"""
Training loop for MAE-style domain-adaptive SSL pretraining.

Wires together:
  - src/data/ssl_dataset.py: PalmSSLDataset (random-cropped, normalized
    tiles, no labels).
  - src/models/mae.py: adapt_patch_embed + MaskedAutoencoder.

Design decisions made for this project:
  - Optimizer: AdamW, matching the original MAE paper's config
    (betas=(0.9, 0.95), weight_decay=0.05).
  - LR schedule: linear warmup into cosine decay (see adjust_learning_rate),
    same as the paper — computed continuously per training step, not per
    epoch. warmup_epochs is set as ~5% of total_epochs (the paper's
    40/800-epoch ratio), not a fixed count, since this project's runs are
    much shorter than the paper's 800-epoch pretraining.
  - Freezing: most of the pretrained backbone's transformer blocks are
    frozen initially (see build_model) — this is CONTINUED pretraining on
    a small (~27 train tile), domain-specific dataset, far smaller than
    DINOv2's original pretraining corpus, so full fine-tuning risks
    overwriting useful general-purpose features before the model has seen
    enough data to responsibly relearn them. patch_embed (newly
    channel-widened), the last few blocks, and the entire decoder stay
    trainable.
  - Checkpointing: every checkpoint_every epochs, saving model/optimizer
    state, the epoch number, and the ChannelStats used for normalization
    (needed to reproduce the exact preprocessing this checkpoint expects).
"""
from __future__ import annotations

import math
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ChannelStats, compute_channel_stats, spatial_split
from src.data.ssl_dataset import PalmSSLDataset
from src.models.mae import MaskedAutoencoder, adapt_patch_embed


def build_dataloaders(
    tile_dir: Path,
    stats: ChannelStats,
    crop_size: int,
    batch_size: int,
    num_workers: int,
    val_frac: float,
    test_frac: float,
    block_size_m: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders over PalmSSLDataset."""
    tile_paths = list(tile_dir.glob("*_rgbchm.tif"))
    train, val, _ = spatial_split(tile_paths, val_frac, test_frac, block_size_m, seed)

    train_data = PalmSSLDataset(train, stats, crop_size)
    val_data = PalmSSLDataset(val, stats, crop_size)

    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def build_model(
    backbone_name: str,
    in_chans: int,
    img_size: int,
    patch_size: int,
    embed_dim: int,
    decoder_embed_dim: int,
    decoder_depth: int,
    decoder_num_heads: int,
    mask_ratio: float,
    device: torch.device,
) -> MaskedAutoencoder:
    """Build the channel-adapted, backbone-frozen MaskedAutoencoder.

    Freezes all but the last 5 backbone transformer blocks; patch_embed
    and the entire decoder remain trainable by default (nothing here
    touches their requires_grad).
    """
    backbone = timm.create_model(backbone_name, pretrained=True)
    backbone = adapt_patch_embed(backbone, in_chans)

    model = MaskedAutoencoder(
        backbone,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        decoder_embed_dim,
        decoder_depth,
        decoder_num_heads,
        mask_ratio,
    )
    model = model.to(device)

    num_unfrozen = 5
    frozen_blocks = model.backbone.blocks[:-num_unfrozen]
    for block in frozen_blocks:
        for p in block.parameters():
            p.requires_grad = False

    return model


def build_optimizer(
    model: MaskedAutoencoder, lr: float, weight_decay: float = 0.05
) -> torch.optim.Optimizer:
    """AdamW over trainable parameters only, matching the MAE paper's config."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr, betas=(0.9, 0.95), weight_decay=weight_decay)


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch_frac: float,
    warmup_epochs: float,
    total_epochs: float,
    lr: float,
    min_lr: float = 0.0,
) -> float:
    """Linear warmup into cosine decay, applied to every optimizer param
    group. `epoch_frac` is the continuous (non-integer) point in training
    (e.g. epoch 3, 40% through its batches, is 3.4) — call once per
    training step for a smooth per-step schedule, not once per epoch.
    """
    if epoch_frac < warmup_epochs:
        computed_lr = lr * (epoch_frac / warmup_epochs)
    else:
        progress = (epoch_frac - warmup_epochs) / (total_epochs - warmup_epochs)
        computed_lr = min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    for group in optimizer.param_groups:
        group["lr"] = computed_lr

    return computed_lr


def train_one_epoch(
    model: MaskedAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    warmup_epochs: float,
    base_lr: float,
    log_interval: int,
) -> float:
    """One full training pass over `loader`. Returns the epoch's average loss."""
    model.train()
    total_loss = 0.0
    min_lr = 0.0

    for batch_idx, batch in enumerate(loader):
        epoch_frac = epoch + batch_idx / len(loader)
        current_lr = adjust_learning_rate(
            optimizer=optimizer,
            epoch_frac=epoch_frac,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            lr=base_lr,
            min_lr=min_lr,
        )
        batch = batch.to(device)

        optimizer.zero_grad()
        loss, _, _ = model(batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % log_interval == 0:
            print(loss.item(), current_lr)

    return total_loss / len(loader)


@torch.no_grad()
def validate(model: MaskedAutoencoder, loader: DataLoader, device: torch.device) -> float:
    """One pass over `loader` without updating weights. Returns the average loss."""
    model.eval()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        loss, _, _ = model(batch)
        total_loss += loss.item()

    return total_loss / len(loader)


def save_checkpoint(
    path: Path,
    model: MaskedAutoencoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    stats: ChannelStats,
) -> None:
    """Save model/optimizer state, epoch, and normalization stats to `path`."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "stats": stats,
    }
    torch.save(checkpoint, f=path)


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tile_dir = Path("data/processed/lugano_example/feature_stack")
    backbone_name = "vit_small_patch14_dinov2.lvd142m"
    in_chans = 4
    img_size = 224
    patch_size = 14
    embed_dim = 384
    decoder_embed_dim = 192
    decoder_depth = 4
    decoder_num_heads = 6
    mask_ratio = 0.75

    crop_size = 224
    batch_size = 4
    num_workers = 2
    val_frac = 0.15
    test_frac = 0.15
    block_size_m = 150
    seed = 42

    base_lr = 1.5e-4
    weight_decay = 0.05
    total_epochs = 20
    warmup_epochs = total_epochs * 0.05
    log_interval = 5
    checkpoint_dir = Path("checkpoints")
    checkpoint_every = 5

    tile_paths = list(tile_dir.glob("*_rgbchm.tif"))
    train_paths, _, _ = spatial_split(tile_paths, val_frac, test_frac, block_size_m, seed)
    stats = compute_channel_stats(train_paths)
    lr = base_lr * batch_size / 256

    train_loader, val_loader = build_dataloaders(
        tile_dir, stats, crop_size, batch_size, num_workers, val_frac, test_frac, block_size_m, seed
    )

    model = build_model(
        backbone_name, in_chans, img_size, patch_size, embed_dim,
        decoder_embed_dim, decoder_depth, decoder_num_heads, mask_ratio, device,
    )
    optimizer = build_optimizer(model, lr, weight_decay)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(total_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, total_epochs, warmup_epochs, base_lr, log_interval
        )
        val_loss = validate(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(checkpoint_dir / f"checkpoint_epoch{epoch}.pt", model, optimizer, epoch, stats)


if __name__ == "__main__":
    main()
