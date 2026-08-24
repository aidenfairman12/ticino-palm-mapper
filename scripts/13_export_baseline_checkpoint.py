#!/usr/bin/env python
"""
13_export_baseline_checkpoint.py
================================
Exports a checkpoint in the same format pretrain_ssl.py's save_checkpoint
produces, but using the PLAIN pretrained DINOv2 backbone — no continued
MAE pretraining on domain imagery at all. Not a trained checkpoint; the
model weights are exactly what timm.create_model(..., pretrained=True)
loads, with only the minimum architecture change needed to accept this
project's channel count (adapt_patch_embed widens the RGB patch-embed conv
to in_chans, zero-initializing the new channels — at inference this means
the extra channels contribute exactly zero, so the model behaves
identically to the untouched 3-channel DINOv2 model, not some partially-
adapted variant).

Purpose: MAE-style continued pretraining is documented in the SSL
literature to sometimes produce LESS linearly-separable features than
contrastive pretraining (DINOv2's original objective) — MAE's strength is
usually under fine-tuning, not linear probing. Since this project's
backbone started as DINOv2 and got continued-pretrained via MAE
reconstruction, it's worth directly checking whether that continued
pretraining helped or hurt linear-probe performance, rather than assuming
domain-adaptive pretraining is strictly better. Run linear_probe.py
against this checkpoint's output and compare to checkpoints/bellinzona/
checkpoint_best.pt's numbers — if this baseline does BETTER, that's a real
finding worth taking seriously, not just a sanity check that failed to
find anything.

epoch=0 and best_val_loss=nan are used as sentinels marking this as NOT a
real trained checkpoint, in case anyone is tempted to --resume-from it —
that would silently "resume" training from epoch 1 of a checkpoint that
was never actually trained.

STATUS: implemented.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import compute_channel_stats, spatial_split  # noqa: E402
from src.training.pretrain_ssl import MODEL_CONFIGS, build_model, glob_tiles, save_checkpoint  # noqa: E402


def parse_export_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a plain-pretrained-DINOv2 (no continued pretraining) baseline checkpoint."
    )
    p.add_argument(
        "--tile-dirs", type=Path, nargs="+", required=True,
        help="Same tile-dirs used for the real pretraining run — determines the train split "
             "(and therefore the normalization stats) the same way, so the comparison is "
             "apples-to-apples. Entirely unused (kept required just for a consistent CLI "
             "shape) if --stats-from is given instead.",
    )
    p.add_argument(
        "--stats-from", type=Path, default=None,
        help="Optional: reuse the stats already saved in an existing checkpoint (e.g. "
             "checkpoints/bellinzona/checkpoint_best.pt) instead of recomputing via "
             "compute_channel_stats. Since the split params below match pretrain_ssl.py's "
             "exactly, the same --tile-dirs would produce byte-identical stats anyway — this "
             "just skips re-reading every train tile from disk to get there. Only valid if "
             "--tile-dirs is the same set that checkpoint was actually trained on.",
    )
    p.add_argument("--in-chans", type=int, default=6, choices=[4, 6])
    p.add_argument("--model-size", type=str, default="small", choices=sorted(MODEL_CONFIGS))
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--block-size-m", type=float, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("checkpoints/baseline_dinov2/checkpoint_best.pt"))
    return p.parse_args()


def main() -> None:
    args = parse_export_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    backbone_name, embed_dim = MODEL_CONFIGS[args.model_size]

    if args.stats_from is not None:
        stats = torch.load(args.stats_from, map_location="cpu", weights_only=False)["stats"]
        print(f"reusing stats from {args.stats_from} — skipped recomputing over the train split")
    else:
        tile_paths = glob_tiles(args.tile_dirs, args.in_chans)
        train_paths, _, _ = spatial_split(tile_paths, args.val_frac, args.test_frac, args.block_size_m, args.seed)
        stats = compute_channel_stats(train_paths)
        print(f"computed stats over {len(train_paths)} train tiles (of {len(tile_paths)} total)")

    model = build_model(
        backbone_name, args.in_chans, img_size=224, patch_size=14, embed_dim=embed_dim,
        decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=6, mask_ratio=0.75, device=device,
    )
    print(f"loaded plain pretrained {backbone_name} — no continued pretraining applied")

    # Never actually used for training — save_checkpoint just needs something with
    # an optimizer-shaped state_dict to stay a self-contained, faithful reproduction
    # of the real checkpoint format.
    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        args.output, model, dummy_optimizer,
        epoch=0, stats=stats, best_val_loss=float("nan"),
        backbone_name=backbone_name, embed_dim=embed_dim,
    )
    print(f"wrote baseline checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
