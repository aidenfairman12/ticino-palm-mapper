"""
MAE-style masked autoencoder for domain-adaptive SSL pretraining.

Standard MAE (He et al. 2021), adapted for this project:
  - Input: (B, C, H, W) tiles from PalmSSLDataset, C=4 (RGB+CHM) or 6
    (NIR+RGB+NDVI+CHM), H=W=crop_size (224 in current testing).
  - Patchify into non-overlapping patch_size x patch_size patches.
  - Randomly mask a high fraction of patches (paper default 0.75). The
    ENCODER ONLY EVER SEES THE VISIBLE PATCHES — this is what makes MAE
    cheap: encoder compute scales with the ~25% kept, not the full image.
  - The decoder is lightweight: it takes the encoded visible tokens plus a
    shared learned "mask token" (same vector, repeated) for every masked
    position, plus full positional embeddings, and predicts pixel values
    for every patch.
  - Loss is MSE between predicted and actual pixel values, computed ONLY on
    masked patches (visible patches contribute nothing — the model isn't
    graded on copying what it already saw).

Backbone: transfer learning from a pretrained DINOv2 ViT (via timm) rather
than training an encoder from scratch, per the project's domain-adaptive
continued-pretraining plan. DINOv2 was pretrained on 3-channel RGB at
518x518 (patch_size=14, 37x37 patch grid), so two adaptations are needed
to reuse it here:
  - the patch-embedding input projection is widened from 3 to 4/6 channels
    (adapt_patch_embed),
  - the positional embeddings are interpolated at runtime from the 37x37
    pretraining grid to whatever grid size the current input resolution
    produces (see forward_encoder) — this backbone was NOT built with
    dynamic_img_size=True, so timm does not do this automatically.

Reference: https://github.com/facebookresearch/mae
"""
from __future__ import annotations

import torch
import torch.nn as nn
from timm.layers import resample_abs_pos_embed


def adapt_patch_embed(backbone: nn.Module, in_chans: int) -> nn.Module:
    """Widen a pretrained ViT's patch-embedding conv from 3 to `in_chans`
    input channels, in place.

    The pretrained RGB weights are copied into the first 3 input-channel
    slots; the remaining channels (CHM, and NIR/NDVI if in_chans==6) are
    zero-initialized. This means the model starts out numerically
    identical to the pretrained RGB-only model — the new channels
    contribute nothing until training gives them a reason to.
    """
    old_proj = backbone.patch_embed.proj
    new_proj = nn.Conv2d(in_chans, old_proj.out_channels, old_proj.kernel_size, old_proj.stride)

    with torch.no_grad():
        new_proj.weight[:, :3, :, :] = old_proj.weight
        new_proj.weight[:, 3:, :, :] = 0
        new_proj.bias = old_proj.bias

    backbone.patch_embed.proj = new_proj
    return backbone


def random_masking(
    x: torch.Tensor, mask_ratio: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly drop a fraction of patch tokens, per-sample.

    Args:
        x: (B, N, D) patch tokens.
        mask_ratio: fraction of the N tokens to mask, e.g. 0.75.

    Returns:
        x_masked: (B, N_keep, D) — the visible tokens only.
        mask: (B, N) binary, 1 = masked, 0 = kept.
        ids_restore: (B, N) — maps shuffled/kept order back to original
            patch order; needed by the decoder to reinsert mask tokens at
            the correct positions.
    """
    B, N, D = x.shape
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)

    N_keep = int((1 - mask_ratio) * N)
    ids_keep = ids_shuffle[:, :N_keep]
    ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, D)
    x_masked = torch.gather(x, 1, ids_keep_expanded)

    mask = torch.ones(B, N, device=x.device)
    mask[:, :N_keep] = 0
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.gather(mask, 1, ids_restore)

    return x_masked, mask, ids_restore


class MaskedAutoencoder(nn.Module):
    """MAE wrapper around a pretrained ViT encoder + a small custom decoder.

    Composition, not inheritance: this class HOLDS a backbone (a timm
    DINOv2 ViT with its patch embed widened via adapt_patch_embed) and adds
    the masking + lightweight decoder + loss machinery around it.
    """

    def __init__(
        self,
        backbone: nn.Module,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        mask_ratio: float,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        # Backbone was pretrained at a fixed resolution (518); positional
        # embeddings are interpolated by hand in forward_encoder, so the
        # backbone's own strict input-size check needs to be disabled.
        self.backbone.patch_embed.strict_img_size = False

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, decoder_embed_dim) * 0.02
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim, nhead=decoder_num_heads, batch_first=True
        )
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N, patch_size**2 * C), the pixel-space target
        the decoder's output is compared against in the loss.
        """
        B, C, H, W = imgs.shape
        p = self.patch_size
        h = H // p
        w = W // p

        tens = imgs.reshape(B, C, h, p, w, p)
        tens = tens.permute(0, 2, 4, 3, 5, 1)
        tens = tens.reshape(B, h * w, p * p * C)

        return tens

    def forward_encoder(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Patch-embed, mask, and encode the visible patches.

        Bypasses the backbone's own forward() to insert masking between
        patch-embedding and the transformer blocks: positional embeddings
        must be added to the FULL patch sequence (so they reflect true
        patch position) before masking drops most of it. The CLS token is
        reattached after masking, since it isn't a maskable patch.

        Returns: (encoded_visible_tokens, mask, ids_restore).
        """
        x = self.backbone.patch_embed(x)
        B = x.shape[0]

        # Backbone's pos_embed was learned at a 37x37 patch grid (518px
        # pretraining resolution); resize to this input's actual grid size.
        patch_pos_embed = resample_abs_pos_embed(
            self.backbone.pos_embed[:, 1:, :],
            new_size=(self.img_size // self.patch_size, self.img_size // self.patch_size),
            old_size=(37, 37),
            num_prefix_tokens=0,
        )
        x = x + patch_pos_embed

        x_masked, mask, ids_restore = random_masking(x, self.mask_ratio)

        cls_token = self.backbone.cls_token.expand(B, -1, -1) + self.backbone.pos_embed[:, :1, :]
        x = torch.cat([cls_token, x_masked], dim=1)

        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Reinsert mask tokens at their true positions and decode to pixel
        predictions for every patch (both visible and masked).

        Returns: (B, N, patch_size**2 * in_chans).
        """
        x = self.decoder_embed(x)
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]

        num_masked = ids_restore.shape[1] - patch_tokens.shape[1]
        mask_tokens = self.mask_token.expand(x.shape[0], num_masked, -1)
        combined = torch.cat([patch_tokens, mask_tokens], dim=1)

        ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim)
        x = torch.gather(combined, 1, ids_restore)

        x = x + self.decoder_pos_embed
        x = torch.cat([cls_token, x], dim=1)
        x = self.decoder_blocks(x)
        x = self.decoder_pred(x)

        return x[:, 1:, :]  # drop CLS — loss target has no CLS entry

    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean squared error between predicted and target pixel values,
        averaged over masked patches only.
        """
        target = self.patchify(imgs)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full MAE forward pass: encode -> decode -> loss.

        Returns (loss, pred, mask) — pred and mask are kept around for
        visualization/debugging (e.g. reconstructing a masked tile to
        eyeball quality) even though only loss is needed for training.
        """
        encoded, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(encoded, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)

        return loss, pred, mask


if __name__ == "__main__":
    # Sanity-check scaffold: build the model, run one random batch through
    # it, and confirm loss/pred/mask shapes make sense before wiring up a
    # real training loop.
    import timm

    backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
    backbone = adapt_patch_embed(backbone, in_chans=4)

    model = MaskedAutoencoder(
        backbone=backbone,
        img_size=224,
        patch_size=14,
        in_chans=4,
        embed_dim=384,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=6,
        mask_ratio=0.75,
    )

    x = torch.randn(2, 4, 224, 224)
    loss, pred, mask = model(x)
    print(f"loss: {loss.item():.4f}")
    print(f"pred shape: {pred.shape}")
    print(f"mask shape: {mask.shape}, mask sum per sample: {mask.sum(dim=1)}")
