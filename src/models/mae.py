"""
MAE-style masked autoencoder for domain-adaptive SSL pretraining.

STUBBED ON PURPOSE — this file is yours to implement. Every function/method
below has a docstring describing what it needs to do, expected shapes, and
relevant context. No tensor logic is filled in.

Big picture (standard MAE, He et al. 2021, adapted for this project):
  - Input: (B, C, H, W) tiles from PalmSSLDataset, C=4 (RGB+CHM) or 6
    (NIR+RGB+NDVI+CHM), H=W=crop_size (224 in current testing).
  - Patchify into non-overlapping patch_size x patch_size patches (16 is the
    ViT-standard choice; 224/16 = 14x14 = 196 patches).
  - Randomly mask a high fraction of patches (paper default 0.75). The
    ENCODER ONLY EVER SEES THE VISIBLE PATCHES — this is what makes MAE
    cheap: encoder compute scales with the ~25% kept, not the full image.
  - Decoder is lightweight, takes encoded visible tokens + a shared learned
    "mask token" (same vector, repeated) for every masked position + full
    positional embeddings, and predicts pixel values for the masked patches.
  - Loss is MSE between predicted and actual pixel values, computed ONLY on
    masked patches (visible patches contribute nothing — the model isn't
    graded on copying what it already saw).

Backbone choice: this project uses transfer learning from a pretrained
DINOv2 ViT (via timm) rather than training an encoder from scratch — see
project notes on domain-adaptive continued pretraining. DINOv2 was
pretrained on 3-channel RGB, so the patch-embedding input projection needs
adapting to accept 4 or 6 channels here (see adapt_patch_embed below) while
keeping the rest of the pretrained transformer weights intact.

Reference if you want to check your work against the canonical
implementation later (don't copy from it while writing — the point is to
derive the masking/restore-index logic yourself first):
https://github.com/facebookresearch/mae
"""
from __future__ import annotations

import torch
import torch.nn as nn


def adapt_patch_embed(backbone: nn.Module, in_chans: int) -> nn.Module:
    """Swap a pretrained ViT's patch-embedding conv to accept `in_chans`
    input channels instead of its pretrained 3.

    DINOv2 (and most timm ViTs) patchify via a single Conv2d with
    kernel_size=stride=patch_size, living at some backbone-specific
    attribute path (e.g. `backbone.patch_embed.proj` for timm's VisionTransformer).
    You need to:
      - locate that conv layer,
      - build a new Conv2d with the same out_channels/kernel_size/stride but
        in_channels=in_chans,
      - initialize it so training doesn't start from noise: e.g. copy the
        pretrained RGB weights into the first 3 input channels, and init the
        remaining channels (CHM, and NDVI if in_chans==6) some sensible way
        (zeros so the extra channels start silent and the model leans on the
        pretrained RGB response first, or replicate/average the RGB weights
        as a starting point — pick one and document why in a comment here
        once decided).
      - replace the layer on the backbone in place.

    Returns the modified backbone (or modify in place and return it either way).
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
    """Randomly mask a fraction of patch tokens, per-sample, and return only
    the visible ones. generates random masks by sorting random integers by argsort. maps back to the original location
    by argsorting the shuffled tensor
    """
    
    B, N, D = x.shape
    noise = torch.rand(B, N)
    ids_shuffle = torch.argsort(noise, dim=1)

    N_keep = int((1 - mask_ratio) * N)
    ids_keep = ids_shuffle[:,:N_keep]
    ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, D)
    
    patches = torch.gather(x, 1, ids_keep_expanded)
    
    mask = torch.ones(B,N)
    mask[:, :N_keep] = 0
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    mask = torch.gather(mask, 1, ids_restore)
    
    return patches, mask, ids_restore
    

class MaskedAutoencoder(nn.Module):
    """MAE wrapper around a pretrained ViT encoder + a small custom decoder.

    Composition, not inheritance: this class HOLDS a backbone (e.g. a timm
    DINOv2 ViT with its patch embed swapped via adapt_patch_embed) and adds
    the masking + lightweight decoder + loss machinery around it. It does not
    subclass the backbone.
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
        """Store the (already channel-adapted) backbone as the encoder, and
        build the decoder-side modules:
          - a Linear projecting encoder embed_dim -> decoder_embed_dim,
          - a single learned mask_token parameter, shape (1, 1, decoder_embed_dim),
            broadcast to every masked position,
          - decoder positional embeddings, shape (1, num_patches, decoder_embed_dim)
            (fixed sin-cos or learned — your call, document which),
          - a small transformer stack for the decoder (decoder_depth blocks,
            decoder_num_heads heads) — can reuse nn.TransformerEncoderLayer/
            nn.TransformerEncoder as a lightweight stand-in rather than writing
            attention from scratch,
          - a final Linear projecting decoder_embed_dim -> patch_size**2 * in_chans
            (i.e. predicting raw pixel values for each masked patch).
        Also store mask_ratio, patch_size, in_chans, img_size — you'll need
        them for patchify/unpatchify below.
        """
        super().__init__()
        raise NotImplementedError

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N, patch_size**2 * C), the pixel-space target
        the decoder's output gets compared against in the loss.

        Think in terms of a reshape + permute: split H and W into
        (num_patches_per_side, patch_size) each, then rearrange so each patch's
        pixels end up contiguous in the last dimension.
        """
        raise NotImplementedError

    def forward_encoder(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Patch-embed the input via the backbone, apply random_masking to
        keep only the visible tokens, run those through the backbone's
        transformer blocks (skipping its own patch-embed step since you've
        already done that), and return (encoded_visible_tokens, mask, ids_restore).

        Watch for: the backbone may add its own positional embeddings and/or
        a CLS token internally — you need those applied to the FULL patch
        sequence before masking (positions must correspond to true patch
        locations), not after. Check how your chosen timm backbone exposes
        patch_embed vs. pos_embed vs. the transformer blocks themselves to
        do this correctly, since you're bypassing its normal forward().
        """
        raise NotImplementedError

    def forward_decoder(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """Project encoded visible tokens to decoder_embed_dim, insert
        mask_token at every masked position (using ids_restore to get them
        back in correct spatial order), add decoder positional embeddings,
        run through the decoder transformer stack, and project to pixel space.

        Returns: (B, N, patch_size**2 * in_chans) — a prediction for EVERY
        patch position (both visible and masked); the loss will select out
        only the masked ones.
        """
        raise NotImplementedError

    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """MSE between predicted and target pixel values, masked-positions only.

        Steps: patchify(imgs) to get the target in the same (B, N, patch_size**2*C)
        layout as pred, compute per-patch MSE (mean over the pixel dim, keep
        the N dim), multiply elementwise by `mask` (so visible-patch error
        contributes 0), sum and divide by mask.sum() (average error PER
        MASKED PATCH, not per all patches — otherwise your loss scale shifts
        with mask_ratio, which you don't want if you ever sweep it).

        Optional refinement from the original paper worth knowing about even
        if you skip it initially: normalizing each patch's target pixels to
        zero mean/unit variance before computing the loss (per-patch, not
        per-channel-global) tends to improve reconstruction quality. Decide
        whether to include this and note the decision here once made.
        """
        raise NotImplementedError

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Wire the three pieces together: forward_encoder -> forward_decoder
        -> forward_loss. Return (loss, pred, mask) — pred and mask are handy
        to keep around for visualization/debugging (e.g. reconstructing a
        masked tile to eyeball quality) even though only loss is needed for
        the actual training step.
        """
        raise NotImplementedError


if __name__ == "__main__":
    # Sanity-check scaffold — fill in once the pieces above work.
    # Suggested first check: build the model with a small/fast config,
    # run one batch from PalmSSLDataset through it, and confirm loss is a
    # finite scalar and pred/mask shapes make sense, e.g.:
    #
    #   backbone = timm.create_model("vit_small_patch16_224.dino", pretrained=True)
    #   backbone = adapt_patch_embed(backbone, in_chans=4)
    #   model = MaskedAutoencoder(backbone, img_size=224, patch_size=16, in_chans=4,
    #                              embed_dim=384, decoder_embed_dim=192,
    #                              decoder_depth=4, decoder_num_heads=6, mask_ratio=0.75)
    #   loss, pred, mask = model(batch)
    #   print(loss.item(), pred.shape, mask.shape)
    #
    # Get this verified correct before wiring up a full training loop.
    import timm
    backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
    backbone = adapt_patch_embed(backbone, in_chans=4)
    print(backbone.patch_embed.proj)
