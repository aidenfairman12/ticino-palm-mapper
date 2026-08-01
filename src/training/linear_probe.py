"""
Linear probe: does the SSL-pretrained encoder's features actually separate
palm crops from non-palm crops?

STUBBED ON PURPOSE — this file is yours to implement. Every function below
has a docstring describing what it needs to do. No logic is filled in.

This is the "go/no-go" checkpoint discussed at length this session: a small,
frozen-encoder classifier trained on a handful of labeled crops
(PalmSSLDataset for pretraining, PalmProbeDataset for this). Deliberately
simple relative to pretrain_ssl.py:
  - No DataLoader/multi-worker machinery — dataset size is dozens of
    examples, not thousands; a plain Python loop is simpler and there's no
    real performance cost at this scale.
  - No LR schedule — training a single Linear layer on a tiny, fixed
    feature set is not a place warmup/cosine decay earns its complexity.
  - Features are computed ONCE via the frozen encoder and cached (as plain
    tensors), not recomputed every epoch — the encoder never changes during
    probing, so re-running it repeatedly would be pure waste.

Evaluation strategy: leave-one-tile-out cross-validation, specifically
because positive TILES (not just points) are so scarce. Rather than one
train/val split, hold out each positive-containing tile in turn, train on
everything else, evaluate on the held-out tile's examples, and aggregate
across folds — this is about the only way to get a genuine (if
high-variance, small-sample) read on generalization with this little data.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.data.probe_dataset import PalmProbeDataset
from src.models.mae import MaskedAutoencoder


def extract_all_features(
    model: MaskedAutoencoder, dataset: PalmProbeDataset, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[Path]]:
    """Run every example in `dataset` through the frozen encoder once,
    pooling each to a single feature vector.

    For each (image, label) in the dataset: move image to device, add a
    batch dim (encode_full expects (B, C, H, W)), run model.encode_full(x)
    under torch.no_grad() (no gradients needed — you're not training the
    encoder), and pool the (1, 1+N, D) output down to (D,). Decide your
    pooling strategy here (per encode_full's docstring: CLS token
    `encoded[0, 0]` is well-motivated for a DINOv2 backbone, or mean-pool
    the patch tokens `encoded[0, 1:].mean(dim=0)` — pick one, document why).

    Also collect each example's tile_path (dataset.examples[i][0]) — needed
    by leave_one_tile_out_cv to know which fold an example belongs to.

    Returns: (features, labels, tile_paths) — features (N, D) float32,
    labels (N,) float32, tile_paths a plain list of N Paths (parallel to
    the tensors, not itself a tensor).
    """
    model.eval()
    features, labels, tile_paths = [], [], []
    
    for idx, (image, label) in enumerate(dataset):
      x = image.unsqueeze(0).to(device)
      with torch.no_grad():
        encoded = model.encode_full(x)
      
      feature = encoded[0,0]
      features.append(feature)
      labels.append(label)
      tile_paths.append(dataset.examples[idx][0])
      
    
    return (torch.stack(features), torch.stack(labels), tile_paths)


def train_probe(
    features: torch.Tensor, labels: torch.Tensor, embed_dim: int, epochs: int, lr: float
) -> nn.Linear:
    """Train a single nn.Linear(embed_dim, 1) classifier on precomputed
    features (no encoder involved at all here — pure feature -> label
    classification).

    Standard loop: build the Linear layer, an optimizer (plain Adam is
    fine — no need for AdamW's weight decay on a single tiny layer), and
    nn.BCEWithLogitsLoss. For `epochs` iterations: zero grad, forward pass
    (raw logits, no sigmoid — BCEWithLogitsLoss applies it internally,
    more numerically stable than doing it yourself), compute loss against
    `labels`, backward, step. No batching needed — with this few examples,
    just pass the whole `features` tensor through every iteration (full-
    batch gradient descent).

    Returns the trained nn.Linear.
    """
    raise NotImplementedError


def evaluate_probe(
    probe: nn.Linear, features: torch.Tensor, labels: torch.Tensor
) -> dict:
    """Evaluate a trained probe on held-out features/labels.

    Under torch.no_grad(): get logits from probe(features), threshold at 0
    (equivalent to thresholding sigmoid(logits) at 0.5) to get predicted
    labels, compare against true labels.

    Return a dict with at least: accuracy, and the raw counts (true
    positives/negatives, false positives/negatives) — with folds this
    small, a single accuracy number can be misleading (e.g. 1 wrong
    prediction out of 3 examples looks dramatic), so keeping the raw
    counts around lets you (and anyone reading results later) judge
    fold-by-fold results in context rather than trusting one aggregate
    number blindly.
    """
    raise NotImplementedError


def leave_one_tile_out_cv(
    model: MaskedAutoencoder,
    dataset: PalmProbeDataset,
    device: torch.device,
    epochs: int,
    lr: float,
) -> list[dict]:
    """Run leave-one-tile-out CV: for each unique tile that has at least
    one POSITIVE example, hold it out, train on everything else, evaluate
    on the held-out tile's examples, repeat.

    Steps:
      - call extract_all_features ONCE up front (expensive part, don't
        repeat it per fold),
      - find the set of unique tile_paths that have at least one positive
        (label==1) example among their crops — these are your folds; tiles
        with only negative examples aren't meaningful holdouts on their own
        (there's nothing positive to test generalization against),
      - for each such tile: build a boolean mask splitting features/labels
        into "this tile" (held out) vs "everything else" (train), call
        train_probe on the train split, evaluate_probe on the held-out
        split, record the result (include which tile, for readability),
      - return the list of per-fold result dicts.

    Note on negatives: since negatives were sampled independently of any
    specific tile grouping, decide how they get distributed across folds —
    e.g. all held-out-tile's negatives (if any landed there) go with that
    fold's test set, everything else's negatives go to train, same
    tile-membership logic as positives. No special-casing needed if you
    already have each example's tile_path from extract_all_features.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Sanity-check scaffold — fill in once the pieces above work. Suggested
    # first check: build a PalmProbeDataset + a MaskedAutoencoder (loaded
    # from a real checkpoint, per mae.py's __main__ pattern), run
    # leave_one_tile_out_cv, and print each fold's results.
    pass
