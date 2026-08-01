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
    
    Returns the trained nn.Linear.
    """
    probe = nn.Linear(embed_dim, 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
      optimizer.zero_grad()
      logits = probe(features).squeeze(-1)
      loss = loss_fn(logits, labels)
      loss.backward()
      optimizer.step()
      
    return probe
      


def evaluate_probe(
    probe: nn.Linear, features: torch.Tensor, labels: torch.Tensor
) -> dict:
    """Evaluate a trained probe on held-out features/labels.
    """
    
    with torch.no_grad():
      logits = probe(features).squeeze(-1)
      preds = (logits > 0).float()
      acc = (preds == labels).float().mean().item()
      
      tPos = ((preds == 1) & (labels == 1)).sum().item()
      tNeg = ((preds == 0) & (labels == 0)).sum().item()
      fPos = ((preds == 1) & (labels == 0)).sum().item()
      fNeg = ((preds == 0) & (labels == 1)).sum().item()
    
    ret = {
      "accuracy": acc,
      "true_pos": tPos,
      "true_neg": tNeg,
      "false_pos": fPos,
      "false_neg": fNeg
    }
    
    return ret


def leave_one_tile_out_cv(
    model: MaskedAutoencoder,
    dataset: PalmProbeDataset,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict:
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
        split, record the result,
      - return a dict keyed by held-out tile path, one entry per fold.

    Note on negatives: since negatives were sampled independently of any
    specific tile grouping, decide how they get distributed across folds —
    e.g. all held-out-tile's negatives (if any landed there) go with that
    fold's test set, everything else's negatives go to train, same
    tile-membership logic as positives. No special-casing needed if you
    already have each example's tile_path from extract_all_features.
    """
    
    ret = {}
    
    features, labels, tile_paths = extract_all_features(model, dataset, device)
    embed_dim = features.shape[1]
    positive_tiles = {tp for tp, label in zip(tile_paths, labels) if label == 1}
    
    for tile in positive_tiles:
      mask = torch.tensor([tp == tile for tp in tile_paths])
      
      test_feat = features[mask]
      train_feat = features[~mask]
      
      train_lab, test_label = labels[~mask], labels[mask]
      
      probe = train_probe(train_feat, train_lab, embed_dim, epochs, lr)

      ret[tile] = evaluate_probe(probe, test_feat, test_label)
      
    return ret
      


if __name__ == "__main__":
    # Sanity-check scaffold — fill in once the pieces above work. Suggested
    # first check: build a PalmProbeDataset + a MaskedAutoencoder (loaded
    # from a real checkpoint, per mae.py's __main__ pattern), run
    # leave_one_tile_out_cv, and print each fold's results.
    pass
