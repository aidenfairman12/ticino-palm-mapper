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

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import timm
import torch
import torch.nn as nn

from src.data.dataset import load_confirmed_points
from src.data.probe_dataset import PalmProbeDataset, sample_negative_points
from src.models.mae import MaskedAutoencoder, adapt_patch_embed
from src.training.pretrain_ssl import glob_tiles


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
      labels.append(label.to(device))
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
    probe = nn.Linear(embed_dim, 1).to(features.device)
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
    """
    
    ret = {}

    features, labels, tile_paths = extract_all_features(model, dataset, device)
    embed_dim = features.shape[1]

    # Fold identity is the tile FILENAME, not the full path — the same
    # geographic tile shows up as separate files across different
    # feature_stack_rs_<date> directories (identical filename, since
    # script 07 derives it from the input RGB tile's name, not rs_date;
    # only the containing directory differs). Grouping by full Path would
    # split one location's multi-date versions across different "folds",
    # leaking the same location into both train and test simultaneously.
    positive_tiles = {tp.name for tp, label in zip(tile_paths, labels) if label == 1}

    for tile_name in positive_tiles:
      mask = torch.tensor([tp.name == tile_name for tp in tile_paths])

      test_feat = features[mask]
      train_feat = features[~mask]

      train_lab, test_label = labels[~mask], labels[mask]

      probe = train_probe(train_feat, train_lab, embed_dim, epochs, lr)

      ret[tile_name] = evaluate_probe(probe, test_feat, test_label)

    return ret
      


def parse_probe_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leave-one-tile-out linear probe eval of an SSL checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to a pretrain_ssl.py checkpoint (e.g. checkpoint_best.pt).")
    p.add_argument("--tile-dirs", type=Path, nargs="+", required=True, help="feature_stack_rs_<date> dir(s) to search for covering tiles — same set (or a superset) of what the checkpoint was trained on.")
    p.add_argument("--confirmed-points", type=Path, required=True, help="Path to the MASTER confirmed-palms GeoJSON.")
    p.add_argument(
        "--scouted-points", type=Path, default=None,
        help="Optional path to a scouted-candidates GeoJSON (e.g. from KML-based satellite "
             "scouting) — all its points are included as positives alongside the confirmed "
             "'distinct' set, regardless of confidence tier. Accepts some false positives in "
             "exchange for meaningfully more positive-tile fold coverage.",
    )
    p.add_argument("--n-negatives", type=int, default=30)
    p.add_argument("--min-distance-m", type=float, default=20.0)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_probe_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load checkpoint first — its stats tell us in_chans (len of the mean
    # vector) rather than needing a separate, error-prone CLI flag that
    # could drift out of sync with what the checkpoint actually expects.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stats = checkpoint["stats"]
    in_chans = len(stats.mean)

    # Architecture constants must match pretrain_ssl.py's exactly, or
    # load_state_dict below will fail on a shape mismatch.
    backbone_name = "vit_small_patch14_dinov2.lvd142m"
    img_size = 224
    patch_size = 14
    embed_dim = 384
    decoder_embed_dim = 192
    decoder_depth = 4
    decoder_num_heads = 6
    mask_ratio = 0.75

    backbone = timm.create_model(backbone_name, pretrained=True)
    backbone = adapt_patch_embed(backbone, in_chans)
    model = MaskedAutoencoder(
        backbone, img_size, patch_size, in_chans, embed_dim,
        decoder_embed_dim, decoder_depth, decoder_num_heads, mask_ratio,
    )
    model.backbone.patch_embed.strict_img_size = False
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Positives: "distinct"-signal confirmed points, plus (if provided) ALL
    # scouted candidates regardless of confidence tier — accepting a few
    # likely false positives in exchange for meaningfully more positive-
    # tile fold coverage than the original 3-tile ground truth allowed.
    # Negatives: sampled locations, excluded from being near either a
    # confirmed point OR a "none"-signal point (already-reviewed-but-
    # inconclusive — a reasonable stand-in for "don't sample here"
    # alongside the confirmed positives, without needing a separate
    # raw-GBIF-occurrences file for this first real run).
    points = load_confirmed_points(args.confirmed_points)
    distinct = points[points["ndvi_signal_current"] == "distinct"]
    none_signal = points[points["ndvi_signal_current"] == "none"]

    positive_geoms = distinct.geometry
    if args.scouted_points is not None:
        scouted = gpd.read_file(args.scouted_points)
        positive_geoms = pd.concat([positive_geoms, scouted.geometry], ignore_index=True)
        print(f"positives: {len(distinct)} confirmed + {len(scouted)} scouted = {len(positive_geoms)} total")

    tile_paths = glob_tiles(args.tile_dirs, in_chans)

    negatives = sample_negative_points(
        positive_points=positive_geoms,
        candidate_points=none_signal.geometry,
        tile_paths=tile_paths,
        n_negatives=args.n_negatives,
        min_distance_m=args.min_distance_m,
        seed=args.seed,
    )

    dataset = PalmProbeDataset(positive_geoms, negatives, tile_paths, stats, args.crop_size)
    print(f"probe dataset: {len(dataset)} examples "
          f"({sum(1 for e in dataset.examples if e[3] == 1.0)} positive, "
          f"{sum(1 for e in dataset.examples if e[3] == 0.0)} negative)")

    results = leave_one_tile_out_cv(model, dataset, device, args.epochs, args.lr)

    print(f"\n{len(results)} folds:")
    total_tp = total_tn = total_fp = total_fn = 0
    accuracies = []
    for tile_name, res in results.items():
        print(f"  {tile_name}: {res}")
        accuracies.append(res["accuracy"])
        total_tp += res["true_pos"]
        total_tn += res["true_neg"]
        total_fp += res["false_pos"]
        total_fn += res["false_neg"]

    mean_acc = sum(accuracies) / len(accuracies) if accuracies else float("nan")
    total = total_tp + total_tn + total_fp + total_fn
    pooled_acc = (total_tp + total_tn) / total if total else float("nan")
    print(f"\nmean per-fold accuracy: {mean_acc:.3f}")
    print(f"pooled accuracy (all folds combined): {pooled_acc:.3f} "
          f"(tp={total_tp}, tn={total_tn}, fp={total_fp}, fn={total_fn})")


if __name__ == "__main__":
    main()
