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
    features: torch.Tensor, labels: torch.Tensor, embed_dim: int, epochs: int, lr: float,
    pos_weight_multiplier: float = 1.0,
    hidden_dim: int = 0,
) -> nn.Module:
    """Train a classifier on precomputed features (no encoder involved at all
    here — pure feature -> label classification).

    hidden_dim=0 (default) trains a plain nn.Linear(embed_dim, 1), same as
    before. hidden_dim>0 instead trains a small one-hidden-layer MLP
    (embed_dim -> hidden_dim -> 1, ReLU + dropout) — a minimal, cheap test of
    whether the frozen features need a mildly non-linear boundary rather than
    a straight hyperplane. Kept small and regularized (dropout=0.3) since the
    labeled dataset is tiny (a few hundred examples after tile fan-out,
    spread thin across leave-one-tile-out folds) — a bigger head risks
    overfitting rather than finding real structure. Watch for the CV numbers
    getting LESS stable (wider fold-to-fold swings) than the linear probe —
    that's the overfitting signature, not evidence this helped.

    Returns the trained nn.Module (nn.Linear or nn.Sequential depending on
    hidden_dim) — evaluate_probe and everything downstream just calls it,
    indifferent to which.
    """
    if hidden_dim > 0:
        probe = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        ).to(features.device)
    else:
        probe = nn.Linear(embed_dim, 1).to(features.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    # Negatives outnumber positives (~3:1 with hard negatives included, since they
    # fan out across more multi-date tiles than positives do), and unweighted BCE
    # was found to bias the boundary toward predicting "not palm" — recall collapsed
    # to ~50% while specificity sat at 96%. pos_weight rebalances the loss's gradient
    # contribution per class without discarding any hard negatives, unlike
    # subsampling the negative pool would. pos_weight_multiplier=1.0 (default) fully
    # compensates for the imbalance and was found to overshoot the other direction
    # (recall 90.6%, specificity 79.6%) — a fractional value (e.g. 0.5) lands at a
    # more moderate point on the recall/specificity tradeoff instead of either extreme.
    n_pos = labels.sum()
    n_neg = labels.numel() - n_pos
    pos_weight = pos_weight_multiplier * (n_neg / n_pos) if n_pos > 0 else torch.tensor(1.0, device=labels.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
      optimizer.zero_grad()
      logits = probe(features).squeeze(-1)
      loss = loss_fn(logits, labels)
      loss.backward()
      optimizer.step()
      
    return probe
      


def evaluate_probe(
    probe: nn.Module, features: torch.Tensor, labels: torch.Tensor
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
    pos_weight_multiplier: float = 1.0,
    hidden_dim: int = 0,
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
    positive_tiles = sorted({tp.name for tp, label in zip(tile_paths, labels) if label == 1})
    n_folds = len(positive_tiles)

    # Negatives aren't tied to a specific tile the way positives are, and
    # in practice almost never happen to share a tile with a positive
    # point — relying on tile-membership alone leaves every fold's test
    # set with zero negatives (true_neg always 0), never actually testing
    # specificity. Instead, split negatives into n_folds roughly-equal
    # groups up front, one group held out per fold — every negative gets
    # tested in exactly one fold across the whole CV process, and a fold's
    # "other" negative groups remain safely usable for that fold's
    # training (they're just reserved as a DIFFERENT fold's test set,
    # which doesn't leak anything into this fold's own train/test split).
    neg_idx = [i for i, label in enumerate(labels) if label == 0]
    g = torch.Generator().manual_seed(0)
    shuffled_neg_idx = [neg_idx[i] for i in torch.randperm(len(neg_idx), generator=g).tolist()]
    neg_groups = [shuffled_neg_idx[i::n_folds] for i in range(n_folds)]

    for tile_name, neg_test_idx in zip(positive_tiles, neg_groups):
      pos_test_mask = torch.tensor([tp.name == tile_name and lab == 1 for tp, lab in zip(tile_paths, labels)])
      neg_test_mask = torch.zeros(len(labels), dtype=torch.bool)
      neg_test_mask[neg_test_idx] = True

      test_mask = pos_test_mask | neg_test_mask
      train_mask = ~test_mask

      test_feat, test_label = features[test_mask], labels[test_mask]
      train_feat, train_lab = features[train_mask], labels[train_mask]

      probe = train_probe(train_feat, train_lab, embed_dim, epochs, lr, pos_weight_multiplier, hidden_dim)

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
    p.add_argument(
        "--hard-negatives", type=Path, default=None,
        help="Optional path to active_learning_hard_negatives.geojson (from "
             "12_merge_review_verdicts.py) — locations that scored high but were "
             "confirmed not-palm on manual review. Added ON TOP of the --n-negatives "
             "random negatives (not a replacement), since they're a targeted, scarce "
             "resource meant to teach the probe the confusable cases random sampling "
             "essentially never produces.",
    )
    p.add_argument(
        "--reviewed-positives", type=Path, default=None,
        help="Optional path to active_learning_confirmed_palms.geojson (from "
             "12_merge_review_verdicts.py) — locations confirmed as palms on manual "
             "review. Added on top of --confirmed-points/--scouted-points as positives, "
             "same reasoning as score_candidates.py's flag of the same name.",
    )
    p.add_argument("--n-negatives", type=int, default=30)
    p.add_argument("--min-distance-m", type=float, default=20.0)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument(
        "--pos-weight-multiplier", type=float, default=1.0,
        help="Scales pos_weight relative to the full n_neg/n_pos ratio. 1.0 (default) "
             "fully compensates for class imbalance and was found to overshoot toward "
             "recall (90.6% recall / 79.6% specificity on the real bellinzona run); "
             "try e.g. 0.5 for a more moderate point on the recall/specificity tradeoff.",
    )
    p.add_argument(
        "--hidden-dim", type=int, default=0,
        help="0 (default) trains a plain linear probe. >0 (e.g. 32) instead trains a "
             "small one-hidden-layer MLP (embed_dim -> hidden_dim -> 1) on the frozen "
             "features, testing whether the boundary needs mild non-linearity rather "
             "than a straight hyperplane. Keep small given the tiny labeled dataset — "
             "a bigger value risks overfitting rather than finding real signal.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_probe_args()
    print(f"pos-weight-multiplier: {args.pos_weight_multiplier} "
          f"(scales pos_weight relative to the full n_neg/n_pos ratio per fold)")

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

    # backbone_name/embed_dim are read from the checkpoint itself (saved by
    # pretrain_ssl.py's save_checkpoint) rather than hardcoded, since --model-size
    # now lets a run use vit_small or vit_base — hardcoding here would silently
    # mismatch and fail on load_state_dict for anything but the default. Fallback
    # covers checkpoints saved before this field existed (all were vit_small).
    backbone_name = checkpoint.get("backbone_name", "vit_small_patch14_dinov2.lvd142m")
    img_size = 224
    patch_size = 14
    embed_dim = checkpoint.get("embed_dim", 384)
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

    if args.reviewed_positives is not None:
        reviewed_pos = gpd.read_file(args.reviewed_positives)
        positive_geoms = pd.concat([positive_geoms, reviewed_pos.geometry], ignore_index=True)
        print(f"positives: +{len(reviewed_pos)} active-learning-confirmed = {len(positive_geoms)} total")

    tile_paths = glob_tiles(args.tile_dirs, in_chans)

    negatives = sample_negative_points(
        positive_points=positive_geoms,
        candidate_points=none_signal.geometry,
        tile_paths=tile_paths,
        n_negatives=args.n_negatives,
        min_distance_m=args.min_distance_m,
        seed=args.seed,
    )

    if args.hard_negatives is not None:
        hard_neg = gpd.read_file(args.hard_negatives)
        negatives = negatives + [(pt.x, pt.y) for pt in hard_neg.geometry]
        print(f"negatives: {len(negatives) - len(hard_neg)} random + {len(hard_neg)} hard = {len(negatives)} total")

    dataset = PalmProbeDataset(positive_geoms, negatives, tile_paths, stats, args.crop_size)
    print(f"probe dataset: {len(dataset)} examples "
          f"({sum(1 for e in dataset.examples if e[3] == 1.0)} positive, "
          f"{sum(1 for e in dataset.examples if e[3] == 0.0)} negative)")

    results = leave_one_tile_out_cv(
        model, dataset, device, args.epochs, args.lr, args.pos_weight_multiplier, args.hidden_dim
    )

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
