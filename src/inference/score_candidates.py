"""
Score random locations across the full NIR footprint with a trained linear
probe, surfacing candidates for manual review — an active-learning pass to
find both new positives (palms the current label set missed) and hard
negatives (locations that fool the model despite not being palms).

Workflow:
  1. Train one "production" probe on ALL currently available labeled data
     (confirmed + scouted positives, sampled negatives) — no held-out fold,
     unlike leave_one_tile_out_cv, since this isn't an evaluation, it's the
     actual classifier we're about to use.
  2. Sample --n-samples random locations across the given --tile-dirs'
     combined footprint — genuinely unbiased (no exclusion zones), since
     the whole point is broad coverage, not "definitely safe" negatives.
  3. Score every sampled location with the production probe.
  4. Write out every scored location (sorted by predicted probability,
     highest first) to a GeoJSON for manual review — high-probability
     ones are the actual candidates worth looking at: either genuine
     unlabeled palms, or hard negatives that fooled the model.

Reuses sample_negative_points for the random sampling step by passing
min_distance_m=0.0 — since distance is never negative, this makes the
exclusion check a no-op, giving pure unbiased random sampling without
duplicating that logic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import timm
import torch
import torch.nn as nn
from shapely.geometry import Point

from src.data.dataset import load_confirmed_points, load_tile, normalize
from src.data.probe_dataset import (
    PalmProbeDataset,
    crop_centered_on_point,
    find_covering_tiles,
    load_tile_bounds,
    sample_negative_points,
)
from src.models.mae import MaskedAutoencoder, adapt_patch_embed
from src.training.linear_probe import extract_all_features, train_probe
from src.training.pretrain_ssl import glob_tiles


def build_production_probe(
    model: MaskedAutoencoder,
    dataset: PalmProbeDataset,
    device: torch.device,
    embed_dim: int,
    epochs: int,
    lr: float,
) -> nn.Linear:
    """Train one probe on ALL of `dataset` — no held-out fold. This is the
    real classifier used to score new candidates, as opposed to the CV
    probes in leave_one_tile_out_cv (which only exist to measure how well
    this approach generalizes, and are discarded after evaluation)."""
    features, labels, _ = extract_all_features(model, dataset, device)
    return train_probe(features, labels, embed_dim, epochs, lr)


def score_locations(
    model: MaskedAutoencoder,
    probe: nn.Linear,
    locations: list[tuple[float, float]],
    tile_paths: list[Path],
    stats,
    crop_size: int,
    device: torch.device,
) -> list[dict]:
    """Score every (x, y) in `locations` with `probe`, one result per
    location. Unlike PalmProbeDataset (which deliberately fans a labeled
    point out across every covering NIR-date tile for training diversity),
    here a location covered by multiple dates is scored on only the most
    recent date's tile — this is a review/candidate list, so a single real-
    world spot should appear as ONE card, not once per date it happens to
    have extra coverage for. (Tile filenames are identical across date
    directories — only the parent directory differs — so sorting covering
    tiles by parent dir name and taking the last picks the newest date,
    since feature_stack_rs_<YYYYMMDD> sorts chronologically as a string.)
    Returns predicted_prob via sigmoid(logit), so results can be
    sorted/thresholded meaningfully rather than just a hard 0/1."""
    model.eval()
    results = []

    # Precompute once — reopening every tile's bounds per location (as the
    # old find_covering_tiles(point, tile_paths) signature required) would
    # be wasteful at this scale (hundreds/thousands of scored locations).
    tile_boxes = load_tile_bounds(tile_paths)

    for x, y in locations:
        point = Point(x, y)
        covering = find_covering_tiles(point, tile_boxes)
        if not covering:
            continue
        tile_path = sorted(covering, key=lambda p: p.parent.name)[-1]

        arr, transform, _ = load_tile(tile_path)
        arr = normalize(arr, stats)
        crop = crop_centered_on_point(arr, transform, point, crop_size)

        img = torch.from_numpy(crop).float().unsqueeze(0).to(device)
        with torch.no_grad():
            encoded = model.encode_full(img)
            feature = encoded[0, 0].unsqueeze(0)
            logit = probe(feature).squeeze(-1)
            prob = torch.sigmoid(logit).item()

        results.append({
            "x": x, "y": y, "tile": tile_path.name, "predicted_prob": prob,
        })

    return results


def parse_score_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score random locations with a trained probe, for active-learning-style manual review."
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tile-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--confirmed-points", type=Path, required=True)
    p.add_argument("--scouted-points", type=Path, default=None)
    p.add_argument(
        "--hard-negatives", type=Path, default=None,
        help="Optional path to active_learning_hard_negatives.geojson (from "
             "12_merge_review_verdicts.py) — added ON TOP of the --n-negatives random "
             "negatives when training the production probe, same reasoning as "
             "linear_probe.py's --hard-negatives.",
    )
    p.add_argument("--n-samples", type=int, default=500, help="Number of random locations to score.")
    p.add_argument("--n-negatives", type=int, default=30, help="Negatives for training the production probe.")
    p.add_argument("--min-distance-m", type=float, default=20.0)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("candidate_scores.geojson"))
    return p.parse_args()


def main() -> None:
    args = parse_score_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stats = checkpoint["stats"]
    in_chans = len(stats.mean)

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

    points = load_confirmed_points(args.confirmed_points)
    distinct = points[points["ndvi_signal_current"] == "distinct"]
    none_signal = points[points["ndvi_signal_current"] == "none"]

    positive_geoms = distinct.geometry
    if args.scouted_points is not None:
        scouted = gpd.read_file(args.scouted_points)
        positive_geoms = pd.concat([positive_geoms, scouted.geometry], ignore_index=True)

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
    print(f"training production probe on {len(dataset)} examples "
          f"({sum(1 for e in dataset.examples if e[3] == 1.0)} positive, "
          f"{sum(1 for e in dataset.examples if e[3] == 0.0)} negative)")

    probe = build_production_probe(model, dataset, device, embed_dim, args.epochs, args.lr)

    # Pure unbiased random sampling — min_distance_m=0.0 makes
    # sample_negative_points' exclusion check a no-op (distance is never
    # negative), reusing it rather than duplicating the sampling logic.
    random_locations = sample_negative_points(
        positive_points=positive_geoms,
        candidate_points=none_signal.geometry,
        tile_paths=tile_paths,
        n_negatives=args.n_samples,
        min_distance_m=0.0,
        seed=args.seed + 1,  # different seed than the training negatives
    )
    print(f"scoring {len(random_locations)} random locations...")

    results = score_locations(model, probe, random_locations, tile_paths, stats, args.crop_size, device)

    gdf = gpd.GeoDataFrame(
        results,
        geometry=[Point(r["x"], r["y"]) for r in results],
        crs="EPSG:2056",
    )
    gdf = gdf.sort_values("predicted_prob", ascending=False)
    gdf.to_file(args.output, driver="GeoJSON")

    n_flagged = (gdf["predicted_prob"] > 0.5).sum()
    print(f"wrote {len(gdf)} scored locations -> {args.output}")
    print(f"{n_flagged} flagged as predicted-positive (prob > 0.5) — review these first")


if __name__ == "__main__":
    main()
