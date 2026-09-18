"""
Score random locations across the full NIR footprint with a trained
classical (RF/XGBoost) model on physically-grounded features, surfacing
candidates for manual review — the classical-features analog of
score_candidates.py, for the pipeline that outperformed the deep
ViT+linear-probe approach by ~14 points (see
classical-model-beats-deep-pipeline project memory: 96.7% pooled accuracy
vs the deep pipeline's 82.9% best-ever, at a fraction of the compute).

Deliberately reuses score_candidates.py's hard-won machinery rather than
reimplementing it — grid generation (generate_grid_points), atomic
resumable writes (_write_results/_load_prior_results), and the bounded
LRU tile cache pattern that's the difference between this finishing and
getting OOM-killed at full-grid scale. The only genuinely new piece is
HOW a location gets turned into a feature vector: scripts/15's
extract_features_from_array against cached tile arrays, aggregated across
every covering date (not just the newest) via temporal_features_from_dicts,
since the trained model's feature set includes cross-date stability
columns that only exist if every covering date gets inspected — unlike
score_candidates.py's deep pipeline, which deliberately scores only the
newest non-nodata date to avoid one location producing multiple review
cards.

Workflow:
  1. Train ONE production model on ALL of classical_features.csv (no
     held-out fold — this isn't evaluation, it's the classifier actually
     used to score). Hyperparameters default to the winning XGBoost
     config from the hyperparameter sweep (see module memory above).
  2. Generate the same systematic grid score_candidates.py would, over
     the same --tile-dirs footprint, excluding already-reviewed locations.
  3. For each location: find every covering tile (across all NIR dates),
     extract classical features from each non-nodata one, aggregate
     temporal-stability features across them, and score with the
     production model — batched, with a bounded tile cache and periodic
     atomic flush/resume, mirroring score_candidates.py's score_locations.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections import OrderedDict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import rowcol
from shapely.geometry import Point

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# scripts/15_extract_classical_features.py can't be `import`ed normally —
# a module name can't start with a digit — so load it by file path instead.
# It's also deliberately torch-free (see its own module comment), which is
# why this whole file avoids src.data.dataset / src.data.probe_dataset /
# src.inference.score_candidates below too — all three import torch at the
# top level (for unrelated Dataset/model classes), and torch + xgboost
# loaded in the same process segfaults (a known OpenMP-runtime conflict
# between the two libraries). generate_grid_points/_write_results/
# _load_prior_results are duplicated from score_candidates.py below rather
# than imported, for the same reason.
_spec = importlib.util.spec_from_file_location(
    "extract_classical_features", _REPO_ROOT / "scripts" / "15_extract_classical_features.py"
)
_extract_classical_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_extract_classical_features)
extract_features_from_array = _extract_classical_features.extract_features_from_array
temporal_features_from_dicts = _extract_classical_features.temporal_features_from_dicts
glob_tiles = _extract_classical_features.glob_tiles
load_confirmed_points = _extract_classical_features.load_confirmed_points
load_tile = _extract_classical_features.load_tile
find_covering_tiles = _extract_classical_features.find_covering_tiles
load_tile_bounds = _extract_classical_features.load_tile_bounds

from src.training.classical_probe import build_model, feature_columns, load_feature_table  # noqa: E402

RADII_M = [1.0, 3.0, 5.0]  # must match scripts/15's default — the trained model's columns depend on it


def generate_grid_points(tile_paths, spacing_m: float, exclude_points: gpd.GeoSeries, min_distance_m: float) -> list[tuple[float, float]]:
    """Duplicated from src.inference.score_candidates (torch-free reasons
    above) — identical logic, see that module for the full rationale."""
    tile_boxes = load_tile_bounds(tile_paths)
    boxes_gdf = gpd.GeoDataFrame(geometry=[b for _, b in tile_boxes], crs="EPSG:2056")
    minx, miny, maxx, maxy = boxes_gdf.total_bounds

    xs = np.arange(minx, maxx, spacing_m)
    ys = np.arange(miny, maxy, spacing_m)
    xx, yy = np.meshgrid(xs, ys)
    grid_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs="EPSG:2056")

    covered = gpd.sjoin(grid_gdf, boxes_gdf, predicate="within", how="inner")
    covered = covered[~covered.index.duplicated()]

    if len(exclude_points) > 0:
        exclusion_zone = exclude_points.buffer(min_distance_m).union_all()
        covered = covered[~covered.geometry.within(exclusion_zone)]

    return [(pt.x, pt.y) for pt in covered.geometry]


def _write_results(results: list[dict], output_path: Path) -> None:
    """Duplicated from src.inference.score_candidates (torch-free reasons
    above) — identical atomic-write-via-temp-file-and-rename logic."""
    gdf = gpd.GeoDataFrame(
        results,
        geometry=[Point(r["x"], r["y"]) for r in results],
        crs="EPSG:2056",
    )
    gdf = gdf.sort_values("predicted_prob", ascending=False)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    gdf.to_file(tmp_path, driver="GeoJSON")
    tmp_path.replace(output_path)


def _load_prior_results(output_path: Path) -> tuple[list[dict], set[tuple[float, float]]]:
    """Duplicated from src.inference.score_candidates (torch-free reasons above)."""
    if not output_path.exists():
        return [], set()
    prior_gdf = gpd.read_file(output_path)
    prior_results = prior_gdf.drop(columns="geometry").to_dict("records")
    already_scored = {(round(r["x"], 3), round(r["y"], 3)) for r in prior_results}
    print(f"resuming from {output_path}: {len(already_scored)} location(s) already scored")
    return prior_results, already_scored


def build_production_model(features_csv: Path, model_type: str, class_weight_multiplier: float,
                            model_seed: int, **hyperparams):
    """Train ONE model on ALL of classical_features.csv — no held-out
    fold, since this is the real classifier used to score new candidates,
    not a CV run measuring generalization. Mirrors
    score_candidates.py's build_production_probe.
    """
    df = load_feature_table(features_csv)
    feat_cols = feature_columns(df)
    n_pos = df["label"].sum()
    n_neg = len(df) - n_pos
    pos_weight = (n_neg / n_pos) * class_weight_multiplier if n_pos > 0 else 1.0
    class_weights = {0: 1, 1: pos_weight}
    model = build_model(model_type, model_seed, class_weight=class_weights, **hyperparams)
    model.fit(df[feat_cols], df["label"])
    return model, feat_cols


def score_locations_classical(
    model, feat_cols: list[str], locations: list[tuple[float, float]], tile_paths: list[Path],
    batch_size: int = 256, prior_results: list[dict] | None = None,
    output_path: Path | None = None, flush_every: int = 20000,
) -> list[dict]:
    """Score every (x, y) in `locations`. Unlike score_candidates.py's
    score_locations (one date per location, to avoid duplicate review
    cards), a location's feature vector here is built from EVERY covering
    date — the trained model's temporal-stability columns are only
    meaningful if computed that way, same as at training time
    (scripts/15's build_rows fans a point across every covering tile).
    The "tile" field in the output (for the review page's image) is still
    just the newest covering date, for a single representative card.
    """
    t_start = time.time()
    tile_boxes = load_tile_bounds(tile_paths)

    # Same bounded-LRU reasoning as score_candidates.py's tile cache — an
    # unbounded cache at full-grid scale is what OOM-killed every early
    # attempt at that pipeline.
    max_cached_tiles = 150
    tile_cache: OrderedDict[Path, tuple[np.ndarray, object]] = OrderedDict()

    def _load_tile_cached(path: Path) -> tuple[np.ndarray, object]:
        if path in tile_cache:
            tile_cache.move_to_end(path)
            return tile_cache[path]
        arr, transform, _ = load_tile(path)
        tile_cache[path] = (arr, transform)
        if len(tile_cache) > max_cached_tiles:
            tile_cache.popitem(last=False)
        return tile_cache[path]

    results = list(prior_results) if prior_results else []
    since_last_flush = 0
    n_skipped_nodata = 0
    n_skipped_no_tile = 0
    pending: list[tuple[float, float, str, dict]] = []

    def _run_pending_batch() -> None:
        nonlocal since_last_flush
        if not pending:
            return
        rows = pd.DataFrame([feats for *_, feats in pending])[feat_cols]
        probs = model.predict_proba(rows)[:, 1]
        for (x, y, tile_name, _), prob in zip(pending, probs):
            results.append({"x": x, "y": y, "tile": tile_name, "predicted_prob": float(prob)})
        since_last_flush += len(pending)
        pending.clear()

    for x, y in locations:
        point = Point(x, y)
        covering = find_covering_tiles(point, tile_boxes)
        if not covering:
            n_skipped_no_tile += 1
            continue

        per_date_feats = []
        display_tile = None
        for candidate_tile in sorted(covering, key=lambda p: p.parent.name, reverse=True):
            arr, transform = _load_tile_cached(candidate_tile)
            row, col = rowcol(transform, x, y)
            row = max(0, min(row, arr.shape[1] - 1))
            col = max(0, min(col, arr.shape[2] - 1))
            if (arr[:, row, col] == 0).all():
                continue  # nodata at this point in this date's tile
            feats = extract_features_from_array(arr, transform, point, RADII_M)
            per_date_feats.append(feats)
            if display_tile is None:
                display_tile = candidate_tile.name  # newest non-nodata date, for the review card image

        if not per_date_feats:
            n_skipped_nodata += 1
            continue

        combined = dict(per_date_feats[0])  # point/neighborhood stats from the newest non-nodata date
        combined.update(temporal_features_from_dicts(per_date_feats))
        pending.append((x, y, display_tile, combined))

        if len(pending) >= batch_size:
            _run_pending_batch()
            if output_path is not None and since_last_flush >= flush_every:
                _write_results(results, output_path)
                elapsed = time.time() - t_start
                rate = (len(results) - len(prior_results or [])) / elapsed if elapsed > 0 else 0.0
                print(
                    f"flushed {len(results)} total scored ({len(results) - len(prior_results or [])} this run, "
                    f"{elapsed/60:.1f} min elapsed, {rate:.1f} locations/s) -> {output_path}"
                )
                since_last_flush = 0

    _run_pending_batch()

    if n_skipped_nodata:
        print(f"[warn] skipped {n_skipped_nodata} location(s) — nodata in every covering tile")
    if n_skipped_no_tile:
        print(f"[warn] skipped {n_skipped_no_tile} location(s) — no covering tile")

    if output_path is not None and since_last_flush > 0:
        _write_results(results, output_path)

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-csv", type=Path, required=True, help="Output of scripts/15_extract_classical_features.py — trains the production model.")
    p.add_argument("--tile-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--confirmed-points", type=Path, required=True)
    p.add_argument("--scouted-points", type=Path, default=None)
    p.add_argument("--reviewed-positives", type=Path, default=None)
    p.add_argument("--hard-negatives", type=Path, default=None)
    p.add_argument("--model", choices=["rf", "xgboost"], default="xgboost")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--min-child-weight", type=float, default=1.0, help="xgboost only.")
    p.add_argument("--min-samples-leaf", type=int, default=1, help="rf only.")
    p.add_argument("--class-weight-multiplier", type=float, default=1.0)
    p.add_argument("--model-seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--n-negatives", type=int, default=30, help="Only used to build the already-reviewed exclusion zone, same as score_candidates.py — the production model itself always trains on the full features CSV.")
    p.add_argument("--min-distance-m", type=float, default=20.0)
    p.add_argument("--spacing-m", type=float, default=11.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--flush-every", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("candidate_scores/scores_classical.geojson"))
    return p.parse_args()


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # see score_candidates.py's main() for why this matters
    args = parse_args()

    t_load_start = time.time()

    model_type = args.model
    hyperparams = {"n_estimators": args.n_estimators, "max_depth": args.max_depth}
    if model_type == "rf":
        hyperparams["min_samples_leaf"] = args.min_samples_leaf
    else:
        hyperparams["min_child_weight"] = args.min_child_weight

    model, feat_cols = build_production_model(
        args.features_csv, model_type, args.class_weight_multiplier, args.model_seed, **hyperparams
    )
    print(f"trained production {model_type} model on {args.features_csv} ({len(feat_cols)} features)")

    points = load_confirmed_points(args.confirmed_points)
    distinct = points[points["ndvi_signal_current"] == "distinct"]
    none_signal = points[points["ndvi_signal_current"] == "none"]

    positive_geoms = distinct.geometry
    if args.scouted_points is not None:
        scouted = gpd.read_file(args.scouted_points)
        positive_geoms = pd.concat([positive_geoms, scouted.geometry], ignore_index=True)
    if args.reviewed_positives is not None:
        reviewed_pos = gpd.read_file(args.reviewed_positives)
        positive_geoms = pd.concat([positive_geoms, reviewed_pos.geometry], ignore_index=True)

    tile_paths = glob_tiles(args.tile_dirs, 6)

    hard_neg_geoms = gpd.GeoSeries([], crs="EPSG:2056")
    if args.hard_negatives is not None:
        hard_neg = gpd.read_file(args.hard_negatives)
        hard_neg_geoms = hard_neg.geometry

    t_load_end = time.time()

    already_reviewed = pd.concat([positive_geoms, none_signal.geometry, hard_neg_geoms], ignore_index=True)
    grid_locations = generate_grid_points(tile_paths, args.spacing_m, already_reviewed, args.min_distance_m)
    print(f"grid: {len(grid_locations)} candidate locations at {args.spacing_m}m spacing")

    if len(grid_locations) > args.n_samples:
        rng = np.random.default_rng(args.seed + 1)
        chosen_idx = rng.choice(len(grid_locations), size=args.n_samples, replace=False)
        scored_locations = [grid_locations[i] for i in chosen_idx]
    else:
        scored_locations = grid_locations
    print(f"scoring {len(scored_locations)} of them this run...")

    t_grid_end = time.time()

    prior_results, already_scored = _load_prior_results(args.output)
    locations_to_score = [
        loc for loc in scored_locations
        if (round(loc[0], 3), round(loc[1], 3)) not in already_scored
    ]
    n_already_done = len(scored_locations) - len(locations_to_score)
    if n_already_done:
        print(f"{n_already_done} of {len(scored_locations)} target location(s) already scored, "
              f"{len(locations_to_score)} remaining this run")

    results = score_locations_classical(
        model, feat_cols, locations_to_score, tile_paths, args.batch_size,
        prior_results=prior_results, output_path=args.output, flush_every=args.flush_every,
    )

    t_score_end = time.time()
    ms_per_location = (
        (t_score_end - t_grid_end) / len(locations_to_score) * 1000 if locations_to_score else 0.0
    )
    print(
        f"timing — load+train: {t_load_end - t_load_start:.1f}s, "
        f"grid gen: {t_grid_end - t_load_end:.1f}s, scoring: {t_score_end - t_grid_end:.1f}s "
        f"({ms_per_location:.2f} ms/location, {len(locations_to_score)} newly scored)"
    )

    n_flagged = sum(1 for r in results if r["predicted_prob"] > 0.5)
    print(f"wrote {len(results)} scored locations -> {args.output}")
    print(f"{n_flagged} flagged as predicted-positive (prob > 0.5) — review these first")


if __name__ == "__main__":
    main()
