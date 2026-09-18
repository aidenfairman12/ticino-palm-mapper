#!/usr/bin/env python
"""
15_extract_classical_features.py
=================================
Builds a tabular feature file (one row per labeled point x covering tile,
same unit-of-example as PalmProbeDataset/linear_probe.py) for training a
classical model (random forest / gradient-boosted trees) as a cheaper,
interpretable alternative to the ViT-embedding + linear-probe pipeline.

Motivation: T. fortunei has a genuinely distinctive physical signature —
evergreen (stable NDVI year-round), a specific height range, and a
radially-symmetric fan crown. The deep pipeline tries to re-derive that
implicitly through a natural-photo-pretrained ViT's frozen embedding space,
adapted to 6 channels via a hacked patch-embed conv. This script instead
extracts the physical signal directly, from the raw [NIR,R,G,B,NDVI,CHM] /
[R,G,B,CHM] stack, as a feature table any sklearn/xgboost/lightgbm model
can consume.

Per-band features, per --radii-m (default 1/3/5m):
  - <band>_point: value at the exact labeled coordinate (nearest pixel)
  - <band>_mean_r<R>, _std_r<R>, _max_r<R>, _min_r<R>: neighborhood stats
    within radius R meters
Derived:
  - ndvi_contrast: point NDVI minus the widest-radius neighborhood mean —
    formalizes the same idea as lugano_MASTER_confirmed_palms.geojson's
    hand-tuned ndvi_contrast_current field, computed uniformly for every
    point instead of by manual review.
  - chm_peakiness: point CHM minus the widest-radius neighborhood mean —
    a palm crown should be a localized height bump above its surroundings;
    a flat rooftop or uniform canopy should not be.
  - <band>_temporal_std, <band>_temporal_range, n_dates_covered: computed
    across every date-tile covering the SAME point (see fold_tile below)
    for ndvi and chm — an evergreen palm should show LOW variance across
    imagery dates, unlike seasonal vegetation or transient artifacts. This
    is the one feature category the deep pipeline structurally cannot see,
    since PalmProbeDataset/leave_one_tile_out_cv treat each date-tile as an
    independent example, never fusing multiple dates for the same point.

fold_tile is the tile's filename (shared across date directories — see
adapt-patch-embed-channel-misalignment-bug / c021_r076 investigation) so a
leave-one-tile-out CV loop can group rows identically to linear_probe.py's,
for an apples-to-apples comparison against its reported accuracy.

STATUS: feature extraction only — deliberately does not include the
model-training/CV loop. Load the output CSV and take it from there (a
leave-one-tile-out split grouped by `fold_tile`, mirroring
leave_one_tile_out_cv in src/training/linear_probe.py, is the natural
apples-to-apples comparison against the existing probe numbers).

Not included in this first pass (flagged, not built, to avoid scope creep
before knowing if the simpler feature set is even competitive): texture/
GLCM/Gabor features, an explicit radial-symmetry/matched-filter shape
feature for the fan crown. Worth adding only if v1 underperforms the deep
pipeline and texture looks like the missing signal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.transform import rowcol
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import PROJECT_CRS  # noqa: E402

# The obvious versions of these five helpers already exist in src/data/dataset.py
# and src/data/probe_dataset.py, but BOTH of those modules import torch at the
# top level (for unrelated Dataset classes) — and torch + xgboost loaded in the
# same process segfaults (a known OpenMP-runtime conflict between the two
# libraries). This script (and anything that dynamically loads it, like
# src/inference/score_candidates_classical.py) has no other reason to need
# torch, so these are duplicated here, torch-free, rather than imported.


def load_tile(path: Path) -> tuple[np.ndarray, "rasterio.Affine", "rasterio.crs.CRS"]:
    with rasterio.open(path) as src:
        arr = src.read()
        transform = src.transform
        crs = src.crs
    assert arr.dtype == np.float32, f"expected float32, got {arr.dtype} in {path}"
    assert arr.shape[0] in (4, 6), f"expected 4 or 6 channels but got {arr.shape[0]} from {path}"
    assert crs.to_epsg() == int(PROJECT_CRS.split(":")[1]), f"expected EPSG:2056, got {crs} in {path}"
    return arr, transform, crs


def load_confirmed_points(geojson_path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(geojson_path)


def load_tile_bounds(tile_paths: list[Path]) -> list[tuple[Path, "shapely.geometry.base.BaseGeometry"]]:
    ret = []
    for path in tile_paths:
        with rasterio.open(path) as src:
            coords = src.bounds
        ret.append((path, box(*coords)))
    return ret


def find_covering_tiles(point, tile_boxes: list[tuple[Path, "shapely.geometry.base.BaseGeometry"]]) -> list[Path]:
    return [path for path, tile_box in tile_boxes if point.within(tile_box)]


def sample_negative_points(
    positive_points: gpd.GeoSeries, candidate_points: gpd.GeoSeries, tile_paths: list[Path],
    n_negatives: int, min_distance_m: float, seed: int,
) -> list[tuple[float, float]]:
    results = []
    rng = np.random.default_rng(seed)
    while len(results) < n_negatives:
        idx = rng.integers(0, len(tile_paths))
        with rasterio.open(tile_paths[idx]) as src:
            coords = src.bounds
        x, y = rng.uniform(coords.left, coords.right), rng.uniform(coords.bottom, coords.top)
        point = shapely.geometry.Point(x, y)
        if positive_points.distance(point).min() >= min_distance_m and candidate_points.distance(point).min() >= min_distance_m:
            results.append((x, y))
    return results


def glob_tiles(tile_dirs: list[Path], in_chans: int) -> list[Path]:
    """Same behavior as src.training.pretrain_ssl.glob_tiles, duplicated
    here rather than imported — that module pulls in torch purely for this
    trivial suffix-glob, and torch + xgboost loaded in the same process
    segfaults (a known OpenMP-runtime conflict between the two libraries).
    This module has no other reason to need torch at all.
    """
    suffix = "*_rgbchm.tif" if in_chans == 4 else "*_nirchm.tif"
    tile_paths = []
    for d in tile_dirs:
        tile_paths.extend(sorted(d.glob(suffix)))
    return tile_paths

BAND_NAMES_6 = ["nir", "r", "g", "b", "ndvi", "chm"]
BAND_NAMES_4 = ["r", "g", "b", "chm"]
RES_M = 0.1  # SwissImage / feature-stack pixel size, meters/px


def window_stats(arr: np.ndarray, transform, point: shapely.geometry.Point, radius_m: float):
    """(point_value, mean, std, max, min) per band within radius_m of point."""
    row, col = rowcol(transform, point.x, point.y)
    H, W = arr.shape[1], arr.shape[2]
    row = max(0, min(row, H - 1))
    col = max(0, min(col, W - 1))
    radius_px = max(1, round(radius_m / RES_M))
    r0, r1 = max(0, row - radius_px), min(H, row + radius_px + 1)
    c0, c1 = max(0, col - radius_px), min(W, col + radius_px + 1)
    window = arr[:, r0:r1, c0:c1]
    point_val = arr[:, row, col]
    return point_val, window.mean(axis=(1, 2)), window.std(axis=(1, 2)), window.max(axis=(1, 2)), window.min(axis=(1, 2))


def extract_features_from_array(arr: np.ndarray, transform, point: shapely.geometry.Point, radii_m: list[float]) -> dict:
    """Same feature set as extract_features, but against an already-loaded
    (arr, transform) pair rather than a tile path — lets a caller that's
    caching tiles in memory (e.g. a full-grid scoring pass touching the
    same handful of tiles repeatedly) avoid re-reading from disk per point,
    same reasoning as score_candidates.py's tile cache.
    """
    band_names = BAND_NAMES_6 if arr.shape[0] == 6 else BAND_NAMES_4

    feats: dict[str, float] = {}
    for r in radii_m:
        pt_val, mean, std, mx, mn = window_stats(arr, transform, point, r)
        for i, name in enumerate(band_names):
            if r == radii_m[0]:
                feats[f"{name}_point"] = float(pt_val[i])
            feats[f"{name}_mean_r{r}"] = float(mean[i])
            feats[f"{name}_std_r{r}"] = float(std[i])
            feats[f"{name}_max_r{r}"] = float(mx[i])
            feats[f"{name}_min_r{r}"] = float(mn[i])

    widest = radii_m[-1]
    if "ndvi" in band_names:
        feats["ndvi_contrast"] = feats["ndvi_point"] - feats[f"ndvi_mean_r{widest}"]
    if "chm" in band_names:
        feats["chm_peakiness"] = feats["chm_point"] - feats[f"chm_mean_r{widest}"]

    return feats


def extract_features(tile_path: Path, point: shapely.geometry.Point, radii_m: list[float]) -> dict:
    arr, transform, _ = load_tile(tile_path)
    return extract_features_from_array(arr, transform, point, radii_m)


def build_rows(points: gpd.GeoSeries, label: float, tile_boxes, radii_m: list[float]) -> list[dict]:
    rows = []
    for point_id, point in enumerate(points):
        covering = find_covering_tiles(point, tile_boxes)
        for tile_path in covering:
            feats = extract_features(tile_path, point, radii_m)
            feats.update(
                point_id=f"{'pos' if label else 'neg'}_{point_id}",
                label=label,
                fold_tile=tile_path.name,
                x=point.x,
                y=point.y,
            )
            rows.append(feats)
    return rows


def temporal_features_from_dicts(per_date_feats: list[dict]) -> dict:
    """Same computation as add_temporal_features, but for one point's list
    of per-date feature dicts directly rather than a groupby over a
    DataFrame — for streaming callers (e.g. full-grid scoring) that never
    materialize the whole dataset as a table the way build_rows does.
    """
    ndvi_vals = [f["ndvi_point"] for f in per_date_feats if "ndvi_point" in f]
    chm_vals = [f["chm_point"] for f in per_date_feats if "chm_point" in f]
    out = {
        "chm_temporal_std": float(np.std(chm_vals)) if len(chm_vals) > 1 else 0.0,
        "chm_temporal_range": float(max(chm_vals) - min(chm_vals)) if chm_vals else 0.0,
        "n_dates_covered": len(per_date_feats),
    }
    if ndvi_vals:
        out["ndvi_temporal_std"] = float(np.std(ndvi_vals)) if len(ndvi_vals) > 1 else 0.0
        out["ndvi_temporal_range"] = float(max(ndvi_vals) - min(ndvi_vals))
    return out


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """For each point_id covered by multiple date-tiles, compute the std/range
    of ndvi_point and chm_point across those tiles and broadcast back onto
    every row for that point — an evergreen palm should be stable across
    dates, unlike seasonal vegetation or transient imagery artifacts.
    """
    agg = df.groupby("point_id").agg(
        ndvi_temporal_std=("ndvi_point", "std") if "ndvi_point" in df else ("chm_point", lambda s: np.nan),
        chm_temporal_std=("chm_point", "std"),
        n_dates_covered=("fold_tile", "count"),
    )
    if "ndvi_point" in df:
        rng = df.groupby("point_id")["ndvi_point"].agg(lambda s: s.max() - s.min())
        agg["ndvi_temporal_range"] = rng
    chm_rng = df.groupby("point_id")["chm_point"].agg(lambda s: s.max() - s.min())
    agg["chm_temporal_range"] = chm_rng
    agg = agg.fillna(0.0)  # std/range of a single-date point is 0, not NaN — "no variation observed" is the honest value
    return df.merge(agg, on="point_id", how="left")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--confirmed-points", type=Path, required=True, help="Path to the MASTER confirmed-palms GeoJSON.")
    p.add_argument("--scouted-points", type=Path, default=None)
    p.add_argument("--reviewed-positives", type=Path, default=None, help="active_learning_confirmed_palms.geojson")
    p.add_argument("--hard-negatives", type=Path, default=None, help="active_learning_hard_negatives.geojson")
    p.add_argument("--n-negatives", type=int, default=30)
    p.add_argument("--min-distance-m", type=float, default=20.0)
    p.add_argument("--radii-m", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    p.add_argument("--in-chans", type=int, default=6, choices=[4, 6])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("classical_features.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

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
    print(f"positives: {len(positive_geoms)} total")

    tile_paths = glob_tiles(args.tile_dirs, args.in_chans)
    tile_boxes = load_tile_bounds(tile_paths)

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
    negative_geoms = gpd.GeoSeries([shapely.geometry.Point(x, y) for x, y in negatives], crs=positive_geoms.crs)
    print(f"negatives: {len(negative_geoms)} total")

    rows = build_rows(positive_geoms, 1.0, tile_boxes, args.radii_m)
    rows += build_rows(negative_geoms, 0.0, tile_boxes, args.radii_m)
    df = pd.DataFrame(rows)
    df = add_temporal_features(df)

    df.to_csv(args.output, index=False)
    n_folds = df["fold_tile"].nunique()
    print(f"=== wrote {len(df)} rows ({int(df['label'].sum())} positive, {int((1 - df['label']).sum())} negative), "
          f"{n_folds} distinct fold_tile values -> {args.output} ===")


if __name__ == "__main__":
    main()
