#!/usr/bin/env python
"""
19_harvest_deciduous_negatives.py
==================================
Harvest confident non-palm negatives from the March (leaf-off) / leaf-on
NDVI-drop signal validated visually in scripts/18's `grid` mode, at the scale
a real training run needs (thousands, not the ~40 eyeballed by hand).

Why this exists: dense scoring of the classical model (see
dense-scoring-reveals-vegetation-detector) showed it learned "tall
vegetation" rather than "palm", because its negatives are sampled >=20m from
any positive and so almost never include another tree. Fixing that needs
real vegetation negatives — but the user cannot reliably tell a palm from
other canopy by eye except right at the roadside (checkpoint-2026-09-18).
The March flight sidesteps that entirely: T. fortunei is evergreen, so a
location that visibly loses NDVI between March and leaf-on is deciduous,
which is a confident non-palm — no visual species identification required.

WHY THIS SCANS TILES, NOT RANDOM POINTS
----------------------------------------
scripts/18's sample_canopy_controls draws a random tile from the FULL
leaf-on set, then rejects most draws for lacking March coverage (the March
flight covers ~29% of the footprint). That is fine for a handful of
examples; wasteful at the scale a real negative set needs. Every March tile
has March coverage by definition, so this scans the March tile list
directly and looks up the matching leaf-on tile by grid cell (tile_key, from
src/inference/dense_classical.py) — no rejection sampling, and every tile is
touched exactly once.

For each paired tile: compute NDVI drop for EVERY pixel (vectorized, ~1s per
tile — see dense_classical.py's timing), then filter down to a smallish
number of representative points per tile (--per-tile-cap) rather than using
every qualifying pixel, since neighbouring pixels are not independent
samples and a training set of literally every canopy pixel in Bellinzona
would be redundant and slow to extract features for.

THE SHADOW GATE
----------------
NDVI is a ratio of NIR and red reflectance, and shadow depresses both
unevenly — a textbook confound, not specific to this data. March (low winter
sun, long shadows) and leaf-on (high summer sun, short shadows) cast
DIFFERENT shadow patterns on the same tree, so a shadowed evergreen can show
a large apparent NDVI "drop" that is really a sun-angle artifact, not leaf
loss. Confirmed as a real risk on this data by direct visual inspection
(scripts/18 grid mode) before this script was written, not a hypothetical.

The gate: at each pixel, compare its brightness (mean R+G+B) to the LOCAL
neighbourhood mean brightness (--shadow-window-m), on BOTH dates. A point
much darker than its own surroundings is flagged as likely-shadowed and
excluded from being called a confident negative, on either date. This is a
heuristic — self-relative, not universal DN thresholds, since raw band
values here are uncalibrated ~hundreds-to-thousands range digital numbers,
not reflectance — and --shadow-ratio is deliberately exposed as a tunable,
not baked in as a fixed constant.

OUTPUT
------
A GeoJSON of harvested points, same shape as active_learning_hard_negatives
.geojson (geometry + provenance columns). scripts/15's --hard-negatives
loader only ever reads .geometry — see load_confirmed_points /
build_rows in scripts/15_extract_classical_features.py — so extra columns
here (ndvi_drop, tile, harvest_method) are free metadata, not a compatibility
risk.

--dry-run prints a threshold sweep (candidate counts at several
--drop-threshold values, computed from the SAME per-tile arrays without
re-scanning) instead of writing anything, since the right cutoff is a
judgement call the user should make by eye against scripts/18's grid output,
not a value baked into this script's default alone.

Torch-free, like the rest of the classical pipeline. Loads scripts/15 and
src/inference/dense_classical.py by file path for the same reason both of
those already do.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import Point

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_spec15 = importlib.util.spec_from_file_location(
    "extract_classical_features", _REPO_ROOT / "scripts" / "15_extract_classical_features.py"
)
_s15 = importlib.util.module_from_spec(_spec15)
_spec15.loader.exec_module(_s15)
load_tile = _s15.load_tile
glob_tiles = _s15.glob_tiles
BAND_NAMES_6 = _s15.BAND_NAMES_6
RES_M = _s15.RES_M

from src.inference.dense_classical import tile_key, valid_mask  # noqa: E402

NIR, RED, GREEN, BLUE, NDVI, CHM = 0, 1, 2, 3, 4, 5

# GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR — see scripts/18's module docstring
# for the full story: GDAL lists the CONTAINING DIRECTORY on every open by
# default, and with thousands of tiles per directory on a network filesystem
# that turns tile scanning into a run that can exceed a login-node time
# limit with no output to show it's alive. Applies to every open this
# process makes.
import os
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")


def local_brightness_and_ratio(rgb_sum: np.ndarray, canopy_mask: np.ndarray, window_m: float,
                               min_canopy_frac: float = 0.1) -> np.ndarray:
    """Per-pixel ratio of its own brightness to the MEAN BRIGHTNESS OF NEARBY
    CANOPY ONLY. < 1 means darker than the other trees around it.

    Restricted to canopy neighbours specifically, not "whatever is nearby" —
    an earlier version averaged over the whole window regardless of what was
    in it, which meant a sunlit tree standing next to a bright roof or
    driveway got its local mean pulled UP by the building and could register
    as "darker than its surroundings" despite being perfectly lit and simply
    a naturally darker-toned species. That failure mode concentrates exactly
    in garden/residential settings — trees next to houses — which is the
    population this whole harvest most needs good examples from. Masking the
    average to canopy pixels removes the building-contamination pathway: the
    comparison is now "darker than other nearby TREES", which is what a
    shadow-from-an-adjacent-tree or self-shading case actually looks like.

    Where too little canopy exists nearby to form a meaningful comparison
    (an isolated specimen with no neighbouring canopy in the window — plenty
    of ornamental palms are exactly this) the ratio defaults to 1.0 (treated
    as NOT shadowed) rather than excluded: absence of a comparison should not
    default to penalizing isolated trees, and isolated specimens are common
    for exactly the ornamental plantings this project cares about.

    uniform_filter with a truncated (constant, cval=0) window plus a
    canopy-weighted divisor gives the TRUE local canopy mean even at tile
    edges — same normalized-convolution technique as dense_classical.py's
    _truncated_window_mean, duplicated narrowly since that helper is private
    to that module.
    """
    size = max(3, 2 * round(window_m / RES_M) + 1)
    mask = canopy_mask.astype(np.float64)
    canopy_count = ndimage.uniform_filter(mask, size=size, mode="constant", cval=0.0) * (size * size)
    canopy_sum = ndimage.uniform_filter(rgb_sum * mask, size=size, mode="constant", cval=0.0) * (size * size)
    enough = canopy_count >= max(1.0, min_canopy_frac * size * size)
    local_mean = np.where(enough, canopy_sum / np.maximum(canopy_count, 1.0), rgb_sum)
    return rgb_sum / np.maximum(local_mean, 1e-6)


def load_exclusion_tree(paths: list[Path]) -> tuple[cKDTree | None, np.ndarray]:
    if not paths:
        return None, np.zeros((0, 2))
    frames = [gpd.read_file(p).to_crs("EPSG:2056") for p in paths]
    xy_arr = np.array([[g.x, g.y] for f in frames for g in f.geometry])
    if len(xy_arr) == 0:
        return None, xy_arr
    return cKDTree(xy_arr), xy_arr


def process_tile_pair(march_path: Path, leafon_path: Path, args, excl_tree: cKDTree | None,
                      thresholds_to_sweep: list[float]) -> tuple[list[dict], dict]:
    """Returns (harvested rows at args.drop_threshold, sweep counts dict)."""
    march_arr, march_tr, _ = load_tile(march_path)
    leafon_arr, leafon_tr, leafon_crs = load_tile(leafon_path)
    if march_arr.shape != leafon_arr.shape:
        print(f"  [skip] {march_path.name}: shape mismatch march {march_arr.shape} "
              f"vs leafon {leafon_arr.shape}")
        return [], {}
    # Grid-alignment assumption already relied on throughout this pipeline
    # (dense_classical.py's dense_feature_stack asserts the same thing across
    # SAME-date neighbour tiles); here it is ACROSS dates for the SAME cell,
    # which every prior cross-tile check in this project has held for.
    if not march_tr.almost_equals(leafon_tr):
        print(f"  [skip] {march_path.name}: transform mismatch vs {leafon_path.name}")
        return [], {}

    valid = valid_mask(march_arr) & valid_mask(leafon_arr)
    if not valid.any():
        return [], {}

    canopy = leafon_arr[CHM] >= args.canopy_min_m
    drop = leafon_arr[NDVI] - march_arr[NDVI]

    march_bright = march_arr[RED].astype(np.float64) + march_arr[GREEN] + march_arr[BLUE]
    leafon_bright = leafon_arr[RED].astype(np.float64) + leafon_arr[GREEN] + leafon_arr[BLUE]
    march_ratio = local_brightness_and_ratio(march_bright, canopy, args.shadow_window_m)
    leafon_ratio = local_brightness_and_ratio(leafon_bright, canopy, args.shadow_window_m)
    shadowed = (march_ratio < args.shadow_ratio) | (leafon_ratio < args.shadow_ratio)

    base_mask = valid & canopy & ~shadowed

    sweep = {t: int((base_mask & (drop >= t)).sum()) for t in thresholds_to_sweep}

    accept_mask = base_mask & (drop >= args.drop_threshold)
    rows_rc = np.argwhere(accept_mask)
    if len(rows_rc) == 0:
        return [], sweep

    # Exclude anything near an existing confirmed/scouted point — a control
    # sampled near a known palm could be an unlabeled neighbour, not a
    # negative (same reasoning as scripts/18's --exclude-points).
    if excl_tree is not None:
        xs, ys = xy(leafon_tr, rows_rc[:, 0], rows_rc[:, 1])
        xs, ys = np.atleast_1d(xs), np.atleast_1d(ys)
        d, _ = excl_tree.query(np.c_[xs, ys], k=1)
        far_enough = d >= args.min_dist_m
        rows_rc = rows_rc[far_enough]
        xs, ys = xs[far_enough], ys[far_enough]
    else:
        xs, ys = xy(leafon_tr, rows_rc[:, 0], rows_rc[:, 1])
        xs, ys = np.atleast_1d(xs), np.atleast_1d(ys)

    if len(rows_rc) == 0:
        return [], sweep

    # Cap per tile: neighbouring accepted pixels are not independent samples,
    # and a training set of every qualifying pixel in Bellinzona would be
    # both redundant and slow to run scripts/15's feature extraction over
    # (~140ms/row, dominated by a full tile re-read per point).
    rng = np.random.default_rng(hash(march_path.name) & 0xFFFFFFFF)
    keep = rng.permutation(len(rows_rc))[:args.per_tile_cap]

    out = []
    for i in keep:
        r, c = rows_rc[i]
        out.append({
            "geometry": Point(float(xs[i]), float(ys[i])),
            "tile": leafon_path.name,
            "ndvi_drop": float(drop[r, c]),
            "chm_m": float(leafon_arr[CHM, r, c]),
            "harvest_method": "march_ndvi_drop_filter",
        })
    return out, sweep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--march-dir", type=Path, required=True, help="Leaf-off flight tile directory.")
    p.add_argument("--leafon-dir", type=Path, required=True, help="Leaf-on flight tile directory to diff against.")
    p.add_argument("--exclude-points", type=Path, nargs="+", default=[],
                   help="Confirmed/scouted palm GeoJSONs to stay --min-dist-m away from.")
    p.add_argument("--in-chans", type=int, default=6, choices=[4, 6])
    p.add_argument("--canopy-min-m", type=float, default=1.0)
    p.add_argument("--drop-threshold", type=float, default=0.20,
                   help="Minimum NDVI drop (leaf-on - March) to call a location confidently "
                        "deciduous. Pick this by eye against scripts/18 grid-mode output — "
                        "confirmed palms there sit at roughly 0.05-0.15 drop, unambiguous "
                        "deciduous controls at 0.35+; the default sits in between with margin "
                        "below where palms were observed, not a value to trust blindly.")
    p.add_argument("--shadow-ratio", type=float, default=0.6,
                   help="A pixel darker than this fraction of its own local neighbourhood mean "
                        "brightness, on EITHER date, is excluded as likely-shadowed rather than "
                        "trusted as a confident negative. Heuristic, self-relative — tune it.")
    p.add_argument("--shadow-window-m", type=float, default=3.0)
    p.add_argument("--per-tile-cap", type=int, default=5,
                   help="Max harvested points per tile — keeps the set spatially diverse "
                        "rather than many near-duplicate pixels from one qualifying patch.")
    p.add_argument("--min-dist-m", type=float, default=20.0)
    p.add_argument("--max-tiles", type=int, default=None, help="Debug: stop after N tile pairs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print a threshold sweep and skip writing output.")
    p.add_argument("--output", type=Path, default=Path("data/interim/labels/harvested_deciduous_negatives.geojson"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    march_tiles = {tile_key(p): p for p in glob_tiles([args.march_dir], args.in_chans)}
    leafon_tiles = {tile_key(p): p for p in glob_tiles([args.leafon_dir], args.in_chans)}
    print(f"march tiles: {len(march_tiles)}   leaf-on tiles: {len(leafon_tiles)}")
    paired = sorted(set(march_tiles) & set(leafon_tiles))
    print(f"paired by grid cell: {len(paired)} "
          f"({len(march_tiles) - len(paired)} march tile(s) have no matching leaf-on tile)")
    if args.max_tiles:
        paired = paired[:args.max_tiles]

    excl_tree, _ = load_exclusion_tree(args.exclude_points)
    if args.exclude_points:
        print(f"excluding within {args.min_dist_m} m of {excl_tree.n if excl_tree else 0} known point(s)")

    sweep_values = sorted({0.10, 0.15, 0.20, 0.25, 0.30, 0.35, round(args.drop_threshold, 2)})
    all_rows: list[dict] = []
    sweep_totals = {t: 0 for t in sweep_values}
    t0 = time.time()
    for i, key in enumerate(paired):
        rows, sweep = process_tile_pair(march_tiles[key], leafon_tiles[key], args, excl_tree, sweep_values)
        all_rows.extend(rows)
        for t, n in sweep.items():
            sweep_totals[t] += n
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(paired)} tile pairs, {len(all_rows)} harvested so far "
                  f"({time.time()-t0:.0f}s elapsed)")

    print(f"\nscanned {len(paired)} tile pairs in {time.time()-t0:.1f}s")
    print("\nthreshold sweep — qualifying PIXELS before per-tile capping "
          "(canopy + not-shadowed, drop >= threshold):")
    for t in sweep_values:
        marker = "  <- --drop-threshold" if abs(t - args.drop_threshold) < 1e-9 else ""
        print(f"  {t:.2f}: {sweep_totals[t]:>10,d} pixels{marker}")

    print(f"\nharvested {len(all_rows)} points at drop-threshold={args.drop_threshold} "
          f"(capped at {args.per_tile_cap}/tile)")

    if args.dry_run:
        print("\n--dry-run: nothing written. Re-run without it once --drop-threshold looks right.")
        return

    if not all_rows:
        print("[warn] nothing harvested — output not written. Lower --drop-threshold or "
              "--shadow-ratio, or check --canopy-min-m isn't excluding everything.")
        return

    gdf = gpd.GeoDataFrame(all_rows, crs="EPSG:2056")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(args.output, driver="GeoJSON")
    print(f"wrote {len(gdf)} rows -> {args.output}")


if __name__ == "__main__":
    main()
