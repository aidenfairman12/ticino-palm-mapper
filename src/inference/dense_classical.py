"""
Per-pixel ("dense") scoring with the trained classical RF/XGBoost model,
writing one probability GeoTIFF per tile instead of a point GeoJSON.

Why this exists: score_candidates_classical.py evaluates the model at a
sparse set of grid LOCATIONS — one feature vector per location. That answers
"is there a palm here?" but cannot answer "how many palms are in this blob?",
because there is no surface to find peaks in. This module evaluates the SAME
trained model at every pixel of a tile, producing a continuous probability
surface you can open in QGIS on top of the confirmed-palm points and look at.

The immediate question it exists to answer: at a known multi-palm cluster
(c010_r060 holds a group of 6; c057_r040 holds a 3 and a 2; c061_r038 holds
a 3), does the surface show one broad blob or N separate local maxima? That
decides whether the point-classification framing is structurally unable to
count clusters, and therefore whether a density-regression rebuild is worth
the exhaustive annotation it would require.

NOT a density map: these are per-pixel probabilities from a model trained on
point labels. They carry no count semantics — the surface does not integrate
to a palm count, and nothing constrains neighbouring pixels to agree. Peak
structure is all you should read out of it.

HOW IT REPRODUCES scripts/15's FEATURES
---------------------------------------
Every statistic in window_stats is a sliding-window reduction, so the whole
per-point feature set has an exact whole-tile equivalent in scipy.ndimage:

    <band>_point       -> the band array itself
    <band>_mean_r<R>   -> uniform_filter, normalized by the VALID pixel count
    <band>_std_r<R>    -> sqrt(E[x^2] - E[x]^2) from two uniform_filter passes
    <band>_max_r<R>    -> maximum_filter          (mode="nearest")
    <band>_min_r<R>    -> minimum_filter          (mode="nearest")
    ndvi_contrast      -> ndvi  - ndvi_mean at the widest radius
    chm_peakiness      -> chm   - chm_mean  at the widest radius

Verified against window_stats over random pixels including tile edges: mean
and std agree to ~2e-6 (float32 noise), max and min are bit-exact.

Three details are load-bearing, and getting any of them wrong yields a
plausible-looking but wrong surface:

  1. window_stats slices a SQUARE box (r0:r1, c0:c1), not a disc, so a
     square filter of size 2*radius_px+1 matches by construction.

  2. window_stats TRUNCATES its window at tile edges (max(0, ...)/min(H, ...)),
     which shrinks the divisor. Reproducing that needs uniform_filter with
     mode="constant", cval=0 applied to both the data and a ones array, then
     dividing — NOT mode="nearest", which extends values instead of shrinking
     the count and would silently change every feature within a 5 m (50 px)
     border, ~10% of a 1024x1024 tile.
     max/min are the opposite case: mode="nearest" is exactly right there,
     because nearest-extension only replicates border values that are already
     inside the truncated window. Hence bit-exact.

  3. E[x^2] - E[x]^2 cancels catastrophically in float32 on CHM-scale values,
     so the accumulators are float64.

MULTI-DATE SEMANTICS
--------------------
score_locations_classical picks, per location, the NEWEST covering date whose
pixel is not nodata, takes all spatial features from that one date, and
computes the temporal-stability columns across EVERY non-nodata date. This
module reproduces that per pixel, which means the date supplying a pixel's
spatial features varies across the tile — dates are visited newest-first and
each pixel is filled by the first one valid there. Only one date's dense
stack is held in memory at a time (they are ~335 MB each at 1024x1024).

Nodata matches score_locations_classical's test exactly: a pixel is nodata in
a date when ALL bands are 0 there.

Requires the date grids to be pixel-identical; this is asserted per tile
rather than assumed, and the assertion is the thing to check first if a run
dies on a new AOI.

TILE OVERLAP AND n_dates_covered
-------------------------------
That feature does not count dates. build_rows fans a point across every tile
find_covering_tiles returns, and this project's tiles OVERLAP (~12% of a tile
in lugano_example), so a point in a 4-tile corner covered by 3 dates yields 12.
dense_coverage_count reproduces that by counting covering non-nodata tile
instances across the whole tile set, including neighbouring cells. Overlapping
tiles are bit-identical on shared ground, so this multiplicity leaves the
temporal std/range columns alone and affects only the count.

WHERE THIS DIVERGES FROM THE POINT PATH (inherent, not a defect here)
--------------------------------------------------------------------
score_locations_classical takes its spatial features from the FIRST covering
tile in `sorted(covering, key=lambda p: p.parent.name, reverse=True)`. That
sort orders date directories, but among several tiles of the SAME newest date
— which overlap here — it is stable, so the winner is whichever find_covering_
tiles globbed first. For a location in an overlap region that may be a
NEIGHBOURING cell's tile, and the two disagree: each truncates its windows at
its own edge, so a pixel 10 px inside this tile is 200 px inside its
neighbour, with an untruncated window there.

This module always uses the cell it is scoring. Measured on 250 random pixels
of an overlapping 3-date tile set:

    point path chose this cell's tile   -> 187 px, dense matches 187/187
    point path chose a neighbour's tile ->  63 px, dense matches  16/63

So the two agree exactly wherever the point path picks the same tile, and the
residual is entirely the arbitrary tile choice. The point path's choice is an
accident of glob order — the same ground location yields different features
depending on filename ordering — so this is a pre-existing wart, not something
the dense path introduces. A per-tile raster has no way to reproduce it and no
reason to.

Practical consequence: pixels within the widest radius (5 m = 50 px) of a tile
seam may differ from what score_candidates_classical reported there. Before
reading peak structure at a cluster, check the cluster is not within ~5 m of a
tile boundary. The clean fix, if it ever matters, is to compute features on a
mosaic of the target tile plus its neighbours and crop back — no internal
truncation at all, better than either current path.

TEMPORAL STD ESTIMATOR
----------------------
scripts/15 previously disagreed with itself: add_temporal_features (training)
used pandas .std() = ddof=1, while temporal_features_from_dicts (scoring) used
np.std() = ddof=0 — a factor of sqrt(2) apart at the two dates most points
have. scripts/15 now uses ddof=0 on both paths, so a features CSV regenerated
after that fix matches what this module and score_candidates_classical compute.
--temporal-ddof 1 reproduces the old training behaviour for comparison against
a model fitted on a pre-fix CSV.

Torch-free, like everything else on this pipeline: torch and xgboost loaded
in the same process segfault (OpenMP runtime conflict). scripts/15 is loaded
by file path because a module name cannot start with a digit.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage, signal

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "extract_classical_features", _REPO_ROOT / "scripts" / "15_extract_classical_features.py"
)
_extract_classical_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_extract_classical_features)
load_tile = _extract_classical_features.load_tile
glob_tiles = _extract_classical_features.glob_tiles
BAND_NAMES_4 = _extract_classical_features.BAND_NAMES_4
BAND_NAMES_6 = _extract_classical_features.BAND_NAMES_6
RES_M = _extract_classical_features.RES_M

from src.inference.score_candidates_classical import RADII_M, build_production_model  # noqa: E402

# c010_r060 -> the grid cell a tile covers. Date directories repeat the same
# cell, but the rest of the filename can carry an AOI/year token that differs,
# so tiles are grouped on this rather than on the full filename.
TILE_KEY_RE = re.compile(r"(c\d+_r\d+)")


def tile_key(path: Path) -> str:
    """The c###_r### grid cell a tile covers — its identity across date dirs."""
    m = TILE_KEY_RE.search(path.name)
    if m is None:
        raise ValueError(f"no c###_r### grid key in tile name: {path.name}")
    return m.group(1)


def group_tiles_by_cell(tile_paths: list[Path]) -> dict[str, list[Path]]:
    """Group tiles by grid cell, each group ordered NEWEST DATE FIRST.

    Date ordering is by parent directory name (feature_stack_rs_20240821 >
    ..._20210811 > ..._20210324), matching score_locations_classical's
    `sorted(covering, key=lambda p: p.parent.name, reverse=True)`.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in tile_paths:
        groups[tile_key(p)].append(p)
    return {k: sorted(v, key=lambda p: p.parent.name, reverse=True) for k, v in groups.items()}


def valid_mask(arr: np.ndarray) -> np.ndarray:
    """(H, W) bool — False where the pixel is nodata in this date.

    Matches score_locations_classical's `(arr[:, row, col] == 0).all()`.
    """
    return ~(arr == 0).all(axis=0)


def _valid_mask_cached(path: Path, cache: "OrderedDict[Path, np.ndarray]", max_cached: int = 300) -> np.ndarray:
    """valid_mask(path), memoized. Only the bool mask is retained (~1 MB per
    1024x1024 tile), not the ~25 MB array, so a whole neighbourhood stays
    cached cheaply across successive target tiles."""
    if path in cache:
        cache.move_to_end(path)
        return cache[path]
    arr, _, _ = load_tile(path)
    cache[path] = valid_mask(arr)
    if len(cache) > max_cached:
        cache.popitem(last=False)
    return cache[path]


def _pixel_offset(ref_transform, other_transform) -> tuple[int, int]:
    """(row, col) offset to add to a ref-frame index to index `other`.

    Requires the two grids to differ by a whole number of pixels; tiles cut
    from a common mosaic do, and the assert catches an AOI where they don't.
    """
    dc = (ref_transform.c - other_transform.c) / ref_transform.a
    dr = (other_transform.f - ref_transform.f) / ref_transform.e
    assert abs(dc - round(dc)) < 1e-6 and abs(dr - round(dr)) < 1e-6, (
        f"tile grids are not whole-pixel aligned (offset {dr:.4f}, {dc:.4f} px) — "
        "coverage counting assumes a shared pixel grid"
    )
    return int(round(dr)), int(round(dc))


def dense_coverage_count(
    ref_transform, shape: tuple[int, int], candidate_paths: list[Path],
    cache: "OrderedDict[Path, np.ndarray]",
) -> np.ndarray:
    """Per-pixel count of covering, non-nodata TILE INSTANCES — the dense
    equivalent of n_dates_covered.

    Despite the name, that feature does not count dates. scripts/15's
    build_rows fans a point across every tile returned by find_covering_tiles,
    and this project's tiles OVERLAP (~12% of a tile's area in lugano_example),
    so a point in a 4-tile corner covered by 3 dates yields n_dates_covered=12,
    not 3. Counting only the target cell's dates would hand the model a feature
    in the wrong range — the model was fitted on values up to 12.

    Overlapping tiles are bit-identical on shared ground (verified on
    lugano_example), so this multiplicity does NOT disturb the temporal
    std/range columns: replicating every date's value the same number of times
    leaves min, max and the ddof=0 std unchanged. It only affects this count.
    (Under pandas' old ddof=1 it would have perturbed the std too — one more
    reason scripts/15 now uses ddof=0.)

    Note this makes n_dates_covered partly a TILING artifact: it encodes how
    close a location sits to a tile seam, which has nothing to do with palms.
    Reproduced faithfully here rather than silently corrected, since the
    trained model depends on it — but it is worth revisiting as a feature.
    """
    H, W = shape
    left, top = ref_transform.c, ref_transform.f
    right, bottom = left + W * ref_transform.a, top + H * ref_transform.e
    count = np.zeros((H, W), dtype=np.float32)

    for p in candidate_paths:
        with rasterio.open(p) as src:
            b, other_transform, (oh, ow) = src.bounds, src.transform, (src.height, src.width)
        if b.left >= right or b.right <= left or b.bottom >= top or b.top <= bottom:
            continue  # no shared ground

        dr, dc = _pixel_offset(ref_transform, other_transform)
        # rows r of the target map to r+dr in `other`; keep those in range of both
        r0, r1 = max(0, -dr), min(H, oh - dr)
        c0, c1 = max(0, -dc), min(W, ow - dc)
        if r0 >= r1 or c0 >= c1:
            continue
        other_valid = _valid_mask_cached(p, cache)
        count[r0:r1, c0:c1] += other_valid[r0 + dr:r1 + dr, c0 + dc:c1 + dc]

    return count


def _truncated_window_mean(band: np.ndarray, size: int, counts: np.ndarray) -> np.ndarray:
    """Mean over a square window truncated at the array edge.

    uniform_filter divides by the full window area; multiplying back out
    recovers the window SUM, and dividing by the count of in-bounds pixels
    (precomputed once per radius, see _window_counts) gives the truncated-
    window mean that window_stats computes.
    """
    return ndimage.uniform_filter(band, size=size, mode="constant", cval=0.0) * (size * size) / counts


def _window_counts(shape: tuple[int, int], size: int) -> np.ndarray:
    """In-bounds pixel count of the square window centered on each pixel."""
    ones = np.ones(shape, dtype=np.float64)
    return ndimage.uniform_filter(ones, size=size, mode="constant", cval=0.0) * (size * size)

def _wedge_masks(radius_px: int, n_sectors: int = 8) -> list[np.ndarray]:
    """8 boolean wedge masks, same disc/angle definition as
    chm_radial_asymmetry, sized to enclose exactly the radius_px disc."""
    size = 2 * radius_px + 1
    rows, cols = np.indices((size, size))
    center = radius_px
    dy, dx = rows - center, cols - center
    dist = np.hypot(dy, dx)
    valid = (dist > 0) & (dist <= radius_px)
    angle = np.arctan2(dy, dx) % (2 * np.pi)
    sector_width = 2 * np.pi / n_sectors
    sector_idx = np.minimum((angle // sector_width).astype(int), n_sectors - 1)
    
    return [(valid & (sector_idx == s)).astype(np.float64) for s in range(n_sectors)]



def dense_spatial_features(arr: np.ndarray, radii_m: list[float]) -> dict[str, np.ndarray]:
    """Whole-tile equivalent of extract_features_from_array, minus the
    temporal columns (which need more than one date — see dense_temporal_features).

    Returns {feature_name: (H, W) float32}, with exactly the names
    extract_features_from_array produces for the same `radii_m`.
    """
    band_names = BAND_NAMES_6 if arr.shape[0] == 6 else BAND_NAMES_4
    H, W = arr.shape[1], arr.shape[2]
    feats: dict[str, np.ndarray] = {}
    
    if "chm" in band_names:
        chm_laplacian = ndimage.laplace(arr[band_names.index("chm")].astype(np.float64))
        
    for r in radii_m:
        radius_px = max(1, round(r / RES_M))  # same as window_stats
        size = 2 * radius_px + 1
        counts = _window_counts((H, W), size)

        for i, name in enumerate(band_names):
            band = arr[i].astype(np.float64)  # float64: E[x^2]-E[x]^2 cancels badly in float32

            if r == radii_m[0]:
                feats[f"{name}_point"] = arr[i].astype(np.float32)

            mean = _truncated_window_mean(band, size, counts)
            mean_sq = _truncated_window_mean(band * band, size, counts)
            std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))  # clamp float noise below 0

            feats[f"{name}_mean_r{r}"] = mean.astype(np.float32)
            feats[f"{name}_std_r{r}"] = std.astype(np.float32)
            # mode="nearest" is exact for max/min: extending the border can only
            # replicate values already inside the truncated window.
            feats[f"{name}_max_r{r}"] = ndimage.maximum_filter(arr[i], size=size, mode="nearest")
            feats[f"{name}_min_r{r}"] = ndimage.minimum_filter(arr[i], size=size, mode="nearest")
            
            if name == "chm":
                chm_band, chm_mean, chm_mean_sq, chm_std = band, mean, mean_sq, std

        if "chm" in band_names:
            max_r = feats[f"chm_max_r{r}"]
            mean_r = feats[f"chm_mean_r{r}"]
            feats[f"chm_peak_ratio_r{r}"] = np.where(mean_r > 0, max_r / mean_r, 0.0).astype(np.float32)
            
            mean_cube = _truncated_window_mean(chm_band ** 3, size, counts)
            mean_quad = _truncated_window_mean(chm_band ** 4, size, counts)

            m2 = chm_std * chm_std  # already float64, already clamped >= 0
            m3 = mean_cube - 3 * chm_mean * chm_mean_sq + 2 * chm_mean ** 3
            m4 = mean_quad - 4 * chm_mean * mean_cube + 6 * chm_mean ** 2 * chm_mean_sq - 3 * chm_mean ** 4

            flat = m2 <= 1e-6  # chm variance below (1mm)^2 — treat as flat; catches the
                    # float64 cancellation residual a perfectly flat window
                    # leaves after two independent convolution passes, which
                    # the point path never sees (window.std() on one array
                    # cancels exactly, this doesn't)
            safe_m2 = np.where(flat, 1.0, m2)  # placeholder denominator, discarded by the outer where
            skew = np.where(flat, 0.0, m3 / safe_m2 ** 1.5)
            kurt = np.where(flat, 0.0, m4 / safe_m2 ** 2 - 3.0)

            feats[f"chm_skew_r{r}"] = skew.astype(np.float32)
            feats[f"chm_kurtosis_r{r}"] = kurt.astype(np.float32)
            
            lap_mean = _truncated_window_mean(chm_laplacian, size, counts)
            lap_mean_sq = _truncated_window_mean(chm_laplacian ** 2, size, counts)
            feats[f"chm_local_roughness_r{r}"] = np.sqrt(np.maximum(lap_mean_sq - lap_mean * lap_mean, 0.0)).astype(np.float32)
            
            masks = _wedge_masks(radius_px)
            ones = np.ones((H, W), dtype=np.float64)
            wedge_means, wedge_valid = [], []
            for mask in masks:
                # fftconvolve does true convolution (flips the kernel); flipping the
                # mask first cancels that, giving correlation — matching the point
                # path's window[mask] lookup, which is unflipped by construction.
                flipped = mask[::-1, ::-1]
                wsum = signal.fftconvolve(chm_band, flipped, mode="same")
                wcount = signal.fftconvolve(ones, flipped, mode="same")
                valid = wcount > 0.5  # a wedge with no in-bounds pixels here (tile corners)
                safe_count = np.where(valid, wcount, 1.0)
                wedge_means.append(np.where(valid, wsum / safe_count, 0.0))
                wedge_valid.append(valid)

            wedge_stack = np.stack(wedge_means, axis=0)  # (8, H, W)
            valid_stack = np.stack(wedge_valid, axis=0)  # (8, H, W) bool
            n_valid = valid_stack.sum(axis=0)
            enough = n_valid >= 2  # matches the point path's `len(wedge_means) < 2: return 0.0`
            safe_n = np.where(enough, n_valid, 1)

            # masked mean/variance across ONLY the valid sectors at each pixel --
            # an invalid sector's fabricated 0.0 above must not leak into either
            # the mean or the variance, or a fake 0.0 sitting among several-metre
            # real heights would inflate asymmetry right at tile corners.
            masked_vals = np.where(valid_stack, wedge_stack, 0.0)
            valid_mean = masked_vals.sum(axis=0) / safe_n
            sq_dev = np.where(valid_stack, (wedge_stack - valid_mean) ** 2, 0.0)
            variance = sq_dev.sum(axis=0) / safe_n

            feats[f"chm_asymmetry_r{r}"] = np.where(enough, variance, 0.0).astype(np.float32)
 
            row_idx, col_idx = np.indices((H, W))
            feats[f"chm_window_clipped_r{r}"] = (
                    (row_idx < radius_px) | (row_idx >= H - radius_px)|
                    (col_idx < radius_px) | (col_idx >= W - radius_px)
                    ).astype(np.float32)

    widest = radii_m[-1]
    if "ndvi" in band_names:
        feats["ndvi_contrast"] = feats["ndvi_point"] - feats[f"ndvi_mean_r{widest}"]
    if "chm" in band_names:
        feats["chm_peakiness"] = feats["chm_point"] - feats[f"chm_mean_r{widest}"]
    return feats


def dense_temporal_features(
    date_arrs: list[np.ndarray], masks: list[np.ndarray], band_names: list[str], ddof: int = 0
) -> dict[str, np.ndarray]:
    """Whole-tile equivalent of temporal_features_from_dicts.

    Pixelwise std/range of ndvi and chm across every date valid AT THAT PIXEL,
    plus the count of such dates. `<band>_point` densely is just the band, so
    the per-date values being reduced over are the band arrays themselves.

    ddof=0 matches temporal_features_from_dicts (np.std). See the module
    docstring — scripts/15's TRAINING path uses pandas .std(), i.e. ddof=1.
    """
    H, W = date_arrs[0].shape[1], date_arrs[0].shape[2]
    n = np.zeros((H, W), dtype=np.float64)
    acc = {b: {"sum": np.zeros((H, W)), "sumsq": np.zeros((H, W)),
               "max": np.full((H, W), -np.inf), "min": np.full((H, W), np.inf)}
           for b in ("ndvi", "chm") if b in band_names}

    for arr, mask in zip(date_arrs, masks):
        n += mask
        for b, a in acc.items():
            band = np.where(mask, arr[band_names.index(b)].astype(np.float64), 0.0)
            a["sum"] += band
            a["sumsq"] += band * band
            a["max"] = np.where(mask, np.maximum(a["max"], band), a["max"])
            a["min"] = np.where(mask, np.minimum(a["min"], band), a["min"])

    out: dict[str, np.ndarray] = {"n_dates_covered": n.astype(np.float32)}
    # A single date has no spread: temporal_features_from_dicts returns 0.0,
    # not NaN — "no variation observed" is the honest value, same reasoning as
    # scripts/15's fillna(0.0).
    multi = n > 1
    seen = n > 0
    safe_n = np.where(seen, n, 1.0)
    for b, a in acc.items():
        mean = a["sum"] / safe_n
        var = np.maximum(a["sumsq"] / safe_n - mean * mean, 0.0)
        if ddof == 1:
            var = var * np.where(multi, safe_n / np.maximum(safe_n - 1.0, 1.0), 0.0)
        out[f"{b}_temporal_std"] = np.where(multi, np.sqrt(var), 0.0).astype(np.float32)
        out[f"{b}_temporal_range"] = np.where(seen, a["max"] - a["min"], 0.0).astype(np.float32)
    return out


def dense_feature_stack(
    date_paths: list[Path], radii_m: list[float], temporal_ddof: int = 0,
    coverage_paths: list[Path] | None = None,
    coverage_cache: "OrderedDict[Path, np.ndarray] | None" = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, object, object]:
    """Build the full per-pixel feature stack for one grid cell.

    `date_paths` must be NEWEST FIRST (group_tiles_by_cell guarantees this).
    Spatial features come from the newest date valid at each pixel — the
    per-pixel form of score_locations_classical's per-location date choice —
    and temporal features span every date valid there.

    `coverage_paths` should be EVERY tile across every date dir, not just this
    cell's — n_dates_covered counts overlapping neighbour tiles too (see
    dense_coverage_count). Omitting it falls back to counting this cell's valid
    dates, which understates the feature wherever tiles overlap.

    Returns (features, valid_mask, transform, crs). valid_mask is False where
    no date had data, i.e. where the model must not be asked for a prediction.
    """
    date_arrs, masks = [], []
    ref_transform = ref_crs = ref_shape = None
    for p in date_paths:
        arr, transform, crs = load_tile(p)
        if ref_transform is None:
            ref_transform, ref_crs, ref_shape = transform, crs, arr.shape
        else:
            # Pixelwise date reductions are meaningless on misaligned grids.
            # Fail loudly rather than silently comparing different ground.
            assert arr.shape == ref_shape, f"shape mismatch across dates for {p.name}: {arr.shape} vs {ref_shape}"
            assert transform.almost_equals(ref_transform), f"transform mismatch across dates for {p.name}"
        date_arrs.append(arr)
        masks.append(valid_mask(arr))

    band_names = BAND_NAMES_6 if ref_shape[0] == 6 else BAND_NAMES_4
    H, W = ref_shape[1], ref_shape[2]

    feats: dict[str, np.ndarray] = {}
    filled = np.zeros((H, W), dtype=bool)
    for arr, mask in zip(date_arrs, masks):  # newest first
        take = mask & ~filled
        if not take.any():
            continue
        # One date's dense stack at a time — each is ~335 MB at 1024x1024.
        date_feats = dense_spatial_features(arr, radii_m)
        for k, v in date_feats.items():
            if k not in feats:
                feats[k] = np.zeros((H, W), dtype=np.float32)
            feats[k][take] = v[take]
        filled |= take
        del date_feats

    feats.update(dense_temporal_features(date_arrs, masks, band_names, ddof=temporal_ddof))
    if coverage_paths is not None:
        cache = coverage_cache if coverage_cache is not None else OrderedDict()
        feats["n_dates_covered"] = dense_coverage_count(
            ref_transform, (H, W), coverage_paths, cache
        )
    return feats, filled, ref_transform, ref_crs


def predict_dense(model, feat_cols: list[str], feats: dict[str, np.ndarray],
                  valid: np.ndarray, chunk_rows: int = 200_000) -> np.ndarray:
    """Run the trained model over every valid pixel; NaN elsewhere.

    Columns are assembled in `feat_cols` order. XGBoost validates column names
    on a DataFrame but silently accepts a misordered ndarray, which would give
    a plausible-looking and completely meaningless surface — the same failure
    mode as the patch-embed channel misalignment. Missing columns raise here
    instead.
    """
    missing = [c for c in feat_cols if c not in feats]
    if missing:
        raise KeyError(
            f"trained model expects {len(missing)} feature(s) this stack does not provide: {missing[:8]}"
            " — check --radii-m and --in-chans match the run that produced the features CSV"
        )

    H, W = valid.shape
    flat_valid = valid.reshape(-1)
    idx = np.flatnonzero(flat_valid)
    out = np.full(H * W, np.nan, dtype=np.float32)
    if idx.size == 0:
        return out.reshape(H, W)

    cols = [feats[c].reshape(-1) for c in feat_cols]
    for start in range(0, idx.size, chunk_rows):
        sel = idx[start:start + chunk_rows]
        X = np.empty((sel.size, len(feat_cols)), dtype=np.float32)
        for j, col in enumerate(cols):
            X[:, j] = col[sel]
        out[sel] = model.predict_proba(X)[:, 1].astype(np.float32)
    return out.reshape(H, W)


def write_probability_tif(path: Path, prob: np.ndarray, transform, crs) -> None:
    """Single-band float32 GeoTIFF, NaN nodata — drops straight into QGIS on
    top of the confirmed-palm GeoJSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=prob.shape[0], width=prob.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform,
        nodata=np.nan, compress="deflate", tiled=True,
    ) as dst:
        dst.write(prob, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--features-csv", type=Path, required=True,
                   help="scripts/15 output the production model is trained on.")
    p.add_argument("--tile-dirs", type=Path, nargs="+", required=True,
                   help="One per date, e.g. feature_stack_rs_20210324 ... _20240821.")
    p.add_argument("--tiles", nargs="+", default=None,
                   help="Grid cells to score, e.g. c010_r060 c057_r040. Default: every cell found.")
    p.add_argument("--out-dir", type=Path, default=Path("dense_scores"))
    p.add_argument("--in-chans", type=int, default=6, choices=[4, 6])
    p.add_argument("--radii-m", type=float, nargs="+", default=list(RADII_M),
                   help="Must match the run that produced --features-csv.")
    p.add_argument("--temporal-ddof", type=int, default=0, choices=[0, 1],
                   help="0 matches scoring (np.std); 1 matches training (pandas .std). See module docstring.")
    p.add_argument("--no-neighbour-coverage", action="store_true",
                   help="Count only this cell's valid dates for n_dates_covered, skipping overlapping "
                        "neighbour tiles. Faster, but understates the feature near tile seams and "
                        "diverges from how the model was trained. Diagnostic use only.")
    p.add_argument("--model", choices=["rf", "xgboost"], default="xgboost")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--min-samples-leaf", type=int, default=1, help="rf only.")
    p.add_argument("--class-weight-multiplier", type=float, default=1.0)
    p.add_argument("--model-seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    hyperparams = {"n_estimators": args.n_estimators, "max_depth": args.max_depth}
    if args.model == "rf":
        hyperparams["min_samples_leaf"] = args.min_samples_leaf
    else:
        hyperparams["min_child_weight"] = args.min_child_weight

    model, feat_cols = build_production_model(
        args.features_csv, args.model, args.class_weight_multiplier, args.model_seed, **hyperparams
    )
    print(f"production {args.model} trained on {args.features_csv} ({len(feat_cols)} features)")

    all_tiles = glob_tiles(args.tile_dirs, args.in_chans)
    groups = group_tiles_by_cell(all_tiles)
    if args.tiles:
        unknown = [t for t in args.tiles if t not in groups]
        if unknown:
            raise SystemExit(f"no tiles found for grid cell(s): {unknown}")
        groups = {k: groups[k] for k in args.tiles}
    print(f"scoring {len(groups)} grid cell(s)")

    # n_dates_covered is excluded from the model as a leak (see
    # classical_probe.NON_FEATURE_COLUMNS), so normally there is nothing to
    # count and the neighbour-tile pass is skipped entirely. It still runs for
    # a model fitted on an older CSV that kept the column, where the count must
    # include overlapping neighbour cells or the feature lands in the wrong
    # range. Only bool masks are cached, so it stays cheap across tiles.
    needs_coverage = "n_dates_covered" in feat_cols
    coverage_paths = all_tiles if (needs_coverage and not args.no_neighbour_coverage) else None
    if needs_coverage:
        print("[note] this model still uses n_dates_covered — counting overlapping neighbour tiles")
    coverage_cache: OrderedDict[Path, np.ndarray] = OrderedDict()

    for key, date_paths in groups.items():
        t0 = time.time()
        feats, valid, transform, crs = dense_feature_stack(
            date_paths, args.radii_m, args.temporal_ddof, coverage_paths, coverage_cache
        )
        prob = predict_dense(model, feat_cols, feats, valid)
        out_path = args.out_dir / f"{key}_prob.tif"
        write_probability_tif(out_path, prob, transform, crs)
        finite = prob[np.isfinite(prob)]
        summary = f"max={finite.max():.3f} p99={np.percentile(finite, 99):.3f}" if finite.size else "all nodata"
        print(f"  {key}: {len(date_paths)} date(s), {valid.mean()*100:.1f}% valid, "
              f"{summary}, {time.time()-t0:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
