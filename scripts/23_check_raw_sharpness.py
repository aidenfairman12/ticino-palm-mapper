"""Quick, standalone sharpness check on the RAW delivered strips themselves
(data/raw/swissimage_rs/bellinzona_delivery_2026-07), before investing effort in
running 07_build_nir_stack.py for the three never-processed leaf-on dates
(2022-07-15, 2024-07-20, 2024-08-10).

scripts/20_check_acquisition_sharpness.py already established, on the
PROCESSED feature_stack_rs_* tiles, that flight line 12504 (2021-08-11,
2024-08-21) is ~50-60x blurrier than line 12501 (2021-03-24) by variance-of-
Laplacian. All three never-processed dates are ALSO line 12504 (confirmed via
the delivery's own SIRS_footprint .dbf LINE_UUID field), so the prior is that
they're equally blurry, not better — but that is an assumption worth checking
directly rather than assuming, especially since it changes whether it's worth
running the full NIR-stack pipeline on them at all.

Reads small windows directly from the raw strips via rasterio windowed reads
(the strips are ~1-1.3 GB each; this never loads a whole one into memory) and
applies the exact same metric as scripts/20 (mean per-band variance of the
Laplacian over R/G/B) so the numbers are comparable to that script's published
baseline (12501 ~137,909; 12504 ~2,400-2,900).

Raw band order is [NIR, R, G, B] uint16 (confirmed against
scripts/07_build_nir_stack.py's rs_arr[0..3] unpacking) — R/G/B are bands 1,2,3
(0-indexed after NIR).

IMPORTANT: this must sample the SAME ground locations across every date, not
independent random windows per date — scripts/21's first version made that
exact mistake and had to be redone with --paired, because Laplacian variance
is dominated by scene content (a rooftop edge vs. open canopy), not focus,
unless location is held fixed. This script finds candidate points inside the
bounding-box overlap of every requested date's strips, keeps only points
actually covered (non-nodata) by EVERY date, and reads a same-centered window
from each date's own raster via its own affine transform.

Run on the HPC login node, where rasterio + these raw files both live:
    python3 scripts/23_check_raw_sharpness.py \
        --rs-dir data/raw/swissimage_rs/bellinzona_delivery_2026-07 \
        --dates 20210324 20210811 20240821 20220715 20240720 20240810
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

WIN = 512  # window side in pixels, centered on each sampled point
N_POINTS = 30
RED, GREEN, BLUE = 1, 2, 3  # 0-indexed band position after NIR (band 0)


def laplacian_sharpness(band: np.ndarray) -> float:
    return float(ndimage.laplace(band.astype(np.float64)).var())


def open_strips(rs_dir: Path, date: str) -> list[rasterio.DatasetReader]:
    return [rasterio.open(p) for p in sorted(rs_dir.glob(f"{date}_*.tif"))]


def read_window_at(strips: list[rasterio.DatasetReader], x: float, y: float, half: int) -> np.ndarray | None:
    """RGB window of side 2*half centered at (x, y) in map coords, from
    whichever strip covers it, or None if no strip covers it with margin."""
    for src in strips:
        row, col = src.index(x, y)
        if row - half < 0 or col - half < 0 or row + half > src.height or col + half > src.width:
            continue
        window = Window(col - half, row - half, 2 * half, 2 * half)
        arr = src.read([RED + 1, GREEN + 1, BLUE + 1], window=window)
        if (arr == 0).all():
            continue
        return arr
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rs-dir", type=Path, required=True)
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    by_date = {d: open_strips(args.rs_dir, d) for d in args.dates}
    for d, strips in by_date.items():
        print(f"{d}: {len(strips)} strips")

    # overlap of bounding boxes across all dates
    minx = max(min(s.bounds.left for s in strips) for strips in by_date.values())
    maxx = min(max(s.bounds.right for s in strips) for strips in by_date.values())
    miny = max(min(s.bounds.bottom for s in strips) for strips in by_date.values())
    maxy = min(max(s.bounds.top for s in strips) for strips in by_date.values())
    if minx >= maxx or miny >= maxy:
        raise SystemExit("no bounding-box overlap across the given dates' strips")
    print(f"\nbbox overlap: x[{minx:.0f},{maxx:.0f}] y[{miny:.0f},{maxy:.0f}]")

    half = WIN // 2
    margin = half * 0.1 + 5  # meters, at 0.1 m/px
    rng = np.random.default_rng(args.seed)
    results: dict[str, list[float]] = {d: [] for d in args.dates}
    n_found = 0
    attempts = 0
    while n_found < N_POINTS and attempts < N_POINTS * 40:
        attempts += 1
        x = rng.uniform(minx + margin, maxx - margin)
        y = rng.uniform(miny + margin, maxy - margin)
        crops = {d: read_window_at(strips, x, y, half) for d, strips in by_date.items()}
        if any(c is None for c in crops.values()):
            continue
        n_found += 1
        for d, arr in crops.items():
            results[d].append(float(np.mean([laplacian_sharpness(arr[b]) for b in range(3)])))

    print(f"\npaired locations found: {n_found} (of {attempts} attempts)")
    if n_found == 0:
        raise SystemExit("no location was covered by every requested date — dates may not overlap enough")

    print(f"{'date':<12}{'n':>5}{'mean':>14}{'median':>14}")
    for d in args.dates:
        vals = np.array(results[d])
        print(f"{d:<12}{len(vals):>5}{vals.mean():>14.1f}{np.median(vals):>14.1f}")

    ranked = sorted(results, key=lambda d: np.mean(results[d]), reverse=True)
    print(f"\nsharpest to blurriest (raw strips, SAME {n_found} locations): {' > '.join(ranked)}")


if __name__ == "__main__":
    main()
