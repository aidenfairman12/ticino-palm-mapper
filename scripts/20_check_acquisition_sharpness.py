#!/usr/bin/env python
"""Quantify apparent RGB sharpness per acquisition date, across the SAME grid
cells in every date, to check whether "March looks sharper than leaf-on"
(observed by eye in scripts/18's grid output) is a consistent, real
difference between flights, or just a couple of unlucky tiles.

Metric: variance of the Laplacian, per RGB band, averaged. This is the
standard blur-detection heuristic (equivalent to cv2.Laplacian().var(),
just via scipy) — a sharp image has a lot of high-frequency detail, so a
second-derivative filter has large magnitude in many places and high
variance; a blurred image has little high-frequency content left and a
correspondingly low variance. Unlike the shadow gate in scripts/19, this
IS a standard, widely-used technique, not an ad hoc heuristic.

Only tile cells present in EVERY given --tile-dirs directory are compared,
so the same ~50 physical locations are being judged under each date — a
date-level sharpness difference on the SAME ground is the signal that
would indicate a real acquisition difference, as opposed to two random
samples that just happen to look different because they show different
scenes (city vs open field, say).

Torch-free, like the rest of the classical pipeline.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_spec15 = importlib.util.spec_from_file_location(
    "extract_classical_features", _REPO_ROOT / "scripts" / "15_extract_classical_features.py"
)
_s15 = importlib.util.module_from_spec(_spec15)
_spec15.loader.exec_module(_s15)
load_tile = _s15.load_tile
glob_tiles = _s15.glob_tiles

from src.inference.dense_classical import tile_key  # noqa: E402

import os
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

RED, GREEN, BLUE = 1, 2, 3


def laplacian_sharpness(band: np.ndarray) -> float:
    """Variance of the Laplacian — higher means more high-frequency detail,
    i.e. sharper. A uniformly blurred copy of the same image has a lower
    value than the original; that relationship (not the raw number, which
    has no inherent scale) is what makes this a useful comparison metric.
    """
    lap = ndimage.laplace(band.astype(np.float64))
    return float(lap.var())


def tile_sharpness(path: Path) -> float:
    arr, _, _ = load_tile(path)
    return float(np.mean([laplacian_sharpness(arr[b]) for b in (RED, GREEN, BLUE)]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile-dirs", type=Path, nargs="+", required=True,
                   help="One directory per acquisition date (date read from the directory name, "
                        "e.g. feature_stack_rs_20210324), 2 or more.")
    p.add_argument("--in-chans", type=int, default=6, choices=[4, 6])
    p.add_argument("--n-tiles", type=int, default=50,
                   help="How many grid cells common to EVERY given directory to compare.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    by_dir: dict[Path, dict[str, Path]] = {}
    for d in args.tile_dirs:
        tiles = {tile_key(p): p for p in glob_tiles([d], args.in_chans)}
        by_dir[d] = tiles
        print(f"{d.name}: {len(tiles)} tiles")

    common = set.intersection(*(set(t) for t in by_dir.values()))
    print(f"\ngrid cells present in ALL {len(args.tile_dirs)} directories: {len(common)}")
    if len(common) < 5:
        raise SystemExit("fewer than 5 common cells — directories may cover different areas "
                          "entirely, or --in-chans doesn't match one of them")

    rng = np.random.default_rng(args.seed)
    sample = rng.choice(sorted(common), size=min(args.n_tiles, len(common)), replace=False)
    print(f"sampling {len(sample)} common cells\n")

    results: dict[Path, list[float]] = {d: [] for d in args.tile_dirs}
    for key in sample:
        for d in args.tile_dirs:
            results[d].append(tile_sharpness(by_dir[d][key]))

    print(f"{'directory':<35}{'n':>5}{'mean':>12}{'median':>12}{'std':>12}")
    means = {}
    for d in args.tile_dirs:
        vals = np.array(results[d])
        means[d] = vals.mean()
        print(f"{d.name:<35}{len(vals):>5}{vals.mean():>12.1f}{np.median(vals):>12.1f}{vals.std():>12.1f}")

    ranked = sorted(means, key=means.get, reverse=True)
    print(f"\nsharpest to blurriest (by mean): {' > '.join(d.name for d in ranked)}")
    spread = (means[ranked[0]] - means[ranked[-1]]) / means[ranked[-1]] * 100
    print(f"gap between sharpest and blurriest: {spread:.0f}% relative to the blurriest")
    print("\nA consistent ranking across ~50 varied, real locations (not just the one tile "
          "eyeballed earlier) is what would confirm a genuine cross-flight difference, as "
          "opposed to a couple of unlucky tiles.")


if __name__ == "__main__":
    main()
