#!/usr/bin/env python
"""
21_visualize_sharpness_range.py
================================
Visual sanity check on scripts/20's Laplacian-variance sharpness ranking: for
each acquisition date, render real RGB crops spanning that date's OWN
sharpness distribution from lowest to highest, so the ~50x gap reported
between March and the two leaf-on dates can be checked against actual
imagery rather than trusted as a number alone.

Reuses the same common-tile-key restriction as scripts/20 (only grid cells
present in EVERY given --tile-dirs directory), then within each date sorts
those tiles by ITS OWN sharpness score and picks --n-examples evenly spaced
by RANK — so you see that date's blurriest tile, its sharpest, and several
representative points between, not an arbitrary sample.

Each date gets its own output PNG (one row per selected tile, sorted
ascending — blurriest at top, sharpest at bottom), since a low-sharpness
tile in one date and a low-sharpness tile in another may be entirely
different physical locations (each date is ranked against itself). The row
label shows the tile's rank position and numeric score so it can be
cross-checked directly against scripts/20's output.

Crop is centred on each tile at TRUE pixel resolution (no synthetic
upsampling, unlike scripts/18's --zoom — a 20m crop is already 200x200 real
pixels, plenty to judge sharpness by eye without needing to fake detail).

Torch-free, like the rest of the classical pipeline.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_spec15 = importlib.util.spec_from_file_location(
    "extract_classical_features", _REPO_ROOT / "scripts" / "15_extract_classical_features.py"
)
_s15 = importlib.util.module_from_spec(_spec15)
_spec15.loader.exec_module(_s15)
load_tile = _s15.load_tile
glob_tiles = _s15.glob_tiles
RES_M = _s15.RES_M

_spec20 = importlib.util.spec_from_file_location(
    "check_acquisition_sharpness", _REPO_ROOT / "scripts" / "20_check_acquisition_sharpness.py"
)
_s20 = importlib.util.module_from_spec(_spec20)
_spec20.loader.exec_module(_s20)
laplacian_sharpness = _s20.laplacian_sharpness

from src.inference.dense_classical import tile_key  # noqa: E402

import os
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

RED, GREEN, BLUE = 1, 2, 3
DATE_RE = re.compile(r"(\d{8})")


def date_of(d: Path) -> str:
    m = DATE_RE.search(d.name)
    return m.group(1) if m else d.name


def stretch(band: np.ndarray, lo_hi=(2, 98)) -> np.ndarray:
    """Percentile contrast stretch to [0,1] — matches scripts/10/18's convention."""
    lo, hi = np.percentile(band, lo_hi)
    return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)


def center_crop(arr: np.ndarray, crop_px: int) -> np.ndarray:
    h, w = arr.shape[1], arr.shape[2]
    r0, c0 = (h - crop_px) // 2, (w - crop_px) // 2
    return arr[:, r0:r0 + crop_px, c0:c0 + crop_px]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile-dirs", type=Path, nargs="+", required=True,
                   help="One directory per acquisition date, 2 or more — same restriction as "
                        "scripts/20: only tiles present in EVERY directory are compared.")
    p.add_argument("--in-chans", type=int, default=6, choices=[4, 6])
    p.add_argument("--n-tiles", type=int, default=50,
                   help="How many common cells to score per date before selecting examples from "
                        "them — should match whatever --n-tiles you used with scripts/20 if you "
                        "want the SAME underlying scores, not a fresh independent sample.")
    p.add_argument("--n-examples", type=int, default=15,
                   help="How many tiles per date to render, evenly spaced by RANK across that "
                        "date's own sharpness distribution (low to high).")
    p.add_argument("--crop-m", type=float, default=20.0,
                   help="Crop side length in metres, centred on each tile. True resolution, "
                        "no upsampling — 20m is already 200x200 real pixels.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("sharpness_examples"))
    p.add_argument("--paired", action="store_true",
                   help="Instead of ranking each date independently (which mostly just "
                        "re-discovers which tiles contain buildings vs forest — Laplacian "
                        "variance is dominated by SCENE CONTENT, a tile with roof/road edges "
                        "always scores higher than a canopy tile regardless of focus), show "
                        "the SAME physical location under every date side by side. This holds "
                        "content fixed and is the direct test of whether one date is really "
                        "sharper than another, rather than just showing different scenes.")
    p.add_argument("--n-paired", type=int, default=12, help="--paired: how many locations to show.")
    return p.parse_args()




def run_paired(by_dir, sample_keys, args, crop_px):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(sample_keys[:args.n_paired])
    fig, axes = plt.subplots(len(keys), len(args.tile_dirs),
                             figsize=(4.2 * len(args.tile_dirs), 4.4 * len(keys)), squeeze=False)
    for row, key in enumerate(keys):
        for col, d in enumerate(args.tile_dirs):
            arr, _, _ = load_tile(by_dir[d][key])
            crop = center_crop(arr, crop_px)
            score = float(np.mean([laplacian_sharpness(crop[b]) for b in (RED, GREEN, BLUE)]))
            rgb = np.dstack([stretch(crop[b]) for b in (RED, GREEN, BLUE)])
            ax = axes[row][col]
            ax.imshow(rgb, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(date_of(d), fontsize=11)
            if col == 0:
                ax.set_ylabel(key, fontsize=9, rotation=0, ha="right", va="center")
            ax.text(0.5, -0.06, f"sharp={score:,.0f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=8)

    fig.suptitle("SAME physical location, every date — the direct test (content held fixed)",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path = args.out_dir / "sharpness_paired.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote paired comparison ({len(keys)} locations x {len(args.tile_dirs)} dates) -> {out_path}")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    by_dir: dict[Path, dict[str, Path]] = {}
    for d in args.tile_dirs:
        by_dir[d] = {tile_key(p): p for p in glob_tiles([d], args.in_chans)}
        print(f"{d.name}: {len(by_dir[d])} tiles")

    common = sorted(set.intersection(*(set(t) for t in by_dir.values())))
    print(f"grid cells common to all {len(args.tile_dirs)} directories: {len(common)}")
    if len(common) < 5:
        raise SystemExit("fewer than 5 common cells — see scripts/20 for the same check")

    rng = np.random.default_rng(args.seed)
    sample_keys = rng.choice(common, size=min(args.n_tiles, len(common)), replace=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    crop_px = max(4, round(args.crop_m / RES_M))

    if args.paired:
        run_paired(by_dir, sample_keys, args, crop_px)
        return

    for d in args.tile_dirs:
        scored = []
        for key in sample_keys:
            arr, _, _ = load_tile(by_dir[d][key])
            score = float(np.mean([laplacian_sharpness(arr[b]) for b in (RED, GREEN, BLUE)]))
            scored.append((score, key, arr))
        scored.sort(key=lambda t: t[0])  # ascending: blurriest first

        n = min(args.n_examples, len(scored))
        idx = np.unique(np.round(np.linspace(0, len(scored) - 1, n)).astype(int))
        picks = [scored[i] for i in idx]

        fig, axes = plt.subplots(len(picks), 1, figsize=(4.5, 4.5 * len(picks)), squeeze=False)
        for row, (rank, (score, key, arr)) in enumerate(zip(idx, picks)):
            crop = center_crop(arr, crop_px)
            rgb = np.dstack([stretch(crop[b]) for b in (RED, GREEN, BLUE)])
            ax = axes[row][0]
            ax.imshow(rgb, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            pct = rank / (len(scored) - 1) * 100 if len(scored) > 1 else 0
            ax.set_title(f"{key}   rank {rank+1}/{len(scored)} ({pct:.0f}th pct)   "
                        f"sharpness={score:,.0f}", fontsize=9)

        fig.suptitle(f"{date_of(d)}  —  sorted blurriest (top) to sharpest (bottom)", fontsize=12, y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out_path = args.out_dir / f"sharpness_{date_of(d)}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  {date_of(d)}: scores range {scored[0][0]:,.0f} - {scored[-1][0]:,.0f} "
              f"-> wrote {len(picks)} example(s) to {out_path}")


if __name__ == "__main__":
    main()
