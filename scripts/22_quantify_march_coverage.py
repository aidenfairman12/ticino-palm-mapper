"""Quantify how much of the confirmed (+ scouted) palm population falls
inside vs outside the March 2021 (leaf-off) delivery's footprint.

Context: scripts/19's deciduous-negative harvest only draws from inside the
March footprint. checkpoint-2026-09-18-labeling-ceiling.md (memory) found
that all three validation tiles have ZERO March coverage and that March
covers only 8/96 palms in an earlier, partial count. This script re-checks
that against the FULL confirmed+scouted population using every points file
in data/interim/labels/, not just the leaf-on-set sample used earlier.

Tile grid geometry (derived from HPC rasters, EPSG:2056, verified against
active_learning_confirmed_palms.geojson's recorded `tile` property — 97/98
exact matches, one boundary case within float tolerance):
    origin (c=0, r=0 tile's top-left): x0=2715098.0, y0=1120181.0
    stride=89.6 m between adjacent cell origins, tile size=102.4 m
    (tiles overlap by 12.8 m, so a point can legitimately fall in >1 cell)

Needs data/interim/labels/_tile_cell_manifest.csv, a `date_dir,cell_key` list
pulled from the HPC scratch tile directories (basic `ls`, not a job):
    ssh af26g813@submit01.unibe.ch '
      cd /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir
      for d in feature_stack_rs_20210324 feature_stack_rs_20210811 feature_stack_rs_20240821; do
        ls $d | grep -oE "c[0-9]+_r[0-9]+" | sed "s/^/$d,/"
      done'
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS_DIR = REPO_ROOT / "data" / "interim" / "labels"
MANIFEST_PATH = LABELS_DIR / "_tile_cell_manifest.csv"

X0, Y0, STRIDE, SIZE = 2715098.0, 1120181.0, 89.6, 102.4
MARCH_DIR = "feature_stack_rs_20210324"

CONFIRMED_FILES = [
    "lugano_MASTER_confirmed_palms.geojson",       # merges batch2/3/4 + example
    "active_learning_confirmed_palms.geojson",
]
SCOUTED_FILES = [
    "lugano_new_candidates_scouted.geojson",
]


def load_manifest() -> dict[str, set[str]]:
    by_date: dict[str, set[str]] = defaultdict(set)
    for line in MANIFEST_PATH.read_text().splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        date_dir, cell = line.split(",", 1)
        by_date[date_dir].add(cell)
    return by_date


def covering_cells(x: float, y: float) -> list[tuple[int, int]]:
    c_lo = math.ceil((x - X0 - SIZE) / STRIDE - 1e-6)
    c_hi = math.floor((x - X0) / STRIDE + 1e-6)
    r_lo = math.ceil((Y0 - y - SIZE) / STRIDE - 1e-6)
    r_hi = math.floor((Y0 - y) / STRIDE + 1e-6)
    return [(c, r) for c in range(c_lo, c_hi + 1) for r in range(r_lo, r_hi + 1)]


def cell_key(c: int, r: int) -> str:
    return f"c{c:03d}_r{r:03d}"


def load_points(fname: str) -> list[tuple[str, float, float]]:
    path = LABELS_DIR / fname
    data = json.loads(path.read_text())
    out = []
    for feat in data["features"]:
        pid = feat["properties"].get("id") or feat["properties"].get("tile") or "?"
        x, y = feat["geometry"]["coordinates"][:2]
        out.append((f"{fname}:{pid}", x, y))
    return out


def classify(points: list[tuple[str, float, float]], by_date: dict[str, set[str]]):
    all_cells: set[str] = set()
    for cells in by_date.values():
        all_cells |= cells
    march_cells = by_date[MARCH_DIR]

    in_march, in_footprint_not_march, outside_footprint = [], [], []
    for pid, x, y in points:
        cells = {cell_key(c, r) for c, r in covering_cells(x, y)}
        if cells & march_cells:
            in_march.append(pid)
        elif cells & all_cells:
            in_footprint_not_march.append(pid)
        else:
            outside_footprint.append(pid)
    return in_march, in_footprint_not_march, outside_footprint


def report(label: str, points, by_date):
    in_march, in_fp_not_march, outside = classify(points, by_date)
    n = len(points)
    print(f"\n=== {label} (n={n}) ===")
    print(f"  inside March footprint:      {len(in_march):4d}  ({100*len(in_march)/n:.1f}%)")
    print(f"  in AOI but outside March:    {len(in_fp_not_march):4d}  ({100*len(in_fp_not_march)/n:.1f}%)")
    print(f"  outside bellinzona_full_nir footprint entirely: {len(outside):4d}  ({100*len(outside)/n:.1f}%)")
    if outside:
        print(f"    (likely lugano_example / other-AOI points, not bellinzona): {outside[:5]}{'...' if len(outside) > 5 else ''}")
    return in_march, in_fp_not_march, outside


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"missing {MANIFEST_PATH} — pull it via the ssh command in this script's docstring first"
        )
    by_date = load_manifest()
    for d, cells in by_date.items():
        print(f"{d}: {len(cells)} tiles")

    confirmed = []
    for f in CONFIRMED_FILES:
        confirmed.extend(load_points(f))
    seen_xy = set()
    deduped = []
    for pid, x, y in confirmed:
        key = (round(x, 1), round(y, 1))
        if key in seen_xy:
            continue
        seen_xy.add(key)
        deduped.append((pid, x, y))
    if len(deduped) != len(confirmed):
        print(f"\n(dropped {len(confirmed) - len(deduped)} exact-coordinate duplicates across confirmed files)")

    report("Confirmed palms (all sources, deduped)", deduped, by_date)

    scouted = []
    for f in SCOUTED_FILES:
        scouted.extend(load_points(f))
    if scouted:
        report("Scouted candidates (not yet confirmed)", scouted, by_date)


if __name__ == "__main__":
    main()
