"""Pick new validation tiles for the 10cm-only migration.

Context: c010_r060, c057_r040, c061_r038 (used in every ablation so far) have
ZERO coverage in the two new native-10cm leaf-on dates (2024-07-20,
2024-08-10), confirmed via bounding-box intersection. This finds candidate
tiles containing a confirmed multi-palm cluster that DO have coverage in the
three native-10cm dates (March 2021 + 2024-07-20 + 2024-08-10), preferring
tiles covered by all three over tiles covered by only two.

Tile grid geometry + manifest format reused from scripts/22 (EPSG:2056,
pure-Python grid math, no geopandas/rasterio needed). Clustering reuses
scripts/17's single-linkage-at-4.0m definition of a "cluster" so results are
directly comparable to the per-cluster verdicts already reported.

Needs data/interim/labels/_tile_cell_manifest.csv refreshed to include the
two new date dirs (basic `ls`, not a job):
    ssh af26g813@submit01.unibe.ch '
      cd /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir
      for d in feature_stack_rs_20210324 feature_stack_rs_20210811 feature_stack_rs_20240821 feature_stack_rs_20240720 feature_stack_rs_20240810; do
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

NATIVE_10CM_DIRS = [
    "feature_stack_rs_20210324",  # March (leaf-off)
    "feature_stack_rs_20240720",
    "feature_stack_rs_20240810",
]

CONFIRMED_FILES = [
    "lugano_MASTER_confirmed_palms.geojson",
    "active_learning_confirmed_palms.geojson",
    "google_earth_review_batch1_confirmed_palms.geojson",
    "google_earth_review_batch2_confirmed_palms.geojson",
]

OLD_VALIDATION_TILES = {"c010_r060", "c057_r040", "c061_r038"}


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


def dedupe(points: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    seen, out = set(), []
    for pid, x, y in points:
        key = (round(x, 1), round(y, 1))
        if key in seen:
            continue
        seen.add(key)
        out.append((pid, x, y))
    return out


def cluster_points(points: list[tuple[str, float, float]], link_m: float) -> list[list[int]]:
    """Single-linkage grouping, O(n^2) — fine at label-file scale (no scipy locally)."""
    n = len(points)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    link2 = link_m * link_m
    for i in range(n):
        _, xi, yi = points[i]
        for j in range(i + 1, n):
            _, xj, yj = points[j]
            if (xi - xj) ** 2 + (yi - yj) ** 2 <= link2:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cluster-link-m", type=float, default=4.0)
    ap.add_argument("--min-cluster-size", type=int, default=2)
    args = ap.parse_args()

    if not MANIFEST_PATH.exists():
        raise SystemExit(f"missing {MANIFEST_PATH} — pull it via the ssh command in this script's docstring first")
    by_date = load_manifest()
    missing = [d for d in NATIVE_10CM_DIRS if d not in by_date]
    if missing:
        raise SystemExit(f"manifest missing date dirs: {missing} — re-pull it (see docstring)")

    points = []
    for f in CONFIRMED_FILES:
        points.extend(load_points(f))
    points = dedupe(points)
    print(f"{len(points)} deduped confirmed palms across {len(CONFIRMED_FILES)} files")

    clusters = cluster_points(points, args.cluster_link_m)
    multi = [c for c in clusters if len(c) >= args.min_cluster_size]
    print(f"{len(multi)} clusters with >= {args.min_cluster_size} palms (single-linkage @ {args.cluster_link_m} m)\n")

    candidates = []
    for idxs in multi:
        pts = [points[i] for i in idxs]
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))

        # tiles containing the FULL cluster: intersection of each point's covering cells
        common_cells = None
        for _, x, y in pts:
            cells = {cell_key(c, r) for c, r in covering_cells(x, y)}
            common_cells = cells if common_cells is None else (common_cells & cells)
        if not common_cells:
            continue  # cluster straddles a tile boundary with no single tile covering all of it

        for tile in sorted(common_cells):
            covering_dates = [d for d in NATIVE_10CM_DIRS if tile in by_date[d]]
            if len(covering_dates) < 2:
                continue  # need at least a pairwise overlap
            candidates.append({
                "tile": tile,
                "n_palms": len(pts),
                "span_m": span,
                "covering_dates": covering_dates,
                "is_old_tile": tile in OLD_VALIDATION_TILES,
                "example_ids": [p[0] for p in pts[:3]],
            })

    candidates.sort(key=lambda c: (-len(c["covering_dates"]), -c["n_palms"]))

    print(f"{'tile':<14}{'palms':>6}{'span_m':>8}  dates covered (of 3 native-10cm)")
    for c in candidates:
        flag = "  [SAME AS OLD VALIDATION TILE]" if c["is_old_tile"] else ""
        dates_short = ",".join(d.replace("feature_stack_rs_", "") for d in c["covering_dates"])
        print(f"{c['tile']:<14}{c['n_palms']:>6}{c['span_m']:>8.1f}  {dates_short}{flag}")
        print(f"    example ids: {c['example_ids']}")

    if not candidates:
        print("No multi-palm cluster has >=2-of-3 native-10cm coverage. "
              "Consider lowering --min-cluster-size or checking single-palm tiles instead.")


if __name__ == "__main__":
    main()
