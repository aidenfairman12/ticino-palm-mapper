#!/usr/bin/env python
"""
17_inspect_dense_surface.py
===========================
Read out a probability surface from src/inference/dense_classical.py against
the confirmed-palm points, and answer the question the dense pass exists for:
at a known multi-palm cluster, does the surface resolve into one maximum per
palm, or one blob for the whole group?

Eyeballing the raster cannot answer that — a bright patch over a cluster looks
the same whether it has one internal maximum or six. This applies the peak
detection a density-regression pipeline would use and counts.

What it does:
  1. Local-maxima detection on the surface: maximum_filter of footprint
     --nms-radius-m acts as non-maximum suppression (keep pixels equal to
     their neighbourhood max), then a --threshold cut. These are the two knobs
     a real detector would tune; they are exposed rather than buried because
     the answer genuinely depends on them, and a sweep is the honest report.
  2. Hungarian matching of peaks to ground-truth points within --match-dist-m,
     giving precision / recall / F1 and mean localization error. Greedy
     nearest-neighbour matching double-counts a peak sitting between two
     palms, which is the exact case under test, so this uses an optimal
     assignment instead.
  3. Per-cluster breakdown. Points are grouped by single-linkage at
     --cluster-link-m; for each group the report gives palms-in-group vs
     peaks-found-nearby. A 6-palm group reported as 1 peak is the merged-blob
     failure; as 6 peaks means point classification was never the blocker.
  4. Optional PNG: the whole tile plus a zoomed panel per multi-palm cluster,
     with ground-truth points and detected peaks marked.

The global precision number will look terrible and that is expected, not a
bug: the model was fitted on ~300 curated points and is here applied to ~1M
pixels of roofs, roads, cars and shadow it never saw. Read the per-cluster
counts, which are local and are what the experiment is about.

Torch-free, matching the rest of the classical pipeline.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.geometry import box


def load_points_in_tile(point_paths: list[Path], src) -> gpd.GeoDataFrame:
    """Every confirmed point falling inside this raster, de-duplicated.

    Points are pooled across files because the confirmed set is split over
    several (MASTER, active-learning, scouted) and a tile may draw from more
    than one. Exact-duplicate coordinates are dropped: the same palm is
    recorded in more than one file, and counting it twice would inflate the
    ground-truth count for precisely the clusters under test.
    """
    tile_box = box(*src.bounds)
    frames = []
    for p in point_paths:
        g = gpd.read_file(p).to_crs(src.crs)
        g = g[g.geometry.within(tile_box)]
        if len(g):
            frames.append(g[["geometry"]])
    if not frames:
        return gpd.GeoDataFrame({"geometry": []}, crs=src.crs)
    out = gpd.GeoDataFrame(
        {"geometry": [geom for f in frames for geom in f.geometry]}, crs=src.crs
    )
    return out.drop_duplicates(subset=["geometry"]).reset_index(drop=True)


def find_peaks(prob: np.ndarray, nms_radius_px: int, threshold: float) -> np.ndarray:
    """(N, 2) array of (row, col) local maxima above `threshold`.

    A pixel survives when it equals the maximum of its neighbourhood, which is
    non-maximum suppression: within any nms_radius_px window only the top
    pixel is kept, so one broad blob yields one peak unless it has genuinely
    separate internal maxima. That is the property being tested.

    NaN (nodata) is replaced by -inf so it neither wins a comparison nor
    poisons its neighbourhood, as NaN would propagate through maximum_filter.
    """
    filled = np.where(np.isfinite(prob), prob, -np.inf)
    size = 2 * nms_radius_px + 1
    local_max = ndimage.maximum_filter(filled, size=size, mode="nearest")
    peaks = (filled == local_max) & (filled >= threshold)
    return np.argwhere(peaks)


def match_peaks(peak_xy: np.ndarray, true_xy: np.ndarray, max_dist_m: float):
    """Optimal one-to-one assignment within max_dist_m.

    Hungarian rather than greedy: a single peak sitting midway between two
    adjacent palms is exactly the merged-blob case, and greedy matching would
    let it claim both, reporting perfect recall for a detector that found one
    object where there were two.

    Returns (matched pairs, unmatched truth idx, unmatched peak idx).
    """
    if len(peak_xy) == 0 or len(true_xy) == 0:
        return [], list(range(len(true_xy))), list(range(len(peak_xy)))

    d = np.linalg.norm(true_xy[:, None, :] - peak_xy[None, :, :], axis=2)
    big = max_dist_m * 1e3 + 1.0  # effectively forbids a pair beyond the radius
    cost = np.where(d <= max_dist_m, d, big)
    ti, pi = linear_sum_assignment(cost)

    pairs = [(t, p, d[t, p]) for t, p in zip(ti, pi) if d[t, p] <= max_dist_m]
    matched_t = {t for t, _, _ in pairs}
    matched_p = {p for _, p, _ in pairs}
    return (pairs,
            [i for i in range(len(true_xy)) if i not in matched_t],
            [i for i in range(len(peak_xy)) if i not in matched_p])


def cluster_points(xy_arr: np.ndarray, link_m: float) -> dict[int, list[int]]:
    """Single-linkage grouping of ground-truth points at `link_m`."""
    parent = list(range(len(xy_arr)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if len(xy_arr) > 1:
        for a, b in cKDTree(xy_arr).query_pairs(link_m):
            parent[find(a)] = find(b)

    groups = defaultdict(list)
    for i in range(len(xy_arr)):
        groups[find(i)].append(i)
    return dict(groups)


def render(prob, true_xy, peak_xy, groups, src, out_path: Path, pad_m: float) -> None:
    """Whole-tile view plus a zoomed panel per multi-palm cluster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    multi = [g for g in groups.values() if len(g) > 1]
    multi.sort(key=len, reverse=True)
    multi = multi[:5]
    n = 1 + len(multi)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6.4))
    axes = np.atleast_1d(axes)

    def draw(ax, extent_px, title):
        r0, r1, c0, c1 = extent_px
        ax.imshow(np.where(np.isfinite(prob), prob, 0.0)[r0:r1, c0:c1],
                  cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        for arr, style in ((true_xy, dict(marker="+", c="#39FF14", s=150, lw=2.0)),
                           (peak_xy, dict(marker="o", c="#00D4FF", s=70,
                                          facecolors="none", lw=1.6))):
            if len(arr) == 0:
                continue
            rr, cc = rowcol(src.transform, arr[:, 0], arr[:, 1])
            rr, cc = np.atleast_1d(rr), np.atleast_1d(cc)
            keep = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
            if keep.any():
                ax.scatter(cc[keep] - c0, rr[keep] - r0, **style)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    draw(axes[0], (0, prob.shape[0], 0, prob.shape[1]),
         f"whole tile — {len(true_xy)} palms, {len(peak_xy)} peaks")

    res = abs(src.transform.a)
    for ax, g in zip(axes[1:], multi):
        gxy = true_xy[g]
        rr, cc = rowcol(src.transform, gxy[:, 0], gxy[:, 1])
        pad = int(pad_m / res)
        r0, r1 = max(0, min(rr) - pad), min(prob.shape[0], max(rr) + pad)
        c0, c1 = max(0, min(cc) - pad), min(prob.shape[1], max(cc) + pad)
        near = sum(1 for p in peak_xy
                   if r0 <= rowcol(src.transform, p[0], p[1])[0] < r1
                   and c0 <= rowcol(src.transform, p[0], p[1])[1] < c1)
        draw(ax, (r0, r1, c0, c1), f"cluster of {len(g)} — {near} peaks in view")

    fig.suptitle("green + = confirmed palm     blue o = detected peak", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prob-tif", type=Path, required=True, help="dense_classical.py output.")
    p.add_argument("--points", type=Path, nargs="+", required=True,
                   help="Confirmed-palm GeoJSONs; pooled and de-duplicated.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--nms-radius-m", type=float, default=1.5,
                   help="Suppression radius. Set below half the minimum palm spacing you "
                        "intend to resolve, or adjacent palms cannot produce separate peaks.")
    p.add_argument("--match-dist-m", type=float, default=2.0)
    p.add_argument("--cluster-link-m", type=float, default=4.0)
    p.add_argument("--sweep", action="store_true",
                   help="Report the metrics across a grid of thresholds and NMS radii.")
    p.add_argument("--png", type=Path, default=None)
    p.add_argument("--pad-m", type=float, default=15.0, help="Zoom margin around a cluster.")
    return p.parse_args()


def report(prob, true_xy, src, thr, nms_m, match_m, link_m, verbose=True):
    res = abs(src.transform.a)
    peaks_rc = find_peaks(prob, max(1, round(nms_m / res)), thr)
    if len(peaks_rc):
        px, py = xy(src.transform, peaks_rc[:, 0], peaks_rc[:, 1])
        peak_xy = np.c_[np.atleast_1d(px), np.atleast_1d(py)]
    else:
        peak_xy = np.zeros((0, 2))

    pairs, miss_t, miss_p = match_peaks(peak_xy, true_xy, match_m)
    tp, fn, fp = len(pairs), len(miss_t), len(miss_p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    if verbose:
        loc = np.mean([d for _, _, d in pairs]) if pairs else float("nan")
        print(f"\nthreshold={thr}  nms_radius={nms_m} m  match_dist={match_m} m")
        print(f"  ground-truth palms : {len(true_xy)}")
        print(f"  detected peaks     : {len(peak_xy)}")
        print(f"  matched            : {tp}   (recall {rec:.2f}, precision {prec:.3f}, F1 {f1:.3f})")
        print(f"  mean localization error of matched pairs: {loc:.2f} m")
        print(f"  COUNT ERROR        : {len(peak_xy) - len(true_xy):+d} "
              f"({len(peak_xy)} predicted vs {len(true_xy)} actual)")

        groups = cluster_points(true_xy, link_m)
        multi = sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)
        print(f"\n  per-cluster (single-linkage at {link_m} m) — the actual question:")
        if not multi:
            print("    no multi-palm clusters in this tile")
        for g in multi:
            gxy = true_xy[g]
            lo, hi = gxy.min(axis=0) - match_m, gxy.max(axis=0) + match_m
            near = int(((peak_xy[:, 0] >= lo[0]) & (peak_xy[:, 0] <= hi[0]) &
                        (peak_xy[:, 1] >= lo[1]) & (peak_xy[:, 1] <= hi[1])).sum()) if len(peak_xy) else 0
            spread = np.linalg.norm(gxy.max(axis=0) - gxy.min(axis=0))
            verdict = "MERGED" if near < len(g) else ("resolved" if near == len(g) else "over-split")
            print(f"    {len(g)} palms spanning {spread:5.1f} m -> {near} peak(s)   [{verdict}]")
    return peak_xy, (prec, rec, f1)


def main() -> None:
    args = parse_args()
    with rasterio.open(args.prob_tif) as src:
        prob = src.read(1)
        pts = load_points_in_tile(args.points, src)
        true_xy = np.c_[pts.geometry.x, pts.geometry.y] if len(pts) else np.zeros((0, 2))

        print(f"{args.prob_tif.name}: {prob.shape[0]}x{prob.shape[1]} px, "
              f"{np.isfinite(prob).mean()*100:.1f}% valid")
        finite = prob[np.isfinite(prob)]
        if finite.size:
            for q in (50, 90, 99, 99.9):
                print(f"  p{q:<5} = {np.percentile(finite, q):.3f}", end="")
            print(f"   max = {finite.max():.3f}")
            print(f"  fraction of tile above 0.5: {(finite >= 0.5).mean()*100:.1f}%")
        if not len(true_xy):
            raise SystemExit("no confirmed points inside this tile — nothing to compare against")

        if args.sweep:
            print("\nsweep (recall / precision / F1 / peak count):")
            print(f"  {'thr':>5} " + "".join(f"{f'nms={n}m':>26}" for n in (0.5, 1.0, 1.5, 2.5)))
            for thr in (0.3, 0.5, 0.7, 0.9):
                row = f"  {thr:>5.1f} "
                for nms in (0.5, 1.0, 1.5, 2.5):
                    pk, (p_, r_, f_) = report(prob, true_xy, src, thr, nms,
                                              args.match_dist_m, args.cluster_link_m, verbose=False)
                    row += f"{f'{r_:.2f}/{p_:.3f}/{f_:.2f}/{len(pk)}':>26}"
                print(row)

        peak_xy, _ = report(prob, true_xy, src, args.threshold, args.nms_radius_m,
                            args.match_dist_m, args.cluster_link_m)
        if args.png:
            render(prob, true_xy, peak_xy, cluster_points(true_xy, args.cluster_link_m),
                   src, args.png, args.pad_m)
            print(f"\nwrote {args.png}")


if __name__ == "__main__":
    main()
