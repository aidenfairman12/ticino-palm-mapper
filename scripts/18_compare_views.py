#!/usr/bin/env python
"""
18_compare_views.py
===================
Test whether a different VIEW of the same data makes palms adjudicable by eye,
and whether the March flight separates evergreen from deciduous numerically.

Why this exists: the binding constraint on this project is not the model, it is
what can be labeled. Dense scoring established that the classical model learned
"tall vegetation" rather than "palm" because its negatives contain almost no
other trees (see dense-scoring-reveals-vegetation-detector). The fix is real
vegetation negatives — but they can only be collected if a human can actually
tell a palm from a non-palm in the imagery, which currently only works for
roadside palms with street view to confirm. Every downstream choice (more
labeling, a U-Net, density regression) is gated on lifting that ceiling, so it
is worth an hour before any of them.

Two modes.

`panels` renders each point as a row of views, so you can see at a glance which
one separates palms from other canopy:
    RGB               natural colour, what you have been looking at
    false-colour IR   NIR/R/G — vegetation reads by structure, not greenness
    NDVI March        early spring: evergreen bright, deciduous bare
    NDVI August       full leaf-on, everything bright
    NDVI drop         August minus March — high means deciduous
    CHM hillshade     shaded relief of the height model; a fan crown's radial
                      micro-relief is invisible under a flat colour ramp

--blind interleaves confirmed palms with canopy controls in shuffled order and
writes the answer key to a separate file. This matters: the question is not
"do palms look distinctive once I know where they are" (they always do) but
"can I identify one without being told". Grade yourself before trusting any
labeling campaign built on that judgement.

`ndvi` skips rendering and measures the March separation directly — per-group
NDVI distributions and ROC AUC for each candidate statistic. If March is
already past leaf-out in the Ticino lowlands the evergreen filter is worthless,
and ten minutes here beats discovering that after a labeling push.

Controls are sampled from CANOPY (CHM above --canopy-min-m), not from anywhere
in the tile: the discrimination that matters is palm vs other tree. Sampling
uniformly is what produced the blind spot in the first place.

Torch-free, like the rest of the classical pipeline.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RES_M = 0.1
NIR, RED, GREEN, BLUE, NDVI, CHM = 0, 1, 2, 3, 4, 5
DATE_RE = re.compile(r"(\d{8})")


def date_of(tile_path: Path) -> str:
    """Acquisition date from the containing directory (feature_stack_rs_20210324)."""
    m = DATE_RE.search(tile_path.parent.name)
    if m is None:
        raise ValueError(f"no YYYYMMDD in directory name: {tile_path.parent.name}")
    return m.group(1)


def glob_tiles(tile_dirs: list[Path]) -> list[Path]:
    out = []
    for d in tile_dirs:
        out.extend(sorted(d.glob("*_nirchm.tif")))
    return out


class TileSet:
    """Tiles indexed by date, with bounds cached and arrays loaded lazily.

    Bounds are read once up front (cheap, ~3 ms/tile) so point-in-tile lookups
    do not reopen files; full arrays are ~25 MB each and are cached behind a
    cap, the same reasoning as score_candidates_classical's LRU.
    """

    def __init__(self, tile_paths: list[Path], max_cached: int = 24):
        self.by_date: dict[str, list[tuple[Path, object]]] = {}
        for p in tile_paths:
            with rasterio.open(p) as src:
                self.by_date.setdefault(date_of(p), []).append((p, box(*src.bounds)))
        self._cache: dict[Path, tuple[np.ndarray, object]] = {}
        self._order: list[Path] = []
        self.max_cached = max_cached

    @property
    def dates(self) -> list[str]:
        return sorted(self.by_date)

    def load(self, path: Path):
        if path not in self._cache:
            with rasterio.open(path) as src:
                self._cache[path] = (src.read(), src.transform)
            self._order.append(path)
            if len(self._order) > self.max_cached:
                del self._cache[self._order.pop(0)]
        return self._cache[path]

    def crop(self, point: Point, date: str, half_px: int):
        """(C, 2*half_px, 2*half_px) window centred on `point`, or None.

        None when no tile of that date covers the point, when the window would
        run off the tile edge, or when the centre is nodata — a partial or
        empty crop is worse than a gap in a comparison figure, since it invites
        a judgement about imagery that is not there.
        """
        for path, bounds in self.by_date.get(date, []):
            if not point.within(bounds):
                continue
            arr, transform = self.load(path)
            r, c = rowcol(transform, point.x, point.y)
            r0, c0 = r - half_px, c - half_px
            if r0 < 0 or c0 < 0 or r0 + 2 * half_px > arr.shape[1] or c0 + 2 * half_px > arr.shape[2]:
                continue
            win = arr[:, r0:r0 + 2 * half_px, c0:c0 + 2 * half_px]
            if (win[:, half_px, half_px] == 0).all():
                continue
            return win
        return None


def stretch(band: np.ndarray, lo_hi=(2, 98)) -> np.ndarray:
    """Percentile contrast stretch to [0,1] — matches scripts/10's _rgb_crop."""
    lo, hi = np.percentile(band, lo_hi)
    return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)


def build_views(crop_a: np.ndarray, crop_march: np.ndarray | None,
                exag: float = 1.0, smooth_px: float = 2.0,
                seasonal: bool = True) -> list[tuple[str, np.ndarray, dict]]:
    """The view stack for one location. `crop_a` is leaf-on, `crop_march` early
    spring (may be None if that date does not cover the point).

    `seasonal` is decided once by the caller for the WHOLE figure, never per
    row: the subplot grid is rectangular, so a row that returned fewer views
    than the first row left blank columns, and one that returned more raised
    IndexError. With the bellinzona flights only 8 of 96 palms have March data,
    so ragged rows were the normal case rather than an edge case. A location
    missing March now renders explicit NaN panels, keeping the grid aligned and
    making the gap visible instead of silently shifting columns."""
    views: list[tuple[str, np.ndarray, dict]] = []

    rgb = np.dstack([stretch(crop_a[b]) for b in (RED, GREEN, BLUE)])
    views.append(("RGB", rgb, {}))

    # NIR into red: healthy vegetation glows, and crown structure separates from
    # background greenness in a way natural colour flattens out.
    cir = np.dstack([stretch(crop_a[b]) for b in (NIR, RED, GREEN)])
    views.append(("false-colour IR", cir, {}))

    blank = np.full_like(crop_a[NDVI], np.nan, dtype=float)
    if seasonal:
        march_ndvi = crop_march[NDVI] if crop_march is not None else blank
        views.append(("NDVI March", march_ndvi, dict(cmap="RdYlGn", vmin=-0.2, vmax=0.9)))
    views.append(("NDVI leaf-on", crop_a[NDVI], dict(cmap="RdYlGn", vmin=-0.2, vmax=0.9)))
    if seasonal:
        drop = (crop_a[NDVI] - crop_march[NDVI]) if crop_march is not None else blank
        views.append(("NDVI drop (leaf-on - Mar)", drop, dict(cmap="coolwarm", vmin=-0.4, vmax=0.4)))

    from matplotlib.colors import LightSource
    from scipy import ndimage
    chm = np.nan_to_num(crop_a[CHM].astype(float))
    # A 10 cm CHM over canopy is per-pixel noisy, and raw hillshade of it is
    # speckle that hides the metre-scale crown relief we are looking for. A
    # light blur suppresses the noise without touching crown-scale structure.
    if smooth_px > 0:
        chm = ndimage.gaussian_filter(chm, sigma=smooth_px)
    shade = LightSource(azdeg=315, altdeg=35).hillshade(chm, vert_exag=exag, dx=RES_M, dy=RES_M)
    views.append(("CHM hillshade", shade, dict(cmap="gray")))
    return views


def sample_canopy_controls(tiles: TileSet, date: str, palms: gpd.GeoSeries,
                           n: int, min_dist_m: float, canopy_min_m: float,
                           seed: int, height_range: tuple[float, float] | None = None) -> list[Point]:
    """Points on canopy (CHM >= canopy_min_m) at least min_dist_m from any palm.

    Deliberately NOT uniform over the tile: the model already separates palm
    from road and roof, and the discrimination it fails at — and that a human
    must supply — is palm vs other tree.
    """
    rng = np.random.default_rng(seed)
    entries = tiles.by_date.get(date, [])
    if not entries:
        raise SystemExit(f"no tiles for date {date}")
    out: list[Point] = []
    for _ in range(n * 400):
        if len(out) >= n:
            break
        path, bounds = entries[rng.integers(len(entries))]
        lo_x, lo_y, hi_x, hi_y = bounds.bounds
        pt = Point(rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y))
        if len(palms) and palms.distance(pt).min() < min_dist_m:
            continue
        arr, transform = tiles.load(path)
        r, c = rowcol(transform, pt.x, pt.y)
        if not (0 <= r < arr.shape[1] and 0 <= c < arr.shape[2]):
            continue
        if (arr[:, r, c] == 0).all() or arr[CHM, r, c] < canopy_min_m:
            continue
        # Height matching. Without it a flat --canopy-min-m floor manufactures
        # a CHM separation whenever the palms sit below it: measured on
        # bellinzona the confirmed palms are ~1.4 m against 14 m canopy
        # controls, an AUC of 0.98 that says nothing except that the sampler
        # was told to pick tall things. Negatives drawn that way would teach
        # "short = palm", the mirror of the vegetation leak being fixed.
        if height_range is not None and not (height_range[0] <= arr[CHM, r, c] <= height_range[1]):
            continue
        out.append(pt)
    if len(out) < n:
        print(f"[warn] only found {len(out)}/{n} canopy controls — "
              f"try lowering --canopy-min-m (currently {canopy_min_m} m)")
    return out


def run_panels(tiles: TileSet, palms: list[Point], controls: list[Point],
               march: str, leafon: str, half_px: int, blind: bool,
               out: Path, seed: int, exag: float = 1.0, smooth_px: float = 2.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(p, "PALM") for p in palms] + [(p, "control") for p in controls]
    if blind:
        np.random.default_rng(seed).shuffle(rows)

    seasonal = march != leafon
    built, dropped, no_march = [], {"PALM": 0, "control": 0}, 0
    for pt, label in rows:
        ca = tiles.crop(pt, leafon, half_px)
        if ca is None:
            dropped[label] += 1
            continue
        cm = tiles.crop(pt, march, half_px) if seasonal else None
        if seasonal and cm is None:
            no_march += 1
        built.append((build_views(ca, cm, exag, smooth_px, seasonal), label, pt))
    if not built:
        raise SystemExit("no location had a usable crop — check --tile-dirs and --crop-m")

    # Silence here would be dangerous: a figure containing only controls looks
    # perfectly normal, and in blind mode you cannot tell by reading it.
    n_pal = sum(1 for _, l, _ in built if l == "PALM")
    print(f"rendered {n_pal} palms + {len(built)-n_pal} controls "
          f"(dropped {dropped['PALM']} palms, {dropped['control']} controls — "
          f"not covered by the leaf-on date, or too near a tile edge for the crop)")
    if n_pal == 0:
        raise SystemExit("no PALM had a usable crop — the points may be outside these tile dirs")
    if dropped["PALM"]:
        print("       raise --n-palms to compensate, or widen --tile-dirs")
    if no_march:
        print(f"       {no_march}/{len(built)} location(s) have no {march} data — their seasonal "
              f"panels are blank (white). Drop the early date from --tile-dirs to remove those columns.")

    ncol = len(built[0][0])
    fig, axes = plt.subplots(len(built), ncol, figsize=(2.5 * ncol, 2.6 * len(built)), squeeze=False)
    for i, (views, label, pt) in enumerate(built):
        for j, (name, img, kw) in enumerate(views):
            ax = axes[i][j]
            ax.imshow(img, interpolation="nearest", **kw)
            # A marker over the centre would give away the answer in blind mode
            # and bias the judgement in labelled mode; the target is the centre
            # pixel by construction, so ticks are enough.
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=10)
        axes[i][0].set_ylabel(f"#{i}" if blind else f"#{i} {label}",
                              fontsize=9, rotation=0, ha="right", va="center")

    fig.suptitle("target is the CENTRE of each crop" + ("  —  BLIND: identify the palms, key in the .txt"
                 if blind else ""), fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if len(built) < 4 else 0.99))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({len(built)} locations x {ncol} views)")

    if blind:
        key = out.with_suffix(".key.txt")
        key.write_text("\n".join(
            f"#{i}\t{label}\t{pt.x:.2f}\t{pt.y:.2f}" for i, (_, label, pt) in enumerate(built)) + "\n")
        print(f"wrote {key}  — do not open until you have scored yourself")


def run_ndvi(tiles: TileSet, palms: list[Point], controls: list[Point],
             march: str, leafon: str, out: Path | None) -> None:
    from sklearn.metrics import roc_auc_score

    def gather(pts):
        rows = []
        for pt in pts:
            ca = tiles.crop(pt, leafon, 1)
            cm = tiles.crop(pt, march, 1)
            if ca is None or cm is None:
                continue
            rows.append((float(cm[NDVI, 1, 1]), float(ca[NDVI, 1, 1]), float(ca[CHM, 1, 1])))
        return np.array(rows).reshape(-1, 3)

    P, C = gather(palms), gather(controls)
    if len(P) < 5 or len(C) < 5:
        raise SystemExit(f"too few points covered by BOTH dates (palms {len(P)}, controls {len(C)})")

    print(f"\ncovered by both {march} and {leafon}: {len(P)}/{len(palms)} palms, "
          f"{len(C)}/{len(controls)} canopy controls "
          f"({len(C)/max(len(controls),1)*100:.0f}% of sampled canopy is in the {march} footprint)")
    if len(P) < 20:
        print(f"  [WARN] only {len(P)} palms — AUC standard error is roughly "
              f"{(0.25/len(P))**0.5:.2f}, so treat these as indicative only.")
        print("         Pass every points file to --points to pull more palms into the overlap.")
    stats = {
        "NDVI March": (P[:, 0], C[:, 0], "higher = evergreen"),
        "NDVI August": (P[:, 1], C[:, 1], ""),
        "NDVI drop (Aug-Mar)": (P[:, 1] - P[:, 0], C[:, 1] - C[:, 0], "lower = evergreen"),
        "CHM (m)": (P[:, 2], C[:, 2], ""),
    }
    print(f"\n{'statistic':<22}{'palm median':>13}{'control median':>16}{'ROC AUC':>10}   note")
    for name, (a, b, note) in stats.items():
        y = np.r_[np.ones(len(a)), np.zeros(len(b))]
        auc = roc_auc_score(y, np.r_[a, b])
        print(f"{name:<22}{np.median(a):>13.3f}{np.median(b):>16.3f}{max(auc, 1-auc):>10.3f}   {note}")

    print("\n  AUC 0.5 = no separation, 1.0 = perfect. If 'NDVI March' and 'NDVI drop'")
    print("  are both near 0.5, March is already leafed out here and the evergreen")
    print("  filter is not worth building.")

    if out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, (name, (a, b, _)) in zip(axes, list(stats.items())[:3]):
            bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 30)
            ax.hist(b, bins=bins, alpha=0.6, label=f"canopy control (n={len(b)})", color="#888")
            ax.hist(a, bins=bins, alpha=0.7, label=f"confirmed palm (n={len(a)})", color="#2E8B57")
            ax.set_title(name); ax.legend(fontsize=8)
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=125, bbox_inches="tight")
        plt.close(fig)
        print(f"\nwrote {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["panels", "ndvi"])
    p.add_argument("--tile-dirs", type=Path, nargs="+", required=True,
                   help="One per date; the date is read from each directory name.")
    p.add_argument("--points", type=Path, nargs="+", required=True, help="Confirmed-palm GeoJSONs.")
    p.add_argument("--march-date", default=None, help="Default: earliest date found.")
    p.add_argument("--leafon-date", default=None, help="Default: latest date found.")
    p.add_argument("--n-palms", type=int, default=12)
    p.add_argument("--n-controls", type=int, default=12)
    p.add_argument("--crop-m", type=float, default=12.0, help="Crop side length in metres.")
    p.add_argument("--canopy-min-m", type=float, default=1.0,
                   help="Minimum CHM for a control — keeps controls on woody vegetation, not lawn.")
    p.add_argument("--match-height", action="store_true",
                   help="Restrict controls to the palms' own CHM 5-95 percentile range. Without "
                        "this, any separation on a height-derived statistic may just reflect the "
                        "--canopy-min-m floor rather than anything about palms.")
    p.add_argument("--min-dist-m", type=float, default=25.0)
    p.add_argument("--blind", action="store_true")
    p.add_argument("--hillshade-exag", type=float, default=1.0,
                   help="Vertical exaggeration. Above ~2 the slopes saturate at 10 cm resolution.")
    p.add_argument("--hillshade-smooth-px", type=float, default=2.0,
                   help="Gaussian blur on CHM before hillshading; 0 disables. Suppresses per-pixel "
                        "canopy noise that otherwise hides crown-scale relief.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("view_comparison.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tiles = TileSet(glob_tiles(args.tile_dirs))
    if not tiles.dates:
        raise SystemExit("no *_nirchm.tif found under --tile-dirs")
    march = args.march_date or tiles.dates[0]
    leafon = args.leafon_date or tiles.dates[-1]
    print(f"dates found: {tiles.dates}   using march={march}, leaf-on={leafon}")
    if march == leafon:
        print("[warn] march and leaf-on are the same date — the seasonal views will be empty")

    frames = [gpd.read_file(p).to_crs("EPSG:2056") for p in args.points]
    palms_all = gpd.GeoSeries(
        [g for f in frames for g in f.geometry], crs="EPSG:2056"
    ).drop_duplicates().reset_index(drop=True)
    print(f"{len(palms_all)} unique confirmed palms loaded")

    rng = np.random.default_rng(args.seed)
    pick = rng.permutation(len(palms_all))[:args.n_palms if args.mode == "panels" else len(palms_all)]
    palms = [palms_all.iloc[i] for i in pick]

    height_range = None
    if args.match_height:
        # The range must come from the SAME palms the comparison will run on.
        # In ndvi mode that is the both-dates intersection, which can differ
        # sharply from the leaf-on set: on bellinzona the March-covered palms
        # are 1.4 m median against 4.8 m for the leaf-on set, so matching to
        # the latter puts controls 4 m taller than the palms they are meant to
        # control for, and CHM keeps a separation the matching was added to
        # remove.
        heights = []
        for pt in palms_all:
            c = tiles.crop(pt, leafon, 1)
            if c is None:
                continue
            if args.mode == "ndvi" and tiles.crop(pt, march, 1) is None:
                continue
            heights.append(float(c[CHM, 1, 1]))
        if len(heights) < 5:
            raise SystemExit("--match-height needs at least 5 palms covered by the leaf-on date")
        height_range = (float(np.percentile(heights, 5)), float(np.percentile(heights, 95)))
        print(f"palm CHM 5-95 percentile over the {len(heights)} palms in the comparison set: "
              f"{height_range[0]:.2f}-{height_range[1]:.2f} m (median {np.median(heights):.2f} m) "
              f"— controls restricted to this range")

    controls = sample_canopy_controls(
        tiles, leafon, palms_all,
        args.n_controls if args.mode == "panels" else max(args.n_controls, 200),
        args.min_dist_m, args.canopy_min_m, args.seed, height_range)
    print(f"using {len(palms)} palms and {len(controls)} canopy controls")

    if args.mode == "panels":
        run_panels(tiles, palms, controls, march, leafon,
                   max(4, round(args.crop_m / RES_M / 2)), args.blind, args.out, args.seed,
                   args.hillshade_exag, args.hillshade_smooth_px)
    else:
        run_ndvi(tiles, palms, controls, march, leafon, args.out)


if __name__ == "__main__":
    main()
