#!/usr/bin/env python
"""
02_align_labels.py
==================
Align Info Flora presence points to the imagery tiles produced by
`00_fetch_swissimage.py`, producing model-ready labels.

For each image tile this writes:
  - a per-tile GeoJSON of the points falling inside it (in pixel + map coords), and
  - optionally a rasterized label mask (disk of `labels.crown_radius_m` around each
    point) for segmentation-style training.

This is the bridge from "geodata" to "ML dataset". Detection-style training can use
the point/box GeoJSON directly; segmentation-style can use the raster mask.

Reminder (proposal §3/§5): Info Flora points are WEAK POSITIVES. This script places
labels where palms were *recorded*; it does not invent labels where none exist and
does not guarantee completeness. For honest evaluation you still need a small,
EXHAUSTIVELY hand-labeled held-out test set.

STATUS: skeleton. The geometry/affine math is real; it assumes tiles from script 00
exist as GeoTIFFs with valid georeferencing. Until script 00 is wired up, run this
against any georeferenced test raster to sanity-check the alignment logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import PROJECT_CRS, ensure_dir, load_config, parse_args  # noqa: E402


def points_to_pixel(gdf, transform):
    """
    Convert point geometries (in raster CRS) to (col, row) pixel coordinates
    using the raster's affine transform. Returns a list of (col, row, geometry).
    """
    from rasterio.transform import rowcol

    out = []
    for geom in gdf.geometry:
        r, c = rowcol(transform, geom.x, geom.y)
        out.append((c, r, geom))
    return out


def rasterize_disks(points_px, shape, radius_px):
    """Burn filled disks of radius `radius_px` around each point into a uint8 mask."""
    import numpy as np
    from skimage.draw import disk

    mask = np.zeros(shape, dtype="uint8")
    h, w = shape
    for col, row, _ in points_px:
        rr, cc = disk((row, col), radius_px, shape=shape)
        mask[rr, cc] = 1
    return mask


def main() -> None:
    args = parse_args("Align occurrence points to image tiles.")
    cfg = load_config(args.config)

    import geopandas as gpd
    import rasterio

    tiles_dir = Path(cfg["paths"]["interim_dir"]) / "tiles" / cfg["aoi"]["name"]
    occ_path = (
        Path(cfg["paths"]["interim_dir"]) / "labels" / f"{cfg['aoi']['name']}_occurrences.geojson"
    )
    out_dir = ensure_dir(Path(cfg["paths"]["processed_dir"]) / cfg["aoi"]["name"])

    if not occ_path.exists():
        raise FileNotFoundError(f"Run 01_fetch_infoflora.py first; missing {occ_path}")
    gdf = gpd.read_file(occ_path).to_crs(PROJECT_CRS)
    print(f"[load] {len(gdf)} occurrence points")

    tiles = sorted(tiles_dir.glob("*.tif"))
    if not tiles:
        print(f"[warn] no tiles in {tiles_dir} — wire up 00_fetch_swissimage.py first.")
        print("[info] alignment logic is ready; this is a no-op until tiles exist.")
        return

    res_m = 0.10
    crown_radius_px = int(round(cfg["labels"]["crown_radius_m"] / res_m))
    n_labeled_tiles = 0

    for tif in tiles:
        with rasterio.open(tif) as src:
            # clip points to this tile's bounds (fast) then to pixel coords
            left, bottom, right, top = src.bounds
            sub = gdf.cx[left:right, bottom:top]
            if len(sub) == 0:
                continue
            pts_px = points_to_pixel(sub, src.transform)

            # 1) detection-style: save points (pixel + map coords) as GeoJSON
            sub_out = sub.copy()
            sub_out["px_col"] = [c for c, _, _ in pts_px]
            sub_out["px_row"] = [r for _, r, _ in pts_px]
            gj = out_dir / f"{tif.stem}_points.geojson"
            sub_out.to_file(gj, driver="GeoJSON")

            # 2) segmentation-style: rasterize disks into a mask alongside the tile
            mask = rasterize_disks(pts_px, (src.height, src.width), crown_radius_px)
            import numpy as np
            np.save(out_dir / f"{tif.stem}_mask.npy", mask)

            n_labeled_tiles += 1
            print(f"[ok] {tif.name}: {len(sub)} points -> {gj.name} (+mask)")

    print(f"=== {n_labeled_tiles} tiles labeled -> {out_dir} ===")
    print("Next: hand-label a seed set + an exhaustive held-out test set "
          "(proposal §5), then Phase 1 baseline.")


if __name__ == "__main__":
    main()
