#!/usr/bin/env python
"""
04_fetch_temporal.py
====================
Fetch SWISSIMAGE RGB tiles for the AOI across MULTIPLE vintages, into per-year
folders, so you get a co-registered time series for change / persistence features.

  interim/tiles/<aoi>/<year>/<aoi>_<year>_cCCC_rRRR.tif

Because every year is tiled with the identical map-coordinate grid
(swisstopo.iter_tile_bounds), the tile at a given (col,row) covers the SAME ground
across years — so stacking <aoi>_2018_c003_r003 / _2021_ / _2024_ gives a temporal
stack with no extra registration.

WHY (see feasibility notes): an evergreen palm is persistent across vintages while
deciduous canopy and land-use (e.g. construction) change — temporal persistence is
one of the few signals that helps separate palms from look-alike crowns. The Lugano
area has 2018 / 2021 / 2024.

Config:
  imagery.years: [2018, 2021, 2024]   # optional; default = all available for the AOI
(falls back to [imagery.year] semantics if you only set a single year elsewhere).

STATUS: implemented (STAC route). Mirrors 00's fetch but loops vintages.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import swisstopo as st  # noqa: E402
from src.data.config import PROJECT_CRS, ensure_dir, load_config, parse_args  # noqa: E402


def fetch_year(year, hrefs, bbox, tile_size, overlap, res_m, out_dir, max_tiles):
    """Tile the COGs for one vintage into out_dir; returns number written."""
    written = 0
    for col, row, tb in st.iter_tile_bounds(bbox, tile_size, overlap, res_m):
        arr, transform = st.read_window(hrefs, tb, res_m, resampling="nearest")
        if arr is None or int(arr.max()) == 0:
            continue  # no coverage / empty (lake, AOI corner)
        out_path = out_dir / f"{out_dir.parent.name}_{year}_c{col:03d}_r{row:03d}.tif"
        st.write_geotiff(out_path, arr, transform, nodata=None)
        written += 1
        if max_tiles and written >= max_tiles:
            print(f"  [{year}] reached max_tiles={max_tiles}; stopping.")
            break
    return written


def main() -> None:
    args = parse_args("Fetch SWISSIMAGE tiles across multiple vintages for an AOI.")
    cfg = load_config(args.config)

    assert cfg["aoi"]["crs"] == PROJECT_CRS, f"AOI CRS must be {PROJECT_CRS}"
    aoi = cfg["aoi"]["name"]
    bbox = cfg["aoi"]["bbox_lv95"]
    img = cfg["imagery"]
    tile_size, overlap = img["tile_size_px"], img["tile_overlap_px"]
    res_m = img.get("res_m", 0.10)
    max_tiles = img.get("max_tiles")

    feats = st.search_stac_items(st.COLLECTION_RGB, bbox)
    by_year = st.assets_by_year(feats, st.TAG_RGB)
    available = sorted(by_year)
    requested = [str(y) for y in img.get("years", available)]
    years = [y for y in requested if y in by_year]
    missing = [y for y in requested if y not in by_year]

    print(f"=== temporal fetch :: AOI '{aoi}' ===")
    print(f"[stac] available vintages here: {available}")
    if missing:
        print(f"[warn] requested but unavailable (skipped): {missing}")
    if not years:
        raise SystemExit(f"No requested vintage is available. Pick from {available}.")

    for year in years:
        out_dir = ensure_dir(Path(cfg["paths"]["interim_dir"]) / "tiles" / aoi / year)
        n = fetch_year(year, by_year[year], bbox, tile_size, overlap, res_m, out_dir, max_tiles)
        print(f"[ok] {year}: wrote {n} tiles -> {out_dir}")

    print(f"=== done: {len(years)} vintages ({years}). "
          f"Tiles at the same (col,row) align across years. ===")


if __name__ == "__main__":
    main()
