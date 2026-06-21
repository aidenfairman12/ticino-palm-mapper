#!/usr/bin/env python
"""
00_fetch_swissimage.py
======================
Fetch SWISSIMAGE 10 cm orthophoto coverage for the configured AOI and cut it into
model-ready tiles (single vintage). For a multi-year time series use
04_fetch_temporal.py; both share src/data/swisstopo.py.

Two routes (set `imagery.source` in the config):

  - "stac" : swisstopo publishes SWISSIMAGE as Cloud-Optimized GeoTIFFs (COGs)
             discoverable via its STAC API. Free, no account. We query the
             collection for items intersecting the AOI, then read the COGs with
             windowed/mosaicked reads (rasterio.merge with bounds) — memory-safe.
             IMPLEMENTED.

  - "gee"  : Google Earth Engine export. Needs an EE account; still a stub since
             the STAC route needs no account and is the default.

Confirmed live (2026-06): STAC v1, collection ch.swisstopo.swissimage-dop10, each
item a 1 km² tile with a "_0.1_2056.tif" 10 cm COG already in EPSG:2056. Coverage
is flown on a ~3-year cycle, so not every year exists for every AOI — the script
lists available years and errors (rather than writing zero tiles) if yours is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

# allow running as `python scripts/00_fetch_swissimage.py` without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import swisstopo as st  # noqa: E402
from src.data.config import PROJECT_CRS, ensure_dir, load_config, parse_args  # noqa: E402


def fetch_stac(cfg: dict, out_dir: Path) -> None:
    """Query swisstopo STAC for the AOI and tile the 10 cm COGs for one vintage."""
    bbox = cfg["aoi"]["bbox_lv95"]
    year = str(cfg["imagery"]["year"])
    img = cfg["imagery"]
    tile_size, overlap = img["tile_size_px"], img["tile_overlap_px"]
    res_m = img.get("res_m", 0.10)
    max_tiles = img.get("max_tiles")

    print(f"[stac] AOI bbox (EPSG:2056): {bbox}")
    print(f"[stac] collection={st.COLLECTION_RGB} year={year} res={res_m} m/px")

    feats = st.search_stac_items(st.COLLECTION_RGB, bbox)
    by_year = st.assets_by_year(feats, st.TAG_RGB)
    available = sorted(by_year)
    print(f"[stac] {len(feats)} items intersect the AOI | years available: {available}")
    if year not in by_year:
        raise SystemExit(
            f"[stac] no SWISSIMAGE 10 cm coverage for year {year} in this AOI.\n"
            f"        available years here: {available or '(none — check the bbox)'}\n"
            f"        set imagery.year to one of the above."
        )
    hrefs = by_year[year]
    print(f"[stac] {len(hrefs)} COG(s) for year {year}")

    n_planned = sum(1 for _ in st.iter_tile_bounds(bbox, tile_size, overlap, res_m))
    print(f"[stac] AOI tiles to cut: {n_planned} of {tile_size}px (overlap {overlap}px)")
    if max_tiles and n_planned > max_tiles:
        print(f"[stac] max_tiles={max_tiles} set -> stopping after {max_tiles} "
              f"(shrink aoi.bbox_lv95 for a full run).")

    written = 0
    for col, row, tb in st.iter_tile_bounds(bbox, tile_size, overlap, res_m):
        arr, transform = st.read_window(hrefs, tb, res_m, resampling="nearest")
        if arr is None or int(arr.max()) == 0:
            continue  # AOI corner with no imagery, or empty/lake tile
        out_path = out_dir / f"{cfg['aoi']['name']}_{year}_c{col:03d}_r{row:03d}.tif"
        st.write_geotiff(out_path, arr, transform, nodata=None)
        written += 1
        if written == 1 or written % 10 == 0:
            print(f"[stac]   wrote {written} tiles... (latest {out_path.name})")
        if max_tiles and written >= max_tiles:
            print(f"[stac] reached max_tiles={max_tiles}; stopping.")
            break
    print(f"[stac] done: {written} tiles written")


def fetch_gee(cfg: dict, out_dir: Path) -> None:
    """Export SWISSIMAGE 10 cm from Google Earth Engine for the AOI."""
    print("[gee] route selected.")
    # ---- TODO: Earth Engine export (optional; STAC route needs no account) ------
    # import ee; ee.Initialize()
    # img = (ee.ImageCollection("Switzerland/SWISSIMAGE/orthos/10cm")
    #          .filterDate(f"{cfg['imagery']['year']}-01-01", f"{cfg['imagery']['year']}-12-31")
    #          .mosaic())
    # region = ee.Geometry.Rectangle(cfg["aoi"]["bbox_lv95"], proj="EPSG:2056", geodesic=False)
    # ee.batch.Export.image.toDrive(image=img, region=region, scale=0.10,
    #                               crs="EPSG:2056", maxPixels=1e13).start()
    # ----------------------------------------------------------------------------
    print("[gee] TODO: wire up ee export if ever needed; STAC is the default.")


def main() -> None:
    args = parse_args("Fetch SWISSIMAGE tiles for an AOI (single vintage).")
    cfg = load_config(args.config)

    assert cfg["aoi"]["crs"] == PROJECT_CRS, (
        f"AOI CRS must be {PROJECT_CRS}; got {cfg['aoi']['crs']}"
    )

    out_dir = ensure_dir(Path(cfg["paths"]["interim_dir"]) / "tiles" / cfg["aoi"]["name"])
    source = cfg["imagery"]["source"]

    print(f"=== SWISSIMAGE fetch :: AOI '{cfg['aoi']['name']}' :: source={source} ===")
    if source == "stac":
        fetch_stac(cfg, out_dir)
    elif source == "gee":
        fetch_gee(cfg, out_dir)
    else:
        raise ValueError(f"Unknown imagery.source: {source!r} (use 'stac' or 'gee')")

    print(f"=== output dir: {out_dir} ===")


if __name__ == "__main__":
    main()
