#!/usr/bin/env python
"""
check_rs.py — inspect the delivered SWISSIMAGE RS sample tiles.

NB: do NOT name this file inspect.py / any stdlib module name. A local file that
shadows a stdlib module (e.g. inspect.py) is put first on sys.path and breaks the
imports of rasterio/attr with "module 'inspect' has no attribute 'signature'".

The RS GeoTIFFs ship with NO embedded CRS (only a .tfw world file). Their
coordinates are valid EPSG:2056 (CH1903+/LV95) — this script assigns that CRS so
everything lines up, and checks each tile against the Lugano palm AOI.
"""
from __future__ import annotations

import glob
import os
import sys

import rasterio
from rasterio.coords import disjoint_bounds

DEFAULT_RS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "swissimage_rs", "lugano_delivery_2026-07",
)
RS_DIR = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RS_DIR
AOI = (2718000, 1115000, 2718500, 1115500)  # Lugano palm AOI, EPSG:2056
CRS = "EPSG:2056"


def main() -> None:
    tifs = sorted(glob.glob(os.path.join(RS_DIR, "*.tif")))
    if not tifs:
        raise SystemExit(f"No .tif found in {RS_DIR!r}")
    print(f"=== SWISSIMAGE RS check :: {len(tifs)} tiles in {RS_DIR} ===\n")

    dates, covering, no_crs = set(), [], 0
    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    for f in tifs:
        with rasterio.open(f) as s:
            b = s.bounds
            crs = s.crs
            if crs is None:
                no_crs += 1
            date = os.path.basename(f)[:8]
            dates.add(date)
            minx, miny = min(minx, b.left), min(miny, b.bottom)
            maxx, maxy = max(maxx, b.right), max(maxy, b.top)
            hits_aoi = not disjoint_bounds(AOI, (b.left, b.bottom, b.right, b.top))
            if hits_aoi:
                covering.append(os.path.basename(f))
            flag = "  <-- covers AOI" if hits_aoi else ""
            print(f"{os.path.basename(f)}: {s.count}-band {s.dtypes[0]} "
                  f"{s.width}x{s.height} crs={crs}{flag}")

    print("\n--- summary ---")
    print(f"tiles with NO embedded CRS: {no_crs}/{len(tifs)}  "
          f"(coords are EPSG:2056 from the .tfw; assign it on read)")
    print(f"acquisition dates present: {sorted(dates)}")
    print(f"overall footprint (LV95): [{minx:.0f}, {miny:.0f}, {maxx:.0f}, {maxy:.0f}]")
    print(f"AOI {AOI} covered by {len(covering)} tile(s):")
    for c in covering:
        print(f"   {c}")

    if no_crs:
        print("\nFIX: the CRS is missing, not wrong. Stamp EPSG:2056 into the georef "
              "metadata (cheap, no pixel rewrite) so QGIS/rasterio place them correctly:")
        print(f'   for f in {RS_DIR}/*.tif; do gdal_edit.py -a_srs EPSG:2056 "$f"; done')
        print("   (or pass crs=EPSG:2056 explicitly when you read them in the pipeline)")


if __name__ == "__main__":
    main()
