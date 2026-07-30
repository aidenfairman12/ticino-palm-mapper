#!/usr/bin/env python
"""
09_export_nir_extent_kml.py
============================
Export a KML of the FULL delivered NIR/RS footprint (all strips, not just the
current tiny AOI), so a much larger area can be visually scanned for palms
before picking a new, more representative AOI.

The current AOI (configs/aoi_example.yaml) is a 500x500m sanity-check box,
picked only to make the very first pipeline run easy to eyeball — it's far
too small to be a representative sample. The delivered SWISSIMAGE RS (NIR)
data covers a much larger area (multiple strips) that was never clipped down
to that box; nothing outside the AOI has been fetched/processed into tiles
yet, but the raw NIR strips are already sitting on disk and worth scouting.

Workflow this supports: open the output KML in Google Earth, pan the WHOLE
NIR extent, mark palm locations found via satellite imagery (same method as
batches 2-4). Once a strong candidate region emerges, set a new, larger
aoi.bbox_lv95 in the config and rerun the Phase 0 pipeline (00, 01, 02, 03,
07) for it.

Layers in the output:
  - Individual NIR/RS strip footprints (blue outline) — the ACTUAL scoutable
    area; strips don't necessarily tile contiguously, so gaps between them
    are not covered even though they fall inside the overall bounding box.
  - The current AOI box (red outline) — what's already processed into tiles.
  - The 39 existing confirmed points, colored by ndvi_signal_current.

STATUS: implemented. Read-only — writes no labels itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import PROJECT_CRS, ensure_dir, load_config, parse_args  # noqa: E402


def _polygon_kml(name: str, color_abgr: str, coords_lonlat: list[tuple[float, float]],
                  fill_alpha: str = "33") -> str:
    coord_str = " ".join(f"{lon},{lat},0" for lon, lat in coords_lonlat)
    return f"""
    <Placemark>
      <name>{name}</name>
      <Style><LineStyle><color>ff{color_abgr}</color><width>2</width></LineStyle>
      <PolyStyle><color>{fill_alpha}{color_abgr}</color></PolyStyle></Style>
      <Polygon><outerBoundaryIs><LinearRing><coordinates>
        {coord_str}
      </coordinates></LinearRing></outerBoundaryIs></Polygon>
    </Placemark>"""


def _point_kml(name: str, color_abgr: str, lon: float, lat: float, desc: str = "") -> str:
    return f"""
    <Placemark>
      <name>{name}</name>
      <description>{desc}</description>
      <Style><IconStyle><color>ff{color_abgr}</color>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
      </IconStyle></Style>
      <Point><coordinates>{lon},{lat},0</coordinates></Point>
    </Placemark>"""


def _bounds_to_wgs_ring(bounds, to_wgs) -> list[tuple[float, float]]:
    corners = [(bounds.left, bounds.bottom), (bounds.right, bounds.bottom),
               (bounds.right, bounds.top), (bounds.left, bounds.top),
               (bounds.left, bounds.bottom)]
    return [to_wgs.transform(x, y) for x, y in corners]


def main() -> None:
    args = parse_args("Export a KML of the full NIR/RS delivery footprint for AOI scouting.")
    cfg = load_config(args.config)

    import geopandas as gpd
    import rasterio
    from pyproj import Transformer

    aoi = cfg["aoi"]["name"]
    to_wgs = Transformer.from_crs(PROJECT_CRS, "EPSG:4326", always_xy=True)

    rs_dir = Path(cfg["labels"].get("rs_dir", "data/raw/swissimage_rs/lugano_delivery_2026-07"))
    rs_tiles = sorted(rs_dir.glob("*.tif"))
    if not rs_tiles:
        raise SystemExit(f"No NIR/RS tiles found in {rs_dir}")

    strip_placemarks = []
    for t in rs_tiles:
        with rasterio.open(t) as src:
            ring = _bounds_to_wgs_ring(src.bounds, to_wgs)
        strip_placemarks.append(_polygon_kml(t.stem, "ff0000", ring, fill_alpha="26"))  # blue, faint fill

    # current (tiny) AOI box, for reference
    xmin, ymin, xmax, ymax = cfg["aoi"]["bbox_lv95"]
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
    aoi_ring = [to_wgs.transform(x, y) for x, y in corners]
    aoi_placemark = _polygon_kml("CURRENT AOI (already processed)", "0000ff", aoi_ring, fill_alpha="55")

    # existing confirmed points, for reference
    master_path = Path(cfg["paths"]["interim_dir"]) / "labels" / "lugano_MASTER_confirmed_palms.geojson"
    point_placemarks = []
    # KML colors are AABBGGRR; these are the BBGGRR part (no alpha):
    # 00ff00=green, 00ffff=yellow, 0000ff=red.
    signal_colors = {"distinct": "00ff00", "weak": "00ffff", "none": "0000ff"}
    if master_path.exists():
        confirmed = gpd.read_file(master_path).to_crs(PROJECT_CRS)
        for _, row in confirmed.iterrows():
            lon, lat = to_wgs.transform(row.geometry.x, row.geometry.y)
            signal = row.get("ndvi_signal_current", "none")
            color = signal_colors.get(signal, "ff0000")
            point_placemarks.append(_point_kml(f"confirmed ({signal})", color, lon, lat,
                                                 desc=str(row.get("source", ""))))

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>{aoi} — full NIR delivery footprint for AOI scouting</name>
{''.join(strip_placemarks)}
{aoi_placemark}
{''.join(point_placemarks)}
</Document>
</kml>"""

    out_dir = ensure_dir(Path("data/processed/_exploration"))
    out_path = out_dir / f"{aoi}_nir_footprint_scouting.kml"
    out_path.write_text(kml)

    print(f"=== {len(rs_tiles)} NIR strips outlined (blue) ===")
    print(f"=== current AOI outlined (red) for reference ===")
    print(f"=== {len(point_placemarks)} existing confirmed points plotted ===")
    print(f"=== wrote -> {out_path} ===")


if __name__ == "__main__":
    main()
