#!/usr/bin/env python
"""
08_export_kml.py
=================
Export a KML overlay for visually scanning the AOI in Google Earth, instead of
clicking through Street View candidates one at a time.

Three layers:
  - Tile boundaries, color-coded: GREEN = already has a confirmed "distinct"
    palm, YELLOW = has one or more un-reviewed raw GBIF candidate points
    inside it, GRAY = neither.
  - Confirmed points (lugano_MASTER_confirmed_palms.geojson), colored by
    ndvi_signal_current: green=distinct, yellow=weak, red=none.
  - Un-reviewed raw GBIF occurrence points (blue) — candidates worth checking.

Open the output .kml in Google Earth (desktop or web) or Google Maps; pan
around, cross-reference the tile grid against the real satellite imagery, and
note new palm locations directly — no Street View driving required, same
"satellite landmark matching" method already used for batches 2-4.

STATUS: implemented. Read-only — writes no labels itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import PROJECT_CRS, ensure_dir, load_config, parse_args  # noqa: E402


def _polygon_kml(name: str, color_abgr: str, coords_lonlat: list[tuple[float, float]]) -> str:
    coord_str = " ".join(f"{lon},{lat},0" for lon, lat in coords_lonlat)
    return f"""
    <Placemark>
      <name>{name}</name>
      <Style><LineStyle><color>ff{color_abgr}</color><width>2</width></LineStyle>
      <PolyStyle><color>4d{color_abgr}</color></PolyStyle></Style>
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


def main() -> None:
    args = parse_args("Export a KML overlay of tile boundaries + confirmed/candidate points.")
    cfg = load_config(args.config)

    import geopandas as gpd
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import box

    aoi = cfg["aoi"]["name"]
    to_wgs = Transformer.from_crs(PROJECT_CRS, "EPSG:4326", always_xy=True)

    tile_dir = Path(cfg["paths"]["processed_dir"]) / aoi / "feature_stack"
    tile_paths = sorted(tile_dir.glob("*_rgbchm.tif"))
    if not tile_paths:
        raise SystemExit(f"No tiles found in {tile_dir} — run 03_build_feature_stack.py first")

    master_path = Path(cfg["paths"]["interim_dir"]) / "labels" / "lugano_MASTER_confirmed_palms.geojson"
    occ_path = Path(cfg["paths"]["interim_dir"]) / "labels" / f"{aoi}_occurrences.geojson"

    confirmed = gpd.read_file(master_path).to_crs(PROJECT_CRS) if master_path.exists() else None
    occurrences = gpd.read_file(occ_path).to_crs(PROJECT_CRS) if occ_path.exists() else None

    distinct_pts = confirmed[confirmed["ndvi_signal_current"] == "distinct"] if confirmed is not None else None

    # --- tile boundary polygons, color-coded ---
    tile_placemarks = []
    n_green, n_yellow, n_gray = 0, 0, 0
    for tile_path in tile_paths:
        with rasterio.open(tile_path) as src:
            bounds = src.bounds
        tile_box = box(*bounds)

        has_confirmed = distinct_pts is not None and distinct_pts.geometry.within(tile_box).any()
        has_candidate = occurrences is not None and occurrences.geometry.within(tile_box).any()

        if has_confirmed:
            color, n_green = "00ff00", n_green + 1  # ABGR: opaque green
        elif has_candidate:
            color, n_yellow = "00ffff", n_yellow + 1  # ABGR: opaque yellow
        else:
            color, n_gray = "888888", n_gray + 1  # ABGR: gray

        corners = [(bounds.left, bounds.bottom), (bounds.right, bounds.bottom),
                   (bounds.right, bounds.top), (bounds.left, bounds.top),
                   (bounds.left, bounds.bottom)]
        corners_wgs = [to_wgs.transform(x, y) for x, y in corners]
        tile_placemarks.append(_polygon_kml(tile_path.stem, color, corners_wgs))

    # --- confirmed points, colored by ndvi_signal_current ---
    point_placemarks = []
    # KML colors are AABBGGRR; these are the BBGGRR part (no alpha):
    # 00ff00=green, 00ffff=yellow, 0000ff=red.
    signal_colors = {"distinct": "00ff00", "weak": "00ffff", "none": "0000ff"}
    if confirmed is not None:
        for _, row in confirmed.iterrows():
            lon, lat = to_wgs.transform(row.geometry.x, row.geometry.y)
            signal = row.get("ndvi_signal_current", "none")
            color = signal_colors.get(signal, "ff0000")
            point_placemarks.append(_point_kml(f"confirmed ({signal})", color, lon, lat, desc=str(row.get("source", ""))))

    # --- un-reviewed raw candidates: blue ---
    if occurrences is not None and confirmed is not None:
        reviewed_union = confirmed.geometry.buffer(5).union_all()
        unreviewed = occurrences[~occurrences.geometry.within(reviewed_union)]
    else:
        unreviewed = occurrences

    if unreviewed is not None:
        for _, row in unreviewed.iterrows():
            lon, lat = to_wgs.transform(row.geometry.x, row.geometry.y)
            point_placemarks.append(_point_kml("candidate (unreviewed)", "ff8000", lon, lat,
                                                 desc=f"gbifID={row.get('gbifID', '')}"))

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>{aoi} — tile grid + palm candidates</name>
{''.join(tile_placemarks)}
{''.join(point_placemarks)}
</Document>
</kml>"""

    out_dir = ensure_dir(Path(cfg["paths"]["processed_dir"]) / aoi / "labeling")
    out_path = out_dir / f"{aoi}_overview.kml"
    out_path.write_text(kml)

    print(f"=== tiles: {n_green} confirmed-positive (green), {n_yellow} have candidates (yellow), "
          f"{n_gray} empty (gray) ===")
    print(f"=== wrote {len(point_placemarks)} points -> {out_path} ===")
    print(f"Open in Google Earth: file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
