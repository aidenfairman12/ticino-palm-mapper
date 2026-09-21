"""Export a KML of the three NATIVE-10cm-resolution acquisition dates'
footprints (2021-03-24, 2024-07-20, 2024-08-10 — see
checkpoint-2026-09-18-labeling-ceiling.md's 2026-09-21 update for how these
were identified: the other three dates in the delivery are native 25cm,
upsampled to look like 10cm by the existing pipeline, and not directly
comparable), plus every confirmed palm point, so the actual candidate AOI can
be scouted in Google Earth for palm presence before committing to it.

Each date's strips get their own color/layer so overlap (or gaps) between the
three dates is visible at a glance — the model can only use ground that's
covered by ALL THREE for change-based features, so what matters here is less
each date's individual footprint and more their intersection.

Confirmed palms are pooled from both label sources (active_learning +
lugano_MASTER, which itself merges the batch2-4 + example sets), deduplicated
on exact coordinates, same as scripts/22_quantify_march_coverage.py.

Usage (run where geopandas/rasterio/pyproj are available, e.g. the HPC login
node — the repo's data/raw and data/interim/labels are both there):
    python3 scripts/24_export_10cm_footprint_kml.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
RS_DIR = REPO_ROOT / "data" / "raw" / "swissimage_rs" / "lugano_delivery_2026-07"
LABELS_DIR = REPO_ROOT / "data" / "interim" / "labels"
OUT_DIR = REPO_ROOT / "data" / "processed" / "_exploration"

# date -> (label, KML color in BBGGRR, no alpha)
TEN_CM_DATES = {
    "20210324": ("2021-03-24 (leaf-off)", "ff0000"),   # blue
    "20240720": ("2024-07-20", "00ff00"),               # green
    "20240810": ("2024-08-10", "00a5ff"),               # orange
}
CONFIRMED_FILES = ["lugano_MASTER_confirmed_palms.geojson", "active_learning_confirmed_palms.geojson"]


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


def load_confirmed_points() -> list[tuple[str, float, float]]:
    pts, seen = [], set()
    for fname in CONFIRMED_FILES:
        data = json.loads((LABELS_DIR / fname).read_text())
        for feat in data["features"]:
            x, y = feat["geometry"]["coordinates"][:2]
            key = (round(x, 1), round(y, 1))
            if key in seen:
                continue
            seen.add(key)
            pid = feat["properties"].get("id") or feat["properties"].get("tile") or "?"
            pts.append((f"{fname}:{pid}", x, y))
    return pts


def main() -> None:
    import rasterio
    from pyproj import Transformer
    from src.data.config import PROJECT_CRS

    to_wgs = Transformer.from_crs(PROJECT_CRS, "EPSG:4326", always_xy=True)

    strip_placemarks = []
    for date, (label, color) in TEN_CM_DATES.items():
        strips = sorted(RS_DIR.glob(f"{date}_*.tif"))
        if not strips:
            print(f"[warn] no raw strips found for {date}")
            continue
        for t in strips:
            with rasterio.open(t) as src:
                res = round(src.transform.a, 3)
                if res != 0.1:
                    print(f"[warn] {t.name} is native {res}m, not 0.1m — skipping, check TEN_CM_DATES")
                    continue
                ring = _bounds_to_wgs_ring(src.bounds, to_wgs)
            strip_placemarks.append(_polygon_kml(f"{label} :: {t.stem}", color, ring, fill_alpha="40"))
        print(f"{date}: {len(strips)} strips outlined")

    points = load_confirmed_points()
    point_placemarks = []
    for pid, x, y in points:
        lon, lat = to_wgs.transform(x, y)
        point_placemarks.append(_point_kml("confirmed palm", "0000ff", lon, lat, desc=pid))  # red
    print(f"{len(points)} confirmed palms plotted")

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>Native-10cm acquisition footprints + confirmed palms</name>
{''.join(strip_placemarks)}
{''.join(point_placemarks)}
</Document>
</kml>"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ten_cm_footprints_and_palms.kml"
    out_path.write_text(kml)
    print(f"=== wrote -> {out_path} ===")


if __name__ == "__main__":
    main()
