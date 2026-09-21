"""One-off import of the user's first Google Earth visual-review pass over
the native-10cm footprint KML (scripts/24's output). The user added two
folders of new placemarks in Earth ('palms', 'negatives') and deleted two
existing confirmed-palm placemarks they decided were wrong on closer look,
then re-exported the whole project as KML.

Splits that single edited KML into:
  - google_earth_review_20260921_confirmed_palms.geojson   (new 'palms' folder)
  - google_earth_review_20260921_vegetation_negatives.geojson (new 'negatives' folder)
  - a report of which existing confirmed points are missing (present in the
    original 135-point export but absent from the root-level points in the
    edited KML), for manual removal from whichever source file they came from

Root-level placemarks (no folder) are the ORIGINAL confirmed-palm points
scripts/24 wrote in; anything inside 'palms'/'negatives' folders is new. This
only works because the user didn't move anything out of the root level into
a folder — if they had, this diffing approach would misattribute it.

Usage: python3 scripts/25_import_earth_review_batch.py --kml <path>
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from pyproj import Transformer

NS = {"kml": "http://www.opengis.net/kml/2.2"}
REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS_DIR = REPO_ROOT / "data" / "interim" / "labels"
CONFIRMED_FILES = ["lugano_MASTER_confirmed_palms.geojson", "active_learning_confirmed_palms.geojson"]


def walk_points(el, path, out):
    for child in el:
        tag = child.tag.split("}")[-1]
        if tag == "Folder":
            name = child.find("kml:name", NS)
            nm = name.text if name is not None else "?"
            walk_points(child, path + [nm], out)
        elif tag == "Placemark":
            geom = child.find(".//kml:Point", NS)
            if geom is None:
                continue
            name_el = child.find("kml:name", NS)
            desc_el = child.find("kml:description", NS)
            lon, lat, *_ = (float(v) for v in geom.find("kml:coordinates", NS).text.strip().split(","))
            out.append({
                "folder": path[-1] if path else None,
                "name": (name_el.text or "").strip() if name_el is not None else "",
                "note": (desc_el.text or "").strip() if desc_el is not None else "",
                "lon": lon, "lat": lat,
            })
        else:
            walk_points(child, path, out)


def make_geojson(points: list[dict], to_lv95, label: str) -> dict:
    features = []
    for i, p in enumerate(points):
        x, y = to_lv95.transform(p["lon"], p["lat"])
        note = p["note"].replace("<div>", "").replace("</div>", "").strip()
        features.append({
            "type": "Feature",
            "properties": {
                "id": f"gearth_20260921_{label}_{i+1}",
                "x": round(x, 2), "y": round(y, 2),
                "lon": p["lon"], "lat": p["lat"],
                "note": note,
                "source": "google_earth_visual_review_20260921",
            },
            "geometry": {"type": "Point", "coordinates": [round(x, 2), round(y, 2)]},
        })
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::2056"}},
        "features": features,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kml", type=Path, required=True)
    args = ap.parse_args()

    tree = ET.parse(args.kml)
    all_points: list[dict] = []
    walk_points(tree.getroot(), [], all_points)

    root_points = [p for p in all_points if p["folder"] is None]
    palms = [p for p in all_points if p["folder"] == "palms"]
    negatives = [p for p in all_points if p["folder"] == "negatives"]
    print(f"root-level (original) points: {len(root_points)}")
    print(f"new palms: {len(palms)}, new negatives: {len(negatives)}")

    to_wgs = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)

    originals = []
    seen = set()
    for fname in CONFIRMED_FILES:
        data = json.loads((LABELS_DIR / fname).read_text())
        for feat in data["features"]:
            x, y = feat["geometry"]["coordinates"][:2]
            key = (round(x, 1), round(y, 1))
            if key in seen:
                continue
            seen.add(key)
            pid = feat["properties"].get("id") or feat["properties"].get("tile") or "?"
            lon, lat = to_wgs.transform(x, y)
            originals.append({"file": fname, "pid": pid, "x": x, "y": y, "lon": lon, "lat": lat})

    def closest_dist(lon, lat):
        return min(((lon - p["lon"]) ** 2 + (lat - p["lat"]) ** 2) ** 0.5 for p in root_points) if root_points else 1e9

    missing = [o for o in originals if closest_dist(o["lon"], o["lat"]) > 1e-5]
    print(f"\nremoved (present originally, absent from edited root-level points): {len(missing)}")
    for m in missing:
        print(f"  {m['file']} :: {m['pid']}  (x={m['x']}, y={m['y']})")

    palms_out = LABELS_DIR / "google_earth_review_20260921_confirmed_palms.geojson"
    neg_out = LABELS_DIR / "google_earth_review_20260921_vegetation_negatives.geojson"
    palms_out.write_text(json.dumps(make_geojson(palms, to_lv95, "palm"), indent=2))
    neg_out.write_text(json.dumps(make_geojson(negatives, to_lv95, "negative"), indent=2))
    print(f"\nwrote {len(palms)} palms -> {palms_out}")
    print(f"wrote {len(negatives)} negatives -> {neg_out}")
    print("\nNOTE: the 'removed' points above were NOT auto-deleted from their source files "
          "-- remove them by hand (or re-run with a --remove flag if you add one) after confirming "
          "this list matches what you actually deleted in Earth.")


if __name__ == "__main__":
    main()
