"""Import a Google Earth visual-review pass over the native-10cm footprint
KML (scripts/24's output). The user adds placemarks in Earth under 'palms'
and 'negatives' folders (and may delete some of the original root-level
confirmed-palm placemarks they decide are wrong on closer look), then
re-exports the whole project as KML. This is re-run every time the user does
another pass and re-exports.

Root-level placemarks (no folder) are always the ORIGINAL confirmed-palm
points scripts/24 wrote in — this only works because the user doesn't move
anything out of the root level into a folder; if they had, this diffing
approach would misattribute it.

Handles repeat passes over the SAME project (the KML keeps growing as the
user adds more): loads every already-imported batch's points
(google_earth_review_batch*_confirmed_palms.geojson /
..._hard_negatives.geojson) and skips anything within ~1m of a point already
imported, so re-running after another editing pass only writes the NEW
points into a new, separately-numbered batch file — it never re-imports or
duplicates a prior batch.

Usage: python3 scripts/25_import_earth_review_batch.py --kml <path> --batch 2
"""
from __future__ import annotations

import argparse
import glob
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from pyproj import Transformer

NS = {"kml": "http://www.opengis.net/kml/2.2"}
REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS_DIR = REPO_ROOT / "data" / "interim" / "labels"
CONFIRMED_FILES = ["lugano_MASTER_confirmed_palms.geojson", "active_learning_confirmed_palms.geojson"]
DUP_TOLERANCE_DEG = 1e-5  # ~1m


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


def make_geojson(points: list[dict], to_lv95, label: str, batch: int) -> dict:
    features = []
    for i, p in enumerate(points):
        x, y = to_lv95.transform(p["lon"], p["lat"])
        note = p["note"].replace("<div>", "").replace("</div>", "").strip()
        features.append({
            "type": "Feature",
            "properties": {
                "id": f"gearth_batch{batch}_{label}_{i+1}",
                "x": round(x, 2), "y": round(y, 2),
                "lon": p["lon"], "lat": p["lat"],
                "note": note,
                "source": f"google_earth_visual_review_batch{batch}",
            },
            "geometry": {"type": "Point", "coordinates": [round(x, 2), round(y, 2)]},
        })
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::2056"}},
        "features": features,
    }


def load_previously_imported_lonlat(pattern: str) -> list[tuple[float, float]]:
    pts = []
    for fname in glob.glob(str(LABELS_DIR / pattern)):
        data = json.loads(Path(fname).read_text())
        for feat in data["features"]:
            props = feat["properties"]
            if "lon" in props and "lat" in props:
                pts.append((props["lon"], props["lat"]))
    return pts


def dedup_against(points: list[dict], already: list[tuple[float, float]]) -> list[dict]:
    if not already:
        return points
    out = []
    for p in points:
        d = min(((p["lon"] - lo) ** 2 + (p["lat"] - la) ** 2) ** 0.5 for lo, la in already)
        if d > DUP_TOLERANCE_DEG:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kml", type=Path, required=True)
    ap.add_argument("--batch", type=int, required=True,
                     help="Batch number for this pass, e.g. 2 for the second editing session. "
                          "Must not reuse a number already written to disk.")
    args = ap.parse_args()

    tree = ET.parse(args.kml)
    all_points: list[dict] = []
    walk_points(tree.getroot(), [], all_points)

    root_points = [p for p in all_points if p["folder"] is None]
    palms_raw = [p for p in all_points if p["folder"] == "palms"]
    negatives_raw = [p for p in all_points if p["folder"] == "negatives"]
    print(f"root-level (original) points: {len(root_points)}")
    print(f"palms in KML: {len(palms_raw)}, negatives in KML: {len(negatives_raw)}")

    already_palms = load_previously_imported_lonlat("google_earth_review_batch*_confirmed_palms.geojson")
    already_negs = load_previously_imported_lonlat("google_earth_review_batch*_hard_negatives.geojson")
    palms = dedup_against(palms_raw, already_palms)
    negatives = dedup_against(negatives_raw, already_negs)
    print(f"already imported in prior batches: {len(already_palms)} palms, {len(already_negs)} negatives")
    print(f"NEW this batch: {len(palms)} palms, {len(negatives)} negatives")

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

    missing = [o for o in originals if closest_dist(o["lon"], o["lat"]) > DUP_TOLERANCE_DEG]
    print(f"\nremoved from the ORIGINAL confirmed set (present before scripts/24, absent from edited "
          f"root-level points): {len(missing)}")
    for m in missing:
        print(f"  {m['file']} :: {m['pid']}  (x={m['x']}, y={m['y']})")

    # also check whether any PREVIOUSLY-IMPORTED batch point (in the 'palms'/'negatives'
    # folders, not root) is missing from this new export -- e.g. the user deleted one
    # they'd added in an earlier pass.
    def missing_from_kml(prior_lonlat, kml_folder_points, kind):
        kml_set = kml_folder_points
        gone = []
        for lo, la in prior_lonlat:
            d = min(((lo - p["lon"]) ** 2 + (la - p["lat"]) ** 2) ** 0.5 for p in kml_set) if kml_set else 1e9
            if d > DUP_TOLERANCE_DEG:
                gone.append((lo, la))
        if gone:
            print(f"\n[warn] {len(gone)} previously-imported {kind} no longer appear in this KML "
                  f"(deleted in Earth since the last import?) -- not auto-removed, check by hand: {gone}")
    missing_from_kml(already_palms, palms_raw, "palms")
    missing_from_kml(already_negs, negatives_raw, "negatives")

    if not palms and not negatives:
        print("\nnothing new to write.")
        return

    palms_out = LABELS_DIR / f"google_earth_review_batch{args.batch}_confirmed_palms.geojson"
    neg_out = LABELS_DIR / f"google_earth_review_batch{args.batch}_hard_negatives.geojson"
    if palms_out.exists() or neg_out.exists():
        raise SystemExit(f"batch {args.batch} output already exists -- pick an unused --batch number")
    palms_out.write_text(json.dumps(make_geojson(palms, to_lv95, "palm", args.batch), indent=2))
    neg_out.write_text(json.dumps(make_geojson(negatives, to_lv95, "negative", args.batch), indent=2))
    print(f"\nwrote {len(palms)} NEW palms -> {palms_out}")
    print(f"wrote {len(negatives)} NEW negatives -> {neg_out}")
    print("\nNOTE: the 'removed' points above were NOT auto-deleted from their source files "
          "-- remove them by hand after confirming the list matches what was actually deleted in Earth.")


if __name__ == "__main__":
    main()
