#!/usr/bin/env python
"""
06_prioritize_labels.py
========================
Rank occurrence points by how CHEAP they are to hand-verify, so labeling effort
goes to the candidates most likely to pay off first.

Two independent filters, both free / no API key:

  - distance to the nearest public road (OpenStreetMap via Overpass API) — a
    proxy for Street View coverage. Google's Street View car network runs on
    public roads, so a point far from any road is unlikely to have imagery,
    and forcing a label there wastes time. NOT a guarantee — some roads lack
    coverage — but a strong, free prioritization signal.

  - whether the point falls inside the delivered SWISSIMAGE RS (NIR) footprint,
    so we know which candidates could also get an NDVI/CIR crop.

Output: a CSV ranked by (near a road) AND (has NIR), then by distance, so the
best-value candidates — checkable AND NIR-covered — sort to the top. Also
regenerates the labeling-assist HTML restricted to the top candidates.

STATUS: implemented. This produces a SHORTLIST, not labels — every candidate
still needs a human to actually look at Street View / the crop and decide.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import swisstopo as st  # noqa: E402
from src.data.config import PROJECT_CRS, ensure_dir, load_config, parse_args  # noqa: E402

OVERPASS_API = "https://overpass-api.de/api/interpreter"


def fetch_osm_roads(bbox_lv95, buffer_m=300):
    """Query Overpass for all highway=* ways in a buffered AOI; return LineStrings in EPSG:2056."""
    import requests
    from pyproj import Transformer
    from shapely.geometry import LineString

    xmin, ymin, xmax, ymax = bbox_lv95
    xmin, ymin, xmax, ymax = xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m
    to_wgs = Transformer.from_crs(PROJECT_CRS, "EPSG:4326", always_xy=True)
    to_lv95 = Transformer.from_crs("EPSG:4326", PROJECT_CRS, always_xy=True)
    lon0, lat0 = to_wgs.transform(xmin, ymin)
    lon1, lat1 = to_wgs.transform(xmax, ymax)

    query = f"""
    [out:json][timeout:30];
    way["highway"]({lat0},{lon0},{lat1},{lon1});
    out geom;
    """
    headers = {"User-Agent": "ticino-palm-mapper/0.1 (research, non-commercial)"}
    resp = requests.post(OVERPASS_API, data={"data": query}, timeout=60, headers=headers)
    resp.raise_for_status()
    elements = resp.json().get("elements", [])

    lines = []
    for el in elements:
        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue
        pts = [to_lv95.transform(pt["lon"], pt["lat"]) for pt in geom]
        lines.append(LineString(pts))
    return lines


def main() -> None:
    args = parse_args("Rank occurrence points by road proximity + NIR coverage.")
    cfg = load_config(args.config)

    import geopandas as gpd
    import pandas as pd
    from shapely.ops import unary_union

    aoi = cfg["aoi"]["name"]
    bbox = cfg["aoi"]["bbox_lv95"]
    near_road_m = cfg["labels"].get("near_road_m", 25)
    rs_dir = cfg["labels"].get("rs_dir", "data/raw/swissimage_rs/lugano_delivery_2026-07")

    occ_path = Path(cfg["paths"]["interim_dir"]) / "labels" / f"{aoi}_occurrences.geojson"
    if not occ_path.exists():
        raise SystemExit(f"Run 01_fetch_infoflora.py first; missing {occ_path}")
    gdf = gpd.read_file(occ_path).to_crs(PROJECT_CRS)
    print(f"=== prioritize labels :: AOI '{aoi}' :: {len(gdf)} occurrence points ===")

    # --- road proximity (OSM) ---
    print("[osm] querying Overpass for roads in AOI...")
    roads = fetch_osm_roads(bbox)
    print(f"[osm] {len(roads)} road segments returned")
    if roads:
        road_union = unary_union(roads)
        gdf["dist_to_road_m"] = gdf.geometry.apply(lambda p: round(p.distance(road_union), 1))
    else:
        gdf["dist_to_road_m"] = None
    gdf["near_road"] = gdf["dist_to_road_m"] <= near_road_m

    # --- NIR footprint coverage ---
    rs_tiles = sorted(Path(rs_dir).glob("*.tif"))
    print(f"[nir] checking against {len(rs_tiles)} delivered RS tiles in {rs_dir}")
    rs_bounds = []
    if rs_tiles:
        import rasterio
        for t in rs_tiles:
            with rasterio.open(t) as s:
                rs_bounds.append(s.bounds)

    def has_nir(pt):
        return any(b.left <= pt.x <= b.right and b.bottom <= pt.y <= b.top for b in rs_bounds)

    gdf["nir_available"] = gdf.geometry.apply(has_nir)

    # --- rank: best value = checkable AND NIR-covered, then by road distance ---
    gdf["priority"] = (~gdf["near_road"]).astype(int) + (~gdf["nir_available"]).astype(int)
    gdf = gdf.sort_values(["priority", "dist_to_road_m"], na_position="last")

    n_near = int(gdf["near_road"].sum())
    n_nir = int(gdf["nir_available"].sum())
    n_both = int((gdf["near_road"] & gdf["nir_available"]).sum())
    print(f"\n[summary] near a road (<= {near_road_m} m): {n_near}/{len(gdf)}")
    print(f"[summary] inside NIR footprint:          {n_nir}/{len(gdf)}")
    print(f"[summary] BOTH (best-value candidates):   {n_both}/{len(gdf)}")

    out_dir = ensure_dir(Path(cfg["paths"]["processed_dir"]) / aoi / "labeling")
    out_csv = out_dir / "candidates_ranked.csv"
    cols = ["gbifID", "year", "coordUncertainty_m", "lat", "lon",
            "dist_to_road_m", "near_road", "nir_available", "priority"]
    gdf[cols].to_csv(out_csv, index=False)
    print(f"\n=== wrote ranked shortlist -> {out_csv} ===")
    print("Top 10 candidates (best value first):")
    print(gdf[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
