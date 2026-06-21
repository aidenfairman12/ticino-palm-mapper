#!/usr/bin/env python
"""
01_fetch_infoflora.py
=====================
Load *Trachycarpus fortunei* occurrence points, clip to the AOI, reproject to
EPSG:2056, and save a clean GeoJSON of presence points.

Two routes (set `labels.source` in the config):

  - "gbif" : fetch occurrence points directly from the GBIF API (free, no account).
             GBIF mirrors Info Flora's Swiss records and exposes per-record
             coordinates (lat/lon, WGS84) plus coordinate-uncertainty in metres.
             This is the default — it unblocks the pipeline without a manual export.
             IMPLEMENTED.

  - "file" : load a manually-exported GeoJSON/CSV at `labels.occurrence_file`
             (e.g. the official Info Flora point export, once it arrives). Cleaning
             + reprojection logic below handles both formats.

IMPORTANT (see proposal §3): these are PRESENCE-ONLY records — good positive
anchors, NOT complete labels and NOT crown outlines. Absence of a point does not
mean absence of a palm. For honest evaluation you still need a small, EXHAUSTIVELY
hand-labelled held-out test set.

NOTE on the earlier "AtlasWS_*.csv" export: that was a Welten-Sutter species
*checklist* for a mapping square (one row per species, NO coordinates) — it cannot
place point labels. The export you want is one row per *observation* with X/Y. The
GBIF route below produces exactly that.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import PROJECT_CRS, ensure_dir, load_config, parse_args  # noqa: E402

GBIF_API = "https://api.gbif.org/v1"


def _aoi_polygon_wgs84(bbox_lv95):
    """AOI bbox (EPSG:2056) -> a WGS84 WKT polygon for the GBIF `geometry` filter.

    GBIF wants lon/lat, counter-clockwise, with the ring closed.
    """
    from pyproj import Transformer

    xmin, ymin, xmax, ymax = bbox_lv95
    t = Transformer.from_crs(PROJECT_CRS, "EPSG:4326", always_xy=True)
    # corners CCW starting from SW
    corners_lv95 = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
    lonlat = [t.transform(x, y) for x, y in corners_lv95]
    ring = ", ".join(f"{lon:.8f} {lat:.8f}" for lon, lat in lonlat)
    return f"POLYGON(({ring}))"


def fetch_gbif(cfg: dict):
    """Page through GBIF occurrences for the species within the AOI; return a GeoDataFrame."""
    import geopandas as gpd
    import pandas as pd
    import requests

    species = cfg["labels"]["species"]
    taxon_key = cfg["labels"].get("gbif_taxon_key")
    country = cfg["labels"].get("gbif_country", "CH")

    if taxon_key is None:  # resolve from name if not pinned in config
        match = requests.get(
            f"{GBIF_API}/species/match", params={"name": species}, timeout=30
        ).json()
        taxon_key = match.get("usageKey")
        print(f"[gbif] resolved '{species}' -> taxonKey={taxon_key}")
        if taxon_key is None:
            raise SystemExit(f"[gbif] could not resolve a taxonKey for {species!r}")

    geometry = _aoi_polygon_wgs84(cfg["aoi"]["bbox_lv95"])
    params = {
        "taxonKey": taxon_key,
        "country": country,
        "hasCoordinate": "true",
        "hasGeospatialIssue": "false",
        "geometry": geometry,
        "limit": 300,
    }
    print(f"[gbif] taxonKey={taxon_key} country={country} (AOI-clipped query)")

    records, offset = [], 0
    while True:
        params["offset"] = offset
        page = requests.get(f"{GBIF_API}/occurrence/search", params=params, timeout=60).json()
        for o in page.get("results", []):
            lon, lat = o.get("decimalLongitude"), o.get("decimalLatitude")
            if lon is None or lat is None:
                continue
            records.append(
                {
                    "gbifID": o.get("key"),
                    "species": o.get("species") or species,
                    "year": o.get("year"),
                    "coordUncertainty_m": o.get("coordinateUncertaintyInMeters"),
                    "basisOfRecord": o.get("basisOfRecord"),
                    "lon": lon,
                    "lat": lat,
                }
            )
        if page.get("endOfRecords", True):
            break
        offset += params["limit"]

    df = pd.DataFrame.from_records(records)
    print(f"[gbif] fetched {len(df)} georeferenced records")
    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    )


def load_occurrences_file(path: Path):
    """Load occurrences from a user-provided GeoJSON or CSV export."""
    import geopandas as gpd
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(
            f"Occurrence file not found: {path}\n"
            "Export Trachycarpus fortunei OBSERVATION points (one row per record, "
            "with coordinates) from Info Flora, or set labels.source: gbif to fetch "
            "directly. NB a Welten-Sutter 'species list' export has no coordinates "
            "and will not work."
        )

    if path.suffix.lower() in {".geojson", ".json", ".gpkg", ".shp"}:
        gdf = gpd.read_file(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        lon_col = next((c for c in df.columns if c.lower() in {"lon", "longitude", "x"}), None)
        lat_col = next((c for c in df.columns if c.lower() in {"lat", "latitude", "y"}), None)
        if lon_col is None or lat_col is None:
            raise ValueError(f"Could not find lon/lat columns in {path} (cols={list(df.columns)})")
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326",  # CSV lon/lat assumed WGS84; reprojected below
        )
    else:
        raise ValueError(f"Unsupported occurrence file type: {path.suffix}")
    return gdf


def main() -> None:
    args = parse_args("Fetch + reproject occurrence points for an AOI.")
    cfg = load_config(args.config)

    import geopandas as gpd
    from shapely.geometry import box

    source = cfg["labels"].get("source", "gbif")
    print(f"=== Occurrences :: {cfg['labels']['species']} :: source={source} ===")

    if source == "gbif":
        gdf = fetch_gbif(cfg)
    elif source in {"file", "infoflora"}:
        gdf = load_occurrences_file(Path(cfg["labels"]["occurrence_file"]))
    else:
        raise ValueError(f"Unknown labels.source: {source!r} (use 'gbif' or 'file')")
    print(f"[load] {len(gdf)} raw records, crs={gdf.crs}")

    # Reproject everything to the project CRS.
    gdf = gdf.to_crs(PROJECT_CRS)

    # Optional species filter (file exports may contain multiple taxa; GBIF is
    # already filtered by taxonKey but this is harmless).
    name_col = next(
        (c for c in gdf.columns if c.lower() in {"species", "taxon", "scientificname", "name"}),
        None,
    )
    if name_col is not None:
        before = len(gdf)
        gdf = gdf[gdf[name_col].astype(str).str.contains("Trachycarpus", case=False, na=False)]
        print(f"[filter] species filter on '{name_col}': {before} -> {len(gdf)}")

    # Clip to AOI bbox.
    xmin, ymin, xmax, ymax = cfg["aoi"]["bbox_lv95"]
    aoi = gpd.GeoSeries([box(xmin, ymin, xmax, ymax)], crs=PROJECT_CRS)
    gdf = gpd.clip(gdf, aoi)
    print(f"[clip] within AOI bbox: {len(gdf)} points")

    out_dir = ensure_dir(Path(cfg["paths"]["interim_dir"]) / "labels")
    out_path = out_dir / f"{cfg['aoi']['name']}_occurrences.geojson"
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"=== wrote {len(gdf)} points -> {out_path} ===")

    if len(gdf) == 0:
        print("[warn] zero points in AOI — check the bbox and the source, "
              "or pick an AOI known to contain palms (e.g. around Lugano).")


if __name__ == "__main__":
    main()
