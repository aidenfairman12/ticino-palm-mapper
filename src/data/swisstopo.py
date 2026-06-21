"""
Shared swisstopo STAC helpers for the Phase 0 data pipeline.

Centralises the bits that scripts 00 (RGB), 03 (feature stack / CHM) and 04
(temporal stack) all need: STAC search, tiling math, windowed/co-registered COG
reads, and GeoTIFF writing. Everything is EPSG:2056 (see config.PROJECT_CRS).

Confirmed live (2026-06):
  - STAC root : https://data.geo.admin.ch/api/stac/v1
  - RGB       : ch.swisstopo.swissimage-dop10   asset tag "_0.1_2056" (10 cm)
  - DSM (surface) : ch.swisstopo.swisssurface3d-raster  asset tag "_0.5_2056" (0.5 m)
  - DTM (terrain) : ch.swisstopo.swissalti3d           asset tag "_0.5_2056" (0.5 m)
  Canopy height model (object height) = DSM - DTM.
  NIR (SWISSIMAGE RS) is NOT freely served — request/paid only, no STAC collection.
"""
from __future__ import annotations

from pathlib import Path

from .config import PROJECT_CRS

STAC_API = "https://data.geo.admin.ch/api/stac/v1"

COLLECTION_RGB = "ch.swisstopo.swissimage-dop10"
COLLECTION_DSM = "ch.swisstopo.swisssurface3d-raster"
COLLECTION_DTM = "ch.swisstopo.swissalti3d"

TAG_RGB = "_0.1_2056"
TAG_DSM = "_0.5_2056"
TAG_DTM = "_0.5_2056"


def bbox_lv95_to_wgs84(bbox):
    """[xmin,ymin,xmax,ymax] EPSG:2056 -> WGS84 lon/lat (for STAC bbox queries)."""
    from pyproj import Transformer

    xmin, ymin, xmax, ymax = bbox
    t = Transformer.from_crs(PROJECT_CRS, "EPSG:4326", always_xy=True)
    lon0, lat0 = t.transform(xmin, ymin)
    lon1, lat1 = t.transform(xmax, ymax)
    return [lon0, lat0, lon1, lat1]


def iter_tile_bounds(bbox, tile_size_px, overlap_px, res_m=0.10):
    """Yield (col, row, (txmin,tymin,txmax,tymax)) tiles covering the AOI bbox.

    Tiles are in MAP coordinates (metres, EPSG:2056) so they compose across the
    1 km COGs. Row 0 is the TOP of the AOI (decreasing y). Identical math is used
    for every year, so tiles at a given (col,row) align across vintages.
    """
    xmin, ymin, xmax, ymax = bbox
    tile_m = tile_size_px * res_m
    step_m = (tile_size_px - overlap_px) * res_m
    row, ty_top = 0, ymax
    while ty_top > ymin:
        col, tx_left = 0, xmin
        while tx_left < xmax:
            tx_right = min(tx_left + tile_m, xmax)
            ty_bottom = max(ty_top - tile_m, ymin)
            yield col, row, (tx_left, ty_bottom, tx_right, ty_top)
            col += 1
            tx_left += step_m
        row += 1
        ty_top -= step_m


def search_stac_items(collection, bbox_lv95):
    """All STAC features in `collection` intersecting the AOI, following paging."""
    import requests

    bbox_wgs = bbox_lv95_to_wgs84(bbox_lv95)
    url = f"{STAC_API}/collections/{collection}/items"
    params = {"bbox": ",".join(map(str, bbox_wgs)), "limit": 100}
    feats = []
    while url:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        page = r.json()
        feats.extend(page.get("features", []))
        url = next((l["href"] for l in page.get("links", []) if l.get("rel") == "next"), None)
        params = None
    return feats


def assets_by_year(features, tag):
    """Map 'YYYY' -> [asset hrefs matching `tag`] across the given STAC features."""
    out: dict[str, list[str]] = {}
    for f in features:
        yr = f["properties"]["datetime"][:4]
        for k, a in f["assets"].items():
            if tag in k and k.endswith(".tif"):
                out.setdefault(yr, []).append(a["href"])
    return out


def asset_hrefs(features, tag):
    """All asset hrefs matching `tag` (year-agnostic; for single-vintage layers)."""
    return [a["href"] for f in features for k, a in f["assets"].items()
            if tag in k and k.endswith(".tif")]


def _open_intersecting(hrefs, bounds):
    import rasterio
    from rasterio.coords import disjoint_bounds

    srcs = [rasterio.open(h) for h in hrefs]
    keep = [s for s in srcs if not disjoint_bounds(bounds, s.bounds)]
    drop = [s for s in srcs if s not in keep]
    for s in drop:
        s.close()
    return keep


def read_window(hrefs, bounds, res, resampling="nearest"):
    """Mosaic the COGs in `hrefs` over `bounds` at `res` m/px. Returns (arr, transform).

    Memory-safe: rasterio.merge does windowed reads. `resampling` matters when the
    source GSD differs from `res` (e.g. resampling 0.5 m LiDAR onto a 0.1 m grid).
    """
    from rasterio.enums import Resampling
    from rasterio.merge import merge

    srcs = _open_intersecting(hrefs, bounds)
    if not srcs:
        return None, None
    try:
        arr, transform = merge(
            srcs, bounds=bounds, res=res, resampling=Resampling[resampling]
        )
    finally:
        for s in srcs:
            s.close()
    return arr, transform


def read_aligned_to(ref_tif, hrefs, resampling="bilinear"):
    """Read/mosaic `hrefs` onto the exact grid (bounds, res, shape) of `ref_tif`.

    Used to co-register a coarser layer (DSM/DTM at 0.5 m) onto an existing RGB
    tile's 10 cm grid so channels stack pixel-for-pixel. Returns a (bands,H,W)
    array cropped/padded to the reference shape, or None if no coverage.
    """
    import numpy as np
    import rasterio

    with rasterio.open(ref_tif) as ref:
        bounds = ref.bounds
        res = ref.res[0]
        H, W = ref.height, ref.width
    arr, _ = read_window(hrefs, bounds, res, resampling=resampling)
    if arr is None:
        return None
    # merge can be off by a pixel at edges; force the reference shape
    out = np.zeros((arr.shape[0], H, W), dtype=arr.dtype)
    h, w = min(H, arr.shape[1]), min(W, arr.shape[2])
    out[:, :h, :w] = arr[:, :h, :w]
    return out


def write_geotiff(path, arr, transform, nodata=None, dtype=None):
    """Write a (bands,H,W) array as a deflate-compressed EPSG:2056 GeoTIFF."""
    import rasterio

    dtype = dtype or arr.dtype
    profile = dict(
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=dtype,
        transform=transform,
        crs=PROJECT_CRS,
        compress="deflate",
        nodata=nodata,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype))
