#!/usr/bin/env python
"""
05_labeling_assist.py
=====================
Build a self-contained HTML review sheet to VERIFY occurrence points fast, instead
of eyeballing the orthophoto cold.

For each occurrence point (from 01_fetch_infoflora.py) it renders one card:
  - a native-resolution ortho crop centred on the point (red +),
  - GBIF metadata (year, coordinate-uncertainty, id),
  - a Google Street View link (oblique eye-level confirmation),
  - a Google satellite link and a swisstopo map.geo.admin.ch link (both at the point).

Open the resulting index.html and click through: Street View confirms species, the
crop shows the from-above signature. This is the workflow we used to confirm a palm
at Parco giochi Usignolo — it's how you build an HONEST, ground-truthed test set
(occurrence-point coordinates alone are unreliable; some land on rooftops).

Cards are sorted by coordinate-uncertainty (most precise first) and capped at
labels.assist_max (default 60).

STATUS: implemented. Pure review tool — writes no labels itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import swisstopo as st  # noqa: E402
from src.data.config import PROJECT_CRS, ensure_dir, load_config, parse_args  # noqa: E402


def _png_b64(arr_hwc):
    """(H,W,3) uint8 -> base64 PNG data-URI string."""
    import base64
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.fromarray(arr_hwc).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


CARD = """
<div class="card">
  <img src="{img}"/>
  <div class="meta">
    <b>#{i}</b> &nbsp; unc=<b>{unc}</b> m &nbsp; yr={yr}<br/>
    <span class="id">{gid}</span><br/>
    <a href="{sv}" target="_blank">Street View</a> ·
    <a href="{sat}" target="_blank">Sat</a> ·
    <a href="{swiss}" target="_blank">swisstopo</a>
  </div>
</div>"""

PAGE = """<!doctype html><meta charset="utf-8"><title>{aoi} — labeling assist</title>
<style>
 body{{font-family:system-ui,Arial;margin:16px;background:#111;color:#eee}}
 h1{{font-size:18px}} .grid{{display:flex;flex-wrap:wrap;gap:10px}}
 .card{{width:200px;background:#1d1d1d;border:1px solid #333;border-radius:8px;padding:6px}}
 .card img{{width:188px;height:188px;object-fit:cover;border-radius:5px;image-rendering:pixelated}}
 .meta{{font-size:11px;line-height:1.5;margin-top:4px}} .id{{color:#888;font-size:10px}}
 a{{color:#6cf;text-decoration:none}} a:hover{{text-decoration:underline}}
</style>
<h1>{aoi} — {n} candidate occurrence points (sorted by precision)</h1>
<p style="font-size:12px;color:#aaa">Click <b>Street View</b> to confirm the species at eye level,
then read the crop for the from-above signature. Coordinates are weak anchors —
verify before trusting. Crops are {win:.0f} m wide from SWISSIMAGE {yr}.</p>
<div class="grid">{cards}</div>
"""


def main() -> None:
    args = parse_args("Build an HTML labeling-assist sheet for an AOI's occurrence points.")
    cfg = load_config(args.config)

    import geopandas as gpd
    import numpy as np
    from pyproj import Transformer
    from rasterio.plot import reshape_as_image

    aoi = cfg["aoi"]["name"]
    bbox = cfg["aoi"]["bbox_lv95"]
    res_m = cfg["imagery"].get("res_m", 0.10)
    win_m = cfg["labels"].get("assist_window_m", 24)
    cap = cfg["labels"].get("assist_max", 60)

    occ = Path(cfg["paths"]["interim_dir"]) / "labels" / f"{aoi}_occurrences.geojson"
    if not occ.exists():
        raise SystemExit(f"Run 01_fetch_infoflora.py first; missing {occ}")
    gdf = gpd.read_file(occ).to_crs(PROJECT_CRS)
    if "coordUncertainty_m" in gdf.columns:
        gdf = gdf.sort_values("coordUncertainty_m", na_position="last")
    gdf = gdf.head(cap)
    print(f"=== labeling assist :: AOI '{aoi}' :: {len(gdf)} cards ===")

    # latest RGB vintage for the crops
    by_year = st.assets_by_year(st.search_stac_items(st.COLLECTION_RGB, bbox), st.TAG_RGB)
    yr = max(by_year)
    rgb_hrefs = by_year[yr]
    to_wgs = Transformer.from_crs(PROJECT_CRS, "EPSG:4326", always_xy=True)

    cards, half = [], win_m / 2
    for i, (_, row) in enumerate(gdf.iterrows(), 1):
        x, y = row.geometry.x, row.geometry.y
        arr, _ = st.read_window(rgb_hrefs, (x - half, y - half, x + half, y + half), res_m)
        if arr is None:
            continue
        img = reshape_as_image(arr).copy()
        # draw a small red crosshair at centre
        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        img[cy - 1:cy + 2, max(cx - 12, 0):cx + 12] = [255, 0, 0]
        img[max(cy - 12, 0):cy + 12, cx - 1:cx + 2] = [255, 0, 0]
        lon, lat = to_wgs.transform(x, y)
        cards.append(CARD.format(
            i=i, img=_png_b64(img),
            unc=row.get("coordUncertainty_m", "?"), yr=row.get("year", "?"),
            gid=row.get("gbifID", ""),
            sv=f"https://www.google.com/maps?q=&layer=c&cbll={lat:.6f},{lon:.6f}",
            sat=f"https://www.google.com/maps/@{lat:.6f},{lon:.6f},21z/data=!3m1!1e3",
            swiss=f"https://map.geo.admin.ch/?E={x:.0f}&N={y:.0f}&zoom=13&crosshair=marker",
        ))

    out_dir = ensure_dir(Path(cfg["paths"]["processed_dir"]) / aoi / "labeling")
    out = out_dir / "index.html"
    out.write_text(PAGE.format(aoi=aoi, n=len(cards), cards="".join(cards), yr=yr, win=win_m))
    print(f"=== wrote {len(cards)} cards -> {out} ===")
    print(f"Open it: file://{out.resolve()}")


if __name__ == "__main__":
    main()
