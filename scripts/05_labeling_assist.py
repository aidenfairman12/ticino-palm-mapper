#!/usr/bin/env python
"""
05_labeling_assist.py
=====================
Build a self-contained HTML review sheet to VERIFY occurrence points fast, instead
of eyeballing the orthophoto cold.

For each occurrence point it renders one card:
  - a native-resolution RGB ortho crop centred on the point (red +),
  - a colour-infrared (NIR,R,G) crop + the NDVI value at the point, IF the point
    falls inside the delivered SWISSIMAGE RS footprint (06/07's NIR data),
  - GBIF metadata (year, coordinate-uncertainty, id),
  - a Google Street View link (oblique eye-level confirmation),
  - a Google satellite link and a swisstopo map.geo.admin.ch link (both at the point).

Open the resulting index.html and click through: Street View confirms species, the
crops show the from-above signature. This is the workflow we used to confirm a palm
at Parco giochi Usignolo — it's how you build an HONEST, ground-truthed test set
(occurrence-point coordinates alone are unreliable; some land on rooftops).

Card ORDER: if 06_prioritize_labels.py has been run, its ranked shortlist
(candidates_ranked.csv — road proximity + NIR coverage) is used, and each card
shows the road-distance / "near a road" flag. Otherwise falls back to sorting by
coordinate-uncertainty alone. Either way, capped at labels.assist_max (default 60).

NDVI is a SCREENING SIGNAL, not a label: a high value only means "plausible
evergreen vegetation here," not "confirmed palm." Still requires a human look.

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


def _crosshair(img):
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    img[cy - 1:cy + 2, max(cx - 12, 0):cx + 12] = [255, 0, 0]
    img[max(cy - 12, 0):cy + 12, cx - 1:cx + 2] = [255, 0, 0]
    return img


CARD = """
<div class="card">
  <div class="imgs">
    <img src="{img}" title="RGB"/>
    {nir_img}
  </div>
  <div class="meta">
    <b>#{i}</b> &nbsp; unc=<b>{unc}</b> m &nbsp; yr={yr} {roadbadge}<br/>
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
 .card{{width:{cardw}px;background:#1d1d1d;border:1px solid #333;border-radius:8px;padding:6px}}
 .imgs{{display:flex;gap:3px}}
 .imgs img{{width:{imgw}px;height:{imgw}px;object-fit:cover;border-radius:5px;image-rendering:pixelated}}
 .meta{{font-size:11px;line-height:1.5;margin-top:4px}} .id{{color:#888;font-size:10px}}
 a{{color:#6cf;text-decoration:none}} a:hover{{text-decoration:underline}}
 .road-yes{{color:#8fb45c}} .road-no{{color:#666}} .ndvi{{color:#8fb45c;font-size:9.5px}}
</style>
<h1>{aoi} — {n} candidate occurrence points</h1>
<p style="font-size:12px;color:#aaa">{subtitle}</p>
<div class="grid">{cards}</div>
"""


def load_candidates(cfg, gdf):
    """Order + annotate points using 06's ranked shortlist, if it exists."""
    aoi = cfg["aoi"]["name"]
    ranked_csv = Path(cfg["paths"]["processed_dir"]) / aoi / "labeling" / "candidates_ranked.csv"
    if not ranked_csv.exists():
        if "coordUncertainty_m" in gdf.columns:
            gdf = gdf.sort_values("coordUncertainty_m", na_position="last")
        gdf["dist_to_road_m"] = None
        gdf["near_road"] = None
        gdf["nir_available"] = None
        return gdf, False

    import pandas as pd

    ranked = pd.read_csv(ranked_csv)
    gdf = gdf.merge(
        ranked[["gbifID", "dist_to_road_m", "near_road", "nir_available", "priority"]],
        on="gbifID", how="right",  # right: keep 06's order/selection, drop anything it excluded
    )
    gdf = gdf.sort_values(["priority", "dist_to_road_m"], na_position="last")
    return gdf, True


def main() -> None:
    args = parse_args("Build an HTML labeling-assist sheet for an AOI's occurrence points.")
    cfg = load_config(args.config)

    import numpy as np
    import geopandas as gpd
    from pyproj import Transformer
    from rasterio.plot import reshape_as_image

    aoi = cfg["aoi"]["name"]
    bbox = cfg["aoi"]["bbox_lv95"]
    res_m = cfg["imagery"].get("res_m", 0.10)
    win_m = cfg["labels"].get("assist_window_m", 24)
    cap = cfg["labels"].get("assist_max", 60)
    rs_dir = Path(cfg["labels"].get("rs_dir", "data/raw/swissimage_rs/lugano_delivery_2026-07"))
    rs_date = str(cfg["imagery"].get("rs_date", ""))

    occ = Path(cfg["paths"]["interim_dir"]) / "labels" / f"{aoi}_occurrences.geojson"
    if not occ.exists():
        raise SystemExit(f"Run 01_fetch_infoflora.py first; missing {occ}")
    gdf = gpd.read_file(occ).to_crs(PROJECT_CRS)

    gdf, ranked = load_candidates(cfg, gdf)
    gdf = gdf.head(cap)
    print(f"=== labeling assist :: AOI '{aoi}' :: {len(gdf)} cards "
          f"({'ranked shortlist' if ranked else 'unranked, run 06 first for a shortlist'}) ===")

    # latest free RGB vintage for the base crop
    by_year = st.assets_by_year(st.search_stac_items(st.COLLECTION_RGB, bbox), st.TAG_RGB)
    yr = max(by_year)
    rgb_hrefs = by_year[yr]

    # NIR/RS hrefs, if available, for the matching acquisition date
    rs_hrefs = []
    if rs_dir.exists():
        dates = sorted({p.name[:8] for p in rs_dir.glob("*.tif")})
        rs_date = rs_date if rs_date in dates else (dates[0] if dates else "")
        rs_hrefs = [str(p) for p in sorted(rs_dir.glob(f"{rs_date}_*.tif"))]
        print(f"[nir] {len(rs_hrefs)} RS strips available for date {rs_date}")

    to_wgs = Transformer.from_crs(PROJECT_CRS, "EPSG:4326", always_xy=True)
    img_px = 188

    cards, half = [], win_m / 2
    for i, (_, row) in enumerate(gdf.iterrows(), 1):
        x, y = row.geometry.x, row.geometry.y
        bounds = (x - half, y - half, x + half, y + half)

        arr, _ = st.read_window(rgb_hrefs, bounds, res_m)
        if arr is None:
            continue
        img = _crosshair(reshape_as_image(arr).copy())

        nir_img_html, ndvi_txt = "", ""
        if rs_hrefs and row.get("nir_available"):
            rs_arr, _ = st.read_window(rs_hrefs, bounds, res_m, resampling="bilinear")
            if rs_arr is not None:
                nir, r = rs_arr[0].astype("float32"), rs_arr[1].astype("float32")
                ndvi = (nir - r) / (nir + r + 1e-6)
                lo, hi = np.percentile(nir, (2, 98))
                cir = np.dstack([
                    np.clip((nir - lo) / (hi - lo + 1e-6), 0, 1) * 255,
                    np.clip((r - np.percentile(r, 2)) / (np.percentile(r, 98) - np.percentile(r, 2) + 1e-6), 0, 1) * 255,
                    np.clip((rs_arr[2].astype("float32") - np.percentile(rs_arr[2], 2)) /
                            (np.percentile(rs_arr[2], 98) - np.percentile(rs_arr[2], 2) + 1e-6), 0, 1) * 255,
                ]).astype("uint8")
                cir = _crosshair(cir.copy())
                h, w = ndvi.shape
                ndvi_center = float(ndvi[h // 2, w // 2])
                nir_img_html = f'<img src="{_png_b64(cir)}" title="Colour-IR (NIR,R,G)"/>'
                ndvi_txt = f'<div class="ndvi">NDVI@pt={ndvi_center:.2f}</div>'

        lon, lat = to_wgs.transform(x, y)
        near = row.get("near_road")
        if near is True:
            roadbadge = f'<span class="road-yes">● road {row.get("dist_to_road_m", "?")}m</span>'
        elif near is False:
            roadbadge = '<span class="road-no">○ no road nearby</span>'
        else:
            roadbadge = ""

        cards.append(CARD.format(
            i=i, img=_png_b64(img), nir_img=nir_img_html + ndvi_txt,
            unc=row.get("coordUncertainty_m", "?"), yr=row.get("year", "?"), roadbadge=roadbadge,
            gid=row.get("gbifID", ""),
            sv=f"https://www.google.com/maps?q=&layer=c&cbll={lat:.6f},{lon:.6f}",
            sat=f"https://www.google.com/maps/@{lat:.6f},{lon:.6f},21z/data=!3m1!1e3",
            swiss=f"https://map.geo.admin.ch/?E={x:.0f}&N={y:.0f}&zoom=13&crosshair=marker",
        ))

    subtitle = (
        "Ranked by 06_prioritize_labels.py: road-checkable + NIR-covered candidates first. "
        if ranked else
        "Run 06_prioritize_labels.py first to rank by road proximity + NIR coverage. "
    ) + f"Click Street View to confirm the species; crops are {win_m:.0f} m wide (RGB {yr}" + \
        (f", CIR {rs_date}" if rs_hrefs else "") + \
        "). Coordinates are weak anchors and NDVI is a screening signal only — verify before trusting either."

    out_dir = ensure_dir(Path(cfg["paths"]["processed_dir"]) / aoi / "labeling")
    out = out_dir / "index.html"
    cardw = img_px * 2 + 24 if rs_hrefs else img_px + 12
    out.write_text(PAGE.format(aoi=aoi, n=len(cards), cards="".join(cards), subtitle=subtitle,
                                cardw=cardw, imgw=img_px))
    print(f"=== wrote {len(cards)} cards -> {out} ===")
    print(f"Open it: file://{out.resolve()}")


if __name__ == "__main__":
    main()
