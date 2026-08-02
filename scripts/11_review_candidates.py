#!/usr/bin/env python
"""
11_review_candidates.py
========================
Build an HTML review sheet for score_candidates.py's output — same
Street View / satellite / swisstopo cross-referencing workflow as
05_labeling_assist.py, but sourced from model-scored random locations
(ranked by predicted probability) instead of raw GBIF occurrence points.

Purpose: surface both new positives (palms the current label set missed)
and hard negatives (locations that fooled the trained probe despite not
being palms) from a single manual review pass. Cards are shown highest-
probability first — those are the ones actually worth a human look,
either because they're a plausible undiscovered palm or because they're
confusing the model in an informative way.

The RGB crop shown is the EXACT input the model scored (same crop_size,
same point-centering) — reviewing what the model actually saw, not an
arbitrary nearby view. Uses a percentile contrast stretch for display
(same reasoning as the reconstruction-quality notebook fix — this RS-
delivery imagery is not standard 8-bit, a naive 0-255 clip would blow
every pixel to white).

STATUS: implemented. Pure review tool — writes no labels itself.
"""
from __future__ import annotations

import argparse
import base64
import sys
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import PROJECT_CRS, ensure_dir  # noqa: E402
from src.data.dataset import load_tile  # noqa: E402
from src.data.probe_dataset import crop_centered_on_point  # noqa: E402


def _png_b64(arr_hwc: np.ndarray) -> str:
    """(H,W,3) uint8 -> base64 PNG data-URI string."""
    buf = BytesIO()
    Image.fromarray(arr_hwc).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _crosshair(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    img[cy - 1:cy + 2, max(cx - 12, 0):cx + 12] = [255, 0, 0]
    img[max(cy - 12, 0):cy + 12, cx - 1:cx + 2] = [255, 0, 0]
    return img


def _rgb_crop(tile_path: Path, point: Point, crop_size: int) -> np.ndarray:
    """Load `tile_path`, crop centered on `point`, return an (H,W,3) uint8
    RGB image with a per-channel percentile contrast stretch — the same
    fix applied to the reconstruction-quality notebook, since this RS-
    delivery data is not standard 0-255.
    """
    arr, transform, _ = load_tile(tile_path)  # (C,H,W) raw values, NOT normalized
    crop = crop_centered_on_point(arr, transform, point, crop_size)

    in_chans = crop.shape[0]
    rgb_idx = slice(1, 4) if in_chans == 6 else slice(0, 3)
    rgb = crop[rgb_idx].astype("float32")

    out = np.zeros_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[c], (2, 98))
        out[c] = np.clip((rgb[c] - lo) / (hi - lo + 1e-6), 0, 1) * 255
    return np.transpose(out.astype("uint8"), (1, 2, 0))  # (H,W,3)


CARD = """
<div class="card">
  <img src="{img}" title="predicted_prob={prob:.3f}"/>
  <div class="meta">
    <b>#{i}</b> &nbsp; prob=<b class="{probclass}">{prob:.3f}</b><br/>
    <span class="id">{tile}</span><br/>
    <a href="{sv}" target="_blank">Street View</a> ·
    <a href="{sat}" target="_blank">Sat</a> ·
    <a href="{swiss}" target="_blank">swisstopo</a>
  </div>
</div>"""

PAGE = """<!doctype html><meta charset="utf-8"><title>candidate review</title>
<style>
 body{{font-family:system-ui,Arial;margin:16px;background:#111;color:#eee}}
 h1{{font-size:18px}} .grid{{display:flex;flex-wrap:wrap;gap:10px}}
 .card{{width:{imgw}px;background:#1d1d1d;border:1px solid #333;border-radius:8px;padding:6px}}
 .card img{{width:{imgw}px;height:{imgw}px;object-fit:cover;border-radius:5px;image-rendering:pixelated}}
 .meta{{font-size:11px;line-height:1.5;margin-top:4px}} .id{{color:#888;font-size:10px}}
 a{{color:#6cf;text-decoration:none}} a:hover{{text-decoration:underline}}
 .hi{{color:#e07a5f}} .lo{{color:#8fb45c}}
</style>
<h1>candidate review — {n} locations</h1>
<p style="font-size:12px;color:#aaa">{subtitle}</p>
<div class="grid">{cards}</div>
"""


def parse_review_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an HTML review sheet for score_candidates.py's output.")
    p.add_argument("--candidates", type=Path, required=True, help="Path to candidate_scores.geojson.")
    p.add_argument("--tile-dirs", type=Path, nargs="+", required=True, help="Same tile dirs used for scoring, to resolve full paths from filenames.")
    p.add_argument("--max-cards", type=int, default=60)
    p.add_argument("--min-prob", type=float, default=0.0, help="Only show candidates with predicted_prob >= this.")
    p.add_argument("--crop-size", type=int, default=224, help="Should match --crop-size used when scoring.")
    p.add_argument("--img-px", type=int, default=200, help="Display size (pixels) for each card's image.")
    p.add_argument("--output", type=Path, default=Path("candidate_review/index.html"))
    return p.parse_args()


def main() -> None:
    args = parse_review_args()

    gdf = gpd.read_file(args.candidates).to_crs(PROJECT_CRS)
    gdf = gdf[gdf["predicted_prob"] >= args.min_prob].sort_values("predicted_prob", ascending=False)
    gdf = gdf.head(args.max_cards)
    print(f"=== reviewing {len(gdf)} candidates (of {len(gpd.read_file(args.candidates))} scored) ===")

    tile_by_name = {}
    for d in args.tile_dirs:
        for p in d.glob("*_nirchm.tif"):
            tile_by_name[p.name] = p
        for p in d.glob("*_rgbchm.tif"):
            tile_by_name[p.name] = p

    to_wgs = Transformer.from_crs(PROJECT_CRS, "EPSG:4326", always_xy=True)

    cards = []
    for i, (_, row) in enumerate(gdf.iterrows(), 1):
        tile_path = tile_by_name.get(row["tile"])
        if tile_path is None:
            print(f"[warn] tile not found for candidate #{i} ({row['tile']}) — skipping")
            continue

        point = Point(row.geometry.x, row.geometry.y)
        rgb = _rgb_crop(tile_path, point, args.crop_size)
        rgb = _crosshair(rgb.copy())

        lon, lat = to_wgs.transform(row.geometry.x, row.geometry.y)
        prob = row["predicted_prob"]

        cards.append(CARD.format(
            i=i, img=_png_b64(rgb), prob=prob,
            probclass="hi" if prob >= 0.5 else "lo",
            tile=row["tile"],
            sv=f"https://www.google.com/maps?q=&layer=c&cbll={lat:.6f},{lon:.6f}",
            sat=f"https://www.google.com/maps/@{lat:.6f},{lon:.6f},21z/data=!3m1!1e3",
            swiss=f"https://map.geo.admin.ch/?E={row.geometry.x:.0f}&N={row.geometry.y:.0f}&zoom=13&crosshair=marker",
        ))

    subtitle = (
        f"Ranked by predicted probability (production probe trained on all current labels), "
        f"highest first — min_prob={args.min_prob}. High-probability cards are worth a look "
        f"either way: a genuine unlabeled palm, or a hard negative that fooled the model."
    )

    out_dir = ensure_dir(args.output.parent)
    out = args.output
    out.write_text(PAGE.format(n=len(cards), cards="".join(cards), subtitle=subtitle, imgw=args.img_px))
    print(f"=== wrote {len(cards)} cards -> {out} ===")
    print(f"Open it: file://{out.resolve()}")


if __name__ == "__main__":
    main()
