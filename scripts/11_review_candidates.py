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
from rasterio.transform import rowcol, xy
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import PROJECT_CRS, ensure_dir  # noqa: E402
from src.data.dataset import load_tile  # noqa: E402


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


def _rgb_crop(
    tile_path: Path, point: Point, crop_size: int
) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    """Load `tile_path`, crop centered on `point` (same clamped-offset logic
    as crop_centered_on_point), return an (H,W,3) uint8 RGB image with a
    per-channel percentile contrast stretch, plus the crop's real-world
    bounding box (xmin, ymin, xmax, ymax) — needed so the review page can
    map a click on the displayed image back to a real-world coordinate
    (the scored point isn't always exactly on the palm; clicking lets the
    reviewer record where it actually is).

    Returns None if the crop is mostly nodata — tile bounding boxes are
    rectangular, but actual delivery coverage within them can have real
    gaps, and a percentile contrast stretch over an all-zero region just
    produces an unreviewable black card (same underlying issue
    score_candidates.py's score_locations guards against on the scoring
    side; this guards the display side, e.g. for candidates scored before
    that fix existed).
    """
    arr, transform, _ = load_tile(tile_path)  # (C,H,W) raw values, NOT normalized
    H, W = arr.shape[1], arr.shape[2]
    rows_px, cols_px = rowcol(transform, point.x, point.y)
    row_start = max(0, min(rows_px - crop_size // 2, H - crop_size))
    col_start = max(0, min(cols_px - crop_size // 2, W - crop_size))
    crop = arr[:, row_start:row_start + crop_size, col_start:col_start + crop_size]

    if (crop == 0).all(axis=0).mean() > 0.5:
        return None

    x0, y0 = xy(transform, row_start, col_start, offset="ul")
    x1, y1 = xy(transform, row_start + crop_size, col_start + crop_size, offset="ul")
    bbox = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    in_chans = crop.shape[0]
    rgb_idx = slice(1, 4) if in_chans == 6 else slice(0, 3)
    rgb = crop[rgb_idx].astype("float32")

    out = np.zeros_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[c], (2, 98))
        out[c] = np.clip((rgb[c] - lo) / (hi - lo + 1e-6), 0, 1) * 255
    return np.transpose(out.astype("uint8"), (1, 2, 0)), bbox  # (H,W,3), bbox


CARD = """
<div class="card" data-idx="{i}" data-tile="{tile}" data-x="{x:.2f}" data-y="{y:.2f}" data-prob="{prob:.4f}"
     data-cropxmin="{cxmin:.2f}" data-cropymin="{cymin:.2f}" data-cropxmax="{cxmax:.2f}" data-cropymax="{cymax:.2f}">
  <div class="imgwrap"><img src="{img}" title="predicted_prob={prob:.3f} — click the actual palm to correct its location"/></div>
  <div class="meta">
    <b>#{i}</b> &nbsp; prob=<b class="{probclass}">{prob:.3f}</b><br/>
    <span class="id">{tile}</span><br/>
    <a href="{sv}" target="_blank">Street View</a> ·
    <a href="{sat}" target="_blank">Sat</a> ·
    <a href="{swiss}" target="_blank">swisstopo</a>
  </div>
  <div class="verdict-btns">
    <button class="vb vb-palm" onclick="setVerdict({i},'palm')" title="Key: P">Palm</button>
    <button class="vb vb-not" onclick="setVerdict({i},'not_palm')" title="Key: N">Not palm</button>
    <button class="vb vb-unsure" onclick="setVerdict({i},'unsure')" title="Key: U">Unsure</button>
  </div>
</div>"""

PAGE = """<!doctype html><meta charset="utf-8"><title>candidate review</title>
<style>
 body{{font-family:system-ui,Arial;margin:16px;background:#111;color:#eee}}
 h1{{font-size:18px}} .grid{{display:flex;flex-wrap:wrap;gap:10px}}
 .card{{width:{imgw}px;background:#1d1d1d;border:2px solid #333;border-radius:8px;padding:6px;transition:border-color .15s}}
 .imgwrap{{position:relative;width:{imgw}px;height:{imgw}px}}
 .imgwrap img{{width:{imgw}px;height:{imgw}px;object-fit:cover;border-radius:5px;image-rendering:pixelated;cursor:crosshair}}
 .click-marker{{position:absolute;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;background:#4fc3f7;border:2px solid #fff;pointer-events:none;display:none;box-shadow:0 0 3px #000}}
 .meta{{font-size:11px;line-height:1.5;margin-top:4px}} .id{{color:#888;font-size:10px}}
 a{{color:#6cf;text-decoration:none}} a:hover{{text-decoration:underline}}
 .hi{{color:#e07a5f}} .lo{{color:#8fb45c}}
 .verdict-btns{{display:flex;gap:4px;margin-top:6px}}
 .vb{{flex:1;font-size:11px;padding:5px 0;border:1px solid #444;border-radius:4px;background:#262626;color:#ccc;cursor:pointer}}
 .vb:hover{{background:#333}}
 .vb-palm.active{{background:#2e7d32;border-color:#2e7d32;color:#fff}}
 .vb-not.active{{background:#8b3a3a;border-color:#8b3a3a;color:#fff}}
 .vb-unsure.active{{background:#8a6d1a;border-color:#8a6d1a;color:#fff}}
 .card.verdict-palm{{border-color:#2e7d32}}
 .card.verdict-not_palm{{border-color:#8b3a3a;opacity:.55}}
 .card.verdict-unsure{{border-color:#8a6d1a}}
 #toolbar{{position:sticky;top:0;background:#111;padding:10px 0;z-index:10;display:flex;align-items:center;gap:14px;border-bottom:1px solid #333;margin-bottom:10px}}
 #toolbar button{{font-size:13px;padding:7px 14px;border-radius:5px;border:1px solid #555;background:#1d1d1d;color:#eee;cursor:pointer}}
 #toolbar button:hover{{background:#292929}}
 #progress{{font-size:12px;color:#aaa}}
</style>
<div id="toolbar">
  <button onclick="exportVerdicts()">Export verdicts (CSV)</button>
  <span id="progress">0 / {n} reviewed</span>
  <span style="font-size:11px;color:#666">Click a card's image to focus it (P/N/U to verdict) — if the red crosshair isn't on the palm, click the actual palm(s) instead (each click adds a marker; multiple palms in one crop get multiple markers, exported as separate rows). Right-click an image to clear its markers. Unlabeled cards are skipped on export.</span>
</div>
<h1>candidate review — {n} locations</h1>
<p style="font-size:12px;color:#aaa">{subtitle}</p>
<div class="grid">{cards}</div>
<script>
const verdicts = {{}};
const markers = {{}};  // idx -> [[world_x, world_y], ...], accumulates across clicks (multiple palms in one crop)
const TOTAL = {n};
let focusedIdx = null;

function setVerdict(idx, verdict) {{
  verdicts[idx] = verdict;
  const card = document.querySelector(`[data-idx="${{idx}}"]`);
  card.classList.remove('verdict-palm', 'verdict-not_palm', 'verdict-unsure');
  card.classList.add('verdict-' + verdict);
  card.querySelectorAll('.vb').forEach(b => b.classList.remove('active'));
  card.querySelector('.vb-' + (verdict === 'not_palm' ? 'not' : verdict)).classList.add('active');
  updateProgress();
}}

function updateProgress() {{
  document.getElementById('progress').textContent = Object.keys(verdicts).length + ' / ' + TOTAL + ' reviewed';
}}

function exportVerdicts() {{
  const rows = [['index', 'tile', 'x', 'y', 'corrected_x', 'corrected_y', 'predicted_prob', 'verdict']];
  document.querySelectorAll('.card').forEach(card => {{
    const idx = card.dataset.idx;
    if (!(idx in verdicts)) return;
    const verdict = verdicts[idx];
    const pts = markers[idx] || [];

    // Multiple markers only matter for a palm verdict — each one is a separate
    // real palm in the same crop, so each becomes its own row. A not_palm/unsure
    // verdict always exports exactly one row (any markers clicked while figuring
    // out the card don't create meaningless extra negative rows at the same spot).
    if (verdict === 'palm' && pts.length > 0) {{
      pts.forEach(([cx, cy]) => {{
        rows.push([idx, card.dataset.tile, card.dataset.x, card.dataset.y, cx.toFixed(2), cy.toFixed(2), card.dataset.prob, verdict]);
      }});
    }} else {{
      rows.push([idx, card.dataset.tile, card.dataset.x, card.dataset.y, card.dataset.x, card.dataset.y, card.dataset.prob, verdict]);
    }}
  }});
  const csv = rows.map(r => r.join(',')).join('\\n');
  const blob = new Blob([csv], {{type: 'text/csv'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'candidate_verdicts.csv';
  a.click();
  URL.revokeObjectURL(url);
}}

// corrected_x/corrected_y default to the original scored point (x, y) when no
// marker has been placed. Clicking the image records where a palm actually is,
// in case the scored/crosshair point is off by a few meters or there's more
// than one palm in the crop — each click adds another marker (blue dot) rather
// than replacing the previous one.
document.querySelectorAll('.card').forEach(card => {{
  const img = card.querySelector('img');
  const wrap = card.querySelector('.imgwrap');
  const idx = card.dataset.idx;
  markers[idx] = [];

  function clearMarkers() {{
    markers[idx] = [];
    wrap.querySelectorAll('.click-marker').forEach(m => m.remove());
  }}

  img.addEventListener('click', (e) => {{
    focusedIdx = idx;

    const rect = img.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;

    const marker = document.createElement('div');
    marker.className = 'click-marker';
    marker.style.left = (fx * 100) + '%';
    marker.style.top = (fy * 100) + '%';
    marker.style.display = 'block';
    wrap.appendChild(marker);

    const xmin = parseFloat(card.dataset.cropxmin), xmax = parseFloat(card.dataset.cropxmax);
    const ymin = parseFloat(card.dataset.cropymin), ymax = parseFloat(card.dataset.cropymax);
    const worldX = xmin + fx * (xmax - xmin);
    const worldY = ymax - fy * (ymax - ymin);  // image y grows downward, world y grows upward
    markers[idx].push([worldX, worldY]);
  }});

  img.addEventListener('contextmenu', (e) => {{
    e.preventDefault();
    clearMarkers();
  }});
}});

document.addEventListener('keydown', (e) => {{
  if (focusedIdx === null) return;
  const key = e.key.toLowerCase();
  if (key === 'p') setVerdict(focusedIdx, 'palm');
  else if (key === 'n') setVerdict(focusedIdx, 'not_palm');
  else if (key === 'u') setVerdict(focusedIdx, 'unsure');
}});
</script>
"""


def parse_review_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an HTML review sheet for score_candidates.py's output.")
    p.add_argument("--candidates", type=Path, required=True, help="Path to candidate_scores.geojson.")
    p.add_argument("--tile-dirs", type=Path, nargs="+", required=True, help="Same tile dirs used for scoring, to resolve full paths from filenames.")
    p.add_argument("--max-cards", type=int, default=60)
    p.add_argument("--min-prob", type=float, default=0.0, help="Only show candidates with predicted_prob >= this.")
    p.add_argument(
        "--offset", type=int, default=0,
        help="Skip this many top-ranked candidates before taking --max-cards — lets "
             "one big scored file (from a single expensive scoring pass) be reviewed "
             "incrementally across multiple sessions without re-scoring: e.g. "
             "--offset 60 for your second session's next batch after reviewing the "
             "first 60, --offset 120 for the third, etc.",
    )
    p.add_argument("--crop-size", type=int, default=224, help="Should match --crop-size used when scoring.")
    p.add_argument("--img-px", type=int, default=200, help="Display size (pixels) for each card's image.")
    p.add_argument("--output", type=Path, default=Path("candidate_review/index.html"))
    return p.parse_args()


def main() -> None:
    args = parse_review_args()

    gdf = gpd.read_file(args.candidates).to_crs(PROJECT_CRS)
    gdf = gdf[gdf["predicted_prob"] >= args.min_prob].sort_values("predicted_prob", ascending=False)
    gdf = gdf.iloc[args.offset:args.offset + args.max_cards]
    print(f"=== reviewing {len(gdf)} candidates (offset {args.offset}, "
          f"of {len(gpd.read_file(args.candidates))} scored) ===")

    tile_by_name = {}
    for d in args.tile_dirs:
        for p in d.glob("*_nirchm.tif"):
            tile_by_name[p.name] = p
        for p in d.glob("*_rgbchm.tif"):
            tile_by_name[p.name] = p

    to_wgs = Transformer.from_crs(PROJECT_CRS, "EPSG:4326", always_xy=True)

    cards = []
    for i, (_, row) in enumerate(gdf.iterrows(), args.offset + 1):
        tile_path = tile_by_name.get(row["tile"])
        if tile_path is None:
            print(f"[warn] tile not found for candidate #{i} ({row['tile']}) — skipping")
            continue

        point = Point(row.geometry.x, row.geometry.y)
        crop_result = _rgb_crop(tile_path, point, args.crop_size)
        if crop_result is None:
            print(f"[warn] candidate #{i} ({row['tile']}) is mostly nodata — skipping")
            continue
        rgb, (cxmin, cymin, cxmax, cymax) = crop_result
        rgb = _crosshair(rgb.copy())

        lon, lat = to_wgs.transform(row.geometry.x, row.geometry.y)
        prob = row["predicted_prob"]

        cards.append(CARD.format(
            i=i, img=_png_b64(rgb), prob=prob,
            probclass="hi" if prob >= 0.5 else "lo",
            tile=row["tile"], x=row.geometry.x, y=row.geometry.y,
            cxmin=cxmin, cymin=cymin, cxmax=cxmax, cymax=cymax,
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
