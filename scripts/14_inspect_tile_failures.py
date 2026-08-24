#!/usr/bin/env python
"""
14_inspect_tile_failures.py
===========================
Visual inspection tool for specific tiles where leave-one-tile-out CV
showed poor recall on their confirmed positives — renders every known
positive location within the given tiles as a read-only card (RGB crop +
what a probe trained on all current labels currently predicts for it), so
you can look at exactly what the model sees and form a hypothesis about
why it's missing confirmed palms there (dense mixed canopy? occlusion?
unusual lighting? a genuinely atypical palm?).

Not a rerun of the CV number itself: the probe here is trained on ALL
current labels, including this tile's own positives — not the held-out
probe leave_one_tile_out_cv actually evaluated with for that fold. That's
a practical approximation for visual inspection, not a faithful
reproduction of the CV fold's exact prediction. A low predicted_prob here
is still a meaningful signal (this probe has seen strictly more data than
the CV fold's did, so if it's STILL confidently wrong even with this
tile's own examples included in training, that's a stronger sign of a
real, hard case rather than an artifact of what got held out).

STATUS: implemented. Read-only — writes no labels, corrects nothing.
"""
from __future__ import annotations

import argparse
import base64
import sys
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import PROJECT_CRS  # noqa: E402
from src.data.dataset import load_confirmed_points, load_tile, normalize  # noqa: E402
from src.data.probe_dataset import (  # noqa: E402
    PalmProbeDataset,
    crop_centered_on_point,
    sample_negative_points,
)
from src.inference.score_candidates import build_production_probe  # noqa: E402
from src.models.mae import MaskedAutoencoder, adapt_patch_embed  # noqa: E402
from src.training.pretrain_ssl import glob_tiles  # noqa: E402


def _png_b64(arr_hwc: np.ndarray) -> str:
    buf = BytesIO()
    Image.fromarray(arr_hwc).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _load_crop_and_rgb(tile_path: Path, point: Point, crop_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load `tile_path` once, return (raw_crop, display_rgb) — raw_crop for
    normalize()+scoring, display_rgb via the same percentile contrast-stretch
    approach as 10_review_candidates.py (this RS-delivery imagery isn't
    standard 0-255, a naive clip blows every pixel to white)."""
    arr, transform, _ = load_tile(tile_path)
    raw_crop = crop_centered_on_point(arr, transform, point, crop_size)

    in_chans = raw_crop.shape[0]
    rgb_idx = slice(1, 4) if in_chans == 6 else slice(0, 3)
    rgb = raw_crop[rgb_idx].astype("float32")
    out = np.zeros_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[c], (2, 98))
        out[c] = np.clip((rgb[c] - lo) / (hi - lo + 1e-6), 0, 1) * 255
    display_rgb = np.transpose(out.astype("uint8"), (1, 2, 0))

    return raw_crop, display_rgb


CARD = """
<div class="card">
  <img src="{img}" title="predicted_prob={prob:.3f}"/>
  <div class="meta">
    prob=<b class="{probclass}">{prob:.3f}</b> — ground truth: <b>palm</b><br/>
    <span class="id">{tile}</span><br/>
    <a href="{sv}" target="_blank">Street View</a> ·
    <a href="{sat}" target="_blank">Sat</a> ·
    <a href="{swiss}" target="_blank">swisstopo</a>
  </div>
</div>"""

PAGE = """<!doctype html><meta charset="utf-8"><title>tile failure inspection</title>
<style>
 body{{font-family:system-ui,Arial;margin:16px;background:#111;color:#eee}}
 h1{{font-size:18px}} .grid{{display:flex;flex-wrap:wrap;gap:10px}}
 .card{{width:{imgw}px;background:#1d1d1d;border:1px solid #333;border-radius:8px;padding:6px}}
 .card img{{width:{imgw}px;height:{imgw}px;object-fit:cover;border-radius:5px;image-rendering:pixelated}}
 .meta{{font-size:11px;line-height:1.5;margin-top:4px}} .id{{color:#888;font-size:10px}}
 a{{color:#6cf;text-decoration:none}} a:hover{{text-decoration:underline}}
 .hi{{color:#8fb45c}} .lo{{color:#e07a5f}}
</style>
<h1>tile failure inspection — {n} known positives</h1>
<p style="font-size:12px;color:#aaa">Every card here is a CONFIRMED palm (ground truth). Low predicted_prob
(shown in red) means the current probe — trained on all current labels — still gets it wrong even having
seen this tile's own examples during training. Look for a pattern: dense mixed canopy, occlusion, unusual
lighting, an atypical palm shape.</p>
<div class="grid">{cards}</div>
"""


def parse_inspect_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visually inspect confirmed positives in specific tiles the CV eval got wrong."
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tile-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--confirmed-points", type=Path, required=True)
    p.add_argument("--scouted-points", type=Path, default=None)
    p.add_argument("--reviewed-positives", type=Path, default=None)
    p.add_argument("--hard-negatives", type=Path, default=None)
    p.add_argument(
        "--tiles", type=str, nargs="+", required=True,
        help="Tile filenames to restrict to, e.g. bellinzona_full_nir_2024_c021_r076_nirchm.tif "
             "(exactly what leave_one_tile_out_cv's fold names look like).",
    )
    p.add_argument("--n-negatives", type=int, default=30)
    p.add_argument("--min-distance-m", type=float, default=20.0)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--img-px", type=int, default=200)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--pos-weight-multiplier", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("review/tile_failure_inspection/index.html"))
    return p.parse_args()


def main() -> None:
    args = parse_inspect_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stats = checkpoint["stats"]
    in_chans = len(stats.mean)
    backbone_name = checkpoint.get("backbone_name", "vit_small_patch14_dinov2.lvd142m")
    embed_dim = checkpoint.get("embed_dim", 384)

    backbone = timm.create_model(backbone_name, pretrained=True)
    backbone = adapt_patch_embed(backbone, in_chans)
    model = MaskedAutoencoder(backbone, 224, 14, in_chans, embed_dim, 192, 4, 6, 0.75)
    model.backbone.patch_embed.strict_img_size = False
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    points = load_confirmed_points(args.confirmed_points)
    distinct = points[points["ndvi_signal_current"] == "distinct"]
    none_signal = points[points["ndvi_signal_current"] == "none"]

    positive_geoms = distinct.geometry
    if args.scouted_points is not None:
        positive_geoms = pd.concat([positive_geoms, gpd.read_file(args.scouted_points).geometry], ignore_index=True)
    if args.reviewed_positives is not None:
        positive_geoms = pd.concat([positive_geoms, gpd.read_file(args.reviewed_positives).geometry], ignore_index=True)

    tile_paths = glob_tiles(args.tile_dirs, in_chans)

    negatives = sample_negative_points(
        positive_points=positive_geoms, candidate_points=none_signal.geometry,
        tile_paths=tile_paths, n_negatives=args.n_negatives,
        min_distance_m=args.min_distance_m, seed=args.seed,
    )
    if args.hard_negatives is not None:
        negatives = negatives + [(pt.x, pt.y) for pt in gpd.read_file(args.hard_negatives).geometry]

    dataset = PalmProbeDataset(positive_geoms, negatives, tile_paths, stats, args.crop_size)
    print(f"training production probe on {len(dataset)} examples "
          f"({sum(1 for e in dataset.examples if e[3] == 1.0)} positive, "
          f"{sum(1 for e in dataset.examples if e[3] == 0.0)} negative)")
    probe = build_production_probe(
        model, dataset, device, embed_dim, args.epochs, args.lr, args.pos_weight_multiplier, args.hidden_dim
    )

    target_tiles = set(args.tiles)

    # Every (tile, x, y) example in the dataset whose tile matches one of the
    # target tiles AND is a true positive — i.e. exactly the population
    # leave_one_tile_out_cv's per-fold true_pos/false_neg counts summed over.
    targets = [
        (tile, x, y) for tile, x, y, label in dataset.examples
        if label == 1.0 and tile.name in target_tiles
    ]
    print(f"found {len(targets)} confirmed-positive example(s) across {len(target_tiles)} target tile(s)")

    to_wgs = Transformer.from_crs(PROJECT_CRS, "EPSG:4326", always_xy=True)
    model.eval()

    cards = []
    for tile_path, x, y in targets:
        point = Point(x, y)
        raw_crop, rgb = _load_crop_and_rgb(tile_path, point, args.crop_size)

        crop_norm = normalize(raw_crop, stats)
        img_t = torch.from_numpy(crop_norm).float().unsqueeze(0).to(device)
        with torch.no_grad():
            encoded = model.encode_full(img_t)
            logit = probe(encoded[:, 0]).squeeze(-1)
            prob = torch.sigmoid(logit).item()

        lon, lat = to_wgs.transform(x, y)
        cards.append(CARD.format(
            img=_png_b64(rgb), prob=prob,
            probclass="lo" if prob < 0.5 else "hi",
            tile=tile_path.name,
            sv=f"https://www.google.com/maps?q=&layer=c&cbll={lat:.6f},{lon:.6f}",
            sat=f"https://www.google.com/maps/@{lat:.6f},{lon:.6f},21z/data=!3m1!1e3",
            swiss=f"https://map.geo.admin.ch/?E={x:.0f}&N={y:.0f}&zoom=13&crosshair=marker",
        ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(PAGE.format(n=len(cards), cards="".join(cards), imgw=args.img_px))
    print(f"wrote {len(cards)} cards -> {args.output}")


if __name__ == "__main__":
    main()
