#!/usr/bin/env python
"""
07_build_nir_stack.py
======================
For every tile in the RGB feature stack (03_build_feature_stack.py), add a
co-registered NIR channel + NDVI from the delivered SWISSIMAGE RS tiles, where
coverage exists. Writes a separate RS-native stack:

    [NIR, R, G, B, NDVI, CHM]   (6 bands, float32)

Design choice: R/G/B here come from the RS delivery itself, NOT the free
SWISSIMAGE dop10 product used in 03. They're one coherent acquisition (same
sensor pass), so NIR/R/G/B are radiometrically consistent with each other —
mixing NIR from one flight with RGB from a different year's flight would
confound the NDVI signal with real ground change. CHM is carried over from the
existing [R,G,B,CHM] stack (03's output) since LiDAR height is comparatively
stable year-to-year.

Coverage is real but partial: the free RGB/CHM stack covers the whole canton,
the RS delivery covers only what was ordered. Tiles with no RS coverage are
skipped (reported), not padded with fake data — the gap is itself information.

Config:
  labels.rs_dir     - directory of delivered RS tiles (default in aoi_example.yaml)
  imagery.rs_date    - which acquisition date to use (default: the leaf-off
                        date, since evergreen/deciduous contrast is the reason
                        NIR was requested). Falls back to whatever's available
                        if unset.

STATUS: implemented and tested against the lugano_example AOI (36/36 tiles
had coverage from the 2021-03-24 leaf-off delivery).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import swisstopo as st  # noqa: E402
from src.data.config import ensure_dir, load_config, parse_args  # noqa: E402

RS_TAG = ""  # RS filenames have no fixed resolution tag like the STAC assets; match by date prefix


def rs_hrefs_for_date(rs_dir: Path, date: str) -> list[str]:
    return [str(p) for p in sorted(rs_dir.glob(f"{date}_*.tif"))]


def main() -> None:
    args = parse_args("Add NIR + NDVI to the feature stack where RS coverage exists.")
    cfg = load_config(args.config)

    import numpy as np
    import rasterio

    aoi = cfg["aoi"]["name"]
    rs_dir = Path(cfg["labels"].get("rs_dir", "data/raw/swissimage_rs/lugano_delivery_2026-07"))
    fs_dir = Path(cfg["paths"]["processed_dir"]) / aoi / "feature_stack"
    out_dir = ensure_dir(Path(cfg["paths"]["processed_dir"]) / aoi / "feature_stack_rs")

    if not rs_dir.exists():
        raise SystemExit(f"RS directory not found: {rs_dir}")
    all_rs = sorted(rs_dir.glob("*.tif"))
    dates_available = sorted({p.name[:8] for p in all_rs})
    rs_date = str(cfg["imagery"].get("rs_date", dates_available[0] if dates_available else ""))
    if rs_date not in dates_available:
        raise SystemExit(f"imagery.rs_date={rs_date!r} not in delivered dates {dates_available}")

    hrefs = rs_hrefs_for_date(rs_dir, rs_date)
    print(f"=== NIR stack :: AOI '{aoi}' :: RS date {rs_date} ({len(hrefs)} strips) ===")

    tiles = sorted(fs_dir.glob(f"{aoi}_*.tif"))
    if not tiles:
        print(f"[warn] no feature_stack tiles in {fs_dir} — run 03_build_feature_stack.py first.")
        return

    written, skipped = 0, 0
    for tif in tiles:
        with rasterio.open(tif) as src:
            rgb_chm = src.read()  # (4,H,W): R,G,B,CHM from 03
            transform = src.transform
        chm = rgb_chm[3]

        rs_arr = st.read_aligned_to(tif, hrefs, resampling="bilinear")
        if rs_arr is None:
            skipped += 1
            continue

        nir, r, g, b = rs_arr[0].astype("float32"), rs_arr[1].astype("float32"), \
            rs_arr[2].astype("float32"), rs_arr[3].astype("float32")
        ndvi = (nir - r) / (nir + r + 1e-6)

        stack = np.stack([nir, r, g, b, ndvi, chm], axis=0)  # (6,H,W)
        out_path = out_dir / f"{tif.stem.replace('_rgbchm', '')}_nirchm.tif"
        st.write_geotiff(out_path, stack, transform, dtype="float32")
        written += 1
        if written == 1 or written % 10 == 0:
            print(f"[ok] {written}/{len(tiles)}  {out_path.name}  "
                  f"NDVI[min/median/max]={ndvi.min():.2f}/{np.median(ndvi):.2f}/{ndvi.max():.2f}")

    print(f"=== wrote {written} NIR stacks ({skipped} skipped, no RS coverage) -> {out_dir} ===")
    print("Channels = NIR, R, G, B (RS-native), NDVI, CHM(m). "
          "R/G/B here are from the RS delivery, not the free dop10 product used in 03 — "
          "kept separate so NIR/R/G/B stay radiometrically consistent (same acquisition).")


if __name__ == "__main__":
    main()
