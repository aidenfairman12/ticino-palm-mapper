#!/usr/bin/env python
"""Fuse the paid SWISSIMAGE RS delivery's NIR/R/G/B into the [R,G,B,CHM] stack
from 03_build_feature_stack.py, producing [NIR,R,G,B,NDVI,CHM] (6 bands,
float32) per tile wherever RS coverage exists.

R/G/B here come from the RS delivery itself, not the free dop10 product used
in 03: NIR/R/G/B need to be one radiometrically consistent acquisition, or
NDVI mixes real ground change with cross-flight sensor differences. CHM is
carried over unchanged from 03's output, since LiDAR height is stable
year-to-year. Tiles with no RS coverage are skipped and reported, not padded
with fake data — the gap is itself information.

imagery.rs_date defaults to the leaf-off acquisition (the reason NIR was
requested in the first place); override per-run with --rs-date to build a
second, differently-dated stack without touching the config — each date
writes to its own feature_stack_rs_<date>/ dir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import swisstopo as st  # noqa: E402
from src.data.config import ensure_dir, load_config  # noqa: E402

RS_TAG = ""  # RS filenames have no fixed resolution tag like the STAC assets; match by date prefix


def rs_hrefs_for_date(rs_dir: Path, date: str) -> list[str]:
    return [str(p) for p in sorted(rs_dir.glob(f"{date}_*.tif"))]


def parse_nir_stack_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add NIR + NDVI to the feature stack where RS coverage exists."
    )
    p.add_argument("--config", required=True, help="Path to AOI/run YAML config")
    p.add_argument(
        "--rs-date", default=None,
        help="Override imagery.rs_date from the config — lets you build multiple "
             "dated stacks (e.g. leaf-off + a recent date) from one config without "
             "duplicating it. Output goes to a date-specific directory either way.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_nir_stack_args()
    cfg = load_config(args.config)

    import numpy as np
    import rasterio

    aoi = cfg["aoi"]["name"]
    rs_dir = Path(cfg["labels"].get("rs_dir", "data/raw/swissimage_rs/bellinzona_delivery_2026-07"))
    fs_dir = Path(cfg["paths"]["processed_dir"]) / aoi / "feature_stack"

    if not rs_dir.exists():
        raise SystemExit(f"RS directory not found: {rs_dir}")
    all_rs = sorted(rs_dir.glob("*.tif"))
    dates_available = sorted({p.name[:8] for p in all_rs})
    rs_date = args.rs_date or str(cfg["imagery"].get("rs_date", dates_available[0] if dates_available else ""))
    if rs_date not in dates_available:
        raise SystemExit(f"rs_date={rs_date!r} not in delivered dates {dates_available}")

    # Date-specific output dir — running this script twice with different dates
    # (e.g. leaf-off for NDVI contrast + a recent date for temporal alignment
    # with newer labels) must not silently overwrite the other run's output.
    out_dir = ensure_dir(
        Path(cfg["paths"]["processed_dir"]) / aoi / f"feature_stack_rs_{rs_date}"
    )

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
