#!/usr/bin/env python
"""
03_build_feature_stack.py
=========================
For every RGB tile produced by `00_fetch_swissimage.py`, co-register a LiDAR
canopy-height channel and write a 4-band feature stack: [R, G, B, CHM].

  CHM (canopy height model) = DSM - DTM
    DSM = ch.swisstopo.swisssurface3d-raster  (surface incl. vegetation, 0.5 m)
    DTM = ch.swisstopo.swissalti3d            (bare terrain, 0.5 m)

The 0.5 m height layers are resampled (bilinear) onto each RGB tile's exact 10 cm
grid, so the four channels stack pixel-for-pixel. Output is float32 GeoTIFF
(R,G,B in 0-255, CHM in metres) — your Dataset reads it as a (4,H,W) array and
normalises however you like.

WHY (see feasibility notes): RGB alone can't reliably separate palms from other
crowns at 10 cm; height masks lawn/ground and separates understory palms from tall
canopy. Height is a feature, not a stand-alone discriminator — fuse it in the model.

STATUS: implemented. NIR would be the natural 5th channel but SWISSIMAGE RS is not
freely served (request/paid), so it's omitted here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import swisstopo as st  # noqa: E402
from src.data.config import ensure_dir, load_config, parse_args  # noqa: E402


def main() -> None:
    args = parse_args("Build [R,G,B,CHM] feature stacks for an AOI's tiles.")
    cfg = load_config(args.config)

    import numpy as np
    import rasterio

    aoi = cfg["aoi"]["name"]
    bbox = cfg["aoi"]["bbox_lv95"]
    tiles_dir = Path(cfg["paths"]["interim_dir"]) / "tiles" / aoi
    out_dir = ensure_dir(Path(cfg["paths"]["processed_dir"]) / aoi / "feature_stack")

    tiles = sorted(tiles_dir.glob(f"{aoi}_*.tif"))
    if not tiles:
        print(f"[warn] no RGB tiles in {tiles_dir} — run 00_fetch_swissimage.py first.")
        return
    print(f"=== feature stack :: AOI '{aoi}' :: {len(tiles)} RGB tiles ===")

    # Resolve DSM/DTM COG hrefs once for the whole AOI.
    dsm_hrefs = st.asset_hrefs(st.search_stac_items(st.COLLECTION_DSM, bbox), st.TAG_DSM)
    dtm_hrefs = st.asset_hrefs(st.search_stac_items(st.COLLECTION_DTM, bbox), st.TAG_DTM)
    print(f"[lidar] DSM COGs: {len(dsm_hrefs)} | DTM COGs: {len(dtm_hrefs)}")
    if not dsm_hrefs or not dtm_hrefs:
        print("[warn] no LiDAR coverage for this AOI — cannot build CHM. "
              "swissSURFACE3D / swissALTI3D may not be released here yet.")
        return

    written, skipped = 0, 0
    for tif in tiles:
        with rasterio.open(tif) as src:
            rgb = src.read()  # (3,H,W) uint8
            transform = src.transform

        dsm = st.read_aligned_to(tif, dsm_hrefs, resampling="bilinear")
        dtm = st.read_aligned_to(tif, dtm_hrefs, resampling="bilinear")
        if dsm is None or dtm is None:
            skipped += 1
            continue

        chm = (dsm[0] - dtm[0]).astype("float32")
        chm[chm < 0] = 0.0  # clip negatives (noise / overhangs)

        stack = np.concatenate([rgb.astype("float32"), chm[None, :, :]], axis=0)  # (4,H,W)
        out_path = out_dir / f"{tif.stem}_rgbchm.tif"
        st.write_geotiff(out_path, stack, transform, dtype="float32")
        written += 1
        if written == 1 or written % 10 == 0:
            print(f"[ok] {written}/{len(tiles)}  {out_path.name}  "
                  f"CHM[min/median/max]={chm.min():.1f}/{np.median(chm):.1f}/{chm.max():.1f} m")

    print(f"=== wrote {written} feature stacks ({skipped} skipped, no LiDAR) -> {out_dir} ===")
    print("Each file is float32 (4,H,W): channels = R, G, B, CHM(m). "
          "Load with rasterio; normalise in your Dataset.")


if __name__ == "__main__":
    main()
