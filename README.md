# ticino-palm-mapper

Detecting the invasive Chinese windmill palm (*Trachycarpus fortunei*) in southern
Ticino from high-resolution aerial orthophotos, using a self-supervised /
geo-foundation-model approach.

> **Status:** Phase 0 skeleton. See [`docs/PROJECT_PROPOSAL.md`](docs/PROJECT_PROPOSAL.md)
> for the full feasibility assessment and methods plan — read that first.

---

## TL;DR

- **Imagery:** swisstopo SWISSIMAGE 10 cm (free, RGB, EPSG:2056). Southern Ticino
  valleys get the full 10 cm/pixel → a ~2 m palm crown is ~20×20 px. Resolvable.
- **Labels:** Info Flora occurrence points (presence-only; weak positives).
- **Approach:** fine-tune a geo-FM (Phase 1) → domain-adaptive continued
  pretraining on unlabeled Ticino tiles (Phase 2) → canton-wide density map (Phase 3)
  → invasibility / spread model from multi-date detections (Phase 4, downstream).
- **The novelty:** first individual-level remote-sensing map of an invasive palm
  in temperate mixed forest.

## Repo layout

```
ticino-palm-mapper/
├── docs/
│   └── PROJECT_PROPOSAL.md     # full feasibility + methods reference (READ FIRST)
├── configs/
│   └── aoi_example.yaml        # area-of-interest + run config
├── scripts/
│   ├── 00_fetch_swissimage.py  # fetch SWISSIMAGE RGB tiles for an AOI (single vintage)
│   ├── 01_fetch_infoflora.py   # fetch occurrence points (GBIF API) + reproject
│   ├── 02_align_labels.py      # rasterize/align points onto the tile grid
│   ├── 03_build_feature_stack.py # add co-registered LiDAR CHM -> [R,G,B,CHM] stacks
│   ├── 04_fetch_temporal.py    # fetch multiple vintages (2018/2021/2024), aligned
│   ├── 05_labeling_assist.py   # HTML review sheet: crop(s) + GBIF + Street View links
│   ├── 06_prioritize_labels.py # rank points by road proximity (OSM) + NIR coverage
│   ├── 07_build_nir_stack.py   # add co-registered NIR+NDVI -> [NIR,R,G,B,NDVI,CHM]
│   └── check_swissimage_rs.py  # (utility) inspect delivered SWISSIMAGE RS tiles
├── reports/                    # slide decks / write-ups (src/ holds the build tooling)
├── src/data/swisstopo.py       # shared STAC fetch / tiling / co-registration helpers
├── src/
│   ├── data/                   # dataset, tiling, transforms
│   ├── models/                 # backbones / detection heads
│   ├── training/               # train loops (fine-tune, continued-pretrain)
│   └── inference/              # canton-wide inference → density map
├── notebooks/                  # exploration, sanity checks
├── data/
│   ├── raw/                    # downloaded imagery + raw occurrence files
│   │   └── swissimage_rs/      # 4-band NIR deliveries (gitignored, ~66 GB)
│   ├── interim/                # reprojected / tiled intermediates
│   └── processed/              # model-ready tiles + aligned labels
├── environment.yml
└── requirements.txt
```

## Quickstart (Phase 0)

> ✅ Phase 0 is **wired and runnable** end-to-end against live data:
> - `00` streams SWISSIMAGE 10 cm tiles from swisstopo's STAC COGs (no account).
> - `01` fetches *T. fortunei* occurrence points straight from the **GBIF API**
>   (default; `labels.source: gbif`). Swap to a manual Info Flora point export
>   later with `labels.source: file`.
> - `02` aligns the points onto the tile grid (point GeoJSON + raster masks).
>
> Only the optional Earth Engine route in `00` (`imagery.source: gee`) is still a
> stub — the STAC route needs no account and is the default.

```bash
# 1. environment (Phase 0 needs only the geo stack — torch/timm come later)
python3 -m venv .venv && source .venv/bin/activate
pip install rasterio rioxarray geopandas shapely pyproj fiona numpy pandas \
            xarray scikit-image scikit-learn matplotlib pyyaml tqdm requests
# (or the full conda env: conda env create -f environment.yml)

# 2. a palm-dense sanity AOI near Lugano is preset in configs/aoi_example.yaml

# 3. fetch imagery for the AOI
python scripts/00_fetch_swissimage.py --config configs/aoi_example.yaml

# 4. fetch occurrence points (GBIF) for the AOI
python scripts/01_fetch_infoflora.py --config configs/aoi_example.yaml

# 5. align labels onto the tile grid
python scripts/02_align_labels.py --config configs/aoi_example.yaml

# 6. (optional) add a co-registered LiDAR height channel -> [R,G,B,CHM] stacks
python scripts/03_build_feature_stack.py --config configs/aoi_example.yaml

# 7. (optional) fetch the 2018/2021/2024 time series (aligned per col/row)
python scripts/04_fetch_temporal.py --config configs/aoi_example.yaml

# 8. (optional) build an HTML sheet to verify points via Street View before labeling
python scripts/05_labeling_assist.py --config configs/aoi_example.yaml
```

**On inputs (from Phase 0 feasibility work):** RGB alone can't reliably separate
palms from look-alike crowns at 10 cm. The LiDAR **CHM** (script 03) and **temporal
persistence** (script 04) are the free signals that help — fuse them in the model
rather than relying on RGB. **NIR** (SWISSIMAGE RS) would likely help most but is
request/paid only. Treat occurrence points as weak positives and ground-truth your
test set with the labeling-assist sheet (script 05), not by eyeballing orthophotos.

After this, eyeball a few tiles in QGIS with the points overlaid and confirm you
can see palms with your own eyes **before** building anything heavier.

> ⚠️ **Don't export a Welten-Sutter "species list" from Info Flora** — it's a
> per-square checklist with **no coordinates** and can't place labels. You want a
> per-observation export *with* X/Y (which is what the GBIF route returns).

## CRS note

Everything Swiss is **CH1903+ / LV95 = EPSG:2056**. All scripts standardize on it.
Don't mix in WGS84 except at fetch boundaries (e.g. some APIs want lon/lat).

## Data license

swisstopo free geodata (incl. SWISSIMAGE) may be used, processed, and used
commercially **with mandatory source citation**. Cite swisstopo. Check Info Flora's
terms for the occurrence data before any redistribution.

## SSL checkpoints

`checkpoints/bellinzona/checkpoint_best.pt` (small backbone, mask-ratio 0.75, epoch
148, SSL val_loss 0.0918) is the checkpoint currently in use for the linear
probe / active-learning scoring pipeline — **not**
`checkpoints/trial_maskratio/checkpoint_best.pt` (small backbone, mask-ratio
0.5, epoch 137, SSL val_loss 0.0647), despite the latter reaching a lower SSL
reconstruction loss.

The reason: lower SSL val_loss didn't translate to better downstream
performance. Comparing the two on `src/training/linear_probe.py`'s
leave-one-tile-out CV (default `--pos-weight-multiplier`):

| checkpoint | SSL val_loss | mean per-fold accuracy | pooled accuracy | job |
|---|---|---|---|---|
| `checkpoints/bellinzona` | 0.0918 | 0.819 | 0.824 (tp=106, tn=265, fp=68, fn=11) | 11498941 |
| `checkpoints/trial_maskratio` | 0.0647 | 0.743 | 0.753 (tp=99, tn=240, fp=93, fn=18) | 11644022 |

A lower masked-reconstruction loss just means mask-ratio 0.5 is an easier
pretext task (fewer patches to reconstruct from), not that the encoder
learned better features for the actual palm/non-palm discrimination task —
worth remembering before picking a future SSL run's "winner" on val_loss
alone. `checkpoints/trial_maskratio/` is kept on disk rather than deleted, in
case it's useful for future comparison, but nothing in the active pipeline
should point at it.

## Next steps

1. Wire up the `TODO`s in `scripts/00`–`02` against the real endpoints.
2. Hand-label a seed set + an **exhaustive held-out test set** (see proposal §5).
3. Phase 1 baseline → Phase 2 SSL → Phase 3 map.
