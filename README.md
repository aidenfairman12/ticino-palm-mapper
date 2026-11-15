# ticino-palm-mapper

Detecting the invasive Chinese windmill palm (*Trachycarpus fortunei*) in southern
Ticino, Switzerland, from 10 cm aerial orthophotos, near-infrared imagery and LiDAR
canopy height.

The palm is spreading through the region's low-elevation forests, and no
remote-sensing detection model exists for it. Unlike plantation palm detection, the
task here is to find scattered individuals against a heterogeneous native-forest
background, with only presence-only citizen-science records as starting labels.
The original feasibility assessment and methods design are in
[`docs/DESIGN.md`](docs/DESIGN.md).

---

## Status and results

The project currently covers a ~6,000-tile area of interest around Bellinzona. Two
approaches have been built and compared under **leave-one-tile-out cross-validation**,
which keeps every image of a location in the same fold so that nearby crops cannot
leak between train and test.

| Approach | Input | Pooled LOTO accuracy |
|---|---|---|
| Frozen DINOv2 ViT-S + linear probe | 6-channel crop (NIR, R, G, B, NDVI, CHM) | 81.6% → 82.9% as labels grew |
| Same, after continued MAE pretraining on Ticino tiles | same | 72.4% |
| **Classical XGBoost on physical features** | per-point NDVI / CHM / spectral neighbourhood stats + cross-date stability | **96.7%** |

Latest classical run, restricted to the three native-10 cm acquisition dates
(81 positive / 526 negative examples, 147 folds): **96.9% pooled accuracy, 0.84
recall, 0.92 precision** at the point level.

### Findings
- **Continued self-supervised pretraining hurt.** Continuing MAE pretraining of DINOv2
  on unlabeled Ticino imagery reduced linear-probe accuracy by ~9 points against the
  off-the-shelf encoder. The gap appeared in two independent comparisons, before
  and after a channel-alignment bug fix. A lower reconstruction loss also did not
  predict better downstream features: a mask-ratio-0.5 run reached lower SSL loss
  (0.065 vs 0.092) but probed worse (75.3% vs 82.4%). MAE optimizes pixel
  reconstruction, not linear separability.
- **Physical features beat learned embeddings at this label scale.** The palm has a
  distinctive physical signature: evergreen (stable NDVI between leaf-off and leaf-on
  flights), a characteristic height, and a compact crown. Hand-built features capture
  it directly, train in seconds on a CPU, and are interpretable.
- **Point classification does not yet transfer to wall-to-wall detection.** When
  the classical model is run per pixel over whole validation tiles
  ([`src/inference/dense_classical.py`](src/inference/dense_classical.py)), it
  recovers 56–60% of known palms but with only ~2–4% peak precision. It has largely
  learned "tall evergreen vegetation". Dense scoring showed the cause: random
  negatives are rarely other trees. Most of the recent work targets this problem:
  harvesting deciduous negatives from the March NDVI drop, collecting hand-verified
  vegetation negatives through Google Earth review, and moving to native-10 cm dates
  only.
- **Data quality matters as much as the model.** Three of the six delivered
  NIR acquisition dates turned out to be 25 cm imagery upsampled to 10 cm (~50× blurrier
  by Laplacian variance; [`scripts/20`](scripts/20_check_acquisition_sharpness.py),
  [`23`](scripts/23_check_raw_sharpness.py)). A channel-order bug in the 6-channel
  patch-embedding adaptation was found and fixed (commit `fcad6ce`), and every
  checkpoint was regenerated afterwards.

### Labels
Starting labels were GBIF / Info Flora presence points, which are incomplete and
sometimes mislocated. All labels used for training and evaluation were verified by hand:

- **~175 confirmed palm locations**, from Street View–assisted verification of
  occurrence points ([`05`](scripts/05_labeling_assist.py)), active-learning review of
  model-scored candidates ([`10`](scripts/10_review_candidates.py) /
  [`11`](scripts/11_merge_review_verdicts.py)), and Google Earth review
  ([`24`](scripts/24_export_10cm_footprint_kml.py) / [`25`](scripts/25_import_earth_review_batch.py))
- **~320 hand-verified hard negatives** (204 from active learning, 115 from Google Earth review)
- 13,746 automatically harvested deciduous negatives ([`19`](scripts/19_harvest_deciduous_negatives.py)),
  kept for ablations

---

## Pipeline

All scripts take explicit arguments (`--help`), and the data scripts read an AOI config from
[`configs/`](configs/). Everything is in **EPSG:2056 (CH1903+ / LV95)**.

| Stage | Scripts / modules |
|---|---|
| **Imagery and features** | `00` fetch SWISSIMAGE RGB tiles (swisstopo STAC) · `03` add LiDAR canopy height (DSM − DTM) · `04` multi-year RGB stack · `07` fuse SWISSIMAGE RS → `[NIR, R, G, B, NDVI, CHM]` |
| **Occurrence labels** | `01` fetch GBIF / Info Flora points · `02` align points to tiles · `05` Street View verification sheet · `06` prioritize points by verification cost · `08`, `09` KML overlays for Google Earth |
| **Deep pipeline** | [`src/models/mae.py`](src/models/mae.py) MAE on a DINOv2 backbone · [`src/training/pretrain_ssl.py`](src/training/pretrain_ssl.py) continued pretraining · `13` export plain-DINOv2 baseline · [`src/training/linear_probe.py`](src/training/linear_probe.py) LOTO linear probe · [`src/inference/score_candidates.py`](src/inference/score_candidates.py) grid scoring |
| **Classical pipeline** | `15` feature extraction · [`src/training/classical_probe.py`](src/training/classical_probe.py) LOTO RF/XGBoost · `16` hyperparameter sweep · `27` feature-subset ablation · [`src/inference/score_candidates_classical.py`](src/inference/score_candidates_classical.py) grid scoring · [`src/inference/dense_classical.py`](src/inference/dense_classical.py) per-pixel scoring |
| **Active learning and review** | `10` HTML review page for scored candidates · `11` merge verdicts into labels · `12` score summary · `19` harvest deciduous negatives · `24`, `25` Google Earth review round-trip |
| **Diagnostics** | `14` CV failure cases · `17` dense-surface peak readout · `18` view comparison and NDVI separability · `20`–`23` acquisition sharpness and coverage · `26` validation-tile selection · `check_swissimage_rs.py` inspect RS deliveries |

[`slurm/`](slurm/) holds the batch jobs used to run each experiment on the university
HPC. Each job's header comment records its hypothesis and configuration.

---

## Reproducing

```bash
# Environment (conda-forge handles the GDAL/PROJ stack)
conda env create -f environment.yml && conda activate ticino-palm
# or build the container used on the HPC:
apptainer build ticino-palm.sif apptainer.def

# 1. Data: small example AOI near Bellinzona (configs/aoi_example.yaml)
python scripts/00_fetch_swissimage.py   --config configs/aoi_example.yaml
python scripts/01_fetch_infoflora.py    --config configs/aoi_example.yaml
python scripts/02_align_labels.py       --config configs/aoi_example.yaml
python scripts/03_build_feature_stack.py --config configs/aoi_example.yaml
python scripts/07_build_nir_stack.py    --config configs/aoi_example.yaml   # needs SWISSIMAGE RS

# 2. Classical pipeline (CPU, minutes)
python scripts/15_extract_classical_features.py --tile-dirs <feature_stack_rs dirs> \
    --confirmed-points data/interim/labels/bellinzona_MASTER_confirmed_palms.geojson \
    --hard-negatives data/interim/labels/*_hard_negatives.geojson \
    --output classical_features/features.csv
python -m src.training.classical_probe --features-csv classical_features/features.csv --model xgboost
```

Steps `00`–`04` use only free swisstopo data and need no account. The NIR channel
(`07`) requires the SWISSIMAGE RS product, which swisstopo provides on request; it is
not redistributed here. [`slurm/native_10cm_baseline.sbatch`](slurm/native_10cm_baseline.sbatch)
shows the full classical run (extraction → CV → dense scoring → readout).

---

## Repository layout

```
configs/            AOI and run configs
data/interim/labels hand-verified palm points and hard negatives (tracked)
data/raw|processed  imagery and feature stacks (not tracked; large/licensed)
docs/DESIGN.md      feasibility assessment and methods design
notebooks/          MAE reconstruction inspection
review/             generated HTML review pages from active-learning rounds
candidate_scores/   small scored-candidate outputs
scripts/            numbered pipeline, labeling and diagnostic scripts
slurm/              HPC batch jobs, one per experiment
src/
  data/             tiling, normalization, datasets, swisstopo STAC helpers
  models/           MAE with channel-adapted DINOv2 backbone
  training/         SSL pretraining, linear probe, classical probe
  inference/        grid and dense scoring
```

---

## Limitations and next steps
- **Dense precision** is the main open problem (see Findings). Next steps: more
  hand-verified vegetation negatives from inside the native-10 cm footprint, crown-shape
  features computed at a scale the 0.5 m LiDAR can actually support, and a dense
  detection head trained on point-derived density targets.
- **Evaluation** is cross-validated over a few hundred labeled points. An exhaustively
  labeled held-out test area is needed before reporting detection metrics.
- **Scale:** the work so far covers the Bellinzona AOI. Canton-wide mapping and the
  spread/invasibility modeling in [`docs/DESIGN.md`](docs/DESIGN.md) §6 are future work.

## Data and licensing
- swisstopo geodata (SWISSIMAGE, swissSURFACE3D, swissALTI3D) may be used, processed and
  used commercially with mandatory source citation: **© swisstopo**.
- Occurrence records come from GBIF (mirroring Info Flora). Check Info Flora's terms
  before redistributing raw occurrence data.
- SWISSIMAGE RS (NIR) was supplied by swisstopo on request and is not included in this
  repository.
