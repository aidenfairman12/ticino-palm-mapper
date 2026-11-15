# Design: Mapping *Trachycarpus fortunei* in Ticino from Aerial Imagery

*Feasibility assessment and methods plan for individual-level remote-sensing detection of an invasive palm in temperate mixed forest.*

> This is the project's original design document. It records the reasoning behind the
> approach taken. Some plans changed as the work progressed; for example, SWISSIMAGE RS
> (NIR) was obtained and is now a core input rather than an optional upgrade. See the
> [README](../README.md) for current status and results.

---

## 0. Summary

*Trachycarpus fortunei* is an invasive ornamental palm spreading through the thermophilic forests of southern Ticino. Existing research on it focuses on genetics and social perception; **no remote-sensing detection model exists for it**. Switzerland publishes free 10 cm aerial orthophotos (SWISSIMAGE) covering the invaded valleys, and Info Flora provides georeferenced occurrence records for ground truth. The plan is to adapt a geospatial foundation model to detect individual palms, test whether self-supervised *domain-adaptive continued pretraining* improves on it, and produce a palm density map for the region. The project is **feasible**: the palms are large enough to resolve, the imagery and labels are freely available, and the application gap is real.

---

## 1. Motivation and gap

### The species
- *Trachycarpus fortunei* ("Chinese windmill palm", locally "Tessinerpalme") was introduced as an ornamental in the 19th century and has become **aggressively invasive in southern Ticino** over recent decades, displacing native species and inhibiting forest regeneration.
- It thrives in the warm, humid Insubric climate of southern Ticino (Bellinzona, Locarno, Mendrisiotto), the same low-elevation valley terrain that gets the highest-resolution aerial imagery.
- Morphology relevant to detection: single trunk, **2–10 m tall**, crown of **fan-shaped (palmate) fronds ~60–100 cm long**, forming a distinctive radial rosette **~1.5–3 m across**. Juveniles are trunkless rosettes closer to the ground.

### The gap
- Palm detection from aerial and satellite imagery is a mature sub-field, but almost exclusively for **oil palm and date palm plantations**: regularly spaced monocultures in tropical or arid settings, with F-scores of ~90%+.
- Detecting **scattered invasive individuals against a heterogeneous native-forest background in temperate Europe** is a harder variant that has not been addressed.
- The intended contribution is the first individual-level remote-sensing map of an invasive palm in temperate mixed forest.

---

## 2. Feasibility: are the palms resolvable?

Yes. The deciding factor is imagery resolution relative to crown size.

| Factor | Value | Implication |
|---|---|---|
| SWISSIMAGE 10 cm resolution (plains/valleys) | **0.10 m / pixel** | Southern Ticino is valley terrain → full 10 cm |
| SWISSIMAGE resolution (Alps) | 0.25 m / pixel | Only matters for high-elevation tiles (outside the invasion zone) |
| Mature palm crown (~2 m) | **~20×20 px** | Ample for a CNN/ViT |
| Mature rosette (~3 m) | **~30×30 px** | Strong signal |
| Position accuracy (10 cm GSD) | ±0.15 m | Sufficient to align with point labels |

**Reference point:** the foundational oil-palm detection work (Li et al., 2016) used QuickBird imagery at **0.6 m/pixel**, six times coarser than SWISSIMAGE. The radial frond pattern is also a distinctive, partly species-specific visual signature against native broadleaf and conifer crowns.

**Aerial vs. satellite:** satellite optical imagery (Sentinel-2 at 10 m, or even commercial ~0.3–0.5 m) is not sufficient for reliable individual detection here. The project relies on **aerial orthophotos (SWISSIMAGE)**, which are airborne but published as a seamless orthomosaic, and is described as high-resolution aerial imagery rather than satellite imagery.

---

## 3. Data inventory

| Dataset | Provider | Resolution / type | Cost | Use |
|---|---|---|---|---|
| **SWISSIMAGE 10 cm** | swisstopo | 0.10 m RGB, COG GeoTIFF | **Free** (attribution) | Primary imagery. From 2017 onward also on Google Earth Engine |
| **SWISSIMAGE RS** | swisstopo | 4-band incl. **NIR** | By request / paid | NIR strongly aids vegetation discrimination |
| **swissSURFACE3D (LiDAR)** | swisstopo | National point cloud → canopy height model | Free | Height feature; palm vs. other-tree separation |
| **Info Flora occurrences** | Info Flora | Georeferenced *T. fortunei* presence points | Free | Ground-truth anchor for labels and validation |
| **WSL 2023 study** | WSL | Spread analysis of the species in Ticino | Public | Cross-check of the final map against published findings |

**Covariate layers for Phase 4 (invasibility modeling), all free and in EPSG:2056:**
Habitat Map of Switzerland / TypoCH (EnviDat), Vegetation Height Model NFI and
swissSURFACE3D (canopy structure), swissALTI3D DEM (terrain → slope/aspect/solar/TWI),
swissEO VHI (Sentinel-2 vegetation health), CHELSA bioclim (temperature/precipitation).
See §6, Phase 4.

- **CRS:** all Swiss data uses **CH1903+ / LV95 = EPSG:2056**; the project standardizes on it.
- **Tiling:** SWISSIMAGE ships in 1 km² tiles aligned to round LV95 km coordinates.
- **License:** swisstopo free geodata may be used, processed, and used commercially with mandatory source citation.

### Ground-truth caveat
Info Flora points are **presence records**, not segmentation masks or exhaustive surveys. They indicate that *a palm was seen near here*, not *every palm in this tile*, and not *the crown outline*. Consequences:
- They are good **positive anchors** and useful for **validating presence**.
- They are **not** complete labels: absence of a point does not imply absence of a palm, so unlabeled tiles cannot be treated as confirmed negatives.
- A real detection evaluation requires a **hand-labeled held-out test set** (see §5).

---

## 4. Method

### Choice of encoder
Three options were considered:

1. **Fine-tune a geospatial foundation model (baseline).** SSL pretraining quality scales with data volume and diversity, and off-the-shelf foundation models are pretrained on far more imagery than all of Ticino combined. For a niche single-sensor domain, a well-tuned pretrained model is expected to match or beat a from-scratch encoder. This is the Phase 1 baseline.

2. **Domain-adaptive continued pretraining (research question).** Continue SSL pretraining of a foundation model on unlabeled Ticino tiles, of which there is an effectively unlimited supply, before fine-tuning. This adapts generic features to the specific appearance of 10 cm Swiss orthophotos and may outperform plain fine-tuning at low label counts, at far lower cost and risk than pretraining from scratch. This is Phase 2.

3. **Pretraining from scratch (rejected).** Pretraining an encoder on a narrow single-region distribution is high-effort and likely to underperform a foundation model.

Even if Phase 2 does not beat Phase 1, a **label-efficiency curve** comparing the two is an informative result.

### Backbone candidates: match the resolution
Many geospatial foundation models (SatMAE, Prithvi, most Sentinel-pretrained SSL models) are built for **10–30 m multispectral** satellite imagery, the wrong scale for 0.1 m aerial RGB. Better matches for sub-meter imagery:

- **DINOv2:** not geospatial, but a strong general high-resolution RGB encoder that performs well on aerial imagery. Default choice.
- **Clay v1:** designed for variable resolutions and sensors.
- **DOFA:** explicitly multi-resolution and wavelength-aware.
- Any model pretrained on **aerial/UAV** rather than satellite imagery.

An early benchmark of 2–3 backbones on a small labeled slice is planned to choose empirically.

### Task formulation
| Formulation | Output | Label cost | Verdict |
|---|---|---|---|
| Tile-level presence (classification) | yes/no per grid cell → heatmap | Lowest (Info Flora points ≈ free) | Coarse; good warm-up |
| **Detection / counting (points/boxes)** | dot/box per palm → density map | Moderate | **Chosen**: management-relevant abundance, tractable labels |
| Instance segmentation (crown masks) | pixel mask per crown | Highest (polygons) | Deferred to a later version |

### Role of self-supervision
SSL **does not eliminate labels; it relocates them**. The standard recipe:
1. Pretrain or adapt the encoder on **unlabeled** SWISSIMAGE tiles (MAE, DINOv2-style, or contrastive). This needs no labels and can draw on all available Ticino imagery.
2. Fine-tune a small head on a **small labeled set**, which needs far fewer labels than training from scratch.

SSL reduces the number of **training** labels needed, not **evaluation** labels: a hand-labeled **test set** is still required to report detection metrics.

---

## 5. Labeling strategy

Goal: minimize manual annotation while keeping evaluation sound.

- **Weak positives:** Info Flora points as positive anchors.
- **Pseudo-labeling / active learning loop:** train a weak detector on a small seed set, run it over more tiles, and manually accept or reject high-confidence detections, which is much faster than labeling from scratch. Iterate.
- **Seed label budget:** on the order of **a few hundred palms across a handful of representative tiles** (dense-garden, forest-edge, and in-forest contexts).
- **Held-out test set:** a small but *exhaustively* labeled set of tiles never used in training, so that precision, recall and F1 are meaningful. This is the most important piece of manual work.
- **Tooling:** QGIS or a lightweight review UI over the COG tiles; labels exported as GeoJSON in EPSG:2056.

---

## 6. Phased plan

The phases are ordered to establish that the signal exists before investing in SSL.

### Phase 0: Data pipeline
- Define a southern-Ticino bounding box, starting small (one municipality around Bellinzona).
- Fetch SWISSIMAGE 10 cm tiles (swisstopo STAC COGs, or an Earth Engine export).
- Ingest Info Flora occurrence points; reproject to EPSG:2056.
- Build the tiling and label-alignment workflow (rasterize points onto the tile grid).
- **Deliverable:** a reproducible pipeline and a few hundred aligned image/label tiles.

### Phase 1: Baseline
- Hand-label palms on a few tiles (seed set and test set).
- Fine-tune an off-the-shelf detector / foundation-model head.
- **Deliverable:** first detection metrics on the held-out test set.

### Phase 2: Domain-adaptive SSL
- Continue pretraining the encoder on unlabeled Ticino tiles.
- Fine-tune on the same seed set; compare to Phase 1.
- **Deliverable:** a **label-efficiency curve** (accuracy vs. number of labels, plain fine-tuning vs. domain-adapted).

### Phase 3: Density map
- Run inference over the canton or a large representative region.
- Produce a **palm density map**; cross-check it against Info Flora and the WSL findings.
- **Deliverable:** a regional density map, the main management-relevant output.

### Phase 4: Invasibility / spread modeling

> **Dependency:** this phase consumes the detector's output. It is downstream of
> a working Phase 1 (ideally with multi-date detection) and is a separate model, not a
> variant of the SSL phase. It should start only once Phase 1 metrics are trustworthy.

**Question:** which areas of native vegetation are most susceptible to invasion,
i.e. predicting *invasibility* rather than current presence. This is the established field of
invasion susceptibility / habitat-suitability modeling. Existing *T. fortunei*
modeling is climate-based and range-scale (e.g. New Zealand under climate change);
**fine-scale, within-Ticino susceptibility modeling from remotely sensed detections
has not been done.**

**Static vs. temporal: the temporal model is preferred.**
- *Static SDM (weaker):* predict P(present) from covariates at one time. Simple, but it
  **conflates "suitable" with "already reached"**: a suitable cell may be palm-free
  only because no seed has arrived, so the model partly learns where palms already are.
- *Colonization / spread model (preferred):* use the **multi-date
  SWISSIMAGE archive (2017→present)**. Detect palms at t₀ and t₁, find cells that were
  palm-free at t₀ and colonized by t₁, and model **P(colonization | covariates at t₀)**.
  Predicting change conditioned on prior state is more management-relevant
  ("where next") and more defensible. It is also how invasion SDMs are
  retrospectively validated (cf. the *Vespa velutina* study, where early-stage-calibrated
  models predicted later spread well).

**Covariate stack (all free, all EPSG:2056):**

| Layer | Source | Captures |
|---|---|---|
| **Habitat Map of Switzerland (TypoCH)** | EnviDat | Vegetation co-occurrence: 84 habitat types / 32 groups at ~1 m |
| **Vegetation Height Model (NFI)** / swissSURFACE3D | EnviDat / swisstopo | Canopy height and structure, light gaps |
| **swissALTI3D DEM** | swisstopo | Slope, **aspect**, solar radiation, topographic wetness index |
| **swissEO VHI** | swisstopo | Sentinel-2 vegetation health/condition |
| **CHELSA bioclim** | CHELSA | Temperature and precipitation (winter minima matter for this species) |
| **Propagule-pressure layer** *(derived)* | Phase 1 detections | Distance to / density of existing palms (see caveat 2) |

**Methodological caveats:**

1. **Spatial autocorrelation → spatial block cross-validation.** Invaded sites
   cluster because seeds fall near parents. Random k-fold CV leaks neighbouring cells across
   train and test and inflates AUC. Block or spatial CV is required.
2. **Propagule pressure is a major confounder.** Distance to the nearest existing palm
   is usually the strongest single predictor of new colonization and can swamp or
   mimic any vegetation signal, so it must be included explicitly. The question
   of interest becomes: which vegetation predicts invasion *after* controlling for propagule
   pressure?
3. **Suitable ≠ invaded.** Absence of a palm is not evidence of unsuitability; the site may be
   dispersal-limited. Negatives must be framed carefully, and the colonization formulation handles
   this better than a static SDM.
4. **Presence-only sampling bias.** Info Flora records cluster near trails and towns. A detector
   mitigates this because it covers everywhere the imagery does, which is a further
   argument for detection-derived labels over citizen-science points for this task.

**Confounding by shared environmental gradients.** Whether native community X *causes*
higher invasibility, or X and the palm both prefer warm, low,
south-facing slopes, is a classic confounding problem. The analysis would compare a
gradient-boosted SDM baseline against explicit confounder adjustment for the shared
environmental gradient, with causal-discovery methods (PC/GES) used only as an
exploratory layer. Causal discovery on spatial observational data is difficult
(spatial confounding is a contested problem), so careful adjustment-based inference
takes priority over any claim of a definitive causal graph.

**Spatial co-occurrence statistics.** Bivariate Moran's I, join-count
statistics, and neighbourhood-enrichment tests between palm cells and native habitat classes
quantify which habitat types palms associate with.

- **Deliverable:** an **invasion-risk / susceptibility map** indicating where to focus
  eradication next, plus a comparison of predictive vs. confounder-adjusted vegetation drivers.
  This is the output most directly useful to conservation managers (WSL / cantonal forestry).

### Possible extensions
- Add NIR (SWISSIMAGE RS) and/or LiDAR canopy-height features. *(Both were subsequently adopted; see README.)*
- Upgrade detection to instance segmentation for crown delineation.
- Compare across SWISSIMAGE vintages (2017→present) to estimate **spread rate**.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scattered invasives partly occluded by taller native canopy | Accept reduced recall in closed canopy; report by habitat context; LiDAR height helps |
| Class imbalance (palms rare vs. background) | Hard-negative mining, focal loss, tile sampling around positives |
| Info Flora labels are presence-only and incomplete | Treat as weak positives; build a small exhaustively labeled test set for real metrics |
| Foundation-model resolution mismatch | Choose sub-meter/aerial-appropriate backbones (DINOv2/Clay/DOFA); verify empirically |
| Domain adaptation may not beat the baseline | Still informative as a label-efficiency analysis / negative result |
| Confusion with other palms, ornamentals, and native fan-like crowns | Include hard negatives; consider habitat priors |
| *(Phase 4)* Spatial autocorrelation inflates accuracy | Spatial block cross-validation |
| *(Phase 4)* Propagule pressure confounds vegetation signal | Include distance-to-existing-palm explicitly; ask what predicts invasion *after* adjusting for it |
| *(Phase 4)* Overclaiming causality on observational spatial data | Lead with careful adjustment/inference; treat causal discovery (PC/GES) as exploratory only |

---

## 8. Compute and tooling
- **Compute:** university HPC with GPUs under 80 GB, which is sufficient for fine-tuning and continued pretraining at this scale.
- **Main constraints:** labels and a clean evaluation set, not compute.
- **Stack:** Python, PyTorch, `rasterio`/`rioxarray` and `geopandas` for geospatial I/O, `pyproj` for CRS handling, swisstopo STAC (or Earth Engine) for data access, QGIS for labeling, and `timm` for backbones.

---

## 9. References
- swisstopo SWISSIMAGE 10 cm: product page and Earth Engine catalog entry (resolution, license, COG format).
- Info Flora: *T. fortunei* occurrence portal (ground-truth points).
- WSL (2023): study on *T. fortunei* spread in Ticino forests (management context, map cross-check).
- Li et al. (2016): first CNN oil-palm detection from QuickBird (0.6 m), used as the resolution benchmark.
- Culman, Delalieux & Van Tricht (2020), *Remote Sensing* 12:3476: individual palm detection on RGB for tree inventory (closest methodological analogue).
- Foundation-model candidates: DINOv2, Clay v1, DOFA (sub-meter-appropriate encoders).

**Phase 4 (invasibility modeling):**
- Conedera / Walther / Jousson et al. (2022): causes of *T. fortunei* spread in southern Ticino (disturbance, winter temperature, landscape use; dispersal to altitude).
- *T. fortunei* climate-suitability SDM (New Zealand, climate change): range-scale prior work.
- Asian hornet (*Vespa velutina*) SDM validation study: retrospective temporal validation of invasion SDMs.
- Pazúr et al. (2021) / Habitat Map of Switzerland (TypoCH), *Remote Sensing* 15:643: vegetation co-occurrence covariate.
- "Can SDMs predict invasive expansion?": equilibrium-assumption pitfalls in invasion SDMs.

*This list is a starting point for the literature review, not a complete bibliography.*
