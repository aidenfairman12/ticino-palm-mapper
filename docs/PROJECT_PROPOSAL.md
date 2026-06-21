# Mapping the Chinese Windmill Palm (*Trachycarpus fortunei*) in Ticino from Aerial Imagery

**A self-supervised remote-sensing project — feasibility assessment & methods plan**

*Working title: an individual-level remote-sensing map of an invasive palm in temperate forest.*

---

## 0. One-paragraph summary

*Trachycarpus fortunei* is an invasive ornamental palm spreading through the thermophilic forests of southern Ticino. Existing research on it is genetic and social-perception based; **no remote-sensing detection model exists for it**. Switzerland publishes free 10 cm aerial orthophotos (SWISSIMAGE) covering exactly the invaded valleys, and Info Flora provides georeferenced occurrence records for ground truth. The plan is to fine-tune a geospatial foundation model to detect individual palms, with a self-supervised *domain-adaptive continued-pretraining* step as the research contribution, and to produce a canton-wide palm density map as the headline deliverable. The project is **feasible**: the palms are large enough to resolve, the imagery and labels exist and are free, and the application gap is real.

---

## 1. The problem & why it's a good project

### The species
- *Trachycarpus fortunei* ("Chinese windmill palm", locally "Tessinerpalme") was introduced as an ornamental in the 19th century and has become **aggressively invasive in southern Ticino** over recent decades, displacing native species and inhibiting forest regeneration.
- It thrives in the warm, humid Insubric climate of southern Ticino (Lugano, Locarno, Mendrisiotto) — the same low-elevation valley terrain that gets the highest-resolution aerial imagery.
- Morphology relevant to detection: single trunk, **2–10 m tall**, crown of **fan-shaped (palmate) fronds ~60–100 cm long**, forming a distinctive radial "pinwheel/star" rosette **~1.5–3 m across**. Juveniles are trunkless rosettes closer to the ground.

### Why it's portfolio-worthy *and* paper-able
- **The method is established; the application is not.** Palm detection from aerial/satellite imagery is a mature sub-field — but almost exclusively on **oil palm and date palm plantations**: regularly spaced monocultures in tropical/arid settings, F-scores ~90%+. Detecting **scattered invasive individuals against a heterogeneous native-forest background in temperate Europe** is a genuinely harder and unsolved variant.
- **The novelty is defensible in one sentence:** *"First individual-level remote-sensing map of an invasive palm in temperate mixed forest."*
- **It hits target ML competencies:** self-supervised learning, geospatial foundation models, label-efficient learning, a real end-to-end data pipeline, and a tangible map artifact.

---

## 2. Feasibility — can we actually see them?

**Yes, comfortably.** The decisive number is imagery resolution vs. crown size.

| Factor | Value | Implication |
|---|---|---|
| SWISSIMAGE 10 cm resolution (plains/valleys) | **0.10 m / pixel** | Southern Ticino is valley terrain → full 10 cm |
| SWISSIMAGE resolution (Alps) | 0.25 m / pixel | Only matters for high-elevation tiles (not invasion zone) |
| Mature palm crown (~2 m) | **~20×20 px** | Plenty for a CNN/ViT |
| Mature rosette (~3 m) | **~30×30 px** | Strong signal |
| Position accuracy (10 cm GSD) | ±0.15 m | Good enough to align with point labels |

**Reference point from the literature:** the foundational oil-palm detection work (Li et al., 2016) detected palms from QuickBird at **0.6 m/pixel**. We have **6× finer** resolution. The distinctive radial frond pattern is also a strong, somewhat species-specific visual signature against native broadleaf/conifer crowns.

**Note on "satellite" vs "aerial":** true satellite optical (Sentinel-2 at 10 m, even commercial ~0.3–0.5 m) is *not* sufficient for reliable individual detection here. The project rests on **aerial orthophotos (SWISSIMAGE)**, which are airborne but published as a seamless ortho mosaic. Framing-wise, call it "high-resolution aerial/orthophoto imagery," not "satellite," in the writeup.

---

## 3. Data inventory

| Dataset | Provider | Resolution / type | Cost | Use |
|---|---|---|---|---|
| **SWISSIMAGE 10 cm** | swisstopo | 0.10 m RGB, COG GeoTIFF | **Free** (attribution) | Primary imagery. From 2017+, also on Google Earth Engine |
| **SWISSIMAGE RS** | swisstopo | 4-band incl. **NIR** | By request / paid | NIR strongly aids vegetation discrimination — optional upgrade |
| **swissSURFACE3D (LiDAR)** | swisstopo | National point cloud → canopy height model | Free | Height feature; palm-vs-other-tree separation |
| **Info Flora occurrences** | Info Flora | Georeferenced *T. fortunei* presence points | Free | Ground-truth anchor for labels & validation |
| **WSL 2023 study** | WSL | Spread analysis of the species in Ticino | Public | Sanity-check the final map against published findings |

**Covariate layers for Phase 4 (invasibility modeling) — all free, EPSG:2056:**
Habitat Map of Switzerland / TypoCH (EnviDat), Vegetation Height Model NFI &
swissSURFACE3D (canopy structure), swissALTI3D DEM (terrain → slope/aspect/solar/TWI),
swissEO VHI (Sentinel-2 vegetation health), CHELSA bioclim (temperature/precipitation).
See §6 Phase 4 for how these are used.

- **CRS:** everything Swiss is **CH1903+ / LV95 = EPSG:2056**. Standardize on this.
- **Tiling:** SWISSIMAGE ships in 1 km² tiles aligned to round LV95 km coordinates.
- **License:** swisstopo free geodata may be used, processed, and used commercially with mandatory source citation.

### Ground-truth caveat (important)
Info Flora points are **presence records**, not segmentation masks or exhaustive surveys. They tell you *a palm was seen near here*, not *every palm in this tile* and not *the crown outline*. Consequences:
- They are excellent **positive anchors** and for **validation of presence**.
- They are **not** complete labels — absence of a point ≠ absence of a palm. So you cannot treat unlabeled tiles as confirmed negatives without care.
- For a real detection eval you will need a **hand-labeled held-out test set** (see §5).

---

## 4. Method — the central decision and the recommendation

### The encoder fork
Three options, in order of recommendation:

1. **✅ Default / best accuracy-per-effort — fine-tune a geospatial foundation model.** SSL pretraining quality scales with data volume and diversity. Off-the-shelf geo-FMs are pretrained on far more imagery than all of Ticino combined. For a niche single-sensor domain, a well-fine-tuned geo-FM will **match or beat** a from-scratch encoder. This is the Phase-1 baseline.

2. **✅ Research contribution — domain-adaptive continued pretraining.** Take a geo-FM and **continue SSL pretraining on unlabeled Ticino tiles** (of which there is effectively unlimited supply), *then* fine-tune. This adapts generic features to the specific look of 10 cm Swiss orthophotos. It can beat plain fine-tuning at low label counts, it's a legitimate novel result, and it's far cheaper/lower-risk than pretraining from zero. **This is the "trained my own encoder" flavor without the downside.** This is Phase 2.

3. **❌ From scratch — not recommended.** Pretraining an encoder from zero on a narrow single-region distribution is high-effort, likely *worse*, and a weak story ("reinvented a smaller wheel").

> **Bottom line:** start with geo-FM fine-tuning; make domain-adaptive continued pretraining the paper's novelty; skip from-scratch. Even if Phase 2 fails to beat Phase 1, a clean **label-efficiency curve** is itself a publishable result.

### Encoder/backbone candidates — match the resolution!
Many geo-FMs (SatMAE, Prithvi, most Sentinel-pretrained SSL) are built for **10–30 m multispectral** satellite imagery — the wrong scale for 0.1 m RGB aerial. Better matches for sub-meter RGB:

- **DINOv2** — not geospatial, but an excellent general high-res RGB encoder; strong on aerial in practice. Strong default.
- **Clay v1** — designed for variable resolution/sensors.
- **DOFA** — explicitly multi-resolution, wavelength-aware.
- Any model pretrained on **aerial/UAV** rather than satellite.

**Cheap early experiment:** benchmark 2–3 backbones on a small labeled slice. "I compared encoder backbones empirically" is itself a good portfolio signal.

### Task formulation — what the model outputs
| Formulation | Output | Label cost | Verdict |
|---|---|---|---|
| Tile-level presence (classification) | yes/no per grid cell → heatmap | Lowest (Info Flora points ≈ free) | Coarse; good warm-up |
| **Detection / counting (points/boxes)** | dot/box per palm → density map | Moderate | **Recommended** — management-relevant abundance, tractable labels |
| Instance segmentation (crown masks) | pixel mask per crown | Highest (polygons) | Best visual / strongest CV signal; defer to v2 |

**Recommendation:** detection/counting as the core; presence-mapping as a quick warm-up; segmentation as an optional v2 upgrade.

### On self-supervision (your stated preference)
SSL **does not eliminate labels — it relocates them.** Standard recipe:
1. Pretrain/adapt the encoder on **unlabeled** SWISSIMAGE tiles (MAE / DINOv2-style / contrastive). Zero labels, unlimited Ticino imagery — a perfect fit for your situation.
2. Fine-tune a small detection head on a **small labeled set** — far fewer labels than from-scratch.

SSL saves you on **training** labels, **not evaluation** labels. You still need a hand-labeled **test set** to report any detection metric honestly.

---

## 5. Labeling strategy (minimizing hand-work)

Goal: as little manual annotation as possible while keeping the eval honest.

- **Weak positives, free:** Info Flora points → positive anchors.
- **Pseudo-labeling loop:** train a weak detector on a small seed set, run it over more tiles, manually accept/reject high-confidence detections (much faster than labeling from scratch). Iterate.
- **Seed label budget:** plan to hand-mark on the order of a **few hundred palms across a handful of representative tiles** (mix of dense-garden, forest-edge, and in-forest contexts).
- **Non-negotiable:** a **held-out, fully-labeled test set** on tiles never used in training — small but *exhaustively* labeled, so precision/recall/F1 are real. This is the single most important piece of manual work.
- **Tooling:** QGIS for point/box labeling over the COG tiles, or a lightweight labeling UI; export to GeoJSON in EPSG:2056.

---

## 6. Phased plan

> Designed to de-risk early (prove signal before investing in SSL) and to have a paper-able endpoint.

### Phase 0 — Data pipeline *(start here; mostly engineering)*
- Define a southern-Ticino bounding box (start small — e.g. one municipality around Lugano/Mendrisio).
- Fetch SWISSIMAGE 10 cm tiles (Earth Engine export **or** swisstopo COG download).
- Ingest Info Flora occurrence points; reproject to EPSG:2056.
- Build tiling + label-alignment workflow (rasterize points onto tile grid).
- **Deliverable:** reproducible pipeline + a few hundred aligned image/label tiles. Solid demonstrable engineering on its own.

### Phase 1 — Baseline *(prove the signal exists)*
- Hand-label palms on a few tiles (seed set + test set).
- Fine-tune an off-the-shelf detector / geo-FM head.
- **Deliverable:** first detection metrics on the held-out test set.

### Phase 2 — SSL contribution *(the novelty)*
- Continue-pretrain the encoder on unlabeled Ticino tiles (domain adaptation).
- Fine-tune on the same seed set; compare to Phase 1.
- **Deliverable:** a **label-efficiency curve** — accuracy vs. number of labels, plain fine-tune vs. domain-adapted. The figure that sells the paper.

### Phase 3 — The map *(headline visual + management relevance)*
- Run inference over the canton (or a large representative region).
- Produce a **palm density map**; sanity-check against Info Flora and the WSL findings.
- **Deliverable:** canton-wide density map — the management-relevant artifact and paper centerpiece.

### Phase 4 — Invasibility / spread modeling *(turns a detector into a decision tool)*

> **Dependency:** this phase *consumes the detector's output* — it is downstream of
> a working Phase 1 (and ideally multi-date detection). It is a second model, not a
> variant of the SSL phase. Don't start it until Phase 1 metrics are trustworthy.

**The question:** which areas of native flora are most prone to being overtaken —
i.e. predict *invasibility*, not current presence. This is the established field of
invasion susceptibility / habitat-suitability modeling. Existing *T. fortunei*
modeling is climate-based and range-scale (e.g. New Zealand under climate change);
**fine-scale, within-Ticino, vegetation-co-occurrence susceptibility from
remote-sensed detections is the open gap.**

**Static vs. temporal — build the temporal one.**
- *Static SDM (weaker):* predict P(present) from covariates at one time. Easy, but
  **conflates "suitable" with "already reached"** — a suitable cell may be palm-free
  only because no seed has arrived. The model partly learns where palms already are.
- *Colonization / spread model (stronger, recommended):* use the **multi-date
  SWISSIMAGE archive (2017→present)**. Detect palms at t₀ and t₁; find cells that were
  palm-free at t₀ and colonized by t₁; model **P(colonization | covariates at t₀)**.
  Predicting a *change* conditioned on prior state is more management-relevant
  ("where next") and far more defensible. This is also how invasion SDMs are
  retrospectively validated (cf. the *Vespa velutina* study: early-stage-calibrated
  models predicted later spread well).

**Covariate stack (all free, all EPSG:2056) — Switzerland is unusually rich here:**

| Layer | Source | Captures |
|---|---|---|
| **Habitat Map of Switzerland (TypoCH)** | EnviDat | The "co-occurrence with other vegetation" predictor — 84 habitat types / 32 groups at ~1 m. *This is the layer you asked for, served on a plate.* |
| **Vegetation Height Model (NFI)** / swissSURFACE3D | EnviDat / swisstopo | Canopy height & structure, light gaps |
| **swissALTI3D DEM** | swisstopo | → slope, **aspect**, solar radiation, topographic wetness index |
| **swissEO VHI** | swisstopo | Sentinel-2 vegetation health/condition |
| **CHELSA bioclim** | CHELSA | Temperature / precipitation (winter minima matter for this species) |
| **Propagule-pressure layer** *(derived)* | from Phase 1 detections | Distance to / density of existing palms — **see caveat below** |

**Critical methodological caveats (these separate a real result from a fake one):**

1. **Spatial autocorrelation → use spatial block cross-validation.** Invaded sites
   cluster (seeds fall near parents). Random k-fold CV leaks neighbouring cells across
   train/test and yields a beautiful, *fake* AUC. Block/spatial CV is non-negotiable
   and is the first thing a reviewer will check.
2. **Propagule pressure confounds everything.** Distance to the nearest existing palm
   is usually the strongest single predictor of new colonization and can swamp or
   *mimic* any vegetation signal. **Include it explicitly.** The interesting question
   then becomes: what vegetation predicts invasion *after* controlling for propagule
   pressure?
3. **Suitable ≠ invaded.** Absence of a palm is not evidence of unsuitability (could be
   dispersal-limited). Frame negatives carefully; the colonization formulation handles
   this better than static SDM.
4. **Presence-only sampling bias** (Info Flora clusters near trails/towns). The detector
   mitigates this — it sees everywhere the imagery covers — which is itself a good
   argument for why detection-derived labels beat citizen-science points for this task.

**The causal hook (ties this to the wider portfolio).** "Does native community X
*cause* higher invasibility, or do X and the palm both just prefer warm, low,
south-facing slopes?" is a textbook confounding problem — the *same* structure as the
Tennessee Eastman causal-discovery project (predictive baselines misattribute root
cause; causal methods disentangle it). Run the same comparison here: a
gradient-boosted SDM baseline vs. a causal-adjustment layer (explicit confounder
adjustment for the shared environmental gradient; PC/GES as an *exploratory* discovery
layer). **Be honest:** causal discovery on spatial observational data is hard
(spatial confounding is nasty and contested), so lead with careful inference +
adjustment and treat discovery as exploratory rather than claiming a definitive graph.

**Spatial co-occurrence stats reuse prior work.** Bivariate Moran's I, join-count
statistics, and neighbourhood-enrichment between palm cells and native habitat classes
are *methodologically identical* to spatial neighbourhood-enrichment analysis in
spatial transcriptomics (Squidpy/CosMx) — swap cell types for habitat types. Same math,
different domain.

- **Deliverable:** an **invasion-risk / susceptibility map** ("where to focus
  eradication next"), plus a predictive-vs-causal comparison of vegetation drivers.
  This is the artifact a conservation manager (WSL / cantonal forestry) would actually
  use, and a clean second paper or a strong extension of the first.

### Optional v2
- Add NIR (SWISSIMAGE RS) and/or LiDAR canopy-height features.
- Upgrade detection → instance segmentation for crown delineation.
- Temporal: compare across SWISSIMAGE vintages (2017→present) to estimate **spread rate**.

---

## 7. Risks & honest difficulties

| Risk | Mitigation |
|---|---|
| Scattered invasives partly occluded by taller native canopy | Accept reduced recall in closed canopy; report by habitat context; LiDAR height helps |
| Class imbalance (palms rare vs. background) | Hard-negative mining, focal loss, tile sampling around positives |
| Info Flora ≠ complete labels (presence-only) | Treat as weak positives; build a small exhaustively-labeled test set for real metrics |
| Geo-FM resolution mismatch | Pick sub-meter/aerial-appropriate backbones (DINOv2/Clay/DOFA); verify empirically |
| Domain adaptation may not beat baseline | Still publishable as label-efficiency analysis / negative result |
| Confusion with other palms/ornamentals & native fan-like crowns | Include hard negatives; consider habitat priors |
| *(Phase 4)* Spatial autocorrelation inflates accuracy | Spatial block cross-validation — non-negotiable |
| *(Phase 4)* Propagule pressure confounds vegetation signal | Include distance-to-existing-palm explicitly; ask what predicts invasion *after* adjusting for it |
| *(Phase 4)* Overclaiming causality on observational spatial data | Lead with careful adjustment/inference; treat causal discovery (PC/GES) as exploratory only |

---

## 8. Compute & logistics
- **Compute:** university HPC, sub-80 GB GPUs — **more than sufficient**. Not a constraint. Fine-tuning and continued pretraining at this scale fit comfortably. Don't over-optimize for memory.
- **Real constraints are labels and a clean eval set**, not compute.
- **Stack (suggested):** Python, PyTorch, `rasterio`/`rioxarray` + `geopandas` for geo I/O, `pyproj` for CRS, an Earth Engine or COG fetch path, QGIS for labeling, `timm`/HuggingFace for backbones, `lightning` (optional) for training loops.

---

## 9. First concrete step (when you return to this)
Run **Phase 0**: the data pipeline in this repo. Start with a *small* bounding box (one municipality), get ~tens of tiles + aligned Info Flora points rendered correctly in EPSG:2056, and eyeball a few palms in the imagery to confirm the signal with your own eyes before building anything heavier.

See `README.md` and `scripts/` for the runnable skeleton.

---

## 10. Key references / sources to revisit
- swisstopo SWISSIMAGE 10 cm — product page & Earth Engine catalog entry (resolution, license, COG format).
- Info Flora — *T. fortunei* occurrence portal (ground-truth points).
- WSL (2023) — study on *T. fortunei* spread in Ticino forests (management context, map sanity check).
- Li et al. (2016) — first CNN oil-palm detection from QuickBird (0.6 m) — resolution benchmark.
- Culman, Delalieux & Van Tricht (2020), *Remote Sensing* 12:3476 — individual palm detection on RGB for tree inventory (closest methodological analogue).
- Geo-FM candidates: DINOv2, Clay v1, DOFA (sub-meter-appropriate encoders).

**Phase 4 (invasibility modeling):**
- Conedera / Walther / Jousson et al. (2022) — causes of *T. fortunei* spread in southern Ticino (disturbance, winter temperature, landscape use; dispersal to altitude) — covariate shortlist + causal grounding.
- *T. fortunei* climate-suitability SDM (New Zealand, climate change) — range-scale prior art to contrast against fine-scale work.
- Asian hornet (*Vespa velutina*) SDM validation study — retrospective temporal validation of invasion SDMs; supports the colonization-model approach.
- Pazúr et al. (2021) / Habitat Map of Switzerland (TypoCH), *Remote Sensing* 15:643 — the vegetation co-occurrence covariate.
- General caution: "Can SDMs predict invasive expansion?" — equilibrium-assumption pitfalls in invasion SDM.

*(Strengthen this list with proper citations during the literature review — these are anchors, not a final bibliography.)*
