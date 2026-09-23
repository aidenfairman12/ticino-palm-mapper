#!/bin/bash
#SBATCH --job-name=palm-earth-review-pipeline
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/earth_review_pipeline/palm-earth-review-pipeline-%j.out
#SBATCH --error=logs/earth_review_pipeline/palm-earth-review-pipeline-%j.err

# Same chain as harvest_pipeline_overnight.sbatch, re-run with the two Google
# Earth visual-review batches added: 115 hand-picked vegetation negatives
# from residential/garden terrain (the kind of ground the March-only
# deciduous harvest structurally could never reach — see
# checkpoint-2026-09-18-labeling-ceiling.md's 2026-09-20 update, a decisive
# null result that root-caused the harvest's failure to March's coverage
# gap), plus 49 new confirmed palms. This is the actual test of the original
# hypothesis in dense-scoring-reveals-vegetation-detector.md: the model never
# learned palm-vs-tree because its negatives contained almost no other real
# trees. The harvest alone didn't fix it because it drew negatives from the
# wrong part of the map; these do not have that problem.
#
# bellinzona_MASTER_confirmed_palms.geojson already has the two new Earth-review
# confirmed-palm batches merged in (39 -> 88), same as every prior batch
# (2/3/4) was merged before it — scripts/15 only takes one --confirmed-points
# file. The two Earth-review hard-negative files are passed alongside the
# existing ones via --hard-negatives, which already takes a list.
#
# Same 3 validation tiles as every prior comparison (c010_r060, c057_r040,
# c061_r038) so this reads directly against the established numbers:
#   pre-harvest:  15-21% saturation, best precision 0.007-0.026, recall ~92-93%
#   post-harvest: 16.8-24.3% (worse on one tile), best precision 0.014-0.041, recall 84.0%
#
# Worth setting expectations before reading the [4/4] output: the negative
# pool here is dominated by the 13,746-point March harvest + 204 active-
# learning negatives -- the 115 new real-vegetation negatives are only ~0.8%
# of the combined negative set. If they don't move the needle, that alone
# doesn't mean they're useless; it may mean they're too outnumbered for
# XGBoost to weight meaningfully, which would be worth knowing regardless.
#
# Run from the repo root: sbatch slurm/earth_review_pipeline.sbatch
set -e
mkdir -p logs/earth_review_pipeline

TILE_DIRS=(
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210324
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210811
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20240821
)
FEATURES_CSV=classical_features/classical_features_with_earth_review.csv
DENSE_OUT_DIR=dense_scores_earth_review
POINTS=(
    data/interim/labels/active_learning_confirmed_palms.geojson
    data/interim/labels/bellinzona_MASTER_confirmed_palms.geojson
)

module load Python/3.11.3-GCCcore-12.3.0

echo "=== [1/4] feature extraction $(date) ==="
python scripts/15_extract_classical_features.py \
    --tile-dirs "${TILE_DIRS[@]}" \
    --confirmed-points data/interim/labels/bellinzona_MASTER_confirmed_palms.geojson \
    --scouted-points data/interim/labels/bellinzona_new_candidates_scouted.geojson \
    --reviewed-positives data/interim/labels/active_learning_confirmed_palms.geojson \
    --hard-negatives \
        data/interim/labels/active_learning_hard_negatives.geojson \
        data/interim/labels/harvested_deciduous_negatives.geojson \
        data/interim/labels/google_earth_review_batch1_hard_negatives.geojson \
        data/interim/labels/google_earth_review_batch2_hard_negatives.geojson \
    --output "$FEATURES_CSV"

echo "=== [2/4] retrain + leave-one-tile-out CV (diagnostic only) $(date) ==="
python -m src.training.classical_probe --features-csv "$FEATURES_CSV" --model xgboost --max-depth 8

echo "=== [3/4] dense per-pixel re-scoring $(date) ==="
python -m src.inference.dense_classical \
    --features-csv "$FEATURES_CSV" \
    --tile-dirs "${TILE_DIRS[@]}" \
    --tiles c010_r060 c057_r040 c061_r038 \
    --out-dir "$DENSE_OUT_DIR"

echo "=== [4/4] readout — compare against pre-harvest (15-21%) and post-harvest (16.8-24.3%) saturation $(date) ==="
for tile in c010_r060 c057_r040 c061_r038; do
    echo "--- $tile ---"
    python scripts/17_inspect_dense_surface.py \
        --prob-tif "$DENSE_OUT_DIR/${tile}_prob.tif" \
        --points "${POINTS[@]}" \
        --sweep
done

module unload Python/3.11.3-GCCcore-12.3.0

echo "=== DONE $(date) — check the [4/4] section above for the answer ==="
