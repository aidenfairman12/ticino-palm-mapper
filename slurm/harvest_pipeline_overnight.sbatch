#!/bin/bash
#SBATCH --job-name=palm-harvest-pipeline
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/harvest_pipeline/palm-harvest-pipeline-%j.out
#SBATCH --error=logs/harvest_pipeline/palm-harvest-pipeline-%j.err

# Runs the whole overnight chain as one job, independent of any connected
# terminal — the previous interactive attempt at just step 1 died to a
# dropped SSH connection with no partial output saved:
#
#   1. scripts/15  — feature extraction, now including the 13,746-point
#      March-NDVI-drop harvest as additional hard negatives.
#   2. classical_probe.py — leave-one-tile-out CV, a sanity check that
#      accuracy didn't collapse with the much larger, harder negative set.
#      Purely diagnostic; does not feed step 3 (see below).
#   3. dense_classical.py — re-scores the same 3 tiles inspected earlier
#      this session, fitting its OWN production model from the same CSV
#      (build_production_model fits on ALL rows, independent of step 2's
#      CV fits — nothing needs to be passed between steps 2 and 3, they are
#      two different readouts of the same feature table, not a pipeline).
#   4. scripts/17, --sweep only (no --png, for speed) on all 3 tiles —
#      prints the actual "fraction of tile above 0.5" and per-cluster
#      verdicts, i.e. the number that answers whether this whole session's
#      work actually worked: it was 15-21% before the harvest.
#
# `set -e`: if extraction fails or produces something classical_probe/
# dense_classical then chokes on, the job stops there rather than wasting
# the remaining time-budget running later steps against bad/missing data.
# Check the .err log first if this job ends without reaching step 4.
#
# 3h total: ~2h budgeted for step 1 alone (tile-read-bound — see
# extract_classical_features_with_harvest.sbatch's own reasoning), steps
# 2-4 measured/estimated at a few minutes combined.
#
# Run from the repo root: sbatch slurm/harvest_pipeline_overnight.sbatch
set -e
mkdir -p logs/harvest_pipeline

TILE_DIRS=(
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210324
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210811
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20240821
)
FEATURES_CSV=classical_features/classical_features_with_harvest.csv
DENSE_OUT_DIR=dense_scores_harvest
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
    --output "$FEATURES_CSV"

echo "=== [2/4] retrain + leave-one-tile-out CV (diagnostic only) $(date) ==="
python -m src.training.classical_probe --features-csv "$FEATURES_CSV" --model xgboost --max-depth 8

echo "=== [3/4] dense per-pixel re-scoring $(date) ==="
python -m src.inference.dense_classical \
    --features-csv "$FEATURES_CSV" \
    --tile-dirs "${TILE_DIRS[@]}" \
    --tiles c010_r060 c057_r040 c061_r038 \
    --out-dir "$DENSE_OUT_DIR"

echo "=== [4/4] readout — compare against the pre-harvest 15-21% saturation $(date) ==="
for tile in c010_r060 c057_r040 c061_r038; do
    echo "--- $tile ---"
    python scripts/17_inspect_dense_surface.py \
        --prob-tif "$DENSE_OUT_DIR/${tile}_prob.tif" \
        --points "${POINTS[@]}" \
        --sweep
done

module unload Python/3.11.3-GCCcore-12.3.0

echo "=== DONE $(date) — check the [4/4] section above for the answer ==="
