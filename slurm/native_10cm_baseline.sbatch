#!/bin/bash
#SBATCH --job-name=palm-native-10cm-baseline
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/native_10cm_baseline/palm-native-10cm-baseline-%j.out
#SBATCH --error=logs/native_10cm_baseline/palm-native-10cm-baseline-%j.err

# First run on the native-10cm-only tile set (all three of March 2021,
# 2024-07-20, 2024-08-10 are true 10cm deliveries, unlike the two 25cm
# dates used in every prior comparison). New validation tiles picked via
# scripts/26_pick_10cm_validation_tiles.py: c010_r060/c057_r040/c061_r038
# have ZERO coverage in the two new 10cm dates, so they can't be reused.
#
# Same harvest-dropped config as earth_review_no_harvest_chm.sbatch (real
# negatives only, ~1:1 balance) -- the only changes are TILE_DIRS and the
# validation --tiles list. scripts/15's CURRENT feature set (including the
# CHM shape descriptors: chm_peak_ratio/skew/kurtosis/local_roughness/
# asymmetry) is used as-is, unmodified -- this is NOT the "drop CHM shape
# features" baseline discussed in the 2026-09-23 conversation, since that
# needs a column-selection decision left to the user. Treat this run as a
# fresh, resolution-consistent data point with the current feature set; the
# shape-feature question can be answered separately by re-deriving the
# features CSV this job writes.
#
# Not directly comparable in % terms to any prior run (different tiles,
# different palm counts: 5 / 3 / 2 vs the old 9 / 5 / 3) -- read cluster
# resolution and precision/recall on their own terms, not against the old
# percentages.
#
# Run from the repo root: sbatch slurm/native_10cm_baseline.sbatch
set -e
mkdir -p logs/native_10cm_baseline

TILE_DIRS=(
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210324
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20240720
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20240810
)
VALIDATION_TILES=(c044_r049 c047_r048 c035_r055)
FEATURES_CSV=classical_features/classical_features_native_10cm.csv
DENSE_OUT_DIR=dense_scores_native_10cm
POINTS=(
    data/interim/labels/active_learning_confirmed_palms.geojson
    data/interim/labels/bellinzona_MASTER_confirmed_palms.geojson
    data/interim/labels/google_earth_review_batch1_confirmed_palms.geojson
    data/interim/labels/google_earth_review_batch2_confirmed_palms.geojson
)

module load Python/3.11.3-GCCcore-12.3.0

echo "=== [1/4] feature extraction (native 10cm dates, harvest DROPPED) $(date) ==="
python scripts/15_extract_classical_features.py \
    --tile-dirs "${TILE_DIRS[@]}" \
    --confirmed-points data/interim/labels/bellinzona_MASTER_confirmed_palms.geojson \
    --scouted-points data/interim/labels/bellinzona_new_candidates_scouted.geojson \
    --reviewed-positives data/interim/labels/active_learning_confirmed_palms.geojson \
    --hard-negatives \
        data/interim/labels/active_learning_hard_negatives.geojson \
        data/interim/labels/google_earth_review_batch1_hard_negatives.geojson \
        data/interim/labels/google_earth_review_batch2_hard_negatives.geojson \
    --output "$FEATURES_CSV"

echo "=== [2/4] retrain + leave-one-tile-out CV (diagnostic only) $(date) ==="
python -m src.training.classical_probe --features-csv "$FEATURES_CSV" --model xgboost --max-depth 8

echo "=== [3/4] dense per-pixel re-scoring $(date) ==="
python -m src.inference.dense_classical \
    --features-csv "$FEATURES_CSV" \
    --tile-dirs "${TILE_DIRS[@]}" \
    --tiles "${VALIDATION_TILES[@]}" \
    --out-dir "$DENSE_OUT_DIR"

echo "=== [4/4] readout — fresh native-10cm baseline, not %-comparable to prior (25cm-date) runs $(date) ==="
for tile in "${VALIDATION_TILES[@]}"; do
    echo "--- $tile ---"
    python scripts/17_inspect_dense_surface.py \
        --prob-tif "$DENSE_OUT_DIR/${tile}_prob.tif" \
        --points "${POINTS[@]}" \
        --sweep
done

module unload Python/3.11.3-GCCcore-12.3.0

echo "=== DONE $(date) — check the [4/4] section above ==="
