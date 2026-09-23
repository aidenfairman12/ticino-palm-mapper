#!/bin/bash
#SBATCH --job-name=palm-native-10cm-chm-ablation
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/native_10cm_chm_ablation/palm-native-10cm-chm-ablation-%j.out
#SBATCH --error=logs/native_10cm_chm_ablation/palm-native-10cm-chm-ablation-%j.err

# Two ablations on top of the native_10cm_baseline.sbatch run, isolating
# which part of the CHM feature set (if any) was responsible for the
# chm_local_roughness_r1.0-driven accuracy jump seen in that baseline
# (2026-09-23 conversation):
#   A) drop all r=1.0-radius CHM features + chm_point (untrustworthy per-window
#      sample count at that radius, independent of RGB/NIR/NDVI resolution)
#   B) drop the 5 CHM shape descriptors (peak_ratio/skew/kurtosis/
#      local_roughness/asymmetry) at all 3 radii, keep plain chm_mean/std/min/max
#
# Both CSVs already extracted via scripts/27_filter_feature_columns.py from
# classical_features/classical_features_native_10cm.csv (no re-extraction needed -- same tiles,
# same points, just fewer columns). No code changes to scripts/15 or
# dense_classical.py needed: feature_columns() and predict_dense() both derive
# feat_cols from whatever's in the CSV, and dense scoring always computes the
# full superset and subsets to match.
#
# Run from the repo root: sbatch slurm/native_10cm_chm_ablation.sbatch
set -e
mkdir -p logs/native_10cm_chm_ablation

TILE_DIRS=(
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210324
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20240720
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20240810
)
VALIDATION_TILES=(c044_r049 c047_r048 c035_r055)
POINTS=(
    data/interim/labels/active_learning_confirmed_palms.geojson
    data/interim/labels/bellinzona_MASTER_confirmed_palms.geojson
    data/interim/labels/google_earth_review_batch1_confirmed_palms.geojson
    data/interim/labels/google_earth_review_batch2_confirmed_palms.geojson
)

module load Python/3.11.3-GCCcore-12.3.0

run_ablation () {
    LABEL="$1"
    FEATURES_CSV="$2"
    DENSE_OUT_DIR="$3"

    echo "=== [$LABEL] retrain + leave-one-tile-out CV (diagnostic only) $(date) ==="
    python -m src.training.classical_probe --features-csv "$FEATURES_CSV" --model xgboost --max-depth 8

    echo "=== [$LABEL] dense per-pixel re-scoring $(date) ==="
    python -m src.inference.dense_classical \
        --features-csv "$FEATURES_CSV" \
        --tile-dirs "${TILE_DIRS[@]}" \
        --tiles "${VALIDATION_TILES[@]}" \
        --out-dir "$DENSE_OUT_DIR"

    echo "=== [$LABEL] readout $(date) ==="
    for tile in "${VALIDATION_TILES[@]}"; do
        echo "--- $LABEL / $tile ---"
        python scripts/17_inspect_dense_surface.py \
            --prob-tif "$DENSE_OUT_DIR/${tile}_prob.tif" \
            --points "${POINTS[@]}" \
            --sweep
    done
}

run_ablation "A: drop CHM r=1.0" classical_features/classical_features_native_10cm_no_chm_r1.csv dense_scores_native_10cm_no_chm_r1
run_ablation "B: drop CHM shape descriptors" classical_features/classical_features_native_10cm_no_shape.csv dense_scores_native_10cm_no_shape

module unload Python/3.11.3-GCCcore-12.3.0

echo "=== DONE $(date) — compare [A: drop CHM r=1.0] and [B: drop CHM shape descriptors] readouts above against the native_10cm_baseline.sbatch run (chm_local_roughness_r1.0 was its #1 feature at 0.163 importance) ==="
