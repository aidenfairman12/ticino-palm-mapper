#!/bin/bash
#SBATCH --job-name=palm-earth-review-no-harvest
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/earth_review_no_harvest/palm-earth-review-no-harvest-%j.out
#SBATCH --error=logs/earth_review_no_harvest/palm-earth-review-no-harvest-%j.err

# Same chain as earth_review_pipeline.sbatch, with the 13,746-point automatic
# March-NDVI-drop harvest DROPPED from the negative set entirely, leaving
# only human-verified negatives: 204 active_learning_hard_negatives (model-
# flagged, then visually confirmed not-palm) + 115 from the two Google Earth
# review batches (hand-picked from residential/garden terrain). Positives:
# negatives goes from ~1:40 (dominated by the unverified, geographically-
# biased harvest) to roughly 1:1 (356 : ~319+30 baseline-random).
#
# This tests the hypothesis from the 2026-09-22 conversation directly: was
# the harvest run's weak improvement (saturation dropped a little, precision
# barely moved) because 115 real negatives were too diluted by 13,746
# untrusted ones to matter, or because real negatives near-1:1 still aren't
# enough absolute signal for the classical feature set to separate palm from
# other vegetation? Either answer is useful -- if this doesn't help much
# either, the ceiling looks structural to the feature set, not a data-
# imbalance problem, and is worth knowing before more labeling effort.
#
# Same 3 validation tiles as every prior comparison, for direct comparison:
#   pre-harvest:        15-21%    saturation, best precision 0.007-0.026, recall ~92-93%
#   post-harvest:       16.8-24.3% (worse on 1 tile), best precision 0.014-0.041, recall 84.0%
#   +115 real negatives (harvest still in): 11.4-22.8%, best precision 0.010-0.036, recall 78.1%
#
# Run from the repo root: sbatch slurm/earth_review_no_harvest.sbatch
set -e
mkdir -p logs/earth_review_no_harvest

TILE_DIRS=(
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210324
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210811
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20240821
)
FEATURES_CSV=classical_features/classical_features_no_harvest.csv
DENSE_OUT_DIR=dense_scores_no_harvest
POINTS=(
    data/interim/labels/active_learning_confirmed_palms.geojson
    data/interim/labels/bellinzona_MASTER_confirmed_palms.geojson
)

module load Python/3.11.3-GCCcore-12.3.0

echo "=== [1/4] feature extraction (harvest DROPPED) $(date) ==="
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
    --tiles c010_r060 c057_r040 c061_r038 \
    --out-dir "$DENSE_OUT_DIR"

echo "=== [4/4] readout — compare against harvest-in run (11.4-22.8%, best precision 0.010-0.036) $(date) ==="
for tile in c010_r060 c057_r040 c061_r038; do
    echo "--- $tile ---"
    python scripts/17_inspect_dense_surface.py \
        --prob-tif "$DENSE_OUT_DIR/${tile}_prob.tif" \
        --points "${POINTS[@]}" \
        --sweep
done

module unload Python/3.11.3-GCCcore-12.3.0

echo "=== DONE $(date) — check the [4/4] section above for the answer ==="
