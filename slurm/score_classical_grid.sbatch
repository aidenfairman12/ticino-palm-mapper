#!/bin/bash
#SBATCH --job-name=palm-score-classical
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/score_review/palm-score-classical-%j.out
#SBATCH --error=logs/score_review/palm-score-classical-%j.err

# Full-grid scoring with the classical NDVI/CHM/spectral + XGBoost pipeline
# (src/inference/score_candidates_classical.py) — the pipeline that beat
# the deep ViT+linear-probe approach by ~14 points (96.7% vs 82.9% pooled
# accuracy, see classical-model-beats-deep-pipeline project memory).
#
# Two steps, two DIFFERENT environments — deliberately:
#   1. Scoring runs under the module-loaded Python (rasterio + xgboost
#      installed there via `pip install --user`), NOT apptainer, because
#      this pipeline must stay torch-free — torch + xgboost loaded in the
#      same process segfaults (a known OpenMP-runtime conflict — see
#      score_candidates_classical.py's module docstring), and the module
#      Python has no torch installed at all, so there's no risk of it
#      being pulled in by accident.
#   2. The review-page build (scripts/10_review_candidates.py) runs under
#      apptainer as usual, since it imports src.data.dataset (torch at
#      the top level) for load_tile — but that's a SEPARATE process from
#      the scoring step, so torch and xgboost are never loaded together.
#
# Local smoke test (200 locations, tiny example tile set) scored at
# ~22ms/location — scaled to the real full grid (~400-500k locations,
# similar order to the deep pipeline's grid), that's roughly 2.5-3 hours,
# well inside the 12h limit here with real margin. Uses the winning
# XGBoost hyperparameters from the sweep (max_depth=8, min_child_weight=1,
# class_weight_multiplier=1.0) as the script's own defaults.
#
# Run from the repo root: sbatch slurm/score_classical_grid.sbatch
mkdir -p logs/score_review candidate_scores review/full_classical

TILE_DIRS=(
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210324
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210811
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20240821
)

start_time=$SECONDS

module load Python/3.11.3-GCCcore-12.3.0
python -m src.inference.score_candidates_classical \
    --features-csv classical_features/classical_features.csv \
    --tile-dirs "${TILE_DIRS[@]}" \
    --confirmed-points data/interim/labels/bellinzona_MASTER_confirmed_palms.geojson \
    --scouted-points data/interim/labels/bellinzona_new_candidates_scouted.geojson \
    --hard-negatives data/interim/labels/active_learning_hard_negatives.geojson \
    --reviewed-positives data/interim/labels/active_learning_confirmed_palms.geojson \
    --n-samples 2000000 \
    --output candidate_scores/full_classical.geojson
module unload Python/3.11.3-GCCcore-12.3.0

# See train_ssl_bellinzona.sbatch: `import timm` loads its own libjpeg before
# rasterio does, and the incompatible copy wins the SONAME slot, breaking
# rasterio's libtiff link. Preload the correct libjpeg so it claims the slot
# first, regardless of import order.
apptainer exec --cleanenv \
    --env LD_PRELOAD="/opt/conda/envs/ticino-palm/lib/libjpeg.so.8" \
    --bind "$(pwd)":/workspace \
    --bind /rs_scratch:/rs_scratch \
    --pwd /workspace \
    ticino-palm.sif \
    python scripts/10_review_candidates.py \
        --candidates candidate_scores/full_classical.geojson \
        --tile-dirs "${TILE_DIRS[@]}" \
        --offset 0 \
        --max-cards 100 \
        --output review/full_classical/index.html

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
