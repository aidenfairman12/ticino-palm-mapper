#!/bin/bash
#SBATCH --job-name=palm-score-timing-trial
#SBATCH --partition=gpu
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/palm-score-timing-trial-%j.out
#SBATCH --error=logs/palm-score-timing-trial-%j.err

# Purpose: get a real throughput number (ms/location) on the actual GPU, to
# extrapolate how long a full/near-full grid scoring pass will take before
# committing a much longer job to it.
#
# Scores TRIAL_N_SAMPLES locations only. score_candidates.py prints its own
# phase-timed breakdown (load / probe-train / grid-gen / scoring), so this
# script doesn't need to time anything itself — watch the log for a line like:
#   timing — load: 2.4s, probe train: 4.6s, grid gen: 0.4s, scoring: 2.8s
#   (140.61 ms/location, fixed overhead excluded)
# The ms/location figure already excludes the fixed costs (checkpoint load,
# probe training) that won't scale with --n-samples, so it's directly usable:
#   minutes_for_full_grid = (ms_per_location * N) / 1000 / 60
# where N is the real full-grid size, also printed on its own line:
#   grid: N candidate locations at 11.0m spacing
#
# Timing is checkpoint-independent here (same vit_small architecture either
# way), so this can run against whichever checkpoint is on disk right now —
# no need to wait for the reproducibility rerun to get a throughput estimate.
#
# Run from the repo root: sbatch slurm/score_timing_trial.sbatch
mkdir -p logs

TRIAL_N_SAMPLES=5000

start_time=$SECONDS

TILE_DIRS=(
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210324
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210811
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20240821
)

# See train_ssl_bellinzona.sbatch: `import timm` loads its own libjpeg before
# rasterio does, and the incompatible copy wins the SONAME slot, breaking
# rasterio's libtiff link. Preload the correct libjpeg so it claims the slot
# first, regardless of import order.
apptainer exec --nv --cleanenv \
    --env LD_PRELOAD="/opt/conda/envs/ticino-palm/lib/libjpeg.so.8" \
    --bind "$(pwd)":/workspace \
    --bind /rs_scratch:/rs_scratch \
    --pwd /workspace \
    ticino-palm.sif \
    python -m src.inference.score_candidates \
        --checkpoint checkpoints_trial_maskratio/checkpoint_best.pt \
        --tile-dirs "${TILE_DIRS[@]}" \
        --confirmed-points data/interim/labels/lugano_MASTER_confirmed_palms.geojson \
        --scouted-points data/interim/labels/lugano_new_candidates_scouted.geojson \
        --hard-negatives data/interim/labels/active_learning_hard_negatives.geojson \
        --reviewed-positives data/interim/labels/active_learning_confirmed_palms.geojson \
        --n-samples "$TRIAL_N_SAMPLES" \
        --output candidate_scores_timing_trial.geojson

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
