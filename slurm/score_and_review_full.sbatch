#!/bin/bash
#SBATCH --job-name=palm-score-review-full
#SBATCH --partition=gpu
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# TODO: set from slurm/score_timing_trial.sbatch's ms/location figure —
# (ms_per_location * N_SAMPLES below) / 1000 / 60 / 60 hours, plus margin.
#SBATCH --time=00:00:00
#SBATCH --output=logs/palm-score-review-full-%j.out
#SBATCH --error=logs/palm-score-review-full-%j.err

# Full/near-full systematic grid scoring pass + one review page, meant to be
# run rarely (once per meaningfully-updated checkpoint) rather than every
# review session — see 11_review_candidates.py's --offset flag for reviewing
# this same output incrementally across many sessions without re-scoring.
#
# Two things MUST be filled in below before submitting — left as visible
# placeholders rather than silently defaulting to something that would run
# but be wrong:
#   1. CHECKPOINT — blocked on the pos_weight=1.0 reproducibility rerun
#      (comparing checkpoints_bellinzona/checkpoint_best.pt, val_loss=0.0918,
#      against checkpoints_trial_maskratio/checkpoint_best.pt, val_loss=0.0666)
#      resolving which checkpoint's downstream linear-probe numbers actually
#      hold up. Don't run this against a checkpoint that regression might
#      rule out.
#   2. N_SAMPLES / --time above — set N_SAMPLES to what you're willing to
#      commit compute to (up to the real grid size, printed by this script
#      itself as "grid: N candidate locations"), THEN run
#      score_timing_trial.sbatch (if you haven't already) and use its
#      ms/location figure to size --time with real margin, not a guess.
#
# Run from the repo root: sbatch slurm/score_and_review_full.sbatch
mkdir -p logs candidate_review_full

CHECKPOINT="TODO_set_this_checkpoint_path.pt"
N_SAMPLES=0  # TODO: set from the timing trial + your time budget

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
        --checkpoint "$CHECKPOINT" \
        --tile-dirs "${TILE_DIRS[@]}" \
        --confirmed-points data/interim/labels/lugano_MASTER_confirmed_palms.geojson \
        --scouted-points data/interim/labels/lugano_new_candidates_scouted.geojson \
        --hard-negatives data/interim/labels/active_learning_hard_negatives.geojson \
        --reviewed-positives data/interim/labels/active_learning_confirmed_palms.geojson \
        --n-samples "$N_SAMPLES" \
        --output candidate_scores_full.geojson

# First page of the ranked results — subsequent sessions reuse
# candidate_scores_full.geojson directly with --offset (60, 120, ...)
# instead of resubmitting this whole job.
apptainer exec --cleanenv \
    --env LD_PRELOAD="/opt/conda/envs/ticino-palm/lib/libjpeg.so.8" \
    --bind "$(pwd)":/workspace \
    --bind /rs_scratch:/rs_scratch \
    --pwd /workspace \
    ticino-palm.sif \
    python scripts/11_review_candidates.py \
        --candidates candidate_scores_full.geojson \
        --tile-dirs "${TILE_DIRS[@]}" \
        --offset 0 \
        --output candidate_review_full/index.html

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
