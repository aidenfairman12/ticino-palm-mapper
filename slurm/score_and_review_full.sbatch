#!/bin/bash
#SBATCH --job-name=palm-score-review-full
#SBATCH --partition=gpu
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/score_review/palm-score-review-full-%j.out
#SBATCH --error=logs/score_review/palm-score-review-full-%j.err

# Full/near-full systematic grid scoring pass + one review page, meant to be
# run rarely (once per meaningfully-updated checkpoint) rather than every
# review session — see 11_review_candidates.py's --offset flag for reviewing
# this same output incrementally across many sessions without re-scoring.
#
# score_candidates.py now resumes from --output if it already exists (skips
# already-scored locations, deterministic grid/subsample means a re-run
# reproduces the same target list) and flushes progress to disk every
# --flush-every locations instead of only writing once at the end — so unlike
# the original one-shot version of this script, a 24h time-limit kill loses
# at most one flush interval of progress, not the whole run. That makes SLURM
# job chaining actually safe/useful here; see the sbatch submission for it.
#
# CHECKPOINT is checkpoints/baseline_dinov2/checkpoint_best.pt (plain
# pretrained DINOv2, no continued MAE pretraining) — confirmed via a
# controlled linear-probe comparison to massively outperform the domain-
# adaptive checkpoints/bellinzona checkpoint (90.9% vs 72.8% pooled
# accuracy, 84.2% vs 48.8% precision, same dataset/architecture/pos_weight).
# See mae-continued-pretraining-hurts-linear-probe project memory. Also
# confirmed an MLP head (--hidden-dim) doesn't help on this checkpoint —
# its features are already separable enough that a linear head ties the
# MLP's result, so production stays on the simpler plain linear probe.
#
# OUTPUT is a NEW path (not candidate_scores_full.geojson from the old
# checkpoint's run) — score_candidates.py's resume-from-output logic tracks
# "already scored" purely by (x, y) coordinates, with no awareness of which
# checkpoint produced them. Reusing the old output path here would make
# every location look already-scored and get silently skipped, leaving
# stale old-checkpoint probabilities in a file that looks like a fresh
# DINOv2-baseline run. The old file/review page are left alone in case
# they're wanted for comparison.
#
# N_SAMPLES is set well above the real grid size (printed each run as "grid:
# N candidate locations at ...m spacing") specifically so no subsampling
# happens and the entire grid gets scored.
#
# Run from the repo root: sbatch --dependency=afterany:<prev_jobid> slurm/score_and_review_full.sbatch
# (see chain submission notes in conversation — first submission needs no --dependency)
mkdir -p logs/score_review candidate_review_full_dinov2

CHECKPOINT="checkpoints/baseline_dinov2/checkpoint_best.pt"
N_SAMPLES=2000000  # safely above the real grid size (~500k ballpark) so the full grid gets scored, not a subsample
OUTPUT="candidate_scores_full_dinov2.geojson"

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
        --output "$OUTPUT"

# First page of the ranked results — subsequent sessions reuse
# $OUTPUT directly with --offset (60, 120, ...) instead of resubmitting
# this whole job.
apptainer exec --cleanenv \
    --env LD_PRELOAD="/opt/conda/envs/ticino-palm/lib/libjpeg.so.8" \
    --bind "$(pwd)":/workspace \
    --bind /rs_scratch:/rs_scratch \
    --pwd /workspace \
    ticino-palm.sif \
    python scripts/11_review_candidates.py \
        --candidates "$OUTPUT" \
        --tile-dirs "${TILE_DIRS[@]}" \
        --offset 0 \
        --output candidate_review_full_dinov2/index.html

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
