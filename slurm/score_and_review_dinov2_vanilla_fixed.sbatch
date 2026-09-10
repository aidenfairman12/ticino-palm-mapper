#!/bin/bash
#SBATCH --job-name=palm-score-review-dinov2-vanilla-fixed
#SBATCH --partition=gpu
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/score_review/palm-score-review-dinov2-vanilla-fixed-%j.out
#SBATCH --error=logs/score_review/palm-score-review-dinov2-vanilla-fixed-%j.err

# Full/near-full systematic grid scoring pass + one review page, using
# checkpoints/dinov2_vanilla_fixed/checkpoint_best.pt (plain pretrained
# DINOv2, no continued MAE pretraining, channel-alignment fix applied —
# see commit fcad6ce and adapt_patch_embed-channel-misalignment-bug
# project memory).
#
# The existing candidate_scores/full_dinov2_vanilla.geojson (from
# slurm/score_and_review_full.sbatch, dated 2026-08-07) was scored with the
# UN-fixed checkpoints/dinov2_vanilla checkpoint, before the channel-
# alignment fix — it predates fcad6ce entirely and isn't a valid stand-in
# for this checkpoint. Hence a fresh scoring pass here rather than reusing it.
#
# Picked dinov2_vanilla_fixed over dinov2_mae_fixed for this batch: re-probing
# both checkpoints after growing the label set to 75 confirmed/172 hard
# negatives (jobs 14374142/14374143) showed dinov2_vanilla_fixed improving
# with more data (81.6% -> 82.9% pooled) while dinov2_mae_fixed got worse
# almost everywhere except one fold (c021_r076) — see
# adapt-patch-embed-channel-misalignment-bug / mae-continued-pretraining-
# hurts-linear-probe project memory for the full per-fold breakdown. Since
# dinov2_vanilla_fixed is the checkpoint actually responding well to more
# labels, it gets the next active-learning round, not dinov2_mae_fixed.
#
# OUTPUT is a NEW path — score_candidates.py's resume-from-output logic
# tracks "already scored" purely by (x, y) coordinates with no awareness of
# which checkpoint produced them, so reusing an existing output path here
# would silently skip every location as "already scored" and leave stale
# probabilities from a different checkpoint's run in a file that looks fresh.
#
# --max-cards 60 per request (the usual/default batch size).
#
# N_SAMPLES is set well above the real grid size (printed each run as "grid:
# N candidate locations at ...m spacing") so no subsampling happens and the
# entire grid gets scored.
#
# Run from the repo root: sbatch slurm/score_and_review_dinov2_vanilla_fixed.sbatch
mkdir -p logs/score_review review/full_dinov2_vanilla_fixed

CHECKPOINT="checkpoints/dinov2_vanilla_fixed/checkpoint_best.pt"
N_SAMPLES=2000000  # safely above the real grid size (~500k ballpark) so the full grid gets scored, not a subsample
OUTPUT="candidate_scores/full_dinov2_vanilla_fixed.geojson"

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
    python scripts/10_review_candidates.py \
        --candidates "$OUTPUT" \
        --tile-dirs "${TILE_DIRS[@]}" \
        --offset 0 \
        --max-cards 60 \
        --output review/full_dinov2_vanilla_fixed/index.html

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
