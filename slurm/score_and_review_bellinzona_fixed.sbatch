#!/bin/bash
#SBATCH --job-name=palm-score-review-bellinzona-fixed
#SBATCH --partition=gpu
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/score_review/palm-score-review-bellinzona-fixed-%j.out
#SBATCH --error=logs/score_review/palm-score-review-bellinzona-fixed-%j.err

# Full/near-full systematic grid scoring pass + one review page, using
# checkpoints/bellinzona_fixed/checkpoint_best.pt (continued-MAE-pretrained,
# channel-alignment fix applied — see commit fcad6ce and
# adapt_patch_embed-channel-misalignment-bug project memory).
#
# Chosen over baseline_dinov2_fixed for this batch specifically: linear-probe
# CV on the *_fixed checkpoints shows baseline_dinov2_fixed wins pooled
# accuracy (81.6% vs 72.4%), but bellinzona_fixed wins recall on the hard
# residential/garden folds (84% vs 66%) that originally motivated the
# channel-alignment fix — i.e. it catches more true palms in exactly the
# settings the fix was meant to help, at the cost of more false positives to
# sift through in review. See mae-continued-pretraining-hurts-linear-probe /
# adapt-patch-embed-channel-misalignment-bug project memory for the full
# per-fold breakdown.
#
# OUTPUT is a NEW path — score_candidates.py's resume-from-output logic
# tracks "already scored" purely by (x, y) coordinates with no awareness of
# which checkpoint produced them, so reusing an existing output path here
# would silently skip every location as "already scored" and leave stale
# probabilities from a different checkpoint's run in a file that looks fresh.
#
# --max-cards 100 (vs the usual default of 60) per request: a bigger first
# review batch.
#
# N_SAMPLES is set well above the real grid size (printed each run as "grid:
# N candidate locations at ...m spacing") so no subsampling happens and the
# entire grid gets scored.
#
# Run from the repo root: sbatch slurm/score_and_review_bellinzona_fixed.sbatch
mkdir -p logs/score_review candidate_review_bellinzona_fixed

CHECKPOINT="checkpoints/bellinzona_fixed/checkpoint_best.pt"
N_SAMPLES=2000000  # safely above the real grid size (~500k ballpark) so the full grid gets scored, not a subsample
OUTPUT="candidate_scores_bellinzona_fixed.geojson"

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
# $OUTPUT directly with --offset (100, 200, ...) instead of resubmitting
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
        --max-cards 100 \
        --output candidate_review_bellinzona_fixed/index.html

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
