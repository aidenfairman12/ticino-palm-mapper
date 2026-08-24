#!/bin/bash
#SBATCH --job-name=palm-score-review-maskratio
#SBATCH --partition=gpu
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/score_review/palm-score-review-maskratio-%j.out
#SBATCH --error=logs/score_review/palm-score-review-maskratio-%j.err

# Same as score_and_review_bellinzona.sbatch but against the mask-ratio-0.5
# run's checkpoint_best.pt, with its own output files so it doesn't
# overwrite the small/0.75 run's candidate_scores.geojson / review page —
# keeps both comparable.
#
# Run from the repo root: sbatch slurm/score_and_review_maskratio.sbatch
# Assumes ticino-palm.sif has already been built via:
#   apptainer build --fakeroot ticino-palm.sif apptainer.def
mkdir -p logs/score_review review/maskratio

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
        --checkpoint checkpoints/trial_maskratio/checkpoint_best.pt \
        --tile-dirs "${TILE_DIRS[@]}" \
        --confirmed-points data/interim/labels/lugano_MASTER_confirmed_palms.geojson \
        --scouted-points data/interim/labels/lugano_new_candidates_scouted.geojson \
        --hard-negatives data/interim/labels/active_learning_hard_negatives.geojson \
        --reviewed-positives data/interim/labels/active_learning_confirmed_palms.geojson \
        --n-samples 50 \
        --seed 123 \
        --output candidate_scores/maskratio.geojson

apptainer exec --cleanenv \
    --env LD_PRELOAD="/opt/conda/envs/ticino-palm/lib/libjpeg.so.8" \
    --bind "$(pwd)":/workspace \
    --bind /rs_scratch:/rs_scratch \
    --pwd /workspace \
    ticino-palm.sif \
    python scripts/10_review_candidates.py \
        --candidates candidate_scores/maskratio.geojson \
        --tile-dirs "${TILE_DIRS[@]}" \
        --output review/maskratio/index.html

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
