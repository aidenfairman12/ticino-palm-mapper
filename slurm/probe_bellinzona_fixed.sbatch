#!/bin/bash
#SBATCH --job-name=palm-probe-bellinzona-fixed
#SBATCH --partition=gpu
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/linear_probe/palm-probe-bellinzona-fixed-%j.out
#SBATCH --error=logs/linear_probe/palm-probe-bellinzona-fixed-%j.err

# Linear-probe eval of checkpoints/dinov2_mae_fixed/checkpoint_best.pt — the
# continued-MAE-pretraining run redone from a fresh init after fixing the
# R,G,B channel-alignment bug (commit fcad6ce), best checkpoint at epoch 140,
# SSL val_loss 0.0930 (see train_ssl_bellinzona_fixed.sbatch). Compare
# against this same run's sibling, probe_baseline_dinov2_fixed.sbatch (plain
# DINOv2, same channel fix, no continued pretraining) to redo the
# mae-continued-pretraining-hurts-linear-probe comparison now that both
# checkpoints start from a correctly-aligned init — the original version of
# that comparison (README's "SSL checkpoints" section) used the OLD,
# misaligned checkpoints/dinov2_mae, so it's no longer a clean read on
# whether continued pretraining itself helps or hurts.
#
# Run from the repo root: sbatch slurm/probe_bellinzona_fixed.sbatch
# Assumes ticino-palm.sif has already been built via:
#   apptainer build --fakeroot ticino-palm.sif apptainer.def
mkdir -p logs/linear_probe

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
    python -m src.training.linear_probe \
        --checkpoint checkpoints/dinov2_mae_fixed/checkpoint_best.pt \
        --tile-dirs "${TILE_DIRS[@]}" \
        --confirmed-points data/interim/labels/lugano_MASTER_confirmed_palms.geojson \
        --scouted-points data/interim/labels/lugano_new_candidates_scouted.geojson \
        --reviewed-positives data/interim/labels/active_learning_confirmed_palms.geojson \
        --hard-negatives data/interim/labels/active_learning_hard_negatives.geojson \
        --pos-weight-multiplier "${POS_WEIGHT_MULT:-1.0}"

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
