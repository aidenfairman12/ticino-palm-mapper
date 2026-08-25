#!/bin/bash
#SBATCH --job-name=palm-baseline-dinov2-fixed
#SBATCH --partition=gpu
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/linear_probe/palm-baseline-dinov2-fixed-%j.out
#SBATCH --error=logs/linear_probe/palm-baseline-dinov2-fixed-%j.err

# Regenerates the plain-pretrained-DINOv2 baseline checkpoint after fixing
# the R,G,B channel-misalignment bug in adapt_patch_embed (src/models/mae.py,
# commit fcad6ce) — the previous checkpoints/dinov2_vanilla/checkpoint_best.pt
# had pretrained R/G/B weights routed onto real NIR/R/G data, with true
# Blue/NDVI/CHM zero-initialized. No training happens here (--stats-from
# reuses checkpoints/dinov2_mae's existing, still-valid normalization stats,
# which are data- not weight-dependent) — this only needs a GPU node briefly
# for the timm model load, not a training-length allocation.
#
# Writes to checkpoints/dinov2_vanilla_fixed/, NOT checkpoints/dinov2_vanilla/,
# so the old (buggy) checkpoint stays available for comparison rather than
# being silently overwritten.
#
# Run from the repo root: sbatch slurm/export_baseline_dinov2_fixed.sbatch
mkdir -p logs/linear_probe checkpoints/dinov2_vanilla_fixed

start_time=$SECONDS

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
    python scripts/13_export_baseline_checkpoint.py \
        --stats-from checkpoints/dinov2_mae/checkpoint_best.pt \
        --tile-dirs \
            /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210324 \
            /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210811 \
            /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20240821 \
        --output checkpoints/dinov2_vanilla_fixed/checkpoint_best.pt

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
