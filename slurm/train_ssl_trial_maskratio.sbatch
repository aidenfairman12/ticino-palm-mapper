#!/bin/bash
#SBATCH --job-name=palm-ssl-trial-maskratio
#SBATCH --partition=gpu
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/palm-ssl-trial-maskratio-%j.out
#SBATCH --error=logs/palm-ssl-trial-maskratio-%j.err

# Trial run: lower mask ratio (0.5) vs. the real run's 0.75, small backbone
# kept at default. Fresh short run — deliberately NOT resumable, must not
# touch checkpoints_bellinzona/. See train_ssl_bellinzona.sbatch for the
# real, chained, full-budget run.
#
# Run from the repo root: sbatch slurm/train_ssl_trial_maskratio.sbatch
# Assumes ticino-palm.sif has already been built via:
#   apptainer build --fakeroot ticino-palm.sif apptainer.def
mkdir -p logs checkpoints_trial_maskratio

start_time=$SECONDS

# `import timm` (pulled in via torchvision/Pillow) loads its own libjpeg into
# the process before rasterio does. Because shared libs are deduped by SONAME
# once loaded, rasterio's libtiff then silently binds to that incompatible
# copy instead of its own conda-forge libjpeg-turbo, breaking on the missing
# jpeg12_write_raw_data symbol. Preload the correct libjpeg so it claims the
# SONAME slot first, regardless of import order.
apptainer exec --nv --cleanenv \
    --env LD_PRELOAD="/opt/conda/envs/ticino-palm/lib/libjpeg.so.8" \
    --bind "$(pwd)":/workspace \
    --bind /rs_scratch:/rs_scratch \
    --pwd /workspace \
    ticino-palm.sif \
    python -m src.training.pretrain_ssl \
        --tile-dirs \
            /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210324 \
            /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210811 \
            /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20240821 \
        --in-chans 6 \
        --mask-ratio 0.5 \
        --checkpoint-dir checkpoints_trial_maskratio

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
