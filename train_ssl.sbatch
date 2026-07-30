#!/bin/bash
#SBATCH --job-name=palm-ssl-pretrain
#SBATCH --partition=gpu
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/palm-ssl-%j.out
#SBATCH --error=logs/palm-ssl-%j.err

# Run from the repo root: sbatch train_ssl.sbatch
# Assumes ticino-palm.sif has already been built once via:
#   apptainer build --fakeroot ticino-palm.sif apptainer.def
mkdir -p logs checkpoints

# apptainer is on PATH system-wide on UBELIX (/usr/bin/apptainer) — no module needed.

start_time=$SECONDS

apptainer exec --nv \
    --bind "$(pwd)":/workspace \
    --pwd /workspace \
    ticino-palm.sif \
    python -m src.training.pretrain_ssl

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
