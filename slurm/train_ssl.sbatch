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
#SBATCH --output=logs/ssl_pretrain/palm-ssl-%j.out
#SBATCH --error=logs/ssl_pretrain/palm-ssl-%j.err

# Run from the repo root: sbatch slurm/train_ssl.sbatch
# Assumes ticino-palm.sif has already been built once via:
#   apptainer build --fakeroot ticino-palm.sif apptainer.def
#
# --checkpoint-dir passed explicitly (unlike originally) since checkpoints/
# is now a parent directory holding every run's own subdirectory, not a
# single flat run dir — pretrain_ssl.py's own default of "checkpoints"
# would otherwise write loose files back into the parent again.
mkdir -p logs/ssl_pretrain checkpoints/rgb_chm_original

# apptainer is on PATH system-wide on UBELIX (/usr/bin/apptainer) — no module needed.

start_time=$SECONDS

apptainer exec --nv \
    --bind "$(pwd)":/workspace \
    --pwd /workspace \
    ticino-palm.sif \
    python -m src.training.pretrain_ssl \
        --checkpoint-dir checkpoints/rgb_chm_original

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
