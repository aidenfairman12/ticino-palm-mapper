#!/bin/bash
#SBATCH --job-name=palm-ssl-pretrain
#SBATCH --partition=<FILL IN — e.g. gpu>
#SBATCH --account=<FILL IN — your HPC allocation/account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/palm-ssl-%j.out
#SBATCH --error=logs/palm-ssl-%j.err

# Run from the repo root: sbatch train_ssl.sbatch
# Assumes ticino-palm.sif has already been built once via:
#   apptainer build ticino-palm.sif apptainer.def
mkdir -p logs checkpoints

module load apptainer/<FILL IN — check `module avail apptainer` or `singularity`>

apptainer exec --nv \
    --bind "$(pwd)":/workspace \
    --pwd /workspace \
    ticino-palm.sif \
    python -m src.training.pretrain_ssl
