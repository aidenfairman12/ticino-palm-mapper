#!/bin/bash
#SBATCH --job-name=palm-tune-classical
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/classical_tune/palm-tune-classical-%j.out
#SBATCH --error=logs/classical_tune/palm-tune-classical-%j.err

# CPU-only hyperparameter sweep for src/training/classical_probe.py's
# RandomForestClassifier (scripts/16_tune_classical_model.py) — no GPU, no
# apptainer needed, since this only touches classical_features.csv
# (pandas/numpy/sklearn), not the raw rasters. Runs on the epyc2 CPU
# partition instead of tying up a GPU node for a job that can't use one.
#
# Expected runtime: 48 grid combinations + a multi-seed stability check on
# the top few, each combination running a full 101-fold leave-one-tile-out
# CV — roughly 20-35 minutes based on local timing on a much smaller
# synthetic dataset (n_estimators=300 dominates per-fit cost more than row
# count does). The 1h time limit above leaves headroom.
#
# Run from the repo root: sbatch slurm/tune_classical_model.sbatch
mkdir -p logs/classical_tune

module load Python/3.11.3-GCCcore-12.3.0

start_time=$SECONDS

python -m scripts.16_tune_classical_model \
    --features-csv classical_features.csv \
    --fold-seed 0 \
    --model-seed 42 \
    --top-n 3

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
