#!/bin/bash
#SBATCH --job-name=palm-tune-classical-rf
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/classical_tune/palm-tune-classical-rf-%j.out
#SBATCH --error=logs/classical_tune/palm-tune-classical-rf-%j.err

# CPU-only hyperparameter sweep for RandomForestClassifier
# (scripts/16_tune_classical_model.py --model rf) — no GPU, no apptainer
# needed, since this only touches classical_features/classical_features.csv
# (pandas/numpy/sklearn), not the raw rasters.
#
# Expected runtime: measured directly on the real 1071-row
# classical_features/classical_features.csv — ONE full 101-fold leave-one-tile-out CV pass at
# n_estimators=300 took 2m33s (job-free, on the login node's CPU). The RF
# grid is 48 combinations + a 15-run stability check (3 candidates x 5
# seeds) = 63 total CV passes, so ~160 minutes (2h40m) end to end. The
# first attempt at this (job 15515696) used a 1h limit based on a much
# smaller synthetic-data estimate and got killed by the time limit with a
# completely empty log (stdout wasn't flushed) — this sbatch now has a 6h
# limit for real headroom, and scripts/16 itself writes its progress CSV
# incrementally and flushes every print, so even an unexpected kill won't
# lose everything the way it did last time.
#
# Run from the repo root: sbatch slurm/tune_classical_model_rf.sbatch
mkdir -p logs/classical_tune

module load Python/3.11.3-GCCcore-12.3.0

start_time=$SECONDS

python -m scripts.16_tune_classical_model \
    --features-csv classical_features/classical_features.csv \
    --model rf \
    --fold-seed 0 \
    --model-seed 42 \
    --top-n 3

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
