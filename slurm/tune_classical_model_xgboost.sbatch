#!/bin/bash
#SBATCH --job-name=palm-tune-classical-xgb
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/classical_tune/palm-tune-classical-xgb-%j.out
#SBATCH --error=logs/classical_tune/palm-tune-classical-xgb-%j.err

# CPU-only hyperparameter sweep for XGBClassifier
# (scripts/16_tune_classical_model.py --model xgboost).
#
# Unlike the RF sweep, there's no clean per-fit timing number for this one
# — a single-CV-pass timing attempt on the login node was still running
# after 29+ minutes (vs RF's 2m33s for the same thing) and was killed,
# since it was competing for CPU with other users' processes on a shared
# login node rather than getting a clean read, and running a ~400%-CPU job
# directly on a login node isn't good practice regardless. XGBoost's
# sklearn wrapper auto-detects and uses all visible cores by default,
# which is exactly what makes a login-node timing test unreliable here —
# a dedicated SLURM allocation (this job) may behave very differently.
#
# Given that real uncertainty, this gets a generous 24h limit (matching
# the wall-time already proven to work under this account/QoS on other
# jobs) rather than risk another undersized timeout like the RF sweep's
# first attempt. scripts/16 writes its progress CSV incrementally and
# flushes every print, so even if this does need to be killed for time,
# whatever's been computed by then is preserved on disk.
#
# Run from the repo root: sbatch slurm/tune_classical_model_xgboost.sbatch
mkdir -p logs/classical_tune

module load Python/3.11.3-GCCcore-12.3.0

start_time=$SECONDS

python -m scripts.16_tune_classical_model \
    --features-csv classical_features/classical_features.csv \
    --model xgboost \
    --fold-seed 0 \
    --model-seed 42 \
    --top-n 3

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
