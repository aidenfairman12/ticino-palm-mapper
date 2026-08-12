#!/bin/bash
#SBATCH --job-name=palm-ssl-bellinzona-fixed
#SBATCH --partition=gpu
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ssl_pretrain/palm-ssl-bellinzona-fixed-%j.out
#SBATCH --error=logs/ssl_pretrain/palm-ssl-bellinzona-fixed-%j.err

# Re-run of train_ssl_bellinzona.sbatch from a fresh init, after fixing the
# R,G,B channel-alignment bug in adapt_patch_embed (src/models/mae.py) — the
# original checkpoints/bellinzona run started from a misaligned init (NIR/R/G
# routed through pretrained R/G/B weights, true Blue/NDVI/CHM zeroed). This
# run isolates that fix as the only changed variable: same tile-dirs, same
# in-chans, same mask-ratio/model-size/batch-size/total-epochs/patience
# (all left at pretrain_ssl.py's defaults, unchanged from the original run).
#
# Writes to checkpoints/bellinzona_fixed/, NOT checkpoints/bellinzona/ — the
# resume-glob below is scoped to the new dir specifically so this can safely
# chain across multiple 24h submissions WITHOUT ever resuming from the old,
# misaligned checkpoint.
mkdir -p logs/ssl_pretrain checkpoints/bellinzona_fixed

start_time=$SECONDS

LATEST_CKPT=$(ls -v checkpoints/bellinzona_fixed/checkpoint_epoch*.pt 2>/dev/null | tail -n 1)
RESUME_ARGS=()
if [ -n "$LATEST_CKPT" ]; then
    RESUME_ARGS=(--resume-from "$LATEST_CKPT")
fi

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
        --checkpoint-dir checkpoints/bellinzona_fixed \
        "${RESUME_ARGS[@]}"

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
