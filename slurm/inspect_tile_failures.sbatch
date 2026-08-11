#!/bin/bash
#SBATCH --job-name=palm-inspect-tile-failures
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/linear_probe/palm-inspect-tile-failures-%j.out
#SBATCH --error=logs/linear_probe/palm-inspect-tile-failures-%j.err

# Visual inspection of the 5 tiles with the most robust leave-one-tile-out CV
# recall failures on confirmed positives, from job 11764565 (checkpoints_baseline_dinov2,
# 102 hard negatives / 45 reviewed positives, pos_weight_multiplier=1.0):
#   c052_r050: 7 false negatives (7 of 14 confirmed positives missed, 50% recall)
#   c053_r049: 5 false negatives (5 of 14 missed, 64% recall)
#   c021_r076: 4 false negatives (4 of 4 missed, 0% recall)
#   c065_r061: 4 false negatives (4 of 4 missed, 0% recall)
#   c050_r050: 4 false negatives (4 of 8 missed, 50% recall)
# Picked by RAW false-negative count (not just worst recall), so this is
# robust, multi-example failure signal rather than a single unlucky point —
# tiles with only 1-2 confirmed positives and 0% recall were left out for
# that reason. See scripts/15_inspect_tile_failures.py's docstring: this
# probe is trained on ALL current labels (including these tiles' own
# positives), not the actual held-out CV fold, so a low predicted_prob here
# is a STRONGER signal than the CV fold's own miss, not a weaker one.
#
# CPU only (epyc2 partition, no --gres/--nv) — job_gratis's QOS caps rtx3090
# at 0 GPUs (both per-user and cluster-wide, confirmed via sacctmgr), so
# that's not actually available, and this is a small enough job (5 tiles,
# ~24 confirmed positives to score plus probe training) that CPU inference
# is fine for a one-off diagnostic.
#
# Run from the repo root: sbatch slurm/inspect_tile_failures.sbatch
# Assumes ticino-palm.sif has already been built via:
#   apptainer build --fakeroot ticino-palm.sif apptainer.def
mkdir -p logs/linear_probe

start_time=$SECONDS

TILE_DIRS=(
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210324
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210811
    /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20240821
)

FAILING_TILES=(
    bellinzona_full_nir_2024_c052_r050_nirchm.tif
    bellinzona_full_nir_2024_c053_r049_nirchm.tif
    bellinzona_full_nir_2024_c021_r076_nirchm.tif
    bellinzona_full_nir_2024_c065_r061_nirchm.tif
    bellinzona_full_nir_2024_c050_r050_nirchm.tif
)

# See train_ssl_bellinzona.sbatch: `import timm` loads its own libjpeg before
# rasterio does, and the incompatible copy wins the SONAME slot, breaking
# rasterio's libtiff link. Preload the correct libjpeg so it claims the slot
# first, regardless of import order.
apptainer exec --cleanenv \
    --env LD_PRELOAD="/opt/conda/envs/ticino-palm/lib/libjpeg.so.8" \
    --bind "$(pwd)":/workspace \
    --bind /rs_scratch:/rs_scratch \
    --pwd /workspace \
    ticino-palm.sif \
    python scripts/15_inspect_tile_failures.py \
        --checkpoint checkpoints_baseline_dinov2/checkpoint_best.pt \
        --tile-dirs "${TILE_DIRS[@]}" \
        --confirmed-points data/interim/labels/lugano_MASTER_confirmed_palms.geojson \
        --scouted-points data/interim/labels/lugano_new_candidates_scouted.geojson \
        --reviewed-positives data/interim/labels/active_learning_confirmed_palms.geojson \
        --hard-negatives data/interim/labels/active_learning_hard_negatives.geojson \
        --pos-weight-multiplier 1.0 \
        --tiles "${FAILING_TILES[@]}" \
        --output tile_failure_inspection/index.html

elapsed=$((SECONDS - start_time))
printf 'total job time: %02d:%02d:%02d\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
