#!/bin/bash
#SBATCH --job-name=bellinzona-pipeline
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3-12:00:00
#SBATCH --output=logs/data_pipeline/bellinzona-pipeline-%j.out
#SBATCH --error=logs/data_pipeline/bellinzona-pipeline-%j.err

# Run from the repo root: sbatch slurm/process_bellinzona.sbatch
# Assumes ticino-palm.sif has already been built via:
#   apptainer build --fakeroot ticino-palm.sif apptainer.def
set -e
mkdir -p logs/data_pipeline

CONFIG=configs/bellinzona_full_nir.yaml
RUN() {
    step_start=$SECONDS
    echo "=== starting: $* ==="
    apptainer exec \
        --bind "$(pwd)":/workspace \
        --bind /rs_scratch:/rs_scratch \
        --pwd /workspace \
        ticino-palm.sif \
        python "$@"
    step_elapsed=$((SECONDS - step_start))
    printf '=== finished: %s (%02d:%02d:%02d) ===\n' "$*" $((step_elapsed/3600)) $((step_elapsed%3600/60)) $((step_elapsed%60))
}

job_start=$SECONDS

RUN scripts/00_fetch_swissimage.py --config "$CONFIG"
RUN scripts/01_fetch_infoflora.py --config "$CONFIG"
RUN scripts/02_align_labels.py --config "$CONFIG"
RUN scripts/03_build_feature_stack.py --config "$CONFIG"
RUN scripts/07_build_nir_stack.py --config "$CONFIG"
RUN scripts/07_build_nir_stack.py --config "$CONFIG" --rs-date 20240821

job_elapsed=$((SECONDS - job_start))
printf 'total job time: %02d:%02d:%02d\n' $((job_elapsed/3600)) $((job_elapsed%3600/60)) $((job_elapsed%60))
