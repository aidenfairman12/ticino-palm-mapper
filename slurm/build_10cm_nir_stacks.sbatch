#!/bin/bash
#SBATCH --job-name=palm-build-10cm-nir-stacks
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/build_10cm_nir_stacks/palm-build-10cm-nir-stacks-%j.out
#SBATCH --error=logs/build_10cm_nir_stacks/palm-build-10cm-nir-stacks-%j.err

# Step 1 of the native-10cm migration (checkpoint-2026-09-18-labeling-ceiling.md's
# 2026-09-21/22 updates): build feature_stack_rs_20240720 and
# feature_stack_rs_20240810, the two never-processed dates confirmed native
# 10cm (same as March, unlike the two 25cm dates currently in the pipeline).
#
# No code changes needed -- 07_build_nir_stack.py already supports arbitrary
# --rs-date values and the AOI config already anticipated this exact usage
# (see its imagery.rs_date comment). This iterates over EVERY tile in the
# 03-built feature_stack grid (~5900 tiles) and skips ones with no RS
# coverage for that date, so 2024-07-20 (only 2 raw strips, a small
# footprint) should be fast; 2024-08-10 (4 strips) somewhat more so. `set -e`
# is deliberately OMITTED here -- if the first date's run has an issue, the
# second date should still get a chance to run rather than the whole job
# dying after date 1.
#
# Run from the repo root: sbatch slurm/build_10cm_nir_stacks.sbatch
mkdir -p logs/build_10cm_nir_stacks

module load Python/3.11.3-GCCcore-12.3.0

for date in 20240720 20240810; do
    echo "=== building feature_stack_rs_$date  $(date) ==="
    python scripts/07_build_nir_stack.py --config configs/bellinzona_full_nir.yaml --rs-date "$date"
    echo "=== done with $date  $(date) ==="
done

module unload Python/3.11.3-GCCcore-12.3.0

echo "=== ALL DONE $(date) ==="
