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
# Turned out to be a 3-stage chain, not just 07 -- scratch cleanup had
# already claimed 00's raw RGB tiles AND 03's [R,G,B,CHM] feature_stack (both
# purely-derived, regenerable outputs; nothing irreplaceable was lost, see
# the 2026-09-22 conversation), so both need rebuilding across the WHOLE AOI
# (~5900 tiles) before 07 has anything to read for the new dates:
#
#   1. 00_fetch_swissimage.py -- free RGB from swisstopo STAC (no account,
#      no local upload needed)
#   2. 03_build_feature_stack.py -- LiDAR CHM (now using the fixed
#      native-resolution-diff-then-resample method, not the old
#      resample-then-subtract one -- this is also this fix's first real
#      full-AOI exercise)
#   3. 07_build_nir_stack.py, once per date -- fuses in the paid RS delivery
#      (already safe in git-tracked data/raw/, no upload needed)
#
# Step 3 now covers ALL FIVE dates, not just the two new ones: the fixed
# CHM computed in step 2 is shared/identical across every date by
# construction (07 copies the SAME chm band from 03's output into every
# feature_stack_rs_<date> it writes), but the three ALREADY-BUILT dates
# (20210324, 20210811, 20240821) still hold CHM from the OLD, buggy method
# until they're re-fused against this run's corrected 03 output too --
# otherwise the old and new dates would silently disagree on CHM for the
# same ground, breaking the "CHM is date-invariant" property relied on
# throughout scripts/15 and dense_classical.py. Re-running 07 for the old
# three is cheap relative to steps 1-2 (no new fetching, just re-reading
# already-open RS strips and the now-updated 03 tiles).
#
# `set -e`: each stage genuinely depends on the previous succeeding here
# (unlike the original 07-only version, where dates were independent of
# each other) -- no point starting 03 if 00 failed, or 07 if 03 failed.
#
# Confirmed ~1TB+ free needed (per configs/bellinzona_full_nir.yaml's own
# note) -- this is a purely-regenerable CACHE though, not something to
# protect from future scratch cleanup; re-running this job is the recovery
# plan, not a home-directory backup.
#
# Run from the repo root: sbatch slurm/build_10cm_nir_stacks.sbatch
set -e
mkdir -p logs/build_10cm_nir_stacks

module load Python/3.11.3-GCCcore-12.3.0

echo "=== [1/3] 00_fetch_swissimage.py (free RGB, whole AOI) $(date) ==="
python scripts/00_fetch_swissimage.py --config configs/bellinzona_full_nir.yaml

echo "=== [2/3] 03_build_feature_stack.py (LiDAR CHM, fixed method, whole AOI) $(date) ==="
python scripts/03_build_feature_stack.py --config configs/bellinzona_full_nir.yaml

for rs_date in 20210324 20210811 20240821 20240720 20240810; do
    echo "=== [3/3] 07_build_nir_stack.py --rs-date $rs_date $(date) ==="
    python scripts/07_build_nir_stack.py --config configs/bellinzona_full_nir.yaml --rs-date "$rs_date"
    echo "=== done with $rs_date  $(date) ==="
done

module unload Python/3.11.3-GCCcore-12.3.0

echo "=== ALL DONE $(date) ==="
