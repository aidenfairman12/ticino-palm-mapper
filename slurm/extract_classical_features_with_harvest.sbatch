#!/bin/bash
#SBATCH --job-name=palm-extract-harvest
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/extract_features/palm-extract-harvest-%j.out
#SBATCH --error=logs/extract_features/palm-extract-harvest-%j.err

# scripts/15_extract_classical_features.py, run as a real batch job instead
# of at an interactive prompt — the first attempt at this (154 positives +
# 13,980 negatives, the 13,746 from the new March-NDVI-drop harvest) died to
# a dropped SSH connection partway through the negative set, with no partial
# output saved (the script only writes its CSV once, at the very end, after
# both build_rows calls AND add_temporal_features complete). An sbatch job
# runs independent of any connected terminal, so this can't happen again —
# submit it and disconnect; it keeps running.
#
# 2h time limit: extraction is dominated by tile reads on /rs_scratch, one
# per (point, covering tile) even with build_rows' cache (which only helps
# when several points share the SAME tile, e.g. up to 10 from the harvest's
# --per-tile-cap). Rough estimate for ~14,000 points across up to 3 covering
# dates each: tens of minutes; 2h leaves real margin without tying up the
# gratis QoS unreasonably long if something is slower than expected.
#
# Run from the repo root: sbatch slurm/extract_classical_features_with_harvest.sbatch
mkdir -p logs/extract_features

module load Python/3.11.3-GCCcore-12.3.0
python scripts/15_extract_classical_features.py \
    --tile-dirs \
        /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210324 \
        /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20210811 \
        /rs_scratch/users/af26g813/ticino-palm-mapper/data/processed/bellinzona_full_nir/feature_stack_rs_20240821 \
    --confirmed-points data/interim/labels/bellinzona_MASTER_confirmed_palms.geojson \
    --scouted-points data/interim/labels/bellinzona_new_candidates_scouted.geojson \
    --reviewed-positives data/interim/labels/active_learning_confirmed_palms.geojson \
    --hard-negatives \
        data/interim/labels/active_learning_hard_negatives.geojson \
        data/interim/labels/harvested_deciduous_negatives.geojson \
    --output classical_features/classical_features_with_harvest.csv
module unload Python/3.11.3-GCCcore-12.3.0
