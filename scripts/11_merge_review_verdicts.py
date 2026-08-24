#!/usr/bin/env python
"""
11_merge_review_verdicts.py
===========================
Merge a candidate_verdicts.csv (exported from 10_review_candidates.py's
review page) into the labeled dataset.

  - verdict == "palm"      -> appended to --positives-output, using
                               corrected_x/corrected_y (the reviewer's
                               click-corrected location, falling back to
                               the original scored point if never clicked)
                               since PalmProbeDataset centers training
                               crops on this exact coordinate.
  - verdict == "not_palm"  -> appended to --hard-negatives-output, using
                               the ORIGINAL scored x/y — that's the exact
                               point the model scored high and got wrong,
                               which is the point worth training against,
                               not wherever the reviewer happened to click.
  - verdict == "unsure"    -> skipped.

Both outputs are append-only across runs: if the output file already
exists, new rows are concatenated onto it (existing rows untouched), so
running this after every review batch accumulates labels over time rather
than overwriting.

STATUS: implemented. Pure merge tool — does not touch training/eval code.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import PROJECT_CRS, ensure_dir  # noqa: E402


def _append_or_create(path: Path, new_rows: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if path.exists():
        existing = gpd.read_file(path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        return gpd.GeoDataFrame(combined, geometry="geometry", crs=PROJECT_CRS)
    return new_rows


def parse_merge_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge reviewed candidate verdicts into labeled positives/hard-negatives."
    )
    p.add_argument("--verdicts", type=Path, required=True, help="Path to candidate_verdicts.csv exported from the review page.")
    p.add_argument("--positives-output", type=Path, default=Path("data/interim/labels/active_learning_confirmed_palms.geojson"))
    p.add_argument("--hard-negatives-output", type=Path, default=Path("data/interim/labels/active_learning_hard_negatives.geojson"))
    return p.parse_args()


def main() -> None:
    args = parse_merge_args()

    df = pd.read_csv(args.verdicts)
    required = {"tile", "x", "y", "corrected_x", "corrected_y", "predicted_prob", "verdict"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"verdicts CSV missing columns: {missing}")

    n_unsure = int((df["verdict"] == "unsure").sum())

    palms = df[df["verdict"] == "palm"]
    positives = gpd.GeoDataFrame(
        {
            "tile": palms["tile"].tolist(),
            "predicted_prob_at_review": palms["predicted_prob"].tolist(),
            "source": ["active_learning"] * len(palms),
        },
        geometry=[Point(x, y) for x, y in zip(palms["corrected_x"], palms["corrected_y"])],
        crs=PROJECT_CRS,
    )

    not_palms = df[df["verdict"] == "not_palm"]
    hard_negatives = gpd.GeoDataFrame(
        {
            "tile": not_palms["tile"].tolist(),
            "predicted_prob_at_review": not_palms["predicted_prob"].tolist(),
            "source": ["active_learning"] * len(not_palms),
        },
        geometry=[Point(x, y) for x, y in zip(not_palms["x"], not_palms["y"])],
        crs=PROJECT_CRS,
    )

    ensure_dir(args.positives_output.parent)
    ensure_dir(args.hard_negatives_output.parent)

    combined_pos = _append_or_create(args.positives_output, positives)
    combined_neg = _append_or_create(args.hard_negatives_output, hard_negatives)

    combined_pos.to_file(args.positives_output, driver="GeoJSON")
    combined_neg.to_file(args.hard_negatives_output, driver="GeoJSON")

    print(f"=== merged {len(positives)} new palm(s) -> {args.positives_output} (total now {len(combined_pos)}) ===")
    print(f"=== merged {len(hard_negatives)} new hard negative(s) -> {args.hard_negatives_output} (total now {len(combined_neg)}) ===")
    if n_unsure:
        print(f"skipped {n_unsure} 'unsure' verdict(s)")


if __name__ == "__main__":
    main()
