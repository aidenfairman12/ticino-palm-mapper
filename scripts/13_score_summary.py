#!/usr/bin/env python
"""
13_score_summary.py
====================
Summary statistics for a score_candidates.py output file — how many
locations were scored, the shape of the predicted_prob distribution, and
how many fall in each confidence band. Meant as a quick sanity check after
a scoring run finishes, especially a large/expensive full-grid pass, before
diving into the per-card review.

"Classified as palm" below means predicted_prob > 0.5 — the model's raw
threshold output, not a confirmed palm. Confirming still requires manual
review (11_review_candidates.py -> 12_merge_review_verdicts.py); this
script is purely descriptive of what the model produced.

STATUS: implemented.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np


def parse_summary_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summary statistics for a score_candidates.py output file.")
    p.add_argument("--candidates", type=Path, required=True, help="Path to a candidate_scores*.geojson file.")
    p.add_argument("--bins", type=int, default=10, help="Number of equal-width probability bands to report.")
    return p.parse_args()


def main() -> None:
    args = parse_summary_args()

    gdf = gpd.read_file(args.candidates)
    probs = gdf["predicted_prob"].to_numpy()
    n = len(probs)

    print(f"=== {args.candidates} ===")
    print(f"total locations scored: {n}")
    print(f"unique tiles touched: {gdf['tile'].nunique()}")
    print()

    print("predicted_prob distribution:")
    print(f"  mean:   {probs.mean():.4f}")
    print(f"  median: {np.median(probs):.4f}")
    print(f"  std:    {probs.std():.4f}")
    print(f"  min:    {probs.min():.4f}")
    print(f"  max:    {probs.max():.4f}")
    print()

    n_flagged = int((probs > 0.5).sum())
    print(f"classified as palm (predicted_prob > 0.5): {n_flagged} ({n_flagged / n * 100:.2f}% of scored)")
    print()

    print(f"probability bands ({args.bins} bins):")
    counts, edges = np.histogram(probs, bins=args.bins, range=(0.0, 1.0))
    for i in range(len(counts)):
        bar = "#" * int(counts[i] / max(counts.max(), 1) * 40)
        print(f"  [{edges[i]:.2f}, {edges[i+1]:.2f}): {counts[i]:>7} {bar}")
    print()

    # Very-high-confidence tail is usually the most informative slice for a
    # first review pass — surface it explicitly rather than making the user
    # read it off the histogram.
    for threshold in (0.9, 0.95, 0.99):
        n_above = int((probs > threshold).sum())
        print(f"predicted_prob > {threshold}: {n_above} location(s)")


if __name__ == "__main__":
    main()
