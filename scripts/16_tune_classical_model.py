#!/usr/bin/env python
"""
16_tune_classical_model.py
===========================
Grid-searches src/training/classical_probe.py's RandomForestClassifier
hyperparameters, using a FIXED fold split (single --fold-seed) across every
candidate so the comparison is apples-to-apples — see classical_probe.py's
leave_one_tile_out_cv docstring for why fold_seed and model_seed have to
stay decoupled for this to be valid.

n_estimators is intentionally NOT part of the grid — more trees doesn't
fight overfitting (it reduces ensemble variance, with diminishing returns),
so it's fixed high and left alone. The grid covers max_depth,
min_samples_leaf, and class_weight_multiplier, which are the parameters
that actually trade off under/overfitting and precision/recall on this
task.

For every candidate: run leave_one_tile_out_cv once, record pooled
accuracy AND the std of per-fold accuracy (the overfitting-instability
signal from the modeling walkthrough — a config with slightly lower pooled
accuracy but much tighter fold-to-fold spread is often the more trustworthy
pick). Rank by pooled accuracy, then for the top N candidates, re-run
across several different --model-seed values (fold split still fixed) to
check the same accuracy isn't just a lucky RandomForestClassifier init.

STATUS: implemented, driver-only — reuses classical_probe.py's functions,
no modeling logic duplicated here.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.classical_probe import feature_columns, leave_one_tile_out_cv, load_feature_table  # noqa: E402

N_ESTIMATORS = 300  # fixed — see module docstring, not part of the grid

# Small, hand-picked grid rather than a random/Bayesian search — cheap
# enough (a few dozen combinations) that exhaustive is simpler and more
# transparent than sampling it.
MAX_DEPTH_GRID = [3, 5, 8, None]
MIN_SAMPLES_LEAF_GRID = [1, 2, 5, 10]
CLASS_WEIGHT_MULTIPLIER_GRID = [0.5, 1.0, 1.5]

TOP_N_FOR_STABILITY_CHECK = 3
STABILITY_CHECK_SEEDS = [1, 2, 3, 4, 5]


def pooled_accuracy_and_spread(results: dict) -> tuple[float, float]:
    """Return (pooled_accuracy, std_of_per_fold_accuracy) for one CV run's
    results dict, in the same pooled-accuracy definition main() in both
    linear_probe.py and classical_probe.py already use."""
    accs = [r["accuracy"] for r in results.values()]
    tp = sum(r["true_pos"] for r in results.values())
    tn = sum(r["true_neg"] for r in results.values())
    fp = sum(r["false_pos"] for r in results.values())
    fn = sum(r["false_neg"] for r in results.values())
    pooled = (tp + tn) / (tp + tn + fp + fn)
    return pooled, float(np.std(accs))


def run_grid(df: pd.DataFrame, feat_cols: list[str], fold_seed: int, model_seed: int) -> pd.DataFrame:
    rows = []
    combos = list(itertools.product(MAX_DEPTH_GRID, MIN_SAMPLES_LEAF_GRID, CLASS_WEIGHT_MULTIPLIER_GRID))
    for i, (max_depth, min_samples_leaf, class_weight_multiplier) in enumerate(combos, 1):
        results, _ = leave_one_tile_out_cv(
            df, feat_cols, "rf", class_weight_multiplier, fold_seed, model_seed,
            n_estimators=N_ESTIMATORS, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        )
        pooled, spread = pooled_accuracy_and_spread(results)
        rows.append({
            "max_depth": max_depth, "min_samples_leaf": min_samples_leaf,
            "class_weight_multiplier": class_weight_multiplier,
            "pooled_accuracy": pooled, "fold_accuracy_std": spread,
        })
        print(f"[{i}/{len(combos)}] max_depth={max_depth} min_samples_leaf={min_samples_leaf} "
              f"class_weight_multiplier={class_weight_multiplier} -> "
              f"pooled_accuracy={pooled:.3f} fold_std={spread:.3f}")
    return pd.DataFrame(rows)


def stability_check(df: pd.DataFrame, feat_cols: list[str], fold_seed: int, candidates: pd.DataFrame) -> pd.DataFrame:
    """Re-run each of the top candidates across several model_seed values,
    fold split held fixed — a candidate whose accuracy swings a lot across
    seeds is the overfitting-instability signature, even if its
    single-seed number looked good."""
    rows = []
    for _, cand in candidates.iterrows():
        # pandas upcasts max_depth to float64 because the column mixes ints
        # with None (e.g. 3 -> 3.0) — sklearn's RandomForestClassifier
        # rejects a float there, so cast back explicitly rather than pass
        # the DataFrame value straight through.
        max_depth = None if pd.isna(cand["max_depth"]) else int(cand["max_depth"])
        min_samples_leaf = int(cand["min_samples_leaf"])
        seed_accs = []
        for model_seed in STABILITY_CHECK_SEEDS:
            results, _ = leave_one_tile_out_cv(
                df, feat_cols, "rf", cand["class_weight_multiplier"], fold_seed, model_seed,
                n_estimators=N_ESTIMATORS, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            )
            pooled, _ = pooled_accuracy_and_spread(results)
            seed_accs.append(pooled)
        rows.append({
            **cand.to_dict(),
            "mean_pooled_accuracy_across_seeds": float(np.mean(seed_accs)),
            "std_pooled_accuracy_across_seeds": float(np.std(seed_accs)),
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-csv", type=Path, required=True, help="Output of scripts/15_extract_classical_features.py.")
    p.add_argument("--fold-seed", type=int, default=0, help="Held fixed across the whole sweep, same meaning as classical_probe.py's --fold-seed.")
    p.add_argument("--model-seed", type=int, default=42, help="Used for the initial grid pass; the stability check varies this on its own.")
    p.add_argument("--top-n", type=int, default=TOP_N_FOR_STABILITY_CHECK)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_feature_table(args.features_csv)
    feat_cols = feature_columns(df)

    print(f"grid: {len(MAX_DEPTH_GRID) * len(MIN_SAMPLES_LEAF_GRID) * len(CLASS_WEIGHT_MULTIPLIER_GRID)} "
          f"combinations, fold_seed={args.fold_seed} fixed throughout\n")
    grid_results = run_grid(df, feat_cols, args.fold_seed, args.model_seed)

    ranked = grid_results.sort_values("pooled_accuracy", ascending=False)
    print("\n=== top candidates by pooled accuracy ===")
    print(ranked.head(args.top_n).to_string(index=False))

    print(f"\n=== stability check: re-running top {args.top_n} across seeds {STABILITY_CHECK_SEEDS} ===")
    stability = stability_check(df, feat_cols, args.fold_seed, ranked.head(args.top_n))
    print(stability.sort_values("mean_pooled_accuracy_across_seeds", ascending=False).to_string(index=False))

    out_csv = args.features_csv.with_name(args.features_csv.stem + "_hparam_grid.csv")
    grid_results.to_csv(out_csv, index=False)
    print(f"\nfull grid written -> {out_csv}")


if __name__ == "__main__":
    main()
