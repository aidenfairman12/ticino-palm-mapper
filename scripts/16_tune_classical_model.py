#!/usr/bin/env python
"""
16_tune_classical_model.py
===========================
Grid-searches src/training/classical_probe.py's RandomForestClassifier or
XGBClassifier hyperparameters (--model rf|xgboost), using a FIXED fold
split (single --fold-seed) across every candidate so the comparison is
apples-to-apples — see classical_probe.py's leave_one_tile_out_cv
docstring for why fold_seed and model_seed have to stay decoupled for
this to be valid.

n_estimators is intentionally NOT part of either grid — more trees doesn't
fight overfitting (it reduces ensemble variance, with diminishing
returns), so it's fixed high and left alone. Each model gets its OWN grid
rather than sharing one: max_depth=None means "unlimited depth" for
RandomForestClassifier but merely "use XGBoost's own default (6)" for
XGBClassifier, and min_samples_leaf (rf) / min_child_weight (xgboost)
are different statistics, not a 1:1 swap — see classical_probe.py's
build_model docstring.

For every candidate: run leave_one_tile_out_cv once, record pooled
accuracy AND the std of per-fold accuracy (the overfitting-instability
signal from the modeling walkthrough — a config with slightly lower pooled
accuracy but much tighter fold-to-fold spread is often the more trustworthy
pick). Rank by pooled accuracy, then for the top N candidates, re-run
across several different --model-seed values (fold split still fixed) to
check the same accuracy isn't just a lucky model-fit init.

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

N_ESTIMATORS = 300  # fixed — see module docstring, not part of either grid

# Small, hand-picked grids rather than a random/Bayesian search — cheap
# enough (a few dozen combinations) that exhaustive is simpler and more
# transparent than sampling it. One grid per model — see module docstring
# for why they aren't shared.
GRIDS = {
    "rf": {
        "second_param_name": "min_samples_leaf",
        "max_depth": [3, 5, 8, None],
        "second_param": [1, 2, 5, 10],
        "class_weight_multiplier": [0.5, 1.0, 1.5],
    },
    "xgboost": {
        "second_param_name": "min_child_weight",
        "max_depth": [3, 5, 8],  # None here just means "XGBoost's default of 6", not unlimited — left out to avoid implying otherwise
        "second_param": [1, 3, 5, 10],
        "class_weight_multiplier": [0.5, 1.0, 1.5],
    },
}

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


def run_grid(df: pd.DataFrame, feat_cols: list[str], model_type: str, fold_seed: int, model_seed: int) -> pd.DataFrame:
    grid = GRIDS[model_type]
    second_param_name = grid["second_param_name"]
    rows = []
    combos = list(itertools.product(grid["max_depth"], grid["second_param"], grid["class_weight_multiplier"]))
    for i, (max_depth, second_param, class_weight_multiplier) in enumerate(combos, 1):
        results, _ = leave_one_tile_out_cv(
            df, feat_cols, model_type, class_weight_multiplier, fold_seed, model_seed,
            n_estimators=N_ESTIMATORS, max_depth=max_depth, **{second_param_name: second_param},
        )
        pooled, spread = pooled_accuracy_and_spread(results)
        rows.append({
            "max_depth": max_depth, second_param_name: second_param,
            "class_weight_multiplier": class_weight_multiplier,
            "pooled_accuracy": pooled, "fold_accuracy_std": spread,
        })
        print(f"[{i}/{len(combos)}] max_depth={max_depth} {second_param_name}={second_param} "
              f"class_weight_multiplier={class_weight_multiplier} -> "
              f"pooled_accuracy={pooled:.3f} fold_std={spread:.3f}")
    return pd.DataFrame(rows)


def stability_check(df: pd.DataFrame, feat_cols: list[str], model_type: str, fold_seed: int, candidates: pd.DataFrame) -> pd.DataFrame:
    """Re-run each of the top candidates across several model_seed values,
    fold split held fixed — a candidate whose accuracy swings a lot across
    seeds is the overfitting-instability signature, even if its
    single-seed number looked good."""
    second_param_name = GRIDS[model_type]["second_param_name"]
    rows = []
    for _, cand in candidates.iterrows():
        # pandas upcasts max_depth to float64 because the column mixes ints
        # with None (e.g. 3 -> 3.0) — sklearn's RandomForestClassifier
        # rejects a float there, so cast back explicitly rather than pass
        # the DataFrame value straight through. xgboost's max_depth grid
        # never contains None, so this only ever matters for rf.
        max_depth = None if pd.isna(cand["max_depth"]) else int(cand["max_depth"])
        second_param = cand[second_param_name]
        second_param = int(second_param) if second_param_name == "min_samples_leaf" else float(second_param)
        seed_accs = []
        for model_seed in STABILITY_CHECK_SEEDS:
            results, _ = leave_one_tile_out_cv(
                df, feat_cols, model_type, cand["class_weight_multiplier"], fold_seed, model_seed,
                n_estimators=N_ESTIMATORS, max_depth=max_depth, **{second_param_name: second_param},
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
    p.add_argument("--model", choices=["rf", "xgboost"], default="rf")
    p.add_argument("--fold-seed", type=int, default=0, help="Held fixed across the whole sweep, same meaning as classical_probe.py's --fold-seed.")
    p.add_argument("--model-seed", type=int, default=42, help="Used for the initial grid pass; the stability check varies this on its own.")
    p.add_argument("--top-n", type=int, default=TOP_N_FOR_STABILITY_CHECK)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_feature_table(args.features_csv)
    feat_cols = feature_columns(df)

    grid = GRIDS[args.model]
    n_combos = len(grid["max_depth"]) * len(grid["second_param"]) * len(grid["class_weight_multiplier"])
    print(f"model={args.model}, grid: {n_combos} combinations, fold_seed={args.fold_seed} fixed throughout\n")
    grid_results = run_grid(df, feat_cols, args.model, args.fold_seed, args.model_seed)

    ranked = grid_results.sort_values("pooled_accuracy", ascending=False)
    print("\n=== top candidates by pooled accuracy ===")
    print(ranked.head(args.top_n).to_string(index=False))

    print(f"\n=== stability check: re-running top {args.top_n} across seeds {STABILITY_CHECK_SEEDS} ===")
    stability = stability_check(df, feat_cols, args.model, args.fold_seed, ranked.head(args.top_n))
    print(stability.sort_values("mean_pooled_accuracy_across_seeds", ascending=False).to_string(index=False))

    out_csv = args.features_csv.with_name(f"{args.features_csv.stem}_{args.model}_hparam_grid.csv")
    grid_results.to_csv(out_csv, index=False)
    print(f"\nfull grid written -> {out_csv}")


if __name__ == "__main__":
    main()
