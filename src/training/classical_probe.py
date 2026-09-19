"""
Classical baseline: evaluates whether a physically-grounded feature set
(NDVI/CHM/spectral point+neighborhood stats + cross-date stability, from
scripts/15_extract_classical_features.py) separates palm patches from
non-palm patches at least as well as the frozen-ViT-embedding + linear-
probe pipeline (src/training/linear_probe.py).

Mirrors linear_probe.py's leave-one-tile-out CV design (same fold
definition, same evaluate_probe-shaped confusion dict, same per-fold/
pooled-accuracy print format) so results are directly comparable to the
numbers linear_probe.py has been producing (82.9% pooled peak / 77.4%
most recent, on dinov2_vanilla_fixed).

Model: RandomForestClassifier by default; build_model is structured so
"xgboost" can be added as a second --model option without a rewrite.

Runs in seconds on this data size — no GPU/apptainer/sbatch needed.
Use .venv locally, or `module load Python/3.11.3-GCCcore-12.3.0` on the HPC.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Columns in scripts/15's output that are metadata, not model input —
# everything else in the CSV is a feature.
#
# n_dates_covered is excluded as a LEAK, not because it is metadata. It counts
# covering non-nodata tile instances, and this project's tiles overlap, so it
# partly encodes how near a tile seam a location sits plus which flights
# covered it — neither of which has anything to do with palms. Two things make
# it label-correlated rather than merely useless:
#   - sample_negative_points draws a random TILE then a uniform point inside
#     it, so locations in overlaps get several chances to be drawn; measured on
#     lugano_example's tiling, negatives land in overlaps 1.21x more often than
#     an area-uniform point would.
#   - on the real feature table the split is clearer still — positives have
#     median n_dates_covered 2.0 (max 12), negatives median 4.0 (max 8).
# Still extracted by scripts/15 and kept in the CSV, since it is a useful
# diagnostic for coverage; it just must not be a model input.
NON_FEATURE_COLUMNS = {"point_id", "label", "fold_tile", "x", "y", "n_dates_covered"}


def load_feature_table(csv_path: Path) -> pd.DataFrame:
    """Load scripts/15_extract_classical_features.py's output CSV and
    validate its schema invariants (no NaNs in feature columns, binary
    label, non-null fold_tile) — catches a bad upstream extraction run
    before it silently corrupts training."""
    df = pd.read_csv(csv_path)
    feat_cols = feature_columns(df)
    
    assert not df[feat_cols].isna().to_numpy().any(), "One or more feature columns contain null values"
    assert df["label"].isin([0,1]).all(), "lables arent all 0,1"  
    assert df['fold_tile'].notna().all(), "At least one fold tile is na"
 
    return df
    

def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return every column in `df` that isn't in NON_FEATURE_COLUMNS."""
    return [x for x in df.columns if x not in NON_FEATURE_COLUMNS]


def build_model(model_type: str, seed: int, **hyperparams):
    """Construct an unfitted sklearn-compatible classifier.

    `class_weight` (in **hyperparams) must arrive pre-resolved as a
    {0: w0, 1: w1} dict — this function has no visibility into the label
    distribution, so callers are responsible for scaling it themselves.
    RandomForestClassifier takes that dict directly; XGBClassifier has no
    class_weight concept at all, so it's translated to XGBoost's own
    scale_pos_weight (a single scalar ratio) here — class 0's weight is
    always fixed at 1 in this dict, so class_weight[1] IS that ratio.

    xgboost is imported lazily, only inside the "xgboost" branch — it's an
    optional dependency (not installed everywhere this file runs), and the
    "rf" path shouldn't require it to even be present.

    Note max_depth=None means something different per model: unlimited
    depth for RandomForestClassifier, but merely "use XGBoost's own
    default (6)" for XGBClassifier — it is NOT unlimited there. Don't
    reuse the same max_depth grid values across both models expecting
    equivalent behavior at None.
    """
    match model_type:
        case "rf":
            return RandomForestClassifier(n_estimators=hyperparams["n_estimators"],max_depth=hyperparams["max_depth"], min_samples_leaf=hyperparams["min_samples_leaf"], class_weight=hyperparams["class_weight"],random_state=seed)

        case "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=hyperparams["n_estimators"],
                max_depth=hyperparams["max_depth"],
                min_child_weight=hyperparams["min_child_weight"],
                scale_pos_weight=hyperparams["class_weight"][1],
                random_state=seed,
                eval_metric="logloss",
            )

        case _:
            raise ValueError(f"unknown model_type: {model_type!r}")
            
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return accuracy and confusion counts, matching linear_probe.py's
    evaluate_probe output shape (accuracy, true_pos, true_neg, false_pos,
    false_neg)."""
    
    acc = float((y_pred == y_true).mean())
    tPos = int(((y_pred == 1) & (y_true == 1)).sum())
    tNeg = int(((y_pred == 0) & (y_true == 0)).sum())
    fPos = int(((y_pred == 1) & (y_true == 0)).sum())
    fNeg = int(((y_pred == 0) & (y_true == 1)).sum())
    
    ret = {
          "accuracy": acc,
          "true_pos": tPos,
          "true_neg": tNeg,
          "false_pos": fPos,
          "false_neg": fNeg
        }

    return ret

def leave_one_tile_out_cv(
    df: pd.DataFrame, feat_cols: list[str], model_type: str,
    class_weight_multiplier: float, fold_seed: int, model_seed: int, **hyperparams,
) -> tuple[dict,list]:
    """Leave-one-tile-out CV, mirroring linear_probe.py's function of the
    same name. Fits a fresh model per fold on the training rows, evaluates
    on the held-out rows, and returns (per-fold results dict, fitted
    fold models).

    fold_seed and model_seed are deliberately separate: fold_seed drives
    ONLY which negatives land in which fold (hold it fixed across a
    hyperparameter sweep so every candidate is compared on the identical
    train/test split), while model_seed drives ONLY RandomForestClassifier's
    own random_state (vary this alone to check whether a candidate's
    accuracy is stable across different model-fit randomness, or just
    got lucky). Mirrors linear_probe.py's own separation of concerns,
    where the negative-shuffle is hardcoded independent of --seed.
    """
    ret = {}
    fold_models = []
    mask = df['label'] == 1
    folds = sorted(df.loc[mask, "fold_tile"].unique())
    n_folds = len(folds)
    print(f"leave-one-tile-out CV: {n_folds} folds, {len(df):,} rows "
          f"({int(mask.sum()):,} positive, {int((~mask).sum()):,} negative)")
    t0 = time.time()

    # scripts/15 assigns every row a fold_tile, including negatives, but
    # negatives essentially never land in the same tile as a confirmed
    # positive — grouping them by their own fold_tile would leave most
    # folds' test sets with zero negatives. Shuffle them independently
    # into n_folds groups instead, one held out per fold.
    negs = df.index[df["label"] == 0]
    shuffled = np.random.default_rng(fold_seed).permutation(negs.to_numpy())
    neg_groups = [shuffled[i::n_folds] for i in range(n_folds)]


    for tile_name, neg_ind in zip(folds, neg_groups):
        pos_test_mask = (df["fold_tile"] == tile_name) & mask
        neg_test_mask = df.index.isin(neg_ind)
        test_mask = pos_test_mask | neg_test_mask

        train_mask = ~test_mask
        n_pos = df.loc[train_mask, "label"].sum()
        n_neg = train_mask.sum() - n_pos
        # n_pos == 0 isn't reachable with the current dataset (every fold
        # keeps the other ~100 positive tiles in training), but guard it
        # anyway rather than risk a silent ZeroDivisionError — mirrors
        # linear_probe.py's train_probe fallback to a neutral weight.
        pos_weight = (n_neg / n_pos) * class_weight_multiplier if n_pos > 0 else 1.0
        class_weights = {0: 1, 1: pos_weight}

        test_feat = df.loc[test_mask, feat_cols]
        test_label = df.loc[test_mask, "label"]
        train_feat = df.loc[train_mask, feat_cols]
        train_label = df.loc[train_mask, "label"]

        model = build_model(model_type, model_seed, class_weight=class_weights, **hyperparams)
        model.fit(train_feat, train_label)
        y_pred = model.predict(test_feat)

        ret[tile_name] = evaluate_predictions(test_label, y_pred)
        fold_models.append(model)
        done = len(ret)
        elapsed = time.time() - t0
        eta = elapsed / done * (n_folds - done)
        print(f"  fold {done}/{n_folds} ({tile_name}) done — "
              f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)

    return ret, fold_models
        
        


def summarize_feature_importance(fold_models: list, feat_cols: list[str]) -> pd.Series:
    """Average feature_importances_ across every fold's fitted model,
    sorted descending — averaging reduces per-fold noise given how few
    training examples each fold has."""
    metric_arr = np.stack([m.feature_importances_ for m in fold_models])
    metric_arr = metric_arr.mean(axis=0)
    named_metrics = pd.Series(metric_arr,index=feat_cols).sort_values(ascending=False)
    return named_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leave-one-tile-out RF/XGBoost eval of scripts/15's classical feature table.")
    p.add_argument("--features-csv", type=Path, required=True, help="Output of scripts/15_extract_classical_features.py.")
    p.add_argument("--model", choices=["rf", "xgboost"], default="rf")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=None, help="None means unlimited depth for rf, but only 'use XGBoost's own default (6)' for xgboost — not unlimited there.")
    p.add_argument("--min-samples-leaf", type=int, default=1, help="rf only.")
    p.add_argument("--min-child-weight", type=float, default=1.0, help="xgboost only — not the same statistic as min-samples-leaf, just the closest analog.")
    p.add_argument("--class-weight-multiplier", type=float, default=1.0)
    p.add_argument("--fold-seed", type=int, default=0, help="Controls only which negatives land in which fold — hold this fixed across a hyperparameter sweep.")
    p.add_argument("--model-seed", type=int, default=42, help="Controls only RandomForestClassifier's random_state — vary this alone to test a candidate's stability.")
    return p.parse_args()


def main() -> None:
    """Load the feature table, run leave-one-tile-out CV, and print
    per-fold/pooled accuracy plus the top feature importances."""
    args = parse_args()
    df = load_feature_table(args.features_csv)
    feat_cols = feature_columns(df)
    
    # min_samples_leaf (rf) and min_child_weight (xgboost) aren't the same
    # statistic, so only the one relevant to args.model gets passed through.
    model_hyperparams = {"n_estimators": args.n_estimators, "max_depth": args.max_depth}
    if args.model == "rf":
        model_hyperparams["min_samples_leaf"] = args.min_samples_leaf
    else:
        model_hyperparams["min_child_weight"] = args.min_child_weight

    results, fold_models = leave_one_tile_out_cv(
        df, feat_cols, args.model, args.class_weight_multiplier, args.fold_seed, args.model_seed,
        **model_hyperparams,
    )

    print(f"\n{len(results)} folds:")
    total_tp = total_tn = total_fp = total_fn = 0
    accuracies = []
    for tile_name, res in results.items():
        print(f"  {tile_name}: {res}")
        accuracies.append(res["accuracy"])
        total_tp += res["true_pos"]
        total_tn += res["true_neg"]
        total_fp += res["false_pos"]
        total_fn += res["false_neg"]
        
    mean_acc = sum(accuracies) / len(accuracies) if accuracies else float("nan")
    total = total_tp + total_tn + total_fp + total_fn
    pooled_acc = (total_tp + total_tn) / total if total else float("nan")
    print(f"\nmean per-fold accuracy: {mean_acc:.3f}")
    print(f"pooled accuracy (all folds combined): {pooled_acc:.3f} "
        f"(tp={total_tp}, tn={total_tn}, fp={total_fp}, fn={total_fn})")

    importances = summarize_feature_importance(fold_models, feat_cols)
    print("\ntop feature importances:")
    print(importances.head(15))


if __name__ == "__main__":
    main()
