"""
Classical baseline: does a physically-grounded feature set (NDVI/CHM/
spectral point+neighborhood stats + cross-date stability, from
scripts/15_extract_classical_features.py) separate palm patches from
non-palm patches at least as well as the frozen-ViT-embedding + linear-
probe pipeline (src/training/linear_probe.py)?

STUBBED ON PURPOSE — this file is yours to implement. Every function below
has a docstring describing what it needs to do. No logic is filled in.

This deliberately mirrors linear_probe.py's leave-one-tile-out CV design
as closely as possible, operating on scripts/15's precomputed tabular
feature table instead of live-extracted ViT embeddings — same fold
definition, same evaluate_probe-shaped confusion dict, same printed
per-fold/pooled-accuracy format, so a run of this script is directly
comparable to the numbers linear_probe.py has been producing all along
(82.9% pooled peak / 77.4% most recent, on dinov2_vanilla_fixed).

CRITICAL DESIGN NOTE — read before implementing leave_one_tile_out_cv:
scripts/15_extract_classical_features.py assigns EVERY row a `fold_tile`
value, including negatives (whichever tile happens to cover that negative
point). Do NOT naively group negatives by their own `fold_tile` the way
positives are grouped — hard/random negatives essentially never land in
the same tile as a confirmed positive (same reasoning documented in
linear_probe.py's leave_one_tile_out_cv), so doing that would leave most
folds' test sets with zero negatives, never actually testing specificity.
Instead, replicate linear_probe.py's actual approach: derive n_folds from
the number of distinct POSITIVE fold_tile values, then separately shuffle
the negative rows (fixed seed, independent of fold_tile) into n_folds
roughly-equal groups, one held out per fold alongside that fold's
held-out positive tile.

Model choice: starting with RandomForestClassifier (already in
requirements.txt/environment.yml, no new dependency needed). Structure
build_model so adding "xgboost" as a second --model option later is a
small addition, not a rewrite — xgboost isn't installed yet, don't wire
it up until it's actually needed.

No GPU, no apptainer, no sbatch job needed for this file — it trains in
seconds on this data size. Run it directly, locally in .venv or on the
HPC via the module-loaded Python (module load Python/3.11.3-GCCcore-12.3.0
gives a working scikit-learn/pandas environment without the container).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Columns in scripts/15's output that are metadata, not model input —
# everything else in the CSV is a feature.
NON_FEATURE_COLUMNS = {"point_id", "label", "fold_tile", "x", "y"}


def load_feature_table(csv_path: Path) -> pd.DataFrame:
    """Load scripts/15_extract_classical_features.py's output CSV.

    Sanity-check whatever you think is worth checking here (no NaNs in
    feature columns, label is 0/1, fold_tile is non-null) — this is the
    one place a bad upstream extraction run would surface, so a couple of
    asserts here are worth more than they cost.
    """
    raise NotImplementedError


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return every column in `df` that isn't in NON_FEATURE_COLUMNS —
    derive the feature list from the dataframe itself rather than hardcoding
    all ~85 names by hand, so it stays correct if scripts/15 adds/removes a
    feature later.
    """
    raise NotImplementedError


def build_model(model_type: str, class_weight_multiplier: float, seed: int, **hyperparams):
    """Construct an unfitted sklearn-compatible classifier.

    model_type == "rf": RandomForestClassifier. Worth exposing at least
    n_estimators, max_depth, min_samples_leaf as CLI-tunable (see
    parse_args) — min_samples_leaf is probably your primary overfitting
    guard given how few examples you have.

    class_weight_multiplier mirrors linear_probe.py's pos_weight_multiplier
    naming/reasoning exactly: 1.0 should fully compensate for the neg:pos
    imbalance (sklearn's class_weight='balanced' does this out of the box),
    other values should scale it — you'll likely need to hand-roll a
    class_weight dict ({0: w0, 1: w1}) rather than the 'balanced' string if
    you want the same tunable-multiplier behavior linear_probe.py has.

    seed feeds RandomForestClassifier's own random_state, for reproducible
    trees given fixed data — separate from the CV fold-assignment seed
    below.
    """
    raise NotImplementedError


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Same shape as linear_probe.py's evaluate_probe: a dict with keys
    accuracy, true_pos, true_neg, false_pos, false_neg — so results print
    in the identical format and existing eyeballing/tooling doesn't care
    which pipeline produced them.
    """
    raise NotImplementedError


def leave_one_tile_out_cv(
    df: pd.DataFrame, feat_cols: list[str], model_type: str,
    class_weight_multiplier: float, seed: int, **hyperparams,
) -> dict:
    """Leave-one-tile-out CV, mirroring linear_probe.py's function of the
    same name — see the CRITICAL DESIGN NOTE at the top of this file for
    the one part that must NOT be a naive groupby(fold_tile) on both
    labels: positives are grouped by fold_tile, negatives are grouped by a
    separate fixed-seed shuffle into the same number of folds.

    For each fold: fit a FRESH model (build_model) on the training rows
    only, predict on the held-out rows, evaluate_predictions, store under
    the held-out tile's name — same ret[tile_name] = {...} structure
    linear_probe.py returns, for a diffable comparison.
    """
    raise NotImplementedError


def summarize_feature_importance(fold_models: list, feat_cols: list[str]) -> pd.Series:
    """Average feature_importances_ across every fold's fitted model
    (leave_one_tile_out_cv fits a new model per fold — a single fold's
    importances are noisy on this little data, averaging across all of
    them is the more trustworthy summary), sorted descending. This is the
    main artifact for the interpretation step discussed in the modeling
    walkthrough — check whether the top features line up with real palm
    biology (chm_peakiness, ndvi_contrast, temporal stability) or something
    spurious/tile-specific.
    """
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leave-one-tile-out RF/XGBoost eval of scripts/15's classical feature table.")
    p.add_argument("--features-csv", type=Path, required=True, help="Output of scripts/15_extract_classical_features.py.")
    p.add_argument("--model", choices=["rf"], default="rf", help="xgboost intentionally not wired up yet — not installed, add when needed.")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=1)
    p.add_argument("--class-weight-multiplier", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    """Load the feature table, run leave_one_tile_out_cv, print per-fold
    results + mean/pooled accuracy in the same format as linear_probe.py's
    main() (see its tail end for the exact print statements to match), then
    print the top-15 or so averaged feature importances.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
