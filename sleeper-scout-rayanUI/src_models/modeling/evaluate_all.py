"""
Evaluate all saved models on hold-out sets and write results reports.

Loads hold-out feature CSVs (never seen during training or tuning), runs
inference with both Ridge and XGBoost models, and writes:
  - models/results/evaluation_summary.csv — one row per (position_group, model)
  - models/results/feature_importance.csv — XGBoost gain importances

Go/No-Go gate checks are printed to stdout.

Usage:
    python -m src.modeling.evaluate_all
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

from src.modeling.ridge_baseline import RidgeModel
from src.modeling.xgboost_model import XGBoostModel

FEATURES_DIR = pathlib.Path("data/features")
MODELS_DIR = pathlib.Path("models/saved")
RESULTS_DIR = pathlib.Path("models/results")

_BOOL_MAP = {"True": True, "False": False, True: True, False: False}
_FLAG_COLS = [
    "missing_outcome", "missing_combine", "missing_height_weight",
    "missing_age", "missing_college_stats",
]

HOLDOUT_FILES = {
    "skill_pass": FEATURES_DIR / "skill_pass_features_holdout.csv",
    "skill_run":  FEATURES_DIR / "skill_run_features_holdout.csv",
}

MODEL_FILES = {
    "skill_pass": {
        "xgboost": MODELS_DIR / "xgboost_skill_pass.pkl",
        "ridge":   MODELS_DIR / "ridge_baseline_skill_pass.pkl",
    },
    "skill_run": {
        "xgboost": MODELS_DIR / "xgboost_skill_run.pkl",
        "ridge":   MODELS_DIR / "ridge_baseline_skill_run.pkl",
    },
}


def load_holdout(path: pathlib.Path) -> pd.DataFrame:
    """Load a hold-out feature CSV and restore boolean flag dtypes.

    Args:
        path: Path to a hold-out feature CSV.

    Returns:
        DataFrame with flag columns cast to bool.
    """
    df = pd.read_csv(path, low_memory=False)
    for col in _FLAG_COLS:
        if col in df.columns:
            df[col] = df[col].map(_BOOL_MAP).astype(bool)
    return df


def run_evaluation() -> bool:
    """Load models, evaluate on hold-out, write results, check Go/No-Go gates.

    Returns:
        True if all Go/No-Go gates pass; False otherwise.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    importance_rows: list[dict] = []
    gate_pass = True

    for group, holdout_path in HOLDOUT_FILES.items():
        print(f"\n  === {group} ===")
        df = load_holdout(holdout_path)
        total = len(df)

        # Exclude rows without a ground-truth outcome.
        eval_df = df[df["missing_outcome"] == False].copy()
        excluded = total - len(eval_df)
        print(f"  Hold-out rows: {total:,}  |  excluded (missing_outcome): {excluded:,}  "
              f"|  evaluated: {len(eval_df):,}")
        if excluded:
            recent = df.loc[df["missing_outcome"] == True, "draft_year"].value_counts().sort_index()
            print(f"  Excluded by draft_year: {recent.to_dict()}")

        y_true = eval_df["weighted_av"]

        for model_type in ("ridge", "xgboost"):
            model_path = MODEL_FILES[group][model_type]
            if not model_path.exists():
                print(f"  *** MODEL NOT FOUND: {model_path} — skipping ***")
                gate_pass = False
                continue

            model = (RidgeModel if model_type == "ridge" else XGBoostModel).load(model_path)
            metrics = model.evaluate(eval_df, y_true)
            print(f"  {model_type:<8}: RMSE={metrics['rmse']:.2f}  "
                  f"MAE={metrics['mae']:.2f}  Spearman={metrics['spearman']:.3f}")

            summary_rows.append({
                "position_group": group,
                "model": model_type,
                "n_eval": len(eval_df),
                "n_excluded": excluded,
                **metrics,
            })

            # XGBoost feature importances.
            if model_type == "xgboost":
                preds = model.predict(eval_df)
                neg_preds = int((preds < 0).sum())
                pct_neg = neg_preds / len(preds) * 100
                pct_nonneg = 100 - pct_neg
                print(f"  Non-negative predictions: {pct_nonneg:.1f}%  "
                      f"(negative: {neg_preds})")
                if pct_nonneg < 95:
                    print(f"  *** GATE FAIL: <95% non-negative predictions ***")
                    gate_pass = False

                for feat, imp in model.feature_importances().items():
                    importance_rows.append({
                        "position_group": group,
                        "feature": feat,
                        "importance": imp,
                    })

    # Write evaluation summary.
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Wrote {summary_path}")

    # Write feature importances.
    if importance_rows:
        imp_df = pd.DataFrame(importance_rows).sort_values(
            ["position_group", "importance"], ascending=[True, False]
        )
        imp_path = RESULTS_DIR / "feature_importance.csv"
        imp_df.to_csv(imp_path, index=False)
        print(f"  Wrote {imp_path}")

    # Go/No-Go gate: XGBoost must beat Ridge on Spearman for at least one group.
    xgb_rows = summary_df[summary_df["model"] == "xgboost"].set_index("position_group")
    ridge_rows = summary_df[summary_df["model"] == "ridge"].set_index("position_group")
    xgb_beats_ridge = False
    spearman_above_threshold = False

    for group in xgb_rows.index:
        if group not in ridge_rows.index:
            continue
        xgb_sp = xgb_rows.loc[group, "spearman"]
        ridge_sp = ridge_rows.loc[group, "spearman"]
        if xgb_sp > ridge_sp:
            xgb_beats_ridge = True
            print(f"  XGBoost > Ridge (Spearman) for {group}: {xgb_sp:.3f} > {ridge_sp:.3f}")
        if xgb_sp > 0.30:
            spearman_above_threshold = True
            print(f"  Spearman > 0.30 for {group}: {xgb_sp:.3f} ✓")

    if not xgb_beats_ridge:
        print("  *** GATE FAIL: XGBoost did not beat Ridge on Spearman for any group ***")
        gate_pass = False
    if not spearman_above_threshold:
        print("  *** GATE FAIL: Spearman < 0.30 for all groups ***")
        gate_pass = False

    print(f"\n  GATE RESULT: {'PASS — ready for Stage 4' if gate_pass else 'FAIL — fix issues above'}")
    return gate_pass


def main() -> None:
    """Entry point: evaluate all models and exit with 0 (pass) or 1 (fail)."""
    print("Stage 3 — Evaluation")
    passed = run_evaluation()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
