"""
Train XGBoost and Ridge baseline models for the skill_pass position group.

Reads:  data/features/skill_pass_features_train.csv
Writes: models/saved/xgboost_skill_pass.pkl
        models/saved/ridge_baseline_skill_pass.pkl

Hold-out data is never loaded or referenced in this script.

Usage:
    python -m src.modeling.train_skill_pass
"""

from __future__ import annotations

import pathlib

import pandas as pd

from src.modeling.base import EXCLUDE_ALWAYS, build_ridge_features, build_xgb_features
from src.modeling.ridge_baseline import RidgeModel
from src.modeling.xgboost_model import XGBoostModel

FEATURES_DIR = pathlib.Path("data/features")
MODELS_DIR = pathlib.Path("models/saved")
TRAIN_PATH = FEATURES_DIR / "skill_pass_features_train.csv"

_BOOL_MAP = {"True": True, "False": False, True: True, False: False}
_FLAG_COLS = [
    "missing_outcome", "missing_combine", "missing_height_weight",
    "missing_age", "missing_college_stats",
]


def load_feature_csv(path: pathlib.Path) -> pd.DataFrame:
    """Load a feature CSV and restore boolean flag column dtypes.

    Args:
        path: Path to a Stage 2 feature CSV.

    Returns:
        DataFrame with flag columns cast to bool.
    """
    df = pd.read_csv(path, low_memory=False)
    for col in _FLAG_COLS:
        if col in df.columns:
            df[col] = df[col].map(_BOOL_MAP).astype(bool)
    return df


def main() -> None:
    """Run the full skill_pass training pipeline.

    Steps:
        1. Load training feature CSV.
        2. Exclude rows where missing_outcome == True.
        3. Build and log explicit feature allowlists.
        4. Fit Ridge baseline on z-score + binary features.
        5. Fit XGBoost on raw features with Optuna tuning.
        6. Save both models to models/saved/.
    """
    print("Stage 3 — Training (skill_pass)")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Loading {TRAIN_PATH.name} …")
    df = load_feature_csv(TRAIN_PATH)
    print(f"  Total rows loaded: {len(df):,}")

    # Exclude rows with no outcome label — they have no ground truth for training.
    train = df[df["missing_outcome"] == False].copy()
    n_excluded = len(df) - len(train)
    print(f"  Excluded {n_excluded:,} rows (missing_outcome=True)  "
          f"Training rows: {len(train):,}")

    y = train["weighted_av"]
    draft_year = train["draft_year"]

    # Build explicit feature allowlists — logged to catch any silent leakage.
    xgb_features = build_xgb_features(df.columns.tolist())
    ridge_features = build_ridge_features(df.columns.tolist())

    print(f"  XGBoost features ({len(xgb_features)}): {xgb_features}")
    print(f"  Ridge features   ({len(ridge_features)}): {ridge_features}")

    # Hard assertion: no outcome or identifier columns in either list.
    leaked = [f for f in xgb_features + ridge_features if f in EXCLUDE_ALWAYS]
    if leaked:
        raise ValueError(f"Leakage detected — excluded columns in feature list: {leaked}")

    # --- Ridge baseline ---
    print("  Fitting Ridge baseline …")
    ridge = RidgeModel(features=ridge_features)
    ridge.fit(train, y)
    ridge_path = MODELS_DIR / "ridge_baseline_skill_pass.pkl"
    ridge.save(ridge_path)
    print(f"  Saved → {ridge_path}")

    # --- XGBoost ---
    print("  Fitting XGBoost (Optuna, 30 trials) …")
    xgb_model = XGBoostModel(features=xgb_features)
    xgb_model.fit(train, y, draft_year=draft_year)
    xgb_path = MODELS_DIR / "xgboost_skill_pass.pkl"
    xgb_model.save(xgb_path)
    print(f"  Saved → {xgb_path}")

    print("  Done.")


if __name__ == "__main__":
    main()
