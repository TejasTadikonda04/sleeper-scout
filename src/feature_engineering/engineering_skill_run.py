"""
Feature engineering for the skill_run position group (RB).

Pipeline:
  1. Load processed train and holdout CSVs (with boolean flag dtypes restored).
  2. Add shared draft and combine features (shared_features.py).
  3. Fit z-score params on train only; merge into scaler_params.json.
  4. Apply z-scores to both splits.
  5. Add RB-specific production features.
  6. Validate outcome labels.
  7. Write feature CSVs.

Writes:
  data/features/skill_run_features_train.csv
  data/features/skill_run_features_holdout.csv
  data/features/scaler_params.json  (merged with skill_pass params if present)

Usage:
    python -m src.feature_engineering.engineering_skill_run
"""

from __future__ import annotations

import pathlib

import pandas as pd

from src.feature_engineering.shared_features import (
    add_combine_composites,
    add_draft_features,
    apply_z_scores,
    fit_scaler_params,
    load_processed_csv,
    save_scaler_params,
    validate_outcomes,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED_DIR = pathlib.Path("data/processed")
FEATURES_DIR = pathlib.Path("data/features")

TRAIN_PATH = PROCESSED_DIR / "skill_run_train.csv"
HOLDOUT_PATH = PROCESSED_DIR / "skill_run_holdout.csv"
TRAIN_OUT = FEATURES_DIR / "skill_run_features_train.csv"
HOLDOUT_OUT = FEATURES_DIR / "skill_run_features_holdout.csv"

POSITION_GROUP = "skill_run"


# ---------------------------------------------------------------------------
# RB production features
# ---------------------------------------------------------------------------

def add_rb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rushing and receiving production features for the skill_run group.

    All rows in this group are RBs, so no position masking is needed.
    An assertion confirms position_group integrity before computing features.

    Features added:
        yards_per_carry     : col_rush_yards / col_rush_attempts.
                              NaN if col_rush_attempts is null or 0.
        rush_td_rate        : col_rush_tds / col_rush_attempts.
                              NaN if col_rush_attempts is null or 0.
        reception_rate      : col_receptions / col_rush_attempts. Proxy for
                              pass-catching role. NaN if col_rush_attempts
                              is null or 0.
        yards_from_scrimmage: col_rush_yards + col_rec_yards. If only one
                              input is null, the non-null value is used
                              (null treated as 0). NaN only if both are null.
        rush_yards_flag     : True if col_rush_yards is null, else False.
        rec_yards_flag      : True if col_rec_yards is null, else False.

    Args:
        df: Dataframe for the skill_run group containing only RB rows.

    Returns:
        Copy of df with RB feature columns appended.

    Raises:
        AssertionError: If any row has position_group != 'skill_run'.
    """
    assert (df["position_group"] == "skill_run").all(), (
        "add_rb_features called on a dataframe containing non-skill_run rows."
    )

    df = df.copy()

    att_safe = df["col_rush_attempts"].replace(0.0, float("nan"))
    df["yards_per_carry"] = df["col_rush_yards"] / att_safe
    df["rush_td_rate"] = df["col_rush_tds"] / att_safe
    df["reception_rate"] = df["col_receptions"] / att_safe

    # yards_from_scrimmage: use non-null value when only one input is null
    rush_fill = df["col_rush_yards"].fillna(0.0)
    rec_fill = df["col_rec_yards"].fillna(0.0)
    yfs = rush_fill + rec_fill
    both_null = df["col_rush_yards"].isnull() & df["col_rec_yards"].isnull()
    df["yards_from_scrimmage"] = yfs.where(~both_null, float("nan"))

    df["rush_yards_flag"] = df["col_rush_yards"].isnull().fillna(False)
    df["rec_yards_flag"] = df["col_rec_yards"].isnull().fillna(False)

    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_skill_run_features(
    train_path: pathlib.Path = TRAIN_PATH,
    holdout_path: pathlib.Path = HOLDOUT_PATH,
    train_out: pathlib.Path = TRAIN_OUT,
    holdout_out: pathlib.Path = HOLDOUT_OUT,
) -> None:
    """Run the full skill_run feature engineering pipeline.

    Loads train and holdout processed CSVs, applies all feature
    transformations, fits z-score params on train only (merged into
    scaler_params.json alongside skill_pass params), and writes two
    output feature CSVs.

    Args:
        train_path: Path to skill_run_train.csv.
        holdout_path: Path to skill_run_holdout.csv.
        train_out: Destination path for skill_run_features_train.csv.
        holdout_out: Destination path for skill_run_features_holdout.csv.
    """
    train_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Loading {train_path.name} …")
    train = load_processed_csv(train_path)
    print(f"  Loading {holdout_path.name} …")
    holdout = load_processed_csv(holdout_path)

    # Shared features — applied to each split independently
    train = add_draft_features(train)
    train = add_combine_composites(train)
    holdout = add_draft_features(holdout)
    holdout = add_combine_composites(holdout)

    # Z-score: fit on train only, merge into shared scaler_params.json
    params = fit_scaler_params(train, POSITION_GROUP)
    save_scaler_params(params)
    print(f"  Scaler params saved for {len(params)} features.")
    train = apply_z_scores(train, params, POSITION_GROUP)
    holdout = apply_z_scores(holdout, params, POSITION_GROUP)

    # RB-specific features + validation + write
    for label, df, out_path in [
        ("skill_run_train", train, train_out),
        ("skill_run_holdout", holdout, holdout_out),
    ]:
        df = add_rb_features(df)
        validate_outcomes(df, label)
        df.to_csv(out_path, index=False)
        print(f"  Wrote {out_path}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for skill_run feature engineering."""
    print("Stage 2 — Feature Engineering (skill_run)")
    build_skill_run_features()
    print("  Done.")


if __name__ == "__main__":
    main()
