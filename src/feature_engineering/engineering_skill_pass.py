"""
Feature engineering for the skill_pass position group (QB, WR, TE).

Pipeline:
  1. Load processed train and holdout CSVs (with boolean flag dtypes restored).
  2. Add shared draft and combine features (shared_features.py).
  3. Fit z-score params on train only; save to scaler_params.json.
  4. Apply z-scores to both splits.
  5. Add position dummies (is_QB, is_WR, is_TE) with sum-to-1 assertion.
  6. Add position-specific production features (WR/TE and QB).
  7. Validate outcome labels.
  8. Write feature CSVs.

Writes:
  data/features/skill_pass_features_train.csv
  data/features/skill_pass_features_holdout.csv
  data/features/scaler_params.json  (merged with any existing params)

Usage:
    python -m src.feature_engineering.engineering_skill_pass
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

TRAIN_PATH = PROCESSED_DIR / "skill_pass_train.csv"
HOLDOUT_PATH = PROCESSED_DIR / "skill_pass_holdout.csv"
TRAIN_OUT = FEATURES_DIR / "skill_pass_features_train.csv"
HOLDOUT_OUT = FEATURES_DIR / "skill_pass_features_holdout.csv"

POSITION_GROUP = "skill_pass"


# ---------------------------------------------------------------------------
# Position dummies
# ---------------------------------------------------------------------------

def add_position_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary position indicator columns for the skill_pass group.

    Columns added: is_QB, is_WR, is_TE (int: 0 or 1).

    Asserts that is_QB + is_WR + is_TE == 1 for every row, guaranteeing each
    player is mapped to exactly one position.

    Args:
        df: Dataframe with a 'position' column containing QB, WR, or TE values.

    Returns:
        Copy of df with is_QB, is_WR, is_TE appended.

    Raises:
        ValueError: If any row's position dummies do not sum to exactly 1.
    """
    df = df.copy()
    df["is_QB"] = (df["position"] == "QB").astype(int)
    df["is_WR"] = (df["position"] == "WR").astype(int)
    df["is_TE"] = (df["position"] == "TE").astype(int)

    dummy_sum = df["is_QB"] + df["is_WR"] + df["is_TE"]
    if (dummy_sum != 1).any():
        bad = df.loc[dummy_sum != 1, "position"].unique().tolist()
        raise ValueError(
            f"is_QB + is_WR + is_TE != 1 for positions: {bad}. "
            "Only QB, WR, TE are valid in the skill_pass group."
        )
    return df


# ---------------------------------------------------------------------------
# WR / TE production features
# ---------------------------------------------------------------------------

def add_wr_te_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add receiving production features for WR and TE rows.

    All four features are set to NaN for QB rows — never 0. This distinction
    is critical: XGBoost uses NaN to signal "not applicable" and routes those
    rows differently from a genuine zero value.

    Features added:
        dominator_rating   : col_rec_yards / (col_rec_yards + col_rush_yards).
                             col_rush_yards null is treated as 0 in the
                             denominator (WR/TE typically have no rush yards).
                             NaN if col_rec_yards is null. Validated to
                             [0.0, 1.0]; out-of-range values are logged and
                             capped.
        yards_per_reception: col_rec_yards / col_receptions.
                             NaN if col_receptions is null or 0.
        rec_td_rate        : col_rec_tds / col_receptions.
                             NaN if col_receptions is null or 0.
        rec_yards_flag     : True if col_rec_yards is null, else False.
                             Applied to all rows (not WR/TE-only).

    Args:
        df: Dataframe with is_QB, is_WR, is_TE dummies already added.

    Returns:
        Copy of df with WR/TE feature columns appended.
    """
    df = df.copy()
    qb_mask = df["is_QB"] == 1

    # dominator_rating — treat null rush_yards as 0 for WR/TE denominator
    rush_fill = df["col_rush_yards"].fillna(0.0)
    denom = df["col_rec_yards"] + rush_fill
    dominator = df["col_rec_yards"] / denom.replace(0.0, float("nan"))

    over_1 = dominator > 1.0
    if over_1.any():
        print(f"  [dominator_rating] {int(over_1.sum())} value(s) > 1.0 — capping at 1.0")
        dominator = dominator.clip(upper=1.0)
    under_0 = dominator < 0.0
    if under_0.any():
        print(f"  [dominator_rating] {int(under_0.sum())} value(s) < 0.0 — capping at 0.0")
        dominator = dominator.clip(lower=0.0)

    dominator[qb_mask] = float("nan")
    df["dominator_rating"] = dominator

    # yards_per_reception and rec_td_rate
    rec_safe = df["col_receptions"].replace(0.0, float("nan"))
    ypr = df["col_rec_yards"] / rec_safe
    ypr[qb_mask] = float("nan")
    df["yards_per_reception"] = ypr

    td_rate = df["col_rec_tds"] / rec_safe
    td_rate[qb_mask] = float("nan")
    df["rec_td_rate"] = td_rate

    # rec_yards_flag — all rows, not WR/TE-only
    df["rec_yards_flag"] = df["col_rec_yards"].isnull().fillna(False)

    return df


# ---------------------------------------------------------------------------
# QB production features
# ---------------------------------------------------------------------------

def add_qb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add passing production features for QB rows.

    All four features are set to NaN for WR and TE rows — never 0.

    Features added:
        completion_pct   : col_pass_completions / col_pass_attempts.
                           NaN if col_pass_attempts is null or 0.
        yards_per_attempt: col_pass_yards / col_pass_attempts.
                           NaN if col_pass_attempts is null or 0.
        td_int_ratio     : col_pass_tds / col_pass_ints.
                           Capped at 10.0 when col_pass_ints == 0 and
                           col_pass_tds is not null. NaN if col_pass_tds
                           is null.
        pass_yards_flag  : True if col_pass_yards is null, else False.
                           Applied to all rows (not QB-only).

    Args:
        df: Dataframe with is_QB, is_WR, is_TE dummies already added.

    Returns:
        Copy of df with QB feature columns appended.
    """
    df = df.copy()
    non_qb_mask = df["is_QB"] != 1

    att_safe = df["col_pass_attempts"].replace(0.0, float("nan"))

    cmp_pct = df["col_pass_completions"] / att_safe
    cmp_pct[non_qb_mask] = float("nan")
    df["completion_pct"] = cmp_pct

    ypa = df["col_pass_yards"] / att_safe
    ypa[non_qb_mask] = float("nan")
    df["yards_per_attempt"] = ypa

    # td_int_ratio: avoid inf on zero-int seasons by capping at 10.0
    zero_ints = df["col_pass_ints"] == 0
    ints_safe = df["col_pass_ints"].replace(0.0, float("nan"))
    td_int = df["col_pass_tds"] / ints_safe
    cap_mask = zero_ints & df["col_pass_tds"].notna()
    td_int = td_int.where(~cap_mask, 10.0)
    td_int[non_qb_mask] = float("nan")
    df["td_int_ratio"] = td_int

    # pass_yards_flag — all rows, not QB-only
    df["pass_yards_flag"] = df["col_pass_yards"].isnull().fillna(False)

    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_skill_pass_features(
    train_path: pathlib.Path = TRAIN_PATH,
    holdout_path: pathlib.Path = HOLDOUT_PATH,
    train_out: pathlib.Path = TRAIN_OUT,
    holdout_out: pathlib.Path = HOLDOUT_OUT,
) -> None:
    """Run the full skill_pass feature engineering pipeline.

    Loads train and holdout processed CSVs, applies all feature
    transformations, fits z-score params on train only (saved to
    scaler_params.json), and writes two output feature CSVs.

    Args:
        train_path: Path to skill_pass_train.csv.
        holdout_path: Path to skill_pass_holdout.csv.
        train_out: Destination path for skill_pass_features_train.csv.
        holdout_out: Destination path for skill_pass_features_holdout.csv.
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

    # Z-score: fit on train only, apply to both
    params = fit_scaler_params(train, POSITION_GROUP)
    save_scaler_params(params)
    print(f"  Scaler params saved for {len(params)} features.")
    train = apply_z_scores(train, params, POSITION_GROUP)
    holdout = apply_z_scores(holdout, params, POSITION_GROUP)

    # Position-specific features + validation + write
    for label, df, out_path in [
        ("skill_pass_train", train, train_out),
        ("skill_pass_holdout", holdout, holdout_out),
    ]:
        df = add_position_dummies(df)
        df = add_wr_te_features(df)
        df = add_qb_features(df)
        validate_outcomes(df, label)
        df.to_csv(out_path, index=False)
        print(f"  Wrote {out_path}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for skill_pass feature engineering."""
    print("Stage 2 — Feature Engineering (skill_pass)")
    build_skill_pass_features()
    print("  Done.")


if __name__ == "__main__":
    main()
