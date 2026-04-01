"""
Compute Sleeper Scores (0–100) for all skill-position prospects.

Scores ALL players in both feature tables (train + holdout combined) using
the fitted XGBoost models. Players with missing_outcome=True still receive a
Sleeper Score — they have pre-draft features and model predictions even if
their career outcome is unknown.

Score components and weights:
  surplus_value_percentile : 40%   — predicted value vs. round peers
  pro_bowl_probability     : 20%   — logistic function of predicted AV
  peak_value_percentile    : 20%   — percentile rank of predicted AV
  availability_factor      : 20%   — estimated games played proxy

All components are clipped to [0.0, 1.0] before the weighted sum.
Final sleeper_score is clipped to [0, 100].

Outputs:
  data/features/skill_pass_scored.csv
  data/features/skill_run_scored.csv

Usage:
    python -m src.scores.sleeper_score
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from src.modeling.xgboost_model import XGBoostModel

FEATURES_DIR = pathlib.Path("data/features")
MODELS_DIR = pathlib.Path("models/saved")

FEATURE_FILES = {
    "skill_pass": {
        "train":   FEATURES_DIR / "skill_pass_features_train.csv",
        "holdout": FEATURES_DIR / "skill_pass_features_holdout.csv",
    },
    "skill_run": {
        "train":   FEATURES_DIR / "skill_run_features_train.csv",
        "holdout": FEATURES_DIR / "skill_run_features_holdout.csv",
    },
}

MODEL_FILES = {
    "skill_pass": MODELS_DIR / "xgboost_skill_pass.pkl",
    "skill_run":  MODELS_DIR / "xgboost_skill_run.pkl",
}

OUTPUT_FILES = {
    "skill_pass": FEATURES_DIR / "skill_pass_scored.csv",
    "skill_run":  FEATURES_DIR / "skill_run_scored.csv",
}

_BOOL_MAP = {"True": True, "False": False, True: True, False: False}
_FLAG_COLS = [
    "missing_outcome", "missing_combine", "missing_height_weight",
    "missing_age", "missing_college_stats",
]

# Sleeper Score component weights — sum to 1.0.
W_SURPLUS   = 0.40
W_PRO_BOWL  = 0.20
W_PEAK      = 0.20
W_AVAIL     = 0.20

# Pro Bowl probability heuristic parameter.
# 1 / (1 + exp(-0.05 * (pred - 40))) gives ~50% probability at pred=40 AV.
PRO_BOWL_K   = 0.05
PRO_BOWL_MID = 40.0

# Availability factor denominator — games in ~10 NFL seasons.
AVAIL_DENOMINATOR = 160.0

# Rough proxy for games-played from predicted AV.
AVAIL_AV_MULTIPLIER = 2.5


def _load_and_combine(group: str) -> pd.DataFrame:
    """Load train and holdout feature CSVs for a position group and concatenate.

    Args:
        group: Position group key ('skill_pass' or 'skill_run').

    Returns:
        Combined DataFrame of all players (train + holdout), index reset.
    """
    frames = []
    for split, path in FEATURE_FILES[group].items():
        df = pd.read_csv(path, low_memory=False)
        for col in _FLAG_COLS:
            if col in df.columns:
                df[col] = df[col].map(_BOOL_MAP).astype(bool)
        df["_split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _percentile_rank_within_group(series: pd.Series, group_labels: pd.Series) -> pd.Series:
    """Compute percentile rank [0.0, 1.0] within each position group.

    RB surplus values are ranked only against other RBs; QB/WR/TE surplus
    values are ranked only against other skill_pass players.

    Args:
        series: Numeric values to rank.
        group_labels: Group identifier for each row (position_group column).

    Returns:
        Series of percentile ranks in [0.0, 1.0], same index as series.
    """
    ranks = series.copy().astype(float)
    for group in group_labels.unique():
        mask = group_labels == group
        ranks[mask] = series[mask].rank(pct=True, method="average")
    return ranks


def compute_surplus_value(
    df: pd.DataFrame,
    preds: np.ndarray,
) -> tuple[pd.Series, pd.Series]:
    """Compute surplus_value and surplus_value_percentile.

    surplus_value = predicted_av - median(predicted_av for same draft_round
    and position_group). Positive surplus = model expects player to outperform
    round peers.

    surplus_value_percentile = percentile rank of surplus_value within
    position_group. Range [0.0, 1.0].

    Args:
        df: Full scored dataframe with position_group and draft_round columns.
        preds: 1-D array of weighted_av predictions aligned with df.

    Returns:
        Tuple of (surplus_value Series, surplus_value_percentile Series).
    """
    pred_series = pd.Series(preds, index=df.index, name="weighted_av_pred")

    # Compute median predicted AV by (position_group, draft_round) peer group.
    peer_key = df[["position_group", "draft_round"]].copy()
    peer_key["pred"] = pred_series.values
    peer_median = peer_key.groupby(["position_group", "draft_round"])["pred"].transform("median")

    surplus = pred_series - peer_median
    surplus_pct = _percentile_rank_within_group(surplus, df["position_group"])
    return surplus, surplus_pct


def compute_pro_bowl_probability(preds: np.ndarray) -> np.ndarray:
    """Compute Pro Bowl probability using a logistic heuristic.

    Formula: 1 / (1 + exp(-0.05 * (weighted_av_pred - 40)))
    Approximately 50% probability at 40 AV; approaches 1.0 at ~100 AV.
    This is a heuristic — no separate classifier is trained.

    Args:
        preds: 1-D array of predicted weighted_av values.

    Returns:
        1-D array of Pro Bowl probabilities in [0.0, 1.0].
    """
    return 1.0 / (1.0 + np.exp(-PRO_BOWL_K * (preds - PRO_BOWL_MID)))


def compute_sleeper_scores(df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    """Compute all Sleeper Score components and the final score.

    All four components are clipped to [0.0, 1.0] before the weighted sum.
    The final sleeper_score is clipped to [0, 100].

    Components:
        weighted_av_pred        : XGBoost raw prediction (not clipped)
        surplus_value           : pred minus median peer pred (not clipped)
        surplus_value_percentile: percentile rank of surplus within group [0, 1]
        pro_bowl_probability    : logistic heuristic [0, 1]
        peak_value_percentile   : percentile rank of pred within group [0, 1]
        availability_factor     : (pred * 2.5) / 160 clipped to [0, 1]
        sleeper_score           : weighted sum * 100, clipped to [0, 100]

    Args:
        df: Full combined feature dataframe.
        preds: 1-D array of XGBoost predictions aligned with df.

    Returns:
        df with all component columns and sleeper_score appended.
    """
    df = df.copy()
    df["weighted_av_pred"] = preds

    # Surplus value
    surplus, surplus_pct = compute_surplus_value(df, preds)
    df["surplus_value"] = surplus
    df["surplus_value_percentile"] = surplus_pct.clip(0.0, 1.0)

    # Pro Bowl probability
    df["pro_bowl_probability"] = np.clip(compute_pro_bowl_probability(preds), 0.0, 1.0)

    # Peak value percentile
    df["peak_value_percentile"] = _percentile_rank_within_group(
        pd.Series(preds, index=df.index), df["position_group"]
    ).clip(0.0, 1.0)

    # Availability factor
    nfl_games_est = preds * AVAIL_AV_MULTIPLIER
    df["availability_factor"] = np.clip(nfl_games_est / AVAIL_DENOMINATOR, 0.0, 1.0)

    # Assertions: all components must be in [0, 1]
    for col in ("surplus_value_percentile", "pro_bowl_probability",
                "peak_value_percentile", "availability_factor"):
        out_of_range = ((df[col] < 0.0) | (df[col] > 1.0)).sum()
        assert out_of_range == 0, (
            f"{col} has {out_of_range} value(s) outside [0.0, 1.0] — clip failed."
        )

    # Weighted sum
    df["sleeper_score"] = (
        W_SURPLUS  * df["surplus_value_percentile"] +
        W_PRO_BOWL * df["pro_bowl_probability"] +
        W_PEAK     * df["peak_value_percentile"] +
        W_AVAIL    * df["availability_factor"]
    ) * 100.0
    df["sleeper_score"] = df["sleeper_score"].clip(0.0, 100.0)

    return df


def score_group(group: str) -> None:
    """Run the full Sleeper Score pipeline for one position group.

    Loads train + holdout feature CSVs, loads the fitted XGBoost model,
    scores all players, validates bounds, and writes the output CSV.

    Args:
        group: Position group key ('skill_pass' or 'skill_run').
    """
    print(f"  Scoring {group} …")
    df = _load_and_combine(group)
    print(f"    Players (train + holdout): {len(df):,}")

    model = XGBoostModel.load(MODEL_FILES[group])
    preds = model.predict(df)

    df = compute_sleeper_scores(df, preds)

    # Gate: Sleeper Scores must be in [0, 100]
    out_of_bounds = ((df["sleeper_score"] < 0) | (df["sleeper_score"] > 100)).sum()
    assert out_of_bounds == 0, f"sleeper_score out of [0, 100] for {out_of_bounds} players."

    # Gate: all four component columns must have zero nulls
    component_cols = [
        "surplus_value_percentile", "pro_bowl_probability",
        "peak_value_percentile", "availability_factor",
    ]
    for col in component_cols:
        n_null = df[col].isnull().sum()
        assert n_null == 0, f"{col} has {n_null} null value(s) in scored output."

    out_path = OUTPUT_FILES[group]
    df.to_csv(out_path, index=False)

    score_min = df["sleeper_score"].min()
    score_max = df["sleeper_score"].max()
    score_med = df["sleeper_score"].median()
    print(f"    sleeper_score: min={score_min:.1f}  median={score_med:.1f}  max={score_max:.1f}")
    print(f"    Wrote {out_path}  ({len(df):,} rows)")


def main() -> None:
    """Entry point: compute and write Sleeper Scores for both position groups."""
    print("Stage 3 — Sleeper Score computation")
    for group in ("skill_pass", "skill_run"):
        score_group(group)
    print("  Done.")


if __name__ == "__main__":
    main()
