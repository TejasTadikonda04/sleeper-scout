"""
Compute Sleeper Scores (0–100) for 2026 NFL draft skill-position prospects.

Scores all 2026 prospects in both position groups using pre-trained XGBoost
models. Models are not retrained. Players with missing_outcome=True still
receive a Sleeper Score — they have pre-draft features and model predictions
even if their career outcome is unknown.

Score components and weights:
  contract_surplus_percentile : 50%  — predicted value vs. contract cost
  upside_percentile           : 30%  — std deviations above round-peer mean
  round_value_percentile      : 20%  — predicted AV vs. round-peer median

All components are clipped to [0.0, 1.0] before the weighted sum.
Final sleeper_score is clipped to [0, 100].

Outputs:
  data/features/prospects/skill_pass_2026_scored.csv
  data/features/prospects/skill_run_2026_scored.csv

Usage:
    python -m src.scores.sleeper_score
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from src.modeling.xgboost_model import XGBoostModel
from src.feature_engineering.shared_features import (
    add_combine_composites,
    add_draft_features,
    apply_z_scores,
    load_scaler_params,
)
from src.feature_engineering.engineering_skill_pass import (
    add_position_dummies,
    add_wr_te_features,
    add_qb_features,
)
from src.feature_engineering.engineering_skill_run import add_rb_features

_BOOL_MAP = {"True": True, "False": False, True: True, False: False}
_FLAG_COLS = [
    "missing_outcome", "missing_combine", "missing_height_weight",
    "missing_age", "missing_college_stats",
]

# Maps raw position values to position_group keys used by the models.
_POSITION_GROUP_MAP = {
    "QB": "skill_pass",
    "WR": "skill_pass",
    "TE": "skill_pass",
    "RB": "skill_run",
}

# Combine drill columns — used to derive missing_combine flag.
_COMBINE_COLS = [
    "forty_yard", "vertical_jump", "broad_jump",
    "bench_reps", "cone_drill", "shuttle",
]

# Input / output paths
PROSPECTS_2026_PATH   = pathlib.Path("data/raw/2026_prospects.csv")
MARKET_RATES_PATH     = pathlib.Path("data/raw/position_market_rates.json")

OUTPUT_DIR = pathlib.Path("data/features/prospects")

OUTPUT_FILES = {
    "skill_pass": OUTPUT_DIR / "skill_pass_2026_scored.csv",
    "skill_run":  OUTPUT_DIR / "skill_run_2026_scored.csv",
}

MODEL_FILES = {
    "skill_pass": pathlib.Path("models/saved/xgboost_skill_pass.pkl"),
    "skill_run":  pathlib.Path("models/saved/xgboost_skill_run.pkl"),
}

# 2026 NFL salary cap (projected, used for dollar conversions).
CAP_VALUE_USD = 301_200_000

# Sleeper Score component weights — must sum to 1.0.
W_CONTRACT_SURPLUS = 0.50
W_UPSIDE           = 0.30
W_ROUND_VALUE      = 0.20


def _load_and_combine(group: str) -> pd.DataFrame:
    """Load 2026 prospects for one position group and apply feature engineering.

    Reads data/raw/2026_prospects.csv, derives position_group from position,
    filters to the requested group, adds any missing flag columns, then runs
    the full feature engineering pipeline using pre-fitted scaler params
    (no refitting on 2026 data).

    Args:
        group: Position group key ('skill_pass' or 'skill_run').

    Returns:
        DataFrame of 2026 prospects with all model features present, index reset.

    Raises:
        FileNotFoundError: If 2026_prospects.csv or scaler_params.json does not exist.
        ValueError: If no rows exist for the requested position_group.
    """
    df = pd.read_csv(PROSPECTS_2026_PATH, low_memory=False)

    # Derive position_group and filter to skill positions for this group.
    df["position_group"] = df["position"].map(_POSITION_GROUP_MAP)
    df = df[df["position_group"] == group].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"No 2026 prospects found for position_group='{group}'."
        )

    # Restore boolean dtypes for any flag columns already present.
    for col in _FLAG_COLS:
        if col in df.columns:
            df[col] = df[col].map(_BOOL_MAP).astype(bool)

    # Derive draft_round and draft_pick from projected_draft_pick.
    # projected_draft_pick is the pre-draft pick projection; actual picks
    # don't exist yet for 2026 prospects.
    # Round = ceil(pick / 32), capped at [1, 7]. NaN → round 7 (late/undrafted).
    import math
    proj = df["projected_draft_pick"].fillna(224.0)  # ~round 7 midpoint
    df["draft_pick"] = proj.round().astype(int)
    df["draft_round"] = proj.apply(
        lambda x: min(7, max(1, math.ceil(x / 32)))
    )

    # Add flag columns not present in the raw prospects CSV.
    if "missing_outcome" not in df.columns:
        df["missing_outcome"] = True
    if "missing_combine" not in df.columns:
        df["missing_combine"] = df[_COMBINE_COLS].isnull().all(axis=1)
    if "low_confidence_mock" not in df.columns:
        df["low_confidence_mock"] = False

    # Feature engineering — use pre-fitted scaler params, no refitting.
    df = add_draft_features(df)
    df = add_combine_composites(df)
    scaler_params = load_scaler_params()
    df = apply_z_scores(df, scaler_params, group)

    # Position-specific features.
    if group == "skill_pass":
        df = add_position_dummies(df)
        df = add_wr_te_features(df)
        df = add_qb_features(df)
    elif group == "skill_run":
        df = add_rb_features(df)

    return df


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


def _load_market_rates() -> dict[str, dict[str, float]]:
    """Load position market rate tiers from JSON.

    Reads data/raw/position_market_rates.json. Values are fractions of
    CAP_VALUE_USD. Tiers correspond to within-position weighted_av_pred
    quartiles: elite = top 25%, good = 25–50%, average = 50–75%,
    below = bottom 25%.

    Returns:
        Nested dict: position → tier → cap_fraction.

    Raises:
        FileNotFoundError: If position_market_rates.json does not exist.
    """
    import json
    with open(MARKET_RATES_PATH) as f:
        return json.load(f)


def _assign_av_tier(
    pred_series: pd.Series,
    position_series: pd.Series,
) -> pd.Series:
    """Assign each prospect to a market rate tier based on predicted AV quartile.

    Boundaries are computed within each raw position so that a QB's tier
    is relative to other QBs, not all skill_pass players.

    Args:
        pred_series: Series of predicted weighted_av values.
        position_series: Series of raw position strings (QB, WR, TE, RB).

    Returns:
        Series of tier labels ('elite', 'good', 'average', 'below').
    """
    tiers = pd.Series("below", index=pred_series.index, dtype=str)
    for pos in position_series.unique():
        mask = position_series == pos
        vals = pred_series[mask]
        if len(vals) < 4:
            tiers[mask] = "average"
            continue
        q25, q50, q75 = vals.quantile([0.25, 0.50, 0.75])
        tier_labels = pd.cut(
            vals,
            bins=[-np.inf, q25, q50, q75, np.inf],
            labels=["below", "average", "good", "elite"],
        )
        tiers[mask] = tier_labels.astype(str)
    return tiers


def compute_surplus_value(
    df: pd.DataFrame,
    preds: np.ndarray,
    market_rates: dict[str, dict[str, float]],
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute contract surplus and upside components for 2026 prospects.

    Contract surplus:
        performance_value_usd = market_rate_for_tier * CAP_VALUE_USD
        contract_cost_usd     = projected_contract_value from the CSV
        contract_surplus_usd  = performance_value_usd - contract_cost_usd

    projected_contract_value is already present in 2026_prospects.csv.
    No separate contract lookup table is needed.

    Upside:
        upside_score = (pred - round_peer_mean) / round_peer_std
    Ranked within raw position.

    Args:
        df: Prospect dataframe with position, position_group, draft_round,
            and projected_contract_value columns.
        preds: 1-D array of weighted_av predictions aligned with df.
        market_rates: Dict from _load_market_rates().

    Returns:
        Tuple (av_tier, performance_value, contract_cost,
               contract_surplus_percentile, upside_percentile).
        All Series share df's index.
    """
    pred_series = pd.Series(preds, index=df.index)

    # Performance value from position tier
    av_tier = _assign_av_tier(pred_series, df["position"])
    performance_value = pd.Series(0.0, index=df.index)
    for pos in df["position"].unique():
        rates = market_rates.get(pos, {"elite": 0.05, "good": 0.03,
                                       "average": 0.02, "below": 0.01})
        for tier in ("elite", "good", "average", "below"):
            mask = (df["position"] == pos) & (av_tier == tier)
            if mask.any():
                performance_value[mask] = rates[tier] * CAP_VALUE_USD

    # Contract cost read directly from the CSV
    contract_cost = df["projected_contract_value"].fillna(0.0)

    # Contract surplus percentile
    contract_surplus = performance_value - contract_cost
    contract_surplus_pct = _percentile_rank_within_group(
        contract_surplus, df["position"]
    ).clip(0.0, 1.0)

    # Upside percentile
    peer_stats = (
        df[["position_group", "draft_round"]]
        .assign(pred=preds)
        .groupby(["position_group", "draft_round"])["pred"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "peer_mean", "std": "peer_std"})
    )
    df_temp = df[["position_group", "draft_round"]].join(
        peer_stats, on=["position_group", "draft_round"]
    )
    peer_std_safe = df_temp["peer_std"].replace(0, np.nan).fillna(1.0)
    upside_score = (
        (pred_series.values - df_temp["peer_mean"].values)
        / peer_std_safe.values
    )
    upside_pct = _percentile_rank_within_group(
        pd.Series(upside_score, index=df.index), df["position"]
    ).clip(0.0, 1.0)

    return av_tier, performance_value, contract_cost, contract_surplus_pct, upside_pct


def compute_sleeper_scores(
    df: pd.DataFrame,
    preds: np.ndarray,
    market_rates: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Compute all scoring columns for 2026 prospects.

    Components (weights sum to 1.0):
        contract_surplus_percentile : 0.50
        upside_percentile           : 0.30
        round_value_percentile      : 0.20

    All components clipped to [0, 1]. sleeper_score clipped to [0, 100].

    Also computes draft_window_label — a plain-English GM recommendation.

    Args:
        df: 2026 prospect dataframe for one position group.
        preds: 1-D array of XGBoost predictions aligned with df.
        market_rates: Dict from _load_market_rates().

    Returns:
        df with all scoring columns appended.
    """
    df = df.copy()
    df["weighted_av_pred"] = preds

    # Contract surplus and upside
    av_tier, perf_value, contract_cost, contract_surplus_pct, upside_pct = (
        compute_surplus_value(df, preds, market_rates)
    )
    df["av_tier"]                    = av_tier
    df["performance_value_usd"]      = perf_value
    df["contract_cost_usd"]          = contract_cost
    df["contract_surplus_usd"]       = perf_value - contract_cost
    df["contract_surplus_percentile"] = contract_surplus_pct
    df["upside_percentile"]           = upside_pct

    # Round value percentile
    pred_series = pd.Series(preds, index=df.index)
    peer_median = (
        df[["position_group", "draft_round"]]
        .assign(pred=preds)
        .groupby(["position_group", "draft_round"])["pred"]
        .transform("median")
    )
    round_value = pred_series - pd.Series(peer_median.values, index=df.index)
    df["round_value_percentile"] = _percentile_rank_within_group(
        round_value, df["position_group"]
    ).clip(0.0, 1.0)

    # Assertions
    for col in ("contract_surplus_percentile", "upside_percentile",
                "round_value_percentile"):
        bad = ((df[col] < 0.0) | (df[col] > 1.0)).sum()
        assert bad == 0, f"{col} has {bad} value(s) outside [0, 1]."

    # Weighted sum
    df["sleeper_score"] = (
        W_CONTRACT_SURPLUS * df["contract_surplus_percentile"] +
        W_UPSIDE           * df["upside_percentile"] +
        W_ROUND_VALUE      * df["round_value_percentile"]
    ) * 100.0
    df["sleeper_score"] = df["sleeper_score"].clip(0.0, 100.0)

    # Draft window label
    def _make_label(row: pd.Series) -> str:
        score  = row["sleeper_score"]
        rnd    = int(row["draft_round"]) if pd.notna(row["draft_round"]) else 0
        low_conf = bool(row.get("low_confidence_mock", False))
        if score >= 75 and rnd >= 3:
            label = f"High-value sleeper — target in round {rnd}"
        elif score >= 60 and rnd >= 2:
            label = f"Round {rnd} value — monitor board"
        elif score >= 50:
            label = "On the board — situational value"
        else:
            label = "No strong signal at current ADP"
        if low_conf:
            label += " (mock rank uncertainty: \u00b11\u20132 rounds)"
        return label

    df["draft_window_label"] = df.apply(_make_label, axis=1)

    return df


def score_group(group: str) -> None:
    """Score 2026 prospects for one position group and write the output CSV.

    Loads 2026_prospects.csv filtered to the given position group, loads
    the pre-trained XGBoost model (no retraining), scores all prospects,
    validates output bounds, and writes to data/features/prospects/.

    Args:
        group: Position group key ('skill_pass' or 'skill_run').
    """
    print(f"  Scoring 2026 prospects — {group} …")

    market_rates = _load_market_rates()

    df = _load_and_combine(group)
    print(f"    Prospects: {len(df):,}  |  "
          f"positions: {df['position'].value_counts().to_dict()}")

    model = XGBoostModel.load(MODEL_FILES[group])
    preds = model.predict(df)

    df = compute_sleeper_scores(df, preds, market_rates)

    # Gates
    bad_score = ((df["sleeper_score"] < 0) | (df["sleeper_score"] > 100)).sum()
    assert bad_score == 0, f"sleeper_score out of [0, 100] for {bad_score} rows."
    assert df["sleeper_score"].isnull().sum() == 0, "sleeper_score contains nulls."
    assert df["draft_window_label"].isnull().sum() == 0, "draft_window_label contains nulls."

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_FILES[group]
    df.to_csv(out_path, index=False)

    print(f"    sleeper_score  min={df['sleeper_score'].min():.1f}  "
          f"median={df['sleeper_score'].median():.1f}  "
          f"max={df['sleeper_score'].max():.1f}")
    print(f"    contract_surplus_usd  "
          f"min=${df['contract_surplus_usd'].min():,.0f}  "
          f"max=${df['contract_surplus_usd'].max():,.0f}")
    print(f"    Wrote {out_path}  ({len(df):,} rows)")


def main() -> None:
    """Entry point: compute and write Sleeper Scores for both position groups."""
    print("Stage 4 — Sleeper Score computation (2026 prospects)")
    for group in ("skill_pass", "skill_run"):
        score_group(group)
    print("  Done.")


if __name__ == "__main__":
    main()
