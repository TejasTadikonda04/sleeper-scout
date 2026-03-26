"""
Feature engineering logic shared across all position groups.

Provides:
- Stuart's AV-based draft pick value chart (pick_value, is_early_pick, is_day_3_pick)
- Combine composite scores (speed_score, burst_score, agility_score, bmi)
- Z-score normalization: fit on train only, saved to scaler_params.json,
  applied to holdout without refitting
- CSV loading helper that restores boolean flag dtypes after read
- Outcome validation shared by both engineering scripts
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FEATURES_DIR = pathlib.Path("data/features")
SCALER_PARAMS_PATH = FEATURES_DIR / "scaler_params.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns to Z-score normalize.
# Raw combine columns + physical measurements + composite scores.
COLS_TO_NORMALIZE: list[str] = [
    "height_in",
    "weight_lbs",
    "forty_yard",
    "vertical_jump",
    "broad_jump",
    "bench_reps",
    "cone_drill",
    "shuttle",
    "speed_score",
    "burst_score",
    "agility_score",
    "bmi",
]

# Flag columns written by Stage 1 that must be cast back to bool after CSV read.
STAGE1_FLAG_COLS: list[str] = [
    "missing_outcome",
    "missing_combine",
    "missing_height_weight",
    "missing_college_stats",
]

# Stuart's AV-based draft pick value chart.
# Approximates Chase Stuart's expected career AV per draft slot using a
# quadratic decay: 100 * ((263 - pick) / 262) ** 2.
# Picks outside 1–262 receive 0.0 (handled in add_draft_features via .get).
PICK_VALUE_CHART: dict[int, float] = {
    pick: round(100.0 * ((263 - pick) / 262) ** 2, 2)
    for pick in range(1, 263)
}

# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_processed_csv(path: pathlib.Path) -> pd.DataFrame:
    """Load a Stage 1 processed CSV and restore boolean dtypes for flag columns.

    pandas writes True/False as the strings "True"/"False" in CSV. Explicit
    mapping after read restores the correct bool dtype so that .sum() and
    comparisons behave correctly in downstream validation.

    Args:
        path: Path to a processed split CSV from data/processed/.

    Returns:
        DataFrame with Stage 1 flag columns cast to bool.
    """
    df = pd.read_csv(path, low_memory=False)
    bool_map = {"True": True, "False": False, True: True, False: False}
    for col in STAGE1_FLAG_COLS:
        if col in df.columns:
            df[col] = df[col].map(bool_map).astype(bool)
    return df


# ---------------------------------------------------------------------------
# Task 1 — Draft-position features
# ---------------------------------------------------------------------------

def add_draft_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add draft-position and age-derived feature columns.

    Features added:
        pick_value    : Stuart's AV-based draft value for draft_pick (float).
                        0.0 for picks outside the 1–262 chart range or nulls.
        is_early_pick : 1 if draft_pick <= 32 (round 1), else 0 (int).
        is_day_3_pick : 1 if draft_round >= 5 (rounds 5–7), else 0 (int).
        missing_age   : True for every row — age_at_draft cannot be derived
                        from this dataset (no birthdate column present).

    Args:
        df: Processed split dataframe with draft_pick and draft_round columns.

    Returns:
        Copy of df with four draft feature columns appended.
    """
    df = df.copy()
    df["pick_value"] = df["draft_pick"].map(PICK_VALUE_CHART).fillna(0.0)
    df["is_early_pick"] = (df["draft_pick"] <= 32).astype(int)
    df["is_day_3_pick"] = (df["draft_round"] >= 5).astype(int)
    df["missing_age"] = True
    return df


# ---------------------------------------------------------------------------
# Task 2 — Combine composite scores
# ---------------------------------------------------------------------------

def add_combine_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Compute athletic composite scores from raw combine drill values.

    All composites inherit NaN when any required input is null — never
    imputed. XGBoost handles NaN natively; do not fill before model training.

    Composites added:
        speed_score   : (weight_lbs * 200) / (forty_yard ** 4).
                        NaN if forty_yard or weight_lbs is null.
        burst_score   : (vertical_jump + broad_jump) / 2.
                        NaN if either input is null.
        agility_score : (cone_drill + shuttle) / 2. Lower is better — not
                        inverted. NaN if either input is null.
        bmi           : (weight_lbs * 703) / (height_in ** 2).
                        NaN if either input is null.

    Args:
        df: Dataframe with raw combine and physical measurement columns.

    Returns:
        Copy of df with four composite columns appended.
    """
    df = df.copy()
    df["speed_score"] = (df["weight_lbs"] * 200) / (df["forty_yard"] ** 4)
    df["burst_score"] = (df["vertical_jump"] + df["broad_jump"]) / 2
    df["agility_score"] = (df["cone_drill"] + df["shuttle"]) / 2
    df["bmi"] = (df["weight_lbs"] * 703) / (df["height_in"] ** 2)
    return df


# ---------------------------------------------------------------------------
# Z-score normalization
# ---------------------------------------------------------------------------

def fit_scaler_params(
    train_df: pd.DataFrame,
    position_group: str,
) -> dict[str, dict[str, float]]:
    """Compute mean and std for each normalizable column from train data only.

    Only columns present in train_df and listed in COLS_TO_NORMALIZE are
    included. Columns with zero or NaN std are given std=1.0 to avoid
    division by zero.

    Args:
        train_df: Training split dataframe after composites have been added.
        position_group: 'skill_pass' or 'skill_run' — used as JSON key prefix.

    Returns:
        Dict keyed '{position_group}__{col}' with 'mean' and 'std' sub-dicts.
    """
    params: dict[str, dict[str, float]] = {}
    for col in COLS_TO_NORMALIZE:
        if col not in train_df.columns:
            continue
        series = train_df[col].dropna()
        mean = float(series.mean())
        std = float(series.std())
        if std == 0.0 or np.isnan(std):
            std = 1.0
        params[f"{position_group}__{col}"] = {"mean": mean, "std": std}
    return params


def apply_z_scores(
    df: pd.DataFrame,
    params: dict[str, dict[str, float]],
    position_group: str,
) -> pd.DataFrame:
    """Apply pre-fit z-score normalization to each eligible column.

    Adds a '{col}_z' column for every column found in params. NaN values in
    the source column remain NaN after normalization — never imputed.

    Args:
        df: Dataframe to normalize (train or holdout).
        params: Scaler params from fit_scaler_params or load_scaler_params.
        position_group: 'skill_pass' or 'skill_run' — used as JSON key prefix.

    Returns:
        Copy of df with additional '_z' suffix columns appended.
    """
    df = df.copy()
    for col in COLS_TO_NORMALIZE:
        key = f"{position_group}__{col}"
        if col not in df.columns or key not in params:
            continue
        mean = params[key]["mean"]
        std = params[key]["std"]
        df[f"{col}_z"] = (df[col] - mean) / std
    return df


def save_scaler_params(
    new_params: dict[str, dict[str, float]],
    path: pathlib.Path = SCALER_PARAMS_PATH,
) -> None:
    """Merge new_params into scaler_params.json and write.

    If the file does not yet exist it is created. Existing keys from other
    position groups are preserved — only keys in new_params are added or
    overwritten. This allows skill_pass and skill_run to accumulate into the
    same file across independent runs.

    Args:
        new_params: Params dict from fit_scaler_params.
        path: Destination path for the JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update(new_params)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def load_scaler_params(
    path: pathlib.Path = SCALER_PARAMS_PATH,
) -> dict[str, dict[str, float]]:
    """Load scaler parameters from scaler_params.json.

    Args:
        path: Path to scaler_params.json.

    Returns:
        Dict as written by save_scaler_params.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Outcome validation
# ---------------------------------------------------------------------------

def validate_outcomes(df: pd.DataFrame, label: str) -> None:
    """Assert outcome columns are consistent with missing_outcome flags.

    Checks:
        - career_av must be absent from df.columns.
        - weighted_av is null wherever missing_outcome is True.
        - weighted_av is not null wherever missing_outcome is False.

    Args:
        df: Feature dataframe to validate.
        label: Human-readable label for error messages (e.g. 'skill_pass_train').

    Raises:
        ValueError: If career_av is present or missing_outcome flags mismatch.
    """
    if "career_av" in df.columns:
        raise ValueError(f"[{label}] career_av must not be present in any feature file.")

    if "missing_outcome" not in df.columns or "weighted_av" not in df.columns:
        return

    flagged_not_null = (df["missing_outcome"] == True) & df["weighted_av"].notna()
    if flagged_not_null.any():
        n = int(flagged_not_null.sum())
        raise ValueError(
            f"[{label}] {n} row(s) have missing_outcome=True but weighted_av is not null."
        )

    not_flagged_null = (df["missing_outcome"] == False) & df["weighted_av"].isnull()
    if not_flagged_null.any():
        n = int(not_flagged_null.sum())
        raise ValueError(
            f"[{label}] {n} row(s) have missing_outcome=False but weighted_av is null."
        )
