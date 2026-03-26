"""
Abstract base class and feature-allowlist utilities for all DraftSleeper models.

All models target weighted_av. Rows with missing_outcome=True are excluded
from training but retained for Sleeper Score computation.
"""

from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature allowlist constants
# ---------------------------------------------------------------------------

# Outcome columns — contain career information that would cause leakage.
OUTCOME_COLS: frozenset[str] = frozenset({
    "weighted_av",
    "nfl_games",
    "pro_bowls",
    "all_pro",
    "seasons_started",
    "hof",
    "draft_team_av",
})

# Identifier / temporal columns — not predictive features.
ID_COLS: frozenset[str] = frozenset({
    "pfr_id",
    "cfb_id",
    "gsis_id",
    "player_name",
    "position",
    "position_group",
    "draft_year",
    "draft_team",
    "college",
})

# Missingness metadata flags — describe data quality, not player ability.
META_FLAG_COLS: frozenset[str] = frozenset({
    "missing_outcome",
    "missing_combine",
    "missing_height_weight",
    "missing_age",
    "missing_college_stats",
})

# Union of all columns that must never appear in a feature matrix.
EXCLUDE_ALWAYS: frozenset[str] = OUTCOME_COLS | ID_COLS | META_FLAG_COLS

# Binary position/pick indicators that are already in [0, 1] and do not
# need z-score normalization — included in the Ridge feature matrix directly.
BINARY_INDICATORS: frozenset[str] = frozenset({
    "is_QB",
    "is_WR",
    "is_TE",
    "is_early_pick",
    "is_day_3_pick",
})


# ---------------------------------------------------------------------------
# Feature allowlist builders
# ---------------------------------------------------------------------------

def build_xgb_features(df_columns: list[str]) -> list[str]:
    """Build the XGBoost feature allowlist from a dataframe's column names.

    Includes all columns that are:
        - Not in EXCLUDE_ALWAYS (no outcomes, identifiers, or meta flags)
        - Not a z-score column (suffix _z) — XGBoost does not need normalization
        - Not a Stage 2 feature flag (suffix _flag)

    Args:
        df_columns: Full list of column names in the feature dataframe.

    Returns:
        Ordered list of column names for the XGBoost feature matrix.
    """
    return [
        c for c in df_columns
        if c not in EXCLUDE_ALWAYS
        and not c.endswith("_z")
        and not c.endswith("_flag")
    ]


def build_ridge_features(df_columns: list[str]) -> list[str]:
    """Build the Ridge regression feature allowlist from a dataframe's column names.

    Ridge requires normalized inputs, so only z-score columns and binary
    indicators (already in [0, 1]) are included. Raw continuous features are
    excluded to prevent scale-sensitive distortion.

    Args:
        df_columns: Full list of column names in the feature dataframe.

    Returns:
        Ordered list of column names for the Ridge feature matrix.
    """
    z_cols = [c for c in df_columns if c.endswith("_z") and c not in EXCLUDE_ALWAYS]
    binary = [c for c in df_columns if c in BINARY_INDICATORS]
    return z_cols + binary


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseModel(ABC):
    """Abstract interface for DraftSleeper weighted_av regression models.

    All subclasses must implement fit, predict, evaluate, save, and load.
    The evaluate() return dict must use the keys 'rmse', 'mae', 'spearman'
    to ensure compatibility with evaluate_all.py.
    """

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """Fit the model on training data.

        Args:
            X_train: Feature matrix. Rows with missing_outcome=True must be
                     excluded by the caller before passing here.
            y_train: Target vector of weighted_av values, aligned with X_train.
            **kwargs: Subclass-specific arguments (e.g. draft_year for XGBoost).
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate weighted_av predictions for all rows in X.

        Args:
            X: Feature matrix. NaN-tolerant subclasses (XGBoost) pass NaN as-is;
               others impute before calling.

        Returns:
            1-D float array of predicted weighted_av values, same length as X.
        """

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> dict[str, float]:
        """Compute evaluation metrics on a labelled dataset.

        Args:
            X: Feature matrix.
            y_true: Ground truth weighted_av values aligned with X.

        Returns:
            Dict with keys 'rmse' (float), 'mae' (float), 'spearman' (float).
        """

    @abstractmethod
    def save(self, path: pathlib.Path) -> None:
        """Serialize the fitted model to disk using joblib.

        Args:
            path: Destination file path (e.g., models/saved/xgboost_skill_pass.pkl).
        """

    @classmethod
    @abstractmethod
    def load(cls, path: pathlib.Path) -> "BaseModel":
        """Deserialize a fitted model from disk.

        Args:
            path: Path to a joblib-serialized model file.

        Returns:
            A fitted instance of the subclass.
        """
