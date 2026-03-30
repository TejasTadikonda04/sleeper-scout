"""
Ridge regression baseline model for weighted_av prediction.

Uses z-score normalized (_z) feature columns plus binary position/pick
indicators. NaN values are imputed with column medians fit on the training
split — the only imputation permitted in the pipeline, and only for Ridge.

Serialized via joblib to models/saved/.
"""

from __future__ import annotations

import pathlib

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.modeling.base import BaseModel


class RidgeModel(BaseModel):
    """Ridge regression baseline for weighted_av prediction.

    Wraps sklearn Ridge with the BaseModel interface. Uses pre-normalized
    (_z) combine features plus binary indicators. NaN values in feature
    columns are imputed with column medians derived from the training split.
    """

    def __init__(self, features: list[str], alpha: float = 1.0) -> None:
        """Initialise the Ridge model.

        Args:
            features: Ordered list of column names to use as the feature matrix.
                      Should be the output of build_ridge_features().
            alpha: Ridge regularisation strength (default 1.0).
        """
        self.features = features
        self.alpha = alpha
        self._model = Ridge(alpha=alpha)
        self._impute_medians: dict[str, float] = {}

    def _prepare_X(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Select feature columns and impute NaN with stored column medians.

        Args:
            X: Input dataframe containing at least self.features columns.
            fit: If True, compute and store imputation medians from X.
                 If False, apply previously stored medians (for holdout).

        Returns:
            DataFrame with self.features columns and no NaN values.
        """
        X = X[self.features].copy()
        for col in self.features:
            if fit:
                self._impute_medians[col] = float(X[col].median())
            fill = self._impute_medians.get(col, 0.0)
            X[col] = X[col].fillna(fill)
        return X

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """Fit Ridge on normalized training features.

        Imputation medians are derived from X_train and stored for later
        application to hold-out data. Never refit on hold-out.

        Args:
            X_train: Training feature matrix with _z columns present.
            y_train: Target weighted_av values aligned with X_train.
            **kwargs: Ignored by Ridge.
        """
        X = self._prepare_X(X_train, fit=True)
        self._model.fit(X, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict weighted_av for all rows in X.

        Args:
            X: Feature matrix. NaN values are imputed with train-fit medians.

        Returns:
            1-D float array of predicted weighted_av values.
        """
        return self._model.predict(self._prepare_X(X, fit=False))

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> dict[str, float]:
        """Compute RMSE, MAE, and Spearman correlation on labelled data.

        Args:
            X: Feature matrix.
            y_true: Ground truth weighted_av values aligned with X.

        Returns:
            Dict with keys 'rmse', 'mae', 'spearman'.
        """
        preds = self.predict(X)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
            "mae": float(mean_absolute_error(y_true, preds)),
            "spearman": float(spearmanr(y_true, preds).statistic),
        }

    def save(self, path: pathlib.Path) -> None:
        """Serialize the fitted model to disk using joblib.

        Args:
            path: Destination path (.pkl).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: pathlib.Path) -> "RidgeModel":
        """Deserialize a fitted RidgeModel from disk.

        Args:
            path: Path to a joblib-serialized RidgeModel file.

        Returns:
            Fitted RidgeModel instance.
        """
        return joblib.load(path)
