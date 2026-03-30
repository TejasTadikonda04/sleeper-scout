"""
XGBoost regression model for weighted_av prediction.

Uses raw (non-normalized) feature columns — XGBoost handles NaN natively
so no imputation is applied. Hyperparameters are tuned with Optuna (30 trials)
using a time-based inner split: 2000–2016 inner train, 2017–2019 inner val.
Spearman rank correlation is the tuning objective (not RMSE) because the
downstream Sleeper Score is rank-based, not scale-sensitive.

A monotone decreasing constraint is applied to draft_pick: higher pick number
must predict lower (or equal) weighted_av. The constraint tuple is built
programmatically from the feature list to stay in sync with column order.

Serialized via joblib to models/saved/.
"""

from __future__ import annotations

import pathlib

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.modeling.base import BaseModel

# Suppress Optuna's per-trial output; progress is printed at a higher level.
optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_TRIALS = 30
INNER_TRAIN_MAX_YEAR = 2016
INNER_VAL_MIN_YEAR = 2017
INNER_VAL_MAX_YEAR = 2019


class XGBoostModel(BaseModel):
    """XGBoost regressor for weighted_av prediction.

    Accepts NaN values natively via XGBoost's built-in missing-value handling.
    Applies a monotone decreasing constraint on draft_pick. Hyperparameters
    are tuned with Optuna on an internal time-based split; the model is then
    refit on the full training set using the best parameters found.
    """

    def __init__(self, features: list[str]) -> None:
        """Initialise the XGBoost model.

        Args:
            features: Ordered list of column names for the feature matrix.
                      Must remain consistent across fit() and predict() calls.
                      Should be the output of build_xgb_features().
        """
        self.features = features
        self._model: xgb.XGBRegressor | None = None
        self._best_params: dict = {}

    def _monotone_constraints(self) -> tuple[int, ...]:
        """Build the monotone constraint tuple aligned to self.features.

        draft_pick receives a -1 constraint (monotone decreasing): higher pick
        number must predict ≤ weighted_av compared to lower pick numbers.
        All other features are unconstrained (0).

        The tuple is built programmatically so it stays in sync with the
        feature list regardless of column order or group.

        Returns:
            Tuple of ints, one per feature, in the same order as self.features.
        """
        return tuple(-1 if f == "draft_pick" else 0 for f in self.features)

    def _tune(
        self,
        X_inner: pd.DataFrame,
        y_inner: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Run Optuna hyperparameter search optimising Spearman rank correlation.

        Spearman is preferred over RMSE as the tuning objective because the
        Sleeper Score depends on rank ordering of players, not absolute error.
        Minimising negative Spearman is equivalent to maximising Spearman.

        Args:
            X_inner: Inner training features (draft_year ≤ 2016).
            y_inner: Inner training targets.
            X_val: Inner validation features (draft_year 2017–2019).
            y_val: Inner validation targets.

        Returns:
            Dict of best hyperparameter values found by Optuna.
        """
        constraints = self._monotone_constraints()

        def objective(trial: optuna.Trial) -> float:
            """Single Optuna trial: fit XGBoost, return negative Spearman."""
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "objective": "reg:squarederror",
                "monotone_constraints": constraints,
                "verbosity": 0,
            }
            m = xgb.XGBRegressor(**params)
            m.fit(X_inner, y_inner)
            preds = m.predict(X_val)
            corr = spearmanr(y_val, preds).statistic
            # Optuna minimises — return negative Spearman to maximise it.
            return float(-corr)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        return study.best_params

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        draft_year: pd.Series | None = None,
        **kwargs,
    ) -> None:
        """Tune hyperparameters on an inner time-split, then fit on full train.

        The inner split uses draft_year for a temporal boundary:
            inner train : draft_year ≤ 2016
            inner val   : draft_year 2017–2019
        Hold-out data (draft_year ≥ 2020) is never seen during tuning or fitting.

        Args:
            X_train: Training feature matrix. missing_outcome rows must be
                     excluded by the caller before passing here.
            y_train: Target weighted_av values aligned with X_train.
            draft_year: Series of draft_year values aligned with X_train, used
                        for the time-based inner split. If None, falls back to
                        an 80/20 positional split on the training index.
            **kwargs: Ignored.
        """
        X = X_train[self.features]

        if draft_year is not None:
            inner_mask = draft_year <= INNER_TRAIN_MAX_YEAR
            val_mask = (draft_year >= INNER_VAL_MIN_YEAR) & (draft_year <= INNER_VAL_MAX_YEAR)
        else:
            n = len(X)
            cutoff = int(n * 0.8)
            inner_mask = pd.Series(
                [True] * cutoff + [False] * (n - cutoff), index=X.index
            )
            val_mask = ~inner_mask

        X_inner, y_inner = X[inner_mask], y_train[inner_mask]
        X_val, y_val = X[val_mask], y_train[val_mask]

        print(f"    Optuna: inner train={len(X_inner):,}  inner val={len(X_val):,}")
        self._best_params = self._tune(X_inner, y_inner, X_val, y_val)
        print(f"    Best params: {self._best_params}")

        # Retrain on the full training set with the best hyperparameters.
        self._model = xgb.XGBRegressor(
            **self._best_params,
            objective="reg:squarederror",
            monotone_constraints=self._monotone_constraints(),
            verbosity=0,
        )
        self._model.fit(X, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict weighted_av. NaN values are handled natively by XGBoost.

        XGBoost routes NaN values to the default direction learned during
        training. Imputing with 0 or column mean before calling predict()
        would corrupt this routing and produce worse out-of-sample performance.

        Args:
            X: Feature matrix. NaN values are passed as-is — do not impute.

        Returns:
            1-D float array of predicted weighted_av values.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._model.predict(X[self.features])

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

    def feature_importances(self) -> dict[str, float]:
        """Return gain-based feature importances from the fitted model.

        Gain-based importance measures the average improvement in the loss
        function across all splits where a feature is used — more informative
        than frequency-based importance for sparse features.

        Returns:
            Dict mapping feature name to gain importance score.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._model.get_booster().get_score(importance_type="gain")

    def save(self, path: pathlib.Path) -> None:
        """Serialize the fitted model to disk using joblib.

        Args:
            path: Destination path (.pkl).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: pathlib.Path) -> "XGBoostModel":
        """Deserialize a fitted XGBoostModel from disk.

        Args:
            path: Path to a joblib-serialized XGBoostModel file.

        Returns:
            Fitted XGBoostModel instance.
        """
        return joblib.load(path)
