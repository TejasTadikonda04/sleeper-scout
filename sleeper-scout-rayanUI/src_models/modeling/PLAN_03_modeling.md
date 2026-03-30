# PLAN_03 — Modeling + Sleeper Scoring

## 1. Stage Goal
Train, tune, and evaluate `weighted_av` regression models for both position
groups, then combine model outputs with supporting outcome signals into a
Sleeper Score (0–100) for every player in the feature tables.

---

## 2. Inputs & Outputs

**Inputs:**
- `data/features/skill_pass_features_train.csv`
- `data/features/skill_pass_features_holdout.csv`
- `data/features/skill_run_features_train.csv`
- `data/features/skill_run_features_holdout.csv`
- `data/features/scaler_params.json`

**Outputs:**
- `models/saved/xgboost_skill_pass.pkl`
- `models/saved/xgboost_skill_run.pkl`
- `models/saved/ridge_baseline_skill_pass.pkl`
- `models/saved/ridge_baseline_skill_run.pkl`
- `models/results/evaluation_summary.csv`
- `models/results/feature_importance.csv`
- `data/features/skill_pass_scored.csv` — feature table with Sleeper Score columns appended
- `data/features/skill_run_scored.csv` — feature table with Sleeper Score columns appended

---

## 3. Go/No-Go Gate

All five of the following must be true before moving to Stage 4:

1. XGBoost outperforms Ridge baseline on Spearman rank correlation on the
   hold-out set for at least one position group.
2. Spearman rank correlation > 0.30 on the hold-out set for at least one
   position group.
3. `weighted_av` predicted values are non-negative for at least 95% of
   hold-out players — negative predictions indicate a model calibration issue.
4. Sleeper Scores are bounded [0, 100] for every player in both scored files.
5. All four Sleeper Score component columns (`surplus_value_percentile`,
   `pro_bowl_probability`, `peak_value_percentile`, `availability_factor`)
   are present and have zero nulls in both scored files.

---

## 4. Task List

**Task 1 — Implement the base model class**
- Description: Define an abstract base class that all model classes inherit
  from, enforcing a consistent interface.
- Output artifact: `src/modeling/base.py`
- Required abstract methods: `fit(X_train, y_train)`, `predict(X)`,
  `evaluate(X, y_true)`, `save(path)`, `load(path)`
- `evaluate()` must return a dict with at minimum:
  `{'rmse': float, 'mae': float, 'spearman': float}`
- Pitfall: Standardize the `evaluate()` return dict keys across all subclasses.
  `evaluate_all.py` will call this method on any model instance — inconsistent
  keys will break the report.

**Task 2 — Implement Ridge regression baseline**
- Description: A simple regularized linear model used as the performance
  benchmark. One class, instantiated once per position group.
- Output artifact: `src/modeling/ridge_baseline.py`
- Implementation notes:
  - Wrap sklearn `Ridge(alpha=1.0)` with the base class interface.
  - Features must be pre-normalized before passing to Ridge — use the
    `_z` suffixed columns from the feature tables, not raw values.
  - Exclude rows where `missing_outcome == True` from training.
  - Exclude `weighted_av`, all outcome columns (`nfl_games`, `pro_bowls`,
    `all_pro`, `seasons_started`, `hof`), all flag columns, and all
    identifier columns (`pfr_id`, `cfb_id`, `gsis_id`, `player_name`,
    `position`, `position_group`, `draft_year`, `draft_team`, `college`)
    from the feature matrix. Build an explicit feature allowlist.
- Pitfall: Ridge cannot handle NaN inputs — it will raise an error. Before
  passing to Ridge, impute remaining NaN values with column median (fit on
  train, apply to holdout). This is the only place imputation is permitted
  and only for the Ridge baseline. XGBoost does not need this.

**Task 3 — Implement XGBoost model**
- Description: The primary regression model. One class, instantiated once
  per position group.
- Output artifact: `src/modeling/xgboost_model.py`
- Implementation notes:
  - Use `xgboost.XGBRegressor` with `objective="reg:squarederror"`.
  - XGBoost handles NaN natively — do not impute before passing features.
    Pass the raw (non-`_z`) feature columns to XGBoost; normalization is
    not required for tree-based models.
  - Exclude the same columns excluded in Task 2 (outcome columns, flag
    columns, identifier columns). Build an explicit feature allowlist
    consistent with Task 2's allowlist minus the `_z` columns.
  - Apply a monotone decreasing constraint on `draft_pick`: higher pick
    number should predict lower `weighted_av`. Specify via XGBoost's
    `monotone_constraints` parameter using the feature index position.
  - For skill_pass: include `is_QB`, `is_WR`, `is_TE` as features.
    XGBoost will learn position-specific split behavior from these.
  - Exclude rows where `missing_outcome == True` from training.
  - Hyperparameter tuning: use `optuna` with 30 trials. Internal validation
    uses a time-based split within the training window: 2000–2016 as inner
    train, 2017–2019 as inner validation. Never use hold-out data for tuning.
  - Best params to tune: `max_depth` (3–8), `learning_rate` (0.01–0.3),
    `n_estimators` (100–500), `subsample` (0.6–1.0),
    `colsample_bytree` (0.6–1.0), `min_child_weight` (1–10).
- Pitfall: The `monotone_constraints` parameter requires a tuple of integers
  with one entry per feature in the feature matrix, in the same order as the
  columns passed to `fit()`. Build the constraints tuple programmatically
  from the feature list — never hardcode column positions.
- Pitfall: Optuna will attempt to minimize a metric — use negative Spearman
  correlation as the objective (minimize negative = maximize positive). Do
  not use RMSE as the tuning objective; ranking accuracy matters more than
  absolute error for the Sleeper Score.

**Task 4 — Write training scripts**
- Description: One training script per position group that orchestrates
  loading features, splitting train/holdout, fitting all models, and saving
  artifacts.
- Output artifacts: `src/modeling/train_skill_pass.py`,
  `src/modeling/train_skill_run.py`
- Each training script must:
  1. Load train feature CSV.
  2. Exclude rows where `missing_outcome == True`.
  3. Build feature matrix `X` and target vector `y = weighted_av`.
  4. Fit Ridge baseline on `X_z` (normalized columns).
  5. Fit XGBoost model on `X` (raw columns, NaN-safe).
  6. Save both models to `models/saved/` using joblib.
  7. Log: number of training rows, number excluded for missing_outcome,
     feature count, and model save paths.
- Pitfall: Build and log the feature allowlist explicitly at the start of
  each training script. A silent column inclusion error (e.g., `nfl_games`
  leaking into the feature matrix) will inflate model performance
  dramatically and invalidate the entire evaluation.

**Task 5 — Write evaluation script**
- Description: Load trained models, run inference on hold-out data, and
  produce a structured evaluation report.
- Output artifacts: `src/modeling/evaluate_all.py`,
  `models/results/evaluation_summary.csv`,
  `models/results/feature_importance.csv`
- Evaluation steps:
  1. Load hold-out feature CSV for each position group.
  2. Exclude rows where `missing_outcome == True` from evaluation
     (they have no ground truth label).
  3. Run `predict()` for both Ridge and XGBoost on holdout `X`.
  4. Compute per-model, per-position-group metrics:
     - `rmse`: root mean squared error
     - `mae`: mean absolute error
     - `spearman`: Spearman rank correlation between predictions and
       true `weighted_av`
  5. Write one row per (position_group, model_type) to
     `evaluation_summary.csv`.
  6. Extract XGBoost feature importances (gain-based) and write to
     `feature_importance.csv` with columns: `feature`, `importance`,
     `position_group`.
- Pitfall: Evaluate on hold-out players where `missing_outcome == False`
  only — but also log how many hold-out players are excluded for this
  reason. Recent draft classes (2023, 2024) will have incomplete career
  outcomes by design; this is expected and should be noted in the report.

**Task 6 — Compute Sleeper Score**
- Description: Combine model predictions with outcome signals into a
  Sleeper Score for every player in both feature tables (train + holdout
  combined).
- Output artifacts: `src/scores/sleeper_score.py`,
  `data/features/skill_pass_scored.csv`,
  `data/features/skill_run_scored.csv`
- Scored files combine train and holdout rows — score all players, not
  just one split.
- Sleeper Score formula:
  ```
  sleeper_score = (
      0.40 * surplus_value_percentile +
      0.20 * pro_bowl_probability +
      0.20 * peak_value_percentile +
      0.20 * availability_factor
  ) * 100
  ```
- Component definitions:
  - `weighted_av_pred`: XGBoost predicted `weighted_av` for the player.
  - `surplus_value`: `weighted_av_pred` minus the median predicted
    `weighted_av` for all players at the same `draft_round`, within the
    same `position_group`. A positive surplus means the model expects
    the player to outperform round peers.
  - `surplus_value_percentile`: `surplus_value` converted to a percentile
    rank within `position_group`. Range [0.0, 1.0].
  - `pro_bowl_probability`: logistic function of `weighted_av_pred` scaled
    to approximate Pro Bowl likelihood. Use:
    `1 / (1 + exp(-0.05 * (weighted_av_pred - 40)))`.
    This is a heuristic — do not train a separate classifier.
  - `peak_value_percentile`: `weighted_av_pred` as a percentile rank within
    `position_group`. Range [0.0, 1.0].
  - `availability_factor`: `nfl_games_pred / 160.0` clipped to [0.0, 1.0].
    `nfl_games_pred` is estimated as `weighted_av_pred * 2.5` as a rough
    proxy — do not train a separate model for this.
  - All four components must be in [0.0, 1.0] before the weighted sum.
    Add assertions.
- Clip final `sleeper_score` to [0, 100] after computation.
- Retain all four component columns in the scored CSV alongside
  `sleeper_score` and `weighted_av_pred`.
- Pitfall: Percentile ranks must be computed within `position_group` only.
  A RB's surplus value must not be ranked against a QB's.
- Pitfall: Players with `missing_outcome == True` still receive a Sleeper
  Score — they have pre-draft features and model predictions even if their
  career outcome is unknown. Do not exclude them from scoring.

---

## 5. Prompt Template

Use this prompt when implementing Task 3 (XGBoost model) — the most complex
implementation task in this stage.

```
Project context: I'm building DraftSleeper, an NFL draft prospect forecasting
tool (Python 3.13). I'm in Stage 3: Modeling. My target variable is weighted_av
— never career_av (it is 100% null and must not appear anywhere). Train window:
draft_year <= 2019. Hold-out: draft_year >= 2020. Split is always on draft_year.

Task: Write a Python 3.13 class `XGBoostModel` in `src/modeling/xgboost_model.py`
that inherits from `BaseModel` in `src/modeling/base.py`.

The model predicts weighted_av (career value) for NFL draft prospects using
pre-draft features from the position group feature CSVs.

Feature matrix: all engineered pre-draft features from the feature CSV excluding:
  - Outcome columns: weighted_av, nfl_games, pro_bowls, all_pro, seasons_started, hof
  - Flag columns: missing_outcome, missing_combine, missing_height_weight, missing_age
  - Identifier columns: pfr_id, cfb_id, gsis_id, player_name, position,
    position_group, draft_year, draft_team, college
  - Z-score columns (suffix _z): XGBoost does not need normalized inputs
  Pass the feature allowlist as a constructor argument — never hardcode it inside
  the class.

Target: weighted_av (float). Exclude rows where missing_outcome == True before
fitting.

Implementation requirements:
- Use xgboost.XGBRegressor with objective="reg:squarederror".
- XGBoost receives NaN values as-is — do not impute before fit() or predict().
- Apply a monotone decreasing constraint on draft_pick. Build the
  monotone_constraints tuple programmatically from the feature list — never
  hardcode column positions.
- Hyperparameter tuning via optuna with 30 trials. Internal validation split:
  inner train = draft_year <= 2016, inner validation = draft_year 2017–2019.
  Optimize for negative Spearman correlation (not RMSE).
- Implement fit(), predict(), evaluate(), save(), load() matching BaseModel.
  evaluate() returns: {'rmse': float, 'mae': float, 'spearman': float}
- save() uses joblib. load() is a classmethod that returns a fitted instance.
- All functions must have docstrings.

Before writing any implementation code, explain:
1. How XGBoost's monotone_constraints parameter works when the feature matrix
   has many columns, and the correct way to build the constraint tuple
   programmatically so it stays in sync with the feature list
2. Why Spearman correlation is a better tuning objective than RMSE for a
   ranking-based application like Sleeper Score, and how to use it correctly
   as a negative optuna objective
3. How XGBoost handles NaN natively and what would go wrong if NaN values
   were imputed with 0 or column mean before passing to fit()
```

---

## 6. File & Folder Layout

```
src/modeling/
│
├── PLAN_03_modeling.md
├── base.py                         # Task 1 — abstract base class
├── xgboost_model.py                # Task 3 — XGBoost regressor
├── ridge_baseline.py               # Task 2 — Ridge regression benchmark
├── train_skill_pass.py             # Task 4 — training orchestration for skill_pass
├── train_skill_run.py              # Task 4 — training orchestration for skill_run
└── evaluate_all.py                 # Task 5 — unified evaluation report

src/scores/
└── sleeper_score.py                # Task 6 — Sleeper Score computation

models/
├── saved/
│   ├── xgboost_skill_pass.pkl
│   ├── xgboost_skill_run.pkl
│   ├── ridge_baseline_skill_pass.pkl
│   └── ridge_baseline_skill_run.pkl
└── results/
    ├── evaluation_summary.csv
    └── feature_importance.csv

data/features/                      # Appended by sleeper_score.py
├── skill_pass_scored.csv
└── skill_run_scored.csv
```
