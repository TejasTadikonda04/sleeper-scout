# PLAN_02 — Feature Engineering

## 1. Stage Goal
Transform the four processed split CSVs into position-group feature tables
with engineered combine composites, college production metrics, position dummy
variables, and missing-data flags — fully model-ready with no silent nulls.

---

## 2. Inputs & Outputs

**Inputs:**
- `data/processed/skill_pass_train.csv`
- `data/processed/skill_pass_holdout.csv`
- `data/processed/skill_run_train.csv`
- `data/processed/skill_run_holdout.csv`

**Outputs:**
- `data/features/skill_pass_features_train.csv`
- `data/features/skill_pass_features_holdout.csv`
- `data/features/skill_run_features_train.csv`
- `data/features/skill_run_features_holdout.csv`
- `data/processed/feature_audit.txt` — null rates, distributions, and
  flag column inventory for all four feature tables

---

## 3. Go/No-Go Gate

All five of the following must be true before moving to Stage 3:

1. All four feature CSVs exist and contain only players present in the
   corresponding processed CSV — no new rows introduced.
2. No engineered feature column has a null rate above 60% in the train
   files without a corresponding boolean flag column.
3. All flag columns have exactly 0% null rate in all four files.
4. `is_QB + is_WR + is_TE == 1` for every row in both `skill_pass` files.
   Confirmed by assertion in the audit script.
5. Z-score normalization parameters (mean, std) were fit on train data only
   and saved to `data/features/scaler_params.json` — applied to holdout
   without refitting.

---

## 4. Task List

**Task 1 — Build shared features**
- Description: Compute features that apply identically to both position groups.
  Implement in `shared_features.py` and import from both group scripts.
- Output artifact: Shared feature columns added to each group's working dataframe.
- Features to build:
  - `pick_value`: Stuart's draft value chart score mapped from `draft_pick`.
    Use a hardcoded lookup dict — do not scrape. For picks beyond the chart's
    range, use 0.0.
  - `is_early_pick`: 1 if `draft_pick` <= 32, else 0.
  - `is_day_3_pick`: 1 if `draft_round` >= 5, else 0.
  - `age_at_draft`: player age in decimal years on draft day (April 30 of
    `draft_year`). Source column: none available in the data — this feature
    cannot be computed. Set to NaN for all rows with `missing_age=True`.
    Do not attempt to derive from other columns.
- Pitfall: `age_at_draft` is not derivable from the source data — there is
  no birthdate column. Do not impute or estimate it. Flag it and move on.
  Its absence is a known limitation, not a bug.

**Task 2 — Build combine features (both groups)**
- Description: Compute athletic composite metrics from raw combine drill values.
  Apply identically to both groups via `shared_features.py`.
- Output artifact: Combine feature columns appended to each group's dataframe.
- Features to build:
  - `speed_score`: `(weight_lbs * 200) / (forty_yard ^ 4)`. Set to NaN if
    `forty_yard` is null — do not impute.
  - `burst_score`: `(vertical_jump + broad_jump) / 2`. Set to NaN if either
    is null.
  - `agility_score`: `(cone_drill + shuttle) / 2`. Lower is better — do not
    invert. Set to NaN if either is null.
  - `bmi`: `(weight_lbs * 703) / (height_in ^ 2)`. Set to NaN if either
    input is null.
  - Retain all raw combine columns as-is: `forty_yard`, `vertical_jump`,
    `broad_jump`, `bench_reps`, `cone_drill`, `shuttle`, `height_in`,
    `weight_lbs`.
- Z-score normalization:
  - Fit mean and std for each combine feature on the train split only.
  - Save params to `data/features/scaler_params.json` keyed by
    `{position_group}__{feature_name}`.
  - Apply saved params to the holdout split — never refit on holdout.
  - Add `_z` suffix to normalized columns: `speed_score_z`, `forty_yard_z`,
    etc. Retain unnormalized originals alongside them.
- Pitfall: `missing_combine=True` was set in Stage 1. Do not drop these rows.
  XGBoost handles NaN natively — pass NaN as-is. Do not impute with mean,
  median, or 0 before model training.

**Task 3 — Build skill_pass-specific features**
- Description: Compute production metrics for QB, WR, and TE. Implement in
  `engineering_skill_pass.py`.
- Output artifact: `data/features/skill_pass_features_train.csv` and
  `data/features/skill_pass_features_holdout.csv`
- Position dummy variables (required):
  - `is_QB`: 1 if `position == 'QB'`, else 0
  - `is_WR`: 1 if `position == 'WR'`, else 0
  - `is_TE`: 1 if `position == 'TE'`, else 0
  - Assert `is_QB + is_WR + is_TE == 1` for every row before writing.
- WR/TE production features (set to NaN for QBs — never 0):
  - `dominator_rating`: `col_rec_yards / (col_rec_yards + col_rush_yards)`.
    Proxy for target share. Set to NaN if both inputs are null. Validate
    range [0.0, 1.0] — log and cap any violations.
  - `yards_per_reception`: `col_rec_yards / col_receptions`. Set to NaN if
    `col_receptions` is null or 0.
  - `rec_td_rate`: `col_rec_tds / col_receptions`. Set to NaN if
    `col_receptions` is null or 0.
  - `rec_yards_flag`: True if `col_rec_yards` is null, else False.
- QB production features (set to NaN for WR/TE — never 0):
  - `completion_pct`: `col_pass_completions / col_pass_attempts`. Set to NaN
    if `col_pass_attempts` is null or 0.
  - `yards_per_attempt`: `col_pass_yards / col_pass_attempts`. Set to NaN if
    `col_pass_attempts` is null or 0.
  - `td_int_ratio`: `col_pass_tds / col_pass_ints`. Cap at 10.0 when
    `col_pass_ints == 0`. Set to NaN if `col_pass_tds` is null.
  - `pass_yards_flag`: True if `col_pass_yards` is null, else False.
- Pitfall: `col_rec_yards` null rate is ~57% in train for skill_pass — most
  of this is QBs, for whom it is correctly NaN. Confirm null rate drops to a
  reasonable level when filtered to WR/TE only before flagging as a data issue.
- Pitfall: Position-inapplicable features must be NaN, not 0. A WR with
  `completion_pct = 0` is a data error; a WR with `completion_pct = NaN` is
  correct. This distinction matters for XGBoost's NaN-handling logic.

**Task 4 — Build skill_run-specific features**
- Description: Compute production metrics for RB. Implement in
  `engineering_skill_run.py`.
- Output artifact: `data/features/skill_run_features_train.csv` and
  `data/features/skill_run_features_holdout.csv`
- RB production features:
  - `yards_per_carry`: `col_rush_yards / col_rush_attempts`. Set to NaN if
    `col_rush_attempts` is null or 0.
  - `rush_td_rate`: `col_rush_tds / col_rush_attempts`. Set to NaN if
    `col_rush_attempts` is null or 0.
  - `reception_rate`: `col_receptions / col_rush_attempts`. Proxy for
    pass-catching usage. Set to NaN if `col_rush_attempts` is null or 0.
  - `yards_from_scrimmage`: `col_rush_yards + col_rec_yards`. Set to NaN if
    both inputs are null. If only one is null, use the non-null value.
  - `rush_yards_flag`: True if `col_rush_yards` is null, else False.
  - `rec_yards_flag`: True if `col_rec_yards` is null, else False.
- Confirm `position_group == 'skill_run'` for every row — add assertion.
- Pitfall: `col_rush_yards` null rate is ~67% across all skill positions in
  the source data, but for `skill_run` only it should be much lower. Verify
  the null rate for RBs specifically in the audit before treating it as a
  data quality issue.

**Task 5 — Attach and validate outcome labels**
- Description: Confirm `weighted_av` and supporting outcome columns are present
  and correctly typed in all four feature files. These columns come through from
  Stage 1 — no joining required.
- Output artifact: Outcome columns confirmed present; `missing_outcome` flag
  validated.
- Outcome columns to confirm present: `weighted_av`, `nfl_games`, `pro_bowls`,
  `all_pro`, `seasons_started`, `hof`, `missing_outcome`.
- Confirm `missing_outcome == True` wherever `weighted_av` is null, and
  `missing_outcome == False` wherever `weighted_av` is not null. Add an
  assertion — mismatches indicate a bug in Stage 1.
- Confirm `career_av` is absent. If found, raise an error and halt.

**Task 6 — Run feature audit**
- Description: Read all four feature CSVs and write a structured audit report.
- Output artifact: `data/processed/feature_audit.txt`
- Checks per file:
  - Row count and position breakdown
  - Null rate (%) for every feature column, sorted descending
  - Flag column inventory: name, count True, % True
  - `is_QB + is_WR + is_TE == 1` assertion result for skill_pass files
  - Z-score column summary: mean and std (should be ~0 and ~1 for train,
    different for holdout — both are expected)
  - `weighted_av` distribution for rows where `missing_outcome == False`:
    min, 25th pct, median, 75th pct, max, per position group
  - Confirm `career_av` absent from all files

---

## 5. Prompt Template

Use this prompt when implementing Task 3 (`dominator_rating` and position-specific
features for skill_pass) — the most error-prone computation in this stage.

```
Project context: I'm building DraftSleeper, an NFL draft prospect forecasting
tool (Python 3.13). I'm in Stage 2: Feature Engineering, working on the
skill_pass position group (QB, WR, TE). My target variable is weighted_av —
never career_av (it is 100% null and must not appear in any code).

Task: Write a Python 3.13 function in `src/feature_engineering/engineering_skill_pass.py`
that computes position-specific production features for the skill_pass group.

Input: a dataframe loaded from data/processed/skill_pass_train.csv (or holdout).
Relevant source columns:
  position (values: QB, WR, TE), draft_pick, draft_round,
  col_pass_completions, col_pass_attempts, col_pass_yards, col_pass_tds,
  col_pass_ints, col_receptions, col_rec_yards, col_rec_tds, col_rush_yards,
  weighted_av, missing_outcome, missing_combine

Output: same dataframe with these columns appended:
  is_QB (int: 0/1), is_WR (int: 0/1), is_TE (int: 0/1),
  dominator_rating (float — WR/TE only, NaN for QB),
  yards_per_reception (float — WR/TE only, NaN for QB),
  rec_td_rate (float — WR/TE only, NaN for QB),
  rec_yards_flag (bool),
  completion_pct (float — QB only, NaN for WR/TE),
  yards_per_attempt (float — QB only, NaN for WR/TE),
  td_int_ratio (float — QB only, NaN for WR/TE),
  pass_yards_flag (bool)

Constraints:
- is_QB + is_WR + is_TE must equal exactly 1 for every row. Add an assertion
  that raises ValueError if violated.
- Position-inapplicable features must be NaN, never 0.
- dominator_rating: validate range [0.0, 1.0] after computation. Log any
  violations to console and cap at 1.0.
- td_int_ratio: cap at 10.0 when col_pass_ints == 0.
- career_av must not appear anywhere in this function — not as input, output,
  or intermediate variable.
- All functions must have docstrings.
- Libraries: pandas and numpy only.

Before writing any implementation code, explain:
1. The correct pandas pattern for setting a column to NaN for a subset of rows
   (e.g., WR/TE rows only) without affecting the rest of the dataframe
2. Why NaN and 0 must remain distinct for position-inapplicable features, and
   how XGBoost's native NaN handling uses this distinction
3. How to safely divide two columns where the denominator may be 0 or null,
   without producing inf or unintended 0 values
```

---

## 6. Feature Table Schema Summary

All four feature files share this column structure. Position-specific columns
use NaN for inapplicable rows.

**Pass-through from Stage 1 (both groups):**
`pfr_id`, `cfb_id`, `gsis_id`, `player_name`, `position`, `position_group`,
`draft_year`, `draft_round`, `draft_pick`, `draft_team`, `college`

**Shared engineered features (both groups):**
`pick_value`, `is_early_pick`, `is_day_3_pick`, `missing_age`,
`height_in`, `weight_lbs`, `forty_yard`, `vertical_jump`, `broad_jump`,
`bench_reps`, `cone_drill`, `shuttle`,
`speed_score`, `burst_score`, `agility_score`, `bmi`,
`speed_score_z`, `forty_yard_z`, `vertical_jump_z`, `broad_jump_z`,
`bench_reps_z`, `cone_drill_z`, `shuttle_z`, `height_in_z`, `weight_lbs_z`

**skill_pass only:**
`is_QB`, `is_WR`, `is_TE`,
`dominator_rating`, `yards_per_reception`, `rec_td_rate`, `rec_yards_flag`,
`completion_pct`, `yards_per_attempt`, `td_int_ratio`, `pass_yards_flag`

**skill_run only:**
`yards_per_carry`, `rush_td_rate`, `reception_rate`, `yards_from_scrimmage`,
`rush_yards_flag`, `rec_yards_flag`

**Outcome labels (both groups — pass-through from Stage 1):**
`weighted_av`, `nfl_games`, `pro_bowls`, `all_pro`, `seasons_started`, `hof`

**Flag columns (both groups — all must have 0% null rate):**
`missing_outcome`, `missing_combine`, `missing_height_weight`, `missing_age`

---

## 7. File & Folder Layout

```
src/feature_engineering/
│
├── PLAN_02_feature_engineering.md      # This file
├── shared_features.py                  # Tasks 1, 2 — shared and combine features
├── engineering_skill_pass.py           # Task 3 — skill_pass production features
└── engineering_skill_run.py            # Task 4 — skill_run production features

data/features/                          # Written by this stage
├── skill_pass_features_train.csv
├── skill_pass_features_holdout.csv
├── skill_run_features_train.csv
├── skill_run_features_holdout.csv
└── scaler_params.json                  # Z-score params fit on train only

data/processed/                         # Appended by this stage
└── feature_audit.txt
```
