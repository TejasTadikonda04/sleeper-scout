# PLAN_01 — Data Preparation

## 1. Stage Goal
Load `nfl_prospects.csv`, filter to skill positions, audit data quality, and
produce clean train and hold-out CSVs that all downstream stages read from —
replacing the raw source file as the working input for every subsequent stage.

---

## 2. Inputs & Outputs

**Inputs:**
- `data/raw/nfl_prospects.csv` — read-only, never modified

**Outputs:**
- `data/processed/skill_pass_train.csv` — QB, WR, TE | draft_year 2000–2019
- `data/processed/skill_pass_holdout.csv` — QB, WR, TE | draft_year 2020–2024
- `data/processed/skill_run_train.csv` — RB | draft_year 2000–2019
- `data/processed/skill_run_holdout.csv` — RB | draft_year 2020–2024
- `data/processed/data_audit.txt` — null rates, row counts, and flag inventory

---

## 3. Go/No-Go Gate

All five of the following must be true before moving to Stage 2:

1. Four processed CSVs exist and are non-empty.
2. No player with `draft_year <= 2019` appears in any holdout file. No player
   with `draft_year >= 2020` appears in any train file.
3. `career_av` column is absent from all four processed CSVs — it must be
   dropped at load time and never written to any output file.
4. `position_group` column is present on every row with value `skill_pass` or
   `skill_run` only — no raw position strings in this column.
5. Audit report confirms `weighted_av` null rate is below 15% in each train file.

---

## 4. Task List

**Task 1 — Load and filter the source file**
- Description: Read `nfl_prospects.csv`, drop out-of-scope positions and the
  `career_av` column, and assign `position_group`.
- Output artifact: In-memory dataframe ready for splitting — no file written yet.
- Filter rules:
  - Keep only rows where `position` is in `['QB', 'WR', 'TE', 'RB']`
  - Drop `career_av` immediately on load — do not carry it forward
  - Drop the 9 fully-null columns: `col_total_tackles`, `col_tfl`, `col_sacks`,
    `col_ints`, `col_pass_deflections`, `col_qb_hurries`, `col_fg_made`,
    `col_fg_attempted`, `col_xp_made`, `col_xp_attempted`, `col_punts`,
    `col_punt_yards`
  - Assign `position_group`:
    - `skill_pass` for QB, WR, TE
    - `skill_run` for RB
- Pitfall: Drop `career_av` using `df.drop(columns=['career_av'], errors='ignore')`
  so the script does not crash if the column is already absent in a future version
  of the source file.

**Task 2 — Add missing-data flag columns**
- Description: Before any splitting, add boolean flag columns for all meaningful
  missingness patterns. Flags must be computed on the full filtered dataset so
  rates are consistent.
- Output artifact: Flag columns appended to the in-memory dataframe.
- Flag columns to create:
  - `missing_outcome`: True if `weighted_av` is null
  - `missing_combine`: True if ANY of `forty_yard`, `vertical_jump`, `broad_jump`,
    `bench_reps`, `cone_drill`, `shuttle` is null
  - `missing_height_weight`: True if `height_in` or `weight_lbs` is null
  - `missing_college_stats`: True if ALL college stat columns applicable to the
    player's position are null (position-aware — see Task 3 in PLAN_02 for
    position-to-column mapping)
- Pitfall: Flag columns must have zero nulls themselves — use `.fillna(False)`
  after each boolean assignment to guarantee this.

**Task 3 — Split by draft_year and position_group**
- Description: Partition the flagged dataframe into four subsets and write to
  `data/processed/`.
- Output artifact: Four CSVs as specified in Section 2.
- Split rule: `draft_year <= 2019` → train | `draft_year >= 2020` → holdout.
  This is the only valid split — never random.
- Write order: write all four files before running the audit so the audit
  reads from the files, not from memory. This validates the write/read cycle.
- Pitfall: Confirm row counts after writing by reading back each CSV and
  asserting `len(df_read) == len(df_written)`. Silent truncation during CSV
  write is rare but catastrophic if it happens here.

**Task 4 — Run data audit**
- Description: Read all four processed CSVs and produce a structured audit
  report.
- Output artifact: `data/processed/data_audit.txt`
- Audit checks per file:
  - Row count and unique player count (`pfr_id` as the unique key)
  - Draft year range (min, max) — confirm no leakage
  - `position_group` value counts
  - Null rate (%) for every column, sorted descending
  - Flag column summary: count and % True for each flag column
  - `weighted_av` distribution: min, 25th pct, median, 75th pct, max
  - Confirm `career_av` is absent — log a warning if found
- Pitfall: Run the audit by reading from CSV files, not from in-memory
  dataframes. This catches any serialization issues in Task 3.

---

## 5. Prompt Template

Use this prompt when implementing Tasks 1–3 together in `load_and_split.py`.

```
Project context: I'm building DraftSleeper, an NFL draft prospect forecasting
tool (Python 3.13). I'm in Stage 1: Data Preparation. The source file is
data/raw/nfl_prospects.csv — a self-contained dataset with 6,387 rows covering
draft years 2000–2024, all positions. I need to filter it to skill positions
only and split it into train/hold-out files.

Task: Write a Python 3.13 script `src/data_preparation/load_and_split.py` that:

1. Loads data/raw/nfl_prospects.csv
2. Filters to positions QB, WR, TE, RB only
3. Drops career_av immediately (it is 100% null — never use it)
4. Drops these fully-null columns: col_total_tackles, col_tfl, col_sacks,
   col_ints, col_pass_deflections, col_qb_hurries, col_fg_made,
   col_fg_attempted, col_xp_made, col_xp_attempted, col_punts, col_punt_yards
5. Assigns position_group: skill_pass for QB/WR/TE, skill_run for RB
6. Adds flag columns:
   - missing_outcome (bool): True if weighted_av is null
   - missing_combine (bool): True if any of forty_yard, vertical_jump,
     broad_jump, bench_reps, cone_drill, shuttle is null
   - missing_height_weight (bool): True if height_in or weight_lbs is null
7. Splits by draft_year: train = draft_year <= 2019, holdout = draft_year >= 2020
8. Writes four CSVs to data/processed/:
   skill_pass_train.csv, skill_pass_holdout.csv,
   skill_run_train.csv, skill_run_holdout.csv
9. After writing, reads each CSV back and asserts row count matches

Constraints:
- career_av must not appear in any output file
- position_group column must contain only "skill_pass" or "skill_run"
- All flag columns must have zero nulls (use fillna(False) after assignment)
- Split must be on draft_year only — never random
- All functions must have docstrings
- Do not use any library beyond pandas and numpy

Before writing any implementation code, explain:
1. The correct pandas pattern for dropping a column safely when it may or
   may not be present in a future version of the source file
2. How to implement position-aware missing_college_stats flagging without
   hardcoding column lists inside the flag assignment logic
3. Any risk in writing CSVs with boolean columns and reading them back as
   the correct dtype
```

---

## 6. File & Folder Layout

```
src/data_preparation/
│
├── PLAN_01_data_preparation.md     # This file
├── load_and_split.py               # Tasks 1–3 — load, flag, split, write
└── audit_data.py                   # Task 4 — audit report generator

data/raw/
└── nfl_prospects.csv               # Read-only source — never modified

data/processed/                     # Written by this stage
├── skill_pass_train.csv
├── skill_pass_holdout.csv
├── skill_run_train.csv
├── skill_run_holdout.csv
└── data_audit.txt
```
