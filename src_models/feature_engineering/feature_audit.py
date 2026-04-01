"""
Audit the four feature CSVs produced by Stage 2.

Reads from CSV files — not from in-memory dataframes — to catch any
serialization issues from the write step.

Output: data/processed/feature_audit.txt

Usage:
    python -m src.feature_engineering.feature_audit
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

from src.feature_engineering.shared_features import COLS_TO_NORMALIZE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURES_DIR = pathlib.Path("data/features")
PROCESSED_DIR = pathlib.Path("data/processed")
AUDIT_PATH = PROCESSED_DIR / "feature_audit.txt"

FEATURE_FILES: dict[str, pathlib.Path] = {
    "skill_pass_train":   FEATURES_DIR / "skill_pass_features_train.csv",
    "skill_pass_holdout": FEATURES_DIR / "skill_pass_features_holdout.csv",
    "skill_run_train":    FEATURES_DIR / "skill_run_features_train.csv",
    "skill_run_holdout":  FEATURES_DIR / "skill_run_features_holdout.csv",
}

# All boolean flag columns expected in both groups.
FLAG_COLS = [
    "missing_outcome",
    "missing_combine",
    "missing_height_weight",
    "missing_college_stats",
    "missing_age",
]

# skill_pass position dummy columns.
DUMMY_COLS = ["is_QB", "is_WR", "is_TE"]

# Z-score column names derived from COLS_TO_NORMALIZE.
Z_COLS = [f"{c}_z" for c in COLS_TO_NORMALIZE]

# ---------------------------------------------------------------------------
# High-null exemption sets for the >60% gate
#
# These columns are expected to have high null rates and are already covered
# by an existing flag or by the position structure of the dataset.
# They are excluded from the ">60% null, no flag found" gate check.
# ---------------------------------------------------------------------------

# Raw combine columns + composites derived from them.
# All covered by missing_combine=True from Stage 1.
_COMBINE_COVERED: frozenset[str] = frozenset({
    "forty_yard", "vertical_jump", "broad_jump", "bench_reps", "cone_drill", "shuttle",
    "speed_score", "burst_score", "agility_score", "bmi",
})

# skill_pass: columns whose nulls are explained by position structure.
#   QB-only features  → correctly NaN for WR/TE; covered by is_WR/is_TE dummies
#                        and pass_yards_flag (for col_pass_* derivatives).
#   WR/TE-only features → correctly NaN for QB; covered by is_QB dummy
#                          and rec_yards_flag (for col_rec_* derivatives).
#   col_rush_* → not applicable to most pass-catchers; no explicit flag needed
#                 because these are raw pass-through columns, not engineered ones.
_SKILL_PASS_POSITION_INAPPLICABLE: frozenset[str] = frozenset({
    # QB-only raw cols
    "col_pass_completions", "col_pass_attempts", "col_pass_yards",
    "col_pass_tds", "col_pass_ints",
    # QB-only engineered features (covered by pass_yards_flag + is_WR/is_TE)
    "completion_pct", "yards_per_attempt", "td_int_ratio",
    # WR/TE-only raw cols
    "col_receptions", "col_rec_yards", "col_rec_tds",
    # WR/TE-only engineered features (covered by rec_yards_flag + is_QB)
    "dominator_rating", "yards_per_reception", "rec_td_rate",
    # Rush cols — not applicable to most pass-catchers (raw pass-through)
    "col_rush_yards", "col_rush_tds", "col_rush_attempts",
})

# skill_run: passing columns are inapplicable to RBs (raw pass-through from Stage 1).
# No engineered RB feature depends on these columns.
_SKILL_RUN_POSITION_INAPPLICABLE: frozenset[str] = frozenset({
    "col_pass_completions", "col_pass_attempts", "col_pass_yards",
    "col_pass_tds", "col_pass_ints",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _null_rates(df: pd.DataFrame) -> pd.Series:
    """Compute null rate (0–100 %) per column, sorted descending.

    Args:
        df: Any DataFrame.

    Returns:
        Series of null percentages indexed by column name.
    """
    return (df.isnull().mean() * 100).sort_values(ascending=False)


def _restore_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Cast boolean flag columns to bool dtype after CSV read.

    Args:
        df: Dataframe with potentially string-typed flag columns.

    Returns:
        Same dataframe with flag columns cast to bool.
    """
    bool_map = {"True": True, "False": False, True: True, False: False}
    for col in FLAG_COLS + [c for c in df.columns if c.endswith("_flag")]:
        if col in df.columns:
            df[col] = df[col].map(bool_map).astype(bool)
    return df


# ---------------------------------------------------------------------------
# Per-file audit
# ---------------------------------------------------------------------------

def _audit_one(
    name: str,
    path: pathlib.Path,
    lines: list[str],
) -> bool:
    """Audit a single feature CSV and append findings to lines.

    Checks performed:
        - Row count and position breakdown
        - career_av absence (raises flag if found)
        - missing_outcome / weighted_av consistency
        - Flag column inventory: count True, % True, null check
        - Position dummy sum assertion for skill_pass files
        - Null rates for all feature columns (non-zero only)
        - Z-score column summary: mean and std
        - weighted_av distribution for non-missing rows

    Args:
        name: Human-readable split name (e.g. 'skill_pass_train').
        path: Path to the feature CSV.
        lines: Mutable list; audit output is appended here.

    Returns:
        True if all Go/No-Go gates pass for this file; False otherwise.
    """
    df = pd.read_csv(path, low_memory=False)
    df = _restore_flags(df)

    is_skill_pass = "skill_pass" in name
    is_train = name.endswith("_train")
    passed = True

    lines.append("=" * 70)
    lines.append(f"FILE: {path.name}")
    lines.append("=" * 70)

    # Row count and position breakdown.
    n_rows = len(df)
    lines.append(f"Rows: {n_rows:,}")
    if "position" in df.columns:
        pos_counts = df["position"].value_counts().to_dict()
        lines.append(f"Positions: {pos_counts}")
    if "position_group" in df.columns:
        pg_counts = df["position_group"].value_counts().to_dict()
        lines.append(f"Position groups: {pg_counts}")

    # career_av check.
    if "career_av" in df.columns:
        lines.append("  *** career_av column is present — must be absent ***")
        passed = False
    else:
        lines.append("career_av: absent (correct)")

    # missing_outcome / weighted_av consistency.
    if "missing_outcome" in df.columns and "weighted_av" in df.columns:
        mismatch_a = ((df["missing_outcome"] == True) & df["weighted_av"].notna()).sum()
        mismatch_b = ((df["missing_outcome"] == False) & df["weighted_av"].isnull()).sum()
        if mismatch_a or mismatch_b:
            lines.append(f"  *** missing_outcome mismatch: {mismatch_a} flagged-but-present, "
                         f"{mismatch_b} unflagged-but-null ***")
            passed = False
        else:
            lines.append("missing_outcome / weighted_av: consistent")

    # Position dummy assertion for skill_pass.
    if is_skill_pass and all(c in df.columns for c in DUMMY_COLS):
        dummy_sum = df["is_QB"] + df["is_WR"] + df["is_TE"]
        bad_rows = int((dummy_sum != 1).sum())
        if bad_rows:
            lines.append(f"  *** GATE FAIL: {bad_rows} row(s) where is_QB+is_WR+is_TE != 1 ***")
            passed = False
        else:
            lines.append("is_QB + is_WR + is_TE == 1: OK for all rows")

    # Flag column inventory.
    lines.append("")
    lines.append("--- Flag columns ---")
    all_flag_cols = FLAG_COLS + [c for c in df.columns if c.endswith("_flag") and c not in FLAG_COLS]
    for col in all_flag_cols:
        if col not in df.columns:
            continue
        n_true = int(df[col].sum())
        pct = n_true / n_rows * 100 if n_rows else 0
        n_null = int(df[col].isnull().sum())
        note = f"  *** {n_null} null(s) — gate fail ***" if n_null else ""
        if n_null:
            passed = False
        lines.append(f"  {col:<35}: {n_true:,} / {n_rows:,} ({pct:.1f}%){note}")

    # Z-score summary (mean ≈ 0, std ≈ 1 for train; different for holdout is expected).
    present_z = [c for c in Z_COLS if c in df.columns]
    if present_z:
        lines.append("")
        lines.append("--- Z-score columns (mean / std) ---")
        note = "(expect ~0.0 / ~1.0 for train)" if is_train else "(may differ from 0/1 — expected for holdout)"
        lines.append(f"  {note}")
        for col in present_z:
            series = df[col].dropna()
            if len(series):
                lines.append(f"  {col:<35}: mean={series.mean():+.3f}  std={series.std():.3f}")

    # weighted_av distribution (non-missing rows only).
    if "weighted_av" in df.columns and "missing_outcome" in df.columns:
        non_null = df.loc[df["missing_outcome"] == False, "weighted_av"].dropna()
        if len(non_null):
            d = non_null.describe(percentiles=[0.25, 0.50, 0.75])
            lines.append("")
            lines.append(
                f"weighted_av (non-missing, n={len(non_null):,}): "
                f"min={d['min']:.1f}  p25={d['25%']:.1f}  "
                f"median={d['50%']:.1f}  p75={d['75%']:.1f}  max={d['max']:.1f}"
            )

    # Null rates — non-zero only.
    lines.append("")
    lines.append("--- Null rates >0% (feature columns only, descending) ---")
    non_zero = _null_rates(df)
    non_zero = non_zero[non_zero > 0]

    # Build the full exemption set for this file's position group.
    exemptions = _COMBINE_COVERED
    if is_skill_pass:
        exemptions = exemptions | _SKILL_PASS_POSITION_INAPPLICABLE
    else:
        exemptions = exemptions | _SKILL_RUN_POSITION_INAPPLICABLE

    if non_zero.empty:
        lines.append("  (no nulls)")
    else:
        for col, rate in non_zero.items():
            # Gate: flag columns must have 0% null.
            if col in FLAG_COLS or col.endswith("_flag"):
                lines.append(f"  {col:<40}: {rate:.1f}%  *** flag column must be 0% null ***")
                passed = False

            # _z columns: inherit null pattern from their source — skip gate.
            elif col.endswith("_z"):
                lines.append(f"  {col:<40}: {rate:.1f}%")

            # >60% null gate — apply only to columns not in an exemption set.
            elif rate > 60:
                if col in exemptions:
                    lines.append(f"  {col:<40}: {rate:.1f}%  (expected — covered by existing flag/structure)")
                else:
                    # Check for an explicit flag column as a last resort.
                    has_flag = any(
                        c in df.columns
                        for c in (f"{col}_flag", f"missing_{col}", col.replace("col_", "") + "_flag")
                    )
                    if has_flag:
                        lines.append(f"  {col:<40}: {rate:.1f}%")
                    else:
                        lines.append(f"  {col:<40}: {rate:.1f}%  *** >60% null, no flag column found ***")
                        passed = False

            else:
                lines.append(f"  {col:<40}: {rate:.1f}%")

    lines.append("")
    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_feature_audit(features_dir: pathlib.Path = FEATURES_DIR) -> bool:
    """Run the full audit across all four feature CSV files.

    Writes the audit report to data/processed/feature_audit.txt and prints
    it to stdout. Returns False if any Go/No-Go gate fails.

    Args:
        features_dir: Directory containing the four feature CSVs.

    Returns:
        True if all gates pass; False otherwise.
    """
    lines: list[str] = ["DraftSleeper — Stage 2 Feature Audit", ""]
    all_passed = True

    for name, path in FEATURE_FILES.items():
        if not path.exists():
            lines.append(f"*** FILE NOT FOUND: {path} ***")
            all_passed = False
            continue
        passed = _audit_one(name, path, lines)
        if not passed:
            all_passed = False

    lines.append("=" * 70)
    lines.append(
        "GATE RESULT: " + ("PASS — ready for Stage 3" if all_passed else "FAIL — fix issues above")
    )
    lines.append("=" * 70)

    audit_text = "\n".join(lines) + "\n"
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(audit_text, encoding="utf-8")
    print(audit_text)
    return all_passed


def main() -> None:
    """Entry point: run the feature audit and exit with code 0 (pass) or 1 (fail)."""
    passed = run_feature_audit()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
