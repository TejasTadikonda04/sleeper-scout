"""
Audit the four processed split files produced by load_and_split.py.

Reads from CSV files — not from in-memory dataframes — to catch any
serialization issues from the write step.

Output: data/processed/data_audit.txt

Usage:
    python -m src.data_preparation.audit_data
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROCESSED_DIR = pathlib.Path("data/processed")
AUDIT_PATH = PROCESSED_DIR / "data_audit.txt"

FLAG_COLS = [
    "missing_outcome",
    "missing_combine",
    "missing_height_weight",
    "missing_college_stats",
]

SPLIT_FILES: dict[str, pathlib.Path] = {
    "skill_pass_train":   PROCESSED_DIR / "skill_pass_train.csv",
    "skill_pass_holdout": PROCESSED_DIR / "skill_pass_holdout.csv",
    "skill_run_train":    PROCESSED_DIR / "skill_run_train.csv",
    "skill_run_holdout":  PROCESSED_DIR / "skill_run_holdout.csv",
}

# Go/No-Go gate: weighted_av null rate must be below this in train files.
WEIGHTED_AV_NULL_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_split(path: pathlib.Path) -> pd.DataFrame:
    """Read a processed CSV and cast flag columns back to bool dtype.

    pandas writes True/False values as the strings "True"/"False" in CSV.
    Explicit mapping after read restores the correct bool dtype so that
    .sum() and .isnull() behave correctly downstream.

    Args:
        path: Path to a processed split CSV.

    Returns:
        DataFrame with flag columns as bool dtype.
    """
    df = pd.read_csv(path, low_memory=False)
    bool_map = {"True": True, "False": False, True: True, False: False}
    for col in FLAG_COLS:
        if col in df.columns:
            df[col] = df[col].map(bool_map).astype(bool)
    return df


def _null_rates(df: pd.DataFrame) -> pd.Series:
    """Compute null rate (0–100 %) per column, sorted descending.

    Args:
        df: Any DataFrame.

    Returns:
        Series of null percentages indexed by column name, descending order.
    """
    return (df.isnull().mean() * 100).sort_values(ascending=False)


def _audit_one(
    name: str,
    path: pathlib.Path,
    lines: list[str],
) -> bool:
    """Audit a single processed CSV and append findings to lines.

    Checks performed:
        - Row count and unique player count (pfr_id as unique key)
        - Draft year range — confirms no temporal leakage
        - position_group value counts and validity
        - Absence of career_av column
        - weighted_av null rate (Go/No-Go gate for train files)
        - weighted_av distribution (min, p25, median, p75, max)
        - Flag column summary (count and % True; null check)
        - Null rate per column, sorted descending

    Args:
        name: Human-readable split name (e.g. "skill_pass_train").
        path: Path to the CSV file.
        lines: Mutable list of strings; audit output is appended here.

    Returns:
        True if all Go/No-Go gates pass for this file; False otherwise.
    """
    df = _read_split(path)
    is_train = name.endswith("_train")
    passed = True

    lines.append("=" * 70)
    lines.append(f"FILE: {path.name}")
    lines.append("=" * 70)

    # Row and player counts.
    n_rows = len(df)
    n_players = df["pfr_id"].nunique() if "pfr_id" in df.columns else "n/a"
    lines.append(f"Rows          : {n_rows:,}")
    lines.append(f"Unique players: {n_players}")

    # Draft year range + leakage checks.
    yr_min = int(df["draft_year"].min())
    yr_max = int(df["draft_year"].max())
    lines.append(f"Draft years   : {yr_min}–{yr_max}")

    if is_train and yr_max > 2019:
        lines.append("  *** LEAKAGE: train file contains draft_year > 2019 ***")
        passed = False
    if not is_train and yr_min < 2020:
        lines.append("  *** LEAKAGE: holdout file contains draft_year < 2020 ***")
        passed = False

    # position_group distribution and validity.
    if "position_group" in df.columns:
        pg_counts = df["position_group"].value_counts().to_dict()
        lines.append(f"Position groups: {pg_counts}")
        invalid = set(df["position_group"].dropna()) - {"skill_pass", "skill_run"}
        if invalid:
            lines.append(f"  *** INVALID position_group values: {invalid} ***")
            passed = False
    else:
        lines.append("  *** MISSING: position_group column not found ***")
        passed = False

    # career_av check — must be absent.
    if "career_av" in df.columns:
        lines.append("  *** career_av column is present — must be dropped ***")
        passed = False
    else:
        lines.append("career_av     : absent (correct)")

    # weighted_av null rate (gate applies to train files only).
    if "weighted_av" in df.columns:
        null_pct = df["weighted_av"].isnull().mean() * 100
        lines.append(f"weighted_av null rate: {null_pct:.1f}%")
        if is_train and null_pct >= WEIGHTED_AV_NULL_THRESHOLD * 100:
            lines.append(
                f"  *** GATE FAIL: null rate {null_pct:.1f}% >= "
                f"{WEIGHTED_AV_NULL_THRESHOLD * 100:.0f}% threshold ***"
            )
            passed = False

        # Distribution summary (non-null values only).
        non_null = df["weighted_av"].dropna()
        if len(non_null) > 0:
            d = non_null.describe(percentiles=[0.25, 0.50, 0.75])
            lines.append(
                f"weighted_av   : min={d['min']:.1f}  p25={d['25%']:.1f}  "
                f"median={d['50%']:.1f}  p75={d['75%']:.1f}  max={d['max']:.1f}"
            )

    # Flag column summary.
    lines.append("")
    lines.append("--- Flag columns ---")
    for col in FLAG_COLS:
        if col in df.columns:
            n_true = int(df[col].sum())
            pct = n_true / n_rows * 100 if n_rows else 0
            null_in_flag = int(df[col].isnull().sum())
            note = f"  *** {null_in_flag} nulls in flag column ***" if null_in_flag else ""
            lines.append(f"  {col:<30}: {n_true:,} / {n_rows:,} ({pct:.1f}%){note}")
        else:
            lines.append(f"  {col:<30}: MISSING")

    # Null rates per column (non-zero only).
    lines.append("")
    lines.append("--- Null rates (columns with >0% nulls, descending) ---")
    non_zero = _null_rates(df)
    non_zero = non_zero[non_zero > 0]
    if non_zero.empty:
        lines.append("  (no nulls)")
    else:
        for col, rate in non_zero.items():
            lines.append(f"  {col:<40}: {rate:.1f}%")

    lines.append("")
    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_audit(processed_dir: pathlib.Path = PROCESSED_DIR) -> bool:
    """Run the full audit across all four processed split files.

    Writes the audit report to data/processed/data_audit.txt and prints it
    to stdout. Returns False (and exits with code 1) if any Go/No-Go gate
    fails.

    Args:
        processed_dir: Directory containing the four processed CSVs.

    Returns:
        True if all gates pass; False otherwise.
    """
    lines: list[str] = ["DraftSleeper — Stage 1 Data Audit", ""]
    all_passed = True

    for name, path in SPLIT_FILES.items():
        if not path.exists():
            lines.append(f"*** FILE NOT FOUND: {path} ***")
            all_passed = False
            continue
        passed = _audit_one(name, path, lines)
        if not passed:
            all_passed = False

    lines.append("=" * 70)
    lines.append("GATE RESULT: " + ("PASS — ready for Stage 2" if all_passed else "FAIL — fix issues above"))
    lines.append("=" * 70)

    audit_text = "\n".join(lines) + "\n"
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(audit_text, encoding="utf-8")

    print(audit_text)
    return all_passed


def main() -> None:
    """Entry point: run the audit and exit with code 0 (pass) or 1 (fail)."""
    passed = run_audit()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
