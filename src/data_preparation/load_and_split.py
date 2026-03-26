"""
Load nfl_prospects.csv, filter to skill positions (QB, WR, TE, RB),
assign position_group labels, add missing-data flags, and split into
train/hold-out CSVs by draft_year.

Train   : draft_year <= 2019
Hold-out: draft_year >= 2020

Usage:
    python -m src.data_preparation.load_and_split
"""

from __future__ import annotations

import pathlib

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_PATH = pathlib.Path("data/raw/nfl_prospects.csv")
PROCESSED_DIR = pathlib.Path("data/processed")

SKILL_POSITIONS = {"QB", "WR", "TE", "RB"}

POSITION_GROUP_MAP: dict[str, str] = {
    "QB": "skill_pass",
    "WR": "skill_pass",
    "TE": "skill_pass",
    "RB": "skill_run",
}

# Columns that are 100% null in the source file — dropped on load.
# career_av is listed first; the rest are special-teams / defensive cols.
NULL_COLUMNS = [
    "career_av",
    "col_total_tackles",
    "col_tfl",
    "col_sacks",
    "col_ints",
    "col_pass_deflections",
    "col_qb_hurries",
    "col_fg_made",
    "col_fg_attempted",
    "col_xp_made",
    "col_xp_attempted",
    "col_punts",
    "col_punt_yards",
]

# Combine metric columns used for the missing_combine flag.
COMBINE_COLS = [
    "forty_yard",
    "vertical_jump",
    "broad_jump",
    "bench_reps",
    "cone_drill",
    "shuttle",
]

# College stat columns applicable per position — used for missing_college_stats.
# Only these columns count toward a player's college-stats missingness;
# inapplicable columns (e.g. col_pass_yards for an RB) are never checked.
POSITION_COLLEGE_STATS: dict[str, list[str]] = {
    "QB": ["col_pass_completions", "col_pass_attempts", "col_pass_yards", "col_pass_tds", "col_pass_ints"],
    "WR": ["col_receptions", "col_rec_yards", "col_rec_tds"],
    "TE": ["col_receptions", "col_rec_yards", "col_rec_tds"],
    "RB": ["col_rush_attempts", "col_rush_yards", "col_rush_tds"],
}

TRAIN_YEAR_CUTOFF = 2019  # draft_year <= this value → train set


# ---------------------------------------------------------------------------
# Task 1 — Load and filter
# ---------------------------------------------------------------------------

def load_and_filter(path: pathlib.Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw source file, filter to skill positions, and drop dead columns.

    Drops career_av and all other fully-null columns listed in NULL_COLUMNS.
    Uses errors='ignore' so the script does not crash if a column is already
    absent in a future version of the source file.

    Args:
        path: Path to nfl_prospects.csv.

    Returns:
        DataFrame filtered to QB, WR, TE, RB rows with position_group assigned.
    """
    df = pd.read_csv(path, low_memory=False)

    # Keep skill positions only — all other positions are out of scope.
    df = df[df["position"].isin(SKILL_POSITIONS)].copy()

    # Drop career_av and every other fully-null column safely.
    df = df.drop(columns=NULL_COLUMNS, errors="ignore")

    # Assign the two valid position group labels.
    df["position_group"] = df["position"].map(POSITION_GROUP_MAP)

    return df


# ---------------------------------------------------------------------------
# Task 2 — Missing-data flag columns
# ---------------------------------------------------------------------------

def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Append boolean missing-data flag columns to the dataframe.

    Flags are computed on the full filtered dataset (before splitting) so that
    missingness rates are consistent across train and hold-out. Every flag
    column is guaranteed to have zero nulls via fillna(False).

    Flags added:
        missing_outcome       : True if weighted_av is null.
        missing_combine       : True if ANY of the six combine metrics is null.
        missing_height_weight : True if height_in or weight_lbs is null.
        missing_college_stats : True if ALL college stat columns applicable to
                                the player's position are null. Inapplicable
                                columns are never counted against a player.

    Args:
        df: Filtered dataframe with position_group assigned.

    Returns:
        Copy of df with four flag columns appended.
    """
    df = df.copy()

    df["missing_outcome"] = df["weighted_av"].isnull().fillna(False)

    combine_present = [c for c in COMBINE_COLS if c in df.columns]
    df["missing_combine"] = df[combine_present].isnull().any(axis=1).fillna(False)

    df["missing_height_weight"] = (
        df[["height_in", "weight_lbs"]].isnull().any(axis=1).fillna(False)
    )

    # Position-aware college stats flag: True only when ALL applicable columns
    # for that player's position are null. Build per-position, then merge.
    missing_cs = pd.Series(False, index=df.index)
    for pos, cols in POSITION_COLLEGE_STATS.items():
        present_cols = [c for c in cols if c in df.columns]
        if not present_cols:
            continue
        mask = df["position"] == pos
        if mask.any():
            missing_cs.loc[mask] = df.loc[mask, present_cols].isnull().all(axis=1)
    df["missing_college_stats"] = missing_cs.fillna(False)

    return df


# ---------------------------------------------------------------------------
# Task 3 — Split and write
# ---------------------------------------------------------------------------

def split_and_write(
    df: pd.DataFrame,
    out_dir: pathlib.Path = PROCESSED_DIR,
) -> dict[str, pathlib.Path]:
    """Partition the dataframe into four subsets and write them as CSVs.

    Split logic (always by draft_year — never random):
        draft_year <= TRAIN_YEAR_CUTOFF → train
        draft_year >  TRAIN_YEAR_CUTOFF → hold-out

    Each half is further split by position_group (skill_pass, skill_run).
    After writing, each file is read back and its row count is asserted to
    match the written dataframe — guards against silent CSV truncation.

    Args:
        df: Fully flagged dataframe covering all skill positions and years.
        out_dir: Directory where the four CSVs will be written.

    Returns:
        Dict mapping split name to the Path of the written file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    train = df[df["draft_year"] <= TRAIN_YEAR_CUTOFF]
    holdout = df[df["draft_year"] > TRAIN_YEAR_CUTOFF]

    splits: dict[str, pd.DataFrame] = {
        "skill_pass_train":   train[train["position_group"] == "skill_pass"],
        "skill_pass_holdout": holdout[holdout["position_group"] == "skill_pass"],
        "skill_run_train":    train[train["position_group"] == "skill_run"],
        "skill_run_holdout":  holdout[holdout["position_group"] == "skill_run"],
    }

    written: dict[str, pathlib.Path] = {}
    for name, subset in splits.items():
        path = out_dir / f"{name}.csv"
        subset.to_csv(path, index=False)

        # Read back and assert row count — catches silent truncation.
        df_read = pd.read_csv(path)
        assert len(df_read) == len(subset), (
            f"Row count mismatch for {name}: "
            f"wrote {len(subset)}, read back {len(df_read)}"
        )

        written[name] = path
        print(f"  Wrote {path}  ({len(subset):,} rows)")

    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full load → filter → flag → split → write pipeline."""
    print("Stage 1 — Data Preparation")
    print(f"  Source: {RAW_PATH}")

    df = load_and_filter(RAW_PATH)
    print(f"  Loaded {len(df):,} skill-position rows")

    df = add_missing_flags(df)
    print("  Added missing-data flags")

    written = split_and_write(df)
    print(f"  All {len(written)} split files written and verified.")


if __name__ == "__main__":
    main()
