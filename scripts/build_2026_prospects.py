"""
Build a 2026 NFL Draft prospect file in the same column format as nfl_prospects.csv.

Usage (from project root):
    uv run python scripts/build_2026_prospects.py
    uv run python scripts/build_2026_prospects.py --from-cache   # skip CFBD API calls

Input:
    2026_combine/*.csv   — per-position combine CSVs

Output:
    2026_prospects.csv   — one row per prospect, same columns as nfl_prospects.csv
                           (draft fields and NFL outcome columns are all null)

Environment variables (same .env as run_pipeline.py):
    CFBD_API_KEY    Required for college stats; get free at https://collegefootballdata.com/key
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import CFBD_API_KEY
from src.ingest.cfbd import fetch_raw_college_stats
from src.transform.clean import normalize_name, normalize_school, normalize_position
from src.transform.merge import merge_college_stats

COMBINE_DIR  = Path(__file__).parent.parent / "2026_combine"
OUTPUT_PATH  = Path(__file__).parent.parent / "2026_prospects.csv"
CACHE_PATH   = Path(__file__).parent.parent / ".2026_cfbd_cache.parquet"

DRAFT_YEAR   = 2026
# Fetch college stats back to 2018 — earliest a 2026 draftee could have played
CFBD_FETCH_START = 2018

COLUMN_ORDER = [
    "pfr_id", "cfb_id", "gsis_id", "player_name", "position",
    "draft_year", "draft_round", "draft_pick", "draft_team", "college",
    "height_in", "weight_lbs", "forty_yard", "vertical_jump", "broad_jump",
    "bench_reps", "cone_drill", "shuttle",
    "col_pass_completions", "col_pass_attempts", "col_pass_yards",
    "col_pass_tds", "col_pass_ints",
    "col_rush_attempts", "col_rush_yards", "col_rush_tds",
    "col_receptions", "col_rec_yards", "col_rec_tds",
    "col_total_tackles", "col_tfl", "col_sacks", "col_ints",
    "col_pass_deflections", "col_qb_hurries",
    "col_fg_made", "col_fg_attempted", "col_xp_made", "col_xp_attempted",
    "col_punts", "col_punt_yards",
    "career_av", "weighted_av", "draft_team_av",
    "nfl_games", "pro_bowls", "all_pro", "seasons_started", "hof",
]

# Map filename suffix → canonical position (passed through normalize_position after)
_FILE_POSITION_MAP = {
    "QBs": "QB",
    "RBs": "RB",
    "WRs": "WR",
    "TEs": "TE",
    "OL":  "OL",
    "DL":  "DL",
    "LBs": "LB",
    "DBs": "DB",
}


# ---------------------------------------------------------------------------
# Measurement parsers
# ---------------------------------------------------------------------------

def _parse_height(val) -> float | None:
    """
    Parse NFL compact height encoding FIIN → total inches.

    Format: 4-digit number where
      F   = feet (1 digit)
      II  = whole inches (2 digits)
      N   = eighths of an inch (0-7)

    Examples: 6032 → 6'03 2/8" = 75.25", 5113 → 5'11 3/8" = 71.375"
    Also handles plain feet-inches strings like "6-3" or "6'3".
    """
    if pd.isna(val) or val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # "6-3" or "6'3" style
    m = re.match(r"^(\d+)['\-](\d+)\"?$", s)
    if m:
        return float(m.group(1)) * 12 + float(m.group(2))
    # Compact FIIN (exactly 4 digits)
    if re.fullmatch(r"\d{4}", s):
        feet   = int(s[0])
        inches = int(s[1:3])
        eighths = int(s[3])
        return feet * 12 + inches + eighths / 8
    # Plain numeric already in inches
    try:
        return float(s)
    except ValueError:
        return None


def _parse_broad_jump(val) -> float | None:
    """
    Parse broad jump to total inches.

    Formats:
      "10'3"  or "10'09" → split on apostrophe, feet * 12 + inches
      "1003"  or "903"   → last 2 digits = inches, prefix = feet
    """
    if pd.isna(val) or val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # Explicit feet'inches with apostrophe
    if "'" in s:
        parts = s.replace('"', "").split("'")
        try:
            return float(int(parts[0]) * 12 + int(parts[1]))
        except (ValueError, IndexError):
            return None
    # Compact FII or FFII numeric
    try:
        n = int(float(s))
        inches = n % 100    # last two digits
        feet   = n // 100   # everything before
        return float(feet * 12 + inches)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Load + clean combine CSVs
# ---------------------------------------------------------------------------

def load_combine_csvs() -> pd.DataFrame:
    """Load all per-position CSVs, tag each row with its canonical position."""
    frames = []
    for csv_path in sorted(COMBINE_DIR.glob("*.csv")):
        pos_group = csv_path.stem.split(" - ")[-1].strip()   # e.g. "QBs"
        canonical = _FILE_POSITION_MAP.get(pos_group, pos_group)
        df = pd.read_csv(csv_path, dtype=str)
        df["_position_raw"] = canonical
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No CSVs found in {COMBINE_DIR}")

    return pd.concat(frames, ignore_index=True)


def clean_combine_2026(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename, parse, and normalise the combined combine data."""
    df = raw.copy()

    # Normalise column names (strip trailing spaces/colons used in source files)
    df.columns = [c.strip().rstrip(":").strip() for c in df.columns]

    # Some files label the split column slightly differently — harmonise
    df = df.rename(columns={
        "10 Yard Splits": "10 Yard Split",
        "3 Cone":         "cone_drill",
    })

    # Rename to our schema
    df = df.rename(columns={
        "NAME":         "player_name",
        "SCHOOL":       "college",
        "HEIGHT":       "_raw_height",
        "WEIGHT":       "weight_lbs",
        "40 Yard Dash": "forty_yard",
        "Vertical":     "vertical_jump",
        "Broad":        "_raw_broad",
        "Bench":        "bench_reps",
        "Shuttle":      "shuttle",
        "_position_raw": "position",
    })

    df["height_in"]   = df["_raw_height"].apply(_parse_height)
    df["broad_jump"]  = df["_raw_broad"].apply(_parse_broad_jump)

    # Normalise text
    df["player_name"] = df["player_name"].str.strip()
    df["college"]     = df["college"].apply(normalize_school)
    df["position"]    = df["position"].apply(normalize_position)

    # Coerce numeric measurables
    for col in ["weight_lbs", "forty_yard", "vertical_jump",
                "bench_reps", "cone_drill", "shuttle"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop blank rows
    df = df[df["player_name"].notna() & (df["player_name"] != "")]

    # Deduplicate: if a player appears in more than one file, keep the row
    # with the most non-null measurable fields
    measurable_cols = ["height_in", "weight_lbs", "forty_yard", "vertical_jump",
                       "broad_jump", "bench_reps", "cone_drill", "shuttle"]
    df["_fill_count"] = df[[c for c in measurable_cols if c in df.columns]].notna().sum(axis=1)
    df = (
        df.sort_values("_fill_count", ascending=False)
          .drop_duplicates(subset=["player_name"], keep="first")
          .drop(columns=["_fill_count"])
    )

    # Add null columns for draft fields and NFL outcomes
    df["draft_year"] = DRAFT_YEAR
    for col in ["pfr_id", "cfb_id", "gsis_id",
                "draft_round", "draft_pick", "draft_team",
                "career_av", "weighted_av", "draft_team_av",
                "nfl_games", "pro_bowls", "all_pro", "seasons_started", "hof"]:
        df[col] = pd.NA

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_csv(final: pd.DataFrame) -> None:
    ordered = [c for c in COLUMN_ORDER if c in final.columns]
    extras  = [c for c in final.columns if c not in COLUMN_ORDER]
    final[ordered + extras].to_csv(OUTPUT_PATH, index=False)
    print(f"  Wrote {len(final)} rows → {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(from_cache: bool = False) -> None:
    # ── Step 1: Load combine CSVs ──────────────────────────────────────────
    print("\n=== Step 1/3: Loading 2026 combine CSVs ===")
    raw   = load_combine_csvs()
    prospects = clean_combine_2026(raw)
    print(f"  {len(prospects)} prospects loaded across positions:")
    print(f"  {prospects['position'].value_counts().to_dict()}")

    # ── Step 2: CFBD college stats ─────────────────────────────────────────
    print("\n=== Step 2/3: Fetching CFBD college stats ===")
    if from_cache and CACHE_PATH.exists():
        print(f"  Loading from cache ({CACHE_PATH})")
        raw_stats = pd.read_parquet(CACHE_PATH)
        print(f"  {len(raw_stats)} stat records loaded from cache.")
    elif CFBD_API_KEY:
        raw_stats = fetch_raw_college_stats(
            start_year=CFBD_FETCH_START,
            end_year=DRAFT_YEAR - 1,   # only seasons before the draft
        )
        raw_stats.to_parquet(CACHE_PATH, index=False)
        print(f"  Cached CFBD stats → {CACHE_PATH}")
    else:
        print(
            "  WARNING: CFBD_API_KEY not set — college stats will be empty.\n"
            "  Get a free key at https://collegefootballdata.com/key"
        )
        raw_stats = pd.DataFrame(
            columns=["player_id", "player_name", "team", "season",
                     "category", "stat_type", "stat"]
        )

    # ── Step 3: Merge college stats + write ───────────────────────────────
    print("\n=== Step 3/3: Merging college stats and writing CSV ===")

    # merge_college_stats needs a pfr_id column to use as a join key.
    # For 2026 prospects there are no pfr_ids yet, so we generate temporary
    # stable surrogate keys for the merge (dropped from output).
    prospects["pfr_id"] = [f"_2026_{i:04d}" for i in range(len(prospects))]

    final = merge_college_stats(prospects, raw_stats)

    # Remove the surrogate pfr_ids so the output column is null/empty
    final["pfr_id"] = pd.NA

    save_csv(final)
    print("\nDone.")


if __name__ == "__main__":
    from_cache = "--from-cache" in sys.argv
    main(from_cache=from_cache)
