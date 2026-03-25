"""
Main ETL entry point.

Usage (from project root with venv activated):
    python scripts/run_pipeline.py

Output:
    nfl_prospects.csv  — merged combine + draft + college stats

Environment variables (set in .env):
    CFBD_API_KEY    College Football Data API key (free at https://collegefootballdata.com/key)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import CFBD_API_KEY
from src.ingest.nflverse import load_combine, load_draft_picks
from src.ingest.cfbd import fetch_raw_college_stats
from src.transform.clean import clean_combine, clean_draft_picks
from src.transform.merge import build_final_table

OUTPUT_PATH = Path(__file__).parent.parent / "nfl_prospects.csv"
CACHE_PATH  = Path(__file__).parent.parent / ".pipeline_cache.parquet"

# Column order matching the original nfl_prospects schema
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


def save_csv(final: pd.DataFrame) -> None:
    ordered = [c for c in COLUMN_ORDER if c in final.columns]
    extras  = [c for c in final.columns if c not in COLUMN_ORDER]
    final   = final[ordered + extras]
    final.to_csv(OUTPUT_PATH, index=False)
    print(f"  Wrote {len(final)} rows to {OUTPUT_PATH}")


def main(from_cache: bool = False) -> None:
    if from_cache:
        if not CACHE_PATH.exists():
            print(f"No cache found at {CACHE_PATH}. Run without --from-cache first.")
            sys.exit(1)
        print(f"Loading merged data from cache ({CACHE_PATH})...")
        final = pd.read_parquet(CACHE_PATH)
        print(f"  {len(final)} rows loaded.")
        save_csv(final)
        print("\nDone.")
        return

    if not CFBD_API_KEY:
        print(
            "WARNING: CFBD_API_KEY is not set — college stats will be skipped.\n"
            "Get a free key at https://collegefootballdata.com/key and add it to .env."
        )

    # ── Step 1: nflverse combine ───────────────────────────────────────────
    print("\n=== Step 1/4: Loading nflverse combine data ===")
    combine = clean_combine(load_combine())

    # ── Step 2: nflverse draft picks + career AV ──────────────────────────
    print("\n=== Step 2/4: Loading nflverse draft picks + AV ===")
    draft = clean_draft_picks(load_draft_picks())

    # ── Step 3: CFBD college stats ─────────────────────────────────────────
    print("\n=== Step 3/4: Fetching CFBD college stats ===")
    if CFBD_API_KEY:
        raw_stats = fetch_raw_college_stats()
    else:
        raw_stats = pd.DataFrame(
            columns=["player_id", "player_name", "team", "season",
                     "category", "stat_type", "stat"]
        )
        print("  Skipped — no CFBD_API_KEY.")

    # ── Step 4: Merge and write CSV ────────────────────────────────────────
    print("\n=== Step 4/4: Merging and writing CSV ===")
    final = build_final_table(combine, draft, raw_stats)

    # Cache merged result so --from-cache can re-save without re-fetching
    final.to_parquet(CACHE_PATH, index=False)
    print(f"  Cached merged data to {CACHE_PATH}")

    save_csv(final)
    print("\nPipeline complete.")


if __name__ == "__main__":
    from_cache = "--from-cache" in sys.argv
    main(from_cache=from_cache)
