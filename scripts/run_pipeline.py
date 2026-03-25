"""
Main ETL entry point.

Usage (from project root with venv activated):
    python scripts/run_pipeline.py

Environment variables (set in .env):
    DB_URL          PostgreSQL connection string
    CFBD_API_KEY    College Football Data API key (free at https://collegefootballdata.com/key)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import CFBD_API_KEY
from src.db import get_engine, create_table, upsert_prospects
from src.ingest.nflverse import load_combine, load_draft_picks
from src.ingest.cfbd import fetch_raw_college_stats
from src.transform.clean import clean_combine, clean_draft_picks
from src.transform.merge import build_final_table


def main() -> None:
    if not CFBD_API_KEY:
        print(
            "WARNING: CFBD_API_KEY is not set — college stats will be skipped.\n"
            "Get a free key at https://collegefootballdata.com/key and add it to .env."
        )

    # ── Step 1: nflverse combine ───────────────────────────────────────────
    print("\n=== Step 1/5: Loading nflverse combine data ===")
    combine = clean_combine(load_combine())

    # ── Step 2: nflverse draft picks + career AV ──────────────────────────
    print("\n=== Step 2/5: Loading nflverse draft picks + AV ===")
    draft = clean_draft_picks(load_draft_picks())

    # ── Step 3: CFBD college stats ─────────────────────────────────────────
    print("\n=== Step 3/5: Fetching CFBD college stats ===")
    if CFBD_API_KEY:
        raw_stats = fetch_raw_college_stats()
    else:
        raw_stats = pd.DataFrame(
            columns=["player_id", "player_name", "team", "season",
                     "category", "stat_type", "stat"]
        )
        print("  Skipped — no CFBD_API_KEY.")

    # ── Step 4: Merge all sources ─────────────────────────────────────────
    print("\n=== Step 4/5: Merging datasets ===")
    final = build_final_table(combine, draft, raw_stats)

    # ── Step 5: Load to PostgreSQL ────────────────────────────────────────
    print("\n=== Step 5/5: Loading to PostgreSQL ===")
    engine = get_engine()
    create_table(engine)
    n = upsert_prospects(final, engine)
    print(f"  Upserted {n} rows into nfl_prospects.")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
