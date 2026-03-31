import time

import requests
import pandas as pd
from tqdm import tqdm

from ..config import (
    CFBD_API_KEY,
    CFBD_START_YEAR,
    END_YEAR,
    CFBD_STAT_CATEGORIES,
)

_BASE_URL = "https://api.collegefootballdata.com"
_REQUEST_DELAY = 1.0   # base seconds between requests
_MAX_RETRIES   = 4     # retry up to 4 times on 429


def _get_headers() -> dict:
    return {"Authorization": f"Bearer {CFBD_API_KEY}"}


def _get_with_retry(url: str, params: dict) -> requests.Response:
    """GET with exponential backoff on 429 Too Many Requests."""
    delay = _REQUEST_DELAY
    for attempt in range(_MAX_RETRIES):
        resp = requests.get(url, headers=_get_headers(), params=params, timeout=30)
        if resp.status_code == 429:
            wait = delay * (2 ** attempt)
            print(f"\n[CFBD] Rate limited — waiting {wait:.1f}s before retry {attempt + 1}/{_MAX_RETRIES}")
            time.sleep(wait)
            continue
        return resp
    return resp  # return last response even if still 429


def fetch_raw_college_stats(
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """
    Fetch all player season stats from CFBD for the given year range across all
    CFBD_STAT_CATEGORIES.  Defaults to CFBD_START_YEAR–END_YEAR from config.

    Returns a long-format DataFrame:
        player_id | player_name | team | season | category | stat_type | stat
    """
    if start_year is None:
        start_year = CFBD_START_YEAR
    if end_year is None:
        end_year = END_YEAR
    years = range(start_year, end_year + 1)
    rows: list[dict] = []
    total_calls = len(list(years)) * len(CFBD_STAT_CATEGORIES)

    with tqdm(total=total_calls, desc="[CFBD] Fetching stats") as pbar:
        for year in years:
            for category in CFBD_STAT_CATEGORIES:
                try:
                    resp = _get_with_retry(
                        f"{_BASE_URL}/stats/player/season",
                        params={"year": year, "category": category},
                    )
                    resp.raise_for_status()
                    for r in resp.json():
                        rows.append({
                            "player_id":   str(r.get("playerId", "")),
                            "player_name": r.get("player"),
                            "team":        r.get("team"),
                            "season":      year,
                            "category":    category,
                            "stat_type":   r.get("statType"),
                            "stat":        r.get("stat"),
                        })
                except Exception as exc:
                    print(f"[CFBD] Warning — {year}/{category}: {exc}")
                finally:
                    pbar.update(1)
                    time.sleep(_REQUEST_DELAY)

    df = pd.DataFrame(rows)

    # CFBD occasionally returns stat values as strings rather than numbers.
    # Coerce to float here so that groupby.sum() adds numerically instead of
    # concatenating strings (which would produce wildly inflated values like
    # 142183231 instead of 556 for three seasons of 142 + 183 + 231).
    if not df.empty:
        df["stat"] = pd.to_numeric(df["stat"], errors="coerce")

    print(f"[CFBD] Raw stat records fetched: {len(df)}")
    return df
