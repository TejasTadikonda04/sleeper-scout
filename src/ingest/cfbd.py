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


def fetch_raw_college_stats() -> pd.DataFrame:
    """
    Fetch all player season stats from CFBD for seasons CFBD_START_YEAR–END_YEAR
    across all CFBD_STAT_CATEGORIES.

    Returns a long-format DataFrame:
        player_id | player_name | team | season | category | stat_type | stat
    """
    years = range(CFBD_START_YEAR, END_YEAR + 1)
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
    print(f"[CFBD] Raw stat records fetched: {len(df)}")
    return df
