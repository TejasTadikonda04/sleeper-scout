"""
ingest.py — fetch and save all data locally.

Outputs:
  data/combine_av.csv       combine metrics + weighted career AV (for Model 1)
  data/combine_college.csv  same + matched college stats (for Model 2)

Run once (or whenever you want fresh data):
  venv/Scripts/python src/ingest.py
"""

import os
import re
import time
import warnings

import requests
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from rapidfuzz import process, fuzz

warnings.filterwarnings("ignore")

load_dotenv()
CFB_API_KEY = os.getenv("CFB_API_KEY")

YEARS = list(range(1994, 2024))
COLLEGE_FEATURES = ["pass_yds", "rush_yds", "rec_yds", "total_td", "tackles", "sacks"]
REQ_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nfl-research)"}
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")


# --- Helpers ---

def ht_to_inches(v):
    m = re.match(r"(\d+)['\-](\d+)", str(v))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


def norm(name):
    return re.sub(r"[^a-z]", "", str(name).lower()) if pd.notna(name) else ""


# --- Fetchers ---

def fetch_combine_and_av():
    print("Fetching combine data from nflverse...")
    combine = nfl.import_combine_data(YEARS)

    print("Fetching draft picks (career AV) from nflverse...")
    draft = nfl.import_draft_picks(YEARS)

    # nflverse car_av is unpopulated; w_av (Weighted Career AV) is the equivalent PFR metric
    df = combine.merge(
        draft[["pfr_player_id", "w_av"]].drop_duplicates("pfr_player_id"),
        left_on="pfr_id",
        right_on="pfr_player_id",
        how="left",
    )
    df.rename(columns={"w_av": "car_av"}, inplace=True)
    df["car_av"] = pd.to_numeric(df["car_av"], errors="coerce").fillna(0)
    df["ht_in"] = df["ht"].apply(ht_to_inches)
    df["name_norm"] = df["player_name"].apply(norm)
    df["draft_year"] = df["season"].astype(int)

    print(f"  {len(df)} combine players, {(df['car_av'] > 0).sum()} with career AV > 0")
    return df


def fetch_college_stats():
    print("\nFetching CFBD college stats...")
    frames = []
    hdrs = {**REQ_HEADERS, "Authorization": f"Bearer {CFB_API_KEY}"}

    for year in range(1993, 2024):
        try:
            resp = requests.get(
                "https://api.collegefootballdata.com/stats/player/season",
                headers=hdrs, params={"year": year}, timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if data:
                df = pd.DataFrame(data)
                df["college_year"] = year
                frames.append(df)
                print(f"  {year}: {len(df)} stat rows")
        except Exception as e:
            print(f"  {year}: skipped ({type(e).__name__})")
        time.sleep(0.5)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    raw["stat"] = pd.to_numeric(raw["stat"], errors="coerce")
    raw["key"] = raw["category"].str.lower() + "_" + raw["statType"].str.lower()
    pivot = (
        raw.pivot_table(index=["player", "college_year"], columns="key", values="stat", aggfunc="sum")
        .reset_index()
    )
    pivot.columns.name = None

    def gc(col):
        return pivot[col] if col in pivot.columns else pd.Series(0, index=pivot.index)

    td_cols = [c for c in pivot.columns if c.endswith("_td")]
    out = pd.DataFrame({
        "player":       pivot["player"],
        "college_year": pivot["college_year"],
        "pass_yds":     gc("passing_yds"),
        "rush_yds":     gc("rushing_yds"),
        "rec_yds":      gc("receiving_yds"),
        "total_td":     pivot[td_cols].sum(axis=1) if td_cols else 0,
        "tackles":      gc("defensive_tot"),
        "sacks":        gc("defensive_sacks"),
    })
    out["name_norm"] = out["player"].apply(norm)
    return out.fillna(0)


# --- Player Matching ---

def match_college(combine_df, college_df):
    for f in COLLEGE_FEATURES:
        combine_df[f] = 0.0
    if college_df.empty:
        return combine_df

    by_year = {int(yr): grp for yr, grp in college_df.groupby("college_year")}
    by_year_exact = {yr: dict(zip(grp["name_norm"], grp.index)) for yr, grp in by_year.items()}

    matched = 0
    for i, row in combine_df.iterrows():
        name = row["name_norm"]
        dy = int(row["draft_year"])
        found = None

        for offset in [1, 2]:
            cy = dy - offset
            if cy in by_year_exact and name in by_year_exact[cy]:
                found = college_df.loc[by_year_exact[cy][name]]
                if isinstance(found, pd.DataFrame):
                    found = found.iloc[0]
                break

        if found is None:
            cy = dy - 1
            if cy in by_year:
                yr_df = by_year[cy]
                names = yr_df["name_norm"].tolist()
                result = process.extractOne(name, names, scorer=fuzz.ratio)
                if result and result[1] >= 85:
                    rows = yr_df[yr_df["name_norm"] == result[0]]
                    if not rows.empty:
                        found = rows.iloc[0]

        if found is not None:
            for f in COLLEGE_FEATURES:
                if f in found.index:
                    combine_df.at[i, f] = float(found[f])
            matched += 1

    print(f"  Matched {matched}/{len(combine_df)} players to college stats")
    return combine_df


# --- Main ---

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    combine_df = fetch_combine_and_av()
    college_df = fetch_college_stats()

    combine_path = os.path.join(DATA_DIR, "combine_av.csv")
    combine_df.to_csv(combine_path, index=False)
    print(f"\nSaved: {combine_path}")

    print("\nMatching college stats to combine players...")
    combined = match_college(combine_df.copy(), college_df)
    college_path = os.path.join(DATA_DIR, "combine_college.csv")
    combined.to_csv(college_path, index=False)
    print(f"Saved: {college_path}")

    print("\nDone. Run predict.py to train models and generate plots.")


if __name__ == "__main__":
    main()
