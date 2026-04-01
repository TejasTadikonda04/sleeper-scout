from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pandas as pd
import math
import os

app = FastAPI(title="DraftSleeper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COMMON_COLS = [
    "player_name", "position", "college", "draft_year", "draft_round",
    "draft_pick", "draft_team", "height_in", "weight_lbs",
    "forty_yard", "vertical_jump", "broad_jump", "bench_reps",
    "cone_drill", "shuttle",
    "col_rush_attempts", "col_rush_yards", "col_rush_tds",
    "col_receptions", "col_rec_yards", "col_rec_tds",
    "speed_score", "burst_score", "agility_score", "bmi",
    "surplus_value", "surplus_value_percentile",
    "pro_bowl_probability", "peak_value_percentile",
    "availability_factor", "sleeper_score",
    "weighted_av_pred", "weighted_av",
    "nfl_games", "pro_bowls", "all_pro",
    "pfr_id",
    "speed_score_z", "burst_score_z", "agility_score_z",
    "vertical_jump_z", "broad_jump_z", "bench_reps_z",
    "forty_yard_z", "cone_drill_z", "shuttle_z",
    "bmi_z", "height_in_z", "weight_lbs_z",
]

PASS_ONLY_COLS = [
    "is_QB", "is_WR", "is_TE",
    "dominator_rating", "yards_per_reception", "rec_td_rate",
    "completion_pct", "yards_per_attempt", "td_int_ratio",
    "pass_yards_flag",
    "col_pass_completions", "col_pass_attempts",
    "col_pass_yards", "col_pass_tds", "col_pass_ints",
]


def _clean(val):
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def _load() -> pd.DataFrame:
    paths = {
        "skill_pass": "data/scores/skill_pass_scored.csv",
        "skill_run":  "data/scores/skill_run_scored.csv",
    }
    frames = []

    for group, path in paths.items():
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping.")
            continue

        df = pd.read_csv(path, low_memory=False)
        df["_group"] = group

        keep = [c for c in COMMON_COLS if c in df.columns]
        if group == "skill_pass":
            keep += [c for c in PASS_ONLY_COLS if c in df.columns]

        df = df[keep].copy()

        if group == "skill_run":
            for col in PASS_ONLY_COLS:
                df[col] = None

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["id"] = combined.index.astype(str)
    return combined


_df: pd.DataFrame = _load()


def _row_to_dict(row: pd.Series) -> dict:
    return {k: _clean(v) for k, v in row.to_dict().items()}


@app.get("/prospects")
def list_prospects(
    position: Optional[str] = Query(None),
    college: Optional[str] = Query(None),
    draft_round: Optional[int] = Query(None, ge=1, le=7),
    name: Optional[str] = Query(None),
    group: Optional[str] = Query(None),
    min_score: Optional[float] = Query(None, ge=0, le=100),
    limit: int = Query(200, le=500),
):
    df = _df.copy()
    if position:
        df = df[df["position"].str.upper() == position.upper()]
    if college:
        df = df[df["college"].str.lower().str.contains(college.lower(), na=False)]
    if draft_round:
        df = df[df["draft_round"] == draft_round]
    if name:
        df = df[df["player_name"].str.lower().str.contains(name.lower(), na=False)]
    if group:
        df = df[df["_group"] == group]
    if min_score is not None:
        df = df[df["sleeper_score"] >= min_score]

    df = df.sort_values("sleeper_score", ascending=False).head(limit)
    return [_row_to_dict(r) for _, r in df.iterrows()]


@app.get("/prospects/{prospect_id}")
def get_prospect(prospect_id: str):
    row = _df[_df["id"] == prospect_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Prospect {prospect_id!r} not found")
    return _row_to_dict(row.iloc[0])


@app.get("/scores/top")
def top_sleepers(
    limit: int = Query(50, le=200),
    position: Optional[str] = Query(None),
):
    df = _df.copy()
    if position:
        df = df[df["position"].str.upper() == position.upper()]
    df = df.sort_values("sleeper_score", ascending=False).head(limit)
    return [_row_to_dict(r) for _, r in df.iterrows()]


@app.get("/colleges")
def colleges():
    return sorted(_df["college"].dropna().unique().tolist())


@app.get("/positions")
def positions():
    return sorted(_df["position"].dropna().unique().tolist())


@app.get("/health")
def health():
    return {"status": "ok", "prospects_loaded": len(_df)}