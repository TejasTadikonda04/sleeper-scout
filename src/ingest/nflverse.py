import pandas as pd
import nflreadpy as nfl

from ..config import START_YEAR, END_YEAR


def load_combine() -> pd.DataFrame:
    """
    Load NFL combine data from nflverse and return a clean pandas DataFrame.
    Filtered to draft years [START_YEAR, END_YEAR].
    """
    df = nfl.load_combine().to_pandas()

    df = df[df["draft_year"].between(START_YEAR, END_YEAR)].copy()

    df = df.rename(columns={
        "pos":        "position",
        "school":     "college",
        "ht":         "height_in",
        "wt":         "weight_lbs",
        "forty":      "forty_yard",
        "bench":      "bench_reps",
        "vertical":   "vertical_jump",
        "broad_jump": "broad_jump",
        "cone":       "cone_drill",
    })

    keep = [
        "pfr_id", "cfb_id", "player_name", "position", "college",
        "draft_year", "draft_round", "draft_ovr", "draft_team",
        "height_in", "weight_lbs", "forty_yard", "bench_reps",
        "vertical_jump", "broad_jump", "cone_drill", "shuttle",
    ]
    df = df[[c for c in keep if c in df.columns]]

    # Normalise cfb_id to string (values can be numeric IDs or slug strings)
    if "cfb_id" in df.columns:
        df["cfb_id"] = df["cfb_id"].where(df["cfb_id"].notna(), other=None).astype(str)
        df.loc[df["cfb_id"] == "None", "cfb_id"] = None
        df.loc[df["cfb_id"] == "nan",  "cfb_id"] = None

    print(f"[nflverse] Combine rows loaded: {len(df)}")
    return df.reset_index(drop=True)


def load_draft_picks() -> pd.DataFrame:
    """
    Load draft picks with career AV and NFL career stats from nflverse.
    Filtered to draft seasons [START_YEAR, END_YEAR].
    """
    df = nfl.load_draft_picks().to_pandas()

    df = df[df["season"].between(START_YEAR, END_YEAR)].copy()

    df = df.rename(columns={
        "season":         "draft_year",
        "pfr_player_id":  "pfr_id",
        "cfb_player_id":  "cfb_id",
        "pfr_player_name":"player_name",
        "round":          "draft_round",
        "pick":           "draft_pick",
        "team":           "draft_team",
        "w_av":           "weighted_av",
        "car_av":         "career_av",
        "dr_av":          "draft_team_av",
        "games":          "nfl_games",
        "probowls":       "pro_bowls",
        "allpro":         "all_pro",
        "seasons_started":"seasons_started",
    })

    keep = [
        "pfr_id", "cfb_id", "gsis_id", "player_name", "position",
        "draft_year", "draft_round", "draft_pick", "draft_team", "college",
        "career_av", "weighted_av", "draft_team_av",
        "nfl_games", "pro_bowls", "all_pro", "seasons_started", "hof",
    ]
    df = df[[c for c in keep if c in df.columns]]

    if "cfb_id" in df.columns:
        df["cfb_id"] = df["cfb_id"].where(df["cfb_id"].notna(), other=None).astype(str)
        df.loc[df["cfb_id"] == "None", "cfb_id"] = None
        df.loc[df["cfb_id"] == "nan",  "cfb_id"] = None

    print(f"[nflverse] Draft pick rows loaded: {len(df)}")
    return df.reset_index(drop=True)
