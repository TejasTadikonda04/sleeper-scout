import os
from dotenv import load_dotenv

load_dotenv()

DB_URL: str = os.getenv("DB_URL", "postgresql://localhost/sleeper_scout")
CFBD_API_KEY: str = os.getenv("CFBD_API_KEY", "")

START_YEAR: int = 2000
END_YEAR: int = 2024

# College seasons to pull from CFBD (4 years before earliest draft class)
CFBD_START_YEAR: int = START_YEAR - 4  # 1996

# CFBD stat categories to fetch
CFBD_STAT_CATEGORIES: list[str] = [
    "passing",
    "rushing",
    "receiving",
    "defensive",
    "kicking",
    "punting",
]

# Maps (cfbd_category, cfbd_stat_type) → nfl_prospects column name.
# Only stat types listed here will be written; all others are silently ignored.
STAT_COLUMN_MAP: dict[tuple[str, str], str] = {
    # Passing
    ("passing", "YDS"):         "col_pass_yards",
    ("passing", "TD"):          "col_pass_tds",
    ("passing", "INT"):         "col_pass_ints",
    ("passing", "ATT"):         "col_pass_attempts",
    ("passing", "COMPLETIONS"): "col_pass_completions",
    # Rushing
    ("rushing", "YDS"):  "col_rush_yards",
    ("rushing", "TD"):   "col_rush_tds",
    ("rushing", "ATT"):  "col_rush_attempts",
    ("rushing", "CAR"):  "col_rush_attempts",  # CFBD uses CAR and ATT interchangeably
    # Receiving
    ("receiving", "YDS"): "col_rec_yards",
    ("receiving", "TD"):  "col_rec_tds",
    ("receiving", "REC"): "col_receptions",
    # Defensive
    ("defensive", "TOT"):    "col_total_tackles",
    ("defensive", "TFL"):    "col_tfl",
    ("defensive", "SACKS"):  "col_sacks",
    ("defensive", "INT"):    "col_ints",
    ("defensive", "PD"):     "col_pass_deflections",
    ("defensive", "QB HUR"): "col_qb_hurries",
    # Kicking
    ("kicking", "FGM"): "col_fg_made",
    ("kicking", "FGA"): "col_fg_attempted",
    ("kicking", "XPM"): "col_xp_made",
    ("kicking", "XPA"): "col_xp_attempted",
    # Punting
    ("punting", "NO"):  "col_punts",
    ("punting", "YDS"): "col_punt_yards",
}

# Common school name mismatches between PFR/nflverse and CFBD
SCHOOL_NAME_MAP: dict[str, str] = {
    "USC":            "Southern California",
    "LSU":            "Louisiana State",
    "TCU":            "Texas Christian",
    "UCF":            "Central Florida",
    "Ole Miss":       "Mississippi",
    "Pitt":           "Pittsburgh",
    "UConn":          "Connecticut",
    "USF":            "South Florida",
    "SMU":            "Southern Methodist",
    "UNLV":           "Nevada-Las Vegas",
    "UNC":            "North Carolina",
    "UMass":          "Massachusetts",
    "UAB":            "Alabama-Birmingham",
    "UTSA":           "Texas-San Antonio",
    "UTEP":           "Texas-El Paso",
    "FIU":            "Florida International",
    "FAU":            "Florida Atlantic",
    "Miami (FL)":     "Miami",
    "Miami (OH)":     "Miami (OH)",
}
