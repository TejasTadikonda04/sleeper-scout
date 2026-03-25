import re
import unicodedata

import pandas as pd

from ..config import SCHOOL_NAME_MAP

# Canonical position codes used in the final table
_POSITION_MAP = {
    "FB":   "RB",
    "HB":   "RB",
    "WB":   "WR",
    "SE":   "WR",
    "FL":   "WR",
    "SLB":  "LB",
    "WLB":  "LB",
    "ILB":  "LB",
    "MLB":  "LB",
    "OLB":  "LB",
    "NT":   "DT",
    "DL":   "DT",
    "DE/DT":"DE",
    "SAF":  "S",
    "SS":   "S",
    "FS":   "S",
    "CB/S": "CB",
    "DB":   "CB",
    "OG":   "G",
    "OT":   "T",
    "OC":   "C",
    "OL":   "G",   # fallback
    "LS":   "LS",
    "P/K":  "P",
}


def normalize_name(name: str) -> str:
    """Lower-case, strip accents, remove punctuation and suffixes."""
    if not isinstance(name, str):
        return ""
    # Strip accents
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    # Remove common suffixes
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv|v)$", "", name, flags=re.IGNORECASE)
    # Remove punctuation except spaces and hyphens
    name = re.sub(r"[^a-z0-9\s\-]", "", name.lower())
    return name.strip()


def normalize_school(school: str) -> str:
    """Normalise school name using SCHOOL_NAME_MAP, then lower-case for fuzzy use."""
    if not isinstance(school, str):
        return ""
    mapped = SCHOOL_NAME_MAP.get(school, school)
    return mapped.lower().strip()


def normalize_position(pos: str) -> str:
    """Map variant position labels to a canonical set."""
    if not isinstance(pos, str):
        return pos
    return _POSITION_MAP.get(pos.upper(), pos.upper())


def _height_to_inches(val) -> float | None:
    """Convert '6-5' or '6-05' style height strings to total inches."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    m = re.match(r"^(\d+)['\-](\d+)\"?$", str(val).strip())
    if m:
        return float(m.group(1)) * 12 + float(m.group(2))
    return None


def clean_combine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "position" in df.columns:
        df["position"] = df["position"].apply(normalize_position)
    if "college" in df.columns:
        df["college"] = df["college"].apply(normalize_school)
    if "height_in" in df.columns:
        df["height_in"] = df["height_in"].apply(_height_to_inches)
    return df


def clean_draft_picks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "position" in df.columns:
        df["position"] = df["position"].apply(normalize_position)
    if "college" in df.columns:
        df["college"] = df["college"].apply(normalize_school)
    return df
