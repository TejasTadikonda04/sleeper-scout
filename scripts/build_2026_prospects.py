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
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rapidfuzz import fuzz, process as fuzz_process

from src.config import CFBD_API_KEY
from src.ingest.cfbd import fetch_raw_college_stats
from src.transform.clean import normalize_name, normalize_school, normalize_position
from src.transform.merge import merge_college_stats

COMBINE_DIR  = Path(__file__).parent.parent / "2026_combine"
OUTPUT_PATH  = Path(__file__).parent.parent / "2026_prospects.csv"
CACHE_PATH   = Path(__file__).parent.parent / ".2026_cfbd_cache.parquet"

MOCK_DRAFTS_DIR = Path(__file__).parent.parent / "mock_drafts"

# Projected 2026 rookie wage scale (total contract value by pick number).
# Based on the 2024 scale with a 15.5% inflation multiplier (~$295M cap).
ROOKIE_WAGE_SCALE: dict[int, int] = {
    1:45600000,2:43600000,3:42300000,4:40900000,5:38100000,6:35700000,
    7:33300000,8:30900000,9:28500000,10:26200000,11:24900000,12:23600000,
    13:22300000,14:21000000,15:19800000,16:18600000,17:18300000,18:18000000,
    19:17800000,20:17500000,21:17200000,22:17000000,23:16700000,24:16500000,
    25:16200000,26:15900000,27:15700000,28:15400000,29:15200000,30:14900000,
    31:14600000,32:14400000,33:14100000,34:13900000,35:13700000,36:13400000,
    37:13200000,38:13000000,39:12800000,40:12600000,41:12300000,42:12100000,
    43:11900000,44:11700000,45:11500000,46:11200000,47:11000000,48:10800000,
    49:10600000,50:10400000,51:10200000,52:9900000,53:9700000,54:9500000,
    55:9300000,56:9100000,57:8800000,58:8600000,59:8400000,60:8200000,
    61:8000000,62:7800000,63:7600000,64:7400000,65:7360000,66:7330000,
    67:7300000,68:7260000,69:7230000,70:7200000,71:7160000,72:7130000,
    73:7100000,74:7060000,75:7030000,76:7000000,77:6960000,78:6930000,
    79:6900000,80:6860000,81:6830000,82:6800000,83:6760000,84:6730000,
    85:6700000,86:6660000,87:6630000,88:6600000,89:6560000,90:6530000,
    91:6500000,92:6460000,93:6430000,94:6400000,95:6360000,96:6330000,
    97:6300000,98:6260000,99:6230000,100:6200000,101:6180000,102:6160000,
    103:6140000,104:6120000,105:6100000,106:6080000,107:6060000,108:6040000,
    109:6020000,110:6000000,111:5980000,112:5960000,113:5940000,114:5920000,
    115:5900000,116:5880000,117:5860000,118:5840000,119:5820000,120:5800000,
    121:5780000,122:5760000,123:5740000,124:5720000,125:5700000,126:5680000,
    127:5660000,128:5640000,129:5620000,130:5600000,131:5580000,132:5560000,
    133:5540000,134:5520000,135:5500000,136:5480000,137:5460000,138:5440000,
    139:5420000,140:5400000,141:5380000,142:5360000,143:5340000,144:5320000,
    145:5300000,146:5280000,147:5260000,148:5240000,149:5220000,150:5200000,
    151:5190000,152:5180000,153:5170000,154:5160000,155:5150000,156:5140000,
    157:5130000,158:5120000,159:5110000,160:5100000,161:5090000,162:5080000,
    163:5070000,164:5060000,165:5050000,166:5040000,167:5030000,168:5020000,
    169:5010000,170:5000000,171:4990000,172:4980000,173:4970000,174:4960000,
    175:4950000,176:4940000,177:4930000,178:4920000,179:4910000,180:4900000,
    181:4900000,182:4900000,183:4900000,184:4900000,185:4900000,186:4900000,
    187:4900000,188:4900000,189:4900000,190:4900000,191:4900000,192:4900000,
    193:4900000,194:4900000,195:4900000,196:4900000,197:4900000,198:4900000,
    199:4900000,200:4900000,201:4890000,202:4890000,203:4890000,204:4880000,
    205:4880000,206:4880000,207:4870000,208:4870000,209:4870000,210:4860000,
    211:4860000,212:4860000,213:4850000,214:4850000,215:4850000,216:4840000,
    217:4840000,218:4840000,219:4830000,220:4830000,221:4830000,222:4820000,
    223:4820000,224:4820000,225:4810000,226:4810000,227:4810000,228:4800000,
    229:4800000,230:4800000,231:4790000,232:4790000,233:4790000,234:4780000,
    235:4780000,236:4780000,237:4770000,238:4770000,239:4770000,240:4760000,
    241:4760000,242:4760000,243:4750000,244:4750000,245:4750000,246:4740000,
    247:4740000,248:4740000,249:4730000,250:4730000,251:4730000,252:4720000,
    253:4720000,254:4720000,255:4710000,256:4710000,257:4700000,
}

DRAFT_YEAR   = 2026
# Fetch college stats back to 2018 — earliest a 2026 draftee could have played
CFBD_FETCH_START = 2018

COLUMN_ORDER = [
    "pfr_id", "cfb_id", "gsis_id", "player_name", "position",
    "draft_year", "projected_draft_pick", "projected_contract_value",
    "draft_round", "draft_pick", "draft_team", "college",
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
# Mock draft ADP
# ---------------------------------------------------------------------------

def _parse_mock_file(path: Path) -> list[tuple[int, str]]:
    """
    Parse a mock draft text file of the form:
        1. Player Name
        2. Player Name
        ...
    Returns a list of (pick_number, normalised_name) tuples.
    """
    picks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\.\s+(.+)$", line)
        if m:
            pick = int(m.group(1))
            name = normalize_name(m.group(2).strip())
            picks.append((pick, name))
    return picks


def load_all_mock_drafts() -> dict[str, list[tuple[int, str]]]:
    """Return {filename: [(pick, normalised_name), ...]} for every mock draft."""
    mocks = {}
    for txt in sorted(MOCK_DRAFTS_DIR.rglob("*.txt")):
        picks = _parse_mock_file(txt)
        if picks:
            mocks[str(txt.relative_to(MOCK_DRAFTS_DIR))] = picks
            print(f"  Loaded {len(picks):3d} picks  ← {txt.relative_to(MOCK_DRAFTS_DIR)}")
    return mocks


def compute_adp(
    prospects: pd.DataFrame,
    mocks: dict[str, list[tuple[int, str]]],
) -> pd.Series:
    """
    For each prospect, average their pick number across every mock where they appear.

    Matching strategy (in order):
      1. Exact normalised-name match.
      2. Fuzzy token_set_ratio ≥ 80 against the prospect's normalised name
         (handles "Vega Ioane" → "Olaivavega Ioane", suffix differences, etc.).

    Returns a Series of float ADP values indexed like `prospects`, with NaN for
    players who appear in no mock.
    """
    # Build a flat list of all (pick, norm_name) pairs across all mocks,
    # keeping one entry per (mock_file, player) — some mocks list a player twice
    # (rare, but guard against it).
    mock_appearances: list[tuple[str, int, str]] = []   # (source, pick, norm_name)
    for source, picks in mocks.items():
        seen_in_mock: set[str] = set()
        for pick, norm_name in picks:
            if norm_name not in seen_in_mock:
                mock_appearances.append((source, pick, norm_name))
                seen_in_mock.add(norm_name)

    # Unique normalised names that appear in any mock
    all_mock_names = list({name for _, _, name in mock_appearances})

    # For each prospect, collect all pick numbers they appear at
    prospect_norm_names = prospects["player_name"].apply(normalize_name).tolist()

    # name → list of (source, pick)
    prospect_picks: dict[int, list[tuple[str, int]]] = defaultdict(list)

    for source, pick, mock_name in mock_appearances:
        # Step 1: exact match
        for idx, pname in enumerate(prospect_norm_names):
            if mock_name == pname:
                prospect_picks[idx].append((source, pick))
                break
        else:
            # Step 2: fuzzy match against prospect names
            result = fuzz_process.extractOne(
                mock_name,
                prospect_norm_names,
                scorer=fuzz.token_set_ratio,
                score_cutoff=80,
            )
            if result:
                matched_name, score, idx = result
                prospect_picks[idx].append((source, pick))

    # Compute average pick per prospect
    adp_values = []
    for idx in range(len(prospects)):
        appearances = prospect_picks.get(idx, [])
        if appearances:
            avg = sum(p for _, p in appearances) / len(appearances)
            adp_values.append(round(avg, 1))
        else:
            adp_values.append(None)

    return pd.Series(adp_values, index=prospects.index, name="projected_draft_pick")


# ---------------------------------------------------------------------------
# Contract value lookup
# ---------------------------------------------------------------------------

def adp_to_contract_value(adp: float | None) -> int | None:
    """
    Linearly interpolate the rookie wage scale for a fractional ADP.
    e.g. ADP 5.4 → 60% of pick-5 value + 40% of pick-6 value.
    ADPs beyond pick 257 or null return None.
    """
    if adp is None or pd.isna(adp):
        return None
    lo = int(adp)
    hi = lo + 1
    frac = adp - lo
    val_lo = ROOKIE_WAGE_SCALE.get(lo)
    val_hi = ROOKIE_WAGE_SCALE.get(hi)
    if val_lo is None:
        return None
    if val_hi is None:
        return val_lo
    return round(val_lo + frac * (val_hi - val_lo))


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
    print("\n=== Step 2/4: Fetching CFBD college stats ===")
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

    # ── Step 3: Mock draft ADP ─────────────────────────────────────────────
    print("\n=== Step 3/4: Computing projected draft pick from mock drafts ===")
    mocks = load_all_mock_drafts()
    prospects["projected_draft_pick"] = compute_adp(prospects, mocks)
    matched_adp = prospects["projected_draft_pick"].notna().sum()
    print(f"  {matched_adp}/{len(prospects)} prospects matched to at least one mock draft")
    prospects["projected_contract_value"] = prospects["projected_draft_pick"].apply(
        adp_to_contract_value
    )

    # ── Step 4: Merge college stats + write ───────────────────────────────
    print("\n=== Step 4/4: Merging college stats and writing CSV ===")

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
