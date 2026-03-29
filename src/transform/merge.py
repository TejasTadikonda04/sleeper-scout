from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz, process as fuzz_process

from .clean import normalize_name, normalize_school
from ..config import STAT_COLUMN_MAP

_FUZZY_THRESHOLD = 85


# ---------------------------------------------------------------------------
# Combine + Draft join
# ---------------------------------------------------------------------------

def merge_combine_and_draft(
    combine: pd.DataFrame,
    draft: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join combine measurables onto draft picks using pfr_id.
    Draft is the left (authoritative) table; every draft row is kept.
    """
    combine_cols = [
        "pfr_id",
        "height_in", "weight_lbs", "forty_yard", "bench_reps",
        "vertical_jump", "broad_jump", "cone_drill", "shuttle",
    ]
    combine_slim = combine[[c for c in combine_cols if c in combine.columns]]

    # Drop rows without a pfr_id and deduplicate before joining to avoid a
    # NULL = NULL Cartesian product and many-to-one inflation.
    combine_slim = (
        combine_slim
        .dropna(subset=["pfr_id"])
        .drop_duplicates(subset=["pfr_id"])
    )

    merged = draft.merge(combine_slim, on="pfr_id", how="left")

    print(f"[merge] Draft + combine rows: {len(merged)}")
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CFBD name-match helpers
# ---------------------------------------------------------------------------

def _build_cfbd_lookup(raw: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    """
    Returns a mapping:  normalised_name → [(cfbd_player_id, team), ...]
    built from the unique (player_id, player_name, team) records in raw.
    """
    unique_players = (
        raw[["player_id", "player_name", "team"]]
        .drop_duplicates(subset=["player_id", "team"])
    )
    lookup: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for _, row in unique_players.iterrows():
        key = normalize_name(str(row["player_name"]))
        lookup[key].append((str(row["player_id"]), str(row["team"])))
    return lookup


def _match_player(
    name: str,
    college: str,
    lookup: dict[str, list[tuple[str, str]]],
    all_keys: list[str],
) -> str | None:
    """
    Return the best CFBD player_id for a given prospect name + college.
    1. Exact normalised-name match → prefer candidate whose team matches college.
    2. Fuzzy name match (score ≥ threshold) → same school preference.
    3. No match → None.
    """
    norm_name = normalize_name(name)
    norm_college = normalize_school(college)

    candidates = lookup.get(norm_name)

    if not candidates:
        result = fuzz_process.extractOne(
            norm_name,
            all_keys,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=_FUZZY_THRESHOLD,
        )
        if result:
            candidates = lookup.get(result[0])

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0][0]

    # Disambiguate by school name similarity
    best_id, best_score = candidates[0][0], -1
    for (cid, team) in candidates:
        score = fuzz.token_sort_ratio(normalize_school(team), norm_college)
        if score > best_score:
            best_score, best_id = score, cid
    return best_id


def _build_prospect_cfbd_map(
    prospects: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns a DataFrame [pfr_id, cfbd_player_id] mapping each prospect
    to their best-matched CFBD player_id.
    """
    lookup = _build_cfbd_lookup(raw)
    all_keys = list(lookup.keys())

    rows = []
    seen_pfr: set = set()
    for _, row in prospects.iterrows():
        pfr_id = row.get("pfr_id")
        # Skip rows without a pfr_id — null keys cause cartesian-product fan-out on merge
        if pd.isna(pfr_id) or pfr_id is None or str(pfr_id) in ("nan", "None", ""):
            continue
        if str(pfr_id) in seen_pfr:
            continue
        seen_pfr.add(str(pfr_id))
        cfbd_id = _match_player(
            str(row.get("player_name", "")),
            str(row.get("college", "")),
            lookup,
            all_keys,
        )
        if cfbd_id:
            rows.append({"pfr_id": pfr_id, "cfbd_player_id": cfbd_id})

    df = pd.DataFrame(rows)
    print(f"[merge] Prospects matched to a CFBD player_id: {len(df)} / {len(prospects)}")
    return df


# ---------------------------------------------------------------------------
# College stats aggregation
# ---------------------------------------------------------------------------

def _aggregate_college_stats(
    raw: pd.DataFrame,
    id_map: pd.DataFrame,
    draft_year_map: dict[str, int],
) -> pd.DataFrame:
    """
    Restrict raw stats to seasons before each player's draft year,
    pivot to wide format, and return a DataFrame keyed on cfbd_player_id.
    """
    raw = raw.merge(id_map, on=[], how="inner") if id_map.empty else raw.copy()

    if id_map.empty:
        return pd.DataFrame(columns=["cfbd_player_id"])

    # Only keep CFBD player_ids that appear in our map
    target_ids = set(id_map["cfbd_player_id"].astype(str))
    raw = raw[raw["player_id"].isin(target_ids)].copy()

    if raw.empty:
        return pd.DataFrame(columns=["cfbd_player_id"])

    # Restrict each player's stats to seasons < their draft year
    # Build a player_id → draft_year map via id_map
    pid_to_draft: dict[str, int] = {}
    for _, row in id_map.iterrows():
        dy = draft_year_map.get(str(row["pfr_id"]))
        if dy is not None:
            pid_to_draft[str(row["cfbd_player_id"])] = dy

    raw["draft_year_cutoff"] = raw["player_id"].map(pid_to_draft)
    raw = raw[raw["season"] < raw["draft_year_cutoff"]]

    # Map (category, stat_type) → column name
    raw["col"] = raw.apply(
        lambda r: STAT_COLUMN_MAP.get((r["category"], str(r["stat_type"]).upper())),
        axis=1,
    )
    raw = raw.dropna(subset=["col"])

    if raw.empty:
        return pd.DataFrame(columns=["cfbd_player_id"])

    # Guard: ensure stat is numeric before summing. If the ingest layer already
    # coerced it this is a no-op; it prevents string-concatenation bugs if the
    # cache was written before the fix was applied.
    raw = raw.copy()
    raw["stat"] = pd.to_numeric(raw["stat"], errors="coerce")

    agg = (
        raw.groupby(["player_id", "col"])["stat"]
        .sum()
        .reset_index()
    )

    wide = agg.pivot_table(
        index="player_id", columns="col", values="stat", aggfunc="sum"
    ).reset_index()
    wide = wide.rename(columns={"player_id": "cfbd_player_id"})

    # Ensure all expected columns exist
    for col in set(STAT_COLUMN_MAP.values()):
        if col not in wide.columns:
            wide[col] = None

    # Cast to nullable integer where appropriate
    for c in [col for col in wide.columns if col.startswith("col_")]:
        try:
            wide[c] = pd.to_numeric(wide[c], errors="coerce").round(1)
        except TypeError:
            pass

    print(f"[merge] CFBD career stat rows (wide): {len(wide)}")
    return wide.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def merge_college_stats(
    prospects: pd.DataFrame,
    raw_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match prospects to CFBD players by name, aggregate college career stats,
    and left-join them back onto the prospects DataFrame.
    """
    if raw_stats.empty:
        return prospects

    # Build name-based prospect → CFBD player_id map
    id_map = _build_prospect_cfbd_map(prospects, raw_stats)

    # Build pfr_id → draft_year lookup
    draft_year_map = dict(zip(
        prospects["pfr_id"].astype(str),
        prospects["draft_year"].astype(int),
    ))

    # Aggregate college careers
    college_wide = _aggregate_college_stats(raw_stats, id_map, draft_year_map)

    if college_wide.empty or "cfbd_player_id" not in college_wide.columns:
        return prospects

    # Attach cfbd_player_id to prospects via id_map
    prospects = prospects.merge(id_map, on="pfr_id", how="left")

    # Join aggregated stats
    prospects = prospects.merge(
        college_wide,
        on="cfbd_player_id",
        how="left",
    )
    prospects = prospects.drop(columns=["cfbd_player_id"], errors="ignore")

    matched = prospects[[c for c in college_wide.columns if c.startswith("col_")]].notna().any(axis=1).sum()
    print(f"[merge] Prospects with at least one college stat: {matched}")
    return prospects.reset_index(drop=True)


def build_final_table(
    combine: pd.DataFrame,
    draft: pd.DataFrame,
    raw_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full merge pipeline: combine + draft + college stats → final prospects table.
    """
    prospects = merge_combine_and_draft(combine, draft)
    prospects = merge_college_stats(prospects, raw_stats)

    # Rename draft_ovr → draft_pick when missing
    if "draft_ovr" in prospects.columns and "draft_pick" not in prospects.columns:
        prospects = prospects.rename(columns={"draft_ovr": "draft_pick"})
    elif "draft_ovr" in prospects.columns:
        prospects = prospects.drop(columns=["draft_ovr"], errors="ignore")

    prospects = prospects.dropna(subset=["player_name"])

    # Deduplicate: keep the row with the most non-null values for each player
    dedup_keys = ["pfr_id", "player_name", "draft_year", "draft_pick"]
    dedup_keys = [k for k in dedup_keys if k in prospects.columns]
    prospects["_notnull_count"] = prospects.notna().sum(axis=1)
    prospects = (
        prospects
        .sort_values("_notnull_count", ascending=False)
        .drop_duplicates(subset=dedup_keys, keep="first")
        .drop(columns=["_notnull_count"])
    )

    print(f"[merge] Final table rows: {len(prospects)}")
    return prospects.reset_index(drop=True)
