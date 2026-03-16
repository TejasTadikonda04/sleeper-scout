"""
Compare combine_av vs combine_college for predicting future success (AV share).

Uses the same target (av_share), same train/val split, and same sample of players
so the only difference is feature set: combine-only vs combine + college stats.

Usage:
    python compare_datasets.py

Output:
    - Metrics table (R², Spearman, MAE, RMSE) per position for each dataset
    - Summary: which dataset wins overall (e.g. by mean Spearman across positions)
"""
import os
import pandas as pd
from av_share_predictor import AVSharePredictor

COLLEGE_FEATURES = ['pass_yds', 'rush_yds', 'rec_yds', 'total_td', 'tackles', 'sacks']


def add_av_share(df: pd.DataFrame) -> pd.DataFrame:
    """Add av_share column (car_av / total class AV per draft year)."""
    df = df.copy()
    class_totals = df.groupby('draft_year')['car_av'].sum().reset_index()
    class_totals.columns = ['draft_year', 'total_class_av']
    df = df.merge(class_totals, on='draft_year', how='left')
    df['av_share'] = df['car_av'] / df['total_class_av']
    return df


def main():
    # 1. Load combine-only data (with av_share)
    combine_av_share_path = 'combine_av_share.csv'
    if not os.path.exists(combine_av_share_path):
        raise FileNotFoundError(
            f"Run compute_av_share.py first to create {combine_av_share_path} from combine_av.csv"
        )
    df_combine = pd.read_csv(combine_av_share_path)

    # 2. Load combine + college and add av_share
    df_college = pd.read_csv('combine_college.csv')
    df_college = add_av_share(df_college)

    # 3. Restrict to rows that have all combine AND all college features (same sample for fair comparison)
    for col in COLLEGE_FEATURES:
        if col not in df_college.columns:
            raise ValueError(f"combine_college.csv missing column: {col}")
    # Raw combine columns (engineered ones like bmi are created inside prepare_features)
    raw_combine = ['ht_in', 'wt', 'forty', 'bench', 'vertical', 'broad_jump', 'cone', 'shuttle']
    subset = raw_combine + COLLEGE_FEATURES + ['av_share', 'draft_year', 'pos']
    df_common = df_college.dropna(subset=subset)
    print(f"Common sample (non-null combine + college): {len(df_common)} rows (from {len(df_college)} combine_college)")

    # 4. Train combine-only model on common sample
    predictor_combine = AVSharePredictor(extra_numeric_features=[])
    metrics_combine = predictor_combine.train_position_models(df_common)

    # 5. Train combine+college model on same common sample
    predictor_college = AVSharePredictor(extra_numeric_features=COLLEGE_FEATURES)
    metrics_college = predictor_college.train_position_models(df_common)

    # 6. Align position groups (both may skip small groups)
    positions = sorted(set(metrics_combine.keys()) | set(metrics_college.keys()))
    if not positions:
        print("No position groups had enough data.")
        return

    # 7. Print comparison table
    print("\n" + "=" * 80)
    print("Dataset comparison: Combine-only vs Combine+College (predicting AV share)")
    print("=" * 80)
    print(f"{'Position':<8} {'Dataset':<18} {'R²':>8} {'Spearman':>10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 80)

    for pos in positions:
        m_comb = metrics_combine.get(pos)
        m_coll = metrics_college.get(pos)
        if m_comb:
            print(f"{pos:<8} {'Combine only':<18} {m_comb.r2:>8.3f} {m_comb.spearman:>10.3f} {m_comb.mae:>10.4f} {m_comb.rmse:>10.4f}")
        if m_coll:
            print(f"{'':8} {'Combine+College':<18} {m_coll.r2:>8.3f} {m_coll.spearman:>10.3f} {m_coll.mae:>10.4f} {m_coll.rmse:>10.4f}")
        if m_comb or m_coll:
            print("-" * 80)

    # 8. Summary: compare by mean Spearman (and optionally mean R²) over shared positions
    shared = [p for p in positions if p in metrics_combine and p in metrics_college]
    if shared:
        mean_spearman_comb = sum(metrics_combine[p].spearman for p in shared) / len(shared)
        mean_spearman_coll = sum(metrics_college[p].spearman for p in shared) / len(shared)
        mean_r2_comb = sum(metrics_combine[p].r2 for p in shared) / len(shared)
        mean_r2_coll = sum(metrics_college[p].r2 for p in shared) / len(shared)
        print("\nSummary (mean over shared position groups):")
        print(f"  Combine only:     Spearman = {mean_spearman_comb:.3f},  R² = {mean_r2_comb:.3f}")
        print(f"  Combine+College:  Spearman = {mean_spearman_coll:.3f},  R² = {mean_r2_coll:.3f}")
        if mean_spearman_coll > mean_spearman_comb:
            print("  → Combine+College has better ranking prediction (Spearman).")
        else:
            print("  → Combine only has better or equal ranking prediction (Spearman).")
    print()


if __name__ == "__main__":
    main()
