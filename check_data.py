import pandas as pd
import os

paths = [
    "sleeper-scout-rayanUI/actual_app/backend/data/scores/skill_pass_scored.csv",
    "sleeper-scout-rayanUI/actual_app/backend/data/scores/skill_run_scored.csv"
]

def test_thresholds(score, surplus):
    if score > 75: return 'High Upside'
    if score > 55:
        if surplus > 5: return 'Safe Floor'
        return 'Boom or Bust'
    if score > 40: return 'Developmental'
    return 'Overdrafted'

all_counts = {}

for path in paths:
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        tiers = df.apply(lambda x: test_thresholds(x['sleeper_score'], x['surplus_value']), axis=1)
        counts = tiers.value_counts().to_dict()
        print(f"\n--- {path} ---")
        for tier, count in sorted(counts.items()):
            print(f"{tier}: {count}")
            all_counts[tier] = all_counts.get(tier, 0) + count

print("\n--- TOTALS ---")
for tier, count in sorted(all_counts.items()):
    print(f"{tier}: {count}")
