import os
import pandas as pd
from av_share_predictor import AVSharePredictor, AVShareForecaster
from team_draft_grader import TeamDraftGrader

# Load your data (with av_share already computed)
df = pd.read_csv('combine_av_share.csv')

# 1. Train AV Share predictor
os.makedirs('models', exist_ok=True)
predictor = AVSharePredictor()
metrics = predictor.train_position_models(df)
predictor.save_model('models/av_share_predictor.joblib')

print("Model Performance:")
for pos, metric in metrics.items():
    print(f"{pos}: R²={metric.r2:.3f}, Spearman={metric.spearman:.3f}")

# 2. Forecast new prospects (pass DataFrame with same columns as combine_av_share for real use)
forecaster = AVShareForecaster(predictor)
new_prospects = df.head(0).copy()  # Empty placeholder; replace with 2026 prospects when ready
if len(new_prospects) == 0:
    print("Skipping forecast (no prospects provided).")
else:
    forecasts = forecaster.forecast_5yr_av_share(new_prospects)
    print(forecasts.head())

# 3. Grade team drafting
grader = TeamDraftGrader(df)
team_grades = grader.calculate_team_grades()
os.makedirs('results', exist_ok=True)
grader.plot_team_rankings(team_grades, 'results/team_draft_grades.png')

print("\nTop drafting teams:")
print(team_grades[['team', 'value_over_expected', 'percentile_rank']].head())
