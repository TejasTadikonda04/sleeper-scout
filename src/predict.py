"""
predict.py — load saved data, train models, and save visualizations.

Requires data files produced by ingest.py:
  data/combine_av.csv
  data/combine_college.csv

Run anytime:
  venv/Scripts/python src/predict.py
"""

import os
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
COMBINE_FEATURES = ["ht_in", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]
COLLEGE_FEATURES = ["pass_yds", "rush_yds", "rec_yds", "total_td", "tackles", "sacks"]


def train_and_plot(df, label, features, filename):
    df = df[features + ["car_av"]].copy()
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    print(f"\n{label}")
    print(f"  Training rows: {len(df)}")

    if len(df) < 10:
        print("  Not enough data to train.")
        return

    X = df[features].values
    y = df["car_av"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    r2 = r2_score(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    print(f"  R\u00b2:   {r2:.3f}")
    print(f"  RMSE: {rmse:.2f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_te, preds, alpha=0.35, s=18, color="steelblue", label="Players")
    lo = min(float(y_te.min()), float(preds.min()))
    hi = max(float(y_te.max()), float(preds.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Weighted Career AV (PFR)")
    ax.set_ylabel("Predicted Weighted Career AV (PFR)")
    ax.set_title(f"{label}\nR\u00b2 = {r2:.3f}  |  RMSE = {rmse:.2f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def main():
    combine_path = os.path.join(DATA_DIR, "combine_av.csv")
    college_path = os.path.join(DATA_DIR, "combine_college.csv")

    if not os.path.exists(combine_path) or not os.path.exists(college_path):
        print("Data files not found. Run src/ingest.py first.")
        return

    combine_df = pd.read_csv(combine_path)
    college_df = pd.read_csv(college_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loaded {len(combine_df)} players from {combine_path}")
    print(f"Loaded {len(college_df)} players from {college_path}")

    avail1 = [f for f in COMBINE_FEATURES if f in combine_df.columns]
    train_and_plot(combine_df, "Model 1: Combine Only", avail1, os.path.join(OUTPUT_DIR, "plot_model1_combine.png"))

    avail2 = [f for f in COMBINE_FEATURES if f in college_df.columns] + COLLEGE_FEATURES
    train_and_plot(college_df, "Model 2: Combine + College Stats", avail2, os.path.join(OUTPUT_DIR, "plot_model2_combine_college.png"))


if __name__ == "__main__":
    main()
