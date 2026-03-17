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
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# --- Theme Configuration ---
BG = "#1C1C1C"

# Apply global dark mode and clean sans-serif font settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
})

# dark violet → deep purple → vibrant magenta → fiery orange → gold/yellow
CMAP = LinearSegmentedColormap.from_list("av_cmap", [
    "#0D0221",
    "#6A0572",
    "#E0006B",
    "#FF6B00",
    "#FFD700",
])

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

COMBINE_FEATURES = ["ht_in", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]
COLLEGE_FEATURES = ["pass_yds", "rush_yds", "rec_yds", "total_td", "tackles", "sacks"]


def train_and_plot(df, features, title, filename):
    # Keep the data logic exactly as you had it
    df = df[features + ["car_av"]].copy()
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    print(f"\n{title}")
    print(f"  Training rows: {len(df)}")

    X = df[features].values
    y = df["car_av"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    r2   = r2_score(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    print(f"  R\u00b2:   {r2:.3f}")
    print(f"  RMSE: {rmse:.2f}")

    preds = np.clip(preds, 0, None)

    # Calculate log-based colors for the gradient
    log_av = np.log1p(np.clip(y_te, 0, None))
    vmin, vmax = log_av.min(), log_av.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=(9, 7))

    # Scatter points - bright, discrete, no borders
    ax.scatter(preds, y_te, c=log_av, cmap=CMAP, norm=norm,
               s=18, alpha=0.85, linewidths=0, zorder=3)

    # Trend line clipped so it never goes below y=0
    m, b = np.polyfit(preds, y_te, 1)
    x_line = np.linspace(preds.min(), preds.max(), 300)
    y_line = m * x_line + b
    mask = y_line >= 0
    ax.plot(x_line[mask], y_line[mask], linestyle="--",
            color="#FF69B4", linewidth=2.2, alpha=0.85, zorder=3)

    # Labels & Title
    ax.set_xlabel("Predicted Weighted Career AV", fontsize=11, labelpad=10)
    ax.set_ylabel("Actual Weighted Career AV", fontsize=11, labelpad=10)
    ax.set_title(
        f"{title}\nR\u00b2 = {r2:.3f}  \u2022  RMSE = {rmse:.2f}  \u2022  n = {len(df)}",
        fontsize=13, pad=16, fontweight='bold'
    )

    # Axis Limits & Formatting
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-5)
    
    # Hide ticks but keep labels
    ax.tick_params(axis='both', which='both', length=0, labelsize=9)
    
    # Very faint gridlines behind the data
    ax.grid(True, color="#FFFFFF", alpha=0.08, linewidth=0.5, zorder=0)
    
    # Remove border spines completely for a clean look
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, aspect=35)
    
    tick_vals = np.arange(int(np.floor(vmin)), int(np.ceil(vmax)) + 1)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([str(t) for t in tick_vals])
    cbar.set_label("Career AV (Log Scale)", fontsize=10, labelpad=14)
    cbar.ax.yaxis.set_tick_params(length=0, labelsize=9)
    cbar.outline.set_visible(False) # Remove colorbar border

    plt.tight_layout()

    fig.canvas.draw()
    ax_pos = ax.get_position()

    # White border around the scatter plot area only
    fig.add_artist(plt.Rectangle(
        (ax_pos.x0, ax_pos.y0),
        ax_pos.width, ax_pos.height,
        fill=False, edgecolor="white", linewidth=1.5,
        transform=fig.transFigure, clip_on=False, zorder=10,
    ))

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved: {os.path.relpath(path, ROOT_DIR)}")


def main():
    combine_path = os.path.join(DATA_DIR, "combine_av.csv")
    college_path = os.path.join(DATA_DIR, "combine_college.csv")

    if not os.path.exists(combine_path) or not os.path.exists(college_path):
        print("Data files not found. Run src/ingest.py first.")
        return

    combine_df = pd.read_csv(combine_path)
    college_df = pd.read_csv(college_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loaded {len(combine_df)} players")

    avail_combine = [f for f in COMBINE_FEATURES if f in combine_df.columns]
    avail_college = [f for f in COMBINE_FEATURES if f in college_df.columns] + COLLEGE_FEATURES

    train_and_plot(combine_df, avail_combine,
                   "Linear Regression — Combine Only",
                   "plot_lr_combine.png")

    train_and_plot(college_df, avail_college,
                   "Linear Regression — Combine + College Stats",
                   "plot_lr_combine_college.png")


if __name__ == "__main__":
    main()