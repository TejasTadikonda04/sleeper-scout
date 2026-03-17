"""
Violin plot: Career AV (car_av) by position group for draft classes 2010-2023.
Saves to combine_eda_plots/av_violin_by_position.png

Run: python plot_av_by_position.py
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from plot_style import apply_dark_style

# Match av_share_predictor position groups
POS_GROUP = {
    "QB": "QB",
    "RB": "SKILL", "WR": "SKILL", "TE": "SKILL",
    "OT": "OL", "OG": "OL", "C": "OL",
    "DT": "DL", "DE": "DL", "EDGE": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB",
    "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB",
}


def main():
    df = pd.read_csv("combine_av_share.csv")
    df = df[(df["draft_year"] >= 2010) & (df["draft_year"] <= 2023)].copy()
    df["pos_group"] = df["pos"].map(lambda p: POS_GROUP.get(p, "OTHER"))

    apply_dark_style()
    order = ["QB", "SKILL", "OL", "DL", "LB", "DB", "OTHER"]
    order = [g for g in order if (df["pos_group"] == g).any()]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=df,
        x="pos_group",
        y="car_av",
        order=order,
        palette="muted",
        cut=0,
        ax=ax,
    )
    ax.set_xlabel("Position Group")
    ax.set_ylabel("Career AV")
    ax.set_title("Career AV by Position (Draft Classes 2010-2023)")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("white")
    plt.xticks(rotation=15)
    plt.tight_layout()

    out_dir = Path("combine_eda_plots")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "av_violin_by_position.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
