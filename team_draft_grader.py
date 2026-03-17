import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
from dataclasses import dataclass

from plot_style import (
    AXES_EDGE,
    apply_dark_style,
    get_purple_yellow_cmap,
    trend_line_color,
    trend_line_style,
)

# Directory containing team logo images (e.g. chiefs.png, 49ers.png)
NFL_LOGOS_DIR = Path(__file__).resolve().parent / "NFL_logos"

# Full team name -> logo filename in NFL_logos (no path)
TEAM_LOGO_FILES: Dict[str, str] = {
    "Arizona Cardinals": "cardinals.png",
    "Atlanta Falcons": "falcons.png",
    "Baltimore Ravens": "ravens.png",
    "Buffalo Bills": "bills.png",
    "Carolina Panthers": "panthers.png",
    "Chicago Bears": "bears.png",
    "Cincinnati Bengals": "bengals.png",
    "Cleveland Browns": "browns.png",
    "Dallas Cowboys": "cowboys.png",
    "Denver Broncos": "broncos.png",
    "Detroit Lions": "lions.png",
    "Green Bay Packers": "packers.png",
    "Houston Texans": "texans.png",
    "Indianapolis Colts": "colts.png",
    "Jacksonville Jaguars": "jaguars.png",
    "Kansas City Chiefs": "chiefs.png",
    "Las Vegas Raiders": "raiders.png",
    "Los Angeles Chargers": "chargers.png",
    "Los Angeles Rams": "rams.png",
    "Miami Dolphins": "dolphins.png",
    "Minnesota Vikings": "vikings.png",
    "New England Patriots": "patriots.png",
    "New Orleans Saints": "saints.png",
    "New York Giants": "giants.png",
    "New York Jets": "jets.png",
    "Oakland Raiders": "raiders.png",
    "Philadelphia Eagles": "eagles.png",
    "Pittsburgh Steelers": "steelers.png",
    "San Diego Chargers": "chargers.png",
    "San Francisco 49ers": "49ers.png",
    "Seattle Seahawks": "seahawks.png",
    "St. Louis Rams": "rams.png",
    "Tampa Bay Buccaneers": "buccaneers.png",
    "Tennessee Titans": "titans.png",
    "Washington Commanders": "redskins.png",
    "Washington Football Team": "redskins.png",
    "Washington Redskins": "redskins.png",
}

# Standard NFL team abbreviations for chart labels
TEAM_ABBREV = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Oakland Raiders": "OAK",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Diego Chargers": "SD",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "St. Louis Rams": "STL",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",
    "Washington Redskins": "WSH",
}


@dataclass
class DraftGrade:
    team: str
    total_picks: int
    avg_av_share: float
    expected_av_share: float
    value_over_expected: float
    draft_capital_spent: float
    percentile_rank: float

class TeamDraftGrader:
    """Grades NFL teams on drafting efficiency"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.expected_av_by_slot = self._build_expected_av_curve()
        
    def _build_expected_av_curve(self) -> Dict[int, float]:
        """Historical AV Share by draft slot range"""
        # Group picks into buckets matching Medium article
        self.df['pick_bucket'] = pd.cut(self.df['draft_ovr'], 
                                       bins=[0, 5, 10, 20, 32, 50, 100, 150, 262],
                                       labels=['1-5', '6-10', '11-20', '21-32', 
                                              '33-50', '51-100', '101-150', '151-262'])
        
        expected = self.df.groupby('pick_bucket')['av_share'].mean().to_dict()
        return expected
    
    def calculate_team_grades(self) -> pd.DataFrame:
        """Main grading function"""
        results = []
        
        for team in self.df['draft_team'].unique():
            team_df = self.df[self.df['draft_team'] == team]
            
            if len(team_df) < 10:  # Skip teams with few picks
                continue
                
            # Actual performance
            actual_av_share = team_df['av_share'].sum()
            n_picks = len(team_df)
            
            # Expected performance by slot
            expected_av = 0
            for _, row in team_df.iterrows():
                bucket = pd.cut([row['draft_ovr']], 
                               bins=[0, 5, 10, 20, 32, 50, 100, 150, 262],
                               labels=['1-5', '6-10', '11-20', '21-32', 
                                      '33-50', '51-100', '101-150', '151-262'])[0]
                expected_av += self.expected_av_by_slot.get(bucket, 0.001)
            
            value_over_expected = actual_av_share - expected_av
            
            results.append(DraftGrade(
                team=team,
                total_picks=n_picks,
                avg_av_share=actual_av_share / n_picks,
                expected_av_share=expected_av / n_picks,
                value_over_expected=value_over_expected,
                draft_capital_spent=team_df['draft_ovr'].mean(),
                percentile_rank=np.nan  # Computed later
            ))
        
        grades_df = pd.DataFrame([r.__dict__ for r in results])
        grades_df['percentile_rank'] = grades_df['value_over_expected'].rank(pct=True)
        
        return grades_df.sort_values('value_over_expected', ascending=False)
    
    def plot_team_rankings(self, grades_df: pd.DataFrame, save_path: str = None):
        """Scatter plot: each team as logo, X = drafting efficiency rank (reversed), Y = value over expected.
        Light theme with horizontal line at 0 and shaded trend band."""
        # Light theme for this chart (reference style)
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        ax.grid(True, color="#cccccc", linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        grades_df = grades_df.copy()
        # Rank 1 = best (highest value_over_expected); grades_df is already sorted best-first from calculate_team_grades
        n = len(grades_df)
        grades_df["rank"] = np.arange(1, n + 1)
        x = grades_df["rank"].values
        y = grades_df["value_over_expected"].values

        # Shaded trend band: linear fit with approximate 95% band
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        resid = y - y_pred
        se = np.std(resid) * np.sqrt(1 + 1 / n + (x - np.mean(x)) ** 2 / (np.var(x) * (n - 1) + 1e-9))
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = np.polyval(coeffs, x_line)
        ax.fill_between(
            x_line,
            y_line - 1.96 * np.std(resid),
            y_line + 1.96 * np.std(resid),
            color="grey",
            alpha=0.2,
            zorder=0,
        )
        ax.plot(x_line, y_line, color="grey", linestyle="-", linewidth=1, alpha=0.7, zorder=0.5)

        # Horizontal reference line at 0
        ax.axhline(0, color="black", linewidth=1, zorder=0.5)

        # Logo size (zoom) so markers don't overlap
        logo_zoom = 0.12 if n <= 32 else 0.08
        for _, row in grades_df.iterrows():
            team, rank, voe = row["team"], row["rank"], row["value_over_expected"]
            logo_file = TEAM_LOGO_FILES.get(team)
            if not logo_file or not (NFL_LOGOS_DIR / logo_file).exists():
                ax.scatter(rank, voe, s=80, color="lightgrey", edgecolor="black", zorder=2)
                continue
            img = mpimg.imread(NFL_LOGOS_DIR / logo_file)
            if img.ndim == 3 and img.shape[2] == 4:
                pass  # RGBA
            elif img.ndim == 2:
                img = np.stack([img] * 4, axis=-1)
                img[:, :, 3] = 255
            im = OffsetImage(img, zoom=logo_zoom)
            ab = AnnotationBbox(im, (rank, voe), frameon=False, pad=0)
            ax.add_artist(ab)

        ax.set_xlabel("Drafting Efficiency Rank (1 = best)")
        ax.set_ylabel("Value Over Expected AV Share")
        ax.set_title("Team Drafting Efficiency")
        ax.invert_xaxis()
        ax.set_xlim(n + 0.5, 0.5)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()
    
    def get_team_report(self, team_name: str, grades_df: pd.DataFrame) -> Dict:
        """Detailed report for specific team"""
        team_row = grades_df[grades_df['team'] == team_name]
        if len(team_row) == 0:
            return {"error": "Team not found"}
        
        team_data = self.df[self.df['draft_team'] == team_name]
        return {
            'team_stats': team_row.iloc[0].to_dict(),
            'top_hits': team_data.nlargest(5, 'av_share')[['player_name', 'draft_ovr', 'av_share']].to_dict('records'),
            'biggest_misses': team_data.nsmallest(5, 'av_share')[['player_name', 'draft_ovr', 'av_share']].to_dict('records')
        }
