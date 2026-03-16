import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

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
        """Visualize drafting efficiency of all teams (initials, dotted red = average)."""
        # Map full names to initials; fallback to first two words' initials if unknown
        def to_abbrev(team: str) -> str:
            if team in TEAM_ABBREV:
                return TEAM_ABBREV[team]
            words = team.split()
            return "".join(w[0] for w in words[:2]).upper() if words else team[:3].upper()

        grades_df = grades_df.copy()
        grades_df["abbrev"] = grades_df["team"].map(to_abbrev)
        grades_df = grades_df.sort_values("value_over_expected", ascending=True)  # worst at bottom

        n = len(grades_df)
        fig, ax = plt.subplots(figsize=(10, max(6, n * 0.22)))
        bars = ax.barh(range(n), grades_df["value_over_expected"], color="steelblue", alpha=0.85)
        ax.set_yticks(range(n))
        ax.set_yticklabels(grades_df["abbrev"], fontsize=9)
        ax.set_xlabel("Value Over Expected AV Share")
        ax.set_title("Drafting Efficiency by Team")
        ax.axvline(0, color="gray", linewidth=0.8)
        mean_voe = grades_df["value_over_expected"].mean()
        ax.axvline(mean_voe, color="red", linestyle=":", linewidth=2, label=f"Average ({mean_voe:.2f})")
        ax.legend(loc="lower right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
