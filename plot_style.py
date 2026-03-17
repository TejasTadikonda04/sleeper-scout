"""
Shared dark-theme plot style: purple-to-yellow gradient, light yellow-white text.

Use: call apply_dark_style() at the start of your script or notebook to make
all subsequent matplotlib/seaborn plots use this theme.
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Light yellow-white for text, labels, ticks (matches reference)
TEXT_COLOR = "#f0e6c8"
AXES_EDGE = "#b0a080"
GRID_COLOR = "#3a3640"
# Dashed trend/regression line
TREND_LINE_COLOR = "#e8a0a0"
TREND_LINE_STYLE = "--"

# Purple -> magenta -> orange -> yellow (continuous gradient for data points)
PURPLE_YELLOW_COLORS = [
    "#1a0a2e",  # deep indigo
    "#4a148c",  # purple
    "#8e24aa",  # magenta
    "#d84315",  # orange
    "#ffc107",  # bright yellow
]


def get_purple_yellow_cmap():
    """Colormap from deep purple to bright yellow (for scatter magnitude)."""
    return mcolors.LinearSegmentedColormap.from_list(
        "purple_yellow", PURPLE_YELLOW_COLORS, N=256
    )


def apply_dark_style():
    """Set matplotlib and (if available) seaborn to the dark theme."""
    plt.rcParams.update({
        "figure.facecolor": "#1e1e1e",
        "axes.facecolor": "#1e1e1e",
        "axes.edgecolor": AXES_EDGE,
        "axes.labelcolor": TEXT_COLOR,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.4,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "legend.facecolor": "#2a2a2a",
        "legend.edgecolor": AXES_EDGE,
        "legend.labelcolor": TEXT_COLOR,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#1e1e1e",
        "savefig.edgecolor": "none",
    })
    try:
        import seaborn as sns
        sns.set_style("darkgrid", {"axes.facecolor": "#1e1e1e", "figure.facecolor": "#1e1e1e"})
        sns.set_context("paper", font_scale=1.2)
    except ImportError:
        pass


def trend_line_color():
    """Color for regression/trend lines (dashed light pink/red)."""
    return TREND_LINE_COLOR


def trend_line_style():
    """Linestyle for regression/trend lines."""
    return TREND_LINE_STYLE
