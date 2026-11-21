"""Shared chart style for the CineScore notebooks, matching the web apps' "newspaper review" design:
white page, black ink, greys, one forest-green accent (#1f4d3a), serif titles, light gridlines.

Usage at the top of a notebook:
    from cinescore_style import apply_style, title, ACCENT, INK, GREY, LIGHT, MUTED
    apply_style()
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ACCENT = "#1f4d3a"   # the app's green: the main series / "this model"
ACCENT_LIGHT = "#8fb3a0"
INK = "#111111"      # text, reference lines
MUTED = "#4a4a48"    # secondary text
GREY = "#a3a3a0"     # secondary series
LIGHT = "#d6d6d3"    # tertiary series, rules
GRID = "#e8e8e5"

SERIF = ["Source Serif 4", "Georgia", "DejaVu Serif"]
SANS = ["Libre Franklin", "Segoe UI", "Arial", "DejaVu Sans"]

# sequential (white -> green) for hexbins / densities, diverging (grey <- white -> green) for correlations
CMAP = LinearSegmentedColormap.from_list("cinescore", ["#ffffff", "#c9dbd1", "#5f8a74", ACCENT, "#0b2219"])
CMAP_DIV = LinearSegmentedColormap.from_list("cinescore_div", ["#5c5c59", "#bdbdba", "#ffffff", "#8fb3a0", ACCENT])


def apply_style() -> None:
    """Set matplotlib/seaborn defaults once; every chart in the notebook then matches."""
    for cmap in (CMAP, CMAP_DIV):
        if cmap.name not in mpl.colormaps:
            mpl.colormaps.register(cmap)
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": SANS, "font.serif": SERIF, "font.size": 10.5,
        "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 12, "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.edgecolor": "#8d8d8a", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 0, "xtick.major.width": 0.8,
        "axes.prop_cycle": mpl.cycler(color=[ACCENT, INK, GREY, ACCENT_LIGHT, LIGHT]),
        "lines.linewidth": 2, "patch.edgecolor": "white", "patch.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "image.cmap": "cinescore", "hist.bins": 40,
        "boxplot.flierprops.marker": "o", "boxplot.flierprops.markersize": 2.5,
        "boxplot.flierprops.markerfacecolor": GREY, "boxplot.flierprops.markeredgecolor": "none",
        "boxplot.boxprops.color": INK, "boxplot.whiskerprops.color": INK, "boxplot.capprops.color": INK,
        "boxplot.medianprops.color": ACCENT, "boxplot.medianprops.linewidth": 2,
    })


def title(ax, text: str, sub: str | None = None) -> None:
    """Serif, left-aligned chart title (newspaper style), with an optional small grey subtitle."""
    ax.set_title(text, fontfamily="serif", loc="left", pad=22 if sub else 12)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5, color=MUTED, va="bottom")


# seaborn boxplot arguments in the same style (light box, ink lines, green median)
BOX = dict(color="#eeeeec", linecolor=INK, linewidth=0.9,
           medianprops={"color": ACCENT, "linewidth": 2},
           flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": GREY, "markeredgecolor": "none",
                       "alpha": 0.6})

