#!/usr/bin/env python3
"""
paper_style.py
==============
Shared publication style for all manuscript / SI figures (user preferences + PNAS).
  - Times New Roman everywhere (text + math via STIX, a Times-compatible math font)
  - NO grid lines, NO top/right spines
  - NO figure or panel titles (call sites must not set_title / suptitle)
  - readable axis/label font sizes at final print width (PNAS: >= ~6-8 pt)
  - NO em-dashes anywhere in figure text (use commas / 'to' / parentheses)
Figures are drawn at the final PNAS column width so fonts are correct at 1:1 \\includegraphics.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PNAS print widths (inches)
COL1, COL15, COL2 = 3.42, 4.49, 7.0


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # editable/embedded fonts
        "ps.fonttype": 42,
    })


def panel_label(ax, s, x=-0.02, y=1.04, size=10):
    """Bold (a)/(b) panel tag in the figure corner (replaces scrapped titles)."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=size, fontweight="bold",
            va="bottom", ha="right")
