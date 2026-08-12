#!/usr/bin/env python3
"""
feature_redundancy.py
=====================
Pairwise redundancy audit of the 46 canonical features: signed correlation heatmap (ordered by group)
and the list of near-duplicate pairs (|r| > 0.85) on the synthetic benchmark. Flags features that are
genuinely too similar in definition.

Output: figures/si/figS_feature_redundancy.png + printed pair list.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from paper_style import set_style, COL2
set_style()
from paper_figures import build_features, GROUP, ALL_NAMES, GCOLOR, _tag
from benchmark_feature_spaces import load_synth

SI = "figures/si"; os.makedirs(SI, exist_ok=True)
GORDER = ["A:CSD", "B:Inflect", "C:Phase", "D:Catch22", "E:SN-fp", "F:Transit", "G:TC-fp", "H:Crit"]


def run(thr=0.85):
    Xn, ys = load_synth(1000, 0)
    Z = StandardScaler().fit_transform(np.nan_to_num(build_features(Xn)))
    C = np.corrcoef(Z.T)
    n = C.shape[0]
    pairs = sorted(((abs(C[i, j]), C[i, j], i, j) for i in range(n) for j in range(i + 1, n)), reverse=True)
    dup = [(r, i, j) for a, r, i, j in pairs if a > thr]
    print(f"=== {len(dup)} feature pairs with |r| > {thr} ===")
    for r, i, j in dup:
        print(f"{r:+.2f}  {ALL_NAMES[i]:24s}[{GROUP[i][0]}] ~ {ALL_NAMES[j]:24s}[{GROUP[j][0]}]")

    fig, ax = plt.subplots(figsize=(COL2 * 0.82, COL2 * 0.78))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    bounds = []
    pos = 0
    for g in GORDER:
        c = GROUP.count(g); bounds.append((pos, pos + c - 1, g)); pos += c
    for lo, hi, g in bounds:
        if lo > 0:
            ax.axhline(lo - 0.5, color="k", lw=0.6); ax.axvline(lo - 0.5, color="k", lw=0.6)
        ax.text(-1.6, (lo + hi) / 2, g.split(":")[1], ha="right", va="center", fontsize=6, color=GCOLOR[g], rotation=0)
        ax.text((lo + hi) / 2, -1.6, g.split(":")[1], ha="center", va="bottom", fontsize=6, color=GCOLOR[g], rotation=90)
    # mark the strongest duplicate pairs
    for r, i, j in dup[:8]:
        ax.scatter([j, i], [i, j], s=10, facecolor="none", edgecolor="k", lw=0.6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.ax.tick_params(labelsize=6)
    cb.set_label("feature correlation", fontsize=7)
    _tag(ax, "pairwise feature correlation (46 features, by group)")
    fig.tight_layout(); fig.savefig(f"{SI}/figS_feature_redundancy.png", bbox_inches="tight"); plt.close(fig)
    print("saved figS_feature_redundancy.png")


if __name__ == "__main__":
    run()
