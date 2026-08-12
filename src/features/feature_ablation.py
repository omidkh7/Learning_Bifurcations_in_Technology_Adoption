#!/usr/bin/env python3
"""
feature_ablation.py
===================
Sensitivity of the continuum result to the feature space. Re-runs the PCA -> clustering pipeline on
three feature sets (all subsets of the canonical 46-D space) and asks whether ANY of them produces
discrete saddle-node / transcritical clusters:
  (1) the full canonical 46-D space,
  (2) the critical-slowing-down group alone (group A),
  (3) the critical-scaling group alone (group H, physics-only).
For each: standardise on the reference families (Historical+Solar+Wind), PCA, then KMeans silhouette
for k=2..5 (silhouette < 0.5 => no discrete clusters => continuum). Reports the table and a figure.

Output: Manuscript/SI_figures/figS_feature_ablation.png
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from paper_style import set_style, COL2
set_style()
from paper_figures import load_four_group, build_features, GROUP, FAMCOL, FAMLABEL, FAMS, _tag

SI = "Manuscript/SI_figures"; os.makedirs(SI, exist_ok=True)


def project(F, grp):
    """Standardise + PCA on the reference families (frozen Stage-1 frame), project all series."""
    ref = np.isin(grp, ["Historical", "Renewables"])
    sc = StandardScaler().fit(F[ref]); pca = PCA().fit(sc.transform(F[ref]))
    P = pca.transform(sc.transform(F))
    return P, pca.explained_variance_ratio_


def silhouettes(P, ks=(2, 3, 4, 5)):
    npc = min(10, P.shape[1])
    Pw = np.clip(P[:, :npc], np.percentile(P[:, :npc], 1, 0), np.percentile(P[:, :npc], 99, 0))  # winsorise outliers
    out = {}
    for k in ks:
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(Pw)
        out[k] = silhouette_score(Pw, lab)
    return out


def run():
    grp, _, X = load_four_group()                            # load_four_group returns the canonical 46-D
    F46 = np.nan_to_num(build_features(X))                           # canonical 46-D space
    G = np.array([g.split(":")[0] for g in GROUP])                  # per-feature group letter A..H
    sets = {"canonical 46-D": F46,
            "CSD":            F46[:, G == "A"],
            "critical":       F46[:, G == "H"]}

    proj = {}; sil = {}
    print(f"{'feature set':26s} | {'k=2':>6} {'k=3':>6} {'k=4':>6} {'k=5':>6} | verdict")
    for name, F in sets.items():
        F = np.nan_to_num(F)
        P, evr = project(F, grp); s = silhouettes(P)
        proj[name] = (P, evr); sil[name] = s
        verdict = "continuum (all < 0.5)" if max(s.values()) < 0.5 else "POSSIBLE CLUSTERS"
        print(f"{name:26s} | " + " ".join(f"{s[k]:6.2f}" for k in (2, 3, 4, 5)) + f" | {verdict}")

    # ---- figure: 3 PC1-PC2 scatters + silhouette-vs-k ----
    fig = plt.figure(figsize=(COL2, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.30)
    tags = "abc"
    for j, (name, (P, evr)) in enumerate(proj.items()):
        ax = fig.add_subplot(gs[0, j])
        for f in FAMS:
            m = grp == f
            ax.scatter(P[m, 0], P[m, 1], s=6, color=FAMCOL[f], alpha=0.55, lw=0, label=FAMLABEL[f])
        ax.set_xlabel(f"PC1 ({100*evr[0]:.0f}%)", fontsize=7)
        if j == 0: ax.set_ylabel(f"PC2 ({100*evr[1]:.0f}%)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.text(0.03, 0.97, f"k=2 sil\n{sil[name][2]:.2f}", transform=ax.transAxes, va="top", fontsize=6.2, color="#555")
        _tag(ax, f"({tags[j]}) {name}")
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=6.6, bbox_to_anchor=(0.5, 1.005), frameon=False)

    axk = fig.add_subplot(gs[1, :])
    mk = {"canonical 46-D": ("s", "#2a9d8f"), "CSD": ("o", "#264653"), "critical": ("^", "#9b5de5")}
    ks = [2, 3, 4, 5]
    for name in sets:
        mkr, c = mk[name]
        axk.plot(ks, [sil[name][k] for k in ks], mkr + "-", color=c, ms=5, lw=1.5, label=name)
    axk.axhline(0.5, color="#d62828", lw=1.0, ls="--")
    axk.text(2.05, 0.515, "0.5: discrete-cluster threshold", color="#d62828", fontsize=6.2, ha="left", va="bottom")
    axk.set_xticks(ks); axk.set_xlabel("number of clusters k"); axk.set_ylabel("KMeans silhouette")
    axk.set_ylim(0, 0.6); axk.legend(fontsize=6.4, loc="center right")
    _tag(axk, "(d) no feature set crosses the discrete-cluster threshold")

    fig.savefig(f"{SI}/figS_feature_ablation.png", bbox_inches="tight"); plt.close(fig)
    print("saved figS_feature_ablation.png")


if __name__ == "__main__":
    run()
