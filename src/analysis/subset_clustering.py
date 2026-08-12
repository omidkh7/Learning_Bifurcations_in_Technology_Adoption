#!/usr/bin/env python3
"""
subset_clustering.py
====================
Does concentrating on the THREE strongest-discriminating feature groups change the clustering?
Compares A:CSD + E:SN-fp + H:Crit (the top-3 by class eta^2; 18-D) against the full 46-D space, on
  * REAL data  -> KMeans silhouette vs k + PC scatter (continuum check), and
  * SYNTHETIC benchmark -> ground-truth accuracy (unsupervised GMM vs mechanism-prior nearest-centroid).

Output: figures/si/figS_subset_clustering.png
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from paper_style import set_style, COL2
set_style()
from paper_figures import load_four_group, FAMCOL, FAMLABEL, FAMS, _tag, GROUP, build_features
from benchmark_feature_spaces import load_synth
from unsup_real_world import fit_t_mixture

SI = "figures/si"; os.makedirs(SI, exist_ok=True)
GROUPS = list(GROUP)                                               # canonical 46-D group labels
EHA = [i for i, g in enumerate(GROUPS) if g in ("A:CSD", "E:SN-fp", "H:Crit")]
KS = (2, 3, 4, 5)


def build51(X):
    return np.nan_to_num(build_features(X))                        # canonical 46-D (name kept for compatibility)


def project_sil(F, grp):
    ref = np.isin(grp, ["Historical", "Renewables"])
    sc = StandardScaler().fit(F[ref]); P = PCA().fit(sc.transform(F[ref])).transform(sc.transform(F))
    npc = min(10, P.shape[1])
    Pw = np.clip(P[:, :npc], np.percentile(P[:, :npc], 1, 0), np.percentile(P[:, :npc], 99, 0))  # winsorise outliers
    sil = {k: silhouette_score(Pw, KMeans(k, n_init=10, random_state=0).fit_predict(Pw)) for k in KS}
    return P, sil


def gmm_acc(Z, ys):
    g = fit_t_mixture(Z, 3, seed=0, n_init=3)[0]                  # unsupervised Student-t (GMM dropped)
    Cc = np.array([[((g == k) & (ys == c)).sum() for c in range(3)] for k in range(3)])
    r, cc = linear_sum_assignment(-Cc); mp = {k: cc[i] for i, k in enumerate(r)}
    return (np.array([mp[l] for l in g]) == ys).mean()


def nc_acc(Z, ys):
    cents = np.vstack([Z[ys == c].mean(0) for c in range(3)])
    return (np.argmin(np.linalg.norm(Z[:, None, :] - cents[None], axis=2), axis=1) == ys).mean()


def run():
    # --- real data ---
    grp, _, X = load_four_group(); F = build51(X)
    P_full, s_full = project_sil(F, grp)
    P_eha, s_eha = project_sil(F[:, EHA], grp)

    # --- synthetic ---
    Xn, ys = load_synth(1000, 0); Fs = build51(Xn)
    rows = {"full 46-D": list(range(Fs.shape[1])), "A+E+H 18-D": EHA}
    syn = {}
    for nm, cols in rows.items():
        Z = StandardScaler().fit_transform(Fs[:, cols])
        syn[nm] = (gmm_acc(Z, ys), nc_acc(Z, ys))            # (unsupervised GMM, oracle NC)

    print("real silhouette  full-51:", {k: round(s_full[k], 2) for k in KS})
    print("real silhouette  A+E+H :", {k: round(s_eha[k], 2) for k in KS})
    for nm in rows: print(f"synthetic {nm:11s}: GMM {100*syn[nm][0]:.0f}%   nearest-centroid {100*syn[nm][1]:.0f}%")

    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.9), gridspec_kw=dict(wspace=0.42))

    # (a) real silhouette vs k
    ax = axs[0]
    ax.plot(KS, [s_full[k] for k in KS], "o-", color="#264653", ms=5, lw=1.5, label="full 46-D")
    ax.plot(KS, [s_eha[k] for k in KS], "^-", color="#d62828", ms=5, lw=1.5, label="A+E+H (18-D)")
    ax.axhline(0.5, color="#d62828", lw=0.9, ls="--"); ax.text(5, 0.515, "0.5", color="#d62828", fontsize=6, ha="right")
    ax.set_xticks(KS); ax.set_ylim(0, 0.6); ax.set_xlabel("clusters k"); ax.set_ylabel("KMeans silhouette", fontsize=7)
    ax.legend(fontsize=6, loc="upper right"); _tag(ax, "(a) real: still a continuum")

    # (b) real PC1-PC2 scatter for A+E+H
    ax = axs[1]
    for f in FAMS:
        m = grp == f; ax.scatter(P_eha[m, 0], P_eha[m, 1], s=6, color=FAMCOL[f], alpha=0.55, lw=0, label=FAMLABEL[f])
    ax.set_xlabel("PC1", fontsize=7); ax.set_ylabel("PC2", fontsize=7); ax.tick_params(labelsize=6)
    ax.text(0.03, 0.97, f"k=2 sil {s_eha[2]:.2f}", transform=ax.transAxes, va="top", fontsize=6.2, color="#555")
    ax.legend(fontsize=5.5, loc="lower right", ncol=2); _tag(ax, "(b) A+E+H real projection")

    # (c) synthetic accuracy: unsupervised GMM vs oracle nearest-centroid, both spaces
    ax = axs[2]
    names = list(rows.keys()); xpos = np.arange(len(names)); w = 0.35
    ax.bar(xpos - w/2, [100*syn[n][0] for n in names], w, color="#8ab17d", label="Student-t (unsup.)")
    ax.bar(xpos + w/2, [100*syn[n][1] for n in names], w, color="#1d4e89", label="nearest-cent. (oracle)")
    for i, n in enumerate(names):
        ax.text(i - w/2, 100*syn[n][0] + 1, f"{100*syn[n][0]:.0f}", ha="center", fontsize=6)
        ax.text(i + w/2, 100*syn[n][1] + 1, f"{100*syn[n][1]:.0f}", ha="center", fontsize=6)
    ax.set_xticks(xpos); ax.set_xticklabels(names, fontsize=6.3, rotation=10, ha="right")
    ax.set_ylim(0, 100); ax.set_ylabel("synthetic accuracy (%)", fontsize=7); ax.legend(fontsize=5.8, loc="upper left")
    _tag(ax, "(c) synthetic: ground-truth recovery")

    fig.savefig(f"{SI}/figS_subset_clustering.png", bbox_inches="tight"); plt.close(fig)
    print("saved figS_subset_clustering.png")


if __name__ == "__main__":
    run()
