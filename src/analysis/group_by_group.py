#!/usr/bin/env python3
"""
group_by_group.py
=================
Standalone performance of each of the eight feature groups, reported plainly:
  * SYNTHETIC benchmark (clean logistic null): 3-class accuracy under Student-t (unsupervised) and
    nearest-centroid (oracle), using that group ALONE.
  * REAL data: winsorised k=2 KMeans silhouette of the four families using that group ALONE.
Up front: the CSD group is the single strongest discriminator on the benchmark; on real data no group
crosses the discrete-cluster threshold.

Output: Manuscript/SI_figures/figS_group_by_group.png + printed table.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from paper_style import set_style, COL2
set_style()
from paper_figures import build_features, GROUP, load_four_group, GCOLOR, _tag
from benchmark_data import load_benchmark, CANONICAL_NULL
from unsup_real_world import fit_t_mixture

SI = "Manuscript/SI_figures"; os.makedirs(SI, exist_ok=True)
GORDER = ["A:CSD", "B:Inflect", "C:Phase", "D:Catch22", "E:SN-fp", "F:Transit", "G:TC-fp", "H:Crit"]


def hung(lab, ys):
    Cc = np.array([[((lab == k) & (ys == c)).sum() for c in range(3)] for k in range(3)])
    r, cc = linear_sum_assignment(-Cc); mp = {k: cc[i] for i, k in enumerate(r)}
    return (np.array([mp[l] for l in lab]) == ys).mean()


def nc(Z, ys):
    ce = np.vstack([Z[ys == c].mean(0) for c in range(3)])
    return (np.argmin(np.linalg.norm(Z[:, None, :] - ce[None], axis=2), axis=1) == ys).mean()


def win_sil(Z, k=2):
    if Z.shape[1] < 2: Z = np.hstack([Z, np.zeros((len(Z), 1))])
    Pw = np.clip(Z, np.percentile(Z, 1, 0), np.percentile(Z, 99, 0))
    return silhouette_score(Pw, KMeans(k, n_init=10, random_state=0).fit_predict(Pw))


def per_group(null_kind, n_per, seed):
    Xn, ys = load_benchmark(null_kind, n_per, seed); Fb = np.nan_to_num(build_features(Xn))
    u, o = [], []
    for g in GORDER:
        cols = [i for i, gg in enumerate(GROUP) if gg == g]
        Zb = StandardScaler().fit_transform(Fb[:, cols])
        u.append(hung(fit_t_mixture(Zb, 3, seed=seed, n_init=3)[0], ys)); o.append(nc(Zb, ys))
    return u, o


def run(n_per=800, seed=0):
    uc, oc = per_group(CANONICAL_NULL, n_per, seed)      # clean (logistic) null
    um, om = per_group("mixed", n_per, seed)             # open-set (heterogeneous) null
    print(f"{'group':10s} | clean unsup | clean oracle | mixed unsup | mixed oracle")
    for i, g in enumerate(GORDER):
        print(f"{g:10s} |   {100*uc[i]:4.0f}%    |    {100*oc[i]:4.0f}%    |   {100*um[i]:4.0f}%    |   {100*om[i]:4.0f}%")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(COL2, 3.0), sharey=True, gridspec_kw=dict(wspace=0.12))
    x = np.arange(len(GORDER)); w = 0.38
    for ax, (uu, oo), tag in [(axA, (uc, oc), "(a) each group alone: clean null"),
                              (axB, (um, om), "(b) each group alone: open-set null")]:
        ax.bar(x - w / 2, [100 * v for v in uu], w, color="#8ab17d", label="Student-t (unsup.)")
        ax.bar(x + w / 2, [100 * v for v in oo], w, color="#1d4e89", label="nearest-cent. (oracle)")
        ax.axhline(100 / 3, color="#aaa", lw=0.7, ls=":")
        ax.set_xticks(x); ax.set_xticklabels([g.split(":")[1] for g in GORDER], fontsize=5.8, rotation=25, ha="right")
        ax.set_ylim(0, 100); _tag(ax, tag)
    axA.set_ylabel("benchmark accuracy (%)", fontsize=7.5); axA.legend(fontsize=5.8, loc="upper left")
    fig.savefig(f"{SI}/figS_group_by_group.png", bbox_inches="tight"); plt.close(fig)
    print("saved figS_group_by_group.png")


if __name__ == "__main__":
    run()
