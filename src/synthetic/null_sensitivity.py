#!/usr/bin/env python3
"""
null_sensitivity.py
===================
How hard the benchmark is depends on what the NULL looks like. We benchmark class recovery against
FIVE clean per-shape null classes (each a well-defined process), then -- DEMOTED to an open-set
sensitivity check (Bury et al. 2021 / open-set-recognition style) -- a heterogeneous "mixed" null that
blends shapes. SN/TC are held fixed (canonical normal forms). Methods: Student-t, skew-t (unsupervised)
+ mechanism prior + nearest-centroid (oracle).

Output: Manuscript/SI_figures/figS_null_sensitivity.png + printed accuracy/confusion table.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from paper_style import set_style, COL2
set_style()
from paper_figures import build_features, _tag
from benchmark_data import load_benchmark, CLEAN_NULLS
from unsup_real_world import fit_t_mixture, fit_skew_t_mixture, fit_theory_bayes_gmm

SI = "Manuscript/SI_figures"; os.makedirs(SI, exist_ok=True)
NULLS = CLEAN_NULLS + ["mixed"]                                    # 5 clean + 1 demoted open-set
METHODS = ["Student-t", "skew-t", "mechanism prior", "nearest-centroid"]
MCOL = {"Student-t": "#8ab17d", "skew-t": "#2a9d8f", "mechanism prior": "#1d4e89", "nearest-centroid": "#457b9d"}


def hung(lab, ys):
    Cc = np.array([[((lab == k) & (ys == c)).sum() for c in range(3)] for k in range(3)])
    r, cc = linear_sum_assignment(-Cc); mp = {k: cc[i] for i, k in enumerate(r)}
    pm = np.array([mp[l] for l in lab])
    M = np.array([[((ys == tc) & (pm == pc)).sum() / max((ys == tc).sum(), 1) for pc in range(3)] for tc in range(3)])
    return (pm == ys).mean(), adjusted_rand_score(ys, lab), M


def run(n_per=800, seed=0, n_init=3):
    acc = {m: [] for m in METHODS}; tcnull = {m: [] for m in METHODS}
    print(f"{'null':13s} | " + " ".join(f"{m:>9}" for m in METHODS))
    for kind in NULLS:
        Xn, ys = load_benchmark(kind, n_per, seed)
        Z = StandardScaler().fit_transform(np.nan_to_num(build_features(Xn)))
        cents = np.vstack([Z[ys == c].mean(0) for c in range(3)])
        labs = {
            "Student-t": fit_t_mixture(Z, 3, seed=seed, n_init=n_init)[0],
            "skew-t": fit_skew_t_mixture(Z, 3, seed=seed, n_init=n_init)[0],
            "mechanism prior": fit_theory_bayes_gmm(Z, 3, centroids_prior=cents, seed=seed, n_init=n_init)[0],
            "nearest-centroid": np.argmin(np.linalg.norm(Z[:, None, :] - cents[None], axis=2), axis=1),
        }
        row = []
        for m in METHODS:
            a, ari, M = hung(labs[m], ys)
            acc[m].append(a); tcnull[m].append(M[1, 2] + M[2, 1])
            row.append(f"{100*a:4.0f}%")
        print(f"{kind:13s} | " + " ".join(f"{r:>9}" for r in row))

    nclean = len(CLEAN_NULLS)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(COL2, 3.1), gridspec_kw=dict(wspace=0.32))
    x = np.arange(len(NULLS)); w = 0.2
    for i, m in enumerate(METHODS):
        axA.bar(x + (i - 1.5) * w, [100 * a for a in acc[m]], w, color=MCOL[m], label=m)
    axA.axvline(nclean - 0.5, color="#d62828", lw=1.0, ls="--")
    axA.text(nclean, 100, "open-set", color="#d62828", fontsize=5.8, ha="center", va="bottom")
    axA.set_xticks(x); axA.set_xticklabels(NULLS, fontsize=5.8, rotation=18, ha="right")
    axA.set_ylabel("3-class accuracy (%)", fontsize=7.5); axA.set_ylim(50, 114)
    axA.legend(fontsize=5.4, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.2, handletextpad=0.4); _tag(axA, "(a) recovery vs null class")
    for m in ("Student-t", "skew-t"):
        axB.plot(x, [100 * v for v in tcnull[m]], "o-", color=MCOL[m], ms=4, lw=1.4, label=m)
    axB.axvline(nclean - 0.5, color="#d62828", lw=1.0, ls="--")
    axB.set_xticks(x); axB.set_xticklabels(NULLS, fontsize=5.8, rotation=18, ha="right")
    axB.set_ylabel("TC$\\leftrightarrow$Null confusion (%)", fontsize=7.5)
    axB.legend(fontsize=6, loc="upper left"); _tag(axB, "(b) where the difficulty is")
    fig.savefig(f"{SI}/figS_null_sensitivity.png", bbox_inches="tight"); plt.close(fig)
    print("saved figS_null_sensitivity.png")


if __name__ == "__main__":
    run()
