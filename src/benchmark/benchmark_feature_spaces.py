#!/usr/bin/env python3
"""
benchmark_feature_spaces.py
===========================
Point 2: test the candidate FEATURE SPACES on the SYNTHETIC BENCHMARK (ground-truth SN/TC/Null) with
multiple unsupervised algorithms. Unlike the real-data ablation (silhouette only), here the labels are
known, so we report Hungarian-matched accuracy and ARI -- i.e. whether each feature space actually lets
an algorithm RECOVER the three generative classes.

Feature spaces (all subsets of the canonical 46-D space, sliced by feature group):
  46-D (all)     : the full canonical space used throughout the paper
  CSD            : critical-slowing-down group alone (group A)
  critical       : critical-scaling group alone (group H)
  A+E+H          : the three strongest groups (CSD + saddle-node fingerprint + critical)
Algorithms: Student-t, skew-t (unsupervised) + mechanism prior, nearest-centroid (oracle).

Output: figures/si/figS_benchmark_spaces.png  + printed accuracy/ARI grids.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import PchipInterpolator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from paper_style import set_style, COL2
set_style()
from paper_figures import _tag, build_features, GROUP
from unsup_real_world import fit_t_mixture, fit_skew_t_mixture, fit_theory_bayes_gmm
from benchmark_data import load_benchmark, CANONICAL_NULL

SI = "figures/si"; os.makedirs(SI, exist_ok=True)
# two UNSUPERVISED mixtures (GMM dropped) + two MECHANISM-PRIOR (oracle) references given true centroids
ALGOS = ["Student-t", "skew-t", "mechanism prior", "nearest-centroid"]
N_UNSUP = 2


def load_synth(n_per=1000, seed=0):
    return load_benchmark(CANONICAL_NULL, n_per, seed)              # canonical clean (logistic) null


def acc_ari(labels, ys):
    Cc = np.array([[((labels == k) & (ys == c)).sum() for c in range(3)] for k in range(3)])
    r, cc = linear_sum_assignment(-Cc); mp = {k: cc[i] for i, k in enumerate(r)}
    pm = np.array([mp[l] for l in labels])
    return (pm == ys).mean(), adjusted_rand_score(ys, labels)


def run(n_per=1000, seed=0, n_init=3):
    Xn, ys = load_synth(n_per, seed)
    F46 = np.nan_to_num(build_features(Xn))                         # canonical 46-D space
    G = np.array([g.split(":")[0] for g in GROUP])                  # per-feature group letter A..H
    spaces = {
        "46-D (all)": F46,
        "CSD":        F46[:, G == "A"],
        "critical":   F46[:, G == "H"],
        "A+E+H":      F46[:, np.isin(G, ["A", "E", "H"])],
    }

    accs = {}; aris = {}
    print(f"{'feature space':14s} | " + " ".join(f"{a:>9}" for a in ALGOS) + "   (accuracy / ARI)")
    for sname, F in spaces.items():
        Z = StandardScaler().fit_transform(F)
        cents = np.vstack([Z[ys == c].mean(0) for c in range(3)])
        fits = {
            "Student-t":        fit_t_mixture(Z, 3, seed=seed, n_init=n_init)[0],
            "skew-t":           fit_skew_t_mixture(Z, 3, seed=seed, n_init=n_init)[0],
            "mechanism prior":  fit_theory_bayes_gmm(Z, 3, centroids_prior=cents, seed=seed, n_init=n_init)[0],
            "nearest-centroid": np.argmin(np.linalg.norm(Z[:, None, :] - cents[None], axis=2), axis=1),
        }
        a = {}; r = {}
        for alg in ALGOS:
            a[alg], r[alg] = acc_ari(fits[alg], ys)
        accs[sname] = a; aris[sname] = r
        print(f"{sname:14s} | " + " ".join(f"{100*a[al]:4.0f}/{r[al]:.2f}" for al in ALGOS))

    # ---- figure: accuracy heatmap (spaces x algorithms) + per-space best-accuracy bar ----
    snames = list(spaces.keys())
    A = np.array([[accs[s][al] for al in ALGOS] for s in snames])
    fig, (axH, axB) = plt.subplots(1, 2, figsize=(COL2, 3.3), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.42))
    im = axH.imshow(A * 100, cmap="YlGnBu", vmin=55, vmax=95, aspect="auto")
    axH.set_xticks(range(len(ALGOS))); axH.set_xticklabels(ALGOS, fontsize=6.5, rotation=18, ha="right")
    axH.set_yticks(range(len(snames))); axH.set_yticklabels(snames, fontsize=6.5)
    for i in range(len(snames)):
        for j in range(len(ALGOS)):
            axH.text(j, i, f"{100*A[i, j]:.0f}", ha="center", va="center", fontsize=6.3,
                     color="white" if A[i, j] * 100 > 80 else "#222")
    axH.axvline(N_UNSUP - 0.5, color="#d62828", lw=1.2)          # unsupervised | mechanism-prior (oracle)
    axH.text(N_UNSUP / 2 - 0.5, -0.62, "unsupervised", fontsize=5.6, color="#444", ha="center")
    axH.text((N_UNSUP + len(ALGOS)) / 2 - 0.5, -0.62, "mechanism prior (oracle)", fontsize=5.6, color="#d62828", ha="center")
    cb = fig.colorbar(im, ax=axH, fraction=0.045, pad=0.03); cb.ax.tick_params(labelsize=5.5)
    cb.set_label("accuracy (%)", fontsize=6.2)
    axH.set_title("(a) class recovery on ground-truth synthetic data", fontsize=7.5, loc="left", pad=14)

    best = A.max(1); bestalg = [ALGOS[j] for j in A.argmax(1)]
    axB.barh(range(len(snames)), best * 100, color="#457b9d", alpha=0.85)
    axB.set_yticks(range(len(snames))); axB.set_yticklabels(snames, fontsize=6.5); axB.invert_yaxis()
    axB.set_xlim(50, 100); axB.set_xlabel("best accuracy (%)", fontsize=7)
    for i, (b, al) in enumerate(zip(best, bestalg)):
        axB.text(b * 100 - 1, i, f"{100*b:.0f} ({al})", ha="right", va="center", fontsize=5.6, color="white")
    axB.axvline(100/3, color="#aaa", lw=0.7, ls=":")
    _tag(axB, "(b) best per space")
    fig.savefig(f"{SI}/figS_benchmark_spaces.png", bbox_inches="tight"); plt.close(fig)
    print("saved figS_benchmark_spaces.png")
    return accs, aris


if __name__ == "__main__":
    run()
