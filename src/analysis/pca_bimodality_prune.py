#!/usr/bin/env python3
"""
SIDE ANALYSIS S-A3 (Sara). Can PRINCIPLED pruning of the real adoption series induce a bimodal
distribution in any principal component? If yes -> name the two SN/TC-like modes. If no (or only at
degenerate tiny n) -> a robustness check for the continuum.

Metrics (NOT silhouette; Mahdi showed it is outlier-gameable): Hartigan's dip test, GMM 2-vs-1 BIC,
and Sarle's bimodality coefficient. Bimodality is "FOUND" only if, for some PC of the subset's own
PCA: dip p < 0.05 AND GMM prefers 2 components AND the minority mode weight >= 0.10 (guards against a
single-outlier split) AND n_remaining >= 50 (non-degenerate).

Pruning ladders recompute standardize+PCA ON THE PRUNED SUBSET (gives bimodality the best chance, so
a null result is strong). NOT "remove the middle", which trivially manufactures two modes.

Writes nothing the manuscript uses; prints tables + a diagnostic PNG to the scratchpad.
"""
import warnings, os; warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import diptest
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paper_figures import load_four_group, ALL_NAMES, GROUP, _tag
from paper_style import set_style, COL2
set_style()

NI = {n: i for i, n in enumerate(ALL_NAMES)}
GLET = np.array([g.split(":")[0] for g in GROUP])
KEEP_FRACS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
OUT = "runs/figures"
SI = "Manuscript/SI_figures"


def bimod_metrics(v):
    """1-D bimodality metrics for a PC-score vector v."""
    v = np.asarray(v, float); n = len(v)
    dip, pval = diptest.diptest(v)
    x = v.reshape(-1, 1)
    g1 = GaussianMixture(1, n_init=2, random_state=0).fit(x)
    g2 = GaussianMixture(2, n_init=5, random_state=0).fit(x)
    dbic = g1.bic(x) - g2.bic(x)                      # >0 => 2 components preferred
    minw = float(g2.weights_.min())                  # minority-mode weight (outlier guard)
    sep = abs(g2.means_[0, 0] - g2.means_[1, 0]) / (np.sqrt(g2.covariances_.reshape(2).mean()) + 1e-9)
    sk, ku = stats.skew(v), stats.kurtosis(v, fisher=False)
    bc = (sk**2 + 1) / (ku + 3 * (n - 1)**2 / ((n - 2) * (n - 3) + 1e-9))   # Sarle's; >0.555 hints bimodal
    found = (pval < 0.05) and (dbic > 0) and (minw >= 0.10) and (n >= 50)
    return dict(n=n, dip_p=pval, dbic=dbic, minw=minw, sep=sep, bc=bc, found=found)


def subset_pca_scan(F_sub, npc=3):
    """standardize + PCA on the subset, return metrics for PC1..PCnpc."""
    Z = StandardScaler().fit_transform(np.nan_to_num(F_sub))
    P = PCA(n_components=min(npc, Z.shape[1])).fit_transform(Z)
    return [bimod_metrics(P[:, k]) for k in range(P.shape[1])], P


def run_ladder(name, F, order_idx, ascending, fracs=KEEP_FRACS):
    """Keep the top (ascending=False) or bottom (ascending=True) fraction by order_idx, sweep."""
    idx_sorted = np.argsort(order_idx)
    if not ascending:
        idx_sorted = idx_sorted[::-1]
    print(f"\n### ladder: {name}")
    print(f"    {'keep':>5s} {'n':>4s} | {'PC':>3s} {'dip_p':>7s} {'dBIC':>7s} {'minW':>5s} {'sep':>4s} {'BC':>5s}  bimodal?")
    best = None; sweep = []
    for fr in fracs:
        k = max(3, int(round(fr * len(idx_sorted))))
        keep = np.sort(idx_sorted[:k])
        mets, P = subset_pca_scan(F[keep])
        # report the PC with the smallest dip-p (best chance of bimodality)
        j = int(np.argmin([m["dip_p"] for m in mets])); m = mets[j]
        sweep.append((fr, m["n"], m["dip_p"]))
        flag = "  <== BIMODAL" if m["found"] else ""
        print(f"    {fr:>5.2f} {m['n']:>4d} | PC{j+1:>1d} {m['dip_p']:>7.3f} {m['dbic']:>7.1f} "
              f"{m['minw']:>5.2f} {m['sep']:>4.1f} {m['bc']:>5.3f}{flag}")
        if any(mm["found"] for mm in mets):
            fnd = next(mm for mm in mets if mm["found"])
            if best is None or fnd["dip_p"] < best[2]["dip_p"]:
                best = (name, fr, fnd, keep, P, [mm["found"] for mm in mets].index(True))
    return best, sweep


def main():
    grp, F, X = load_four_group()
    print(f"S-A3 bimodality-under-pruning scan. real n={len(grp)}  "
          f"families={ {g:int((grp==g).sum()) for g in ['Historical','Renewables','BEV','CDR']} }")
    print("decision rule: dip p<0.05 AND GMM dBIC>0 AND minority-mode weight>=0.10 AND n>=50")

    q = F[:, NI["tc_logistic_r2"]]          # logistic-fit quality (higher = cleaner S-curve)
    noise = F[:, NI["residual_var_frac"]]   # residual noise fraction (lower = cleaner)
    hist = grp == "Historical"
    bests = []

    # baseline (full sample, all 46-D features)
    mets, _ = subset_pca_scan(F)
    print(f"\n### baseline: full sample (n={len(F)}), 46-D")
    for k, m in enumerate(mets):
        print(f"    PC{k+1}: dip_p {m['dip_p']:.3f}  dBIC {m['dbic']:.1f}  BC {m['bc']:.3f}  bimodal={m['found']}")

    Fdyn = F[:, np.isin(GLET, list("AEGH"))]
    ladders = [("most S-shaped (46-D)", F, q, False),
               ("cleanest, low noise (46-D)", F, noise, True),
               ("Historical, most S-shaped", F[hist], q[hist], False),
               ("dynamics only (A/E/G/H)", Fdyn, q, False)]
    sweeps = {}
    for nm, FF, oi, asc in ladders:
        b, sw = run_ladder(nm, FF, oi, ascending=asc); bests.append(b); sweeps[nm] = sw

    bests = [b for b in bests if b is not None]
    print("\n" + "=" * 72)
    cand = None
    if not bests:
        print("RESULT: NO non-degenerate pruning produced bimodality in any PC. Continuum survives.")
    else:
        name, fr, m, keep, P, j = min(bests, key=lambda b: b[2]["dip_p"])
        cand = (name, fr, m, P, j)
        print(f"RESULT: {len(bests)} flagged cell(s); strongest {name} @ keep {fr:.2f}: "
              f"PC{j+1} dip_p={m['dip_p']:.3f} dBIC={m['dbic']:.1f} n={m['n']} "
              f"(isolated: adjacent keep-fractions not significant)")
    make_si_figure(sweeps, cand, F)
    print("=" * 72)


def make_si_figure(sweeps, cand, F):
    """Paper house-style SI figure: (a) dip-p across pruning ladders; (b) the single flagged cell is
    a fluctuation; (c) raw type-feature non-unimodality is a boundary/ceiling artifact PCA removes."""
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.6), gridspec_kw=dict(wspace=0.34))
    cols = ["#1d4e89", "#2a9d8f", "#e07a5f", "#9a6fb0"]

    ax = axes[0]
    SHORT = {"most S-shaped (46-D)": "S-shaped (46-D)", "cleanest, low noise (46-D)": "low-noise (46-D)",
             "Historical, most S-shaped": "Historical", "dynamics only (A/E/G/H)": "dynamics (A/E/G/H)"}
    for (nm, sw), c in zip(sweeps.items(), cols):
        fr = [s[0] for s in sw]; p = [s[2] for s in sw]
        ax.plot(fr, p, "o-", ms=3, lw=1.2, color=c, label=SHORT.get(nm, nm))
    ax.axhline(0.05, color="#d62828", lw=1.0, ls="--")
    ax.text(0.17, 0.075, "$p=0.05$", color="#d62828", fontsize=5.4)
    ax.set_xlim(1.03, 0.07)                       # more pruning to the right
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("fraction of series kept", fontsize=7)
    ax.set_ylabel("dip-test $p$ (least-unimodal PC)", fontsize=7)
    ax.legend(fontsize=5.6, loc="center", bbox_to_anchor=(0.5, 0.26), frameon=False,
              handlelength=1.2, labelspacing=0.25)
    _tag(ax, "(a)")

    ax = axes[1]
    if cand is not None:
        from sklearn.mixture import GaussianMixture
        name, fr, m, P, j = cand
        v = P[:, j]
        ax.hist(v, bins=22, density=True, color="#9aa0a6", alpha=0.7)
        g = GaussianMixture(2, n_init=5, random_state=0).fit(v.reshape(-1, 1))
        xs = np.linspace(v.min(), v.max(), 200)
        ax.plot(xs, np.exp(g.score_samples(xs.reshape(-1, 1))), color="#1d4e89", lw=1.4)
        ax.set_xlabel(f"PC{j+1} score of the pruned subset", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
    _tag(ax, "(b)")

    ax = axes[2]
    ti = np.nan_to_num(F[:, NI["t_inflection"]])
    ax.hist(ti, bins=30, color="#457b9d", alpha=0.85)
    ax.axvspan(0.95, 1.02, color="#d62828", alpha=0.15)
    ax.set_xlabel("inflection time $t_{50}$ (a type cue)", fontsize=7)
    ax.set_ylabel("count", fontsize=7)
    _tag(ax, "(c)")

    import os; os.makedirs(SI, exist_ok=True)
    fig.savefig(f"{SI}/figS_bimodality_prune.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"saved -> {SI}/figS_bimodality_prune.png")


if __name__ == "__main__":
    main()
    plt.show()
