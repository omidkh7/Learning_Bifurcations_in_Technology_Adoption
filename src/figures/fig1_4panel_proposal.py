#!/usr/bin/env python3
"""
PROPOSAL (Alan's conceptual-figure request, 2026-07-28). Extend Fig 1 to 4 panels:
  (a) same curve, different mechanism        [current hook]
  (b) typing easy, detection is the hard part [current synthetic-validation panel]
  (c) saddle-node: a threshold you can cross  [NEW bifurcation diagram + intervention arrows]
  (d) transcritical: no threshold to cross    [NEW bifurcation diagram + intervention arrows]

Policy point Alan wants emphasized: for a saddle-node a large enough ONE-OFF change in the STATE
(crossing the unstable threshold) produces a LASTING shift; a small one decays back. For a
transcritical a state push has NO lasting effect; only moving the control PARAMETER changes adoption,
proportionally and without a threshold.

Writes a proposal PNG only; does NOT touch Manuscript/figures/fig1_concept.png.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from paper_style import set_style, COL2
from paper_figures import build_features, load_four_group, _tag, _pca_sign_fix
from benchmark_data import load_benchmark, CANONICAL_NULL
set_style()

SNCOL, TCCOL = "#d62828", "#457b9d"
LAST, FADE, PARAM = "#2a9d8f", "#9aa0a6", "#222222"
#OUT = "/private/tmp/claude-503/-Users-Omidkh7-Downloads-Tech-Adoption-DL/76694a1b-ec86-4b9e-aadf-84219eec0c5e/scratchpad"
OUT = "Manuscript/figures"


def _threeclass_recovery():
    """Panel (d) inset: SN/TC/null three-class recovery on the hardest stable-twin null (SI Sec. S3),
    unsupervised vs supervised. Not a magic number: the supervised bar is read from the committed
    benchmark artifact so it tracks the run; the unsupervised Student-t mixture is the label-free
    partition reported in SI Sec. S3 (Fig. S3)."""
    import json, os
    unsup = 70   # SI Sec. S3: unsupervised Student-t mixture on the logistic stable twin
    superv = 94  # documented fallback if the artifact is absent (94.3 +/- 0.4%, 3 draws x 10 seeds)
    bm = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "second_attempt_deep/runs/benchmark/benchmark_dl_proper.json")
    try:
        superv = round(100 * json.load(open(bm))["aggregate_dl"]["logistic"]["acc_mean"])
    except Exception:
        pass
    return unsup, superv


THREECLASS = _threeclass_recovery()

def integ(f, x0, T=700, dt=0.01):
    x = np.empty(T); x[0] = x0
    for i in range(1, T):
        x[i] = x[i - 1] + f(x[i - 1]) * dt
    return x


# --------------------------------------------------------------- (a) hook
def panel_hook(ax, grp, X):
    t100 = np.linspace(0, 1, X.shape[1]); rng = np.random.default_rng(1)
    tc = integ(lambda x: 1.3 * x * (1 - x), 0.02)
    sn = integ(lambda x: 0.010 + 9.0 * x**2 * (1 - x), 0.002)
    tt = np.linspace(0, 1, len(tc))
    hidx = np.where(grp == "Historical")[0]
    # broader, less cherry-picked backdrop: a random subset of ALL Historical series (no smoothness or
    # inflection-window restriction), still subsampled for legibility
    cand = rng.choice(hidx, min(80, len(hidx)), replace=False)
    for j in cand:
        ax.plot(t100, X[j], color="#c4c4c4", lw=0.5, alpha=0.28)
    ax.plot(tt, (tc - tc.min()) / np.ptp(tc), color=TCCOL, lw=2.2, label="transcritical")
    ax.plot(tt, (sn - sn.min()) / np.ptp(sn), color=SNCOL, lw=2.2, label="saddle-node")
    ax.plot([], [], color="#c4c4c4", lw=0.8, label="real adoption")
    ax.set_xlabel("normalized time", fontsize=7.5); ax.set_ylabel("normalized adoption", fontsize=7.5)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.4, loc="upper left", frameon=False)
    _tag(ax, "(c) one S-curve, two mechanisms")


# --------------------------------------------------------------- (b) validation
def panel_validation(ax):
    Xs, ys = load_benchmark(CANONICAL_NULL, 600, 0)
    Zs = StandardScaler().fit_transform(np.nan_to_num(build_features(Xs)))
    pca = PCA(2).fit(Zs); Ps = pca.transform(Zs); _pca_sign_fix(pca.components_, Ps)
    SLAB = {0: "saddle-node", 1: "transcritical", 2: "null"}; SCOL = {0: SNCOL, 1: TCCOL, 2: "#9aa0a6"}
    for cl in (2, 1, 0):
        m = ys == cl
        ax.scatter(Ps[m, 0], Ps[m, 1], s=5, color=SCOL[cl], alpha=0.45, lw=0, label=SLAB[cl])
    ax.set_xlim(np.percentile(Ps[:, 0], 0.001), np.percentile(Ps[:, 0], 99.999))
    ax.set_ylim(np.percentile(Ps[:, 1], 0.001), np.percentile(Ps[:, 1], 99.999))
    ax.set_xlabel("PC1 (synthetic)", fontsize=7.5); ax.set_ylabel("PC2 (synthetic)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.2, loc="upper right", handletextpad=0.4, borderaxespad=0.4, labelspacing=0.3)
    #ib = ax.inset_axes([0.055, 0.645, 0.27, 0.315])
    ib = ax.inset_axes([0.055, 0.070, 0.27, 0.315])
    ib.bar([0, 1], list(THREECLASS), 0.6, color=["#8d99ae", "#1d4e89"])
    ib.axhline(100 / 3, color="#999", lw=0.6, ls=":")
    for i, v in enumerate(THREECLASS):
        ib.text(i, v + 4, f"{v}%", ha="center", fontsize=4.9)
    ib.text(0.5, 100 / 3 + 3, "chance", ha="center", fontsize=4.0, color="#999")
    ib.set_xticks([0, 1]); ib.set_xticklabels(["unsup.", "superv."], fontsize=4.9)
    ib.set_yticks([]); ib.set_ylim(0, 112); ib.tick_params(length=0)
    ib.set_title("3-class recovery", fontsize=4.8, pad=1.5)
    for sp in ib.spines.values(): sp.set(lw=0.5, color="#bbb")
    _tag(ax, "(d) typing is easy, detection is the hard part")


def _arrow(ax, p0, p1, color, rad=0.0, lw=1.6, mut=8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=mut, lw=lw,
                                 color=color, connectionstyle=f"arc3,rad={rad}", zorder=5))


# --------------------------------------------------------------- (c) saddle-node
def panel_saddlenode(ax):
    x = np.linspace(-1.7, 1.7, 600)
    mu = x**3 - x                                   # sideways-S fold; folds at x=+/-1/sqrt3
    ad = 0.5 + 0.285 * x                            # map state -> adoption ~[0.02,0.98]
    thr = 1 / np.sqrt(3)
    lo = x <= -thr; un = np.abs(x) < thr; hi = x >= thr
    ax.plot(mu[lo], ad[lo], "-", color=SNCOL, lw=2.0)
    ax.plot(mu[hi], ad[hi], "-", color=SNCOL, lw=2.0)
    ax.plot(mu[un], ad[un], "--", color=SNCOL, lw=1.4, alpha=0.8)
    # working point at the centre of the bistable band (fixed parameter, vary the state)
    mu0 = 0.0
    a_lo, a_un, a_hi = 0.5 + 0.285 * np.array([-1.0, 0.0, 1.0])
    ax.axvline(mu0, color="#d8d8d8", lw=0.6, ls=":", zorder=0)
    # HERO: large one-off push in state crosses the threshold -> relaxes onto the upper branch
    _arrow(ax, (mu0, a_lo + 0.02), (mu0, a_hi - 0.01), LAST, rad=-0.16, lw=2.1, mut=11)
    ax.plot(mu0, a_un, "o", ms=4.5, color="#333", zorder=6)
    ax.text(mu0 + 0.035, a_un + 0.09, "threshold\n(unstable)", fontsize=5.5, color="#333",
            ha="left", va="top")
    ax.text(mu0 - 0.02, a_hi + 0.10, "large one-off push\nin state: $lasting$", fontsize=6.5,
            color=LAST, ha="center", va="center")
    # FADES: a small push decays back to the lower state
    _arrow(ax, (mu0 - 0.035, a_lo + 0.02), (mu0 - 0.035, (a_lo + a_un) / 2), FADE, lw=1.3, mut=7)
    _arrow(ax, (mu0 - 0.07, (a_lo + a_un) / 2), (mu0 - 0.07, a_lo + 0.02), FADE, lw=1.3, mut=7)
    ax.text(mu0 - 0.19, (a_lo + a_un) / 2 + 0.01, "small push\nfades", fontsize=6.5,
            color="#6b6b6b", ha="center", va="center")
    # PARAMETER: raise mu along the lower branch to the fold -> tips up
    fold_mu = (-thr)**3 - (-thr); fold_ad = 0.5 + 0.285 * (-thr)
    _arrow(ax, (mu0 + 0.05, a_lo - 0.02), (fold_mu - 0.01, fold_ad - 0.02), PARAM, lw=1.2, mut=7)
    _arrow(ax, (fold_mu, fold_ad), (fold_mu + 0.03, fold_ad + 0.13), PARAM, rad=0.35, lw=1.2, mut=7)
    ax.text(fold_mu + 0.02, fold_ad - 0.05, "raise $\\mu$\nto the fold:\ntips up", fontsize=6.0,
            color=PARAM, ha="center", va="top")
    ax.set_xlim(-1.0, 0.74); ax.set_ylim(-0.02, 1.06)
    ax.set_xlabel("control parameter $\\mu$  (cost $\\cdot$ policy $\\cdot$ support)", fontsize=7)
    ax.set_ylabel("adoption state", fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([0, 0.5, 1]); ax.tick_params(labelsize=6.5)
    ax.plot([], [], "-", color="#555", lw=1.8, label="stable")
    ax.plot([], [], "--", color="#555", lw=1.3, label="unstable")
    ax.legend(fontsize=7, loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.3)
    _tag(ax, "(a) saddle-node: a threshold you can cross")


# --------------------------------------------------------------- (b) transcritical
def panel_transcritical(ax):
    mu = np.linspace(-0.85, 1.0, 400); k = 0.72
    # branch x*=0 (adoption 0): stable mu<0, unstable mu>0
    ax.plot(mu[mu <= 0], np.zeros((mu <= 0).sum()), "-", color=TCCOL, lw=2.0)
    ax.plot(mu[mu > 0], np.zeros((mu > 0).sum()), "--", color=TCCOL, lw=1.4, alpha=0.8)
    # branch x*=mu (adoption k*mu): unstable mu<0, stable mu>0
    mneg = mu[mu < 0]; ax.plot(mneg, k * mneg, "--", color=TCCOL, lw=1.4, alpha=0.5)
    mpos = mu[mu >= 0]; ax.plot(mpos, k * mpos, "-", color=TCCOL, lw=2.0)
    mu0 = 0.6; a0 = k * mu0
    ax.axvline(mu0, color="#d8d8d8", lw=0.6, ls=":", zorder=0)
    # STATE push -> decays back to the branch (no lasting effect)
    _arrow(ax, (mu0 - 0.02, a0 + 0.02), (mu0 - 0.02, a0 + 0.26), LAST, rad=0.0, lw=1.8, mut=9)
    _arrow(ax, (mu0 + 0.02, a0 + 0.26), (mu0 + 0.02, a0 + 0.03), FADE, rad=0.0, lw=1.5, mut=8)
    ax.text(mu0 - 0.03, a0 + 0.30, "state push:\n$no$ $lasting$ $effect$", fontsize=6.5, color=LAST,
            ha="center", va="bottom")
    ax.plot(mu0, a0, "o", ms=3.8, color=TCCOL, zorder=6)
    # PARAMETER move along the stable branch -> proportional response
    _arrow(ax, (mu0 + 0.04, k * (mu0 + 0.04)), (0.97, k * 0.97), PARAM, rad=0.0, lw=1.3, mut=8)
    ax.text(0.86, k * 0.86 - 0.10, "raise $\\mu$:\nproportional,\nno threshold", fontsize=6.0,
            color=PARAM, ha="center", va="top")
    ax.text(0.0, 0.2, "exchange of\nstability", fontsize=5.0, color="#888", ha="center", va="top")
    ax.set_xlim(-0.85, 1.0); ax.set_ylim(-0.06, 1.06)
    ax.set_xlabel("latent control parameter $\\mu$  (cost $\\cdot$ policy $\\cdot$ support)", fontsize=7)
    ax.set_ylabel("adoption state", fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([0, 0.5, 1]); ax.tick_params(labelsize=6.5)
    ax.plot([], [], "-", color="#555", lw=1.8, label="stable")
    ax.plot([], [], "--", color="#555", lw=1.3, label="unstable")
    ax.legend(fontsize=7.0, loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.3)
    _tag(ax, "(b) transcritical: no threshold to cross")


def main():
    grp, F, X = load_four_group()
    fig, axs = plt.subplots(2, 2, figsize=(COL2, 5.6), gridspec_kw=dict(hspace=0.42, wspace=0.28))
    panel_hook(axs[1, 0], grp, X)
    panel_validation(axs[1, 1])
    panel_saddlenode(axs[0, 0])
    panel_transcritical(axs[0, 1])
    import os; os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/fig1_4panel_proposal.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print(f"saved {OUT}/fig1_4panel_proposal.png")


if __name__ == "__main__":
    main()
