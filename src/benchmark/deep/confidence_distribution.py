#!/usr/bin/env python3
"""
confidence_distribution.py — the distribution of the argmax (max-class) probability on real
adoption curves, as a CONTINUUM diagnostic.

Logic: if real curves were genuine discrete types the classifier would saturate near
max-prob ~ 1 (mass at the simplex vertices), exactly as it does on synthetic curves whose
types are real. If adoption is a continuum, the classifier stays hesitant: max-prob piles
at intermediate values and the curves sit along the SN--TC edge / interior of the simplex,
not at the corners.

Trains FeatMLP (5 seeds) on the paper's canonical stable-twin logistic benchmark, seed-
averages the softmax on the 478 real four-family curves and on the synthetic test split,
and reports:
  * max-prob distribution, real vs synthetic (histogram + summary stats)
  * position on the SN/TC/Null probability simplex (real, by family)
  * a continuum coordinate u = P(SN)/(P(SN)+P(TC)) for transition-dominated curves
    (P(SN)+P(TC) > 0.5), and its Spearman correlation with PC1 of the real feature space

Writes runs/checks/confidence_distribution.json + fig_confidence.png
"""
import os, sys, json, warnings
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from benchmark_data import load_benchmark
from paper_figures import build_features, load_four_group, _tag, FAMCOL, FAMLABEL, FAMS
from paper_style import set_style, COL2
from models import FeatMLP
import train as T

set_style()                                                   # paper house style (Times, etc.)
CRUNS = os.path.join(HERE, "runs", "checks")
os.makedirs(CRUNS, exist_ok=True)
os.makedirs(os.path.join(HERE, "runs", "conf"), exist_ok=True)
N_PER, N_SEEDS = 3000, 5
PROPER_N_PER, PROPER_N_SEEDS = 10000, 10     # "proper" mode: python confidence_distribution.py proper
CLASSES = ["SN", "TC", "Null"]


def simplex_xy(P):
    """3-simplex -> 2D. vertices SN=(0,0), TC=(1,0), Null=(0.5, sqrt3/2)."""
    a, b, c = P[:, 0], P[:, 1], P[:, 2]
    return b + 0.5 * c, c * (np.sqrt(3) / 2)


def main(proper=False):
    global N_PER, N_SEEDS
    if proper:
        N_PER, N_SEEDS = PROPER_N_PER, PROPER_N_SEEDS
    tag = "_proper" if proper else ""
    print(f"confidence run: n_per {N_PER}, {N_SEEDS} seeds{' (proper)' if proper else ''}", flush=True)
    Xb, y = load_benchmark("logistic", N_PER, seed=0)
    Fb = np.nan_to_num(build_features(Xb)).astype(np.float32); y = y.astype(np.int64)
    grp, Fr, _ = load_four_group(); Fr = np.nan_to_num(Fr).astype(np.float32)

    Pr_all, Ps_all, yte = [], [], None
    for s in range(N_SEEDS):
        itr, iva, ite = T.make_splits(y, seed=1400 + s); yte = y[ite]
        mu, sd = Fb[itr].mean(0), Fb[itr].std(0) + 1e-9
        Fz = np.clip((Fb - mu) / sd, -8, 8).astype(np.float32)
        Frz = np.clip((Fr - mu) / sd, -8, 8).astype(np.float32)
        torch.manual_seed(s); np.random.seed(s)
        m, _, _ = T.train_model(FeatMLP(), f"conf/{tag or 'c'}_s{s}", "mlp", 0.0,
                                (torch.from_numpy(Fz), torch.from_numpy(y)), itr, iva)
        m.eval()
        with torch.no_grad():
            Ps_all.append(torch.softmax(m(torch.from_numpy(Fz[ite]).to(T.DEVICE)), 1).cpu().numpy())
            Pr_all.append(torch.softmax(m(torch.from_numpy(Frz).to(T.DEVICE)), 1).cpu().numpy())
    # SYMMETRIC confidence protocol (fixes the earlier asymmetry, where real curves were scored by
    # the seed ENSEMBLE mean but synthetic by a single seed; ensembling only the disagreeing side
    # manufactured apparent hesitancy). Here every observation, real or synthetic, is scored by a
    # SINGLE model on its own held-out data, pooled over seeds: real curves (scored by each seed's
    # model) and each seed's synthetic test split. This isolates genuine per-model hesitancy.
    mxr = np.concatenate([P.max(1) for P in Pr_all])            # pooled per-seed real max-prob
    mxs = np.concatenate([P.max(1) for P in Ps_all])            # pooled per-seed synthetic-test max-prob

    def stats(mx):
        return dict(median=float(np.median(mx)), mean=float(mx.mean()),
                    frac_gt_0_9=float((mx > 0.9).mean()), frac_gt_0_8=float((mx > 0.8).mean()),
                    frac_lt_0_6=float((mx < 0.6).mean()), min=float(mx.min()))
    res = {"maxprob_real": stats(mxr), "maxprob_synth": stats(mxs)}

    # continuum coordinate among transition-dominated real curves: the seed-ensemble mean gives the
    # most stable per-curve position estimate for the simplex/PC geometry (panels b, c only).
    Pr = np.mean(Pr_all, axis=0)
    trans = (Pr[:, 0] + Pr[:, 1]) > 0.5
    u = Pr[trans, 0] / (Pr[trans, 0] + Pr[trans, 1] + 1e-9)      # SN share of the SN/TC mass
    Zr = StandardScaler().fit_transform(Fr)
    PC1 = PCA(n_components=2).fit_transform(Zr)[:, 0]
    rho = spearmanr(PC1[trans], u).statistic
    res["n_transition_dominated"] = int(trans.sum())
    res["u_spearman_pc1"] = float(rho)
    res["u_median"] = float(np.median(u))

    print("=" * 64)
    print("Argmax-probability (confidence) distribution on real curves")
    print("=" * 64)
    print(f"  REAL : median {res['maxprob_real']['median']:.2f}  mean {res['maxprob_real']['mean']:.2f}  "
          f">0.9 {res['maxprob_real']['frac_gt_0_9']*100:.0f}%  >0.8 {res['maxprob_real']['frac_gt_0_8']*100:.0f}%  "
          f"<0.6 {res['maxprob_real']['frac_lt_0_6']*100:.0f}%")
    print(f"  SYNTH: median {res['maxprob_synth']['median']:.2f}  mean {res['maxprob_synth']['mean']:.2f}  "
          f">0.9 {res['maxprob_synth']['frac_gt_0_9']*100:.0f}%  >0.8 {res['maxprob_synth']['frac_gt_0_8']*100:.0f}%  "
          f"<0.6 {res['maxprob_synth']['frac_lt_0_6']*100:.0f}%")
    print(f"  transition-dominated real curves: {int(trans.sum())}/{len(grp)};  "
          f"u=P(SN)/(P(SN)+P(TC)) median {res['u_median']:.2f};  Spearman(u, PC1) = {rho:+.2f}")
    json.dump(res, open(f"{CRUNS}/confidence_distribution{tag}.json", "w"), indent=2)

    # ---- figure (paper house style) ----
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5), gridspec_kw=dict(wspace=0.34))
    ax = axes[0]
    bins = np.linspace(0.33, 1.0, 34)
    ax.hist(mxs, bins=bins, density=True, alpha=0.65, color="#457b9d", label="synthetic (true types)")
    ax.hist(mxr, bins=bins, density=True, alpha=0.6, color="#d62828", label="real curves")
    ax.set_xlabel("maximal softmax probability", fontsize=7.5); ax.set_ylabel("density", fontsize=7.5)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    _tag(ax, "(a) real hesitant, synthetic saturates")

    ax = axes[1]
    for vx, vy, nm in [(0, 0, "SN"), (1, 0, "TC"), (0.5, np.sqrt(3) / 2, "Null")]:
        ax.text(vx, vy + (0.04 if nm == "Null" else -0.09), nm, ha="center", fontsize=7, weight="bold")
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color="#bbb", lw=0.9)
    rx, ry = simplex_xy(Pr)
    for g in FAMS:
        mg = grp == g
        if mg.sum() == 0:
            continue
        ax.scatter(rx[mg], ry[mg], s=9, alpha=0.7, lw=0, color=FAMCOL.get(g, "#888"),
                   label=f"{FAMLABEL.get(g, g)} ({mg.sum()})")
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(fontsize=5.6, loc="upper left", frameon=False, bbox_to_anchor=(-0.02, 1.0),
              handletextpad=0.3, labelspacing=0.3)
    _tag(ax, "(b) real curves on the SN / TC / null simplex")

    ax = axes[2]
    ax.scatter(PC1[trans], u, s=9, alpha=0.7, lw=0,
               c=[FAMCOL.get(g, "#888") for g in grp[trans]])
    ax.set_xlabel("PC1 of the real 46-D feature space", fontsize=7.5)
    ax.set_ylabel(r"$u=P(\mathrm{SN})/(P(\mathrm{SN})+P(\mathrm{TC}))$", fontsize=7)
    ax.set_ylim(-0.02, 1.02)
    _tag(ax, f"(c) saddle-node share vs PC1 ($\\rho={rho:+.2f}$)")
    fig.savefig(f"{CRUNS}/fig_confidence{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {CRUNS}/confidence_distribution{tag}.json + fig_confidence{tag}.png", flush=True)


if __name__ == "__main__":
    main(proper="proper" in sys.argv)
