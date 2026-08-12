#!/usr/bin/env python3
"""
mc_mu_observed.py
=================
Monte-Carlo question (single, sharp): IF THE CONTROL PARAMETER mu(t) IS OBSERVED ALONGSIDE THE
ADOPTION CURVE x(t), DO THE PAPER'S OWN UNSUPERVISED METHODS RECOVER THE BIFURCATION TYPE FULLY?

Design rules (per author decision, 2026-07-13):
  - ONLY the benchmark-stage algorithms are used: Student-t mixture, skew-t mixture (unsupervised,
    k=2) and the nearest-centroid oracle (label-informed ceiling, HELD OUT: centroids from a
    stratified half, scored on the other half). NO least-squares / regression classifiers.
  - mu enters exactly the way x does: through analytic features. The x-only arm uses the paper's
    46-D feature space; the x+mu arm appends 7 mu-aware features built from moment and rank
    statistics only (correlations, Kendall tau, path descriptors; no model fitting).
  - Trajectories come from the exact benchmark samplers (Synthetic_Data_Gen): saddle-node with mu
    swept linearly through the fold, transcritical with constant mu > 0, Euler-Maruyama + AR(1)
    noise. No inflection-window rejections.

Three observation regimes x two feature sets x three algorithms:
  low-noise    sigma in {0.005, 0.01, 0.02}          (the clean limit: 'fully?' is judged here)
  canonical    sigma in {0.005 ... 0.20}             (the benchmark protocol)
  marginalised sigma in {0.05 ... 0.20}, 10-25 obs, window truncation (hostile observation)

Output: console table (95% CIs) + runs/unsup/bifurcation_explore/mc_mu_observed.csv
        + Manuscript/SI_figures/figS_mu_observed.png  (showcase + accuracy panels)
Run `python mc_mu_observed.py figure` to redraw the figure from the saved CSV without re-running.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau
from sklearn.preprocessing import StandardScaler
from paper_style import set_style, COL2
set_style()
from Synthetic_Data_Gen import (saddle_node_rhs, transcritical_rhs, sample_saddle_node,
                                sample_transcritical, integrate_sde)
from paper_figures import build_features, _tag
from unsup_real_world import fit_t_mixture, fit_skew_t_mixture

N_PER = 5000
SIGMAS = {"low-noise": [0.005, 0.01, 0.02],
          "canonical": [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
          "marginalised": [0.05, 0.10, 0.15, 0.20]}
REGIMES = ["low-noise", "canonical", "marginalised"]
RCOL = {"low-noise": "#8ab17d", "canonical": "#457b9d", "marginalised": "#e76f51"}


# ---------------------------------------------------------------- generation
def gen_one(label, rng, regime):
    """One SN (0) or TC (1) series with full state kept. Returns dict or None."""
    n_steps = int(rng.integers(15, 80))
    t_end = rng.uniform(5.0, 20.0)
    t = np.linspace(0, t_end, n_steps)
    sigma = rng.choice(SIGMAS[regime])
    phi = rng.uniform(0.0, 0.9)
    if label == 0:
        params, x0, x_ref, mu_sweep = sample_saddle_node(rng, n_steps, t_end)
        x = integrate_sde(saddle_node_rhs, x0, t, params, sigma, rng, phi, mu_sweep)
        mu = mu_sweep.copy()
    else:
        params, x0, x_ref, _ = sample_transcritical(rng, n_steps, t_end)
        x = integrate_sde(transcritical_rhs, x0, t, params, sigma, rng, phi, None)
        mu = np.full(n_steps, params["mu"])
    # validity: a genuine rise that reaches near its stable state and doesn't blow up
    if not np.isfinite(x).all() or np.abs(x).max() > 9: return None
    rise = x[-1] - x[0]
    if rise < 0.3 * abs(x_ref): return None
    if x[-1] < 0.7 * x_ref or x[-1] > 1.6 * x_ref: return None

    if regime == "marginalised":     # hostile observation: window truncation + subsampling
        i0 = int(rng.uniform(0.0, 0.15) * n_steps)
        i1 = int(rng.uniform(0.80, 1.00) * n_steps)
        if i1 - i0 < 10: return None
        t, x, mu = t[i0:i1], x[i0:i1], mu[i0:i1]
        n_obs = int(rng.integers(10, 26))
        if len(t) > n_obs:
            keep = np.unique(np.linspace(0, len(t) - 1, n_obs).astype(int))
            t, x, mu = t[keep], x[keep], mu[keep]
        if x.max() - x.min() < 0.3 * abs(x_ref): return None
    return dict(t=t, x=x, mu=mu, y=label)


def generate(regime, seed):
    rng = np.random.default_rng(seed)
    out = []
    for label in (0, 1):
        k = 0
        while k < N_PER:
            s = gen_one(label, rng, regime)
            if s is not None:
                out.append(s); k += 1
    return out


# ---------------------------------------------------------------- features
def to_grid100(x):
    xn = (x - x.min()) / (x.max() - x.min() + 1e-12)
    tt = np.linspace(0, 1, len(xn))
    return PchipInterpolator(tt, xn)(np.linspace(0, 1, 100))


def mu_feats(s):
    """mu-aware analytic features: descriptors of the observed drive mu(t) and moment/rank
    coupling statistics with x. NO model fitting of any kind. For a constant mu the
    correlation entries are undefined and set to 0; that zero itself encodes 'the drive did
    not move', which is a legitimate observable once mu is measured."""
    t, mu = s["t"], s["mu"]
    x = (s["x"] - s["x"].min()) / (s["x"].max() - s["x"].min() + 1e-12)
    dx = np.diff(x); xs = x[:-1]; mus = mu[:-1]

    def corr(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12: return 0.0
        v = np.corrcoef(a, b)[0, 1]
        return 0.0 if not np.isfinite(v) else float(v)

    if mu.max() - mu.min() > 1e-12:
        tau = kendalltau(np.arange(len(mu)), mu)[0]
        tau = 0.0 if not np.isfinite(tau) else float(tau)
    else:
        tau = 0.0
    return np.array([
        (mu[-1] - mu[0]) / (np.abs(mu).max() + 1e-12),   # net drive change (sweep indicator)
        tau,                                             # drive trend (rank statistic)
        float(np.mean(mu < 0)),                          # fraction of time below mu = 0
        corr(dx, mus),                                   # additive-coupling moment
        corr(dx, mus * xs),                              # multiplicative-coupling moment
        corr(x, mu),                                     # state-drive co-movement
        float(mu.min() / (np.abs(mu).max() + 1e-12)),    # deepest drive value (signed)
    ])


# ---------------------------------------------------------------- typing (benchmark methods only)
def hung_acc(lab, ys):
    C = np.array([[((lab == k) & (ys == c)).sum() for c in range(2)] for k in range(2)])
    r, cc = linear_sum_assignment(-C)
    return sum(C[i, cc[j]] for j, i in enumerate(r)) / len(ys)


def typing_arm(series, seed, with_mu, ablate_mu=()):
    """Student-t + skew-t (unsupervised, k=2) and held-out nearest-centroid oracle, on the 46-D
    feature space (with_mu=False) or the 46-D + 7 mu-feature space (with_mu=True). ablate_mu drops the
    given mu-feature column indices; (2, 6) = fraction-of-time-mu<0 and signed-deepest-mu, the two that
    separate a swept SN from a positive-swept TC by construction, so ablating them isolates the scaling
    channel (the coupling/co-movement moments) from bare sign-of-mu detection."""
    X = np.vstack([to_grid100(s["x"]) for s in series])
    ys = np.array([s["y"] for s in series])
    F = np.nan_to_num(build_features(X))
    if with_mu:
        M = np.vstack([mu_feats(s) for s in series])
        if ablate_mu:
            M = np.delete(M, list(ablate_mu), axis=1)
        F = np.hstack([F, np.nan_to_num(np.clip(M, -50, 50))])
    Z = StandardScaler().fit_transform(F)

    acc_t = hung_acc(fit_t_mixture(Z, 2, seed=seed, n_init=3)[0], ys)
    acc_st = hung_acc(fit_skew_t_mixture(Z, 2, seed=seed, n_init=3)[0], ys)

    rng = np.random.default_rng(seed)
    tr = np.zeros(len(ys), bool)
    for c in (0, 1):
        idx = np.where(ys == c)[0]
        tr[rng.choice(idx, len(idx) // 2, replace=False)] = True
    cents = np.vstack([Z[tr & (ys == c)].mean(0) for c in range(2)])
    te = ~tr
    pred = np.argmin(np.linalg.norm(Z[te][:, None, :] - cents[None], axis=2), axis=1)
    acc_or = (pred == ys[te]).mean()
    return {"Student-t": acc_t, "skew-t": acc_st, "oracle": acc_or}, int(te.sum())


# ---------------------------------------------------------------- run
def main():
    rows = []
    for regime in REGIMES:
        series = generate(regime, seed=REGIMES.index(regime))
        n = len(series)
        ci = lambda p, m=n: 196 * np.sqrt(p * (1 - p) / m)   # 95% CI half-width, pp
        print(f"\n=== regime: {regime} (N={n}) ===")
        for with_mu, fname in [(False, "x only"), (True, "x + mu")]:
            accs, n_te = typing_arm(series, seed=0, with_mu=with_mu)
            for algo, a in accs.items():
                m = n_te if algo == "oracle" else n
                print(f"  {fname:7s} {algo:10s}: {100*a:5.1f}% +- {ci(a, m):.1f}")
                rows.append(dict(regime=regime, featset=fname, algo=algo, acc=100 * a))
    df = pd.DataFrame(rows)
    df.to_csv("runs/unsup/bifurcation_explore/mc_mu_observed.csv", index=False)
    print("\nSaved -> runs/unsup/bifurcation_explore/mc_mu_observed.csv")
    si_figure(df)


# ---------------------------------------------------------------- SI figure
def si_figure(df):
    """(a, b) exemplar SN + TC series at the two observation levels (latent mu ghosted);
    (c, d) typing accuracy of the benchmark algorithms on each feature set, three regimes."""
    SNCOL, TCCOL = "#d62828", "#457b9d"; MUCOL = "#333333"
    rng = np.random.default_rng(7)
    ex = {}
    while len(ex) < 2:                                  # clean, readable exemplars
        for label in (0, 1):
            if label in ex: continue
            n_steps = 60; t_end = 12.0; t = np.linspace(0, t_end, n_steps)
            if label == 0:
                params, x0, x_ref, mu_sweep = sample_saddle_node(rng, n_steps, t_end)
                speed = -mu_sweep[0] / (x_ref ** 2 + 1e-12)
                if x_ref < 0.22 or speed > 1.5: continue    # big fold signal, slow sweep
                x = integrate_sde(saddle_node_rhs, x0, t, params, 0.03, rng, 0.5, mu_sweep)
                mu = mu_sweep.copy()
            else:
                params, x0, x_ref, _ = sample_transcritical(rng, n_steps, t_end)
                x = integrate_sde(transcritical_rhs, x0, t, params, 0.03, rng, 0.5, None)
                mu = np.full(n_steps, params["mu"])
            if np.isfinite(x).all() and x[-1] > 0.8 * x_ref and x[-1] < 1.4 * x_ref \
                    and x[-1] - x[0] > 0.5 * abs(x_ref):
                ex[label] = dict(t=t, x=x, mu=mu)

    cols = [("(a) what the paper observes:\nnormalized curve only, $\\mu$ latent", False),
            ("(b) $\\mu(t)$ observed alongside\nthe normalized curve", True)]

    fig = plt.figure(figsize=(COL2, 7.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.35], hspace=0.62, wspace=0.42)
    for r, (label, col, name) in enumerate([(0, SNCOL, "saddle-node"), (1, TCCOL, "transcritical")]):
        s = ex[label]; t, x, mu = s["t"], s["x"], s["mu"]
        xn = (x - x.min()) / (x.max() - x.min())
        for c, (title, mu_obs) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c]); ax2 = ax.twinx()
            ax2.spines["right"].set_visible(True)
            ax.plot(t, xn, color=col, lw=1.6, zorder=3)
            ax2.plot(t, mu, color=MUCOL, lw=1.1, ls="--", alpha=1.0 if mu_obs else 0.15, zorder=2)
            if label == 0 and mu_obs:
                ax2.axhline(0, color="#999", lw=0.6, ls=":")
                ax2.annotate("fold crossed", xy=(t[np.argmin(np.abs(mu))], 0), fontsize=5.2,
                             color="#777", xytext=(-38, 5), textcoords="offset points")
            if not mu_obs:
                ax2.annotate("$\\mu$ latent", xy=(0.60, 0.10), xycoords="axes fraction",
                             fontsize=6, color="#aaa", style="italic")
                ax2.set_yticks([])
            else:
                ax2.tick_params(labelsize=5.5, colors=MUCOL)
            ax.set_ylabel("$x$ (norm.)", fontsize=6.5)
            ax.tick_params(labelsize=5.5)
            if r == 1: ax.set_xlabel("time", fontsize=6.5)
            if r == 0: _tag(ax, title, y=1.12, size=6.5)
            if c == 0:
                ax.text(-0.30, 0.5, name, transform=ax.transAxes, rotation=90, va="center",
                        fontsize=7, fontweight="bold", color=col)
    handles = [plt.Line2D([], [], color="#666", lw=1.6, label="adoption $x$ (left axis)"),
               plt.Line2D([], [], color=MUCOL, lw=1.1, ls="--",
                          label="control parameter $\\mu$ (right axis)")]
    fig.legend(handles=handles, ncol=2, fontsize=6.5, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.012))

    # (c, d) accuracy panels: one per feature set; benchmark algorithms x three regimes
    algos = ["Student-t", "skew-t", "oracle"]
    for c, (fname, tg) in enumerate([("x only", "(c) typing from the curve alone"),
                                     ("x + mu", "(d) typing with $\\mu$ observed")]):
        ax = fig.add_subplot(gs[2, c])
        xs = np.arange(len(algos)); w = 0.26
        for k, regime in enumerate(REGIMES):
            vals = [df[(df.regime == regime) & (df.featset == fname) & (df.algo == a)].acc.iloc[0]
                    for a in algos]
            for i, v in enumerate(vals):
                ax.bar(xs[i] + (k - 1) * w, v, w, color=RCOL[regime],
                       hatch="///" if algos[i] == "oracle" else None, edgecolor="white", lw=0.3,
                       label=regime if i == 0 and c == 0 else None)
                ax.text(xs[i] + (k - 1) * w, v + 0.6, f"{v:.0f}", ha="center", fontsize=5.6)
        ax.axhline(50, color="#999", lw=0.8, ls=":")
        ax.axhline(100, color="#999", lw=0.6, ls=":")
        if c == 0: ax.text(-0.38, 51.5, "chance", fontsize=5.8, color="#999")
        ax.set_xticks(xs)
        ax.set_xticklabels(["Student-$t$\n(unsup.)", "skew-$t$\n(unsup.)", "oracle\n(held-out)"],
                           fontsize=6.2)
        ax.set_ylabel("SN vs TC typing accuracy (%)", fontsize=7)
        ax.set_ylim(45, 106)
        _tag(ax, tg, y=1.05)
    handles2 = [plt.Rectangle((0, 0), 1, 1, color=RCOL[r]) for r in REGIMES]
    fig.legend(handles2, REGIMES, ncol=3, fontsize=6.2, frameon=False, title="observation regime",
               title_fontsize=6.2, loc="lower center", bbox_to_anchor=(0.5, 0.315))

    import os
    SI = "Manuscript/SI_figures"; os.makedirs(SI, exist_ok=True)
    fig.savefig(f"{SI}/figS_mu_observed.png", bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {SI}/figS_mu_observed.png")


if __name__ == "__main__":
    import sys
    if "figure" in sys.argv:
        si_figure(pd.read_csv("runs/unsup/bifurcation_explore/mc_mu_observed.csv"))
    else:
        main()
