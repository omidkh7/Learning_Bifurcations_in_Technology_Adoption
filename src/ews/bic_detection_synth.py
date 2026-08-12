#!/usr/bin/env python3
"""
SYNTHETIC-FIRST VALIDATION GATE for the Boettiger & Hastings (2012) style BIC detection of a
bifurcation in a COMPLETED adoption curve. Make-or-break, exactly as we vetted S-A1: if the BIC cannot
separate our own synthetic bifurcation classes from the non-bifurcating nulls (where ground truth is
known), we do NOT apply it to real data.

Models fit to the detrended residuals r_t of the full series (trend x_hat_t and inflection t_inf from
the Gaussian smooth, sigma=n/8):
  M_bif  (bifurcation / critical slowing): AR(1) whose restoring rate DIPS at the transition,
         r_t = r_min + a*|t - t_inf|, so the local variance gamma^2_t = sigma^2/(2 r_t) PEAKS at t_inf.
         (Boettiger-Hastings LSN = AMOC MLE, adapted to a passage rather than an approach.) 3 params.
  M_ou   (constant-r OU / adiabatic stable-twin): flat restoring rate, constant variance. 2 params.
         M_bif nests M_ou (a=0), so the deviance 2(logL_bif - logL_ou) >= 0 (Cox's delta).
  M_mult (multiplicative / heteroscedastic null, Seekell 2011): constant restoring, variance tracks the
         level, s^2_t = c * x_hat_t(1 - x_hat_t). No critical slowing. 2 params.
Detection = M_bif has the lowest BIC. Ground-truth classes: SN, TC (bifurcation) vs additive stable-twin
(benchmark null) and a generated multiplicative null. Reports detection/false-alarm rates + AUC; the
multiplicative null is the non-circularity test.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from benchmark_data import load_benchmark, CANONICAL_NULL
from paper_style import set_style, COL2
from paper_figures import _tag
try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None
set_style()
SCR = "runs"   # diagnostic PNG output (repo-relative)
SNCOL, TCCOL, NUCOL, MUCOL = "#d62828", "#457b9d", "#9aa0a6", "#e07a5f"


def resid(v):
    vn = (v - v.min()) / (v.max() - v.min() + 1e-12); n = len(vn)
    trend = gaussian_filter1d(vn, sigma=max(2, n // 8))
    tinf = float(np.argmax(np.gradient(trend)) / (n - 1))
    return vn - trend, np.clip(trend, 0, 1), tinf, n


def _nll_ou(th, r):
    rho = 0.999 / (1 + np.exp(-th[0])); g2 = np.exp(th[1])
    v = g2 * (1 - rho**2); res = r[1:] - rho * r[:-1]
    return 0.5 * np.sum(np.log(2 * np.pi * v) + res**2 / v)


def _nll_bif(th, r, tinf, n):
    s2 = np.exp(th[0]); rmin = np.exp(th[1]); a = np.exp(th[2])
    t = np.arange(n) / (n - 1)
    rt = np.clip(rmin + a * np.abs(t - tinf), 1e-4, 60)
    rho = np.exp(-rt); g2 = s2 / (2 * rt)
    v = np.clip(g2 * (1 - rho**2), 1e-12, None)
    res = r[1:] - rho[1:] * r[:-1]
    return 0.5 * np.sum(np.log(2 * np.pi * v[1:]) + res**2 / v[1:])


def _nll_mult(th, r, w):
    rho = 0.999 / (1 + np.exp(-th[0])); c = np.exp(th[1])
    v = np.clip(c * w * (1 - rho**2), 1e-12, None)
    res = r[1:] - rho * r[:-1]
    return 0.5 * np.sum(np.log(2 * np.pi * v[1:]) + res**2 / v[1:])


def _fit(nll, x0s, args):
    best = None
    for x0 in x0s:
        try:
            res = minimize(nll, x0, args=args, method="Nelder-Mead",
                           options=dict(maxiter=3000, xatol=1e-6, fatol=1e-8))
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            pass
    return best.fun


def bic_all(v, seed=0):
    r, xh, tinf, n = resid(v)
    if np.std(r) < 1e-9 or n < 12:
        return None
    rng = np.random.default_rng(seed); v0 = np.log(max(np.var(r), 1e-6))
    w = np.clip(xh * (1 - xh), 1e-3, None)
    st = [rng.standard_normal(1) for _ in range(3)]
    nll_ou = _fit(_nll_ou, [[s[0], v0] for s in st], (r,))
    nll_bif = _fit(_nll_bif, [[v0, np.log(0.3 + rng.random()), np.log(0.5 + rng.random())] for _ in range(4)], (r, tinf, n))
    nll_mult = _fit(_nll_mult, [[s[0], v0] for s in st], (r, w))
    m = n - 1; lg = np.log(m)
    bic = dict(bif=2 * nll_bif + 3 * lg, ou=2 * nll_ou + 2 * lg, mult=2 * nll_mult + 2 * lg)
    dev = 2 * (nll_ou - nll_bif)                      # Cox deviance, bif vs nested OU (>=0)
    return bic, dev


def make_mult_null(m, rng):
    out = []
    for _ in range(m):
        n = 100; t = np.linspace(0, 1, n)
        t0 = rng.uniform(0.35, 0.7); k = rng.uniform(6, 14)
        x = 1 / (1 + np.exp(-k * (t - t0)))
        phi = rng.uniform(0.0, 0.6); sig = rng.uniform(0.03, 0.09)
        eta = np.zeros(n)
        for i in range(1, n):
            eta[i] = phi * eta[i - 1] + rng.standard_normal()
        eta = eta / (eta.std() + 1e-9)
        out.append(np.clip(x + sig * np.sqrt(x * (1 - x) + 1e-3) * eta, 0, 1.3))
    return np.array(out)


def main():
    X, y = load_benchmark(CANONICAL_NULL, 250, 0)
    rng = np.random.default_rng(7)
    groups = {"SN": X[y == 0], "TC": X[y == 1], "null(add)": X[y == 2], "null(mult)": make_mult_null(250, rng)}
    res = {}
    for name, series in groups.items():
        devs, detect = [], []
        for j, v in enumerate(series):
            out = bic_all(v, seed=j)
            if out is None:
                continue
            bic, dev = out
            devs.append(dev); detect.append(bic["bif"] == min(bic.values()))
        res[name] = dict(dev=np.array(devs), detect=np.array(detect))
        print(f"{name:11s} n={len(detect):3d}  BIC-picks-bifurcation = {100*np.mean(detect):3.0f}%   "
              f"median deviance = {np.median(devs):6.1f}")

    bif_dev = np.r_[res["SN"]["dev"], res["TC"]["dev"]]
    print("\nAUC of the deviance (bifurcation vs null); 0.5 = no power:")
    for nl in ("null(add)", "null(mult)"):
        if roc_auc_score is not None:
            sc = np.r_[bif_dev, res[nl]["dev"]]; lab = np.r_[np.ones(len(bif_dev)), np.zeros(len(res[nl]["dev"]))]
            print(f"  bifurcation vs {nl:11s}: AUC = {roc_auc_score(lab, sc):.2f}")
    tpr = np.mean(np.r_[res['SN']['detect'], res['TC']['detect']])
    print(f"\ntrue-positive (bifurcation detected on SN/TC): {100*tpr:.0f}%")
    print(f"false-alarm (bifurcation 'detected' on additive null): {100*np.mean(res['null(add)']['detect']):.0f}%")
    print(f"false-alarm (bifurcation 'detected' on MULTIPLICATIVE null): {100*np.mean(res['null(mult)']['detect']):.0f}%")
    figure(res, bif_dev)


def figure(res, bif_dev):
    fig, ax = plt.subplots(1, 3, figsize=(COL2, 2.7), gridspec_kw=dict(wspace=0.34))
    A = ax[0]
    bins = np.linspace(0, np.percentile(np.r_[bif_dev, res["null(add)"]["dev"], res["null(mult)"]["dev"]], 97), 26)
    A.hist(bif_dev, bins=bins, density=True, color=SNCOL, alpha=0.5, label="bifurcation (SN+TC)")
    A.hist(res["null(add)"]["dev"], bins=bins, density=True, color=NUCOL, alpha=0.55, label="null (additive)")
    A.hist(res["null(mult)"]["dev"], bins=bins, density=True, color=MUCOL, alpha=0.45, label="null (multiplic.)")
    A.set_xlabel("deviance  $2(\\log L_{bif}-\\log L_{OU})$", fontsize=6.6); A.set_ylabel("density", fontsize=7)
    A.legend(fontsize=5.2, frameon=False); _tag(A, "(a) deviance: bifurcation vs nulls")

    B = ax[1]
    labs = list(res.keys()); vals = [100 * np.mean(res[k]["detect"]) for k in labs]
    cols = [SNCOL, TCCOL, NUCOL, MUCOL]
    B.bar(range(len(labs)), vals, color=cols, alpha=0.8)
    B.set_xticks(range(len(labs))); B.set_xticklabels(labs, fontsize=5.4, rotation=15)
    B.set_ylabel("% BIC picks bifurcation", fontsize=6.6); B.set_ylim(0, 105)
    B.axhline(50, color="#888", lw=0.7, ls=":"); _tag(B, "(b) detection vs false alarm")

    C = ax[2]
    if roc_auc_score is not None:
        from sklearn.metrics import roc_curve
        for nl, c in [("null(add)", NUCOL), ("null(mult)", MUCOL)]:
            sc = np.r_[bif_dev, res[nl]["dev"]]; lab = np.r_[np.ones(len(bif_dev)), np.zeros(len(res[nl]["dev"]))]
            fpr, tpr, _ = roc_curve(lab, sc); auc = roc_auc_score(lab, sc)
            C.plot(fpr, tpr, color=c, lw=1.6, label=f"vs {nl.split('(')[1][:-1]}: {auc:.2f}")
        C.plot([0, 1], [0, 1], color="#bbb", lw=0.8, ls="--")
    C.set_xlabel("false-positive rate", fontsize=6.6); C.set_ylabel("true-positive rate", fontsize=7)
    C.legend(fontsize=5.4, loc="lower right", frameon=False, title="AUC", title_fontsize=5.4)
    _tag(C, "(c) ROC (deviance detector)")
    fig.savefig(f"{SCR}/bic_synth_validation.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved -> {SCR}/bic_synth_validation.png")


if __name__ == "__main__":
    main()
