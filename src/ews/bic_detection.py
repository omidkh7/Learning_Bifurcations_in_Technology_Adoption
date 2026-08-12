#!/usr/bin/env python3
"""
Bootstrap-calibrated deviance detector of a bifurcation in COMPLETED adoption curves
(Boettiger & Hastings 2012 method; the LSN model = the AMOC MLE). Switches the statistic from the
broken min-BIC-of-3 to Cox's deviance delta = 2(logL_bif - logL_OU) with a length-matched parametric
OU-null pool giving a per-series p-value. Detection = p < 0.05 (calibrated 5% false alarm).

  validate  : synthetic gate at 1000/class. Reports SN-vs-null and TC-vs-null AUC (comparable to the
              feature route's ~100% / ~55%), calibrated TPR(SN),TPR(TC) at 5% FPR on the OU null, and
              FPR on the additive stable-twin and the multiplicative-noise null (the confound guard).
  real      : collect() main tier. Per-series p; detection rate; cache + figure.
"""
import warnings, sys; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from bic_detection_synth import resid, _nll_ou, _nll_bif, _nll_mult, _fit, make_mult_null
from benchmark_data import load_benchmark, CANONICAL_NULL
from growth_compare import collect
from paper_style import set_style, COL2
from paper_figures import _tag, FAMCOL, FAMLABEL
set_style()
SCR = "runs"   # diagnostic/cache outputs (repo-relative)
K_POOL = 1200


def _dev(r, tinf, n, rng):
    """Cox deviance 2(logL_bif - logL_OU) on residuals r (>=0)."""
    from scipy.optimize import minimize
    v0 = np.log(max(np.var(r), 1e-6)); b = None
    for _ in range(3):
        res = minimize(_nll_ou, [rng.standard_normal(), v0 + 0.3 * rng.standard_normal()], args=(r,),
                       method="Nelder-Mead", options=dict(maxiter=3000))
        b = res if b is None or res.fun < b.fun else b
    nll_bif = _fit(_nll_bif, [[v0, np.log(0.3 + rng.random()), np.log(0.5 + rng.random())] for _ in range(3)], (r, tinf, n))
    return 2 * (b.fun - nll_bif)


def fit_series(v, seed=0):
    """Return (deviance, trend, resid_std, phi, n); None if unusable. Keeps the deterministic trend
    and residual (std, AR1-phi) so the null pool can regenerate trend + stationary-noise surrogates."""
    from scipy.ndimage import gaussian_filter1d
    vn = (v - v.min()) / (v.max() - v.min() + 1e-12); n = len(vn)
    if np.std(vn) < 1e-9 or n < 12:
        return None
    trend = gaussian_filter1d(vn, sigma=max(2, n // 8)); r = vn - trend
    if np.std(r) < 1e-9:
        return None
    tinf = float(np.argmax(np.gradient(trend)) / (n - 1))
    dev = _dev(r, tinf, n, np.random.default_rng(seed))
    phi = np.clip(np.corrcoef(r[:-1], r[1:])[0, 1] if n > 5 else 0.0, 0.0, 0.95)
    return dev, trend, r.std(), (phi if np.isfinite(phi) else 0.0), n


def build_pool(params, K=K_POOL, seed=99):
    """PROPER null pool: trend + stationary AR(1) noise (no critical slowing), RE-DETRENDED, so the
    trend+detrending variance-bump artifact is present in the null exactly as in a non-bifurcating
    S-curve (the stable twin). This is the null that ews_null_control uses, lifted to the deviance."""
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.default_rng(seed); pn = np.empty(K); pd = np.empty(K)
    for k in range(K):
        _, trend, std, phi, n = params[rng.integers(len(params))]
        eta = np.zeros(n)
        for t in range(1, n):
            eta[t] = phi * eta[t - 1] + rng.standard_normal()
        sur = trend + eta / (eta.std() + 1e-9) * std
        tr = gaussian_filter1d(sur, sigma=max(2, n // 8)); r = sur - tr
        pn[k] = n
        if np.std(r) < 1e-9:
            pd[k] = 0.0; continue
        ti = float(np.argmax(np.gradient(tr)) / (n - 1))
        pd[k] = _dev(r, ti, n, np.random.default_rng(1000 + k))
    return pn, pd


def pval(dev, n, pn, pd, tol=8):
    m = np.abs(pn - n) <= tol
    if m.sum() < 60:
        m = np.abs(pn - n) <= 3 * tol
    return (1 + np.sum(pd[m] >= dev - 1e-9)) / (1 + m.sum())


def auc(pos, neg):
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
    except Exception:
        return np.nan


def validate(n_per=1000):
    print(f"=== SYNTHETIC GATE (calibrated deviance), {n_per}/class ===")
    X, y = load_benchmark(CANONICAL_NULL, n_per, 0); rng = np.random.default_rng(7)
    groups = {"SN": X[y == 0], "TC": X[y == 1], "add-null": X[y == 2], "mult-null": make_mult_null(n_per, rng)}
    F = {}
    for nm, S in groups.items():
        f = [fit_series(v, seed=j) for j, v in enumerate(S)]
        F[nm] = [x for x in f if x is not None]
        print(f"  fit {nm:9s}: {len(F[nm])} series")
    pn, pd = build_pool(F["add-null"])          # null pool from the stable-twin's own trend+noise
    P = {nm: np.array([pval(d, n, pn, pd) for d, _, _, _, n in F[nm]]) for nm in groups}
    dev = {nm: np.array([x[0] for x in F[nm]]) for nm in groups}
    print(f"\nAUC of deviance:  SN-vs-null {auc(dev['SN'],dev['add-null']):.2f}   "
          f"TC-vs-null {auc(dev['TC'],dev['add-null']):.2f}   "
          f"(SN+TC)-vs-mult {auc(np.r_[dev['SN'],dev['TC']],dev['mult-null']):.2f}")
    print(f"calibrated at p<0.05:  TPR(SN)={100*np.mean(P['SN']<0.05):.0f}%  TPR(TC)={100*np.mean(P['TC']<0.05):.0f}%  "
          f"|  FPR(add-null)={100*np.mean(P['add-null']<0.05):.0f}%  FPR(mult-null)={100*np.mean(P['mult-null']<0.05):.0f}%")
    return F, P


def run_real():
    print("\n=== REAL (main tier, collect()) ===")
    fams = ["Historical", "Renewables", "BEV"]
    rows = {f: [] for f in fams}; allf = []
    for g, nm, yy, vv in collect():
        if g not in fams:
            continue
        f = fit_series(np.asarray(vv, float), seed=abs(hash(nm)) % (2**31))
        if f is None:
            continue
        rows[g].append((nm, f)); allf.append(f)
    pn, pd = build_pool(allf)
    det = {}
    for g in fams:
        ps = np.array([pval(f[0], f[4], pn, pd) for _, f in rows[g]])
        det[g] = ps
        print(f"  {g:11s} n={len(ps):3d}  detected (p<0.05) = {100*np.mean(ps<0.05):3.0f}%   median deviance={np.median([f[0] for _,f in rows[g]]):.1f}")
    allp = np.concatenate([det[g] for g in fams])
    import json
    print(f"  OVERALL flagged = {100*np.mean(allp<0.05):.0f}%  (a detection LIMIT, not a rate: the "
          f"stable-twin false-alarm rate is ~60% at this threshold, so real sits at/below the null; Fig S5)")
    np.savez(f"{SCR}/bic_real_cache.npz", **{f"p_{g}": det[g] for g in fams})
    rates = {g: round(100 * float(np.mean(det[g] < 0.05)), 1) for g in fams}
    rates["OVERALL"] = round(100 * float(np.mean(allp < 0.05)), 1)
    json.dump(rates, open("results/bic_real_detection_rates.json", "w"), indent=2)
    return rows, det


CACHE_FIG = "results/bic_fig_cache.npz"
CB, CN, CM, CR = "#d62828", "#3a3a3a", "#e07a5f", "#6a4c93"   # bifurcation, stable-twin, multiplicative, real


def cache_and_plot(F, P_in, rows):
    key = lambda k: k.replace("-", "_")
    dev = {key(nm): np.array([x[0] for x in F[nm]]) for nm in F}
    Pd = {key(nm): np.asarray(P_in[nm]) for nm in P_in}
    real_dev = np.array([f[0] for g in rows for _, f in rows[g]])
    np.savez(CACHE_FIG, real_dev=real_dev, **{f"dev_{k}": dev[k] for k in dev}, **{f"P_{k}": Pd[k] for k in Pd})
    make_figure()


def make_figure():
    from sklearn.metrics import roc_curve, roc_auc_score
    z = np.load(CACHE_FIG)
    dev = {k[4:]: z[k] for k in z.files if k.startswith("dev_")}
    P = {k[2:]: z[k] for k in z.files if k.startswith("P_")}
    real_dev = z["real_dev"]; bif = np.r_[dev["SN"], dev["TC"]]
    hi = np.percentile(np.r_[bif, dev["add_null"], dev["mult_null"]], 97); bins = np.linspace(0, hi, 26)
    fig, ax = plt.subplots(2, 2, figsize=(COL2, 5.2), gridspec_kw=dict(hspace=0.42, wspace=0.3))
    # (a) synthetic deviance distributions -- BORDER-ONLY step outlines
    A = ax[0, 0]
    A.hist(bif, bins=bins, density=True, histtype="step", color=CB, lw=1.8, label="bifurcation (SN+TC)")
    A.hist(dev["add_null"], bins=bins, density=True, histtype="step", color=CN, lw=1.5, label="stable-twin null")
    A.hist(dev["mult_null"], bins=bins, density=True, histtype="step", color=CM, lw=1.5, ls=(0, (4, 1.5)),
           label="multiplicative null")
    A.set_xlabel("deviance $2(\\log L_{bif}-\\log L_{OU})$", fontsize=6.6); A.set_ylabel("density", fontsize=7)
    A.legend(fontsize=6.3, frameon=False); _tag(A, "(a)")
    # (b) ROC
    B = ax[0, 1]
    for pos, neg, c, lab in [(dev["SN"], dev["add_null"], CB, "SN vs null"),
                             (dev["TC"], dev["add_null"], "#457b9d", "TC vs null"),
                             (bif, dev["mult_null"], CM, "vs multiplic.")]:
        yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]; sc = np.r_[pos, neg]
        fpr, tpr, _ = roc_curve(yy, sc); B.plot(fpr, tpr, color=c, lw=1.5, label=f"{lab}: {roc_auc_score(yy, sc):.2f}")
    B.plot([0, 1], [0, 1], color="#bbb", lw=0.8, ls="--")
    B.set_xlabel("false-positive rate", fontsize=6.6); B.set_ylabel("true-positive rate", fontsize=7)
    B.legend(fontsize=6.3, loc="lower right", frameon=False, title="AUC", title_fontsize=6.0)
    _tag(B, "(b)")
    # (c) calibrated rates
    C = ax[1, 0]
    vals = [100 * np.mean(P[k] < 0.05) for k in ("SN", "TC", "add_null", "mult_null")]
    C.bar(range(4), vals, color=[CB, "#457b9d", CN, CM], alpha=0.85)
    C.axhline(5, color="#333", lw=0.8, ls="--"); C.text(3.4, 9, "5% target", fontsize=5.2, ha="right")
    C.set_xticks(range(4)); C.set_xticklabels(["TPR\nSN", "TPR\nTC", "FPR\nstable-twin", "FPR\nmultip."], fontsize=5.3)
    C.set_ylim(0, 105); C.set_ylabel("% flagged at $p<0.05$", fontsize=6.6)
    _tag(C, "(c)")
    # (d) real vs synthetic
    D = ax[1, 1]
    D.hist(np.clip(real_dev, 0, hi), bins=bins, density=True, histtype="step", color=CR, lw=1.8, label="real adoption")
    D.hist(np.clip(bif, 0, hi), bins=bins, density=True, histtype="step", color=CB, lw=1.5, label="synth. bifurcation")
    D.hist(np.clip(dev["add_null"], 0, hi), bins=bins, density=True, histtype="step", color=CN, lw=1.5, label="synth. stable-twin")
    D.set_xlabel("deviance", fontsize=6.6); D.set_ylabel("density", fontsize=7)
    D.legend(fontsize=6.3, frameon=False); _tag(D, "(d)")
    out = "figures/si/figS_bic_detection.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        make_figure()
    else:
        F, P = validate(600)
        rows, det = run_real()
        cache_and_plot(F, P, rows)
