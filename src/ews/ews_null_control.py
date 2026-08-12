#!/usr/bin/env python3
"""
ews_null_control.py
===================
AUDIT RESPONSE (Mahdi, Tier-1 #2): does the Fig-2 "rising pre-takeoff variance" survive a matched
null in which there is NO critical slowing down by construction?

Concern: the residual after Gaussian detrending of an accelerating curve carries a deterministic
curvature term that grows into takeoff, so a rising rolling variance can be produced mechanically.

Test, per qualifying real series (three realised families, len >= 12, >= 4 pre-takeoff windows):
  1. Extract the series' own deterministic skeleton (the same Gaussian trend ews_series uses) and
     its residual amplitude s and lag-1 autocorrelation phi.
  2. Generate M surrogates with NO state-dependent dynamics: skeleton + stationary noise,
     (a) iid N(0, s), and (b) AR(1) with coefficient phi scaled to std s.
  3. Push every surrogate through the EXACT ews_series pipeline and compute the same pre-takeoff
     Kendall tau of the rolling std.
  4. Report: the null positive rate (how often surrogates show tau > 0: the size of the mechanical
     bias) and the per-series exceedance p-value (fraction of surrogates with tau >= tau_real).

Also reports a detrending-free variant: the same statistic computed on rolling std of FIRST
DIFFERENCES (differencing removes smooth trends without a smoother, so curvature leakage is gone).

Output: console summary + runs/unsup/bifurcation_explore/ews_null_control.csv
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau
from ews_groups import ews_series
from growth_compare import collect

M = 200
FAMS = ["Historical", "Renewables", "BEV"]


def pre_tau_sd(v):
    """Pre-takeoff Kendall tau of rolling std via the exact paper pipeline. NaN if unusable."""
    pos, ac, sd, ti = ews_series(np.asarray(v, float))
    if len(pos) < 4: return np.nan
    pre = pos <= max(ti, pos[0] + 1e-9)
    if pre.sum() < 4: return np.nan
    t = kendalltau(pos[pre], sd[pre])[0]
    return t if np.isfinite(t) else np.nan


def pre_tau_diff(v):
    """Detrending-free variant: rolling std of first differences, same windows and takeoff."""
    v = np.asarray(v, float)
    v = (v - v.min()) / (v.max() - v.min() + 1e-12)
    n = len(v)
    trend = gaussian_filter1d(v, sigma=max(2, n // 8))
    ti = float(np.argmax(np.gradient(trend)) / (n - 1))
    d = np.diff(v)
    w = max(5, n // 3)
    pos, sd = [], []
    for i in range(w, len(d) + 1):
        seg = d[i - w:i]
        if seg.std() < 1e-12: continue
        sd.append(seg.std()); pos.append(i / (n - 1))
    pos, sd = np.array(pos), np.array(sd)
    if len(pos) < 4: return np.nan
    pre = pos <= max(ti, pos[0] + 1e-9)
    if pre.sum() < 4: return np.nan
    t = kendalltau(pos[pre], sd[pre])[0]
    return t if np.isfinite(t) else np.nan


def surrogate(trend, s, phi, rng, kind):
    n = len(trend)
    if kind == "iid":
        return trend + rng.normal(0, s, n)
    e = rng.normal(0, 1, n); eta = np.zeros(n)
    for i in range(1, n): eta[i] = phi * eta[i - 1] + e[i]
    eta = eta / (eta.std() + 1e-12) * s
    return trend + eta


def main():
    rng = np.random.default_rng(0)
    rows = []
    for g, nm, yy, vv in collect():
        if g not in FAMS: continue
        v = np.asarray(vv, float)
        if len(v) < 12 or v.max() - v.min() < 1e-9: continue
        tau_real = pre_tau_sd(v)
        if not np.isfinite(tau_real): continue
        # skeleton + residual texture of THIS series
        vn = (v - v.min()) / (v.max() - v.min() + 1e-12)
        n = len(vn)
        trend = gaussian_filter1d(vn, sigma=max(2, n // 8))
        r = vn - trend
        s = r.std()
        phi = np.clip(np.corrcoef(r[:-1], r[1:])[0, 1] if n > 5 else 0.0, 0.0, 0.95)
        if not np.isfinite(phi): phi = 0.0

        out = dict(family=g, name=nm, n=n, tau_real=tau_real,
                   tau_diff_real=pre_tau_diff(v), s=s, phi=phi)
        for kind in ("iid", "ar1"):
            taus, taud = [], []
            for _ in range(M):
                sur = surrogate(trend, s, phi, rng, kind)
                t1 = pre_tau_sd(sur)
                if np.isfinite(t1): taus.append(t1)
                t2 = pre_tau_diff(sur)
                if np.isfinite(t2): taud.append(t2)
            taus, taud = np.array(taus), np.array(taud)
            out[f"nullpos_{kind}"] = np.mean(taus > 0) if len(taus) else np.nan
            out[f"p_{kind}"] = np.mean(taus >= tau_real) if len(taus) else np.nan
            out[f"nullpos_diff_{kind}"] = np.mean(taud > 0) if len(taud) else np.nan
            out[f"p_diff_{kind}"] = (np.mean(taud >= out["tau_diff_real"]) if len(taud)
                                     and np.isfinite(out["tau_diff_real"]) else np.nan)
        rows.append(out)

    df = pd.DataFrame(rows)
    df.to_csv("runs/unsup/bifurcation_explore/ews_null_control.csv", index=False)

    def rep(sub, tag):
        print(f"\n--- {tag} (N={len(sub)}) ---")
        print(f"  real var-tau > 0                    : {100*np.mean(sub.tau_real>0):5.1f}%")
        print(f"  null positive rate, iid  (median)   : {100*sub.nullpos_iid.median():5.1f}%")
        print(f"  null positive rate, AR1  (median)   : {100*sub.nullpos_ar1.median():5.1f}%")
        print(f"  exceeds matched null p<0.05, iid    : {100*np.mean(sub.p_iid<0.05):5.1f}%")
        print(f"  exceeds matched null p<0.05, AR1    : {100*np.mean(sub.p_ar1<0.05):5.1f}%")
        print(f"  exceeds matched null p<0.10, AR1    : {100*np.mean(sub.p_ar1<0.10):5.1f}%")
        ok = sub[np.isfinite(sub.tau_diff_real)]
        print(f"  DIFFERENCED variant: real tau > 0   : {100*np.mean(ok.tau_diff_real>0):5.1f}%  (N={len(ok)})")
        print(f"  differenced null positive, AR1 (med): {100*ok.nullpos_diff_ar1.median():5.1f}%")
        print(f"  differenced exceeds null p<0.05 AR1 : {100*np.mean(ok.p_diff_ar1<0.05):5.1f}%")

    rep(df, "POOLED")
    for g in FAMS: rep(df[df.family == g], g)
    print("\nSaved -> runs/unsup/bifurcation_explore/ews_null_control.csv")
    si_figure(df)


def si_figure(df=None):
    """SI house-style figure: observed positive rates vs the matched-null baseline."""
    import matplotlib.pyplot as plt
    from paper_style import set_style, COL2
    set_style()
    from paper_figures import _tag, FAMCOL, FAMLABEL
    import os
    if df is None:
        df = pd.read_csv("runs/unsup/bifurcation_explore/ews_null_control.csv")
    SI = "Manuscript/SI_figures"; os.makedirs(SI, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.9), gridspec_kw=dict(wspace=0.35))
    groups = ["Historical", "Renewables", "BEV", "POOLED"]
    for ax, (real_col, null_col, tg) in zip(axs, [
            ("tau_real", "nullpos_ar1", "(a) pipeline statistic (rolling std of residuals)"),
            ("tau_diff_real", "nullpos_diff_ar1", "(b) differenced variant (leak-free)")]):
        xs = np.arange(len(groups)); w = 0.36
        for i, g in enumerate(groups):
            sub = df if g == "POOLED" else df[df.family == g]
            ok = sub[np.isfinite(sub[real_col])]
            real = 100 * np.mean(ok[real_col] > 0)
            null = 100 * ok[null_col].median()
            col = "#555555" if g == "POOLED" else FAMCOL[g]
            ax.bar(xs[i] - w / 2, real, w, color=col)
            ax.bar(xs[i] + w / 2, null, w, color=col, alpha=0.35, hatch="///", edgecolor="white", lw=0.3)
            ax.text(xs[i] - w / 2, real + 1, f"{real:.0f}", ha="center", fontsize=6)
            ax.text(xs[i] + w / 2, null + 1, f"{null:.0f}", ha="center", fontsize=6)
        ax.axhline(50, color="#999", lw=0.7, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels([FAMLABEL.get(g, "pooled") for g in groups], fontsize=6.5)
        ax.set_ylabel("positive pre-takeoff variance trend (%)", fontsize=7)
        ax.set_ylim(0, 100)
        _tag(ax, tg, y=1.05)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#555555"),
               plt.Rectangle((0, 0), 1, 1, color="#555555", alpha=0.35, hatch="///")]
    fig.legend(handles, ["observed", "matched null (median surrogate rate)"], ncol=2, fontsize=6.5,
               frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.savefig(f"{SI}/figS_ews_nullcontrol.png", bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {SI}/figS_ews_nullcontrol.png")


if __name__ == "__main__":
    import sys
    if "figure" in sys.argv:
        si_figure()
    else:
        main()
