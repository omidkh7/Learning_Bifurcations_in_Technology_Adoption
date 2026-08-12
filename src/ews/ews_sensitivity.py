#!/usr/bin/env python3
"""
ews_sensitivity.py
==================
EXPLORATION (not in paper/SI): how sensitive are the Fig-2 EWS conclusions to
(i) the detrending bandwidth (Gaussian sigma, + a Lowess variant) and (ii) the rolling-window length?
Canonical Fig-2 setting: sigma = n/8, w = n/3 (trailing window, plotted at right edge).
Mercure et al. (Nat Commun 17:240, 2026) use w = n/2 and cut the series before the transition.

Panels:
  (a) % of series with positive PRE-TAKEOFF variance-tau across the (detrend x window) grid
  (b) same for AC1-tau
  (c) median takeoff-aligned variance curve per family, TRUNCATED at takeoff (canonical params)
  (d) same for AC1  -> preview of the "cut at takeoff" display option.

Output: results/figures/fig_ews_sensitivity.png + console table (incl. per-family canonical numbers).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau
from paper_style import set_style, COL2
set_style()
from paper_figures import _tag, FAMCOL, FAMLABEL
from growth_compare import collect

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOWESS = True
except Exception:
    HAS_LOWESS = False

FAMS3 = ["Historical", "Renewables", "BEV"]
SIGS = [("$\\sigma=n/12$", 1 / 12), ("$\\sigma=n/8$ (main)", 1 / 8), ("$\\sigma=n/4$", 1 / 4)]
if HAS_LOWESS: SIGS.append(("Lowess 0.4", "lowess"))
WINS = [("$w=n/4$", 1 / 4), ("$w=n/3$ (main)", 1 / 3), ("$w=n/2$", 1 / 2)]
GRID_REL = np.linspace(-0.6, 0.0, 40)          # pre-takeoff only


def ews(v, sig, wfrac):
    v = np.asarray(v, float)
    v = (v - v.min()) / (v.max() - v.min() + 1e-12)
    n = len(v)
    if sig == "lowess":
        trend = lowess(v, np.arange(n), frac=0.4, return_sorted=False)
    else:
        trend = gaussian_filter1d(v, sigma=max(2, int(n * sig)))   # floor: matches ews_series (n//8)
    r = v - trend
    w = max(5, int(n * wfrac))                                     # floor: matches ews_series (n//3)
    pos, ac, sd = [], [], []
    for i in range(w, n + 1):
        seg = r[i - w:i]
        if seg.std() < 1e-10: continue
        s0, s1 = seg[:-1] - seg[:-1].mean(), seg[1:] - seg[1:].mean()
        ac.append(float((s0 * s1).sum() / (np.sqrt((s0**2).sum() * (s1**2).sum()) + 1e-12)))
        sd.append(float(seg.std())); pos.append((i - 1) / (n - 1))
    ti = float(np.argmax(np.gradient(trend)) / (n - 1))
    return np.array(pos), np.array(ac), np.array(sd), ti


def pre_tau(pos, y, ti):
    pre = pos <= max(ti, pos[0] + 1e-9)
    if pre.sum() < 4: return np.nan
    return kendalltau(pos[pre], y[pre])[0]


def main():
    series = [(g, np.asarray(vv, float)) for g, nm, yy, vv in collect()
              if g in FAMS3 and len(vv) >= 12 and np.ptp(np.asarray(vv, float)) > 1e-9]
    print(f"{len(series)} qualifying series (>=12 pts, 3 realized families)")

    # ---- grid of % positive pre-takeoff tau ----
    P_sd = np.full((len(SIGS), len(WINS)), np.nan); P_ac = np.full_like(P_sd, np.nan)
    N = np.zeros_like(P_sd, int)
    for a, (slab, sig) in enumerate(SIGS):
        for b, (wlab, wf) in enumerate(WINS):
            tsd, tac = [], []
            for g, v in series:
                pos, ac, sd, ti = ews(v, sig, wf)
                if len(pos) < 4: continue
                ts, ta = pre_tau(pos, sd, ti), pre_tau(pos, ac, ti)
                if np.isfinite(ts): tsd.append(ts)
                if np.isfinite(ta): tac.append(ta)
            if tsd: P_sd[a, b] = 100 * np.mean(np.array(tsd) > 0)
            if tac: P_ac[a, b] = 100 * np.mean(np.array(tac) > 0)
            N[a, b] = len(tsd)
            print(f"{slab:20s} {wlab:16s}  var+ {P_sd[a,b]:5.1f}%   ac+ {P_ac[a,b]:5.1f}%   (N={N[a,b]})")

    # per-family canonical numbers
    print("\ncanonical (sigma=n/8, w=n/3) per family:")
    for f in FAMS3:
        ts = [pre_tau(*(lambda p, a, s, t: (p, s, t))(*ews(v, 1 / 8, 1 / 3))) for g, v in series if g == f]
        ta = [pre_tau(*(lambda p, a, s, t: (p, a, t))(*ews(v, 1 / 8, 1 / 3))) for g, v in series if g == f]
        ts = [x for x in ts if np.isfinite(x)]; ta = [x for x in ta if np.isfinite(x)]
        print(f"  {f:11s} var+ {100*np.mean(np.array(ts)>0):5.1f}%  ac+ {100*np.mean(np.array(ta)>0):5.1f}%  (N={len(ts)})")

    # ---- truncated-at-takeoff median curves (canonical) ----
    curves = {f: dict(sd=[], ac=[]) for f in FAMS3}
    for g, v in series:
        pos, ac, sd, ti = ews(v, 1 / 8, 1 / 3)
        if len(pos) < 4: continue
        xr = pos - ti
        if xr[-1] - xr[0] < 1e-6: continue
        gs = np.interp(GRID_REL, xr, sd, left=np.nan, right=np.nan)
        ga = np.interp(GRID_REL, xr, ac, left=np.nan, right=np.nan)
        curves[g]["sd"].append(gs); curves[g]["ac"].append(ga)

    # ---- SI figure: the two heatmaps only ----
    def heatmaps(axpair):
        for ax, P, tg in [(axpair[0], P_sd, "(a) variance trend: % positive pre-takeoff $\\tau$"),
                          (axpair[1], P_ac, "(b) autocorr. trend: % positive pre-takeoff $\\tau$")]:
            ax.imshow(P, cmap="RdYlGn", vmin=30, vmax=90, aspect="auto")
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    ax.text(j, i, f"{P[i,j]:.0f}%\nN={N[i,j]}", ha="center", va="center", fontsize=6)
            ax.set_xticks(range(len(WINS))); ax.set_xticklabels([w for w, _ in WINS], fontsize=6)
            ax.set_yticks(range(len(SIGS))); ax.set_yticklabels([s for s, _ in SIGS], fontsize=6)
            ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False, ec="k", lw=1.4))
            ax.spines[:].set_visible(True)
            _tag(ax, tg, y=1.06)

    import os
    SI = "figures/si"; os.makedirs(SI, exist_ok=True)
    figS, axsS = plt.subplots(1, 2, figsize=(COL2, 2.7), gridspec_kw=dict(wspace=0.42))
    heatmaps(axsS)
    figS.savefig(f"{SI}/figS_ews_sensitivity.png", bbox_inches="tight"); plt.close(figS)
    print(f"Saved -> {SI}/figS_ews_sensitivity.png")

    # ---- exploration figure (heatmaps + truncated-curve preview) ----
    fig, axs = plt.subplots(2, 2, figsize=(COL2, 5.6), gridspec_kw=dict(hspace=0.55, wspace=0.35))
    heatmaps(axs[0])

    for ax, key, tg, yl in [(axs[1, 0], "sd", "(c) rolling variance, truncated at takeoff", "rolling std"),
                            (axs[1, 1], "ac", "(d) lag-1 autocorr., truncated at takeoff", "lag-1 autocorr.")]:
        for f in FAMS3:
            A = np.array(curves[f][key])
            if not len(A): continue
            ok = np.isfinite(A).sum(0) >= 3
            ax.fill_between(GRID_REL[ok], np.nanpercentile(A[:, ok], 10, 0),
                            np.nanpercentile(A[:, ok], 90, 0), color=FAMCOL[f], alpha=0.13, lw=0)
            ax.plot(GRID_REL[ok], np.nanmedian(A[:, ok], 0), color=FAMCOL[f], lw=1.7, label=FAMLABEL[f])
        ax.axvline(0, color="#888", lw=0.8, ls=":")
        ax.set_xlabel("time to takeoff", fontsize=7.5); ax.set_ylabel(yl, fontsize=7.5)
        ax.legend(fontsize=6, loc="upper left"); _tag(ax, tg, y=1.08)

    fig.savefig("results/figures/fig_ews_sensitivity.png", bbox_inches="tight"); plt.close(fig)
    print("\nSaved -> results/figures/fig_ews_sensitivity.png")


if __name__ == "__main__":
    main()
