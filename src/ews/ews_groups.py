#!/usr/bin/env python3
"""
ews_groups.py
=============
§92.2 — "Adoption goes through a bifurcation" figure (canonical messaging §90): classical
early-warning signals (EWS / critical slowing down) over lifecycle time for ALL series in the
four groups (Historical, Renewables, BEV, CDR), WITHOUT typing the bifurcation.

Per series (real annual data, growth_compare.collect() cleaning):
  residuals = min-max series − Gaussian-smoothed trend (sigma = n/8, as residual_var_frac)
  rolling lag-1 autocorrelation + rolling std of residuals (window = n/3, ≥5 pts)
Overlaid per group (thin) + group median (thick), on normalised lifecycle time.
Quantitative readout: % of series with RISING pre-takeoff EWS (Kendall tau > 0 on t ≤ t_inflection)
— rising autocorrelation/variance on approach = the generic signature of an underlying
bifurcation, agnostic to its type (SN/TC/other).

Output: results/figures/fig_ews_groups.png + console stats
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau
from growth_compare import collect

GCOL = {"Historical": "#9aa0a6", "Renewables": "#2a9d8f", "BEV": "#6a4c93", "CDR": "#fb8500"}
GRID = np.linspace(0, 1, 60)
MIN_N = 12


def ews_series(vals):
    """min-max → detrend → rolling AC1 & std of residuals; returns (pos, ac1, std, t_inf)."""
    v = np.asarray(vals, float)
    v = (v - v.min()) / (v.max() - v.min() + 1e-12)
    n = len(v)
    trend = gaussian_filter1d(v, sigma=max(2, n // 8))
    r = v - trend
    w = max(5, n // 3)
    pos, ac, sd = [], [], []
    for i in range(w, n + 1):
        seg = r[i - w:i]
        if seg.std() < 1e-10:
            continue
        s0, s1 = seg[:-1] - seg[:-1].mean(), seg[1:] - seg[1:].mean()
        ac.append(float((s0 * s1).sum() / (np.sqrt((s0**2).sum() * (s1**2).sum()) + 1e-12)))
        sd.append(float(seg.var()))   # rolling variance (EWS-canonical; matches decline_ews_si + the "variance" panel label)
        pos.append((i - 1) / (n - 1))
    t_inf = float(np.argmax(np.gradient(trend)) / (n - 1))
    return np.array(pos), np.array(ac), np.array(sd), t_inf


def to_grid(pos, y):
    if len(pos) < 4: return None
    g = np.interp(GRID, pos, y, left=np.nan, right=np.nan)
    return g


def main():
    data = collect()
    groups = ["Historical", "Renewables", "BEV", "CDR"]
    res = {g: {"ac": [], "sd": [], "tau_ac": [], "tau_sd": []} for g in groups}
    for g, nm, yy, vv in data:
        if len(vv) < MIN_N: continue
        pos, ac, sd, t_inf = ews_series(vv)
        if len(pos) < 4: continue
        ga, gs = to_grid(pos, ac), to_grid(pos, sd)
        if ga is None: continue
        res[g]["ac"].append(ga); res[g]["sd"].append(gs)
        pre = pos <= max(t_inf, pos[0] + 1e-9)          # pre-takeoff segment
        if pre.sum() >= 4:
            res[g]["tau_ac"].append(kendalltau(pos[pre], ac[pre])[0])
            res[g]["tau_sd"].append(kendalltau(pos[pre], sd[pre])[0])

    print(f"{'group':12s} {'N':>4} {'%rising AC1 (pre-takeoff)':>26} {'%rising var':>12}")
    stats = {}
    for g in groups:
        ta, ts = np.array(res[g]["tau_ac"]), np.array(res[g]["tau_sd"])
        pa = 100 * (ta > 0).mean() if len(ta) else np.nan
        ps = 100 * (ts > 0).mean() if len(ts) else np.nan
        stats[g] = (len(res[g]["ac"]), pa, ps, len(ta))
        print(f"{g:12s} {len(res[g]['ac']):4d} {pa:26.0f} {ps:12.0f}   (pre-takeoff n={len(ta)})")

    fig, axs = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for j, g in enumerate(groups):
        for row, key, lab in [(0, "ac", "rolling lag-1 autocorr (residuals)"),
                              (1, "sd", "rolling std (residuals)")]:
            ax = axs[row, j]
            A = np.array(res[g][key])
            if len(A) == 0: continue
            lw_thin = 0.5 if len(A) > 50 else 1.0
            for a in A:
                ax.plot(GRID, a, color=GCOL[g], alpha=min(.25, 18/max(len(A),1)), lw=lw_thin)
            med = np.nanmedian(A, 0)
            q1, q3 = np.nanpercentile(A, 25, 0), np.nanpercentile(A, 75, 0)
            ax.plot(GRID, med, color="k", lw=2.6, label="median")
            ax.fill_between(GRID, q1, q3, color="k", alpha=.12, label="IQR")
            ax.grid(alpha=.25)
            if row == 0:
                n, pa, ps, npre = stats[g]
                ax.set_title(f"{g}  (N={n})\nrising pre-takeoff: AC1 {pa:.0f}%, var {ps:.0f}%",
                             fontweight="bold", color=GCOL[g], fontsize=10)
                ax.set_ylim(-1.05, 1.05); ax.axhline(0, color="grey", lw=.6)
            else:
                ax.set_xlabel("normalised lifecycle time")
            if j == 0: ax.set_ylabel(lab)
            if j == 0 and row == 0: ax.legend(fontsize=8, loc="lower right")
            if g == "CDR" and row == 0:
                ax.text(.5, .5, "CDR = cumulative project\npipeline — too smooth/few\nfor residual EWS",
                        transform=ax.transAxes, ha="center", fontsize=9, color="#fb8500",
                        fontweight="bold", bbox=dict(fc="white", ec="#fb8500", alpha=.9))
    fig.suptitle("Classical early-warning signals across the four groups — rising residual "
                 "autocorrelation & variance on approach to takeoff\n(the generic signature of an "
                 "underlying bifurcation — agnostic to its TYPE, which is not curve-readable; §90)",
                 fontweight="bold", fontsize=12.5)
    plt.tight_layout()
    fig.savefig("results/figures/fig_ews_groups.png", dpi=150, bbox_inches="tight")
    print("Saved → results/figures/fig_ews_groups.png")


if __name__ == "__main__":
    main()
