#!/usr/bin/env python3
"""
Minimal SI figure: the rising-variance early-warning signature is DIRECTION-AGNOSTIC.

Out-of-scope robustness check (the main text curates to the birth phase and excludes post-peak decline).
On SEPARATE raw-annual data, formerly dominant technologies that were adopted for decades and then
declined as they were displaced -- fixed-telephone/landline (by mobile), telegraph traffic (by
telephone/internet), postal traffic (by email), railroad route length (by road/air) -- show residual
variance RISING as they approach their turnover, above a matched non-bifurcating null. Population-level,
not a per-series detector (consistent with the detection limit in the preceding subsection). These are
NOT failed technologies that never took off; typing the decline is out of scope.

Method matches main-text Fig 2 (residual rolling-variance Kendall-tau) but aligned to the turnover
instead of the takeoff. Writes figures/si/figS_decline_ews.png.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
from paper_style import set_style, COL2
set_style()
# Prefer the shipped decline subset (the 4 obsolescence mechanisms); fall back to the full raw master.
DECLINE = "data/curated/failed_tech/decline_series_raw.csv"
MASTER = "data/curated/master/adoption_master_long.csv"
DATA = DECLINE if os.path.exists(DECLINE) else MASTER
OUT = "figures/si/figS_decline_ews.png"
GENUINE = {"Fixed telephone": "landline", "Telegraph Traffic": "telegraph",
           "Postal Traffic": "postal", "Railroad": "railroad"}
BROWN, GREY = "#7f5539", "#adb5bd"


def approach(v, rng, nnull=299, want_series=False):
    """Residual rolling-variance over the mature approach-to-peak window; real tau, null taus, (series)."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]; n = len(v)
    if n < 18:
        return None
    norm = (v - v.min()) / (v.max() - v.min() + 1e-12)
    sig = max(2, n // 8)
    trend = gaussian_filter1d(norm, sigma=sig)
    tpk = int(np.argmax(trend)); tpeak = trend[tpk]
    if tpk < 6:
        return None
    up = np.where(trend[:tpk + 1] >= 0.5 * tpeak)[0]
    t50 = int(up[0]) if len(up) else 0
    if tpk - t50 < 6:
        t50 = max(0, tpk - 8)
    res = norm - trend
    w = max(5, (tpk - t50) // 2)

    def var_curve(r):
        pos, sd = [], []
        for i in range(w, tpk + 1):
            seg = r[i - w:i]
            if seg.std() < 1e-10:
                continue
            c = i - 1
            if c < t50:
                continue
            sd.append(seg.var()); pos.append(c)
        return np.array(pos), np.array(sd)

    pos, sd = var_curve(res)
    if len(pos) < 5:
        return None
    t_real = kendalltau(pos, sd)[0]
    if not np.isfinite(t_real):
        return None
    s = res.std()
    phi = np.corrcoef(res[:-1], res[1:])[0, 1] if n > 5 else 0.0
    phi = float(np.clip(phi, 0.0, 0.95)) if np.isfinite(phi) else 0.0
    t_null = []
    for _ in range(nnull):
        eta = np.zeros(n)
        for i in range(1, n):
            eta[i] = phi * eta[i - 1] + rng.standard_normal()
        sur = trend + eta / (eta.std() + 1e-9) * s
        p2, s2 = var_curve(sur - gaussian_filter1d(sur, sigma=sig))
        if len(p2) >= 5:
            tn = kendalltau(p2, s2)[0]
            if np.isfinite(tn):
                t_null.append(tn)
    out = dict(tau=t_real, nulls=t_null, n=n)
    if want_series:
        out.update(norm=norm, trend=trend, tpk=tpk, pos=pos, sd=sd, t50=t50)
    return out


def full_var(v, sig_frac=8):
    """Rolling residual variance across the WHOLE series (illustrative), + normalized curve, trend, peak."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]; n = len(v)
    norm = (v - v.min()) / (v.max() - v.min() + 1e-12)
    sig = max(2, n // sig_frac)
    trend = gaussian_filter1d(norm, sigma=sig)
    tpk = int(np.argmax(trend))
    res = norm - trend
    w = max(5, n // 8)
    pos, var = [], []
    for i in range(w, n + 1):
        seg = res[i - w:i]
        pos.append(i - 1); var.append(seg.var())
    return norm, trend, tpk, np.array(pos), np.array(var)


def main():
    m = pd.read_csv(DATA, low_memory=False)
    rng = np.random.default_rng(0)
    real, nullpool, cand = [], [], []
    for sid, g in m.groupby("series_id"):
        grp = GENUINE.get(g.tech.iloc[0])
        if grp is None:
            continue
        g = g.sort_values("year"); v = g.value.to_numpy(float); ok = np.isfinite(v)
        v = v[ok]; yr = g.year.to_numpy(int)[ok]
        if len(v) < 18:
            continue
        pk = int(v.argmax()); peak = v[pk]
        if peak <= 0 or pk >= len(v) - 3 or pk < 3 or v[-1] >= 0.75 * peak:
            continue
        r = approach(v, rng, want_series=True)
        if r is None:
            continue
        real.append(r["tau"]); nullpool += r["nulls"]
        window = r["tpk"] - r["t50"]
        cand.append((grp, g.tech.iloc[0], g.country.iloc[0], v, yr, r["tau"], len(v), peak, window))
    real = np.array(real); nullpool = np.array(nullpool)
    print(f"pooled declining series n={len(real)} | %tau>0={100*np.mean(real>0):.0f}% | "
          f"median real tau={np.median(real):+.2f} | median null tau={np.median(nullpool):+.2f}")

    # exemplar: long, gradual, mainstream-penetration landline collapse (clear run-up + decline tail)
    ll = [c for c in cand if c[0] == "landline" and c[6] >= 55 and c[5] > 0.6
          and 30 <= c[7] <= 90 and c[8] >= 18]
    ll.sort(key=lambda c: (c[8], c[5]), reverse=True)          # longest approach window, then strongest rise
    print("top landline exemplar candidates (country | n | tau | peak | window):")
    for c in ll[:10]:
        print(f"   {c[2]:16s} n={c[6]} tau={c[5]:+.2f} peak={c[7]:.0f} window={c[8]}")
    PREF = ["Korea, Rep.", "Ireland", "Denmark", "Netherlands", "France", "Germany"]
    bycty = {c[2]: c for c in ll}
    ex = next((bycty[p] for p in PREF if p in bycty), ll[0])
    cty = "South Korea" if ex[2] == "Korea, Rep." else ex[2]
    print("exemplar:", ex[2], "landline | n=", ex[6], "tau=", round(ex[5], 2),
          "peak=", round(ex[7], 1), "window=", ex[8])
    exd = approach(ex[3], np.random.default_rng(1), want_series=True)   # analysis variance (approach window)
    yr = ex[4]; tpk = exd["tpk"]

    fig, ax = plt.subplots(1, 2, figsize=(5.9, 2.45), gridspec_kw=dict(wspace=0.44))
    # (a) exemplar: adoption rise-and-fall + the SAME rolling variance that enters tau, up to the turnover
    A = ax[0]
    A.plot(yr, exd["norm"], color=GREY, lw=1.1, label="adoption")
    A.axvline(yr[tpk], color="#aaa", lw=0.8, ls=":")
    A.text(yr[tpk] + 0.6, 0.04, "turnover", fontsize=5.6, color="#999", rotation=90, va="bottom")
    A.set_ylabel("adoption (min-max)", fontsize=7, color="#555")
    A.set_xlabel("year", fontsize=7); A.set_ylim(-0.03, 1.08)
    A2 = A.twinx()
    A2.plot(yr[exd["pos"]], exd["sd"], color=BROWN, lw=1.8, label="rolling variance")
    A2.set_ylabel("residual variance", fontsize=7, color=BROWN)
    A2.tick_params(axis="y", colors=BROWN, labelsize=6); A2.set_ylim(0, None)
    A.tick_params(labelsize=6)
    A.set_title("(a)", fontsize=8.5, loc="left", fontweight="bold")
    # (b) pooled real-vs-null variance-trend tau
    B = ax[1]
    bins = np.linspace(-1, 1, 22)
    B.hist(nullpool, bins=bins, density=True, color=GREY, alpha=0.55, label=f"matched null")
    B.hist(real, bins=bins, density=True, histtype="step", color=BROWN, lw=1.8,
           label=f"declining tech (n={len(real)})")
    B.axvline(np.median(nullpool), color="#666", lw=1.0, ls=":")
    B.axvline(np.median(real), color=BROWN, lw=1.2, ls="--")
    B.set_xlabel("approach-to-turnover variance trend $\\tau$", fontsize=7)
    B.set_ylabel("density", fontsize=7); B.tick_params(labelsize=6)
    B.legend(fontsize=6.4, frameon=False, loc="upper left")
    B.set_title("(b)", fontsize=8.5, loc="left", fontweight="bold")
    fig.savefig(OUT, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
