#!/usr/bin/env python3
"""
Diagnostic: (1) how our takeoff time (t_inf) is defined and whether it is robust; (2) cross-check our
EWS pipeline (ews_groups.ews_series) against Thomas Bury's ewstools library.

Takeoff = inflection = argmax of the gradient of the Gaussian-smoothed, min-max-normalized series,
in normalized lifecycle units [0,1]. Each series is aligned to its own t_inf (standard EWS practice).
Pre-takeoff Kendall-tau is computed over rolling-window EWS points with pos <= t_inf.

Writes a diagnostic PNG (runs/takeoff_ewstools.png) + a console summary.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, ewstools
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau
from ews_groups import ews_series
from growth_compare import collect
from paper_style import set_style, COL2
from paper_figures import _tag, FAMCOL, FAMLABEL
set_style()
FAMS = ["Historical", "Renewables", "BEV"]
SCR = "runs"   # diagnostic PNG output (repo-relative)


def ours(v):
    pos, ac, sd, ti = ews_series(v)
    pre = pos <= max(ti, pos[0] + 1e-9); npre = int(pre.sum())
    if npre < 4:
        return ti, npre, np.nan, np.nan
    return ti, npre, kendalltau(pos[pre], sd[pre])[0], kendalltau(pos[pre], ac[pre])[0]


def ewst(v, ti, bw=0.2, rw=0.25):
    vn = (v - v.min()) / (v.max() - v.min() + 1e-12); n = len(vn)
    ts = ewstools.TimeSeries(pd.Series(vn, index=np.arange(n)), transition=round(ti * (n - 1)))
    ts.detrend(method="Gaussian", bandwidth=bw)
    ts.compute_var(rolling_window=rw); ts.compute_auto(rolling_window=rw, lag=1)
    ts.compute_ktau()
    return ts.ktau.get("variance", np.nan), ts.ktau.get("ac1", np.nan)


def gather():
    R = {f: dict(tinf=[], npre=[], tv=[], ta=[], evd=[], ead=[]) for f in FAMS}
    for g, nm, yy, vv in collect():
        if g not in FAMS:
            continue
        v = np.asarray(vv, float)
        if len(v) < 12 or v.max() - v.min() < 1e-9:
            continue
        ti, npre, tv, ta = ours(v)
        if not np.isfinite(tv):
            continue
        evd, ead = ewst(v, ti)
        for k, val in [("tinf", ti), ("npre", npre), ("tv", tv), ("ta", ta), ("evd", evd), ("ead", ead)]:
            R[g][k].append(val)
    return {f: {k: np.array(val) for k, val in R[f].items()} for f in FAMS}


def main():
    R = gather()
    print(f"{'family':11s} {'n':>4} {'t_inf med[min,max]':>20} {'npre med[min,max]':>18} {'%|tau_v|=1':>10}")
    for f in FAMS:
        ti, npre, tv = R[f]["tinf"], R[f]["npre"], R[f]["tv"]
        print(f"{f:11s} {len(ti):>4} {np.median(ti):.2f}[{ti.min():.2f},{ti.max():.2f}]      "
              f"{int(np.median(npre)):>3}[{npre.min()},{npre.max()}]       {100*np.mean(np.abs(tv)>=0.999):5.0f}%")
    tv = np.concatenate([R[f]["tv"] for f in FAMS]); evd = np.concatenate([R[f]["evd"] for f in FAMS])
    ta = np.concatenate([R[f]["ta"] for f in FAMS]); ead = np.concatenate([R[f]["ead"] for f in FAMS])
    mv = np.isfinite(tv) & np.isfinite(evd); ma = np.isfinite(ta) & np.isfinite(ead)
    print(f"\nvariance tau  ours vs ewstools: r={np.corrcoef(tv[mv],evd[mv])[0,1]:.2f}; "
          f"median ours {np.median(tv[mv]):+.2f} / ewstools {np.median(evd[mv]):+.2f}; "
          f"%>0 ours {100*np.mean(tv[mv]>0):.0f} / ewstools {100*np.mean(evd[mv]>0):.0f}")
    print(f"AC1 tau       ours vs ewstools: r={np.corrcoef(ta[ma],ead[ma])[0,1]:.2f}; "
          f"median ours {np.median(ta[ma]):+.2f} / ewstools {np.median(ead[ma]):+.2f}; "
          f"%>0 ours {100*np.mean(ta[ma]>0):.0f} / ewstools {100*np.mean(ead[ma]>0):.0f}")

    fig, ax = plt.subplots(2, 2, figsize=(COL2, 5.4), gridspec_kw=dict(hspace=0.42, wspace=0.3))
    # (a) t_inf distribution
    A = ax[0, 0]
    for f in FAMS:
        A.hist(R[f]["tinf"], bins=np.linspace(0.3, 0.95, 20), color=FAMCOL[f], alpha=0.55, label=FAMLABEL[f])
    A.set_xlabel("takeoff time $t_{\\mathrm{inf}}$ (normalized lifecycle)", fontsize=7)
    A.set_ylabel("count", fontsize=7); A.legend(fontsize=5.6, frameon=False)
    _tag(A, "(a) takeoff is late and variable, not pathological")
    # (b) npre distribution
    B = ax[0, 1]
    allnp = np.concatenate([R[f]["npre"] for f in FAMS])
    B.hist(allnp, bins=np.arange(3, 60, 3), color="#6a4c93", alpha=0.8)
    B.axvspan(3, 8, color="#d62828", alpha=0.13)
    B.text(8.5, B.get_ylim()[1] * 0.7, "few points\n$\\to$ unstable $\\tau$\n($\\sim$9% give $|\\tau|{=}1$)",
           fontsize=5.4, color="#d62828")
    B.set_xlabel("pre-takeoff rolling-window points", fontsize=7); B.set_ylabel("count", fontsize=7)
    _tag(B, "(b) many series have few pre-takeoff points")
    # (c) variance tau ours vs ewstools
    C = ax[1, 0]
    for f in FAMS:
        C.scatter(R[f]["evd"], R[f]["tv"], s=7, color=FAMCOL[f], alpha=0.5, lw=0)
    C.plot([-1, 1], [-1, 1], color="#888", lw=0.8, ls="--")
    C.set_xlim(-1.1, 1.1); C.set_ylim(-1.1, 1.1)
    C.set_xlabel("ewstools variance $\\tau$", fontsize=7); C.set_ylabel("our variance $\\tau$", fontsize=7)
    C.text(0.04, 0.9, f"r={np.corrcoef(tv[mv],evd[mv])[0,1]:.2f}\nmedians agree\n(+0.5 vs +0.5)",
           transform=C.transAxes, fontsize=5.6, color="#333")
    _tag(C, "(c) variance: ours agrees with ewstools")
    # (d) AC1 tau ours vs ewstools
    D = ax[1, 1]
    for f in FAMS:
        D.scatter(R[f]["ead"], R[f]["ta"], s=7, color=FAMCOL[f], alpha=0.5, lw=0)
    D.plot([-1, 1], [-1, 1], color="#888", lw=0.8, ls="--")
    D.set_xlim(-1.1, 1.1); D.set_ylim(-1.1, 1.1)
    D.set_xlabel("ewstools AC1 $\\tau$", fontsize=7); D.set_ylabel("our AC1 $\\tau$", fontsize=7)
    D.text(0.04, 0.86, f"r={np.corrcoef(ta[ma],ead[ma])[0,1]:.2f}\nours biased LOW\n(ours $-0.1$, ewstools $+0.1$)",
           transform=D.transAxes, fontsize=5.6, color="#d62828")
    _tag(D, "(d) AC1: our estimator is biased negative")
    fig.savefig(f"{SCR}/takeoff_ewstools.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved -> {SCR}/takeoff_ewstools.png")


if __name__ == "__main__":
    main()
