#!/usr/bin/env python3
"""
paper_figures.py
================
Publication MAIN-text figures (PNAS), built on paper_style.py.
  Fig 1  fig1_features.png   feature-space exemplars (adoption=blue, guides=grey), named panels
  Fig 2  fig2_ews_groups.png 3x4 per-family median + 10-90% band; adoption shown at REAL maturity
                              (Historical ~saturated; Renewables/BEV in progress; CDR realized solid
                              + pledged dashed)
  Fig 3  fig3_continuum.png  2x2: continuum scatter + curves at PC1/PC2/PC3 extremes (pooled, all
                              four families)
  Fig 4  fig4_growth.png     2x2: CDR growth vs precedent + the two most family-discriminating PCs
SI helpers: fig2_si -> figS_ews_examples.png ; fig4_si -> figS_discriminating.png
No titles (except the requested named panel tags), no grid, Times New Roman, no em-dashes.
Outputs -> figures/main/
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from paper_style import set_style, panel_label, COL2
from growth_compare import collect
from compare_4groups import clean
from ews_groups import ews_series, GRID, to_grid
from unsup_theory_features import extract_theory_features, FEATURE_NAMES, TCFP_NAMES
from critical_scaling_features import extract_critical_features, CRIT_NAMES

set_style()
OUT = "figures/main"; os.makedirs(OUT, exist_ok=True)

# 51 raw features (38 theory + 5 TC-fingerprint + 8 critical), then DROP 5 near-duplicates (|r|>0.93 with
# another feature) -> 46-D cleaned canonical space.
_GROUP_FULL = (["A:CSD"]*8 + ["B:Inflect"]*7 + ["C:Phase"]*6 + ["D:Catch22"]*7 + ["E:SN-fp"]*5 + ["F:Transit"]*5
               + ["G:TC-fp"]*5 + ["H:Crit"]*8)
_ALL_FULL = list(FEATURE_NAMES) + list(TCFP_NAMES) + list(CRIT_NAMES)
DROP_FEATURES = {"peak_to_rms", "velocity_kurtosis", "center_fraction",   # base-38 duplicates (D, E, F)
                 "percap_linearity", "fold_dwell"}                        # critical duplicates of G, E
KEEP_IDX = [i for i, n in enumerate(_ALL_FULL) if n not in DROP_FEATURES]
GROUP = [_GROUP_FULL[i] for i in KEEP_IDX]
ALL_NAMES = [_ALL_FULL[i] for i in KEEP_IDX]              # 46 cleaned feature names
GCOLOR = {"A:CSD": "#E63946", "B:Inflect": "#F4A261", "C:Phase": "#2A9D8F",
          "D:Catch22": "#8AB17D", "E:SN-fp": "#E76F51", "F:Transit": "#457B9D",
          "G:TC-fp": "#1d4e89", "H:Crit": "#6a4c93"}
FAMCOL = {"Historical": "#7f5539", "Renewables": "#2a9d8f", "BEV": "#6a4c93", "CDR": "#fb8500"}
FAMS = ["Historical", "Renewables", "BEV", "CDR"]
FAMLABEL = {"Historical": "Historical", "Renewables": "Solar+Wind", "BEV": "BEV", "CDR": "CDR"}
PCDESC = {0: "noise / sharpness", 1: "back-loadedness", 2: "step-ness"}   # 51-D loadings (b)
ADOPT = "#1f6fb2"; AUX = "#9aa0a6"
LOWC, HIC = "#3a86ff", "#e63946"
INK = "#222222"


def _tag(ax, s, y=1.05, size=7.3):
    ax.text(0.0, y, s, transform=ax.transAxes, fontsize=size, fontweight="bold", ha="left",
            va="bottom", color=INK)


# ===================================================================== FIG 1
def fig1():
    rng = np.random.default_rng(3); T = 200; t = np.linspace(0, 1, T)
    logi = 1 / (1 + np.exp(-12 * (t - 0.45)))
    noisy = np.clip(logi + 0.026 * rng.standard_normal(T) * (1 + 1.3*t), 0, 1.18)
    sn = np.clip(0.02 + 1/(1 + np.exp(-35*(t-0.62))), 0, 1)
    null = 0.5 + 0.06 * rng.standard_normal(T)
    for i in range(1, T): null[i] = 0.6*null[i-1] + 0.4*null[i]

    fig, axs = plt.subplots(2, 4, figsize=(COL2, 4.3))
    tags = ["(a) Critical slowing-down", "(b) Inflection geometry", "(c) Phase portrait",
            "(d) Generic dynamics", "(e) Saddle-node fingerprint", "(f) Transition vs null",
            "(g) Transcritical fingerprint", "(h) Critical scaling"]

    a = axs[0, 0]; a.plot(t, noisy, color=ADOPT, lw=1.2)
    w = T // 5; rv = np.array([noisy[max(0, i-w):i+1].std() for i in range(T)])
    a2 = a.twinx(); a2.plot(t, rv, color=AUX, lw=1.4, ls="--"); a2.set_yticks([])
    for s in a2.spines.values(): s.set_visible(False)

    b = axs[0, 1]; b.plot(t, logi, color=ADOPT, lw=1.6)
    b.axvline(t[np.argmax(np.gradient(logi, t))], color=AUX, lw=1.0, ls="--")

    c = axs[0, 2]; tau = 12; ph = np.clip(logi + 0.02*rng.standard_normal(T), 0, 1.1)
    c.plot(ph[tau:], ph[:-tau], color=ADOPT, lw=1.0)
    c.fill(ph[tau:], ph[:-tau], color=ADOPT, alpha=0.12)
    c.plot([0, 1], [0, 1], ls=":", color=AUX, lw=0.9)
    c.set_xlabel("x(t)"); c.set_ylabel(r"$x(t-\tau)$")

    d = axs[0, 3]; d.plot(t, noisy, color=ADOPT, lw=1.2)
    d.plot(t, np.polyval(np.polyfit(t, noisy, 1), t), color=AUX, lw=1.3, ls="--")
    d.plot(t, np.polyval(np.polyfit(t, noisy, 3), t), color=AUX, lw=1.3, ls=":")

    e = axs[1, 0]; e.plot(t, sn, color=ADOPT, lw=1.6); e.axhline(0.15, color=AUX, lw=0.9, ls=":")
    e.fill_between(t, 0, 0.15, where=sn < 0.15, color=AUX, alpha=0.18)
    vsn = np.gradient(sn, t); e2 = e.twinx(); e2.plot(t, vsn / vsn.max(), color=AUX, lw=1.2, ls="--")
    e2.set_yticks([]); [s.set_visible(False) for s in e2.spines.values()]   # late (back-loaded) velocity peak

    f = axs[1, 1]; f.plot(t, logi, color=ADOPT, lw=1.6)
    f.plot(t, null, color=AUX, lw=1.3, ls="--")
    f.axhspan(0.3, 0.7, color=AUX, alpha=0.14)

    # (g) transcritical fingerprint: logistic with a SYMMETRIC velocity peak at the midpoint (vs SN's late peak)
    g = axs[1, 2]; g.plot(t, logi, color=ADOPT, lw=1.6)
    vlo = np.gradient(logi, t); g2 = g.twinx(); g2.plot(t, vlo / vlo.max(), color=AUX, lw=1.2, ls="--")
    g2.set_yticks([]); [s.set_visible(False) for s in g2.spines.values()]
    g.axvline(0.45, color=AUX, lw=0.9, ls=":")

    # (h) critical scaling: velocity field dx/dt vs x -- logistic is a symmetric parabola (TC), saddle node is
    # back-loaded / skewed (peak near saturation)
    h = axs[1, 3]; xs = np.linspace(0, 1, 100)
    h.plot(xs, 4 * xs * (1 - xs), color=ADOPT, lw=1.6)                       # logistic velocity field (TC)
    h.plot(xs, 1.9 * xs ** 2 * (1.05 - xs), color=AUX, lw=1.3, ls="--")      # back-loaded (SN-like)
    h.set_xlabel("x"); h.set_ylabel(r"$\dot x$ (velocity field)", fontsize=7)

    special = (c, h)
    for k, ax in enumerate(axs.flat):
        ax.set_xlim(0, 1); ax.tick_params(labelsize=6.5)
        _tag(ax, tags[k])
        if ax not in special:
            ax.set_ylim(-0.03, 1.18); ax.set_yticks([0, 0.5, 1.0])
            ax.set_ylabel("normalized adoption", fontsize=7.5)
        ax.set_xticks([0, 0.5, 1.0])
        if ax not in special: ax.set_xlabel("normalized time", fontsize=7.5)
    fig.tight_layout()
    fig.savefig("figures/si/figS_feature_space.png"); plt.close(fig)
    print("saved figS_feature_space.png (8 groups; moved to SI)")


# ===================================================================== FIG 1 (concept: question + data)
def fig1_concept():
    """Main Fig 1 (2-panel): (a) the hook -- the two mechanisms produce the same observable S-curve,
    with many real adoption curves (faded grey) falling between them and small SCHEMATIC
    equilibrium-diagram insets (the control parameter mu is latent and never observed); (b) validation
    on synthetic ground truth -- the feature space separates well-defined SN/TC/null classes (SN/TC
    labels are legitimate ONLY for synthetic data, where ground truth is known; real series are never
    assigned a type). NOTE: no min-max-normalised family overview here -- that visual falsely shows
    in-progress families (BEV, CDR) reaching 1; real maturity is Fig 2(a-c)."""
    SNCOL, TCCOL = "#d62828", "#457b9d"          # matches the SI benchmark class colours
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 3.1), gridspec_kw=dict(wspace=0.28))
    grp, F, X = load_four_group()
    t100 = np.linspace(0, 1, X.shape[1])
    rng = np.random.default_rng(1)

    # ---- (a) same curve, different mechanism (+ schematic equilibrium-diagram insets) ----
    b = axs[0]; T = 700
    def integ(f, x0, dt=0.01):
        x = np.empty(T); x[0] = x0
        for i in range(1, T): x[i] = x[i-1] + f(x[i-1]) * dt
        return x
    tc = integ(lambda x: 1.3 * x * (1 - x), 0.02)                    # logistic (TC)
    sn = integ(lambda x: 0.010 + 9.0 * x**2 * (1 - x), 0.002)        # ghost bottleneck (SN)
    tt = np.linspace(0, 1, T)
    hidx = np.where(grp == "Historical")[0]
    tv = np.array([np.abs(np.diff(X[j], 2)).sum() for j in hidx])
    tinf = np.array([t100[int(np.argmax(np.gradient(np.maximum.accumulate(X[j]), t100)))] for j in hidx])
    # many faded real S-curves between the two mechanisms (smooth, mid-inflection, so the
    # lower-right inset zone stays clear of curves)
    cand = hidx[(tv < np.percentile(tv, 35)) & (tinf > 0.30) & (tinf < 0.72)]
    show = rng.choice(cand, min(40, len(cand)), replace=False)
    for j in show:
        b.plot(t100, X[j], color="#b7b7b7", lw=0.6, alpha=0.35)
    b.plot(tt, (tc - tc.min()) / np.ptp(tc), color=TCCOL, lw=2.2, label="transcritical")
    b.plot(tt, (sn - sn.min()) / np.ptp(sn), color=SNCOL, lw=2.2, label="saddle-node")
    b.plot([], [], color="#b7b7b7", lw=0.8, label=f"real adoption (n={len(show)})")
    b.set_xlabel("normalized time", fontsize=7.5); b.set_ylabel("normalized adoption", fontsize=7.5)
    b.set_xticks([0, 0.5, 1]); b.set_yticks([0, 0.5, 1]); b.tick_params(labelsize=6.5)
    b.legend(fontsize=5.2, loc="upper left")
    _tag(b, "(a) same curve, different mechanism")
    # schematic insets (x* vs latent mu; axes deliberately unlabelled -- mu is not observed).
    # Stacked vertically in the lower-right corner, clear of the curves (TC/SN are saturated >0.95
    # for x > 0.8, and the grey exemplars stay above y ~ 0.65 there).
    iTC = b.inset_axes([0.815, 0.38, 0.175, 0.22]); iSN = b.inset_axes([0.815, 0.09, 0.175, 0.22])
    mu = np.linspace(-1, 1, 100)
    iTC.plot(mu[mu <= 0], np.zeros((mu <= 0).sum()), color=TCCOL, lw=1.1)
    iTC.plot(mu[mu > 0], np.zeros((mu > 0).sum()), color=TCCOL, lw=0.8, ls="--")
    iTC.plot(mu[mu > 0], mu[mu > 0], color=TCCOL, lw=1.1)
    iTC.plot(mu[mu <= 0], mu[mu <= 0] * 0.6, color=TCCOL, lw=0.8, ls="--")
    iTC.set_title("TC", fontsize=5, pad=1, color=TCCOL)
    mu2 = np.linspace(-1, 0, 100)
    iSN.plot(mu2, 0.5 - np.sqrt(-mu2) * 0.5, color=SNCOL, lw=1.1)
    iSN.plot(mu2, 0.5 + np.sqrt(-mu2) * 0.38, color=SNCOL, lw=0.8, ls="--")
    iSN.plot(0, 0.5, "o", ms=2.2, color=SNCOL); iSN.axhline(1.02, color=SNCOL, lw=1.1)
    iSN.set_xlim(-1, 0.45); iSN.set_ylim(-0.75, 1.25)
    iSN.set_title("SN", fontsize=5, pad=1, color=SNCOL)
    for ii in (iTC, iSN):
        ii.set_xticks([]); ii.set_yticks([]); ii.set_facecolor("white")
        for sp in ii.spines.values(): sp.set(lw=0.5, color="#bbbbbb", visible=True)
    b.text(0.9025, 0.005, "$x^*$ vs latent $\\mu$\n(schematic)", transform=b.transAxes,
           fontsize=4.4, ha="center", va="bottom", color="#888888")

    # ---- (b) validation on synthetic ground truth: typing is easy, detection is the hard part ----
    # SN/TC labels are legitimate here ONLY because the data are synthetic (ground truth known);
    # real series are never assigned a type. The scatter shows the decomposition geometrically
    # (SN separated; TC and its stable twin overlapping); the inset summarises the canonical
    # three-class recovery numbers from SI S3 (Student-t unsupervised 70%; supervised FeatMLP
    # 94.3 +/- 0.4% across three benchmark draws x ten seeds, benchmark_dl_proper.json).
    from sklearn.decomposition import PCA as _PCA
    from benchmark_data import load_benchmark, CANONICAL_NULL
    c = axs[1]
    Xs, ys = load_benchmark(CANONICAL_NULL, 600, 0)
    Zs = StandardScaler().fit_transform(np.nan_to_num(build_features(Xs)))
    _pca_s = _PCA(2).fit(Zs); Ps = _pca_s.transform(Zs)
    _pca_sign_fix(_pca_s.components_, Ps)             # stable orientation across machines
    SLAB = {0: "saddle-node", 1: "transcritical", 2: "null"}
    SCOL = {0: SNCOL, 1: TCCOL, 2: "#9aa0a6"}
    for cl in (2, 1, 0):
        m = ys == cl
        c.scatter(Ps[m, 0], Ps[m, 1], s=5, color=SCOL[cl], alpha=0.45, lw=0, label=SLAB[cl])
    # robust view: a handful of extreme outliers otherwise compress the class geometry
    c.set_xlim(np.percentile(Ps[:, 0], 0.5), np.percentile(Ps[:, 0], 99.5))
    c.set_ylim(np.percentile(Ps[:, 1], 0.5), np.percentile(Ps[:, 1], 99.5))
    c.set_xlabel("PC1 (synthetic)", fontsize=7.5); c.set_ylabel("PC2 (synthetic)", fontsize=7.5)
    c.tick_params(labelsize=6.5)
    c.legend(fontsize=5.2, loc="lower left", handletextpad=0.4, borderaxespad=0.4, labelspacing=0.3)
    # inset: canonical three-class recovery summary, unsupervised vs supervised (SI S3)
    ib = c.inset_axes([0.055, 0.645, 0.27, 0.315])
    vals = [70, 94]
    ib.bar([0, 1], vals, 0.6, color=["#8d99ae", "#1d4e89"])
    ib.axhline(100 / 3, color="#999999", lw=0.6, ls=":")
    for i, v in enumerate(vals):
        ib.text(i, v + 4, f"{v}%", ha="center", fontsize=4.9)
    ib.text(0.5, 100 / 3 + 3, "chance", ha="center", fontsize=4.0, color="#999999")
    ib.set_xticks([0, 1]); ib.set_xticklabels(["unsup.", "superv."], fontsize=4.9)
    ib.set_yticks([]); ib.set_ylim(0, 112); ib.tick_params(length=0)
    ib.set_title("3-class recovery", fontsize=4.8, pad=1.5)
    ib.set_facecolor("white")
    for sp in ib.spines.values():
        sp.set(lw=0.5, color="#bbbbbb", visible=True)
    _tag(c, "(b) typing is easy, detection is the hard part")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_concept.png"); plt.close(fig)
    print("saved fig1_concept.png (2-panel: hook+insets / synthetic separability)")


# ===================================================================== FIG 2 (3x4, real maturity)
def _band(ax, A, color, ls="-", lw=1.8):
    A = np.array(A)
    if not len(A): return 0
    ax.fill_between(GRID, np.nanpercentile(A, 10, 0), np.nanpercentile(A, 90, 0), color=color, alpha=0.16, lw=0)
    ax.plot(GRID, np.nanmedian(A, 0), color=color, lw=lw, ls=ls)
    return len(A)


GRID_REL = np.linspace(-0.6, 0.2, 60)   # lifecycle time relative to takeoff (0 = inflection)


def _grid_rel(pos, t_inf, yv):
    xr = pos - t_inf
    if xr[-1] - xr[0] < 1e-6: return None
    return np.interp(GRID_REL, xr, yv, left=np.nan, right=np.nan)


def _bandx(ax, A, x, color, ls="-"):
    A = np.array(A)
    if not len(A): return 0
    ok = np.isfinite(A).sum(0) >= 1
    if len(A) >= 3:                                    # band only when enough series
        ax.fill_between(x[ok], np.nanpercentile(A[:, ok], 10, 0), np.nanpercentile(A[:, ok], 90, 0),
                        color=color, alpha=0.16, lw=0)
    ax.plot(x[ok], np.nanmedian(A[:, ok], 0), color=color, lw=1.8, ls=ls)
    return len(A)


def _adopt_panel(ax, A, color, ls="-"):
    """All individual adoption curves, faded (no median, no band)."""
    A = np.array(A)
    if not len(A): return
    al = float(np.clip(11.0 / len(A), 0.09, 0.55))
    for c in A:
        ax.plot(GRID, c, color=color, alpha=al, lw=0.6, ls=ls)


def fig2():
    raw = []
    for g, nm, yy, vv in collect():
        if g not in FAMS: continue
        v = np.asarray(vv, float)
        if len(v) < 8 or v.max() - v.min() < 1e-9: continue
        raw.append((g, nm, v))
    cdr_max = max((v.max() for g, nm, v in raw if g == "CDR"), default=1.0)

    store = {f: dict(adopt=[], sd=[], ac=[]) for f in FAMS}
    cdr_ad = {"realised": [], "pledged": []}
    cdr_ews = {"sd": [], "ac": []}                              # EWS for the PLEDGED CDR series only
    for g, nm, v in raw:
        tt = np.linspace(0, 1, len(v))
        if g == "Historical":
            ad = (v - v.min()) / (v.max() - v.min())          # assumed saturated -> ~1
        elif g in ("Renewables", "BEV"):
            ad = v / 100.0                                    # real attained share (fraction)
        else:
            ad = v / cdr_max                                  # CDR as fraction of pledged ceiling
        adg = np.interp(GRID, tt, ad)
        if g == "CDR":
            key = "pledged" if "promised" in nm else "realised"
            cdr_ad[key].append(adg)
            if key == "pledged" and len(v) >= 12:              # residual EWS only on the pledged path
                pos, ac, sd, ti = ews_series(v)
                gsd, gac = _grid_rel(pos, ti, sd), _grid_rel(pos, ti, ac)
                if gsd is not None: cdr_ews["sd"].append(gsd)
                if gac is not None: cdr_ews["ac"].append(gac)
        else:
            store[g]["adopt"].append(adg)
            if len(v) >= 12:
                pos, ac, sd, ti = ews_series(v)                # ALIGN to takeoff (inflection)
                gsd, gac = _grid_rel(pos, ti, sd), _grid_rel(pos, ti, ac)
                if gsd is not None: store[g]["sd"].append(gsd)
                if gac is not None: store[g]["ac"].append(gac)

    fig, axs = plt.subplots(3, 4, figsize=(COL2, 4.8))
    rlab = ["adoption level\n(fraction)", "rolling\nvariance", "lag-1\nautocorr."]
    for ci, f in enumerate(FAMS):
        col = FAMCOL[f]; a0 = axs[0, ci]
        if f == "CDR":
            _adopt_panel(a0, cdr_ad["realised"], col, "-")
            _adopt_panel(a0, cdr_ad["pledged"], col, "--")
            a0.text(0.05, 0.92, "pledged (dashed)\nrealized (solid)", transform=a0.transAxes,
                    fontsize=5.2, color="#777777", va="top")
            _bandx(axs[1, ci], cdr_ews["sd"], GRID_REL, col, "--")   # pledged-only EWS
            _bandx(axs[2, ci], cdr_ews["ac"], GRID_REL, col, "--")
            axs[1, ci].text(0.05, 0.92, "pledged", transform=axs[1, ci].transAxes, fontsize=5.2,
                            color="#777777", va="top")
        else:
            _adopt_panel(a0, store[f]["adopt"], col)
            if f == "Historical":
                a0.text(0.95, 0.08, "saturated\n(assumed)", transform=a0.transAxes,
                        fontsize=5.2, color="#777777", ha="right", va="bottom")
            n_ews = len(store[f]["sd"])
            _bandx(axs[1, ci], store[f]["sd"], GRID_REL, col)
            _bandx(axs[2, ci], store[f]["ac"], GRID_REL, col)
            axs[1, ci].text(0.05, 0.92, f"n={n_ews}", transform=axs[1, ci].transAxes, fontsize=5.2,
                            color="#777777", va="top")
        for ri in (1, 2):
            axs[ri, ci].axvline(0, color="#bbbbbb", lw=0.7, ls=":")     # takeoff marker
        axs[0, ci].set_title(FAMLABEL[f], fontsize=8.5, color=col)
        axs[0, ci].set_xlim(0, 1); axs[0, ci].set_xticks([0, 0.5, 1])
        axs[0, ci].set_xlabel("normalized lifecycle", fontsize=7)
        for ri in (1, 2):
            axs[ri, ci].set_xlim(GRID_REL[0], GRID_REL[-1]); axs[ri, ci].set_xticks([-0.5, 0])
        for ri in range(3): axs[ri, ci].tick_params(labelsize=6.5, length=2)
        axs[2, ci].set_xlabel("time to takeoff", fontsize=7)
    # adoption row: 0-1
    for ci in range(4):
        axs[0, ci].set_ylim(0, 1.05)
        if ci > 0: axs[0, ci].set_yticklabels([])
    # rolling-variance row: TIGHT y-limit driven by the GENUINE families only (CDR's smooth pledged
    # curve has a much higher residual variance and would otherwise squash the visible CSD rise).
    var_med = [np.nanmax(np.nanmedian(np.array(store[f]["sd"]), 0))
               for f in ("Historical", "Renewables", "BEV") if store[f]["sd"]]
    vmax = (max(var_med) * 1.2) if var_med else 0.1
    for ci in range(3):                                    # genuine families: shared tight scale
        axs[1, ci].set_ylim(-0.004, vmax)
        if ci > 0: axs[1, ci].set_yticklabels([])
    axs[1, 3].set_ylim(bottom=0); axs[1, 3].tick_params(labelleft=True, labelsize=6)   # CDR: own scale
    axs[1, 3].text(0.95, 0.06, "own scale\n(flat: no CSD)", transform=axs[1, 3].transAxes,
                   ha="right", va="bottom", fontsize=5.0, color="#999999")
    # autocorr row: shared
    lo = min(axs[2, ci].get_ylim()[0] for ci in range(4)); hi = max(axs[2, ci].get_ylim()[1] for ci in range(4))
    for ci in range(4):
        axs[2, ci].set_ylim(lo, hi)
        if ci > 0: axs[2, ci].set_yticklabels([])
    for ri in range(3): axs[ri, 0].set_ylabel(rlab[ri], fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_ews_groups.png"); plt.close(fig)
    print(f"saved fig2_ews_groups.png  (var ymax={vmax:.3f}; CDR pledged EWS n={len(cdr_ews['sd'])})")


# ----- Fig 2 VARIANT B: Kendall-tau distribution (sharper, quantitative) -----
def fig2_tau():
    raw, cdr_max = [], 1.0
    for g, nm, yy, vv in collect():
        if g not in FAMS: continue
        v = np.asarray(vv, float)
        if len(v) < 8 or v.max() - v.min() < 1e-9: continue
        raw.append((g, nm, v))
    cdr_max = max((v.max() for g, nm, v in raw if g == "CDR"), default=1.0)
    store = {f: dict(adopt=[], tsd=[], tac=[]) for f in FAMS}
    cdr_ad = {"realised": [], "pledged": []}
    for g, nm, v in raw:
        tt = np.linspace(0, 1, len(v))
        ad = ((v - v.min()) / (v.max() - v.min()) if g == "Historical"
              else v / 100.0 if g in ("Renewables", "BEV") else v / cdr_max)
        adg = np.interp(GRID, tt, ad)
        if g == "CDR":
            cdr_ad["pledged" if "promised" in nm else "realised"].append(adg)
        else:
            store[g]["adopt"].append(adg)
        if len(v) >= 12:
            pos, ac, sd, ti = ews_series(v)
            pre = pos <= max(ti, pos[0] + 1e-9)
            if pre.sum() >= 4:
                ts = kendalltau(pos[pre], sd[pre])[0]; ta = kendalltau(pos[pre], ac[pre])[0]
                if np.isfinite(ts): store[g]["tsd"].append(ts)
                if np.isfinite(ta): store[g]["tac"].append(ta)

    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.9), gridspec_kw=dict(wspace=0.38))
    aA = axs[0]
    for f in FAMS:
        if f == "CDR":
            if cdr_ad["realised"]: aA.plot(GRID, np.nanmedian(cdr_ad["realised"], 0), color=FAMCOL[f], lw=1.8, label="CDR realized")
            if cdr_ad["pledged"]: aA.plot(GRID, np.nanmedian(cdr_ad["pledged"], 0), color=FAMCOL[f], lw=1.8, ls="--", label="CDR pledged")
        elif store[f]["adopt"]:
            aA.plot(GRID, np.nanmedian(store[f]["adopt"], 0), color=FAMCOL[f], lw=1.8, label=FAMLABEL[f])
    aA.set_xlim(0, 1); aA.set_ylim(0, 1.05); aA.set_xticks([0, 0.5, 1]); aA.set_yticks([0, 0.5, 1])
    aA.set_xlabel("normalized lifecycle"); aA.set_ylabel("adoption level (fraction)")
    aA.legend(fontsize=5.6, loc="upper left"); _tag(aA, "(a) maturity")

    rng = np.random.default_rng(0)
    for ax, key, tag in [(axs[1], "tsd", "(b) variance trend"), (axs[2], "tac", "(c) autocorr. trend")]:
        data = [store[f][key] for f in FAMS]
        bp = ax.boxplot(data, positions=range(4), widths=0.6, showfliers=False, patch_artist=True)
        for patch, f in zip(bp["boxes"], FAMS): patch.set(facecolor=FAMCOL[f], alpha=0.30, edgecolor=FAMCOL[f])
        for med in bp["medians"]: med.set(color="#333333", lw=1.2)
        for wsk in bp["whiskers"] + bp["caps"]: wsk.set(color="#888888", lw=0.8)
        for i, f in enumerate(FAMS):
            yv = np.array(store[f][key])
            if len(yv): ax.scatter(rng.normal(i, 0.06, len(yv)), yv, s=5, color=FAMCOL[f], alpha=0.5, lw=0)
        ax.axhline(0, color="#bbbbbb", lw=0.8, ls=":")
        ax.set_ylim(-1.05, 1.05); ax.set_xticks(range(4))
        ax.set_xticklabels([FAMLABEL[f] for f in FAMS], fontsize=6.2, rotation=18, ha="right")
        ax.set_ylabel("pre-takeoff Kendall $\\tau$", fontsize=7.5); _tag(ax, tag)
    fig.savefig(f"{OUT}/fig2_ews_tau.png"); plt.close(fig)
    print("saved fig2_ews_tau.png  (median variance-tau: "
          + ", ".join(f"{FAMLABEL[f]}={np.median(store[f]['tsd']) if store[f]['tsd'] else float('nan'):.2f}" for f in FAMS) + ")")


# ----- Fig 2 COMBINED (main): 3 families time-series + Kendall-tau panels; CDR dropped -----
def _tau_inset(parent, real, null, col):
    """Compact inset: variance-trend Kendall-tau, real (family color) vs matched null (gray)."""
    ins = parent.inset_axes([0.07, 0.49, 0.38, 0.45])
    for arr, x0, fc, ec in [(real, 0, col, col), (null, 1, "#c6c6c6", "#8a8a8a")]:
        if len(arr) < 3:
            continue
        bp = ins.boxplot([arr], positions=[x0], widths=0.55, showfliers=False, patch_artist=True)
        bp["boxes"][0].set(facecolor=fc, alpha=0.55, edgecolor=ec, lw=0.5)
        for m in bp["medians"]: m.set(color="#222", lw=0.9)
        for w in bp["whiskers"] + bp["caps"]: w.set(color="#9a9a9a", lw=0.5)
    ins.axhline(0, color="#bbb", lw=0.6, ls=":")
    ins.set_xlim(-0.65, 1.65); ins.set_ylim(-1.05, 1.05)
    ins.set_xticks([0, 1]); ins.set_xticklabels(["real", "null"], fontsize=4.7)
    ins.set_yticks([-1, 0, 1]); ins.set_yticklabels([-1, 0, 1], fontsize=4.4)
    ins.tick_params(length=1.4, pad=1)
    ins.set_title("var-trend $\\tau$ vs null", fontsize=5.0, pad=1.6)
    for sp in ins.spines.values(): sp.set(lw=0.5, color="#bbb")


def fig2_combined(out=None):
    """Adoption is consistent with a bifurcation: ALL adoption curves + rolling variance (aligned to
    takeoff) + variance-trend Kendall-tau REAL vs a matched non-bifurcating null. AC1 is dropped (a
    biased estimator on short annual series; see the ewstools cross-check), so variance carries the
    early-warning claim, backed by the real-vs-null panel and the likelihood detector of SI Fig. S5."""
    from scipy.ndimage import gaussian_filter1d
    fams3 = ["Historical", "Renewables", "BEV"]
    NULL_M = 25
    store = {f: dict(adopt=[], sd=[], tsd=[], tnull=[]) for f in fams3}
    rng = np.random.default_rng(0)
    for g, nm, yy, vv in collect():
        if g not in fams3: continue
        v = np.asarray(vv, float)
        if len(v) < 8 or v.max() - v.min() < 1e-9: continue
        tt = np.linspace(0, 1, len(v))
        ad = (v - v.min()) / (v.max() - v.min()) if g == "Historical" else v / 100.0
        store[g]["adopt"].append(np.interp(GRID, tt, ad))
        if len(v) >= 12:
            pos, ac, sd, ti = ews_series(v)
            gsd = _grid_rel(pos, ti, sd)
            if gsd is not None: store[g]["sd"].append(gsd)
            pre = pos <= max(ti, pos[0] + 1e-9)
            if pre.sum() >= 4:
                ts = kendalltau(pos[pre], sd[pre])[0]
                if np.isfinite(ts): store[g]["tsd"].append(ts)
                # matched non-bifurcating null: same trend skeleton + stationary AR(1) noise (no
                # critical slowing), pushed through the identical ews_series pipeline
                vn = (v - v.min()) / (v.max() - v.min() + 1e-12); n = len(vn)
                trend = gaussian_filter1d(vn, sigma=max(2, n // 8)); resid = vn - trend
                s = resid.std(); phi = np.corrcoef(resid[:-1], resid[1:])[0, 1] if n > 5 else 0.0
                phi = float(np.clip(phi, 0.0, 0.95)) if np.isfinite(phi) else 0.0
                for _ in range(NULL_M):
                    eta = np.zeros(n)
                    for i in range(1, n): eta[i] = phi * eta[i - 1] + rng.standard_normal()
                    sur = trend + eta / (eta.std() + 1e-9) * s
                    p2, a2, s2, t2 = ews_series(sur)
                    pr2 = p2 <= max(t2, p2[0] + 1e-9)
                    if pr2.sum() >= 4:
                        tn = kendalltau(p2[pr2], s2[pr2])[0]
                        if np.isfinite(tn): store[g]["tnull"].append(tn)

    fig, axs = plt.subplots(2, 3, figsize=(COL2, 3.4), gridspec_kw=dict(hspace=0.6, wspace=0.28))
    var_med = []
    for ci, f in enumerate(fams3):
        col = FAMCOL[f]
        a0 = axs[0, ci]
        _adopt_panel(a0, store[f]["adopt"], col)
        a0.set_xlim(0, 1); a0.set_xticks([0, 0.5, 1]); a0.set_ylim(0, 1.05)
        a0.set_title(FAMLABEL[f], fontsize=8.5, color=col, pad=6)
        a0.set_xlabel("normalized lifecycle", fontsize=7)
        if ci == 0: a0.set_ylabel("adoption level\n(fraction)", fontsize=7.5)
        else: a0.set_yticklabels([])
        a0.tick_params(labelsize=6.5, length=2); _tag(a0, f"({'abc'[ci]})", y=1.08, size=7)
        a1 = axs[1, ci]
        _bandx(a1, store[f]["sd"], GRID_REL, col)
        a1.axvline(0, color="#bbbbbb", lw=0.7, ls=":")
        a1.set_xlim(GRID_REL[0], GRID_REL[-1]); a1.set_xticks([-0.5, 0])
        a1.set_xlabel("time to takeoff", fontsize=7)
        if store[f]["sd"]: var_med.append(np.nanmax(np.nanmedian(np.array(store[f]["sd"]), 0)))
        if ci == 0: a1.set_ylabel("rolling\nvariance", fontsize=7.5)
        else: a1.set_yticklabels([])
        a1.tick_params(labelsize=6.5, length=2); _tag(a1, f"({'def'[ci]})", y=1.08, size=7)
        _tau_inset(a1, np.array(store[f]["tsd"]), np.array(store[f]["tnull"]), col)   # panel g absorbed here
    vmax = (max(var_med) * 1.2) if var_med else 0.1
    for ci in range(3): axs[1, ci].set_ylim(-0.004, vmax)
    out = out or f"{OUT}/fig2_ews_groups.png"
    fig.savefig(out); plt.close(fig)
    print(f"saved {out}  (adoption + rolling variance w/ real-vs-null tau insets; AC1 dropped)")


# ===================================================================== shared loaders
def build_features(X):
    """Canonical 46-D feature matrix: 38 theory + 5 transcritical-fingerprint (G) + 8 critical (H) = 51,
    minus 5 near-duplicate columns (DROP_FEATURES). Single source of truth for the feature space."""
    F = np.hstack([np.nan_to_num(extract_theory_features(X, verbose=False, include_tc_fingerprint=True)),
                   np.nan_to_num(extract_critical_features(X))])
    return F[:, KEEP_IDX]


def load_four_group():
    grp, curves = [], []
    for g, nm, yy, vv in collect():
        c, _ = clean(np.asarray(vv, float), min_n=8 if g == "CDR" else 10)
        if c is None: continue
        grp.append(g); curves.append(c)
    grp = np.array(grp); X = np.stack(curves)
    F = build_features(X)
    return grp, F, X


def four_group_names():
    """Series names aligned 1:1 with the rows of load_four_group()/four_group_pca() (same filter),
    so CDR's three series can be identified for per-series markers."""
    names = []
    for g, nm, yy, vv in collect():
        c, _ = clean(np.asarray(vv, float), min_n=8 if g == "CDR" else 10)
        if c is not None: names.append(nm)
    return np.array(names)


def cdr_marker(nm):
    """(marker, size, color) for a CDR series name: pledge (promised IEA) = smaller red star matching
    the Fig 4b,c pledge star; realized IEA = orange circle; realized SoCDR = orange triangle."""
    if "promised" in nm: return "*", 100, "#d62828"
    if "SoCDR" in nm:    return "^", 42, "#fb8500"
    return "o", 40, "#fb8500"


# Anchor each interpreted PC to a feature whose POSITIVE loading gives the manuscript's labelled
# orientation (SI S4, Fig 3b-d): high PC1 = noisy/sharp, high PC2 = late/back-loaded, high PC3 =
# step-like. Fixes the eigenvector-sign ambiguity so panels/text do not mirror across BLAS/LAPACK.
_PC_SIGN_ANCHORS = ["residual_var_frac", "t_inflection", "bimodality_coeff"]   # PC1, PC2, PC3


def _pca_sign_fix(components, *score_arrays):
    """Deterministic PCA orientation, in place: force the anchor feature's loading positive for each
    labelled PC (and largest-|loading| positive for any further PCs). Flips matching score columns."""
    idx = {n: i for i, n in enumerate(ALL_NAMES)}
    for k in range(components.shape[0]):
        j = idx[_PC_SIGN_ANCHORS[k]] if k < len(_PC_SIGN_ANCHORS) else int(np.argmax(np.abs(components[k])))
        if components[k, j] < 0:
            components[k] *= -1
            for S in score_arrays:
                S[:, k] *= -1
    return components


def four_group_pca():
    grp, F, X = load_four_group()
    ref = np.isin(grp, ["Historical", "Renewables"])
    sc = StandardScaler().fit(F[ref]); pca = PCA().fit(sc.transform(F[ref]))
    P = pca.transform(sc.transform(F))
    _pca_sign_fix(pca.components_, P)                 # stable orientation across machines
    return grp, P, pca.explained_variance_ratio_, pca.components_, X


# ===================================================================== FIG 3 (2x2)
def fig3():
    grp, P, evr, L, X = four_group_pca()
    # k=2 silhouette, winsorised at the 1st/99th percentile so a handful of extreme curves (e.g. Hydro,
    # Steamships) cannot inflate it via a degenerate outlier-vs-bulk split (raw 0.54 -> robust 0.25).
    Pw = np.clip(P[:, :10], np.percentile(P[:, :10], 1, axis=0), np.percentile(P[:, :10], 99, axis=0))
    km = KMeans(2, n_init=10, random_state=0).fit_predict(Pw); sil = silhouette_score(Pw, km)
    t = np.linspace(0, 1, X.shape[1])

    fig, axs = plt.subplots(2, 2, figsize=(COL2, 5.0))
    # panel (a): ridgeline of the PC1/PC2/PC3 distributions (PC1 front -> PC3 back).
    # A single unimodal hump per PC, with no gap, is the direct signature of a continuum.
    axS = axs[0, 0]
    PCcol = {0: "#1f6fb2", 1: "#2a9d8f", 2: "#9b5de5"}
    xs = np.linspace(P[:, :3].min() - 1, P[:, :3].max() + 1, 300)
    dy, dx = 0.55, 1.4
    for i in (2, 1, 0):                                  # back (PC3) drawn first
        kde = gaussian_kde(P[:, i]); d = kde(xs); d = d / d.max()
        axS.fill_between(xs + i*dx, i*dy, d + i*dy, color=PCcol[i], alpha=0.55, lw=0)
        axS.plot(xs + i*dx, d + i*dy, color=PCcol[i], lw=1.2)
        axS.text(xs[0] + i*dx - 0.3, i*dy + 0.10, f"PC{i+1}", fontsize=7.5, ha="right",
                 color=PCcol[i], fontweight="bold")
    axS.set_yticks([]); axS.set_xlabel("principal-component score", fontsize=8)
    axS.set_ylabel("probability density (scaled per PC, offset)", fontsize=7)
    axS.text(0.98, 0.96, f"each PC unimodal:\na continuum\n(k=2 silhouette {sil:.2f})",
             transform=axS.transAxes, va="top", ha="right", fontsize=6.3, color="#555555")
    for sp in ("left", "right", "top"): axS.spines[sp].set_visible(False)
    _tag(axS, "(a) PC distributions (continuum)")

    cells = [(0, 1, "(b) PC1"), (1, 0, "(c) PC2"), (1, 1, "(d) PC3")]
    for k, (r, cc, tag) in enumerate(cells):
        ax = axs[r, cc]; s = P[:, k]
        lo = s <= np.percentile(s, 10); hi = s >= np.percentile(s, 90)
        for idx in np.where(lo)[0][:80]: ax.plot(t, X[idx], color=LOWC, alpha=0.06, lw=0.5)
        for idx in np.where(hi)[0][:80]: ax.plot(t, X[idx], color=HIC, alpha=0.06, lw=0.5)
        ax.plot(t, X[lo].mean(0), color="#1d4ed8", lw=2.0, label="low (bottom 10%)")
        ax.plot(t, X[hi].mean(0), color="#c1121f", lw=2.0, label="high (top 10%)")
        ax.set_ylim(-0.03, 1.03); ax.set_yticks([0, 0.5, 1]); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1])
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("normalized time", fontsize=7); ax.set_ylabel("normalized adoption", fontsize=7)
        _tag(ax, f"{tag} ({evr[k]*100:.0f}%)")
    axs[0, 1].legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_continuum.png"); plt.close(fig)
    print(f"saved fig3_continuum.png  (PC1 {evr[0]*100:.0f}% PC2 {evr[1]*100:.0f}% PC3 {evr[2]*100:.0f}%; "
          f"k=2 sil {sil:.2f}; extreme curves pooled over all {len(grp)} series)")


# ===================================================================== FIG 4 (2x2: growth + PC by family)
def fig4():
    # Option A (1x3): (a) PC3 by family (the one family-discriminating continuum axis; PC1/PC2 demoted
    # to the SI), (b) average pace to peak, (c) peak sustained pace + CDR pledge star.
    R = pd.read_csv("results/unsup/bifurcation_explore/growth_compare.csv")
    cats = ["Historical", "Renewables", "BEV"]
    grp, P, evr, L, X = four_group_pca()
    x = P[:, 2]; gm = x.mean()
    eta3 = sum((grp == f).sum() * (x[grp == f].mean() - gm) ** 2 for f in FAMS) / (((x - gm) ** 2).sum() + 1e-12)

    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.9), gridspec_kw=dict(wspace=0.44))
    rng = np.random.default_rng(0)

    # (a) PC3 by family
    ax = axs[0]
    boxed = [f for f in FAMS if f != "CDR"]            # CDR (n=3) shown points-only, no IQR box
    data = [P[grp == f, 2] for f in boxed]
    bp = ax.boxplot(data, positions=range(len(boxed)), widths=0.6, showfliers=False, patch_artist=True)
    for patch, f in zip(bp["boxes"], boxed): patch.set(facecolor=FAMCOL[f], alpha=0.30, edgecolor=FAMCOL[f])
    for med in bp["medians"]: med.set(color="#333333", lw=1.2)
    for wsk in bp["whiskers"] + bp["caps"]: wsk.set(color="#888888", lw=0.8)
    names = four_group_names()                          # per-series names aligned with grp/P
    for i, f in enumerate(FAMS):
        y = P[grp == f, 2]
        if f == "CDR":                                  # 3 series, distinct markers (explained in caption)
            for yv, nm in zip(y, names[grp == f]):
                mk, sz, col = cdr_marker(nm)
                ax.scatter(i, yv, marker=mk, s=sz, color=col, edgecolor="k", lw=0.5, zorder=6)
        else:
            ax.scatter(rng.normal(i, 0.06, len(y)), y, s=5, color=FAMCOL[f], alpha=0.5, lw=0)
    ax.set_xticks(range(4)); ax.set_xticklabels([FAMLABEL[f] for f in FAMS], fontsize=6.2, rotation=18, ha="right")
    ax.set_ylabel(f"PC3 score ({PCDESC[2]})", fontsize=7)
    print("  fig4a PC3 n per family: " + ", ".join(f"{FAMLABEL[f]}={(grp==f).sum()}" for f in FAMS))
    _tag(ax, f"(a) PC3 by family  ($\\eta^2$={eta3:.2f})")

    # (b, c) average pace to peak and peak sustained pace, each with the CDR pledge star
    def growth_box(ax, col, ylab, tag):
        real_all = R[R.group.isin(cats)][col].dropna().values
        cdr = R[(R.group == "CDR") & R.name.str.contains("promised")][col].max()
        pct = 100 * (real_all < cdr).mean()
        bp = ax.boxplot([R[R.group == g][col].dropna().values for g in cats], positions=range(3),
                        widths=0.55, showfliers=False, patch_artist=True)
        for patch, g in zip(bp["boxes"], cats): patch.set(facecolor=FAMCOL[g], alpha=0.30, edgecolor=FAMCOL[g])
        for med in bp["medians"]: med.set(color="#333333", lw=1.2)
        for wsk in bp["whiskers"] + bp["caps"]: wsk.set(color="#888888", lw=0.8)
        for i, g in enumerate(cats):
            y = R[R.group == g][col].dropna().values; ax.scatter(rng.normal(i, 0.06, len(y)), y, s=5, color=FAMCOL[g], alpha=0.5, lw=0)
        ax.scatter([3], [cdr], marker="*", s=185, color="#d62828", edgecolor="k", lw=0.5, zorder=5)
        ax.annotate(f"CDR pledge\n{cdr:.0f}%/yr\n(> {pct:.0f}% realized)", (3, cdr),
                    xytext=(3, cdr - 10), fontsize=6.0, color="#d62828", ha="center", va="top")
        ax.set_xticks(range(4)); ax.set_xticklabels([FAMLABEL[g] for g in cats] + ["CDR"], fontsize=6.5, rotation=12, ha="right")
        ax.set_ylabel(ylab, fontsize=7.5); ax.set_ylim(0, max(100, cdr * 1.3)); _tag(ax, tag)
        return cdr, pct
    growth_box(axs[1], "geo_growth_pct", "geometric-average growth to peak (%/yr)", "(b) average pace to peak")
    cdr_prom, pct = growth_box(axs[2], "peak_growth_pct", "peak rolling growth rate (%/yr)", "(c) peak sustained pace")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_growth.png"); plt.close(fig)
    print(f"saved fig4_growth.png  (Option A 1x3: PC3 eta={eta3:.2f}; peak CDR {cdr_prom:.0f}%/yr > {pct:.0f}% realized)")


# ----- SI: 10 clearest single-series CSD examples -----
def fig2_si(n_show=10):
    cand = []
    for g, nm, yy, vv in collect():
        if g not in ("Historical", "Renewables", "BEV"): continue
        v = np.asarray(vv, float)
        if len(v) < 14: continue
        pos, ac, sd, t_inf = ews_series(v)
        if len(pos) < 6: continue
        pre = pos <= max(t_inf, pos[0] + 1e-9)
        if pre.sum() < 4: continue
        tsd = kendalltau(pos[pre], sd[pre])[0]; tac = kendalltau(pos[pre], ac[pre])[0]
        if not (np.isfinite(tsd) and np.isfinite(tac)): continue
        cand.append((tsd + tac, tsd, tac, g, nm, v, pos, ac, sd, t_inf))
    cand.sort(key=lambda r: -r[0])
    picked_idx, per_base = [], {}
    for cap in (1, 2):
        for ii, r in enumerate(cand):
            if ii in picked_idx or r[1] <= 0 or r[2] <= 0: continue
            base = r[4].split(":")[0]
            if per_base.get(base, 0) >= cap: continue
            per_base[base] = per_base.get(base, 0) + 1; picked_idx.append(ii)
            if len(picked_idx) == n_show: break
        if len(picked_idx) == n_show: break
    picked = [cand[ii] for ii in picked_idx]

    def shorten(g, nm):
        if g == "Renewables":
            tech, cty = nm.split(":", 1); return f"{tech.split()[0]} ({cty[:7]})"
        return nm[:15]
    nC = len(picked)
    fig, axs = plt.subplots(3, nC, figsize=(COL2, 4.0), sharex=True)
    for j, (_, tsd, tac, g, nm, v, pos, ac, sd, t_inf) in enumerate(picked):
        tt = np.linspace(0, 1, len(v)); col = FAMCOL[g]
        axs[0, j].plot(tt, (v-v.min())/(v.max()-v.min()+1e-12), color=col, lw=1.2)
        axs[1, j].plot(pos, sd, color=col, lw=1.2); axs[2, j].plot(pos, ac, color=col, lw=1.2)
        for a in axs[:, j]:
            a.axvline(t_inf, color="#bbbbbb", lw=0.7, ls="--"); a.set_xlim(0, 1)
            a.set_xticks([0, 1]); a.tick_params(labelsize=6, length=2); a.set_yticks([])
        axs[0, j].set_title(shorten(g, nm), fontsize=6.2, pad=2)
    for i, rl in enumerate(["normalized\nadoption", "rolling\nvariance", "lag-1\nautocorr."]):
        axs[i, 0].set_ylabel(rl, fontsize=7.5)
    present = [g for g in ("Historical", "Renewables", "BEV") if any(p[3] == g for p in picked)]
    fig.legend([plt.Line2D([0], [0], color=FAMCOL[g], lw=2) for g in present], [FAMLABEL[g] for g in present],
               loc="upper center", ncol=len(present), fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"figures/si/figS_ews_examples.png"); plt.close(fig)
    print(f"saved figS_ews_examples.png (n={nC})")


# ----- SI: per-family discriminating-feature heatmap -----
def fig4_si(topk=14):
    grp, F, X = load_four_group()
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    GM = np.vstack([Z[grp == f].mean(0) for f in FAMS])
    eta = np.empty(Z.shape[1])
    for j in range(Z.shape[1]):
        x = Z[:, j]; gm = x.mean()
        ssb = sum((grp == f).sum() * (x[grp == f].mean() - gm) ** 2 for f in FAMS)
        eta[j] = ssb / (((x - gm) ** 2).sum() + 1e-12)
    order = np.argsort(-eta)[:topk]; M = GM[:, order].T
    fig, (axh, axb) = plt.subplots(1, 2, figsize=(COL2, 4.4), gridspec_kw=dict(width_ratios=[3, 1], wspace=0.06))
    im = axh.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1.6, vmax=1.6)
    axh.set_xticks(range(4)); axh.set_xticklabels([FAMLABEL[f] for f in FAMS], fontsize=7.5)
    axh.set_yticks(range(topk)); axh.set_yticklabels([ALL_NAMES[j] for j in order], fontsize=6.5)
    for tick, j in zip(axh.get_yticklabels(), order): tick.set_color(GCOLOR[GROUP[j]])
    axh.tick_params(length=0)
    for sp in axh.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=axh, fraction=0.045, pad=0.02); cb.set_label("group mean (pooled z-score)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    axb.barh(range(topk), eta[order], color="#777777", height=0.7); axb.set_ylim(axh.get_ylim()); axb.set_yticks([])
    axb.set_xlabel(r"$\eta^2$ (variance by family)", fontsize=7.5); axb.tick_params(labelsize=6)
    fig.savefig(f"figures/si/figS_discriminating.png"); plt.close(fig)
    print("saved figS_discriminating.png")


# ----- SI: synthetic benchmark (method validation; canonical TMM / SkewT; NO SN/TC type claims) -----
def fig_benchmark(n_per=1000, seed=0, n_init=3, null_kind=None):
    # SI-ONLY figure: ground truth is known, so SN/TC/Null labels are used here. Canonical null =
    # logistic stable twin (the hardest, dynamics-only case); pass null_kind for the per-null
    # variants (linear / exponential / ramp_expsat / sigmoid / mixed). GMM dropped.
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from unsup_real_world import fit_t_mixture, fit_skew_t_mixture, fit_theory_bayes_gmm
    from benchmark_data import load_benchmark, CANONICAL_NULL
    os.makedirs("figures/si", exist_ok=True)
    kind = null_kind or CANONICAL_NULL
    fname = "figS_benchmark.png" if null_kind is None else f"figS_benchmark_{null_kind}.png"
    Xn, ys = load_benchmark(kind, n_per, seed)
    Z = StandardScaler().fit_transform(build_features(Xn))
    cents = np.vstack([Z[ys == c].mean(0) for c in range(3)])            # true-class centroids (oracle prior)

    def ev(labels):
        Cc = np.array([[((labels == k) & (ys == c)).sum() for c in range(3)] for k in range(3)])
        r, cc = linear_sum_assignment(-Cc); mp = {k: cc[i] for i, k in enumerate(r)}
        pm = np.array([mp[l] for l in labels])
        M = np.array([[((ys == tc) & (pm == pc)).sum() / max((ys == tc).sum(), 1) for pc in range(3)]
                      for tc in range(3)])
        sil = silhouette_score(Z, labels) if len(np.unique(labels)) > 1 else np.nan
        return dict(acc=(pm == ys).mean(), ari=adjusted_rand_score(ys, labels), sil=sil, M=M)

    # Two UNSUPERVISED mixtures + one MECHANISM-PRIOR model given the true-class centroids (oracle).
    res = {}
    res["Student-$t$"] = ev(fit_t_mixture(Z, 3, seed=seed, n_init=n_init)[0])
    res["skew-$t$"] = ev(fit_skew_t_mixture(Z, 3, seed=seed, n_init=n_init)[0])
    res["mechanism prior"] = ev(fit_theory_bayes_gmm(Z, 3, centroids_prior=cents, seed=seed, n_init=n_init)[0])
    nc_acc = (np.argmin(np.linalg.norm(Z[:, None, :] - cents[None], axis=2), axis=1) == ys).mean()

    LAB = ["SN", "TC", "Null"]; ccol = {0: "#d62828", 1: "#457b9d", 2: "#9aa0a6"}
    fig, axs = plt.subplots(2, 3, figsize=(COL2, 4.5))

    a = axs[0, 0]; t = np.linspace(0, 1, Xn.shape[1])
    for c in (0, 1, 2):
        for j in np.where(ys == c)[0][:5]:
            a.plot(t, Xn[j], color=ccol[c], alpha=0.55, lw=0.9)
        a.plot([], [], color=ccol[c], lw=2, label=LAB[c])
    a.set_xlim(0, 1); a.set_ylim(-0.03, 1.05); a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1])
    a.set_xlabel("normalized time"); a.set_ylabel("normalized state")
    a.legend(fontsize=6.0, loc="lower right", handlelength=0.9, handletextpad=0.4, borderaxespad=0.3, labelspacing=0.35)
    _tag(a, f"(a) synthetic SN / TC / null ({kind})")

    def confmat(ax, M, tag):
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_xticklabels(LAB, fontsize=6.5)
        ax.set_yticks(range(3)); ax.set_yticklabels(LAB, fontsize=6.5)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]*100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.5 else "#222222")
        ax.set_xlabel("assigned", fontsize=6.5); ax.set_ylabel("true", fontsize=6.5)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(length=0); _tag(ax, tag)

    order = ["Student-$t$", "skew-$t$", "mechanism prior"]
    cells = [axs[0, 1], axs[0, 2], axs[1, 0]]
    tags = ["(b) unsup:", "(c) unsup:", "(d) oracle:"]
    for ax, m, tg in zip(cells, order, tags):
        confmat(ax, res[m]["M"], f"{tg} {m} ({res[m]['acc']*100:.0f}%)")

    axS = axs[1, 1]; x = np.arange(3); w = 0.26
    SCOL = {"accuracy": "#1d4e89", "ARI": "#2a9d8f", "silhouette": "#e07a5f"}
    axS.bar(x - w, [res[m]["acc"] for m in order], w, color=SCOL["accuracy"], label="accuracy")
    axS.bar(x, [res[m]["ari"] for m in order], w, color=SCOL["ARI"], label="ARI")
    axS.bar(x + w, [res[m]["sil"] for m in order], w, color=SCOL["silhouette"], label="silhouette")
    axS.axvline(1.5, color="#bbbbbb", lw=0.8, ls=":")            # unsupervised | oracle divider
    axS.set_xticks(x); axS.set_xticklabels(["Stud-$t$", "skew-$t$", "mech.\nprior"], fontsize=6, rotation=18, ha="right")
    axS.set_ylim(0, 1.24); axS.set_ylabel("score", fontsize=7.5)
    axS.legend(fontsize=5.6, loc="upper center", ncol=3, frameon=False, columnspacing=0.9,
               handletextpad=0.4, bbox_to_anchor=(0.5, 1.0))
    _tag(axS, "(e) clustering scores", y=1.13)

    # (f) per-class recovery (diagonal of each confusion matrix) across all three methods
    axF = axs[1, 2]; xf = np.arange(3); wf = 0.26
    for ci, cl in enumerate((0, 1, 2)):
        axF.bar(xf + (ci - 1) * wf, [res[m]["M"][cl, cl] for m in order], wf,
                color=ccol[cl], label=LAB[cl])
    axF.axvline(1.5, color="#bbbbbb", lw=0.8, ls=":")            # unsupervised | oracle divider
    axF.set_xticks(xf); axF.set_xticklabels(["Stud-$t$", "skew-$t$", "mech.\nprior"], fontsize=6, rotation=18, ha="right")
    axF.set_ylim(0, 1.24); axF.set_ylabel("recovery (frac. of true class)", fontsize=7)
    axF.legend(fontsize=5.6, loc="upper center", ncol=3, frameon=False, columnspacing=0.9,
               handletextpad=0.4, bbox_to_anchor=(0.5, 1.0))
    _tag(axF, "(f) per-class recovery", y=1.13)

    fig.tight_layout()
    fig.savefig(f"figures/si/{fname}"); plt.close(fig)
    print(f"saved {fname} ({kind}) | nearest-centroid (prior only) {nc_acc*100:.0f}%; " + "; ".join(
        f"{m}: acc {res[m]['acc']*100:.0f}% ARI {res[m]['ari']:.2f} sil {res[m]['sil']:.2f}" for m in order))


def fig_benchmark_stack(n_per=1000, seed=0, n_init=3):
    # SI Fig S3 (combined): one row per null class (canonical logistic stable twin + the five
    # others), columns = example trajectories | clustering scores | per-class recovery. Replaces
    # the six separate benchmark-anatomy figures; the confusion matrices' load-bearing content
    # (their diagonals) appears as the per-class recovery bars.
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from unsup_real_world import fit_t_mixture, fit_skew_t_mixture, fit_theory_bayes_gmm
    from benchmark_data import load_benchmark, CANONICAL_NULL
    os.makedirs("figures/si", exist_ok=True)
    kinds = [CANONICAL_NULL, "linear", "exponential", "ramp_expsat", "sigmoid", "mixed"]
    KLAB = {CANONICAL_NULL: "logistic stable twin (canonical)", "linear": "linear null",
            "exponential": "exponential null", "ramp_expsat": "ramp-expsat null",
            "sigmoid": "sigmoid null", "mixed": "mixed null (open set)"}
    LAB = ["SN", "TC", "Null"]; ccol = {0: "#d62828", 1: "#457b9d", 2: "#9aa0a6"}
    SCOL = {"accuracy": "#1d4e89", "ARI": "#2a9d8f", "silhouette": "#e07a5f"}
    order = ["Student-$t$", "skew-$t$", "mechanism prior"]

    fig, axs = plt.subplots(6, 3, figsize=(COL2, 10.2),
                            gridspec_kw=dict(hspace=0.60, wspace=0.32, width_ratios=[1.2, 1, 1]))
    for r, kind in enumerate(kinds):
        Xn, ys = load_benchmark(kind, n_per, seed)
        Z = StandardScaler().fit_transform(build_features(Xn))
        cents = np.vstack([Z[ys == c].mean(0) for c in range(3)])

        def ev(labels):
            Cc = np.array([[((labels == k) & (ys == c)).sum() for c in range(3)] for k in range(3)])
            rr, cc = linear_sum_assignment(-Cc); mp = {k: cc[i] for i, k in enumerate(rr)}
            pm = np.array([mp[l] for l in labels])
            M = np.array([[((ys == tc) & (pm == pc)).sum() / max((ys == tc).sum(), 1)
                           for pc in range(3)] for tc in range(3)])
            sil = silhouette_score(Z, labels) if len(np.unique(labels)) > 1 else np.nan
            return dict(acc=(pm == ys).mean(), ari=adjusted_rand_score(ys, labels), sil=sil, M=M)

        res = {"Student-$t$": ev(fit_t_mixture(Z, 3, seed=seed, n_init=n_init)[0]),
               "skew-$t$": ev(fit_skew_t_mixture(Z, 3, seed=seed, n_init=n_init)[0]),
               "mechanism prior": ev(fit_theory_bayes_gmm(Z, 3, centroids_prior=cents,
                                                          seed=seed, n_init=n_init)[0])}
        print(f"[stack {kind}] " + "; ".join(f"{m}: {res[m]['acc']*100:.0f}%" for m in order), flush=True)

        # -- column 1: example trajectories --
        a = axs[r, 0]; t = np.linspace(0, 1, Xn.shape[1])
        for cl in (0, 1, 2):
            for j in np.where(ys == cl)[0][:5]:
                a.plot(t, Xn[j], color=ccol[cl], alpha=0.55, lw=0.8)
            a.plot([], [], color=ccol[cl], lw=1.8, label=LAB[cl])
        a.set_xlim(0, 1); a.set_ylim(-0.03, 1.05)
        a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1]); a.tick_params(labelsize=6)
        a.set_ylabel("normalized state", fontsize=6.5)
        if r == 5:
            a.set_xlabel("normalized time", fontsize=7)
        else:
            a.set_xticklabels([])
        if r == 0:
            a.legend(fontsize=5.2, loc="lower right", handlelength=0.9, handletextpad=0.4,
                     borderaxespad=0.3, labelspacing=0.3)
        _tag(a, f"({'abcdef'[r]}) {KLAB[kind]}")

        # -- column 2: clustering scores --
        axS = axs[r, 1]; x = np.arange(3); w = 0.26
        axS.bar(x - w, [res[m]["acc"] for m in order], w, color=SCOL["accuracy"], label="accuracy")
        axS.bar(x, [res[m]["ari"] for m in order], w, color=SCOL["ARI"], label="ARI")
        axS.bar(x + w, [res[m]["sil"] for m in order], w, color=SCOL["silhouette"], label="silhouette")
        for xi, m in enumerate(order):
            axS.text(xi - w, res[m]["acc"] + 0.04, f"{res[m]['acc']*100:.0f}",
                     ha="center", fontsize=4.6, color="#222222")
        axS.axvline(1.5, color="#bbbbbb", lw=0.8, ls=":")
        axS.set_xticks(x); axS.set_ylim(0, 1.30); axS.tick_params(labelsize=6)
        axS.set_ylabel("score", fontsize=6.5)
        if r == 5:
            axS.set_xticklabels(["Stud-$t$", "skew-$t$", "mech.\nprior"], fontsize=5.6,
                                rotation=18, ha="right")
        else:
            axS.set_xticklabels([])
        if r == 0:
            axS.legend(fontsize=4.9, loc="upper center", ncol=3, frameon=False,
                       columnspacing=0.7, handletextpad=0.3, bbox_to_anchor=(0.5, 1.02))
            _tag(axS, "clustering scores")

        # -- column 3: per-class recovery --
        axF = axs[r, 2]; xf = np.arange(3); wf = 0.26
        for ci in (0, 1, 2):
            axF.bar(xf + (ci - 1) * wf, [res[m]["M"][ci, ci] for m in order], wf,
                    color=ccol[ci], label=LAB[ci])
        axF.axvline(1.5, color="#bbbbbb", lw=0.8, ls=":")
        axF.set_xticks(xf); axF.set_ylim(0, 1.30); axF.tick_params(labelsize=6)
        axF.set_ylabel("recovery", fontsize=6.5)
        if r == 5:
            axF.set_xticklabels(["Stud-$t$", "skew-$t$", "mech.\nprior"], fontsize=5.6,
                                rotation=18, ha="right")
        else:
            axF.set_xticklabels([])
        if r == 0:
            axF.legend(fontsize=4.9, loc="upper center", ncol=3, frameon=False,
                       columnspacing=0.7, handletextpad=0.3, bbox_to_anchor=(0.5, 1.02))
            _tag(axF, "per-class recovery")

    fig.savefig("figures/si/figS_benchmark_all.png", bbox_inches="tight")
    plt.close(fig)
    print("saved figS_benchmark_all.png")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["1", "2", "3", "4", "si", "bench"]
    if "1" in which: fig1(); fig1_concept()   # audit fix: fig1_concept (main Fig 1) now has a caller
    if "2" in which: fig2_combined()
    if "2a" in which: fig2()          # legacy variant A (time-series only, with CDR)
    if "2t" in which: fig2_tau()      # legacy variant B (maturity + tau)
    if "3" in which: fig3()
    if "4" in which: fig4()
    if "si" in which: fig2_si(); fig4_si()
    if "bench" in which: fig_benchmark()
    if "benchall" in which:                     # per-null S3-style figures (legacy, single figures)
        for k in ("linear", "exponential", "ramp_expsat", "sigmoid", "mixed"):
            fig_benchmark(null_kind=k)
    if "benchstack" in which: fig_benchmark_stack()   # combined 6x3 SI figure (canonical + 5 nulls)
