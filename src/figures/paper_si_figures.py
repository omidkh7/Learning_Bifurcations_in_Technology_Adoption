#!/usr/bin/env python3
"""
paper_si_figures.py
===================
SI figures re-plotted to the manuscript house style (Times New Roman, no grid, no titles,
named panel tags, no em-dashes), output to figures/si/.
Covers: CDR by method (IEA), realized-vs-pipeline CDR, IEA-vs-SoCDR source comparison,
SoCDR realized + growth, four-group distributions, historical era split.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from paper_style import set_style, COL2
set_style()
from paper_figures import FAMCOL, FAMLABEL, FAMS, GROUP, GCOLOR, four_group_pca, load_four_group, _tag, PCDESC
from unsup_theory_features import FEATURE_NAMES
from compare_4groups import clean
from growth_compare import collect, geo_avg_to_peak, peak_growth

SI = "figures/si"; os.makedirs(SI, exist_ok=True)
MCOL = ["#1d4e89", "#e07a5f", "#2a9d8f", "#9b5de5", "#bc6c25", "#457b9d", "#06aed5",
        "#8ab17d", "#e76f51", "#adb5bd", "#264653"]


# ===================================================================== IEA by method
def si_cdr_iea():
    d = pd.read_csv("data/curated/cdr/cdr_by_method.csv")
    meths = sorted(d.method.unique())
    cm = {m: MCOL[i] for i, m in enumerate(meths)}
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 3.1), gridspec_kw=dict(wspace=0.28))
    for ax, track, tag in [(axs[0], "realized", "(a) realized (operational)"),
                           (axs[1], "promised", "(b) announced pipeline (to 2050)")]:
        for m in meths:
            s = d[(d.method == m) & (d.track == track)].sort_values("year")
            if len(s): ax.plot(s.year, s.cum_capacity_Mt, "-o", ms=2.5, lw=1.6, color=cm[m],
                               label=f"{m} ({s.cum_capacity_Mt.max():.2g})")
        ax.set_yscale("log"); ax.set_xlabel("year"); ax.set_ylabel("cumulative capacity (Mt CO$_2$/yr)")
        ax.legend(fontsize=5.8, loc="upper left"); _tag(ax, tag)
    fig.tight_layout(); fig.savefig(f"{SI}/figS_cdr_iea.png"); plt.close(fig); print("saved figS_cdr_iea.png")


# ===================================================================== realized vs pipeline (IEA + SoCDR)
def si_cdr_curves():
    iea = pd.read_csv("data/curated/cdr/cdr_by_method.csv")
    tot = iea.groupby(["track", "year"]).cum_capacity_Mt.sum().reset_index()
    so = pd.read_csv("data/curated/cdr/socdr_novel_by_method.csv"); so = so.rename(columns={so.columns[0]: "year"})
    so_tot = so.set_index("year").sum(axis=1, min_count=1).dropna()
    fig, ax = plt.subplots(figsize=(COL2 * 0.62, 3.1))
    r = tot[tot.track == "realized"]; p = tot[tot.track == "promised"]
    ax.plot(p.year, p.cum_capacity_Mt, "--s", ms=2.5, lw=1.6, color="#1d4e89", label="IEA announced capacity")
    ax.plot(r.year, r.cum_capacity_Mt, "-o", ms=2.5, lw=2.0, color="#1d4e89", label="IEA realized capacity")
    ax.plot(so_tot.index, so_tot.values, ":D", ms=4, lw=1.8, color="#e07a5f", mfc="white", label="SoCDR realized removals")
    ax.set_yscale("symlog", linthresh=0.1); ax.set_xlabel("year")
    ax.set_ylabel("cumulative CDR (Mt CO$_2$/yr)"); ax.legend(fontsize=6.5, loc="upper left")
    ax.axvline(2026, color="#bbbbbb", lw=0.7, ls=":")
    fig.tight_layout(); fig.savefig(f"{SI}/figS_cdr_curves.png"); plt.close(fig); print("saved figS_cdr_curves.png")


# ===================================================================== IEA vs SoCDR sources
def si_cdr_sources():
    pool = pd.read_csv("data/curated/cdr/cdr_sources_pooled.csv")
    ov = pd.read_csv("data/curated/cdr/cdr_sources_overlap.csv")
    SC = {"IEA": "#1d4e89", "SoCDR": "#e07a5f"}
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 3.1), gridspec_kw=dict(wspace=0.34))
    for ax, m, tag in [(axs[0], "DACCS", "(a) DACCS"), (axs[1], "BECCS", "(b) BECCS")]:
        for src in ("IEA", "SoCDR"):
            v = pool[(pool.method == m) & (pool.source == src)].sort_values("year")
            if len(v): ax.plot(v.year, v.value_Mt, "-o", ms=3, lw=1.8, color=SC[src],
                               label=f"{src} ({v.value_Mt.iloc[-1]:.3g} Mt)")
        ax.set_yscale("log"); ax.set_xlabel("year"); ax.set_ylabel("Mt CO$_2$/yr")
        rr = ov[ov.method == m].ratio_IEA_over_SoCDR.iloc[0]
        ax.legend(fontsize=6, loc="best"); _tag(ax, f"{tag} (IEA $\\approx${rr:.0f}$\\times$ SoCDR)")
    methods = sorted(set(pool.method)); y = np.arange(len(methods)); h = 0.38
    last = pool.sort_values("year").groupby(["method", "source"]).value_Mt.last().unstack("source").reindex(methods)
    for k, src in enumerate(("IEA", "SoCDR")):
        vals = last[src].values if src in last else np.full(len(methods), np.nan)
        axs[2].barh(y + (0.5 - k) * h, np.nan_to_num(vals), h, color=SC[src], alpha=0.9, label=src)
    axs[2].set_xscale("log"); axs[2].set_yticks(y); axs[2].set_yticklabels(methods, fontsize=5.6)
    axs[2].invert_yaxis(); axs[2].set_xlabel("latest value (Mt CO$_2$/yr)"); axs[2].legend(fontsize=6)
    _tag(axs[2], "(c) coverage by source")
    fig.tight_layout(); fig.savefig(f"{SI}/figS_cdr_sources.png"); plt.close(fig); print("saved figS_cdr_sources.png")


# ===================================================================== SoCDR realized + growth
def si_socdr():
    w = pd.read_csv("data/curated/cdr/socdr_novel_by_method.csv"); w = w.rename(columns={w.columns[0]: "year"})
    gr = pd.read_csv("data/curated/cdr/socdr_growth_rates.csv").sort_values("CAGR_pct")
    meths = [c for c in w.columns if c != "year"]
    cm = {m: MCOL[i % len(MCOL)] for i, m in enumerate(meths)}
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 3.2), gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.3))
    for m in meths:
        s = w[["year", m]].dropna(); s = s[s[m] > 0]
        if len(s): axs[0].plot(s.year, s[m], "-o", ms=2.5, lw=1.4, color=cm[m], label=m[:26])
    axs[0].set_yscale("log"); axs[0].set_xlabel("year"); axs[0].set_ylabel("realized removals (Mt CO$_2$/yr)")
    axs[0].legend(fontsize=5.2, loc="lower left", ncol=1); _tag(axs[0], "(a) realized novel CDR by method")
    axs[1].barh(range(len(gr)), gr.CAGR_pct, color=[cm.get(m, "#888") for m in gr.method], height=0.7)
    axs[1].set_yticks(range(len(gr))); axs[1].set_yticklabels([m[:24] for m in gr.method], fontsize=5.6)
    axs[1].set_xlabel("growth rate (%/yr, realized window)"); _tag(axs[1], "(b) per-method growth")
    fig.tight_layout(); fig.savefig(f"{SI}/figS_socdr.png"); plt.close(fig); print("saved figS_socdr.png")


# ===================================================================== four-group distributions
def si_compare4_dist():
    grp, P, evr, L, X = four_group_pca()
    # maturity (real attained level): Historical=1 assumed; Solar+Wind/BEV from collect; CDR pledge
    lvl = {f: [] for f in FAMS}
    for g, nm, yy, vv in collect():
        if g not in FAMS: continue
        v = np.asarray(vv, float)
        if g == "Historical": lvl[g].append(1.0)
        elif g in ("Renewables", "BEV"): lvl[g].append(v.max() / 100.0)
    fig, axs = plt.subplots(2, 2, figsize=(COL2, 5.0))
    for ax, pci, tag in [(axs[0, 0], 0, "(a)"), (axs[0, 1], 2, "(b)")]:
        data = [P[grp == f, pci] for f in FAMS]
        bp = ax.boxplot(data, positions=range(4), widths=0.6, showfliers=False, patch_artist=True)
        for patch, f in zip(bp["boxes"], FAMS): patch.set(facecolor=FAMCOL[f], alpha=0.30, edgecolor=FAMCOL[f])
        for med in bp["medians"]: med.set(color="#333", lw=1.2)
        for wsk in bp["whiskers"] + bp["caps"]: wsk.set(color="#888", lw=0.8)
        ax.set_xticks(range(4)); ax.set_xticklabels([FAMLABEL[f] for f in FAMS], fontsize=6.2, rotation=18, ha="right")
        ax.set_ylabel(f"PC{pci+1} ({PCDESC[pci]})", fontsize=7); _tag(ax, f"{tag} PC{pci+1} by family")
    med_lvl = [100 * np.nanmedian(lvl[f]) if lvl[f] else np.nan for f in FAMS]
    axs[1, 0].bar(range(4), [v if np.isfinite(v) else 0 for v in med_lvl], color=[FAMCOL[f] for f in FAMS], alpha=0.8)
    for i, f in enumerate(FAMS):
        if not np.isfinite(med_lvl[i]): axs[1, 0].text(i, 4, "pledge\n(n/a)", ha="center", fontsize=6, color="#777")
    axs[1, 0].set_xticks(range(4)); axs[1, 0].set_xticklabels([FAMLABEL[f] for f in FAMS], fontsize=6.2, rotation=18, ha="right")
    axs[1, 0].set_ylabel("median attained level (%)"); axs[1, 0].set_ylim(0, 105); _tag(axs[1, 0], "(c) maturity")
    # geometric-average-to-peak growth by family
    R = pd.read_csv("results/unsup/bifurcation_explore/growth_compare.csv")
    dd = [R[R.group == f].geo_growth_pct.values for f in FAMS]
    bp = axs[1, 1].boxplot([x[np.isfinite(x)] for x in dd], positions=range(4), widths=0.6, showfliers=False, patch_artist=True)
    for patch, f in zip(bp["boxes"], FAMS): patch.set(facecolor=FAMCOL[f], alpha=0.30, edgecolor=FAMCOL[f])
    for med in bp["medians"]: med.set(color="#333", lw=1.2)
    for wsk in bp["whiskers"] + bp["caps"]: wsk.set(color="#888", lw=0.8)
    axs[1, 1].set_xticks(range(4)); axs[1, 1].set_xticklabels([FAMLABEL[f] for f in FAMS], fontsize=6.2, rotation=18, ha="right")
    axs[1, 1].set_ylabel("avg-to-peak growth (%/yr)"); _tag(axs[1, 1], "(d) average pace to peak")
    fig.tight_layout(); fig.savefig(f"{SI}/figS_compare4_dist.png"); plt.close(fig); print("saved figS_compare4_dist.png")


# ===================================================================== historical era split
def si_era_split():
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    bins = [0, 1950, 1990, 3000]; labs = ["pre-1950", "1950-1990", "post-1990"]
    rows = []
    for g, nm, yy, vv, meta in collect(with_meta=True):
        if g != "Historical": continue
        c, _ = clean(np.asarray(vv, float))
        if c is None: continue
        era = pd.cut([meta["year_start"]], bins, labels=labs)[0]
        gr = geo_avg_to_peak(np.asarray(yy, float), np.asarray(vv, float))
        rows.append((era, gr, c))
    eras = [r[0] for r in rows]; grw = np.array([r[1] for r in rows]); X = np.stack([r[2] for r in rows])
    from unsup_theory_features import extract_theory_features
    F = np.nan_to_num(extract_theory_features(X, verbose=False))
    P = PCA().fit_transform(StandardScaler().fit_transform(F))
    ecol = {"pre-1950": "#003049", "1950-1990": "#669bbc", "post-1990": "#c1121f"}
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 3.1), gridspec_kw=dict(wspace=0.3))
    rng = np.random.default_rng(0)
    for ax, yv, ylab, tag in [(axs[0], grw, "avg-to-peak growth (%/yr)", "(a) growth by era"),
                              (axs[1], P[:, 0], f"PC1 ({PCDESC[0]})", "(b) PC1 by era")]:
        data = [yv[[e == L for e in eras]] for L in labs]
        bp = ax.boxplot(data, positions=range(3), widths=0.6, showfliers=False, patch_artist=True)
        for patch, L in zip(bp["boxes"], labs): patch.set(facecolor=ecol[L], alpha=0.30, edgecolor=ecol[L])
        for med in bp["medians"]: med.set(color="#333", lw=1.2)
        for wsk in bp["whiskers"] + bp["caps"]: wsk.set(color="#888", lw=0.8)
        for i, L in enumerate(labs):
            d = yv[[e == L for e in eras]]
            ax.scatter(rng.normal(i, 0.06, len(d)), d, s=5, color=ecol[L], alpha=0.5, lw=0)
        ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=6.5); ax.set_ylabel(ylab, fontsize=7.5); _tag(ax, tag)
    axs[0].set_ylim(0, min(120, np.nanpercentile(grw, 98)))
    fig.tight_layout(); fig.savefig(f"{SI}/figS_era_split.png"); plt.close(fig); print("saved figS_era_split.png")


# ===================================================================== peak rolling growth (robustness)
# (peak_growth and geo_avg_to_peak are imported from growth_compare so the SI uses the same code path
#  as Fig 4 and growth_compare.csv.)
def si_pc12():
    """Demoted from main-text Fig 4: PC1 and PC2 scores by family. Both are nearly flat across the four
    families ($\\eta^2$ small), which is why only the family-discriminating PC3 is kept in Fig 4."""
    grp, P, evr, L, X = four_group_pca()
    eta = []
    for i in range(3):
        x = P[:, i]; gm = x.mean()
        ssb = sum((grp == f).sum() * (x[grp == f].mean() - gm) ** 2 for f in FAMS)
        eta.append(ssb / (((x - gm) ** 2).sum() + 1e-12))
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(1, 2, figsize=(COL2 * 0.72, 2.9), gridspec_kw=dict(wspace=0.36))
    for k, pci in enumerate((0, 1)):
        ax = axs[k]
        data = [P[grp == f, pci] for f in FAMS]
        bp = ax.boxplot(data, positions=range(4), widths=0.6, showfliers=False, patch_artist=True)
        for patch, f in zip(bp["boxes"], FAMS): patch.set(facecolor=FAMCOL[f], alpha=0.30, edgecolor=FAMCOL[f])
        for med in bp["medians"]: med.set(color="#333333", lw=1.2)
        for wsk in bp["whiskers"] + bp["caps"]: wsk.set(color="#888888", lw=0.8)
        for i, f in enumerate(FAMS):
            y = P[grp == f, pci]; ax.scatter(rng.normal(i, 0.06, len(y)), y, s=5, color=FAMCOL[f], alpha=0.5, lw=0)
        ax.set_xticks(range(4)); ax.set_xticklabels([FAMLABEL[f] for f in FAMS], fontsize=6.2, rotation=18, ha="right")
        ax.set_ylabel(f"PC{pci+1} score ({PCDESC[pci]})", fontsize=7)
        _tag(ax, f"({'ab'[k]}) PC{pci+1} by family  ($\\eta^2$={eta[pci]:.2f})")
    fig.savefig(f"{SI}/figS_pc12.png", bbox_inches="tight", dpi=200); plt.close(fig)
    print(f"saved figS_pc12.png  (PC1 eta={eta[0]:.2f}, PC2 eta={eta[1]:.2f})")


def si_peak_grid():
    """Growth SI (folds in the former peak_growth figure): (a) realized peak-rate distribution vs the CDR
    pledge; (b-g) sensitivity of the peak-rate comparison to window length (2, 3 yr) and denominator floor
    (2, 5, 10% of observed range; shared log y-axis)."""
    cats = ["Historical", "Renewables", "BEV"]
    series = [(g, np.asarray(yy, float), np.asarray(vv, float), nm) for g, nm, yy, vv in collect()]
    floors = [0.02, 0.05, 0.10]; wins = [2, 3]; labs = "bcdefg"
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(COL2, 6.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.0, 1.0], hspace=0.5, wspace=0.12)

    # (a) realized peak-rate distribution (3-yr window, default 10% floor, matching main-text Fig 4) vs
    # the CDR pledge (folded in from the former standalone peak-growth figure)
    axh = fig.add_subplot(gs[0, :])
    peak_all = []; cdr_h = np.nan
    for g, y, v, nm in series:
        p = peak_growth(y, v, w=3)
        if g == "CDR":
            if "promised" in nm: cdr_h = p
        elif g in cats and np.isfinite(p):
            peak_all.append(p)
    peak_all = np.array(peak_all); pcth = 100 * (peak_all < cdr_h).mean()
    axh.hist(np.clip(peak_all, 0, 150), bins=np.linspace(0, 150, 31), color="#c9ccd1", edgecolor="white", lw=0.3)
    axh.axvline(cdr_h, color="#d62828", lw=2.0)
    axh.text(cdr_h - 4, axh.get_ylim()[1] * 0.82, f"CDR pledge\n{cdr_h:.0f}%/yr\nexceeds {pcth:.0f}% of\nrealized peaks",
             ha="right", va="top", fontsize=6.2, color="#d62828")
    axh.set_xlabel("peak 3-yr growth rate (%/yr)", fontsize=7); axh.set_ylabel("number of adoptions", fontsize=7)
    axh.set_xlim(0, 150); _tag(axh, "(a) realized peak-rate distribution")

    # (b-g) sensitivity of the peak-rate comparison to rolling-window length and denominator floor
    base = None; k = 0
    for r, w in enumerate(wins):
        for c, fl in enumerate(floors):
            ax = fig.add_subplot(gs[r + 1, c], sharey=base) if base is not None else fig.add_subplot(gs[r + 1, c])
            if base is None: base = ax
            fam = {g: [] for g in cats}; cdr = np.nan; allr = []
            for g, y, v, nm in series:
                p = peak_growth(y, v, w=w, floor=fl)
                if g == "CDR":
                    if "promised" in nm: cdr = p
                elif g in cats and np.isfinite(p):
                    fam[g].append(p); allr.append(p)
            allr = np.array(allr); pct = 100 * (allr < cdr).mean()
            bp = ax.boxplot([fam[g] for g in cats], positions=range(3), widths=0.55, showfliers=False, patch_artist=True)
            for patch, g in zip(bp["boxes"], cats): patch.set(facecolor=FAMCOL[g], alpha=0.30, edgecolor=FAMCOL[g])
            for med in bp["medians"]: med.set(color="#333333", lw=1.1)
            for wsk in bp["whiskers"] + bp["caps"]: wsk.set(color="#888888", lw=0.7)
            for i, g in enumerate(cats):
                yv = np.array(fam[g]); ax.scatter(rng.normal(i, 0.06, len(yv)), yv, s=3.5, color=FAMCOL[g], alpha=0.45, lw=0)
            ax.scatter([3], [cdr], marker="*", s=130, color="#d62828", edgecolor="k", lw=0.4, zorder=5)
            ax.set_yscale("log"); ax.set_ylim(4, 700); ax.set_xticks(range(4))
            if c > 0: ax.tick_params(labelleft=False)
            if r == 1: ax.set_xticklabels([FAMLABEL[g] for g in cats] + ["CDR"], fontsize=5.8, rotation=18, ha="right")
            else: ax.set_xticklabels([])
            if c == 0: ax.set_ylabel(f"{w}-yr window\npeak rate (%/yr, log)", fontsize=7)
            if r == 0: ax.set_title(f"floor {int(fl*100)}%", fontsize=8)
            _tag(ax, f"({labs[k]})"); k += 1
            ax.text(0.04, 0.97, f"CDR {cdr:.0f}%\nexceeds {pct:.0f}%", transform=ax.transAxes, fontsize=5.8, va="top", color="#d62828")
    fig.savefig(f"{SI}/figS_peak_grid.png", bbox_inches="tight"); plt.close(fig)
    print(f"saved figS_peak_grid.png  (folded histogram + 2x3 sensitivity; CDR peak > {pcth:.0f}% realized)")


def si_peak_growth(w=3):
    cats = ["Historical", "Renewables", "BEV"]
    peak = {c: [] for c in cats}; geo = {c: [] for c in cats}
    cdr_peak = cdr_geo = np.nan; real_all = []; real_geo = []
    for g, nm, yy, vv in collect():
        y = np.asarray(yy, float); v = np.asarray(vv, float)
        p = peak_growth(y, v, w); a = geo_avg_to_peak(y, v)
        if g == "CDR":
            if "promised" in nm: cdr_peak, cdr_geo = p, a
        elif g in cats:
            if np.isfinite(p): peak[g].append(p); real_all.append(p)
            if np.isfinite(a): geo[g].append(a); real_geo.append(a)
    real_all = np.array(real_all); pct = 100 * (real_all < cdr_peak).mean()
    real_geo = np.array(real_geo); pct_geo = 100 * (real_geo < cdr_geo).mean()

    fig, (axG, axA, axB) = plt.subplots(1, 3, figsize=(COL2, 3.1), gridspec_kw=dict(wspace=0.36))
    rng = np.random.default_rng(0)

    def fambox(ax, dat, star, ylab, tag):
        bp = ax.boxplot([dat[g] for g in cats], positions=range(3), widths=0.55, showfliers=False, patch_artist=True)
        for patch, g in zip(bp["boxes"], cats): patch.set(facecolor=FAMCOL[g], alpha=0.30, edgecolor=FAMCOL[g])
        for med in bp["medians"]: med.set(color="#333333", lw=1.2)
        for wsk in bp["whiskers"] + bp["caps"]: wsk.set(color="#888888", lw=0.8)
        for i, g in enumerate(cats):
            yv = np.array(dat[g]); ax.scatter(rng.normal(i, 0.06, len(yv)), yv, s=4, color=FAMCOL[g], alpha=0.5, lw=0)
        ax.scatter([3], [star], marker="*", s=150, color="#d62828", edgecolor="k", lw=0.5, zorder=5)
        ax.annotate("CDR\npledged", (3, star), xytext=(3, star - 10), fontsize=6.0, color="#d62828", ha="center", va="top")
        ax.set_xticks(range(4)); ax.set_xticklabels([FAMLABEL[g] for g in cats] + ["CDR"], fontsize=6, rotation=18, ha="right")
        ax.set_ylabel(ylab, fontsize=7); ax.set_ylim(0, max(110, star * 1.2)); _tag(ax, tag)

    fambox(axG, geo, cdr_geo, "geometric-average growth to peak (%/yr)", "(a) average pace to peak")
    fambox(axA, peak, cdr_peak, "peak 3-yr growth rate (%/yr)", "(b) peak rolling rate")

    axB.hist(np.clip(real_all, 0, 150), bins=np.linspace(0, 150, 31), color="#c9ccd1", edgecolor="white", lw=0.3)
    axB.axvline(cdr_peak, color="#d62828", lw=2.0)
    axB.text(cdr_peak - 4, axB.get_ylim()[1] * 0.82, f"CDR pledge\n{cdr_peak:.0f}%/yr\nexceeds {pct:.0f}% of\nrealized peaks",
             ha="right", va="top", fontsize=6.2, color="#d62828")
    axB.set_xlabel("peak 3-yr growth rate (%/yr)", fontsize=7); axB.set_ylabel("number of adoptions", fontsize=7)
    axB.set_xlim(0, 150); _tag(axB, "(c) realized peak-rate distribution")
    fig.tight_layout(); fig.savefig(f"{SI}/figS_peak_growth.png"); plt.close(fig)
    print(f"saved figS_peak_growth.png  (CDR pledge: avg {cdr_geo:.0f}%/yr > {pct_geo:.0f}%; "
          f"peak {cdr_peak:.0f}%/yr > {pct:.0f}%)")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["iea", "curves", "sources", "socdr", "dist", "era", "peak", "peakgrid"]
    if "iea" in todo: si_cdr_iea()
    if "curves" in todo: si_cdr_curves()
    if "sources" in todo: si_cdr_sources()
    if "socdr" in todo: si_socdr()
    if "dist" in todo: si_compare4_dist()
    if "era" in todo: si_era_split()
    if "peak" in todo: si_peak_growth()
    if "peakgrid" in todo: si_peak_grid()
