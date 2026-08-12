#!/usr/bin/env python3
"""
compare_4groups.py
==================
Compare the 38-D feature space across FOUR groups with ONE identical cleaning pipeline:
  1. Historical  — genuine_v2 mature techs (excl. renewables)
  2. Renewables  — Solar PV, Onshore/Offshore Wind, Renewable/Geothermal/Marine/Solar-thermal
  3. BEV         — battery-EV share of new-car sales per country, MONTHLY→YEARLY (to match)
  4. CDR         — TWO labeled source pools (§94): IEA CCUS realized/promised (operational/announced
                   capacity) + SoCDR Edition-3 realized (delivered novel-CDR removals)

Identical cleaning for every raw ANNUAL series: onset-trim (drop leading <2% of peak) → require
≥10 annual points → min-max [0,1] → PCHIP-100 → 38 theory features. (Min-max is the only normalization
valid across mixed metrics; the real LEVEL is reported separately, §80.) PCA fit on the genuine_v2
reference (groups 1+2); BEV/CDR projected onto it.

Outputs: runs/figures/{fig_compare4_scatter.png, fig_compare4_dist.png},
         runs/unsup/bifurcation_explore/compare4_summary.csv
"""
import warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import unsup_real_world
from unsup_theory_features import extract_theory_features, FEATURE_NAMES
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
NG = 100
CARS = "all_carsales_monthly.csv"
IEA = "IEA CCUS Projects Database 2026.xlsx"
RENEW = {"Solar Photovoltaic", "Onshore Wind Energy", "Offshore Wind Energy", "Renewable Power",
         "Geothermal Energy", "Solar Thermal Energy", "Marine Energy"}
EXCLUDE = {"EU + EFTA + UK", "EUROPEAN UNION", "EFTA", "California CNCDA",
           "Netherlands including used", "Spain including used", "Nepal Comtrade",
           "United Kingdom SMMT", "California"}
GCOL = {"Historical": "#9aa0a6", "Renewables": "#2a9d8f", "BEV": "#6a4c93", "CDR": "#fb8500"}


def clean(v, min_n=10):
    """IDENTICAL cleaning for every raw annual series → (curve100, n_after) or (None,0).
    Onset-trim leading <2% + §56 post-peak DECLINE truncation (keep rise+plateau) + min-max + grid.
    min_n is the post-trim length gate (default 10; relaxed for curated global CDR aggregates)."""
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 3 or v.max() <= 0: return None, 0
    onset = int(np.argmax(v > 0.02 * v.max()))
    v = v[max(0, onset - 2):]
    pk = int(v.argmax())                                   # §56: drop post-peak structural decline
    if v[-1] < 0.90 * v.max() and pk < len(v) - 1:
        end = pk
        while end + 1 < len(v) and v[end + 1] >= 0.90 * v.max():
            end += 1
        v = v[:end + 1]
    if len(v) < min_n or v.max() - v.min() < 1e-9: return None, 0
    mm = (v - v.min()) / (v.max() - v.min())
    xs = PchipInterpolator(np.linspace(0, 1, len(mm)), mm, extrapolate=False)(np.linspace(0, 1, NG))
    b = np.isnan(xs)
    if b.any(): xs[b] = np.interp(np.where(b)[0], np.where(~b)[0], xs[~b])
    return np.clip(xs, 0, 1).astype(np.float32), len(v)


class _U(pickle.Unpickler):
    def find_class(self, m, n):
        return getattr(unsup_real_world, n) if hasattr(unsup_real_world, n) else super().find_class(m, n)


def _scale_maps():
    """Raw (tech,country) peak and per-tech leader peak from technologies.csv (for the §56
    meaningful-scale gate that removes niche low-capacity 'saturated' stalls)."""
    raw = pd.read_csv("technologies.csv"); yc = [c for c in raw.columns if c.isdigit()]
    peak = raw[yc].max(axis=1)
    key = raw.assign(peak=peak).groupby(["Technology Name", "Country Name"])["peak"].max()
    leader = raw.assign(peak=peak).groupby("Technology Name")["peak"].max()
    return key.to_dict(), leader.to_dict()


def load_gv2(scale_frac=0.01):
    s = _U(open("data/combined_genuine_v2/real_world_samples.pkl", "rb")).load()
    md = pd.read_csv("data/combined_genuine_v2/metadata.csv")
    key, leader = _scale_maps()
    rows = []; n_niche = 0
    for i, smp in enumerate(s):
        yf = np.asarray(smp.years_full, float)
        if len(yf) < 10 or yf[-1] - yf[0] < 1: continue
        raw = np.asarray(smp.x_full)[np.clip(np.round((yf - yf[0]) / (yf[-1] - yf[0]) * 499).astype(int), 0, 499)]
        c, n = clean(raw)
        if c is None: continue
        # §56 meaningful-scale gate: drop niche stalls (raw peak < 1% of the tech's global leader)
        pk = key.get((md.tech_name[i], md.country[i]), np.nan); ld = leader.get(md.tech_name[i], np.nan)
        if np.isfinite(pk) and np.isfinite(ld) and ld > 0 and pk < scale_frac * ld:
            n_niche += 1; continue
        grp = "Renewables" if md.tech_name[i] in RENEW else "Historical"
        rows.append((grp, md.tech_name[i], n, md.series_type[i] == "saturating", np.nan, c))
    print(f"  [scale gate] dropped {n_niche} niche stalls (raw peak < {scale_frac*100:.0f}% of tech leader)")
    return rows


def load_bev():
    df = pd.read_csv(CARS); df = df[df.Fuel != "Others"]
    df["year"] = df.YYYYMM // 100
    df = df[df.year < 2026]        # drop the partial year (Jan-May 2026 reads as a seasonal decline)
    bev = df[df.Fuel == "BatteryElectric"].groupby(["year", "Country"]).Value.sum().rename("bev").reset_index()
    tot = df.groupby(["year", "Country"]).Value.sum().rename("tot").reset_index()
    m = bev.merge(tot, on=["year", "Country"]); m["share"] = 100 * m.bev / m.tot
    rows = []
    for c, g in m.groupby("Country"):
        if c in EXCLUDE: continue
        g = g.sort_values("year"); sh = g.share.values
        curve, n = clean(sh)
        if curve is None: continue
        rows.append(("BEV", c, n, bool(sh[-1] >= 0.9 * sh.max() and sh.max() >= 50), sh.max() / 100.0, curve))
    return rows


def load_cdr():
    df = pd.read_excel(IEA, sheet_name="DRAFT CCUS Projects Database")
    df = df[pd.to_numeric(df["CDR capacity (Mt CO2/yr)"], errors="coerce").fillna(0) > 0].copy()
    df["yr"] = pd.to_numeric(df["Operation"], errors="coerce"); df["cap"] = pd.to_numeric(df["CDR capacity (Mt CO2/yr)"], errors="coerce")
    def cum(stat, yto):
        d = df[df["Project status"].isin(stat)].dropna(subset=["yr", "cap"]); d = d[d.yr <= yto]
        if not len(d): return None
        yrs = np.arange(int(d.yr.min()), yto + 1)
        return np.array([d[d.yr.astype(int) <= y]["cap"].sum() for y in yrs])
    pledge = cum({"Operational", "Under construction", "Planned"}, 2050).max()
    out = []
    for nm, cv, lvl in [("CDR realized", cum({"Operational"}, 2026), None),
                        ("CDR promised", cum({"Operational", "Under construction", "Planned"}, 2050), 1.0),
                        ("DAC promised", None, None), ("Bio-CDR promised", None, None)]:
        if cv is None and nm == "DAC promised":
            sub = df[df.Subsector.str.contains("Direct Air", na=False)]
            d = sub.dropna(subset=["yr", "cap"]); yrs = np.arange(int(d.yr.min()), 2051); cv = np.array([d[d.yr.astype(int) <= y]["cap"].sum() for y in yrs])
        if cv is None and nm == "Bio-CDR promised":
            sub = df[~df.Subsector.str.contains("Direct Air", na=False)]
            d = sub.dropna(subset=["yr", "cap"]); yrs = np.arange(int(d.yr.min()), 2051); cv = np.array([d[d.yr.astype(int) <= y]["cap"].sum() for y in yrs])
        c, n = clean(cv)
        if c is None: continue
        lvl = (cv.max() / pledge) if nm == "CDR realized" else lvl
        out.append(("CDR", nm, n, False, lvl, c))
    return out


def main():
    # §84: data via shared collector (honest gen-share renewables + start-gate + scale/decline gates)
    from growth_compare import collect as _collect4
    grp, name, nobs, level, curves = [], [], [], [], []
    for g, nm, yy, vv in _collect4():
        c, _ = clean(np.asarray(vv, float), min_n=8 if g == "CDR" else 10)   # CDR globals: relaxed gate
        if c is None: continue
        grp.append(g); name.append(nm); nobs.append(len(vv)); curves.append(c)
        # §85: Historical mature techs assumed saturated (level≈1) — justified by age + the many
        # peak-then-decline (substitution) curves; absolute-magnitude metrics can't be measured vs a
        # denominator so this is an assumption, not a computed value (caveat in paper caption).
        level.append(float(np.asarray(vv).max() / 100) if g in ("BEV", "Renewables")
                     else (1.0 if g == "Historical" else np.nan))
    grp = np.array(grp); name = np.array(name); nobs = np.array(nobs); level = np.array(level, float); X = np.stack(curves)
    F = extract_theory_features(X, verbose=False)

    # PCA fit on genuine_v2 reference (Historical+Renewables), project all
    ref = np.isin(grp, ["Historical", "Renewables"])
    sc = StandardScaler().fit(np.nan_to_num(F[ref])); pca = PCA().fit(sc.transform(np.nan_to_num(F[ref])))
    PC = pca.transform(sc.transform(np.nan_to_num(F)))
    fn = {n: j for j, n in enumerate(FEATURE_NAMES)}

    # ── summary table ──
    rows = []
    for g in ["Historical", "Renewables", "BEV", "CDR"]:
        mk = grp == g
        rows.append(dict(group=g, N=int(mk.sum()), median_nobs=int(np.median(nobs[mk])),
                         median_level=(round(float(np.nanmedian(level[mk])), 2) if np.isfinite(level[mk]).any() else np.nan),
                         PC1=round(float(PC[mk, 0].mean()), 2), PC2=round(float(PC[mk, 1].mean()), 2),
                         PC3=round(float(PC[mk, 2].mean()), 2),
                         t_inflection=round(float(F[mk, fn["t_inflection"]].mean()), 2),
                         net_rise=round(float(F[mk, fn["net_rise_score"]].mean()), 2)))
    S = pd.DataFrame(rows); S.to_csv("runs/unsup/bifurcation_explore/compare4_summary.csv", index=False)
    print("=" * 90)
    print("  38-D FEATURE SPACE — 4-GROUP COMPARISON (identical cleaning; BEV yearly)")
    print("=" * 90)
    print(S.to_string(index=False))

    # CDR source tag (§94): IEA pool (○) vs SoCDR pool (◇), from series name
    cdr_src = np.array(["SoCDR" if "SoCDR" in n else "IEA" for n in name])
    # ── fig 1: scatter PC1×PC2, PC1×PC3 ──
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
    for a, (yi, ylab) in zip(ax, [(1, "PC2 (noise)"), (2, "PC3 (step-ness)")]):
        for g in ["Historical", "Renewables", "BEV"]:
            mk = grp == g
            a.scatter(PC[mk, 0], PC[mk, yi],
                      s=(18 if g == "Historical" else 60),
                      c=GCOL[g], alpha=(.55 if g == "Historical" else .9),
                      edgecolor="k", lw=(.2 if g == "Historical" else .5),
                      zorder=(2 if g == "Historical" else 4), label=f"{g} (N={mk.sum()})")
        # CDR split by source pool: IEA = circle, SoCDR = diamond, both orange + annotated
        for src, mrk in [("IEA", "o"), ("SoCDR", "D")]:
            mk = (grp == "CDR") & (cdr_src == src)
            if not mk.any(): continue
            a.scatter(PC[mk, 0], PC[mk, yi], s=90, marker=mrk, c=GCOL["CDR"], alpha=.95,
                      edgecolor="k", lw=.7, zorder=5, label=f"CDR — {src} (N={mk.sum()})")
        for i in np.where(grp == "CDR")[0]:
            lab = name[i].replace("CDR ", "").replace("(", "").replace(")", "")
            a.annotate(lab, (PC[i, 0], PC[i, yi]), fontsize=6.8, xytext=(5, 3),
                       textcoords="offset points", zorder=6)
        a.set_xlabel("PC1 (sharpness / back-loadedness)"); a.set_ylabel(ylab); a.legend(fontsize=7.5)
    ax[0].set_title("PC1 × PC2"); ax[1].set_title("PC1 × PC3")
    fig.suptitle("38-D feature space — Historical vs Renewables vs BEV vs CDR "
                 "(CDR shown as two labeled source pools: IEA ○, SoCDR ◇)",
                 fontweight="bold")
    plt.tight_layout(); fig.savefig("runs/figures/fig_compare4_scatter.png", dpi=140, bbox_inches="tight")

    # ── fig 2: distributions + maturity + group-mean feature heatmap ──
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    order = ["Historical", "Renewables", "BEV", "CDR"]
    for k, (axi, pck, lab) in enumerate([(ax[0, 0], 0, "PC1 sharpness"), (ax[0, 1], 2, "PC3 step-ness")]):
        axi.boxplot([PC[grp == g, pck] for g in order], labels=order, showfliers=False,
                    patch_artist=True, boxprops=dict(facecolor="#eee"))
        for i, g in enumerate(order, 1):
            axi.scatter(np.full((grp == g).sum(), i) + np.random.uniform(-.12, .12, (grp == g).sum()),
                        PC[grp == g, pck], s=10, c=GCOL[g], alpha=.5)
        axi.set_title(f"{lab} by group"); axi.set_ylabel(lab)
    # maturity = REAL adoption level reached
    medlev = [100 * np.nanmedian(level[grp == g]) if np.isfinite(level[grp == g]).any() else 0 for g in order]
    ax[1, 0].bar(range(4), medlev, color=[GCOL[g] for g in order])
    for i, g in enumerate(order):
        if not np.isfinite(level[grp == g]).any(): ax[1, 0].text(i, 4, "n/a\n(pledge)", ha="center", fontsize=8)
    ax[1, 0].text(0, 90, "assumed saturated\n(age + decline)", ha="center", fontsize=7, style="italic")
    ax[1, 0].set_xticks(range(4)); ax[1, 0].set_xticklabels(order)
    ax[1, 0].set_ylabel("median REAL level reached (%)"); ax[1, 0].set_ylim(0, 105)
    ax[1, 0].set_title("maturity: Historical≈saturated (assumed); Renewables 12% gen-share, BEV 23% sales,\n"
                       "CDR ~3% — only historical has run its course")
    # group-mean z-scored feature heatmap (top discriminating features)
    Z = (F - F[ref].mean(0)) / (F[ref].std(0) + 1e-9)
    gm = np.vstack([Z[grp == g].mean(0) for g in order])
    spread = gm.std(0); topj = np.argsort(-spread)[:16]
    im = ax[1, 1].imshow(gm[:, topj], aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    ax[1, 1].set_yticks(range(4)); ax[1, 1].set_yticklabels(order)
    ax[1, 1].set_xticks(range(len(topj))); ax[1, 1].set_xticklabels([FEATURE_NAMES[j] for j in topj], rotation=90, fontsize=7)
    ax[1, 1].set_title("group-mean feature z-scores (top-16 discriminating)"); plt.colorbar(im, ax=ax[1, 1], fraction=.04)
    fig.suptitle("38-D feature space — group distributions, maturity, and discriminating features",
                 fontweight="bold")
    plt.tight_layout(); fig.savefig("runs/figures/fig_compare4_dist.png", dpi=140, bbox_inches="tight")
    print("\nSaved → runs/figures/fig_compare4_{scatter,dist}.png + .../compare4_summary.csv")


if __name__ == "__main__":
    main()
