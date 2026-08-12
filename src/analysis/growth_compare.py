#!/usr/bin/env python3
"""
growth_compare.py
=================
Compare GROWTH RATE / scale-up speed across the 4 groups (Historical, Renewables, BEV, CDR) and
showcase that the CDR ANNOUNCED pipeline demands a historically unprecedented growth rate.
Plus a few policy-relevant 38-D feature comparisons.

Per series (real years, onset-trimmed, ≥10 pts, §56 meaningful-scale-gated for magnitude metrics).
Two growth metrics (no 10→90 "diffusion time"; the 90% threshold is misleading for still-incomplete
curves, where it is 90% of the level reached SO FAR rather than of a real ceiling):
  geo_growth   = geometric-average (compound) %/yr from 10% of the attained PEAK up to the peak
                 = (10**(1/Δt) − 1)·100, Δt = years from the 10%-of-peak crossing to the peak.
  peak_growth  = fastest sustained rolling-window rate (3-yr window, start above 10% of observed range)
Policy features (38-D): early_plateau_frac (length of the pre-takeoff "valley"), saturation_speed.

Output: results/figures/fig_growth_compare.png + results/unsup/bifurcation_explore/growth_compare.csv
"""
import warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import unsup_real_world
from unsup_theory_features import extract_theory_features, FEATURE_NAMES
NG = 100
START_MAX = 0.10            # §84: discard left-truncated series whose first point > 10% of peak
CARS = "data/raw/all_carsales_monthly.csv"; IEA = "data/raw/IEA CCUS Projects Database 2026.xlsx"
RENEW_OWID = "data/raw/owid_solar_wind_share.csv"   # §84: honest solar/wind generation share
RENEW = {"Solar Photovoltaic", "Onshore Wind Energy", "Offshore Wind Energy", "Renewable Power",
         "Geothermal Energy", "Solar Thermal Energy", "Marine Energy"}
EXCLUDE = {"EU + EFTA + UK", "EUROPEAN UNION", "EFTA", "California CNCDA", "Netherlands including used",
           "Spain including used", "Nepal Comtrade", "United Kingdom SMMT", "California"}
GCOL = {"Historical": "#9aa0a6", "Renewables": "#2a9d8f", "BEV": "#6a4c93", "CDR": "#fb8500"}
HATCH_MEDIAN = 6.2          # Nemet et al. 2023 median historical growth %/yr


class _U(pickle.Unpickler):
    def find_class(self, m, n):
        return getattr(unsup_real_world, n) if hasattr(unsup_real_world, n) else super().find_class(m, n)


def truncate_decline(years, vals, thr=0.90):
    """§56 consistency: drop post-peak STRUCTURAL decline (keep rise+plateau = the adoption S-curve).
    Only acts when the series ended below thr·peak (genuinely declined, e.g. abandoned/substituted)."""
    v = np.asarray(vals, float); pk = int(v.argmax()); peak = v[pk]
    if v[-1] < thr * peak and pk < len(v) - 1:
        end = pk
        while end + 1 < len(v) and v[end + 1] >= thr * peak:
            end += 1
        return years[:end + 1], vals[:end + 1]
    return years, vals


def onset_trim(years, vals, gate=True):
    vals = np.asarray(vals, float); years = np.asarray(years, float)
    ok = ~np.isnan(vals)
    vals, years = vals[ok], years[ok]
    if len(vals) < 3 or vals.max() <= 0: return None, None
    if gate and vals[0] > START_MAX * vals.max(): return None, None   # §84: left-truncated → discard
    o = int(np.argmax(vals > 0.02 * vals.max()))
    years, vals = years[max(0, o - 2):], vals[max(0, o - 2):]
    years, vals = truncate_decline(years, vals)            # §56: remove post-peak decline
    if len(vals) < 10 or vals.max() - vals.min() < 1e-9: return None, None
    return years, vals


def diffusion(years, vals):
    """years from 10%→90% of observed range, via linear interpolation on real years."""
    lo = vals.min() + 0.10 * (vals.max() - vals.min()); hi = vals.min() + 0.90 * (vals.max() - vals.min())
    def cross(thr):
        for i in range(1, len(vals)):
            if vals[i] >= thr:
                if vals[i] == vals[i-1]: return years[i]
                f = (thr - vals[i-1]) / (vals[i] - vals[i-1])
                return years[i-1] + f * (years[i] - years[i-1])
        return years[-1]
    dt = cross(hi) - cross(lo)
    return max(dt, 2.0)   # annual data can't resolve a 10→90 rise faster than ~2 yr (avoid artifacts)
    # NB: retained only for the (excluded) pushback diagnostics; the manuscript uses the two metrics below.


def geo_avg_to_peak(years, vals):
    """Geometric-average (compound) growth rate, %/yr, from 10% of the ATTAINED PEAK up to the peak.
    No ceiling/saturation is assumed: the 'peak' is simply the highest observed level, so the metric is
    well defined for still-incomplete curves. Δt = years from the 10%-of-peak crossing to the peak.
    When the series starts below 10% of its peak, peak/base = 10 and the rate is (10**(1/Δt) − 1)·100;
    when it starts above (possible only where the onset gate is off, i.e. CDR), the ACTUAL rise ratio
    peak/v[0] is used, since assuming 10x would inflate the rate."""
    v = np.asarray(vals, float); y = np.asarray(years, float)
    pk = v.max(); ip = int(v.argmax()); base = 0.10 * pk
    if v[0] >= base:
        t0 = y[0]; ratio = pk / max(v[0], 1e-12)
    else:
        ratio = 10.0
        t0 = y[ip]
        for i in range(1, ip + 1):
            if v[i] >= base:
                t0 = (y[i] if v[i] == v[i-1]
                      else y[i-1] + (base - v[i-1]) / (v[i] - v[i-1]) * (y[i] - y[i-1]))
                break
    dt = max(y[ip] - t0, 2.0)               # annual data: floor at 2 yr
    return (ratio ** (1.0 / dt) - 1.0) * 100.0


def peak_growth(years, vals, w=3, floor=0.10):
    """Peak rolling growth rate, %/yr: the fastest w-year compound growth over the series, with the
    window START above `floor` of the observed range (a floor that removes the x→0 blow-up). The
    reported value is the maximum over all admissible windows; returns nan if none qualify."""
    y = np.asarray(years, float); v = np.asarray(vals, float); rng = v.max() - v.min()
    if rng <= 0: return np.nan
    lo = v.min() + floor * rng; best = np.nan
    for i in range(len(v)):
        j = np.searchsorted(y, y[i] - w)
        if j < i and v[j] >= lo and v[j] > 0:
            dt = y[i] - y[j]
            if dt >= 2:
                r = (v[i] / v[j]) ** (1.0 / dt) - 1.0
                best = r if np.isnan(best) else max(best, r)
    return 100 * best if not np.isnan(best) else np.nan


def curve100(vals):
    mm = (vals - vals.min()) / (vals.max() - vals.min())
    xs = PchipInterpolator(np.linspace(0, 1, len(mm)), mm, extrapolate=False)(np.linspace(0, 1, NG))
    b = np.isnan(xs)
    if b.any(): xs[b] = np.interp(np.where(b)[0], np.where(~b)[0], xs[~b])
    return np.clip(xs, 0, 1).astype(np.float32)


def scale_maps():
    raw = pd.read_csv("data/raw/technologies.csv"); yc = [c for c in raw.columns if c.isdigit()]
    pk = raw[yc].max(axis=1)
    return (raw.assign(p=pk).groupby(["Technology Name", "Country Name"])["p"].max().to_dict(),
            raw.assign(p=pk).groupby("Technology Name")["p"].max().to_dict())


def collect(with_meta=False):
    """Returns list of (group, name, years, values). with_meta=True appends a 5th element:
    dict(scale='national'|'global', country, year_start) — for §92.8 national/global &
    §92.10 era splits."""
    out = []
    def add(g, nm, yy, vv, scale, country):
        out.append((g, nm, yy, vv, dict(scale=scale, country=country, year_start=int(yy[0])))
                   if with_meta else (g, nm, yy, vv))
    # gv2 historical + renewables (scale-gated)
    s = _U(open("data/curated/combined_genuine_v2/real_world_samples.pkl", "rb")).load()
    md = pd.read_csv("data/curated/combined_genuine_v2/metadata.csv"); key, leader = scale_maps()
    for i, smp in enumerate(s):
        yf = np.asarray(smp.years_full, float)
        if len(yf) < 10 or yf[-1] - yf[0] < 1: continue
        pk = key.get((md.tech_name[i], md.country[i]), np.nan); ld = leader.get(md.tech_name[i], np.nan)
        if np.isfinite(pk) and np.isfinite(ld) and ld > 0 and pk < 0.01 * ld: continue
        if md.tech_name[i] in RENEW: continue        # §84: capacity renewables replaced by gen-share
        vals = np.asarray(smp.x_full)[np.clip(np.round((yf - yf[0]) / (yf[-1] - yf[0]) * 499).astype(int), 0, 499)]
        yy, vv = onset_trim(yf, vals)
        if yy is None: continue
        sc = str(md.scale[i]) if "scale" in md.columns else "national"
        add("Historical", md.tech_name[i], yy, vv, sc, str(md.country[i]))
    # Renewables: HONEST solar/wind SHARE OF ELECTRICITY GENERATION (OWID), real level (not min-max→1)
    rn = pd.read_csv(RENEW_OWID)
    for (tech, country), g in rn.groupby(["tech", "country"]):
        g = g.sort_values("year")
        if g.share.max() < 5: continue               # meaningful: reached ≥5% of generation
        yy, vv = onset_trim(g.year.values, g.share.values)
        sc = "global" if str(country) in ("World", "OWID World") else "national"
        if yy is not None: add("Renewables", f"{tech}:{country}", yy, vv, sc, str(country))
    # BEV yearly
    df = pd.read_csv(CARS); df = df[df.Fuel != "Others"]; df["year"] = df.YYYYMM // 100
    df = df[df.year < 2026]        # drop the partial year (Jan-May 2026 reads as a seasonal decline)
    bev = df[df.Fuel == "BatteryElectric"].groupby(["year", "Country"]).Value.sum().rename("bev").reset_index()
    tot = df.groupby(["year", "Country"]).Value.sum().rename("tot").reset_index()
    m = bev.merge(tot, on=["year", "Country"]); m["share"] = 100 * m.bev / m.tot
    for c, g in m.groupby("Country"):
        if c in EXCLUDE: continue
        g = g.sort_values("year"); yy, vv = onset_trim(g.year.values, g.share.values)
        sc = "global" if str(c) in ("World", "Global") else "national"
        if yy is not None: add("BEV", c, yy, vv, sc, str(c))
    # CDR — kept as TWO separately-labeled source pools (§94): IEA-CCUS (operational capacity) and
    # SoCDR Edition-3 (realized novel-CDR removals). Same family, different count → both shown.
    #   (a) IEA realized + promised cumulative capacity
    cd = pd.read_excel(IEA, sheet_name="DRAFT CCUS Projects Database")
    cd = cd[pd.to_numeric(cd["CDR capacity (Mt CO2/yr)"], errors="coerce").fillna(0) > 0].copy()
    cd["yr"] = pd.to_numeric(cd["Operation"], errors="coerce"); cd["cap"] = pd.to_numeric(cd["CDR capacity (Mt CO2/yr)"], errors="coerce")
    for nm, stat, yto in [("CDR realized (IEA)", {"Operational"}, 2026),
                          ("CDR promised (IEA)", {"Operational", "Under construction", "Planned"}, 2050)]:
        d = cd[cd["Project status"].isin(stat)].dropna(subset=["yr", "cap"]); d = d[d.yr <= yto]
        yrs = np.arange(int(d.yr.min()), yto + 1); cum = np.array([d[d.yr.astype(int) <= y]["cap"].sum() for y in yrs])
        yy, vv = onset_trim(yrs, cum, gate=False)     # CDR = cumulative-from-zero pipeline; exempt start-gate
        if yy is not None: add("CDR", nm, yy, vv, "global", "World")   # CDR pipeline = global aggregate
    #   (b) SoCDR realized total novel CDR (sum across methods/yr) — independent delivery accounting.
    #   Only 9 annual points (2017-2025) → below onset_trim's 10-pt gate, but it is a curated global
    #   aggregate (like the IEA pipeline), so add directly after stripping any post-peak decline.
    try:
        so = pd.read_csv("data/curated/cdr/socdr_novel_by_method.csv")
        so = so.rename(columns={so.columns[0]: "year"})
        tot = so.set_index("year").sum(axis=1, min_count=1).dropna()
        yy, vv = truncate_decline(tot.index.values.astype(float), tot.values)
        if len(vv) >= 3 and vv.max() - vv.min() > 1e-9:
            add("CDR", "CDR realized (SoCDR)", yy, vv, "global", "World")
    except FileNotFoundError:
        pass
    return out


def main():
    data = collect()
    rows = []
    for g, nm, yy, vv in data:
        geo = geo_avg_to_peak(yy, vv); pk = peak_growth(yy, vv)
        F = extract_theory_features(curve100(vv)[None, :], verbose=False)[0]
        rows.append(dict(group=g, name=nm,
                         geo_growth_pct=round(geo, 1),
                         peak_growth_pct=round(pk, 1) if np.isfinite(pk) else np.nan,
                         early_plateau=round(float(F[FEATURE_NAMES.index("early_plateau_frac")]), 3),
                         sat_speed=round(float(F[FEATURE_NAMES.index("saturation_speed")]), 3)))
    R = pd.DataFrame(rows); R.to_csv("results/unsup/bifurcation_explore/growth_compare.csv", index=False)
    order = ["Historical", "Renewables", "BEV", "CDR"]
    print("Median geometric-average-to-peak and peak rolling growth (%/yr) by group:")
    print(R.groupby("group")[["geo_growth_pct", "peak_growth_pct"]].median().round(1).reindex(order).to_string())
    cats = ["Historical", "Renewables", "BEV"]
    for col, lab in [("geo_growth_pct", "geometric-average-to-peak"), ("peak_growth_pct", "peak rolling")]:
        real = R[R.group.isin(cats)][col].dropna().values
        cdr_prom = R[(R.group == "CDR") & (R.name == "CDR promised (IEA)")][col].values[0]
        pct = (real < cdr_prom).mean() * 100
        print(f"  {lab:26s}: CDR pledge {cdr_prom:.0f}%/yr → exceeds {pct:.0f}% of realised "
              f"(realised median {np.median(real):.0f}%/yr, max {real.max():.0f}%/yr)")

    # ── internal diagnostic figure (not the manuscript figure) ──
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    def box(a, col, ylab):
        a.boxplot([R[R.group == g][col].dropna() for g in order], labels=order, showfliers=False,
                  patch_artist=True, boxprops=dict(facecolor="#eee"))
        for i, g in enumerate(order, 1):
            v = R[R.group == g][col].dropna()
            a.scatter(np.full(len(v), i) + np.random.uniform(-.12, .12, len(v)), v, s=18, c=GCOL[g], alpha=.6, zorder=3)
        cp = R[(R.group == "CDR") & (R.name == "CDR promised (IEA)")][col].values[0]
        a.scatter([order.index("CDR") + 1], [cp], s=320, marker="*", c="red", edgecolor="k", zorder=6,
                  label=f"CDR pledge {cp:.0f}%/yr")
        a.set_yscale("log"); a.set_ylabel(ylab); a.legend(fontsize=8)
    box(ax[0], "geo_growth_pct", "geometric-average-to-peak growth (%/yr)")
    ax[0].set_title("(A) Average pace to peak — CDR pledge mid-pack")
    box(ax[1], "peak_growth_pct", "peak rolling growth (%/yr)")
    ax[1].set_title("(B) Peak sustained pace — CDR pledge at the historical extreme")
    fig.suptitle("Growth-rate comparison — Historical · Solar+Wind · BEV · CDR (two metrics)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout(); fig.savefig("results/figures/fig_growth_compare.png", dpi=140, bbox_inches="tight")
    print("\nSaved → results/figures/fig_growth_compare.png + .../growth_compare.csv")


if __name__ == "__main__":
    main()
