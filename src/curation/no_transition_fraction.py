#!/usr/bin/env python3
"""
no_transition_fraction.py
=========================
§92.9 — What % of series genuinely DO NOT undergo a bifurcation/transition?
(canonical messaging §90: adoption goes through a bifurcation — quantify the exceptions.)

A series shows NO resolvable transition if it looks like a stationary/noisy plateau rather than
a state-to-state move. Three transparent, complementary criteria on each series (raw resolution,
y kept as-is/min-max, §83 decline-truncated, identical to feature_space_7datasets):
  C1 net-rise      : net_rise_score < 0.5         (did not clearly move from low→high state)
  C2 no-S-shape    : logistic R² − linear R² ≤ 0  (an S/threshold curve fits no better than a line)
  C3 noise-domin.  : residual_var_frac > 0.5      (more than half the variance is high-freq noise)
"NO transition" (headline) = C1 OR C3 (state didn't move OR signal is mostly noise). We also report
each criterion alone and the strict AND. Datasets: combined, filter10, genuine_v2, hardexcl, s002,
s005 + the four groups (Historical/Renewables/BEV/CDR).

Output: figures/si/figS_no_transition.png + results/unsup/bifurcation_explore/no_transition.csv
"""
import os, warnings, pickle, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paper_style import set_style, COL2
from paper_figures import _tag, INK
set_style()
from scipy.optimize import curve_fit
import unsup_real_world
from unsup_theory_features import extract_theory_features, FEATURE_NAMES
from feature_space_7datasets import _U, load_pkl, recon_raw
DATASETS = ["combined", "filter10", "genuine_v2", "hardexcl", "s002", "s005"]
from adoption_clean import truncate_decline
from growth_compare import collect

I_RISE = FEATURE_NAMES.index("net_rise_score")
I_RESID = FEATURE_NAMES.index("residual_var_frac")


def s_score(v):
    """R²(logistic) − R²(linear) on a raw [0,1] series (the §42 transition-strength score)."""
    v = np.asarray(v, float); n = len(v)
    if n < 6 or v.max() - v.min() < 1e-9: return np.nan
    y = (v - v.min()) / (v.max() - v.min()); t = np.linspace(0, 1, n)
    sst = ((y - y.mean())**2).sum() + 1e-12
    r2_lin = 1 - ((y - np.polyval(np.polyfit(t, y, 1), t))**2).sum() / sst
    try:
        def logi(x, x0, k): return 1 / (1 + np.exp(-k * (x - x0)))
        p, _ = curve_fit(logi, t, y, p0=[0.5, 10], maxfev=4000, bounds=([0, 0.5], [1, 80]))
        r2_log = 1 - ((y - logi(t, *p))**2).sum() / sst
    except Exception:
        r2_log = r2_lin
    return float(r2_log - r2_lin)


def criteria(F, raws):
    rise = F[:, I_RISE]; resid = F[:, I_RESID]
    ss = np.array([s_score(r) for r in raws])
    c1 = rise < 0.5
    c2 = ~(ss > 0)                       # logistic not better than linear (incl. nan)
    c3 = resid > 0.5
    no_trans = c1 | c3
    return dict(c1_lowrise=c1, c2_noS=c2, c3_noise=c3, no_transition=no_trans,
                strict=c1 & c3, s_score=ss)


def main():
    # ── historical datasets: reconstruct raw + features (combined base, subset) ──
    base = load_pkl("combined"); rid = [s.row_id for s in base]
    raws, keep = [], []
    for i, s in enumerate(base):
        r = recon_raw(s.x_full, s.years_full)
        if r is None: continue
        raws.append(r); keep.append(i)
    F = np.vstack([extract_theory_features(r[None, :], verbose=False) for r in raws])
    base_idx = {rid[keep[j]]: j for j in range(len(keep))}

    q = pd.read_csv("results/unsup/shape_diagnostic/scurve_quality.csv")
    gv2 = load_pkl("combined_genuine_v2")
    def sub_gv2(mask): return [gv2[i].row_id for i in mask if i < len(gv2)]
    nx = ~q.excluded
    sets = {"combined": rid, "filter10": [s.row_id for s in load_pkl("combined_filter10")],
            "genuine_v2": [s.row_id for s in gv2],
            "hardexcl": sub_gv2(q.loc[nx, "idx"].astype(int).values),
            "s002": sub_gv2(q.loc[nx & (q.s_score >= 0.02), "idx"].astype(int).values),
            "s005": sub_gv2(q.loc[nx & (q.s_score >= 0.05), "idx"].astype(int).values)}

    rows = []
    crit_by = {}
    for name in DATASETS:
        ix = np.array([base_idx[r] for r in sets[name] if r in base_idx])
        cr = criteria(F[ix], [raws[i] for i in ix])
        crit_by[name] = cr
        rows.append(dict(dataset=name, N=len(ix),
                         pct_no_transition=round(100 * cr["no_transition"].mean(), 1),
                         pct_low_rise=round(100 * cr["c1_lowrise"].mean(), 1),
                         pct_no_Sshape=round(100 * cr["c2_noS"].mean(), 1),
                         pct_noise_dom=round(100 * cr["c3_noise"].mean(), 1),
                         pct_strict_AND=round(100 * cr["strict"].mean(), 1)))

    # ── four groups (real-level, growth_compare cleaning) ──
    data = collect()
    for g in ["Historical", "Renewables", "BEV", "CDR"]:
        curves = [vv for (gg, nm, yy, vv) in data if gg == g]
        rr = []
        for vv in curves:
            v = truncate_decline(np.asarray(vv, float))
            if v.max() - v.min() < 1e-9: continue
            rr.append((v - v.min()) / (v.max() - v.min()))
        if not rr: continue
        Fg = np.vstack([extract_theory_features(r[None, :], verbose=False) for r in rr])
        cr = criteria(Fg, rr)
        rows.append(dict(dataset=g, N=len(rr),
                         pct_no_transition=round(100 * cr["no_transition"].mean(), 1),
                         pct_low_rise=round(100 * cr["c1_lowrise"].mean(), 1),
                         pct_no_Sshape=round(100 * cr["c2_noS"].mean(), 1),
                         pct_noise_dom=round(100 * cr["c3_noise"].mean(), 1),
                         pct_strict_AND=round(100 * cr["strict"].mean(), 1)))

    df = pd.DataFrame(rows)
    df.to_csv("results/unsup/bifurcation_explore/no_transition.csv", index=False)
    print(df.to_string(index=False))

    # ── figure (SI house style) ──
    SI = "figures/si"; os.makedirs(SI, exist_ok=True)
    NICE = {"combined": "Combined", "filter10": "Filter-10", "genuine_v2": "Genuine",
            "hardexcl": "Hard-excl.", "s002": r"$s\geq0.02$", "s005": r"$s\geq0.05$",
            "Historical": "Historical", "Renewables": "Solar+Wind", "BEV": "BEV", "CDR": "CDR"}
    order = df.dataset.tolist(); x = np.arange(len(order))
    labels = [NICE.get(o, o) for o in order]
    ncur = len(DATASETS)                                   # divider: curation tiers | target families

    fig, ax = plt.subplots(1, 2, figsize=(COL2, 3.1), gridspec_kw=dict(wspace=0.30))

    # (a) headline: no resolvable transition (low net-rise OR noise-dominated)
    ax[0].bar(x, df.pct_no_transition, color="#bc4749", width=0.72)
    for i, v in enumerate(df.pct_no_transition):
        ax[0].text(i, v + 0.2, f"{v:.0f}", ha="center", va="bottom", fontsize=6, color=INK)
    ax[0].axvline(ncur - 0.5, color="#9aa0a6", lw=0.8, ls=":")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=6)
    ax[0].set_ylabel("no resolvable transition (% of series)", fontsize=7.5)
    ax[0].set_ylim(0, max(df.pct_no_transition) * 1.20)
    _tag(ax[0], "(a) headline")

    # (b) each criterion separately
    w = 0.26
    for k, (c, lab, col) in enumerate([("pct_low_rise", "low net-rise", "#e76f51"),
                                       ("pct_no_Sshape", "no S-shape", "#e9c46a"),
                                       ("pct_noise_dom", "noise-dominated", "#457b9d")]):
        ax[1].bar(x + (k - 1) * w, df[c], w, label=lab, color=col)
    ax[1].axvline(ncur - 0.5, color="#9aa0a6", lw=0.8, ls=":")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=6)
    ax[1].set_ylabel("meets criterion (% of series)", fontsize=7.5)
    ax[1].legend(fontsize=6, loc="upper right", handlelength=1.2)
    _tag(ax[1], "(b) by individual criterion")

    fig.savefig(f"{SI}/figS_no_transition.png", bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved → {SI}/figS_no_transition.png + results/unsup/bifurcation_explore/no_transition.csv")


if __name__ == "__main__":
    main()
