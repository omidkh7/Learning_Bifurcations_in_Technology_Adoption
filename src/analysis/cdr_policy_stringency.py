#!/usr/bin/env python3
"""
cdr_policy_stringency.py
========================
§95 — A per-country CDR-POLICY-STRINGENCY covariate, built from the SoCDR Edition-3 Chapter-5 G20 CDR
policy database (327 policy records across the 20 G20 jurisdictions). The idea: policy stringency is
the institutional analogue of the "pushback" axis — where pushback measures fossil-incumbent
RESISTANCE, stringency measures the state's PULL for CDR. We then ask whether stringency yet predicts
realized novel-CDR deployment (SoCDR Ch7, 2025 by-country snapshot).

Stringency index per jurisdiction = mean of six min-max-normalized components (each 0-1):
  n_policies     breadth   — number of CDR-relevant policy records
  n_binding      teeth     — records with a BINDING target (Target binding == 'Yes')
  n_novel        relevance — records touching NOVEL (engineered) CDR, not just land sinks
  n_economic     money     — Economic-instrument records (subsidies, tax, crediting markets)
  n_instr_types  breadth   — distinct instrument TYPES deployed (comprehensiveness)
  n_recent       momentum  — records enacted in 2021+ (the post-pledge wave)

Outcome = total realized NOVEL CDR by country in 2025 (Mt CO2/yr), Ch7. Cross-section is the 19
individual G20 countries (EU bloc excluded to avoid double-counting France/Germany/Italy).

Caveat up front: realized novel CDR is tiny (a few Mt globally) and concentrated in cheap-biochar
geographies (US, India, Brazil), NOT in the policy-richest jurisdictions (EU/UK/Australia) — so a
weak or even NEGATIVE stringency↔deployment correlation is the EXPECTED early-stage signature, not a
failure: policy leadership precedes, and does not yet coincide with, delivery.

Inputs : SoCDR-Edition-3/SoCDR-Edition-3-Chapter-5-G20-CDR-policy-database.csv
         SoCDR-Edition-3/SoCDR-Edition-3-Chapter-7.csv  (Novel CDR by method and country in 2025)
Outputs: data/cdr/cdr_policy_stringency.csv   per-country covariate (components + index + outcome)
         runs/figures/fig_cdr_policy_stringency.png
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

os.makedirs("data/cdr", exist_ok=True)
CH5 = "SoCDR-Edition-3/SoCDR-Edition-3-Chapter-5-G20-CDR-policy-database.csv"
CH7 = "SoCDR-Edition-3/SoCDR-Edition-3-Chapter-7.csv"

# ISO3 (policy DB)  →  country name (Ch7 snapshot). EU bloc handled separately.
ISO2NAME = {"USA": "United States", "GBR": "United Kingdom", "DEU": "Germany", "FRA": "France",
            "ITA": "Italy", "JPN": "Japan", "CAN": "Canada", "AUS": "Australia", "BRA": "Brazil",
            "CHN": "China", "IND": "India", "IDN": "Indonesia", "MEX": "Mexico",
            "RUS": "Russian Federation", "SAU": "Saudi Arabia", "ZAF": "South Africa",
            "TUR": "Turkey", "ARG": "Argentina", "KOR": "Korea, Rep."}
COMPS = ["n_policies", "n_binding", "n_novel", "n_economic", "n_instr_types", "n_recent"]


def stringency_table():
    p = pd.read_csv(CH5, low_memory=False)
    p["yr"] = pd.to_numeric(p["Year enacted"], errors="coerce")
    rows = []
    for iso, g in p.groupby("Iso"):
        broad = g["Broad method"].astype(str)
        instr = g["Instrument type"].astype(str).str.split(";").explode().str.strip()
        rows.append(dict(
            iso=iso, jurisdiction=g["Jurisdiction"].iloc[0],
            n_policies=len(g),
            n_binding=int((g["Target binding"].astype(str).str.strip() == "Yes").sum()),
            n_novel=int(broad.str.contains("Novel", case=False, na=False).sum()),
            n_economic=int((g["Instrument category"].astype(str) == "Economic instrument").sum()),
            n_instr_types=int(instr[instr.ne("nan") & instr.ne("Other") & instr.ne("")].nunique()),
            n_recent=int((g.yr >= 2021).sum()),
        ))
    s = pd.DataFrame(rows)
    # min-max normalize each component → composite stringency index (0-1)
    for c in COMPS:
        rng = s[c].max() - s[c].min()
        s[c + "_n"] = (s[c] - s[c].min()) / rng if rng else 0.0
    s["stringency"] = s[[c + "_n" for c in COMPS]].mean(axis=1)
    return s.sort_values("stringency", ascending=False).reset_index(drop=True)


def realized_2025():
    d = pd.read_csv(CH7, low_memory=False)
    snap = d[d["Indicator"] == "Novel CDR by method and country in 2025"].copy()
    snap["v"] = pd.to_numeric(snap["Value"], errors="coerce")
    return snap.groupby("Country").v.sum()


def main():
    s = stringency_table()
    by_c = realized_2025()
    s["novel_cdr_2025_Mt"] = [by_c.get(ISO2NAME.get(i, ""), 0.0) for i in s.iso]
    s.loc[s.iso == "EU", "novel_cdr_2025_Mt"] = np.nan          # bloc: no own country outcome
    s.to_csv("data/cdr/cdr_policy_stringency.csv", index=False)

    print("=== G20 CDR-policy stringency (sorted) ===")
    print(s[["jurisdiction", "iso"] + COMPS + ["stringency", "novel_cdr_2025_Mt"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # cross-section: individual G20 countries (exclude EU bloc)
    cs = s[s.iso != "EU"].dropna(subset=["novel_cdr_2025_Mt"]).copy()
    rho, pv = spearmanr(cs.stringency, cs.novel_cdr_2025_Mt)
    # also vs binding-target count alone (the 'teeth' sub-index)
    rho_b, pv_b = spearmanr(cs.n_binding, cs.novel_cdr_2025_Mt)
    print(f"\nCross-section (n={len(cs)} G20 countries):")
    print(f"  Spearman(stringency, realized novel CDR 2025) = {rho:+.3f} (p={pv:.3f})")
    print(f"  Spearman(binding-targets, realized novel CDR) = {rho_b:+.3f} (p={pv_b:.3f})")
    lead_pol = s.iloc[0].jurisdiction
    lead_dep = cs.sort_values("novel_cdr_2025_Mt").iloc[-1].jurisdiction
    print(f"  policy leader = {lead_pol}; deployment leader = {lead_dep} "
          f"→ {'mismatch' if lead_pol != lead_dep else 'match'} (early-stage decoupling)")

    # ── figure (SI house style; canonical generator of figS_cdr_policy_stringency) ──
    from paper_style import set_style, COL2
    set_style()
    from paper_figures import _tag
    SI = "Manuscript/SI_figures"; os.makedirs(SI, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(COL2, 3.4), gridspec_kw=dict(wspace=0.5))
    # (a) composite stringency index, ranked (policy leaders on top; EU excluded, duplicates members)
    sp = s[s.iso != "EU"].sort_values("stringency")
    y = np.arange(len(sp))
    ax[0].barh(y, sp.stringency, color="#457b9d", height=0.72)
    ax[0].set_yticks(y); ax[0].set_yticklabels(sp.jurisdiction, fontsize=6)
    ax[0].set_xlim(0, 1); ax[0].set_xlabel("CDR-policy stringency index (0 to 1)", fontsize=7.5)
    _tag(ax[0], "(a) policy stringency, G20")

    # (b) stringency vs realized novel CDR 2025 (symlog keeps zero-deployment countries visible)
    ax[1].scatter(cs.stringency, cs.novel_cdr_2025_Mt, s=38, c="#e76f51", edgecolor="k", lw=0.5, zorder=3)
    lab = set(cs.sort_values("novel_cdr_2025_Mt").iso.tail(3)) | set(cs.sort_values("stringency").iso.tail(2))
    for _, r in cs.iterrows():
        if r.iso in lab:
            ax[1].annotate(r.iso, (r.stringency, r.novel_cdr_2025_Mt), fontsize=6,
                           xytext=(3, 3), textcoords="offset points")
    ax[1].set_yscale("symlog", linthresh=1e-3)
    ax[1].set_ylim(bottom=0)
    ax[1].set_xlabel("policy stringency", fontsize=7.5)
    ax[1].set_ylabel("realized novel CDR 2025 (Mt CO$_2$/yr)", fontsize=7.5)
    ax[1].text(0.04, 0.96, f"Spearman rho={rho:+.2f}\n(p={pv:.2f})", transform=ax[1].transAxes,
               fontsize=7, color="#777", va="top")
    _tag(ax[1], "(b) stringency vs realized deployment")

    fig.savefig(f"{SI}/figS_cdr_policy_stringency.png", bbox_inches="tight")
    fig.savefig("runs/figures/fig_cdr_policy_stringency.png", bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved → data/cdr/cdr_policy_stringency.csv + {SI}/figS_cdr_policy_stringency.png")


if __name__ == "__main__":
    main()
