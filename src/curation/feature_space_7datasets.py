#!/usr/bin/env python3
"""
feature_space_7datasets.py
==========================
Classical 38-D feature-space analysis across 7 datasets × 2 representations, with n_obs controls.

Datasets (all subsets of the curated `combined`, 4217 series):
  1 combined            (unfiltered)
  2 filter10            (start-truncation ≤10%)
  3 genuine_v1          (combined_genuine)
  4 genuine_v2          (combined_genuine_v2)
  5 hardexcl            (genuine_v2, shape-excluded removed)   ─┐ via scurve_quality.csv
  6 s002                (hardexcl ∩ s_score≥0.02)              ─┤  (idx = local genuine_v2)
  7 s005                (hardexcl ∩ s_score≥0.05)              ─┘

Representations (per §discussion — PCHIP-500 was only for the retired NN; analyse shape directly):
  RAW       : original-resolution series (x_full sampled at the observed years), min-max amplitude,
              variable length — faithful values, but irregular spacing; the upsampling/n_obs confound
              is removed.
  PCHIP-100 : re-grid raw → 100 uniform points (modest grid: fixes spacing with minimal upsampling).

For each (dataset, representation): PCA (PC1%, #PCs for 90%), Hopkins, KMeans silhouette (k=2..8),
PC1/PC2 Sarle bimodality, and corr(PC1/PC2, n_obs) as the n_obs control. Outputs big per-rep grids
+ a summary figure + JSON.
"""
import os, json, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.stats import skew, kurtosis, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import unsup_real_world
from unsup_theory_features import extract_theory_features
from adoption_clean import truncate_decline

OUTD = "results/unsup/bifurcation_explore"; os.makedirs(OUTD, exist_ok=True)
FIGD = "results/figures"
DATASETS = ["combined", "filter10", "genuine_v1", "genuine_v2", "hardexcl", "s002", "s005"]


class _U(pickle.Unpickler):
    def find_class(self, m, n):
        return getattr(unsup_real_world, n) if hasattr(unsup_real_world, n) else super().find_class(m, n)


def load_pkl(folder):
    return _U(open(f"data/curated/{folder}/real_world_samples.pkl", "rb")).load()


def recon_raw(x_full, years_full):
    yf = np.asarray(years_full, float)
    if len(yf) < 10 or yf[-1] - yf[0] < 1:
        return None
    pos = (yf - yf[0]) / (yf[-1] - yf[0])
    raw = np.asarray(x_full)[np.clip(np.round(pos * 499).astype(int), 0, 499)]
    raw = truncate_decline(raw)                               # §83: drop post-peak decline
    if raw.max() - raw.min() < 1e-9:
        return None
    return ((raw - raw.min()) / (raw.max() - raw.min())).astype(np.float64)


def to_grid(raw, n=100):
    t1 = np.linspace(0, 1, n)
    xs = np.clip(PchipInterpolator(np.linspace(0, 1, len(raw)), raw, extrapolate=False)(t1), 0, 1)
    b = np.isnan(xs)
    if b.any():
        xs[b] = np.interp(np.where(b)[0], np.where(~b)[0], xs[~b])
    return xs.astype(np.float32)


def hopkins(Xz, m=200, seed=0):
    rng = np.random.default_rng(seed); n, d = Xz.shape; m = min(m, n // 2)
    nb = NearestNeighbors(n_neighbors=2).fit(Xz); si = rng.choice(n, m, False)
    u = rng.uniform(Xz.min(0), Xz.max(0), (m, d))
    wd = nb.kneighbors(Xz[si], 2)[0][:, 1]; ud = nb.kneighbors(u, 1)[0][:, 0]
    return float(ud.sum() / (ud.sum() + wd.sum() + 1e-12))


def bc(x):
    n = len(x); g = skew(x); k = kurtosis(x, fisher=True)
    return float((g**2 + 1) / (k + 3 * (n - 1)**2 / ((n - 2) * (n - 3)) + 1e-9))


def analyse(F, nobs):
    Xz = StandardScaler().fit_transform(np.nan_to_num(F))
    pca = PCA().fit(Xz); cum = np.cumsum(pca.explained_variance_ratio_); P = pca.transform(Xz)
    n90 = int(np.argmax(cum >= 0.90) + 1)
    sil = []
    samp = np.random.default_rng(0).choice(len(Xz), min(2000, len(Xz)), False)
    for k in range(2, 9):
        sil.append(silhouette_score(Xz[samp], KMeans(k, n_init=4, random_state=0).fit(Xz).labels_[samp]))
    return dict(N=len(F), pc1=float(pca.explained_variance_ratio_[0]), n90=n90,
                hopkins=hopkins(Xz), max_sil=float(np.nanmax(sil)),
                bc_pc1=bc(P[:, 0]), bc_pc2=bc(P[:, 1]),
                corr_pc1_nobs=abs(spearmanr(P[:, 0], nobs)[0]),
                corr_pc2_nobs=abs(spearmanr(P[:, 1], nobs)[0]),
                cum=cum, sil=sil, P=P)


def main():
    # ── base = combined; build raw & pchip100 features once, then subset ──
    print("Loading combined base + reconstructing series …")
    base = load_pkl("combined")
    rid = [s.row_id for s in base]; rid2i = {r: i for i, r in enumerate(rid)}
    raws, nobs, keep = [], [], []
    for i, s in enumerate(base):
        r = recon_raw(s.x_full, s.years_full)
        if r is None:
            continue
        raws.append(r); nobs.append(len(s.years_full)); keep.append(i)
    keep = np.array(keep); nobs = np.array(nobs)
    print(f"  {len(keep)} usable (≥10 obs). Extracting RAW features (per-series)…")
    F_raw = np.vstack([extract_theory_features(r[None, :], verbose=False) for r in raws])
    print("  Extracting PCHIP-100 features…")
    F_100 = extract_theory_features(np.vstack([to_grid(r, 100) for r in raws]), verbose=False)
    base_idx = {rid[keep[j]]: j for j in range(len(keep))}     # row_id → row in F_raw/F_100

    # ── member row_ids per dataset ──
    def members(folder):
        return [s.row_id for s in load_pkl(folder)]
    q = pd.read_csv("results/unsup/shape_diagnostic/scurve_quality.csv")
    gv2 = load_pkl("combined_genuine_v2")
    def sub_gv2(mask_idx):
        return [gv2[i].row_id for i in mask_idx if i < len(gv2)]
    not_ex = ~q.excluded
    sets = {
        "combined":   rid,
        "filter10":   members("combined_filter10"),
        "genuine_v1": members("combined_genuine"),
        "genuine_v2": members("combined_genuine_v2"),
        "hardexcl":   sub_gv2(q.loc[not_ex, "idx"].astype(int).values),
        "s002":       sub_gv2(q.loc[not_ex & (q.s_score >= 0.02), "idx"].astype(int).values),
        "s005":       sub_gv2(q.loc[not_ex & (q.s_score >= 0.05), "idx"].astype(int).values),
    }

    res = {"raw": {}, "pchip100": {}}; idxs = {}
    for name in DATASETS:
        ix = np.array([base_idx[r] for r in sets[name] if r in base_idx])
        idxs[name] = ix
        res["raw"][name] = analyse(F_raw[ix], nobs[ix])
        res["pchip100"][name] = analyse(F_100[ix], nobs[ix])
        rr, pp = res["raw"][name], res["pchip100"][name]
        print(f"  {name:11s} N={len(ix):5d} nobs_med={int(np.median(nobs[ix])):3d} | "
              f"RAW: dim{rr['n90']} sil{rr['max_sil']:.2f} ρ(PC,nobs){rr['corr_pc1_nobs']:.2f} | "
              f"P100: dim{pp['n90']} sil{pp['max_sil']:.2f} ρ{pp['corr_pc1_nobs']:.2f}")

    # ── JSON summary (drop big arrays) ──
    slim = {rep: {n: {k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in d.items() if k not in ("cum", "sil", "P")}
                  for n, d in res[rep].items()} for rep in res}
    json.dump(slim, open(f"{OUTD}/fs7_summary.json", "w"), indent=2)

    _fig_grid(res["raw"], nobs, idxs, "RAW (original resolution)", f"{FIGD}/fig_fs7_raw.png")
    _fig_grid(res["pchip100"], nobs, idxs, "PCHIP-100 (uniform grid)", f"{FIGD}/fig_fs7_pchip100.png")
    _fig_summary(res, nobs, idxs)
    print(f"\nSaved → {OUTD}/fs7_summary.json + {FIGD}/fig_fs7_{{raw,pchip100,summary}}.png")


def _fig_grid(R, nobs, idxs, rep, path):
    fig, ax = plt.subplots(len(DATASETS), 3, figsize=(13, 3.0 * len(DATASETS)))
    for row, name in enumerate(DATASETS):
        d = R[name]
        ax[row, 0].plot(range(1, 21), d["cum"][:20], "-o", ms=3); ax[row, 0].axhline(.9, ls="--", c="r")
        ax[row, 0].set_ylim(0.15, 1.02); ax[row, 0].set_ylabel(f"{name}\nN={d['N']}", fontsize=10, fontweight="bold")
        if row == 0: ax[row, 0].set_title("PCA scree (cum var)")
        ax[row, 0].text(.95, .2, f"{d['n90']} PCs→90%", transform=ax[row, 0].transAxes, ha="right", fontsize=8)
        ax[row, 1].plot(range(2, 9), d["sil"], "-o"); ax[row, 1].axhline(.5, ls="--", c="r")
        ax[row, 1].set_ylim(0, 0.6)
        if row == 0: ax[row, 1].set_title("KMeans silhouette vs k")
        ax[row, 1].text(.95, .85, f"max {d['max_sil']:.2f}", transform=ax[row, 1].transAxes, ha="right", fontsize=8)
        ax[row, 2].hexbin(d["P"][:, 0], d["P"][:, 1], gridsize=35, cmap="viridis", mincnt=1)
        if row == 0: ax[row, 2].set_title("PC1–PC2 density")
        ax[row, 2].text(.95, .9, f"H{d['hopkins']:.2f} BC{d['bc_pc1']:.2f}/{d['bc_pc2']:.2f}",
                        transform=ax[row, 2].transAxes, ha="right", fontsize=7, color="w")
        ax[row, 2].text(.95, .05, f"ρ(PC,nobs){d['corr_pc1_nobs']:.2f}", transform=ax[row, 2].transAxes,
                        ha="right", fontsize=7, color="w")
    for a in ax[-1]: a.set_xlabel("")
    fig.suptitle(f"38-D feature space across 7 datasets — {rep}\n"
                 "(silhouette <0.5 ⇒ continuum; one blob ⇒ no discrete clusters; "
                 "ρ(PC,nobs) = sampling-density confound)", fontweight="bold", fontsize=12)
    plt.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close()


def _fig_summary(res, nobs, idxs):
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    x = np.arange(len(DATASETS)); w = 0.38
    # n_obs distributions
    ax[0, 0].boxplot([nobs[idxs[n]] for n in DATASETS], labels=DATASETS, showfliers=False)
    ax[0, 0].set_title("raw n_obs per dataset (sampling-density control)"); ax[0, 0].set_ylabel("n_obs")
    ax[0, 0].tick_params(axis="x", rotation=30)
    # effective dimensionality
    ax[0, 1].bar(x - w/2, [res["raw"][n]["n90"] for n in DATASETS], w, label="raw", color="#3a86ff")
    ax[0, 1].bar(x + w/2, [res["pchip100"][n]["n90"] for n in DATASETS], w, label="pchip100", color="#ff006e")
    ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(DATASETS, rotation=30); ax[0, 1].legend()
    ax[0, 1].set_title("effective dimensionality (#PCs for 90% var)"); ax[0, 1].set_ylabel("#PCs")
    # max silhouette
    ax[1, 0].bar(x - w/2, [res["raw"][n]["max_sil"] for n in DATASETS], w, label="raw", color="#3a86ff")
    ax[1, 0].bar(x + w/2, [res["pchip100"][n]["max_sil"] for n in DATASETS], w, label="pchip100", color="#ff006e")
    ax[1, 0].axhline(0.5, ls="--", c="k"); ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(DATASETS, rotation=30)
    ax[1, 0].set_title("max KMeans silhouette  (<0.5 = continuum, no discrete clusters)")
    ax[1, 0].set_ylabel("silhouette"); ax[1, 0].legend()
    # n_obs confound
    ax[1, 1].bar(x - w/2, [res["raw"][n]["corr_pc1_nobs"] for n in DATASETS], w, label="raw PC1", color="#3a86ff")
    ax[1, 1].bar(x + w/2, [res["pchip100"][n]["corr_pc1_nobs"] for n in DATASETS], w, label="pchip100 PC1", color="#ff006e")
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(DATASETS, rotation=30)
    ax[1, 1].set_title("|corr(PC1, n_obs)|  (sampling-density confound)"); ax[1, 1].legend()
    fig.suptitle("7-dataset feature-space comparison — raw vs PCHIP-100, with n_obs controls",
                 fontweight="bold", fontsize=13)
    plt.tight_layout(); fig.savefig(f"{FIGD}/fig_fs7_summary.png", dpi=140, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    main()
