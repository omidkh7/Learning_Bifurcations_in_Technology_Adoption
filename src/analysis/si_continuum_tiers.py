#!/usr/bin/env python3
"""SI Fig. S9 (figS:tiers): the continuum is robust to the curation choice. For each curated dataset
tier, independent PCA on the 46-D feature space, then three per-tier panels:
  (a) clustering silhouette at k=2,3,4 (all below the 0.5 well-separated threshold),
  (b) Hartigan dip-test p-value on PC1/PC2/PC3 (all above 0.05: no component is multimodal),
  (c) the PC1 score distribution (a single skewed peak).
Writes figures/si/figS_continuum_tiers.png. Run from the repo root."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import gaussian_kde
import diptest
from paper_figures import build_features, _pca_sign_fix
from feature_space_7datasets import load_pkl, recon_raw, to_grid

TIERS = ["combined", "filter10", "genuine_v1", "genuine_v2", "hardexcl", "s002", "s005"]
LAB = {"genuine_v2": "genuine_v2\n(main)"}
KS = [2, 3, 4]
KCOL = {2: "#1d4e89", 3: "#4c8fb0", 4: "#9dc3d8"}
DIP_THR = 0.05          # dip test: p > 0.05 = cannot reject unimodality
SIL_THR = 0.5           # conventional "well-separated" clustering
OUT = "figures/si/figS_continuum_tiers.png"

# ---- 46-D features on the combined base once, then subset per tier ----
base = load_pkl("combined"); rid = [s.row_id for s in base]
raws, keep = [], []
for i, s in enumerate(base):
    r = recon_raw(s.x_full, s.years_full)
    if r is not None: raws.append(to_grid(r, 100)); keep.append(rid[i])
F = build_features(np.vstack(raws)); base_idx = {r: j for j, r in enumerate(keep)}

q = pd.read_csv("results/unsup/shape_diagnostic/scurve_quality.csv"); ne = ~q.excluded
gv2 = load_pkl("combined_genuine_v2"); sub = lambda m: [gv2[i].row_id for i in m if i < len(gv2)]
sets = {"combined": rid, "filter10": [s.row_id for s in load_pkl("combined_filter10")],
        "genuine_v1": [s.row_id for s in load_pkl("combined_genuine")], "genuine_v2": [s.row_id for s in gv2],
        "hardexcl": sub(q.loc[ne, "idx"].astype(int).values),
        "s002": sub(q.loc[ne & (q.s_score >= 0.02), "idx"].astype(int).values),
        "s005": sub(q.loc[ne & (q.s_score >= 0.05), "idx"].astype(int).values)}

sil, dip, sc1, Ns = {}, {}, {}, {}
rng = np.random.default_rng(0)
for t in TIERS:
    ix = np.array([base_idx[r] for r in sets[t] if r in base_idx]); Ns[t] = len(ix)
    Z = StandardScaler().fit_transform(np.nan_to_num(F[ix]))
    smp = rng.choice(len(Z), min(2000, len(Z)), replace=False)
    sil[t] = [silhouette_score(Z[smp], KMeans(k, n_init=6, random_state=0).fit(Z).labels_[smp]) for k in KS]
    p = PCA(3).fit(Z); S = p.transform(Z); C = p.components_.copy(); _pca_sign_fix(C, S)
    dip[t] = [diptest.diptest(S[:, k])[1] for k in range(3)]; sc1[t] = S[:, 0]

xl = [LAB.get(t, t) for t in TIERS]; x = np.arange(len(TIERS))
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw=dict(wspace=0.34, width_ratios=[1.1, 1.05, 1.15]))

# (a) silhouette at k=2,3,4 per tier
w = 0.26
for j, k in enumerate(KS):
    axs[0].bar(x + (j - 1) * w, [sil[t][j] for t in TIERS], w, color=KCOL[k], label=f"k={k}")
axs[0].axhline(SIL_THR, ls="--", color="#d62828", lw=1.1)
axs[0].text(len(TIERS) - 0.5, SIL_THR + 0.01, "well-separated (0.5)", ha="right", fontsize=6.5, color="#d62828")
axs[0].set_xticks(x); axs[0].set_xticklabels(xl, rotation=35, ha="right", fontsize=6.5)
axs[0].set_ylim(0, 0.6); axs[0].set_ylabel("clustering silhouette", fontsize=8)
axs[0].legend(fontsize=7, loc="upper left", ncol=3, columnspacing=1.0, handlelength=1.1)
axs[0].set_title("(a) no clustering is well-separated ($k=2,3,4$)", fontsize=8.5)

# (b) dip test p-value on PC1/PC2/PC3 per tier
mk = {0: ("o", "#6a4c93", "PC1"), 1: ("s", "#2a9d8f", "PC2"), 2: ("^", "#e07a5f", "PC3")}
for k in range(3):
    axs[1].plot(x, [dip[t][k] for t in TIERS], mk[k][0] + "-", color=mk[k][1], lw=1.2, ms=5, label=mk[k][2])
axs[1].axhline(DIP_THR, ls="--", color="#d62828", lw=1.1)
axs[1].text(len(TIERS) - 0.5, DIP_THR + 0.02, "reject unimodality below p=0.05", ha="right", fontsize=6.5, color="#d62828")
axs[1].set_xticks(x); axs[1].set_xticklabels(xl, rotation=35, ha="right", fontsize=6.5)
axs[1].set_ylim(0, 1.05); axs[1].set_ylabel("Hartigan dip test p-value", fontsize=8)
axs[1].legend(fontsize=6.5, loc="lower left", ncol=3, columnspacing=1.0)
axs[1].set_title("(b) every component stays unimodal", fontsize=8.5)

# (c) PC1 distribution per tier (ridgeline)
xs = np.linspace(-6, 6, 300)
for r, t in enumerate(TIERS):
    s = sc1[t]; s = (s - s.mean()) / (s.std() + 1e-9); d = gaussian_kde(s)(xs); d /= d.max()
    axs[2].fill_between(xs, r + d, r, color="#6a4c93", alpha=0.28, lw=0); axs[2].plot(xs, r + d, color="#4a2c73", lw=1.0)
    axs[2].text(-5.8, r + 0.12, f"{t} (n={Ns[t]})", fontsize=6)
axs[2].set_yticks([]); axs[2].set_xlabel("PC1 score (standardized)", fontsize=8)
axs[2].set_title("(c) PC1 distribution, every tier", fontsize=8.5)

fig.savefig(OUT, dpi=160, bbox_inches="tight"); plt.close(fig)
print("saved", OUT)
print("silhouette k=2,3,4 per tier:", {t: [round(v, 2) for v in sil[t]] for t in TIERS})
print("min dip p (any PC any tier):", round(min(min(v) for v in dip.values()), 3))
