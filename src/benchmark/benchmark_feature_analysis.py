#!/usr/bin/env python3
"""
benchmark_feature_analysis.py
=============================
Family/class-by-class feature analysis on the SYNTHETIC BENCHMARK (ground-truth SN/TC/Null).
Answers: which feature GROUPS actually carry the class signal, do the SN- and TC-fingerprint groups
fire on their intended class (validity check on the feature design), and how much does each group
contribute (leave-one-group-out). Uses the BALANCED 43-D space (38 + Group G TC-fingerprint).

Output: Manuscript/SI_figures/figS_feature_analysis.png  + printed summary.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from paper_style import set_style, COL2
set_style()
from paper_figures import _tag, GROUP, ALL_NAMES, build_features
from benchmark_feature_spaces import load_synth

SI = "Manuscript/SI_figures"; os.makedirs(SI, exist_ok=True)
LAB = ["SN", "TC", "Null"]; CCOL = {0: "#d62828", 1: "#457b9d", 2: "#9aa0a6"}
NAMES = list(ALL_NAMES)                                             # canonical 46-D names
GROUPS = list(GROUP)                                               # canonical 46-D group labels
GORDER = ["A:CSD", "B:Inflect", "C:Phase", "D:Catch22", "E:SN-fp", "F:Transit", "G:TC-fp", "H:Crit"]
GCOL = {"A:CSD": "#1d4e89", "B:Inflect": "#2a9d8f", "C:Phase": "#e07a5f", "D:Catch22": "#9b5de5",
        "E:SN-fp": "#d62828", "F:Transit": "#bc6c25", "G:TC-fp": "#457b9d", "H:Crit": "#6a4c93"}


def eta2(col, ys):
    gm = col.mean()
    ssb = sum((ys == c).sum() * (col[ys == c].mean() - gm) ** 2 for c in range(3))
    sst = ((col - gm) ** 2).sum()
    return ssb / (sst + 1e-12)


def nc_acc(Z, ys, cols=None):
    Zs = Z[:, cols] if cols is not None else Z
    cents = np.vstack([Zs[ys == c].mean(0) for c in range(3)])
    d = np.linalg.norm(Zs[:, None, :] - cents[None], axis=2)
    return (d.argmin(1) == ys).mean()


def run(n_per=1000, seed=0):
    Xn, ys = load_synth(n_per, seed)
    F = np.nan_to_num(build_features(Xn))                            # canonical 46-D
    Z = StandardScaler().fit_transform(F)

    e2 = np.array([eta2(Z[:, j], ys) for j in range(Z.shape[1])])
    grp_eta = {g: e2[[i for i, gg in enumerate(GROUPS) if gg == g]].mean() for g in GORDER}
    # per-class z-scored mean signature (positive = high for that class)
    sig = np.stack([Z[ys == c].mean(0) for c in range(3)], 1)            # (43, 3)
    # leave-one-group-out nearest-centroid accuracy
    full = nc_acc(Z, ys)
    logo = {g: nc_acc(Z, ys, cols=[i for i, gg in enumerate(GROUPS) if gg != g]) for g in GORDER}

    print(f"full {Z.shape[1]}-D nearest-centroid acc = {100*full:.0f}%")
    print("group | mean eta^2 | acc if dropped (delta)")
    for g in GORDER:
        print(f"  {g:10s} {grp_eta[g]:.3f}   {100*logo[g]:.0f}% ({100*(logo[g]-full):+.0f})")
    # validate fingerprints: which class does each designed group fire on?
    for g in ("E:SN-fp", "G:TC-fp", "H:Crit"):
        gi = [i for i, gg in enumerate(GROUPS) if gg == g]
        print(f"{g:9s} |mean| signal: SN={np.abs(sig[gi,0]).mean():.2f} TC={np.abs(sig[gi,1]).mean():.2f} "
              f"Null={np.abs(sig[gi,2]).mean():.2f}")

    # ---- figure ----
    fig = plt.figure(figsize=(COL2, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], width_ratios=[1, 1.15], hspace=0.4, wspace=0.35)

    # (a) per-group discriminability
    axA = fig.add_subplot(gs[0, 0])
    axA.bar(range(len(GORDER)), [grp_eta[g] for g in GORDER], color=[GCOL[g] for g in GORDER], alpha=0.9)
    axA.set_xticks(range(len(GORDER))); axA.set_xticklabels([g.split(":")[1] for g in GORDER], fontsize=5.8, rotation=25, ha="right")
    axA.set_ylabel(r"mean $\eta^2$ (class sep.)", fontsize=7); _tag(axA, "(a) group discriminability")

    # (b) leave-one-group-out accuracy drop
    axB = fig.add_subplot(gs[0, 1])
    drops = [100 * (logo[g] - full) for g in GORDER]
    axB.bar(range(len(GORDER)), drops, color=[GCOL[g] for g in GORDER], alpha=0.9)
    axB.axhline(0, color="#333", lw=0.6)
    axB.set_xticks(range(len(GORDER))); axB.set_xticklabels([g.split(":")[1] for g in GORDER], fontsize=5.8, rotation=25, ha="right")
    axB.set_ylabel("acc change if dropped (pts)", fontsize=7)
    _tag(axB, f"(b) leave-one-group-out (full {100*full:.0f}%)")

    # (c) per-class signature heatmap for the fingerprint / physics groups E, F, G, H
    axC = fig.add_subplot(gs[1, :])
    hgroups = ("E:SN-fp", "F:Transit", "G:TC-fp", "H:Crit")
    sel = [i for i, g in enumerate(GROUPS) if g in hgroups]
    M = sig[sel]                                                     # (nsel, 3)
    im = axC.imshow(M.T, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
    axC.set_yticks(range(3)); axC.set_yticklabels(LAB, fontsize=7)
    axC.set_xticks(range(len(sel))); axC.set_xticklabels([NAMES[i] for i in sel], fontsize=4.8, rotation=42, ha="right")
    # group dividers / labels
    bounds = []
    for g in hgroups:
        gi = [k for k, i in enumerate(sel) if GROUPS[i] == g]
        bounds.append((min(gi), max(gi), g))
    for lo, hi, g in bounds:
        if lo > 0: axC.axvline(lo - 0.5, color="k", lw=0.8)
        axC.text((lo + hi) / 2, 2.75, g.split(":")[1], ha="center", fontsize=6.2, color=GCOL[g], fontweight="bold")
    cb = fig.colorbar(im, ax=axC, fraction=0.025, pad=0.02); cb.ax.tick_params(labelsize=5.5)
    cb.set_label("z-scored class mean", fontsize=6)
    axC.set_title("(c) fingerprint validity: SN-fp fires strongly on SN; TC-fp fires weakly on TC",
                  fontsize=7.5, loc="left", pad=4)

    fig.savefig(f"{SI}/figS_feature_analysis.png", bbox_inches="tight"); plt.close(fig)
    print("saved figS_feature_analysis.png")


if __name__ == "__main__":
    run()
