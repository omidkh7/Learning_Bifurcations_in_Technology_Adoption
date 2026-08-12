#!/usr/bin/env python3
"""
check4b_ood.py — is the "everything is TC" verdict on real curves a CLASSIFICATION or an
EXTRAPOLATION?

Check 4 found the FeatMLP assigns TC to ~95% of real adoption curves with mean P(SN) ~ 0.01
and no correlation with PC1. Two very different explanations:

  (i)  real completed curves genuinely sit in the TC region of the synthetic feature space
       (the §72b "all completed adoptions are logistic-consistent" result), or
  (ii) real curves fall OUTSIDE the synthetic training distribution entirely, and the
       network is extrapolating — in which case the TC label is meaningless, not evidence.

We distinguish them by asking where the real 46-D vectors sit relative to the synthetic
training distribution, per feature (fraction outside the synthetic 1st-99th percentile)
and jointly (Mahalanobis distance vs the synthetic chi-square reference).

Writes runs/checks/check4b_ood.json + fig_check4b.png
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from features import FEAT_NAMES_46, FEAT_GROUPS_46

CDATA = os.path.join(HERE, "data", "checks")
CRUNS = os.path.join(HERE, "runs", "checks")
CLASSES = ["SN", "TC", "Null"]


def main():
    F = np.load(f"{CDATA}/F_g100.npy").astype(np.float64)      # synthetic, 100-grid
    y = np.load(f"{CDATA}/y_g100.npy").astype(int)
    from paper_figures import load_four_group
    grp, Fr, Xr = load_four_group()
    Fr = np.nan_to_num(Fr).astype(np.float64)
    print(f"synthetic {F.shape}, real {Fr.shape}")

    res = {}

    # ---- per-feature: fraction of real curves outside the synthetic 1st-99th pct ----
    lo = np.percentile(F, 1, axis=0); hi = np.percentile(F, 99, axis=0)
    out_mask = (Fr < lo) | (Fr > hi)                    # (n_real, 46)
    frac_feat = out_mask.mean(0)                        # per feature
    frac_curve = out_mask.mean(1)                       # per real curve
    res["per_feature_frac_outside"] = {FEAT_NAMES_46[j]: float(frac_feat[j]) for j in range(46)}
    res["mean_frac_features_outside_per_curve"] = float(frac_curve.mean())
    res["frac_real_curves_with_any_feature_outside"] = float((out_mask.any(1)).mean())
    res["frac_real_curves_over_25pct_features_outside"] = float((frac_curve > 0.25).mean())

    # ---- joint: Mahalanobis distance to the synthetic distribution ----
    # Fit on STANDARDISED features: the raw 46-D columns span many orders of magnitude, which
    # leaves the covariance ill-conditioned and collapses the distances to ~0.
    mu = F.mean(0)
    sd0 = F.std(0) + 1e-9
    Zs = (F - mu) / sd0
    Zrr = (Fr - mu) / sd0
    cov = LedoitWolf().fit(Zs)
    d_real = cov.mahalanobis(Zrr)          # squared Mahalanobis
    d_syn = cov.mahalanobis(Zs)
    res["mahalanobis_median_synth"] = float(np.median(d_syn))
    res["mahalanobis_median_real"] = float(np.median(d_real))
    res["mahalanobis_p99_synth"] = float(np.percentile(d_syn, 99))
    res["frac_real_beyond_synth_p99"] = float((d_real > np.percentile(d_syn, 99)).mean())

    # ---- distance to each class centroid (in standardised space) ----
    sd = F.std(0) + 1e-9
    Z = (F - mu) / sd; Zr = (Fr - mu) / sd
    cents = {c: Z[y == i].mean(0) for i, c in enumerate(CLASSES)}
    res["mean_dist_to_centroid"] = {}
    for c, ct in cents.items():
        dr = np.linalg.norm(Zr - ct, axis=1)
        ds = np.linalg.norm(Z[y == CLASSES.index(c)] - ct, axis=1)
        res["mean_dist_to_centroid"][c] = dict(real_median=float(np.median(dr)),
                                               synth_own_class_median=float(np.median(ds)))

    print("\n" + "=" * 70)
    print("CHECK 4b — are real curves inside the synthetic training distribution?")
    print(f"  mean fraction of the 46 features outside synth 1-99 pct, per real curve: "
          f"{res['mean_frac_features_outside_per_curve']*100:.1f}%")
    print(f"  real curves with ANY feature outside:        "
          f"{res['frac_real_curves_with_any_feature_outside']*100:.1f}%")
    print(f"  real curves with >25% of features outside:   "
          f"{res['frac_real_curves_over_25pct_features_outside']*100:.1f}%")
    print(f"  Mahalanobis median: synthetic {res['mahalanobis_median_synth']:.1f}  "
          f"real {res['mahalanobis_median_real']:.1f}   "
          f"(synth 99th pct = {res['mahalanobis_p99_synth']:.1f})")
    print(f"  real curves beyond the synthetic 99th pct:   "
          f"{res['frac_real_beyond_synth_p99']*100:.1f}%")
    print("\n  median distance to each synthetic class centroid (standardised units):")
    for c, d in res["mean_dist_to_centroid"].items():
        print(f"    {c:5s} real {d['real_median']:6.2f}   "
              f"(synthetic own-class {d['synth_own_class_median']:.2f})")

    worst = sorted(res["per_feature_frac_outside"].items(), key=lambda kv: -kv[1])[:10]
    print("\n  most out-of-range features on real data:")
    for n, v in worst:
        g = FEAT_GROUPS_46[FEAT_NAMES_46.index(n)]
        print(f"    {n:26s} [{g:10s}] {v*100:5.1f}% of real curves outside")

    ood = res["frac_real_beyond_synth_p99"]
    if ood > 0.5:
        v = (f"EXTRAPOLATION: {ood*100:.0f}% of real curves lie beyond the synthetic 99th "
             f"percentile in the 46-D space. The model has never seen data like this, so its "
             f"'TC' verdict on real curves is an extrapolation artifact, NOT evidence that "
             f"real adoptions are transcritical.")
    elif ood > 0.15:
        v = (f"PARTIAL EXTRAPOLATION: {ood*100:.0f}% of real curves are outside the synthetic "
             f"99th percentile; the real-data verdict is unreliable for a substantial subset.")
    else:
        v = (f"IN-DISTRIBUTION: only {ood*100:.0f}% of real curves are beyond the synthetic "
             f"99th percentile, so the TC verdict is a genuine classification, consistent with "
             f"§72b (completed curves are logistic-consistent).")
    res["verdict"] = v
    print("\nVERDICT:", v)
    json.dump(res, open(f"{CRUNS}/check4b_ood.json", "w"), indent=2)

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.logspace(np.log10(max(d_syn.min(), 1e-2)),
                       np.log10(max(d_real.max(), d_syn.max())), 50)
    ax1.hist(d_syn, bins=bins, alpha=0.6, color="#457b9d", density=True, label="synthetic")
    ax1.hist(d_real, bins=bins, alpha=0.6, color="#e63946", density=True, label="real")
    ax1.axvline(res["mahalanobis_p99_synth"], color="k", ls="--", lw=1,
                label="synthetic 99th pct")
    ax1.set_xscale("log")
    ax1.set_xlabel("Mahalanobis distance to the synthetic distribution")
    ax1.set_ylabel("density"); ax1.legend(fontsize=7.5)
    ax1.set_title("(a) are real curves inside the training distribution?", fontsize=9, loc="left")

    order = np.argsort(-frac_feat)[:14]
    ax2.barh(range(len(order)), frac_feat[order] * 100, color="#6a4c93", alpha=0.9)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([FEAT_NAMES_46[j] for j in order], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("% of real curves outside synthetic 1-99 pct")
    ax2.set_title("(b) which features are out of range on real data", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(f"{CRUNS}/fig_check4b.png", dpi=180, bbox_inches="tight")
    print(f"saved {CRUNS}/check4b_ood.json + fig_check4b.png")


if __name__ == "__main__":
    main()
