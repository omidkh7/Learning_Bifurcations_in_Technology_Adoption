#!/usr/bin/env python3
"""
check4_real_data.py — apply the FeatMLP to REAL adoption curves: does the DL classifier
reproduce the CONTINUUM story (§74-77), or does it claim discrete types?

(Two-track typing is deliberately out of scope here — the question is only whether the DL
read-out of the real feature space is continuous or discrete.)

Domain handling. Real curves live on the 100-point grid, PCHIP-interpolated from ~10-40
annual points (compare_4groups.clean). The SI resamples synthetic curves to that same
100-point grid, and so do we: the model is trained on 100-grid synthetic curves, never on
500-grid ones, so the grid is not a confound. What remains is the physics gap that check 2
quantifies (real series carry observation noise, not process-noise CSD) — the two results
must be read together.

Diagnostics for continuum vs discrete:
  1. distribution of P(SN) on real curves vs on the synthetic test set (where discrete
     classes genuinely exist, so it is the reference for what "discrete" looks like)
  2. bimodality coefficient of P(SN); BC > 5/9 suggests bimodality
  3. confidence: fraction of curves with max class probability > 0.9
  4. SEED STABILITY: each curve is labelled by 5 independently trained models. Real types
     should be seed-stable; an arbitrary cut through a continuum flips boundary curves.
  5. P(SN) versus PC1 (the continuum axis of §76): smooth ramp = continuum, step = types.

Writes runs/checks/check4_real_data.json + fig_check4.png
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import skew, kurtosis, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from checks_common import gen_pool_meta, load_mixed_null
from data import match_t50_3class
from features import build_features46
from models import FeatMLP
import train as T

CDATA = os.path.join(ROOT, "data", "curated", "deep", "checks")
CRUNS = os.path.join(ROOT, "results", "benchmark", "checks")
os.makedirs(CDATA, exist_ok=True)
os.makedirs(CRUNS, exist_ok=True)

GRID = 100          # the real-data grid
N_SNTC = 12000
N_NULL = 10000
N_SEEDS = 5
CLASSES = ["SN", "TC", "Null"]
FAMCOL = {"Historical": "#9aa0a6", "Renewables": "#2a9d8f", "BEV": "#6a4c93", "CDR": "#fb8500"}


def bimodality(v):
    """Sarle's bimodality coefficient: (skew^2 + 1) / kurtosis. > 5/9 suggests bimodal."""
    v = np.asarray(v, float)
    n = len(v)
    g = skew(v); k = kurtosis(v, fisher=True)
    denom = k + 3 * (n - 1) ** 2 / max((n - 2) * (n - 3), 1)
    return float((g ** 2 + 1) / denom) if abs(denom) > 1e-9 else np.nan


def build_synth_100():
    if os.path.exists(f"{CDATA}/F_g100.npy"):
        print("100-grid synthetic cached")
        return
    t0 = time.time()
    print(f"generating 100-grid synthetic ({N_SNTC}/class SN,TC + mixed null) ...")
    Xsn, tsn, *_ = gen_pool_meta(0, N_SNTC, 801, grid=GRID)
    print(f"  SN done ({time.time()-t0:.0f}s)")
    Xtc, ttc, *_ = gen_pool_meta(1, N_SNTC, 802, grid=GRID)
    print(f"  TC done ({time.time()-t0:.0f}s)")
    Xnu, tnu = load_mixed_null(N_NULL, 803, grid=GRID)      # the SI heterogeneous null
    Xm, ym, tm = match_t50_3class([(Xsn, tsn), (Xtc, ttc), (Xnu, tnu)], seed=0)
    print(f"  matched N={len(ym)} {np.bincount(ym).tolist()}")
    F = build_features46(Xm).astype(np.float32)
    np.save(f"{CDATA}/F_g100.npy", F); np.save(f"{CDATA}/y_g100.npy", ym)
    print(f"cached ({time.time()-t0:.0f}s)")


def main():
    build_synth_100()
    F = np.load(f"{CDATA}/F_g100.npy").astype(np.float32)
    y = np.load(f"{CDATA}/y_g100.npy").astype(np.int64)

    from paper_figures import load_four_group
    grp, Fr, Xr = load_four_group()
    Fr = np.nan_to_num(Fr).astype(np.float32)
    print(f"real curves: {len(grp)}  " + str({g: int((grp == g).sum()) for g in np.unique(grp)}))

    res = {"n_real": int(len(grp)), "synth_matched_N": int(len(y)), "n_seeds": N_SEEDS,
           "groups": {g: int((grp == g).sum()) for g in np.unique(grp)}}

    Preal, Psyn, syn_acc = [], [], []
    for seed in range(N_SEEDS):
        itr, iva, ite = T.make_splits(y, seed=900 + seed)
        mu, sd = F[itr].mean(0), F[itr].std(0) + 1e-9
        Fz = np.clip((F - mu) / sd, -8, 8).astype(np.float32)
        Frz = np.clip((Fr - mu) / sd, -8, 8).astype(np.float32)
        torch.manual_seed(seed); np.random.seed(seed)
        m, _, _ = T.train_model(FeatMLP(), f"checks/c4_s{seed}", "mlp", 0.0,
                                (torch.from_numpy(Fz), torch.from_numpy(y)), itr, iva)
        m.eval()
        with torch.no_grad():
            ps = torch.softmax(m(torch.from_numpy(Fz[ite]).to(T.DEVICE)), 1).cpu().numpy()
            pr = torch.softmax(m(torch.from_numpy(Frz).to(T.DEVICE)), 1).cpu().numpy()
        acc = float((ps.argmax(1) == y[ite]).mean())
        syn_acc.append(acc); Psyn.append(ps); Preal.append(pr)
        print(f"  seed {seed}: synthetic(100-grid) test acc {acc*100:.1f}%")

    res["synth_test_acc"] = [float(np.mean(syn_acc)), float(np.std(syn_acc))]
    Pr = np.mean(Preal, axis=0)                      # seed-averaged real probabilities
    Ps = Psyn[0]
    ys_te = y[T.make_splits(y, seed=900)[2]]

    # ---- 1-3. distribution, bimodality, confidence ----
    psn_real = Pr[:, 0]
    psn_syn = Ps[ys_te < 2, 0]                       # synthetic SN/TC only, for a fair shape compare
    res["bimodality_psn_real"] = bimodality(psn_real)
    res["bimodality_psn_synth"] = bimodality(psn_syn)
    res["conf_real_frac_gt0.9"] = float((Pr.max(1) > 0.9).mean())
    res["conf_synth_frac_gt0.9"] = float((Ps.max(1) > 0.9).mean())

    # ---- 4. seed stability ----
    lab_real = np.array([p.argmax(1) for p in Preal])          # (seeds, n_real)
    stable_real = float((lab_real == lab_real[0]).all(0).mean())
    lab_syn = np.array([p.argmax(1) for p in Psyn])
    stable_syn = float((lab_syn == lab_syn[0]).all(0).mean())
    res["seed_stable_frac_real"] = stable_real
    res["seed_stable_frac_synth"] = stable_syn

    # ---- 5. P(SN) vs PC1 ----
    Zr = StandardScaler().fit_transform(Fr)
    pca = PCA(n_components=3).fit(Zr)
    PCr = pca.transform(Zr)
    rho = spearmanr(PCr[:, 0], psn_real).statistic
    res["spearman_pc1_psn"] = float(rho)
    res["pc_var"] = [float(v) for v in pca.explained_variance_ratio_]

    # ---- per-group type fractions ----
    lab = Pr.argmax(1)
    res["group_fractions"] = {}
    for g in np.unique(grp):
        mgrp = grp == g
        res["group_fractions"][g] = {c: float((lab[mgrp] == i).mean()) for i, c in enumerate(CLASSES)}
        res["group_fractions"][g]["mean_P(SN)"] = float(psn_real[mgrp].mean())

    print("\n" + "=" * 70)
    print("CHECK 4 — DL classifier on real adoption curves (continuum test)")
    print(f"  synthetic (100-grid) test acc     {res['synth_test_acc'][0]*100:.1f} "
          f"+/- {res['synth_test_acc'][1]*100:.1f}%")
    print(f"  bimodality of P(SN): synthetic {res['bimodality_psn_synth']:.3f}  "
          f"real {res['bimodality_psn_real']:.3f}   (>0.555 = bimodal)")
    print(f"  confident (max p > 0.9): synthetic {res['conf_synth_frac_gt0.9']*100:.0f}%  "
          f"real {res['conf_real_frac_gt0.9']*100:.0f}%")
    print(f"  seed-stable labels: synthetic {stable_syn*100:.0f}%  real {stable_real*100:.0f}%")
    print(f"  Spearman(PC1, P(SN)) on real = {rho:+.2f}")
    print("\n  per-family:")
    for g, d in res["group_fractions"].items():
        print(f"    {g:12s} SN {d['SN']*100:5.1f}%  TC {d['TC']*100:5.1f}%  "
              f"Null {d['Null']*100:5.1f}%   mean P(SN) {d['mean_P(SN)']:.2f}")

    json.dump(res, open(f"{CRUNS}/check4_real_data.json", "w"), indent=2)

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    ax = axes[0]
    ax.hist(psn_syn, bins=30, alpha=0.65, color="#457b9d", density=True,
            label=f"synthetic (BC {res['bimodality_psn_synth']:.2f})")
    ax.hist(psn_real, bins=30, alpha=0.65, color="#e63946", density=True,
            label=f"real (BC {res['bimodality_psn_real']:.2f})")
    ax.set_xlabel("P(SN)"); ax.set_ylabel("density")
    ax.legend(fontsize=7)
    ax.set_title("(a) P(SN): synthetic (true types) vs real", fontsize=9, loc="left")

    ax = axes[1]
    for g in np.unique(grp):
        mgrp = grp == g
        ax.scatter(PCr[mgrp, 0], psn_real[mgrp], s=14, alpha=0.7,
                   color=FAMCOL.get(g, "#888"), label=f"{g} (n={mgrp.sum()})")
    ax.set_xlabel("PC1 of the real 46-D feature space")
    ax.set_ylabel("P(SN)")
    ax.legend(fontsize=6.5)
    ax.set_title(f"(b) P(SN) vs continuum axis, rho = {rho:+.2f}", fontsize=9, loc="left")

    ax = axes[2]
    labels = ["synthetic", "real"]
    conf = [res["conf_synth_frac_gt0.9"] * 100, res["conf_real_frac_gt0.9"] * 100]
    stab = [stable_syn * 100, stable_real * 100]
    xp = np.arange(2)
    ax.bar(xp - 0.2, conf, 0.4, color="#2a9d8f", alpha=0.9, label="confident (max p>0.9)")
    ax.bar(xp + 0.2, stab, 0.4, color="#6a4c93", alpha=0.9, label="seed-stable label")
    ax.set_xticks(xp); ax.set_xticklabels(labels)
    ax.set_ylabel("% of curves"); ax.set_ylim(0, 105)
    ax.legend(fontsize=7.5)
    ax.set_title("(c) decisiveness and stability", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(f"{CRUNS}/fig_check4.png", dpi=180, bbox_inches="tight")
    print(f"\nsaved {CRUNS}/check4_real_data.json + fig_check4.png")


if __name__ == "__main__":
    main()
