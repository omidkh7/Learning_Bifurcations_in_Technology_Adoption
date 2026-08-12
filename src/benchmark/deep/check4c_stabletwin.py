#!/usr/bin/env python3
"""
check4c_stabletwin.py — re-run check 4 (real-data continuum test) but TRAIN on the paper's
CURRENT canonical benchmark (stable-twin null), to see whether the new null/TC class
definitions change the real-data verdict.

Difference from check4_real_data.py: training data now comes from
benchmark_data.load_benchmark (NULL_VERSION = 2) — SN/TC from data/synthetic_optionc and
the stable-twin logistic null, all on the 100-point grid — instead of the earlier
Option-C open-set null. Everything downstream (paper build_features, 5-seed FeatMLP,
real four-family curves, continuum diagnostics) is identical, so any change is attributable
to the null construction.

Writes runs/checks/check4c_stabletwin.json + fig_check4c.png
"""
import os, sys, json, warnings
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
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
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from benchmark_data import load_benchmark, NULL_VERSION
from paper_figures import build_features
from models import FeatMLP
import train as T

CRUNS = os.path.join(HERE, "runs", "checks")
os.makedirs(CRUNS, exist_ok=True)
N_PER = 3000
N_SEEDS = 5
CLASSES = ["SN", "TC", "Null"]
FAMCOL = {"Historical": "#9aa0a6", "Renewables": "#2a9d8f", "BEV": "#6a4c93", "CDR": "#fb8500"}


def bimodality(v):
    v = np.asarray(v, float); n = len(v)
    g = skew(v); k = kurtosis(v, fisher=True)
    denom = k + 3 * (n - 1) ** 2 / max((n - 2) * (n - 3), 1)
    return float((g ** 2 + 1) / denom) if abs(denom) > 1e-9 else np.nan


def main():
    # paper canonical benchmark (stable-twin logistic null), 100-grid
    Xb, yb = load_benchmark("logistic", N_PER, seed=0)
    Fb = np.nan_to_num(build_features(Xb)).astype(np.float32)
    y = yb.astype(np.int64)
    print(f"training benchmark (stable-twin, NULL_VERSION={NULL_VERSION}): "
          f"F {Fb.shape} per class {np.bincount(y).tolist()}")

    from paper_figures import load_four_group
    grp, Fr, Xr = load_four_group()
    Fr = np.nan_to_num(Fr).astype(np.float32)
    print(f"real curves: {len(grp)} " + str({g: int((grp == g).sum()) for g in np.unique(grp)}))

    res = {"null_version": NULL_VERSION, "trained_null": "stable-twin logistic",
           "n_real": int(len(grp)), "n_seeds": N_SEEDS,
           "groups": {g: int((grp == g).sum()) for g in np.unique(grp)}}

    Preal, Psyn, syn_acc = [], [], []
    for seed in range(N_SEEDS):
        itr, iva, ite = T.make_splits(y, seed=1400 + seed)
        mu, sd = Fb[itr].mean(0), Fb[itr].std(0) + 1e-9
        Fz = np.clip((Fb - mu) / sd, -8, 8).astype(np.float32)
        Frz = np.clip((Fr - mu) / sd, -8, 8).astype(np.float32)
        torch.manual_seed(seed); np.random.seed(seed)
        m, _, _ = T.train_model(FeatMLP(), f"checks/c4c_s{seed}", "mlp", 0.0,
                                (torch.from_numpy(Fz), torch.from_numpy(y)), itr, iva)
        m.eval()
        with torch.no_grad():
            ps = torch.softmax(m(torch.from_numpy(Fz[ite]).to(T.DEVICE)), 1).cpu().numpy()
            pr = torch.softmax(m(torch.from_numpy(Frz).to(T.DEVICE)), 1).cpu().numpy()
        syn_acc.append(float((ps.argmax(1) == y[ite]).mean()))
        Psyn.append(ps); Preal.append(pr)
        print(f"  seed {seed}: synthetic test acc {syn_acc[-1]*100:.1f}%")

    res["synth_test_acc"] = [float(np.mean(syn_acc)), float(np.std(syn_acc))]
    Pr = np.mean(Preal, axis=0)
    psn_real = Pr[:, 0]
    ys_te = y[T.make_splits(y, seed=1400)[2]]
    psn_syn = Psyn[0][ys_te < 2, 0]

    res["bimodality_psn_real"] = bimodality(psn_real)
    res["conf_real_frac_gt0.9"] = float((Pr.max(1) > 0.9).mean())
    lab_real = np.array([p.argmax(1) for p in Preal])
    res["seed_stable_frac_real"] = float((lab_real == lab_real[0]).all(0).mean())

    Zr = StandardScaler().fit_transform(Fr)
    PCr = PCA(n_components=3).fit_transform(Zr)
    res["spearman_pc1_psn"] = float(spearmanr(PCr[:, 0], psn_real).statistic)

    lab = Pr.argmax(1)
    res["group_fractions"] = {}
    for g in np.unique(grp):
        mg = grp == g
        res["group_fractions"][g] = {c: float((lab[mg] == i).mean()) for i, c in enumerate(CLASSES)}
        res["group_fractions"][g]["mean_P(SN)"] = float(psn_real[mg].mean())

    print("\n" + "=" * 70)
    print("CHECK 4c — real-data continuum test, TRAINED ON PAPER STABLE-TWIN NULL")
    print(f"  synthetic test acc {res['synth_test_acc'][0]*100:.1f} +/- {res['synth_test_acc'][1]*100:.1f}%")
    print(f"  bimodality of P(SN) on real: {res['bimodality_psn_real']:.3f}  (>0.555=bimodal)")
    print(f"  seed-stable labels on real:  {res['seed_stable_frac_real']*100:.0f}%")
    print(f"  Spearman(PC1, P(SN)) on real = {res['spearman_pc1_psn']:+.2f}")
    print("  per-family:")
    for g, d in res["group_fractions"].items():
        print(f"    {g:12s} SN {d['SN']*100:5.1f}%  TC {d['TC']*100:5.1f}%  "
              f"Null {d['Null']*100:5.1f}%   mean P(SN) {d['mean_P(SN)']:.2f}")

    # compare to the earlier (Option-C null) check 4, if present
    prev = os.path.join(CRUNS, "check4_real_data.json")
    if os.path.exists(prev):
        p = json.load(open(prev))
        res["compare_prev_optionC_null"] = {
            "spearman_pc1_psn": p.get("spearman_pc1_psn"),
            "mean_P(SN)_by_family": {g: p["group_fractions"][g]["mean_P(SN)"]
                                     for g in p.get("group_fractions", {})}}
        print(f"\n  [prev Option-C null] Spearman(PC1,P(SN)) = {p.get('spearman_pc1_psn'):+.2f}; "
              f"mean P(SN) all ~ "
              f"{np.mean([p['group_fractions'][g]['mean_P(SN)'] for g in p['group_fractions']]):.2f}")

    json.dump(res, open(f"{CRUNS}/check4c_stabletwin.json", "w"), indent=2)

    # figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))
    ax1.hist(psn_syn, bins=30, alpha=0.65, color="#457b9d", density=True, label="synthetic (true types)")
    ax1.hist(psn_real, bins=30, alpha=0.65, color="#e63946", density=True,
             label=f"real (BC {res['bimodality_psn_real']:.2f})")
    ax1.set_xlabel("P(SN)"); ax1.set_ylabel("density"); ax1.legend(fontsize=7.5)
    ax1.set_title("(a) P(SN): stable-twin-trained model", fontsize=9, loc="left")
    for g in np.unique(grp):
        mg = grp == g
        ax2.scatter(PCr[mg, 0], psn_real[mg], s=14, alpha=0.7, color=FAMCOL.get(g, "#888"),
                    label=f"{g} (n={mg.sum()})")
    ax2.set_xlabel("PC1 of real 46-D feature space"); ax2.set_ylabel("P(SN)")
    ax2.legend(fontsize=6.5)
    ax2.set_title(f"(b) P(SN) vs continuum axis, rho={res['spearman_pc1_psn']:+.2f}",
                  fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(f"{CRUNS}/fig_check4c.png", dpi=180, bbox_inches="tight")
    print(f"\nsaved {CRUNS}/check4c_stabletwin.json + fig_check4c.png")


if __name__ == "__main__":
    main()
