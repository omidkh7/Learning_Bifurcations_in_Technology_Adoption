#!/usr/bin/env python3
"""
null_sweep.py — null-class sensitivity for the best model (FeatMLP on the 46-D features),
mirroring the SI's null-sensitivity analysis (null_sensitivity.py / benchmark_data.py).

SN/TC are held FIXED: the t50-matched wide-parameter pairs from the headline experiment
(classes 0/1 of data/X_matched — equal SN/TC per t50 bin, so the inflection shortcut stays
dead for the SN-vs-TC decision). The NULL class is swapped through the SI's six
constructions:

  linear, exponential, ramp_expsat, sigmoid, logistic   (clean per-shape nulls, SI make_null)
  mixed                                                 (Option-C heterogeneous open-set null)

For each null kind a FeatMLP is trained from scratch (same split seed, same standardiser
protocol) and scored on its own test split. In addition, every trained model is scored
against every OTHER kind's test nulls (SN/TC test rows fixed) — the train-null x test-null
generalisation matrix, an open-set question the unsupervised SI analysis could not ask.

Nulls follow the SI construction exactly (fixed 0.03 noise amplitude, SI parameter ranges),
unlike the headline experiment's widened logistic null; numbers are therefore not directly
comparable to the headline table.

Writes: runs/null_sweep.json, runs/fig_null_sweep.png. Reads only; touches nothing outside
second_attempt_deep/.
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from benchmark_data import make_null, CLEAN_NULLS                    # SI null constructions
from features import build_features46
from models import FeatMLP
import train as T

DATA, RUNS = T.DATA, T.RUNS
NULLS = CLEAN_NULLS + ["mixed"]                                      # same order as the SI
GRID = 500
SEED = 42
CLASSES = ["SN", "TC", "Null"]


def gen_null(kind, n, rng):
    if kind == "mixed":
        X = np.load(os.path.join(ROOT, "data/synthetic_optionc/X_full.npy"))
        y = np.load(os.path.join(ROOT, "data/synthetic_optionc/y.npy"))
        nu = X[rng.choice(np.where(y == 2)[0], n, replace=False)]
    else:
        nu = make_null(kind, n, GRID, rng)
    nu = (nu - nu.min(1, keepdims=True)) / (np.ptp(nu, 1, keepdims=True) + 1e-12)
    return nu.astype(np.float32)


def main():
    # fixed SN/TC: the t50-matched pairs of the headline experiment
    Xm = np.load(f"{DATA}/X_matched.npy").astype(np.float32)
    Fm = np.load(f"{DATA}/F_matched.npy").astype(np.float32)
    ym = np.load(f"{DATA}/y_matched.npy").astype(int)
    sntc = ym < 2
    Xb, Fb, yb = Xm[sntc], Fm[sntc], ym[sntc]
    n_null = int((ym == 0).sum())                     # class-balanced with SN and TC
    print(f"fixed SN/TC: {len(yb)} matched series; nulls per kind: {n_null}")

    # nulls + features per kind (features are the slow part; do once)
    rng = np.random.default_rng(SEED)
    Fnull, Xnull = {}, {}
    for kind in NULLS:
        t0 = time.time()
        Xnull[kind] = gen_null(kind, n_null, rng)
        Fnull[kind] = np.nan_to_num(build_features46(Xnull[kind])).astype(np.float32)
        print(f"  {kind:12s} nulls + 46-D features in {time.time()-t0:.0f}s")

    results = {"nulls": NULLS, "per_kind": {}, "cross": {}}
    models, scalers, test_idx = {}, {}, {}

    for kind in NULLS:
        F = np.vstack([Fb, Fnull[kind]])
        y = np.concatenate([yb, np.full(n_null, 2)]).astype(np.int64)
        itr, iva, ite = T.make_splits(y, seed=SEED)
        mu, sd = F[itr].mean(0), F[itr].std(0) + 1e-9
        Fz = np.clip((F - mu) / sd, -8, 8).astype(np.float32)

        torch.manual_seed(SEED); np.random.seed(SEED)
        model, _, best_va = T.train_model(FeatMLP(), f"null_sweep_{kind}", "mlp", 0.0,
                                          (torch.from_numpy(Fz), torch.from_numpy(y)),
                                          itr, iva)
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(Fz[ite]).to(T.DEVICE)).argmax(1).cpu().numpy()
        yt = y[ite]
        acc = float((pred == yt).mean())
        rec = {c: float((pred[yt == i] == i).mean()) for i, c in enumerate(CLASSES)}
        sn_tc_mask = yt < 2
        sn_tc_acc = float((pred[sn_tc_mask] == yt[sn_tc_mask]).mean())
        results["per_kind"][kind] = dict(acc=acc, recall=rec, sn_tc_acc=sn_tc_acc,
                                         best_val=best_va)
        print(f"  => {kind:12s} test acc {acc*100:5.1f}%  "
              f"recall SN {rec['SN']*100:.0f} TC {rec['TC']*100:.0f} "
              f"Null {rec['Null']*100:.0f}  (SN/TC-only {sn_tc_acc*100:.1f}%)")
        models[kind] = model
        scalers[kind] = (mu, sd)
        test_idx[kind] = (itr, ite)

    # cross-null generalisation: model trained with kind A scored on kind B's test nulls
    # (SN/TC test rows fixed to the split of the TRAINING kind)
    for ka in NULLS:
        mu, sd = scalers[ka]
        itr, ite = test_idx[ka]
        row = {}
        for kb in NULLS:
            # rebuild the evaluation set: SN/TC test rows of ka's split + kb's test nulls
            F = np.vstack([Fb, Fnull[kb]])
            y = np.concatenate([yb, np.full(n_null, 2)]).astype(np.int64)
            Fz = np.clip((F - mu) / sd, -8, 8).astype(np.float32)
            with torch.no_grad():
                pred = models[ka](torch.from_numpy(Fz[ite]).to(T.DEVICE)).argmax(1).cpu().numpy()
            row[kb] = float((pred == y[ite]).mean())
        results["cross"][ka] = row

    with open(f"{RUNS}/null_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

    # console cross table
    print("\ncross-null generalisation (rows: trained-with, cols: tested-on, overall acc %):")
    print(f"{'':13s}" + "".join(f"{k:>12s}" for k in NULLS))
    for ka in NULLS:
        print(f"{ka:13s}" + "".join(f"{results['cross'][ka][kb]*100:11.1f} " for kb in NULLS))

    # figure: per-kind bars + cross matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.1),
                                   gridspec_kw=dict(width_ratios=[1, 1.15]))
    accs = [results["per_kind"][k]["acc"] * 100 for k in NULLS]
    snl = [results["per_kind"][k]["sn_tc_acc"] * 100 for k in NULLS]
    xpos = np.arange(len(NULLS))
    ax1.bar(xpos - 0.2, accs, 0.4, color="#2a9d8f", alpha=0.9, label="3-class acc")
    ax1.bar(xpos + 0.2, snl, 0.4, color="#457b9d", alpha=0.9, label="SN/TC-only acc (matched)")
    ax1.axhline(100 / 3, color="k", lw=0.8, ls=":")
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(NULLS, rotation=25, ha="right", fontsize=8)
    ax1.axvline(len(CLEAN_NULLS) - 0.5, color="#d62828", lw=1.0)
    ax1.text(len(CLEAN_NULLS) - 0.45, 40, "open-set", color="#d62828", fontsize=7, rotation=90)
    ax1.set_ylabel("FeatMLP test accuracy (%)")
    ax1.set_ylim(30, 102)
    ax1.legend(fontsize=7, loc="lower left")
    ax1.set_title("(a) accuracy by null construction", fontsize=9, loc="left")
    for i, v in enumerate(accs):
        ax1.text(i - 0.2, v + 0.7, f"{v:.0f}", ha="center", fontsize=7)

    M = np.array([[results["cross"][ka][kb] for kb in NULLS] for ka in NULLS]) * 100
    im = ax2.imshow(M, cmap="YlGnBu", vmin=33, vmax=100)
    for i in range(len(NULLS)):
        for j in range(len(NULLS)):
            ax2.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                     color="white" if M[i, j] > 85 else "#222")
    ax2.set_xticks(range(len(NULLS)))
    ax2.set_xticklabels(NULLS, rotation=25, ha="right", fontsize=8)
    ax2.set_yticks(range(len(NULLS)))
    ax2.set_yticklabels(NULLS, fontsize=8)
    ax2.set_xlabel("tested on null kind")
    ax2.set_ylabel("trained with null kind")
    ax2.set_title("(b) cross-null generalisation (overall acc %)", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"{RUNS}/fig_null_sweep.png", dpi=180, bbox_inches="tight")
    print(f"\nsaved {RUNS}/null_sweep.json + fig_null_sweep.png")


if __name__ == "__main__":
    main()
