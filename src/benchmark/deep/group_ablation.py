#!/usr/bin/env python3
"""
group_ablation.py — which of the eight SI feature groups carries the FeatMLP's learning?

Three complementary measurements on the headline t50-matched dataset (same split, same
standardiser protocol as train.py):

  ALONE      : retrain FeatMLP on ONLY that group's columns (is the group sufficient?)
  DROP-ONE   : retrain FeatMLP with that group's columns removed (is it necessary, or
               is its information redundant with the rest?)
  PERMUTE    : take the full 46-D model and shuffle that group's columns at test time
               (how much does the trained model actually lean on it?)

Accuracies are matched-test, overall and SN/TC-only (the shortcut-protected decision).
Writes runs/group_ablation.json + runs/fig_group_ablation.png.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from models import FeatMLP
from features import FEAT_GROUPS_46, FEAT_NAMES_46
import train as T

DATA, RUNS = T.DATA, T.RUNS
SEED = 42
GROUPS = sorted(set(FEAT_GROUPS_46), key=FEAT_GROUPS_46.index)
GIDX = {g: [j for j in range(46) if FEAT_GROUPS_46[j] == g] for g in GROUPS}


def fit_mlp(Fz, y, itr, iva, tag):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model, _, _ = T.train_model(FeatMLP(n_in=Fz.shape[1]), f"ablation_{tag}", "mlp", 0.0,
                                (torch.from_numpy(Fz), torch.from_numpy(y)), itr, iva)
    model.eval()
    return model


def score(model, Fz, y, ite):
    with torch.no_grad():
        pred = model(torch.from_numpy(Fz[ite]).to(T.DEVICE)).argmax(1).cpu().numpy()
    yt = y[ite]
    overall = float((pred == yt).mean())
    m = yt < 2
    sntc = float((pred[m] == yt[m]).mean())
    return overall, sntc


def main():
    F = np.load(f"{DATA}/F_matched.npy").astype(np.float32)
    y = np.load(f"{DATA}/y_matched.npy").astype(np.int64)
    sp = np.load(f"{RUNS}/splits.npz")
    itr, iva, ite = sp["train"], sp["val"], sp["test"]
    mu, sd = F[itr].mean(0), F[itr].std(0) + 1e-9
    Fz = np.clip((F - mu) / sd, -8, 8).astype(np.float32)

    res = {"groups": GROUPS, "n_features": {g: len(GIDX[g]) for g in GROUPS}}

    full = fit_mlp(Fz, y, itr, iva, "full")
    res["full"] = dict(zip(("acc", "sn_tc_acc"), score(full, Fz, y, ite)))
    print(f"full 46-D: acc {res['full']['acc']*100:.1f}%  "
          f"SN/TC-only {res['full']['sn_tc_acc']*100:.1f}%")

    rng = np.random.default_rng(SEED)
    res["alone"], res["drop_one"], res["permute"] = {}, {}, {}
    for g in GROUPS:
        cols = GIDX[g]
        others = [j for j in range(46) if j not in cols]

        m_alone = fit_mlp(Fz[:, cols], y, itr, iva, f"alone_{g[0]}")
        res["alone"][g] = dict(zip(("acc", "sn_tc_acc"), score(m_alone, Fz[:, cols], y, ite)))

        m_drop = fit_mlp(Fz[:, others], y, itr, iva, f"drop_{g[0]}")
        res["drop_one"][g] = dict(zip(("acc", "sn_tc_acc"), score(m_drop, Fz[:, others], y, ite)))

        accs, sns = [], []
        for _ in range(5):
            Fp = Fz.copy()
            perm = rng.permutation(len(ite))
            for j in cols:
                Fp[ite, j] = Fz[ite[perm], j]
            a, s = score(full, Fp, y, ite)
            accs.append(a); sns.append(s)
        res["permute"][g] = dict(acc=float(np.mean(accs)), sn_tc_acc=float(np.mean(sns)))

        print(f"{g:12s} ({len(cols)} feats)  alone {res['alone'][g]['acc']*100:5.1f}%  "
              f"drop-one {res['drop_one'][g]['acc']*100:5.1f}%  "
              f"permuted {res['permute'][g]['acc']*100:5.1f}%   "
              f"[SN/TC alone {res['alone'][g]['sn_tc_acc']*100:5.1f}%]")

    with open(f"{RUNS}/group_ablation.json", "w") as f:
        json.dump(res, f, indent=2)

    # figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))
    x = np.arange(len(GROUPS))
    alone = [res["alone"][g]["acc"] * 100 for g in GROUPS]
    alone_sntc = [res["alone"][g]["sn_tc_acc"] * 100 for g in GROUPS]
    ax1.bar(x - 0.2, alone, 0.4, color="#2a9d8f", alpha=0.9, label="3-class")
    ax1.bar(x + 0.2, alone_sntc, 0.4, color="#457b9d", alpha=0.9, label="SN/TC-only")
    ax1.axhline(res["full"]["acc"] * 100, color="#d62828", lw=1.0, ls="--",
                label="full 46-D")
    ax1.axhline(100 / 3, color="k", lw=0.8, ls=":")
    ax1.text(len(GROUPS) - 0.4, 35, "chance", fontsize=6.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([g.split(":")[1] for g in GROUPS], rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("matched-test accuracy (%)")
    ax1.set_ylim(25, 103)
    ax1.legend(fontsize=7, loc="lower right")
    ax1.set_title("(a) each group ALONE (sufficiency)", fontsize=9, loc="left")
    for i, v in enumerate(alone):
        ax1.text(i - 0.2, v + 0.8, f"{v:.0f}", ha="center", fontsize=7)

    drop = [res["full"]["acc"] * 100 - res["drop_one"][g]["acc"] * 100 for g in GROUPS]
    perm = [res["full"]["acc"] * 100 - res["permute"][g]["acc"] * 100 for g in GROUPS]
    ax2.bar(x - 0.2, drop, 0.4, color="#e76f51", alpha=0.9, label="drop-one retrain")
    ax2.bar(x + 0.2, perm, 0.4, color="#6a4c93", alpha=0.9, label="permute at test")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([g.split(":")[1] for g in GROUPS], rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("accuracy DROP vs full 46-D (pts)")
    ax2.legend(fontsize=7)
    ax2.set_title("(b) necessity / reliance of the trained model", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(f"{RUNS}/fig_group_ablation.png", dpi=180, bbox_inches="tight")
    print(f"saved {RUNS}/group_ablation.json + fig_group_ablation.png")


if __name__ == "__main__":
    main()
