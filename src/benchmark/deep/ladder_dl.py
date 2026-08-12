#!/usr/bin/env python3
"""
ladder_dl.py — the supervised FeatMLP on every stage of the null-construction LADDER.

Complements null_ladder.py (unsupervised mixtures + oracle across the four null
constructions) with the supervised upper bound at each stage, so the ladder figure
shows how much each closed loophole costs the label-free methods versus the
supervised ceiling. Data exactly matches null_ladder.py: load_benchmark("logistic",
800/class, seed=0, null_version=stage), the paper's 46-D features, 5 seeds, 70/15/15.

Run from the repo root. Output: runs/unsup/bifurcation_explore/null_ladder_dl.csv
(read by null_ladder.py's draw() to overlay the FeatMLP lines on figS_null_ladder).
"""
import os, sys, warnings
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from benchmark_data import load_benchmark, LADDER
from paper_figures import build_features
from models import FeatMLP
import train as T

N_PER, N_SEEDS = 800, 5                     # N_PER matches null_ladder.py exactly
os.makedirs(os.path.join(ROOT, "results", "benchmark", "ladder"), exist_ok=True)


def main():
    rows = []
    for nv in LADDER:
        Xn, ys = load_benchmark("logistic", N_PER, seed=0, null_version=nv)
        F = np.nan_to_num(build_features(Xn)).astype(np.float32)
        y = ys.astype(np.int64)
        accs, tcnulls = [], []
        for s in range(N_SEEDS):
            itr, iva, ite = T.make_splits(y, seed=1000 + s)
            mu, sd = F[itr].mean(0), F[itr].std(0) + 1e-9
            Fz = np.clip((F - mu) / sd, -8, 8).astype(np.float32)
            torch.manual_seed(s); np.random.seed(s)
            m, _, _ = T.train_model(FeatMLP(), f"ladder/{nv}_s{s}", "mlp", 0.0,
                                    (torch.from_numpy(Fz), torch.from_numpy(y)), itr, iva)
            m.eval()
            with torch.no_grad():
                pred = m(torch.from_numpy(Fz[ite]).to(T.DEVICE)).argmax(1).cpu().numpy()
            yt = y[ite]
            M = np.array([[((yt == tc) & (pred == pc)).sum() / max((yt == tc).sum(), 1)
                           for pc in range(3)] for tc in range(3)])
            accs.append((pred == yt).mean()); tcnulls.append(M[1, 2] + M[2, 1])
        rows.append(dict(stage=str(nv), acc_dl=100 * np.mean(accs), acc_sd=100 * np.std(accs),
                         tcnull_dl=100 * np.mean(tcnulls)))
        print(f"{str(nv):11s}: FeatMLP {rows[-1]['acc_dl']:5.1f} +/- {rows[-1]['acc_sd']:.1f}%  "
              f"TC<->null {rows[-1]['tcnull_dl']:5.1f}", flush=True)
    out = os.path.join(ROOT, "results", "unsup", "bifurcation_explore", "null_ladder_dl.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
