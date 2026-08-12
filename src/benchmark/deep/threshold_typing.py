#!/usr/bin/env python3
"""
threshold_typing.py — real-data typing with an abstention rule: call a curve only if the
seed-averaged max class probability exceeds a threshold, else "undecided".

Trains the FeatMLP (5 seeds) on the paper's stable-twin benchmark AND, for contrast, the
old Option-C open-set null, applies to the 478 real four-family curves, seed-averages the
softmax, and reports the decided/undecided split at thresholds 0.7 and 0.8.
"""
import os, sys, warnings
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from benchmark_data import load_benchmark
from paper_figures import build_features, load_four_group
from models import FeatMLP
import train as T

N_PER, N_SEEDS = 3000, 5
CLASSES = ["SN", "TC", "Null"]


def real_probs(null_kind):
    Xb, y = load_benchmark(null_kind, N_PER, seed=0)
    Fb = np.nan_to_num(build_features(Xb)).astype(np.float32); y = y.astype(np.int64)
    grp, Fr, _ = load_four_group(); Fr = np.nan_to_num(Fr).astype(np.float32)
    P = []
    for s in range(N_SEEDS):
        itr, iva, _ = T.make_splits(y, seed=1400 + s)
        mu, sd = Fb[itr].mean(0), Fb[itr].std(0) + 1e-9
        Fz = np.clip((Fb - mu) / sd, -8, 8).astype(np.float32)
        Frz = np.clip((Fr - mu) / sd, -8, 8).astype(np.float32)
        torch.manual_seed(s); np.random.seed(s)
        m, _, _ = T.train_model(FeatMLP(), f"thr/{null_kind}_s{s}", "mlp", 0.0,
                                (torch.from_numpy(Fz), torch.from_numpy(y)), itr, iva)
        m.eval()
        with torch.no_grad():
            P.append(torch.softmax(m(torch.from_numpy(Frz).to(T.DEVICE)), 1).cpu().numpy())
    return grp, np.mean(P, axis=0)


def report(grp, P, null_kind):
    lab = P.argmax(1); mx = P.max(1)
    print(f"\n{'='*64}\n{null_kind} null — argmax label distribution (no threshold):")
    print("  overall: " + ", ".join(f"{c} {int((lab==i).sum())}" for i, c in enumerate(CLASSES)))
    for thr in (0.7, 0.8):
        dec = mx > thr
        print(f"\n  threshold max-prob > {thr}:  decided {dec.sum()}/{len(dec)} "
              f"({dec.mean()*100:.0f}%),  undecided {(~dec).sum()} ({(~dec).mean()*100:.0f}%)")
        for i, c in enumerate(CLASSES):
            n = int((dec & (lab == i)).sum())
            print(f"      {c:5s} {n:4d}")
        print("      by family (decided / total, dominant call):")
        for g in np.unique(grp):
            mg = grp == g; nd = int((dec & mg).sum()); tot = int(mg.sum())
            if nd:
                sub = lab[dec & mg]; dom = CLASSES[np.bincount(sub, minlength=3).argmax()]
                brk = "/".join(f"{c}{int((sub==k).sum())}" for k, c in enumerate(CLASSES))
            else:
                dom, brk = "-", "-"
            print(f"        {g:11s} {nd:3d}/{tot:3d}  dominant {dom:5s}  [{brk}]")


def main():
    for nk in ("logistic", "mixed"):
        grp, P = real_probs(nk)
        report(grp, P, "stable-twin " + ("logistic" if nk == "logistic" else "mixed(open-set)"))


if __name__ == "__main__":
    main()
