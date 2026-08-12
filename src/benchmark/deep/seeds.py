#!/usr/bin/env python3
"""
seeds.py — 3-seed robustness check for the two encoders (§71 taught us not to trust one run).

Keeps the data split fixed (train.py's SEED=42 split) and varies the weight-initialisation /
batch-order seed. Reports matched-test and natural accuracy per seed and mean +/- sd.
Writes runs/seeds.json; checkpoints go to runs/seeds/ (the headline checkpoints from
train.py are untouched).
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train as T
from models import TSEncoder

DATA, RUNS = T.DATA, T.RUNS
os.makedirs(f"{RUNS}/seeds", exist_ok=True)
SEEDS = [0, 1, 2]


def main():
    X = np.load(f"{DATA}/X_matched.npy").astype(np.float32)
    F = np.load(f"{DATA}/F_matched.npy").astype(np.float32)
    y = np.load(f"{DATA}/y_matched.npy").astype(np.int64)
    Xn = np.load(f"{DATA}/X_natural.npy").astype(np.float32)
    yn = np.load(f"{DATA}/y_natural.npy").astype(int)

    sp = np.load(f"{RUNS}/splits.npz")
    itr, iva, ite = sp["train"], sp["val"], sp["test"]
    sc = np.load(f"{RUNS}/scaler.npz")
    Fz = np.clip((F - sc["mu"]) / sc["sd"], -8, 8).astype(np.float32)

    tX, tF, ty = torch.from_numpy(X), torch.from_numpy(Fz), torch.from_numpy(y)

    def test_acc(model):
        model.eval()
        accs = []
        for Xe, ye in [(X[ite], y[ite]), (Xn, yn)]:
            preds = []
            with torch.no_grad():
                for i in range(0, len(Xe), 1024):
                    lo, _ = model(torch.from_numpy(Xe[i:i + 1024]).to(T.DEVICE))
                    preds.append(lo.argmax(1).cpu().numpy())
            accs.append(float((np.concatenate(preds) == ye).mean()))
        return accs  # [matched_test, natural]

    out = {"lambda_fs": T.LAMBDA, "seeds": SEEDS, "runs": []}
    for seed in SEEDS:
        for name, lam in [("encoder_fs", T.LAMBDA), ("encoder_0", 0.0)]:
            torch.manual_seed(seed); np.random.seed(seed)
            m, _, best_va = T.train_model(TSEncoder(), f"seeds/{name}_s{seed}", "enc", lam,
                                          (tX, tF, ty), itr, iva)
            am, an = test_acc(m)
            out["runs"].append(dict(model=name, seed=seed, val=best_va,
                                    matched_test=am, natural=an))
            print(f"  -> {name} seed {seed}: matched {am*100:.1f}%  natural {an*100:.1f}%")

    print("\n" + "=" * 60)
    for name in ("encoder_fs", "encoder_0"):
        for k, lab in [("matched_test", "matched"), ("natural", "natural")]:
            v = [r[k] for r in out["runs"] if r["model"] == name]
            print(f"{name:12s} {lab:8s} {np.mean(v)*100:5.1f}% +/- {np.std(v)*100:.1f}  "
                  f"(runs: {', '.join(f'{x*100:.1f}' for x in v)})")
    with open(f"{RUNS}/seeds.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {RUNS}/seeds.json")


if __name__ == "__main__":
    main()
