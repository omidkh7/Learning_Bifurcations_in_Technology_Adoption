#!/usr/bin/env python3
"""
data.py — synthetic 3-class data for the second deep-learning attempt.

SN / TC: wide-parameter, NO-inflection-rejection normal-form SDEs, reusing
matched_inflection_experiment.gen_pool (the honest regime of §71).
Null: logistic shape (the hardest clean null of the SI benchmark, identical to the TC
solution) with WIDE centre t0 so its t50 spans the SN/TC support, carrying stationary
AR(1) noise of varied amplitude — no CSD by construction, and no noise-amplitude shortcut.

All three classes are then t50-MATCHED (equal counts per t50 bin) so inflection position
carries zero class information, in training as well as testing. A natural (unmatched)
balanced set is kept as a control. 500-point grid throughout (same as §71/§72a evidence).

Classes: 0 = SN, 1 = TC, 2 = Null.
Writes: second_attempt_deep/data/{X,y,t50}_{matched,natural}.npy + meta.json
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from matched_inflection_experiment import gen_pool, t50 as t50_of   # wide SN/TC generator (§71)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

T = 500
N_PER_POOL = 15000       # raw pool size per class before matching
N_BINS = 18              # t50 bins for stratified matching (same as §71)
SEED = 0


def gen_null_pool(n, seed):
    """Logistic null, wide t50, stationary AR(1) noise with varied amplitude.

    Same construction as benchmark_data.make_null('logistic') except (a) t0 is wide so the
    null's t50 overlaps the SN/TC support instead of clustering at 0.5, and (b) the noise
    amplitude is drawn from a range comparable to the SDE residual levels, so 'quiet series
    = Null' is not a shortcut. The noise stays stationary: no critical slowing down.
    """
    rng = np.random.default_rng(seed)
    tt = np.linspace(0, 1, T)
    X, ts = [], []
    while len(X) < n:
        k = rng.uniform(6, 20)
        t0 = rng.uniform(0.12, 0.88)
        x = 1.0 / (1.0 + np.exp(-k * (tt - t0)))
        e = rng.normal(0, 1, T)
        eta = np.zeros(T)
        phi = rng.uniform(0.0, 0.9)
        for i in range(1, T):
            eta[i] = phi * eta[i - 1] + e[i]
        amp = rng.uniform(0.01, 0.06)
        x = x + amp * eta / (np.abs(eta).max() + 1e-9)
        x = (x - x.min()) / (x.max() - x.min() + 1e-12)
        X.append(x.astype(np.float32))
        ts.append(t50_of(x))
    return np.array(X), np.array(ts)


def match_t50_3class(pools, nbins=N_BINS, seed=0):
    """Stratified matching: equal SN/TC/Null counts in every t50 bin.

    pools: list of (X, t50s) in class order. Returns X, y, t50 concatenated and shuffled.
    """
    lo = max(t.min() for _, t in pools)
    hi = min(t.max() for _, t in pools)
    edges = np.linspace(lo, hi, nbins + 1)
    rng = np.random.default_rng(seed)
    Xs, ys, ts = [], [], []
    for b in range(nbins):
        idx_per_class = [np.where((t >= edges[b]) & (t < edges[b + 1]))[0] for _, t in pools]
        k = min(len(ix) for ix in idx_per_class)
        if k == 0:
            continue
        for c, ((X, t), ix) in enumerate(zip(pools, idx_per_class)):
            pick = rng.choice(ix, k, replace=False)
            Xs.append(X[pick]); ys.append(np.full(k, c)); ts.append(t[pick])
    X = np.vstack(Xs); y = np.concatenate(ys).astype(int); t = np.concatenate(ts)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], t[perm]


def natural_balanced(pools, n_per, seed=0):
    """Unmatched control: n_per random series per class, natural t50 distributions."""
    rng = np.random.default_rng(seed)
    Xs, ys, ts = [], [], []
    for c, (X, t) in enumerate(pools):
        pick = rng.choice(len(X), n_per, replace=False)
        Xs.append(X[pick]); ys.append(np.full(n_per, c)); ts.append(t[pick])
    X = np.vstack(Xs); y = np.concatenate(ys).astype(int); t = np.concatenate(ts)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], t[perm]


def main():
    t_start = time.time()
    print(f"Generating wide-parameter pools, {N_PER_POOL} per class (500-pt grid) ...")
    Xsn, tsn = gen_pool(0, N_PER_POOL, SEED + 1)
    print(f"  SN   done ({time.time()-t_start:.0f}s)  t50 median {np.median(tsn):.2f} "
          f"range [{tsn.min():.2f}, {tsn.max():.2f}]")
    Xtc, ttc = gen_pool(1, N_PER_POOL, SEED + 2)
    print(f"  TC   done  t50 median {np.median(ttc):.2f} range [{ttc.min():.2f}, {ttc.max():.2f}]")
    Xnu, tnu = gen_null_pool(N_PER_POOL, SEED + 3)
    print(f"  Null done  t50 median {np.median(tnu):.2f} range [{tnu.min():.2f}, {tnu.max():.2f}]")

    pools = [(Xsn.astype(np.float32), tsn), (Xtc.astype(np.float32), ttc),
             (Xnu.astype(np.float32), tnu)]

    Xm, ym, tm = match_t50_3class(pools, seed=SEED)
    print(f"MATCHED set: N = {len(ym)}  per class {np.bincount(ym).tolist()}")

    n_nat = min(4000, N_PER_POOL)
    Xn, yn, tn = natural_balanced(pools, n_nat, seed=SEED + 10)
    print(f"NATURAL control set: N = {len(yn)}  per class {np.bincount(yn).tolist()}")

    np.save(f"{DATA}/X_matched.npy", Xm); np.save(f"{DATA}/y_matched.npy", ym)
    np.save(f"{DATA}/t50_matched.npy", tm)
    np.save(f"{DATA}/X_natural.npy", Xn); np.save(f"{DATA}/y_natural.npy", yn)
    np.save(f"{DATA}/t50_natural.npy", tn)

    meta = {
        "n_per_pool": N_PER_POOL, "n_bins": N_BINS, "seed": SEED, "grid": T,
        "classes": {"0": "SN", "1": "TC", "2": "Null"},
        "matched_N": int(len(ym)), "natural_N": int(len(yn)),
        "t50_median": {"SN": float(np.median(tsn)), "TC": float(np.median(ttc)),
                       "Null": float(np.median(tnu))},
    }
    with open(f"{DATA}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved to {DATA}  ({time.time()-t_start:.0f}s total)")


if __name__ == "__main__":
    main()
