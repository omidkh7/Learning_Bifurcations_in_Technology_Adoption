#!/usr/bin/env python3
"""
check1_mixed_null.py — the proper (multi-seed) FeatMLP run with the SI's MIXED null.

The proper run (§6b) used the widened logistic null. The null sweep showed null
recognition is shape-specific and that only the heterogeneous null generalises, so the
defensible deployment model is trained with the SI's Option-C mixed null (a blend of
rising 0-to-1 shapes: linear + exp-saturating + sigmoid). This reruns the proper protocol
with that null: all three classes t50-matched, FeatMLP across 5 seeds, per-class metrics.

Writes runs/checks/check1_mixed_null.json + console/markdown table.
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from checks_common import gen_pool_meta, load_mixed_null
from data import match_t50_3class, natural_balanced
from features import build_features46
from models import FeatMLP
import train as T

CDATA = os.path.join(ROOT, "data", "curated", "deep", "checks")
CRUNS = os.path.join(ROOT, "results", "benchmark", "checks")
os.makedirs(CDATA, exist_ok=True)
os.makedirs(CRUNS, exist_ok=True)

N_SNTC = 20000          # SN/TC pool
N_NULL = 10000          # the Option-C null pool has 10k series
N_SEEDS = 5
CLASSES = ["SN", "TC", "Null"]


def build():
    if os.path.exists(f"{CDATA}/F_mixed_matched.npy"):
        print("check1 dataset cached")
        return
    t0 = time.time()
    print(f"generating SN/TC pools ({N_SNTC}/class) ...")
    Xsn, tsn, *_ = gen_pool_meta(0, N_SNTC, 201)
    print(f"  SN done ({time.time()-t0:.0f}s)")
    Xtc, ttc, *_ = gen_pool_meta(1, N_SNTC, 202)
    print(f"  TC done ({time.time()-t0:.0f}s)")
    Xnu, tnu = load_mixed_null(N_NULL, 203)
    print(f"  mixed null loaded: {Xnu.shape}, t50 median {np.median(tnu):.2f} "
          f"range [{tnu.min():.2f}, {tnu.max():.2f}]")

    pools = [(Xsn, tsn), (Xtc, ttc), (Xnu, tnu)]
    Xm, ym, tm = match_t50_3class(pools, seed=0)
    Xn, yn, tn = natural_balanced(pools, min(4000, N_NULL), seed=10)
    print(f"MATCHED N={len(ym)} per class {np.bincount(ym).tolist()}; NATURAL N={len(yn)}")

    print("extracting features ...")
    Fm = build_features46(Xm).astype(np.float32)
    Fn = build_features46(Xn).astype(np.float32)
    np.save(f"{CDATA}/F_mixed_matched.npy", Fm); np.save(f"{CDATA}/y_mixed_matched.npy", ym)
    np.save(f"{CDATA}/F_mixed_natural.npy", Fn); np.save(f"{CDATA}/y_mixed_natural.npy", yn)
    print(f"cached ({time.time()-t0:.0f}s)")


def per_class(pred, yt):
    M = np.zeros((3, 3), int)
    for a, b in zip(yt, pred):
        M[a, b] += 1
    out = {}
    for i, c in enumerate(CLASSES):
        prec = M[i, i] / max(M[:, i].sum(), 1); rec = M[i, i] / max(M[i, :].sum(), 1)
        out[c] = dict(precision=prec, recall=rec,
                      f1=2 * prec * rec / max(prec + rec, 1e-9))
    out["_overall"] = float((pred == yt).mean())
    out["_confusion"] = M.tolist()
    return out


def main():
    build()
    Fm = np.load(f"{CDATA}/F_mixed_matched.npy").astype(np.float32)
    ym = np.load(f"{CDATA}/y_mixed_matched.npy").astype(np.int64)
    Fn = np.load(f"{CDATA}/F_mixed_natural.npy").astype(np.float32)
    yn = np.load(f"{CDATA}/y_mixed_natural.npy").astype(int)
    print(f"matched N={len(ym)} {np.bincount(ym).tolist()}, natural N={len(yn)}")

    runs = []
    for seed in range(N_SEEDS):
        itr, iva, ite = T.make_splits(ym, seed=300 + seed)
        mu, sd = Fm[itr].mean(0), Fm[itr].std(0) + 1e-9
        Fmz = np.clip((Fm - mu) / sd, -8, 8).astype(np.float32)
        Fnz = np.clip((Fn - mu) / sd, -8, 8).astype(np.float32)
        torch.manual_seed(seed); np.random.seed(seed)
        model, _, _ = T.train_model(FeatMLP(), f"checks/mixed_s{seed}", "mlp", 0.0,
                                    (torch.from_numpy(Fmz), torch.from_numpy(ym)), itr, iva)
        model.eval()
        with torch.no_grad():
            pm = model(torch.from_numpy(Fmz[ite]).to(T.DEVICE)).argmax(1).cpu().numpy()
            pn = model(torch.from_numpy(Fnz).to(T.DEVICE)).argmax(1).cpu().numpy()
        runs.append(dict(seed=seed, matched=per_class(pm, ym[ite]), natural=per_class(pn, yn)))
        print(f"  seed {seed}: matched {runs[-1]['matched']['_overall']*100:.1f}%  "
              f"natural {runs[-1]['natural']['_overall']*100:.1f}%")

    summary = {"null": "mixed (Option-C heterogeneous)", "matched_N": int(len(ym)),
               "natural_N": int(len(yn)), "n_seeds": N_SEEDS, "agg": {}, "runs": runs}
    for split in ("matched", "natural"):
        d = {"overall": [float(np.mean([r[split]["_overall"] for r in runs])),
                         float(np.std([r[split]["_overall"] for r in runs]))]}
        for c in CLASSES:
            d[c] = {m: [float(np.mean([r[split][c][m] for r in runs])),
                        float(np.std([r[split][c][m] for r in runs]))]
                    for m in ("precision", "recall", "f1")}
        summary["agg"][split] = d
    json.dump(summary, open(f"{CRUNS}/check1_mixed_null.json", "w"), indent=2)

    print("\n" + "=" * 68)
    print("CHECK 1 — FeatMLP with the SI mixed (Option-C heterogeneous) null")
    for split in ("matched", "natural"):
        om, osd = summary["agg"][split]["overall"]
        print(f"\n{split.upper()}: overall {om*100:.1f} +/- {osd*100:.1f}%")
        for c in CLASSES:
            p = summary["agg"][split][c]
            print(f"  {c:4s} prec {p['precision'][0]*100:5.1f}+/-{p['precision'][1]*100:.1f}  "
                  f"rec {p['recall'][0]*100:5.1f}+/-{p['recall'][1]*100:.1f}  "
                  f"F1 {p['f1'][0]*100:5.1f}+/-{p['f1'][1]*100:.1f}")
    print(f"\nsaved {CRUNS}/check1_mixed_null.json")


if __name__ == "__main__":
    main()
