#!/usr/bin/env python3
"""
proper_run.py — a larger, multi-seed FeatMLP run for solid per-class statistics.

Rationale: the headline matched test is only ~350/class on a single split. Here we
(1) generate a larger t50-matched dataset (bigger raw pools -> more series survive
matching), (2) train the FeatMLP (46-D features in, the best model) across several
seeds, each with a fresh stratified split and fresh weight init, and (3) report per-class
precision/recall/F1 as mean +/- sd on both the matched test (arbiter) and the natural
control.

Writes to data/proper/ and runs/proper/ so the headline artifacts are untouched.
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
from matched_inflection_experiment import gen_pool
from data import gen_null_pool, match_t50_3class, natural_balanced
from features import build_features46
from models import FeatMLP
import train as T

PDATA = os.path.join(HERE, "data", "proper")
PRUNS = os.path.join(HERE, "runs", "proper")
os.makedirs(PDATA, exist_ok=True)
os.makedirs(PRUNS, exist_ok=True)

N_PER_POOL = 40000        # was 15000 in the headline run
N_NATURAL = 6000          # per class in the natural control
N_SEEDS = 5
CLASSES = ["SN", "TC", "Null"]


def build_dataset():
    """Generate + feature-ise the large dataset once; cache to data/proper/."""
    if os.path.exists(f"{PDATA}/F_matched.npy"):
        print("proper dataset already built, loading cache")
        return
    t0 = time.time()
    print(f"Generating {N_PER_POOL}/class wide pools ...")
    Xsn, tsn = gen_pool(0, N_PER_POOL, 101)
    print(f"  SN done ({time.time()-t0:.0f}s)")
    Xtc, ttc = gen_pool(1, N_PER_POOL, 102)
    print(f"  TC done ({time.time()-t0:.0f}s)")
    Xnu, tnu = gen_null_pool(N_PER_POOL, 103)
    print(f"  Null done ({time.time()-t0:.0f}s)")
    pools = [(Xsn.astype(np.float32), tsn), (Xtc.astype(np.float32), ttc),
             (Xnu.astype(np.float32), tnu)]

    Xm, ym, tm = match_t50_3class(pools, seed=0)
    Xn, yn, tn = natural_balanced(pools, N_NATURAL, seed=10)
    print(f"MATCHED N={len(ym)} per class {np.bincount(ym).tolist()};  "
          f"NATURAL N={len(yn)}")

    print("Extracting 46-D features (matched) ...")
    Fm = build_features46(Xm).astype(np.float32)
    print(f"  matched features done ({time.time()-t0:.0f}s); natural ...")
    Fn = build_features46(Xn).astype(np.float32)
    print(f"  natural features done ({time.time()-t0:.0f}s)")

    np.save(f"{PDATA}/F_matched.npy", Fm); np.save(f"{PDATA}/y_matched.npy", ym)
    np.save(f"{PDATA}/F_natural.npy", Fn); np.save(f"{PDATA}/y_natural.npy", yn)
    json.dump({"n_per_pool": N_PER_POOL, "matched_N": int(len(ym)),
               "natural_N": int(len(yn))}, open(f"{PDATA}/meta.json", "w"), indent=2)
    print(f"cached proper dataset ({time.time()-t0:.0f}s)")


def per_class(pred, yt):
    out = {}
    M = np.zeros((3, 3), int)
    for a, b in zip(yt, pred):
        M[a, b] += 1
    for i, c in enumerate(CLASSES):
        prec = M[i, i] / max(M[:, i].sum(), 1)
        rec = M[i, i] / max(M[i, :].sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out[c] = dict(precision=prec, recall=rec, f1=f1)
    out["_overall"] = float((pred == yt).mean())
    out["_confusion"] = M.tolist()
    return out


def main():
    build_dataset()
    Fm = np.load(f"{PDATA}/F_matched.npy").astype(np.float32)
    ym = np.load(f"{PDATA}/y_matched.npy").astype(np.int64)
    Fn = np.load(f"{PDATA}/F_natural.npy").astype(np.float32)
    yn = np.load(f"{PDATA}/y_natural.npy").astype(int)
    print(f"matched N={len(ym)} ({np.bincount(ym).tolist()}), natural N={len(yn)}")

    runs = []
    for seed in range(N_SEEDS):
        itr, iva, ite = T.make_splits(ym, seed=100 + seed)
        mu, sd = Fm[itr].mean(0), Fm[itr].std(0) + 1e-9
        Fmz = np.clip((Fm - mu) / sd, -8, 8).astype(np.float32)
        Fnz = np.clip((Fn - mu) / sd, -8, 8).astype(np.float32)
        torch.manual_seed(seed); np.random.seed(seed)
        model, _, best_va = T.train_model(FeatMLP(), f"proper/featmlp_s{seed}", "mlp", 0.0,
                                           (torch.from_numpy(Fmz), torch.from_numpy(ym)),
                                           itr, iva)
        model.eval()
        with torch.no_grad():
            pm = model(torch.from_numpy(Fmz[ite]).to(T.DEVICE)).argmax(1).cpu().numpy()
            pn = model(torch.from_numpy(Fnz).to(T.DEVICE)).argmax(1).cpu().numpy()
        runs.append(dict(seed=seed, val=best_va,
                         matched=per_class(pm, ym[ite]), natural=per_class(pn, yn)))
        print(f"  seed {seed}: matched {runs[-1]['matched']['_overall']*100:.1f}%  "
              f"natural {runs[-1]['natural']['_overall']*100:.1f}%")

    # aggregate mean/sd
    def agg(split, key, metric):
        v = [r[split][key][metric] for r in runs]
        return float(np.mean(v)), float(np.std(v))

    summary = {"n_per_pool": N_PER_POOL, "matched_N": int(len(ym)),
               "matched_test_per_seed": int(len(T.make_splits(ym, seed=100)[2])),
               "natural_N": int(len(yn)), "n_seeds": N_SEEDS, "agg": {}}
    for split in ("matched", "natural"):
        summary["agg"][split] = {"overall": agg(split, "_overall", None) if False else
                                 (float(np.mean([r[split]["_overall"] for r in runs])),
                                  float(np.std([r[split]["_overall"] for r in runs])))}
        for c in CLASSES:
            summary["agg"][split][c] = {m: agg(split, c, m) for m in ("precision", "recall", "f1")}
    summary["runs"] = runs
    json.dump(summary, open(f"{PRUNS}/proper_summary.json", "w"), indent=2)

    # console + markdown table
    print("\n" + "=" * 70)
    print(f"FeatMLP proper run: {N_SEEDS} seeds, matched test "
          f"{summary['matched_test_per_seed']}/split, natural {len(yn)}")
    lines = [f"# FeatMLP proper run ({N_SEEDS} seeds, {N_PER_POOL}/class pools)", "",
             f"Matched set N = {len(ym)} ({np.bincount(ym).tolist()} per class); "
             f"matched test ~{summary['matched_test_per_seed']}/split; natural N = {len(yn)}.",
             f"Pool size {N_PER_POOL}/class (headline used 15000).", ""]
    for split in ("matched", "natural"):
        om, osd = summary["agg"][split]["overall"]
        print(f"\n{split.upper()}  overall {om*100:.1f} +/- {osd*100:.1f}%")
        lines += [f"## {split.capitalize()} (overall {om*100:.1f} +/- {osd*100:.1f}%)", "",
                  "| class | precision | recall | F1 |", "|---|---|---|---|"]
        for c in CLASSES:
            pr = summary["agg"][split][c]
            print(f"  {c:4s} prec {pr['precision'][0]*100:5.1f}+/-{pr['precision'][1]*100:.1f}  "
                  f"rec {pr['recall'][0]*100:5.1f}+/-{pr['recall'][1]*100:.1f}  "
                  f"F1 {pr['f1'][0]*100:5.1f}+/-{pr['f1'][1]*100:.1f}")
            lines.append(f"| {c} | {pr['precision'][0]*100:.1f} +/- {pr['precision'][1]*100:.1f} "
                         f"| {pr['recall'][0]*100:.1f} +/- {pr['recall'][1]*100:.1f} "
                         f"| {pr['f1'][0]*100:.1f} +/- {pr['f1'][1]*100:.1f} |")
        lines.append("")
    open(f"{PRUNS}/PROPER_RESULTS.md", "w").write("\n".join(lines))

    # figure: mean confusion (matched) + per-class F1 bars with error bars
    Mm = np.mean([np.array(r["matched"]["_confusion"], float) /
                  np.array(r["matched"]["_confusion"], float).sum(1, keepdims=True)
                  for r in runs], axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    im = ax1.imshow(Mm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f"{Mm[i, j]:.3f}", ha="center", va="center", fontsize=9,
                     color="white" if Mm[i, j] > 0.6 else "#222")
    ax1.set_xticks(range(3)); ax1.set_xticklabels(CLASSES)
    ax1.set_yticks(range(3)); ax1.set_yticklabels(CLASSES)
    ax1.set_xlabel("predicted"); ax1.set_ylabel("true")
    ax1.set_title(f"(a) mean matched-test confusion ({N_SEEDS} seeds)", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax1, fraction=0.046)

    x = np.arange(3)
    for off, split, col in [(-0.2, "matched", "#2a9d8f"), (0.2, "natural", "#457b9d")]:
        f1 = [summary["agg"][split][c]["f1"][0] * 100 for c in CLASSES]
        er = [summary["agg"][split][c]["f1"][1] * 100 for c in CLASSES]
        ax2.bar(x + off, f1, 0.4, yerr=er, capsize=3, color=col, alpha=0.9, label=split)
    ax2.set_xticks(x); ax2.set_xticklabels(CLASSES)
    ax2.set_ylim(70, 102); ax2.set_ylabel("F1 (%)")
    ax2.legend(fontsize=8); ax2.set_title("(b) per-class F1 (mean +/- sd)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(f"{PRUNS}/fig_proper.png", dpi=180, bbox_inches="tight")
    print(f"\nsaved {PRUNS}/proper_summary.json, PROPER_RESULTS.md, fig_proper.png")


if __name__ == "__main__":
    main()
