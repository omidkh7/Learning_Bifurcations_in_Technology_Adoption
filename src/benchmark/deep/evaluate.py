#!/usr/bin/env python3
"""
evaluate.py — the verdict script for the second deep-learning attempt.

Scores on the t50-MATCHED test split (the arbiter, §71) and on the NATURAL unmatched
control set:

  baselines : logistic regression on t50 alone (should be ~chance on matched)
              nearest-centroid on the 46-D features (the §72a mechanism-prior reference)
              logistic regression on the 46-D features
  models    : featmlp, encoder_fs (feature-supervised), encoder_0 (ablation)

Also reports the encoder's feature-prediction R^2 by feature group (which of the eight
SI groups the network can actually read off the raw curve).

Writes: runs/summary.json, runs/RESULTS.md, runs/fig_second_attempt.png
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from models import FeatMLP, TSEncoder
from features import FEAT_NAMES_46, FEAT_GROUPS_46

DATA = os.path.join(ROOT, "data", "curated", "deep")
RUNS = os.path.join(ROOT, "results", "benchmark", "deep")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASSES = ["SN", "TC", "Null"]


def batched(fn, X, bs=1024):
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(fn(torch.from_numpy(X[i:i + bs]).to(DEVICE)))
    if isinstance(outs[0], tuple):
        return tuple(torch.cat([o[k] for o in outs]).cpu().numpy() for k in range(len(outs[0])))
    return torch.cat(outs).cpu().numpy()


def confusion(y, p, k=3):
    M = np.zeros((k, k), int)
    for a, b in zip(y, p):
        M[a, b] += 1
    return M


def main():
    # ---------- data ----------
    Xm = np.load(f"{DATA}/X_matched.npy").astype(np.float32)
    Fm = np.load(f"{DATA}/F_matched.npy").astype(np.float32)
    ym = np.load(f"{DATA}/y_matched.npy").astype(int)
    tm = np.load(f"{DATA}/t50_matched.npy")
    Xn = np.load(f"{DATA}/X_natural.npy").astype(np.float32)
    Fn = np.load(f"{DATA}/F_natural.npy").astype(np.float32)
    yn = np.load(f"{DATA}/y_natural.npy").astype(int)
    tn = np.load(f"{DATA}/t50_natural.npy")

    sp = np.load(f"{RUNS}/splits.npz")
    itr, ite = sp["train"], sp["test"]
    sc = np.load(f"{RUNS}/scaler.npz"); mu, sd = sc["mu"], sc["sd"]
    Fmz = np.clip((Fm - mu) / sd, -8, 8).astype(np.float32)
    Fnz = np.clip((Fn - mu) / sd, -8, 8).astype(np.float32)

    res = {"matched_test": {}, "natural": {}}

    # ---------- baselines (fit on matched TRAIN split only) ----------
    lr_t50 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr_t50.fit(tm[itr].reshape(-1, 1), ym[itr])
    res["matched_test"]["t50-only logreg"] = float(
        (lr_t50.predict(tm[ite].reshape(-1, 1)) == ym[ite]).mean())
    res["natural"]["t50-only logreg"] = float(
        (lr_t50.predict(tn.reshape(-1, 1)) == yn).mean())

    cents = np.vstack([Fmz[itr][ym[itr] == c].mean(0) for c in range(3)])
    nc = lambda Z: np.argmin(np.linalg.norm(Z[:, None, :] - cents[None], axis=2), axis=1)
    res["matched_test"]["46-D nearest-centroid"] = float((nc(Fmz[ite]) == ym[ite]).mean())
    res["natural"]["46-D nearest-centroid"] = float((nc(Fnz) == yn).mean())

    lr46 = LogisticRegression(max_iter=3000).fit(Fmz[itr], ym[itr])
    res["matched_test"]["46-D logreg"] = float((lr46.predict(Fmz[ite]) == ym[ite]).mean())
    res["natural"]["46-D logreg"] = float((lr46.predict(Fnz) == yn).mean())

    # ---------- trained models ----------
    mlp = FeatMLP().to(DEVICE)
    mlp.load_state_dict(torch.load(f"{RUNS}/featmlp.pt", map_location=DEVICE))
    mlp.eval()
    pm = batched(mlp, Fmz[ite]).argmax(1)
    pn = batched(mlp, Fnz).argmax(1)
    res["matched_test"]["FeatMLP (46-D in)"] = float((pm == ym[ite]).mean())
    res["natural"]["FeatMLP (46-D in)"] = float((pn == yn).mean())
    conf = {"FeatMLP (46-D in)": confusion(ym[ite], pm).tolist()}

    enc_preds = {}
    for name, label in [("encoder_fs", "Encoder + feature supervision"),
                        ("encoder_0", "Encoder ablation (lambda=0)")]:
        enc = TSEncoder().to(DEVICE)
        enc.load_state_dict(torch.load(f"{RUNS}/{name}.pt", map_location=DEVICE))
        enc.eval()
        lo_m, fp_m = batched(enc, Xm[ite])
        lo_n, _ = batched(enc, Xn)
        res["matched_test"][label] = float((lo_m.argmax(1) == ym[ite]).mean())
        res["natural"][label] = float((lo_n.argmax(1) == yn).mean())
        conf[label] = confusion(ym[ite], lo_m.argmax(1)).tolist()
        enc_preds[name] = (lo_m, fp_m)

    # ---------- feature-prediction R^2 by group (feature-supervised encoder) ----------
    _, fp = enc_preds["encoder_fs"]
    ft = Fmz[ite]
    ss_res = ((ft - fp) ** 2).sum(0)
    ss_tot = ((ft - ft.mean(0)) ** 2).sum(0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    groups = sorted(set(FEAT_GROUPS_46), key=FEAT_GROUPS_46.index)
    r2_group = {g: float(np.mean([r2[j] for j in range(46) if FEAT_GROUPS_46[j] == g]))
                for g in groups}
    res["feature_r2_by_group"] = r2_group
    res["feature_r2"] = {FEAT_NAMES_46[j]: float(r2[j]) for j in range(46)}

    # ---------- verdict ----------
    fs = res["matched_test"]["Encoder + feature supervision"]
    ab = res["matched_test"]["Encoder ablation (lambda=0)"]
    fs_nat = res["natural"]["Encoder + feature supervision"]
    ab_nat = res["natural"]["Encoder ablation (lambda=0)"]
    ceiling = max(res["matched_test"]["46-D nearest-centroid"],
                  res["matched_test"]["46-D logreg"],
                  res["matched_test"]["FeatMLP (46-D in)"])
    chance = 1 / 3
    gain_m, gain_n = fs - ab, fs_nat - ab_nat
    if fs > ceiling - 0.05 and (gain_m > 0.04 or gain_n > 0.08):
        verdict = (f"PASS: the feature-supervised encoder reaches the feature-space ceiling "
                   f"({fs*100:.1f}% vs ceiling {ceiling*100:.1f}% on the matched set) and beats "
                   f"its lambda=0 ablation by {gain_m*100:+.1f} pts matched / {gain_n*100:+.1f} "
                   f"pts natural — the 46-D supervision is doing real work, most visibly as "
                   f"out-of-distribution robustness.")
    elif gain_m > 0.04 or gain_n > 0.08:
        verdict = (f"PARTIAL: feature supervision helps ({gain_m*100:+.1f} pts matched, "
                   f"{gain_n*100:+.1f} pts natural over the ablation) but the encoder stays "
                   f"below the feature-space ceiling ({fs*100:.1f}% vs {ceiling*100:.1f}%).")
    elif fs > chance + 0.10:
        verdict = ("AMBIGUOUS: the encoder is above chance on the matched set but the ablation "
                   "does about as well on both sets, so the gain is not attributable to "
                   "feature supervision.")
    else:
        verdict = ("FAIL: the encoder is near chance on the matched set — same failure mode "
                   "as the old FFT (§71); the raw-curve input still cannot reach the dynamics.")
    res["verdict"] = verdict

    # ---------- seed robustness (overrides the single-run verdict if available) ----------
    seeds_path = f"{RUNS}/seeds.json"
    if os.path.exists(seeds_path):
        with open(seeds_path) as f:
            sj = json.load(f)
        stats = {}
        for name in ("encoder_fs", "encoder_0"):
            for k in ("matched_test", "natural"):
                v = [r[k] for r in sj["runs"] if r["model"] == name]
                stats[f"{name}.{k}"] = dict(mean=float(np.mean(v)), sd=float(np.std(v)),
                                            runs=[float(x) for x in v])
        res["seed_stats"] = stats
        fs_m, ab_m = stats["encoder_fs.matched_test"], stats["encoder_0.matched_test"]
        gap = fs_m["mean"] - ab_m["mean"]
        noise = 2 * max(fs_m["sd"], ab_m["sd"], 1e-9)
        if ab_m["mean"] > 0.75 and abs(gap) < noise:
            verdict = (f"SEED-ROBUST VERDICT: the raw-curve encoder learns the dynamics on the "
                       f"matched set regardless of feature supervision "
                       f"(with FS {fs_m['mean']*100:.1f}+/-{fs_m['sd']*100:.1f}%, without "
                       f"{ab_m['mean']*100:.1f}+/-{ab_m['sd']*100:.1f}%, chance 33.3%, old FFT "
                       f"= chance). The decisive ingredients are the t50-MATCHED training data "
                       f"(no inflection shortcut to exploit) and an architecture that sees the "
                       f"noise structure (dx channel + recurrence), not the feature-regression "
                       f"head: feature supervision is performance-neutral here. The best model "
                       f"overall remains the 46-D features as INPUT (FeatMLP, "
                       f"{res['matched_test']['FeatMLP (46-D in)']*100:.1f}%).")
        elif gap >= noise:
            verdict = (f"SEED-ROBUST VERDICT: feature supervision gives a real gain "
                       f"({gap*100:+.1f} pts over the ablation, beyond 2 sd) on the matched set.")
        else:
            verdict = (f"SEED-ROBUST VERDICT: feature supervision HURTS on the matched set "
                       f"({gap*100:+.1f} pts vs ablation, beyond 2 sd).")
        res["verdict"] = verdict

    with open(f"{RUNS}/summary.json", "w") as f:
        json.dump(res, f, indent=2)

    # ---------- console + RESULTS.md ----------
    lines = ["# Second attempt — results", "",
             f"Matched test N = {len(ite)}, natural control N = {len(yn)}, chance = 33.3%.", "",
             "| method | matched test | natural |", "|---|---|---|"]
    order = ["t50-only logreg", "46-D nearest-centroid", "46-D logreg", "FeatMLP (46-D in)",
             "Encoder + feature supervision", "Encoder ablation (lambda=0)"]
    print("\n" + "=" * 66)
    print(f"{'method':34s} {'matched':>9s} {'natural':>9s}")
    print("-" * 66)
    for k in order:
        a, b = res["matched_test"][k], res["natural"][k]
        print(f"{k:34s} {a*100:8.1f}% {b*100:8.1f}%")
        lines.append(f"| {k} | {a*100:.1f}% | {b*100:.1f}% |")
    print("=" * 66)
    print("\nFeature-prediction R^2 by group (encoder_fs, matched test):")
    for g, v in r2_group.items():
        print(f"  {g:12s} {v:+.2f}")
    print("\nVERDICT:", verdict)
    lines += ["", "## Feature-prediction R2 by group (feature-supervised encoder)", ""]
    lines += [f"- {g}: {v:+.2f}" for g, v in r2_group.items()]
    if "seed_stats" in res:
        lines += ["", "## Seed robustness (3 seeds, same split)", ""]
        for key, s in res["seed_stats"].items():
            lines.append(f"- {key}: {s['mean']*100:.1f}% +/- {s['sd']*100:.1f} "
                         f"(runs: {', '.join(f'{x*100:.1f}' for x in s['runs'])})")
    lines += ["", "## Verdict", "", verdict, ""]
    with open(f"{RUNS}/RESULTS.md", "w") as f:
        f.write("\n".join(lines))

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    ax = axes[0]
    vals = [res["matched_test"][k] * 100 for k in order]
    cols = ["#999999", "#457b9d", "#457b9d", "#2a9d8f", "#e63946", "#f4a261"]
    ax.barh(range(len(order)), vals, color=cols, alpha=0.9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(100 / 3, color="k", lw=0.8, ls=":", label="chance")
    ax.set_xlabel("accuracy on t50-matched test (%)")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=7)
    ax.set_title("(a) the arbiter: matched-inflection test", fontsize=9, loc="left")
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.0f}", va="center", fontsize=8)

    ax = axes[1]
    M = np.array(conf["Encoder + feature supervision"], float)
    M = M / M.sum(1, keepdims=True)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > 0.6 else "#222")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(3))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("(b) encoder+FS confusion, matched test", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    gs = list(r2_group.keys())
    gv = [max(r2_group[g], -0.2) for g in gs]
    ax.bar(range(len(gs)), gv, color="#6a4c93", alpha=0.85)
    ax.set_xticks(range(len(gs)))
    ax.set_xticklabels([g.split(":")[1] for g in gs], rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("mean feature-prediction $R^2$")
    ax.set_title("(c) which SI groups the encoder can read", fontsize=9, loc="left")

    fig.tight_layout()
    fig.savefig(f"{RUNS}/fig_second_attempt.png", dpi=180, bbox_inches="tight")
    print(f"\nSaved summary.json, RESULTS.md, fig_second_attempt.png to {RUNS}")


if __name__ == "__main__":
    main()
