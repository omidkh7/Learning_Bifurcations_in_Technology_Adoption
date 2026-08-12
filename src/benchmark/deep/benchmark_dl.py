#!/usr/bin/env python3
"""
benchmark_dl.py — run the DEEP-LEARNING stage on the PAPER'S CURRENT canonical benchmark.

Uses the manuscript/SI's most recent class definitions verbatim, via the repo's single
source of truth `benchmark_data.load_benchmark` (NULL_VERSION = 2):

  SN, TC : the canonical normal-form simulations (data/synthetic_optionc), resampled to
           the 100-point grid (SI S3).
  Null   : the STABLE TWIN of TC (SI S3, 2026-07 audit). Each null draws the same (mu, a2)
           as the TC sampler, scales its rise template to the physical height x*=mu/a2,
           relaxes toward it at the constant rate lambda=mu through the same Euler-Maruyama
           noise recursion, amplitude palette, AR(1) colour, native grid, resampling, and
           validity gates (incl. visible plateau). Stable throughout => no critical slowing
           down; shape/params/amplitude/noise/selection all matched, so the ONLY difference
           from TC is the dynamics. Canonical = the logistic twin with (k, t0) from logistic
           fits to the actual TC sims.

This is NOT t50-matched between SN and TC (the paper's benchmark is not): SN stays
back-loaded, TC and its null twin are front-loaded and t50-aligned, so TC-vs-null is the
pure-dynamics boundary. Feature space: the paper's own `paper_figures.build_features`
(46-D). We report, on the identical generated benchmark:

  * the paper's UNSUPERVISED methods (Student-t, skew-t, mechanism-prior oracle,
    nearest-centroid) with Hungarian-matched accuracy, as in SI S3/S6, and
  * the SUPERVISED FeatMLP (46-D in), multi-seed, per-class recall + confusion,

so the change the DL stage makes to benchmark-class recovery is directly visible.

Runs the canonical logistic twin and the open-set 'mixed' twin. Writes
runs/benchmark/benchmark_dl.json + fig_benchmark_dl.png (nothing outside src/benchmark/deep/).
"""
import os, sys, json, time, warnings
# Cap BLAS threading BEFORE numpy/torch import: mixing macOS Accelerate multithreaded BLAS
# (scipy t-/skew-t mixtures) with PyTorch MPS in one process segfaults intermittently.
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from benchmark_data import load_benchmark, NULL_VERSION
from paper_figures import build_features, _tag               # the paper's 46-D feature fn + panel tag
from paper_style import set_style, COL2
from unsup_real_world import fit_t_mixture, fit_skew_t_mixture, fit_theory_bayes_gmm
from models import FeatMLP
import train as T

set_style()                                                  # paper house style (Times, etc.)
BRUNS = os.path.join(ROOT, "results", "benchmark")
os.makedirs(BRUNS, exist_ok=True)

N_PER = 3000            # per class for the DL split
N_UNSUP = 1200          # per class subsample for the (slower) unsupervised fits, SI-scale
N_SEEDS = 5
CLASSES = ["SN", "TC", "Null"]
NULLS = ["logistic", "mixed"]      # canonical stable twin, then open-set stable twin

# "proper" mode (python benchmark_dl.py proper): the reviewer-proofing run. Larger classes,
# more model seeds, and THREE independent benchmark draws (data seeds), the axis the default
# run does not cover. Writes *_proper outputs; the canonical files are left untouched.
PROPER_N_PER = 10000
PROPER_N_SEEDS = 10
PROPER_DATA_SEEDS = [0, 1, 2]


def hung_acc(lab, ys):
    """Hungarian-matched accuracy + normalised confusion (rows true, cols matched-pred)."""
    C = np.array([[((lab == i) & (ys == c)).sum() for c in range(3)] for i in range(3)])
    r, cc = linear_sum_assignment(-C)
    mp = {i: cc[j] for j, i in enumerate(r)}
    pm = np.array([mp[l] for l in lab])
    M = np.array([[((ys == tc) & (pm == pc)).sum() / max((ys == tc).sum(), 1)
                   for pc in range(3)] for tc in range(3)])
    return float((pm == ys).mean()), M


def recall_conf(pred, yt):
    M = np.array([[((yt == tc) & (pred == pc)).sum() / max((yt == tc).sum(), 1)
                   for pc in range(3)] for tc in range(3)])
    return float((pred == yt).mean()), M


def run_null(null_kind, seed0=0):
    print(f"\n{'='*70}\nNULL = {null_kind}  (NULL_VERSION={NULL_VERSION}, stable twin, "
          f"data seed {seed0}, n_per {N_PER})\n{'='*70}", flush=True)
    Xn, ys = load_benchmark(null_kind, N_PER, seed=seed0)
    F = np.nan_to_num(build_features(Xn)).astype(np.float32)
    print(f"benchmark: X {Xn.shape}  F {F.shape}  per class {np.bincount(ys).tolist()}")
    out = {"n_per": N_PER, "unsup": {}, "dl": {}}

    # ---------- unsupervised (paper methods), SI-scale subsample ----------
    rng = np.random.default_rng(123)
    sub = np.concatenate([rng.choice(np.where(ys == c)[0], N_UNSUP, replace=False)
                          for c in range(3)])
    Zs = StandardScaler().fit_transform(F[sub]); yss = ys[sub]
    cents = np.vstack([Zs[yss == c].mean(0) for c in range(3)])
    t0 = time.time()
    unsup_fits = {
        "Student-t":        fit_t_mixture(Zs, 3, seed=0, n_init=3)[0],
        "skew-t":           fit_skew_t_mixture(Zs, 3, seed=0, n_init=3)[0],
        "mechanism prior":  fit_theory_bayes_gmm(Zs, 3, centroids_prior=cents, seed=0, n_init=3)[0],
        "nearest-centroid": np.argmin(np.linalg.norm(Zs[:, None, :] - cents[None], axis=2), axis=1),
    }
    for name, lab in unsup_fits.items():
        acc, M = hung_acc(lab, yss)
        out["unsup"][name] = dict(acc=acc, recall=[float(M[i, i]) for i in range(3)],
                                  confusion=M.tolist())
        print(f"  [unsup] {name:16s} acc {acc*100:5.1f}%  recall "
              f"SN {M[0,0]*100:4.0f} TC {M[1,1]*100:4.0f} Null {M[2,2]*100:4.0f}")
    print(f"  (unsupervised fits {time.time()-t0:.0f}s)")

    # ---------- supervised FeatMLP, multi-seed ----------
    runs = []; proba_all = []; ytrue_all = []
    for s in range(N_SEEDS):
        itr, iva, ite = T.make_splits(ys, seed=1000 + s)
        mu, sd = F[itr].mean(0), F[itr].std(0) + 1e-9
        Fz = np.clip((F - mu) / sd, -8, 8).astype(np.float32)
        torch.manual_seed(s); np.random.seed(s)
        m, _, _ = T.train_model(FeatMLP(), f"benchmark/{null_kind}_d{seed0}_s{s}", "mlp", 0.0,
                                (torch.from_numpy(Fz), torch.from_numpy(ys.astype(np.int64))),
                                itr, iva)
        m.eval()
        with torch.no_grad():
            proba = torch.softmax(m(torch.from_numpy(Fz[ite]).to(T.DEVICE)), dim=1).cpu().numpy()
            pred = proba.argmax(1)
        proba_all.append(proba); ytrue_all.append(ys[ite])
        acc, M = recall_conf(pred, ys[ite])
        runs.append(dict(seed=s, acc=acc, recall=[float(M[i, i]) for i in range(3)],
                         confusion=M.tolist()))
        print(f"  [DL s{s}] acc {acc*100:5.1f}%  recall "
              f"SN {M[0,0]*100:4.0f} TC {M[1,1]*100:4.0f} Null {M[2,2]*100:4.0f}")
    accs = [r["acc"] for r in runs]
    rec = np.array([r["recall"] for r in runs])
    conf = np.mean([np.array(r["confusion"]) for r in runs], axis=0)
    from sklearn.metrics import roc_curve, roc_auc_score
    P = np.vstack(proba_all); Y = np.concatenate(ytrue_all)      # columns: SN, TC, null
    def _roc(mask, pos, score):
        yy = np.isin(Y[mask], np.atleast_1d(pos)).astype(int); sc = score[mask]
        fpr, tpr, _ = roc_curve(yy, sc)
        return dict(fpr=fpr.tolist(), tpr=tpr.tolist(), auc=float(roc_auc_score(yy, sc)))
    roc = {"tc_vs_null": _roc(np.isin(Y, [1, 2]), 1, P[:, 1]),           # positive=TC, score=P(TC)
           "bif_vs_null": _roc(np.ones(len(Y), bool), [0, 1], P[:, 0] + P[:, 1])}  # positive=SN|TC
    out["dl"] = dict(acc_mean=float(np.mean(accs)), acc_sd=float(np.std(accs)),
                     recall_mean=[float(v) for v in rec.mean(0)],
                     recall_sd=[float(v) for v in rec.std(0)],
                     confusion_mean=conf.tolist(), runs=runs, roc=roc)
    print(f"  [DL] overall {np.mean(accs)*100:.1f} +/- {np.std(accs)*100:.1f}%  "
          f"recall SN {rec.mean(0)[0]*100:.0f} TC {rec.mean(0)[1]*100:.0f} "
          f"Null {rec.mean(0)[2]*100:.0f}")
    return out


def make_figure(res, tag=""):
    """Paper house-style figure from the results dict (reconstructable from benchmark_dl.json)."""
    UM = ["Student-t", "skew-t", "mechanism prior", "nearest-centroid"]
    LAB = ["Student-$t$", "skew-$t$", "mech.\nprior", "near.-\ncent.", "FeatMLP"]
    BAR = {"overall": "#1d4e89", "tc": "#457b9d", "null": "#9aa0a6"}
    tags = ["(a) logistic stable-twin null", "(b) mixed stable-twin null"]
    fig, axes = plt.subplots(1, len(NULLS) + 1, figsize=(COL2, 2.5),
                             gridspec_kw=dict(width_ratios=[1.4] * len(NULLS) + [1.0], wspace=0.42))
    for ax, nk, tg in zip(axes[:len(NULLS)], NULLS, tags):
        d = res["nulls"][nk]
        acc = [d["unsup"][m]["acc"] for m in UM] + [d["dl"]["acc_mean"]]
        tcrec = [d["unsup"][m]["recall"][1] for m in UM] + [d["dl"]["recall_mean"][1]]
        nurec = [d["unsup"][m]["recall"][2] for m in UM] + [d["dl"]["recall_mean"][2]]
        x = np.arange(len(LAB))
        ax.bar(x - 0.27, np.array(acc) * 100, 0.27, color=BAR["overall"], label="overall acc.")
        ax.bar(x, np.array(tcrec) * 100, 0.27, color=BAR["tc"], label="TC recall")
        ax.bar(x + 0.27, np.array(nurec) * 100, 0.27, color=BAR["null"], label="null recall")
        ax.axhline(100 / 3, color="#999", lw=0.7, ls=":")
        ax.axvline(3.5, color="#d62828", lw=1.0)                 # baselines | supervised DL
        ax.set_xticks(x); ax.set_xticklabels(LAB, rotation=18, ha="right", fontsize=5.6)
        ax.set_ylim(0, 108); ax.set_ylabel("recovery (%)", fontsize=7.5)
        _tag(ax, tg)
        if nk == NULLS[0]:
            ax.legend(fontsize=5.6, loc="upper left", frameon=False, handlelength=1.0,
                      labelspacing=0.3, handletextpad=0.4)

    ax = axes[-1]
    roc = res["nulls"]["logistic"]["dl"].get("roc")
    if roc:
        for key, c, lab in [("bif_vs_null", "#d62828", "bifurc. vs null"),
                            ("tc_vs_null", "#457b9d", "TC vs null")]:
            r = roc[key]; ax.plot(r["fpr"], r["tpr"], color=c, lw=1.6, label=f"{lab}: {r['auc']:.2f}")
        ax.plot([0, 1], [0, 1], color="#bbb", lw=0.8, ls="--")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("false-positive rate", fontsize=6.5); ax.set_ylabel("true-positive rate", fontsize=6.5)
        ax.legend(fontsize=5.3, loc="lower right", frameon=False, title="AUC", title_fontsize=5.3)
    _tag(ax, "(c) FeatMLP detection ROC (logistic null)")
    fig.savefig(f"{BRUNS}/fig_benchmark_dl{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {BRUNS}/fig_benchmark_dl{tag}.png", flush=True)


def main(proper=False):
    global N_PER, N_SEEDS
    data_seeds = PROPER_DATA_SEEDS if proper else [0]
    if proper:
        N_PER, N_SEEDS = PROPER_N_PER, PROPER_N_SEEDS
    tag = "_proper" if proper else ""
    res = {"null_version": NULL_VERSION, "n_per": N_PER, "n_seeds": N_SEEDS,
           "data_seeds": data_seeds, "draws": {}}
    for ds in data_seeds:
        res["draws"][str(ds)] = {}
        for nk in NULLS:
            res["draws"][str(ds)][nk] = run_null(nk, seed0=ds)
    res["nulls"] = res["draws"]["0"]                     # canonical draw (used for the figure)
    if proper:
        # cross-draw aggregate: is the benchmark realization special? (n_runs = draws x seeds)
        res["aggregate_dl"] = {}
        for nk in NULLS:
            accs = [r["acc"] for ds in data_seeds for r in res["draws"][str(ds)][nk]["dl"]["runs"]]
            recs = np.array([r["recall"] for ds in data_seeds
                             for r in res["draws"][str(ds)][nk]["dl"]["runs"]])
            res["aggregate_dl"][nk] = dict(
                n_runs=len(accs), acc_mean=float(np.mean(accs)), acc_sd=float(np.std(accs)),
                acc_range=[float(np.min(accs)), float(np.max(accs))],
                recall_mean=[float(v) for v in recs.mean(0)],
                per_draw_acc={str(ds): res["draws"][str(ds)][nk]["dl"]["acc_mean"]
                              for ds in data_seeds})
    json.dump(res, open(f"{BRUNS}/benchmark_dl{tag}.json", "w"), indent=2)
    make_figure(res, tag=tag)

    # ---------- console summary ----------
    print("\n" + "=" * 72)
    print("SUMMARY — DL vs unsupervised on the paper's current stable-twin benchmark")
    print("=" * 72)
    for ds in data_seeds:
        for nk in NULLS:
            d = res["draws"][str(ds)][nk]
            print(f"\n[{nk} stable-twin null, data seed {ds}]")
            print(f"  {'method':18s} {'overall':>8s} {'SN':>6s} {'TC':>6s} {'Null':>6s}")
            for name in ["Student-t", "skew-t", "mechanism prior", "nearest-centroid"]:
                u = d["unsup"][name]
                print(f"  {name:18s} {u['acc']*100:7.1f}% {u['recall'][0]*100:5.0f} "
                      f"{u['recall'][1]*100:5.0f} {u['recall'][2]*100:5.0f}")
            dl = d["dl"]
            print(f"  {'FeatMLP (DL)':18s} {dl['acc_mean']*100:7.1f}% {dl['recall_mean'][0]*100:5.0f} "
                  f"{dl['recall_mean'][1]*100:5.0f} {dl['recall_mean'][2]*100:5.0f}   "
                  f"(+/- {dl['acc_sd']*100:.1f} overall)")
    if proper:
        print("\n[aggregate across draws x seeds]")
        for nk in NULLS:
            a = res["aggregate_dl"][nk]
            print(f"  {nk:9s}: {a['acc_mean']*100:.1f} +/- {a['acc_sd']*100:.1f}%  "
                  f"range {a['acc_range'][0]*100:.1f}-{a['acc_range'][1]*100:.1f}  "
                  f"per-draw {[round(v*100,1) for v in a['per_draw_acc'].values()]}")
    print(f"\nsaved {BRUNS}/benchmark_dl{tag}.json + fig_benchmark_dl{tag}.png", flush=True)


if __name__ == "__main__":
    if "replot" in sys.argv:                                    # restyle figure from cached JSON (no retrain)
        tag = "_proper" if "proper" in sys.argv else ""
        make_figure(json.load(open(f"{BRUNS}/benchmark_dl{tag}.json")), tag=tag)
    else:
        main(proper="proper" in sys.argv)
