#!/usr/bin/env python3
"""
null_ladder.py
==============
The null-construction LADDER (SI): how much of "detecting a bifurcation" was worth to each
non-dynamical cue. The logistic null is rebuilt in four stages, each closing one loophole
(all stages selectable in benchmark_data.load_benchmark, so this is fully reproducible):

  1            post-hoc stationary noise on templates      (noise texture identifies the null)
  "walk"       integrated noise, no restoring force        (texture closed; broken curves remain)
  "walk_gated" walk + endpoint rise/saturation gates       (broken curves removed)
  2            stable twin: OU-tracking at the classes'    (CANONICAL: shape, parameters, amplitude,
               own (mu, a2) scale + all class gates         noise, and selection all matched;
                                                            dynamics is the only remaining signal)

Output: Manuscript/SI_figures/figS_null_ladder.png
        + runs/unsup/bifurcation_explore/null_ladder.csv
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from paper_style import set_style, COL2
set_style()
from paper_figures import build_features, _tag
from benchmark_data import load_benchmark, LADDER, LADDER_LABEL
from unsup_real_world import fit_t_mixture, fit_skew_t_mixture

N_PER = 800


def hung(lab, ys):
    C = np.array([[((lab == i) & (ys == c)).sum() for c in range(3)] for i in range(3)])
    r, cc = linear_sum_assignment(-C)
    mp = {i: cc[j] for j, i in enumerate(r)}
    pm = np.array([mp[l] for l in lab])
    M = np.array([[((ys == tc) & (pm == pc)).sum() / max((ys == tc).sum(), 1) for pc in range(3)]
                  for tc in range(3)])
    return (pm == ys).mean(), M


def main(figure_only=False):
    if figure_only:
        draw(pd.read_csv("runs/unsup/bifurcation_explore/null_ladder.csv"))
        return
    rows = []
    for nv in LADDER:
        Xn, ys = load_benchmark("logistic", N_PER, seed=0, null_version=nv)
        Z = StandardScaler().fit_transform(np.nan_to_num(build_features(Xn)))
        cents = np.vstack([Z[ys == c].mean(0) for c in range(3)])
        acc_t, Mt = hung(fit_t_mixture(Z, 3, seed=0, n_init=3)[0], ys)
        acc_s, _ = hung(fit_skew_t_mixture(Z, 3, seed=0, n_init=3)[0], ys)
        acc_o = (np.argmin(np.linalg.norm(Z[:, None, :] - cents[None], axis=2), axis=1) == ys).mean()
        conf = Mt[1, 2] + Mt[2, 1]
        sn_tc = Mt[0, 1] + Mt[1, 0]
        print(f"{str(nv):11s}: Student-t {100*acc_t:5.1f}%  skew-t {100*acc_s:5.1f}%  "
              f"oracle {100*acc_o:5.1f}%  TC<->null {100*conf:5.1f}  SN<->TC {100*sn_tc:4.1f}")
        rows.append(dict(stage=str(nv), acc_t=100*acc_t, acc_skewt=100*acc_s, acc_oracle=100*acc_o,
                         tcnull=100*conf, sntc=100*sn_tc))
    df = pd.DataFrame(rows)
    df.to_csv("runs/unsup/bifurcation_explore/null_ladder.csv", index=False)
    draw(df)


def draw(df):
    # optional supervised overlay (FeatMLP per ladder stage; second_attempt_deep/ladder_dl.py)
    import os
    dl_path = "runs/unsup/bifurcation_explore/null_ladder_dl.csv"
    dl = None
    if os.path.exists(dl_path):
        dl = pd.read_csv(dl_path, dtype={"stage": str}).set_index("stage")
        dl = dl.loc[[str(nv) for nv in LADDER]].reset_index()

    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.9), gridspec_kw=dict(wspace=0.35))
    xs = np.arange(len(LADDER))
    labels = [LADDER_LABEL[nv] for nv in LADDER]
    ax = axs[0]
    ax.plot(xs, df.acc_t, "o-", color="#457b9d", lw=1.6, ms=4.5, label="Student-$t$ (unsup.)")
    ax.plot(xs, df.acc_skewt, "s--", color="#2a9d8f", lw=1.2, ms=4, label="skew-$t$ (unsup.)")
    ax.plot(xs, df.acc_oracle, "^:", color="#9a6fb0", lw=1.2, ms=4.5, label="oracle (labels)")
    if dl is not None:
        ax.plot(xs, dl.acc_dl, "D-", color="#1d4e89", lw=1.4, ms=4, label="FeatMLP (superv.)")
    ax.axhline(100 / 3, color="#999", lw=0.7, ls=":")
    ax.text(2.55, 35, "chance", fontsize=5.8, color="#999")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=5.8)
    ax.set_ylabel("three-class accuracy (%)", fontsize=7); ax.set_ylim(28, 102)
    ax.legend(fontsize=5.8, loc="lower left")
    _tag(ax, "(a) accuracy as loopholes close", y=1.05)

    ax = axs[1]
    ax.plot(xs, df.tcnull, "o-", color="#e07a5f", lw=1.6, ms=4.5, label="TC$\\leftrightarrow$null (unsup.)")
    ax.plot(xs, df.sntc, "s--", color="#8d99ae", lw=1.2, ms=4, label="SN$\\leftrightarrow$TC (unsup.)")
    if dl is not None:
        ax.plot(xs, dl.tcnull_dl, "D-", color="#1d4e89", lw=1.4, ms=4,
                label="TC$\\leftrightarrow$null (FeatMLP)")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=5.8)
    ax.set_ylabel("summed cross-assignment (pp)", fontsize=7)
    ax.legend(fontsize=5.8, loc="upper left")
    _tag(ax, "(b) confusion concentrates on detection", y=1.05)

    import os
    SI = "Manuscript/SI_figures"; os.makedirs(SI, exist_ok=True)
    fig.savefig(f"{SI}/figS_null_ladder.png", bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {SI}/figS_null_ladder.png + runs/unsup/bifurcation_explore/null_ladder.csv")


if __name__ == "__main__":
    import sys
    main(figure_only="figure" in sys.argv)
