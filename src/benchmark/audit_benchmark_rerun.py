#!/usr/bin/env python3
"""
audit_benchmark_rerun.py
========================
AUDIT RESPONSE (Mahdi, Tier-1 #1): rerun the synthetic benchmark with the v2 null (texture-matched
noise on the final grid + TC-parameter-matched logistic) against the old v1 null, WITHOUT touching
any manuscript figure. Also reproduces the audit's two-texture-feature probe (residual std + lag-1
AC after Gaussian detrend): if those two shape-free features alone separate null-vs-rest, the null
is identified by noise texture, not dynamics.

Output: console table + results/unsup/bifurcation_explore/audit_benchmark_rerun.csv
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from paper_figures import build_features
from benchmark_data import load_benchmark, CLEAN_NULLS
from unsup_real_world import fit_t_mixture, fit_skew_t_mixture

N_PER = 800


def hung(lab, ys, k=3):
    C = np.array([[((lab == i) & (ys == c)).sum() for c in range(k)] for i in range(k)])
    r, cc = linear_sum_assignment(-C)
    mp = {i: cc[j] for j, i in enumerate(r)}
    pm = np.array([mp[l] for l in lab])
    M = np.array([[((ys == tc) & (pm == pc)).sum() / max((ys == tc).sum(), 1) for pc in range(k)]
                  for tc in range(k)])
    return (pm == ys).mean(), M


def texture_probe(Xn, ys):
    """Audit probe: nearest-true-centroid on ONLY (residual std, residual lag-1 AC)."""
    F = []
    for x in Xn:
        r = x - gaussian_filter1d(x, sigma=12)
        a = np.corrcoef(r[:-1], r[1:])[0, 1] if r.std() > 1e-10 else 0.0
        F.append([r.std(), a if np.isfinite(a) else 0.0])
    Z = StandardScaler().fit_transform(np.array(F))
    cents = np.vstack([Z[ys == c].mean(0) for c in range(3)])
    pred = np.argmin(np.linalg.norm(Z[:, None, :] - cents[None], axis=2), axis=1)
    nullrest = ((pred == 2) == (ys == 2)).mean()
    return (pred == ys).mean(), nullrest


def main():
    rows = []
    for nv in (1, 2):
        print(f"\n================ NULL VERSION {nv} {'(old)' if nv == 1 else '(texture+parameter matched)'} ================")
        for kind in CLEAN_NULLS:
            Xn, ys = load_benchmark(kind, N_PER, seed=0, null_version=nv)
            Z = StandardScaler().fit_transform(np.nan_to_num(build_features(Xn)))
            cents = np.vstack([Z[ys == c].mean(0) for c in range(3)])
            acc_t, Mt = hung(fit_t_mixture(Z, 3, seed=0, n_init=3)[0], ys)
            acc_s, Ms = hung(fit_skew_t_mixture(Z, 3, seed=0, n_init=3)[0], ys)
            pred_o = np.argmin(np.linalg.norm(Z[:, None, :] - cents[None], axis=2), axis=1)
            acc_o = (pred_o == ys).mean()
            acc2, nullrest2 = texture_probe(Xn, ys)
            tcnull = Mt[1, 2] + Mt[2, 1]
            print(f"  {kind:12s}: Student-t {100*acc_t:5.1f}%  skew-t {100*acc_s:5.1f}%  "
                  f"oracle {100*acc_o:5.1f}%  | TC<->null conf {100*tcnull:5.1f}  "
                  f"| texture-probe 3cls {100*acc2:4.1f}%, null-vs-rest {100*nullrest2:5.1f}%")
            rows.append(dict(null_version=nv, null=kind, acc_t=100*acc_t, acc_skewt=100*acc_s,
                             acc_oracle=100*acc_o, tcnull_conf=100*tcnull,
                             texture_3class=100*acc2, texture_nullrest=100*nullrest2))
    pd.DataFrame(rows).to_csv("results/unsup/bifurcation_explore/audit_benchmark_rerun.csv", index=False)
    print("\nSaved -> results/unsup/bifurcation_explore/audit_benchmark_rerun.csv")


if __name__ == "__main__":
    main()
