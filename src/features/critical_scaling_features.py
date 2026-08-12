#!/usr/bin/env python3
"""
critical_scaling_features.py
============================
A compact, physics-grounded "critical-scaling / phase-transition" feature group, motivated by
Corral, Sardanyes & Alseda (2018) and the saddle-node<->first-order / transcritical<->second-order
correspondence. These target what IS recoverable from a single normalised adoption curve (the
finite-time-scaling EXPONENT itself is not -- see finite_time_scaling.py), namely the velocity-field
geometry, the order-parameter onset and approach exponents, and the critical-slowing-down rate.

Input: X (N, L) PCHIP-normalised rising curves on a uniform grid (same X as extract_theory_features).
Output: F (N, 8) and CRIT_NAMES.
"""
import numpy as np
from scipy.stats import kendalltau, skew

CRIT_NAMES = [
    "onset_beta",          # order-parameter onset exponent: x ~ (t-t0)^beta near takeoff (1=linear/TC, >1=algebraic/SN)
    "approach_expfit",     # R^2 of exponential approach to plateau (high=TC exponential relaxation, low=algebraic/SN)
    "percap_linearity",    # R^2 of per-capita growth g=v/x vs x (high=logistic/TC; low=nonlinear/SN ghost)
    "inflection_penetration",  # x at peak velocity (0.5=logistic/TC, higher=back-loaded/SN ghost)
    "velocity_asymmetry",  # skewness of velocity profile over the rise (0=symmetric/TC, +=back-loaded/SN)
    "takeoff_abruptness",  # max(v)/mean(v): high=sharp near-discontinuous jump (1st-order/fold), low=gradual
    "csd_var_rate",        # Kendall-tau of rolling variance over the pre-takeoff phase (CSD divergence rate)
    "fold_dwell",          # fraction of time spent below 10% of peak (saddle-node ghost bottleneck)
]


def _safe(v, fb=0.0):
    return fb if (v is None or not np.isfinite(v)) else float(v)


def _r2(y, yhat):
    ss = np.sum((y - y.mean()) ** 2)
    return _safe(1.0 - np.sum((y - yhat) ** 2) / ss) if ss > 1e-12 else 0.0


def _one(x):
    n = len(x); t = np.linspace(0.0, 1.0, n)
    x = np.clip((x - x.min()) / (x.max() - x.min() + 1e-12), 0.0, 1.0)
    s = np.maximum.accumulate(x)                       # monotone-cleaned rise for shape measures
    v = np.gradient(s, t)                              # velocity field dx/dt
    vpos = v[v > 1e-9]

    # 1. onset exponent beta: log x ~ beta log(t - t0) in the early window x in [0.02, 0.25]
    i0 = int(np.argmax(s > 0.02)); t0 = t[i0]
    m = (s > 0.02) & (s < 0.25) & (t > t0)
    if m.sum() >= 4:
        beta = np.polyfit(np.log(t[m] - t0 + 1e-6), np.log(s[m] + 1e-9), 1)[0]
    else:
        beta = 1.0

    # 2. exponential-approach R^2: log(1-x) ~ linear in t over the saturation window x in [0.7, 0.985]
    d = 1.0 - s; m = (s > 0.7) & (s < 0.985) & (d > 1e-6)
    if m.sum() >= 4:
        p = np.polyfit(t[m], np.log(d[m]), 1); approach = _r2(np.log(d[m]), np.polyval(p, t[m]))
    else:
        approach = 0.0

    # 3. per-capita growth linearity: g = v/x vs x, fit linear, report R^2 (logistic/TC ~ 1)
    m = (s > 0.05) & (s < 0.9) & (v > 1e-9)
    if m.sum() >= 5:
        g = v[m] / s[m]; p = np.polyfit(s[m], g, 1); percap = _r2(g, np.polyval(p, s[m]))
    else:
        percap = 0.0

    # 4. inflection penetration: x at max velocity
    infl_pen = _safe(s[int(np.argmax(v))], 0.5)

    # 5. velocity asymmetry: skewness of v over the active rise (x in [0.02, 0.98])
    rise = v[(s > 0.02) & (s < 0.98)]
    vskew = _safe(skew(rise), 0.0) if len(rise) >= 4 else 0.0

    # 6. takeoff abruptness: peak velocity relative to mean velocity
    abrupt = _safe(v.max() / vpos.mean(), 1.0) if len(vpos) else 1.0

    # 7. CSD variance-rise rate: Kendall-tau of rolling variance over the pre-takeoff phase (x < 0.6)
    w = max(5, n // 8); pre = np.where(s < 0.6)[0]
    if len(pre) > 2 * w:
        rv = np.array([x[i:i + w].var() for i in pre[:-w]])
        csd = _safe(kendalltau(np.arange(len(rv)), rv)[0], 0.0) if len(rv) >= 4 else 0.0
    else:
        csd = 0.0

    # 8. fold dwell: fraction of time below 10% of peak (saddle-node ghost bottleneck)
    fold = _safe(np.mean(s < 0.10), 0.0)

    return np.array([beta, approach, percap, infl_pen, vskew, abrupt, csd, fold], dtype=np.float64)


def extract_critical_features(X, verbose=False):
    X = np.asarray(X, float); F = np.zeros((len(X), len(CRIT_NAMES)))
    for i in range(len(X)):
        try:
            F[i] = _one(X[i])
        except Exception as e:
            if verbose: print(f"  crit-feature fail {i}: {e}")
    for j in range(F.shape[1]):                          # fill non-finite with column median
        bad = ~np.isfinite(F[:, j])
        if bad.any():
            good = F[~bad, j]; F[bad, j] = float(np.median(good)) if len(good) else 0.0
    return F


if __name__ == "__main__":
    from paper_figures import load_four_group
    grp, F38, X = load_four_group()
    Fc = extract_critical_features(X, verbose=True)
    print(f"critical features: {Fc.shape}  names={CRIT_NAMES}")
    import numpy as np
    for f in ["Historical", "Renewables", "BEV", "CDR"]:
        med = np.median(Fc[grp == f], axis=0)
        print(f"  {f:11s} " + "  ".join(f"{n.split('_')[0][:5]}={m:+.2f}" for n, m in zip(CRIT_NAMES, med)))
