#!/usr/bin/env python3
"""
benchmark_data.py
=================
Single source of truth for the synthetic 3-class benchmark (SN / TC / Null). SN and TC come from the
canonical normal-form simulations (data/synthetic_optionc); the NULL class is selectable:

  clean per-shape nulls (the canonical benchmark uses 'logistic', the hardest TC-mimicking clean null):
    'linear'       straight ramp 0->1
    'exponential'  saturating 1-e^{-kt}
    'ramp_expsat'  linear ramp then exponential saturation
    'sigmoid'      generalised logistic (shifted)
    'logistic'     canonical symmetric logistic (the exact TC-solution shape)   <- DEFAULT
  open-set null (DEMOTED to a sensitivity check; Bury et al. 2021 / open-set-recognition style):
    'mixed'        the heterogeneous Option-C null (linear+expsat+sigmoid blended)

load_benchmark(null_kind) returns (Xn 100-grid min-max-normalised, ys) with 0=SN, 1=TC, 2=Null.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

CLEAN_NULLS = ["linear", "exponential", "ramp_expsat", "sigmoid", "logistic"]
CANONICAL_NULL = "logistic"


def make_null(kind, n, L, rng):
    t = np.linspace(0, 1, L); out = []
    for _ in range(n):
        if kind == "linear":
            x = t.copy()
        elif kind == "exponential":
            k = rng.uniform(3, 8); x = (1 - np.exp(-k * t)) / (1 - np.exp(-k))
        elif kind == "ramp_expsat":
            tb = rng.uniform(0.3, 0.6); xb = rng.uniform(0.4, 0.7); k = rng.uniform(5, 12)
            x = np.where(t < tb, (t / tb) * xb, xb + (1 - xb) * (1 - np.exp(-k * (t - tb))))
        elif kind == "sigmoid":
            k = rng.uniform(8, 20); t0 = rng.uniform(0.3, 0.7); x = 1 / (1 + np.exp(-k * (t - t0)))
        else:  # logistic: canonical symmetric logistic centred near 0.5 (exact TC-solution shape)
            k = rng.uniform(8, 14); t0 = rng.uniform(0.45, 0.55); x = 1 / (1 + np.exp(-k * (t - t0)))
        e = rng.normal(0, 1, L); eta = np.zeros(L)
        for i in range(1, L): eta[i] = 0.7 * eta[i - 1] + e[i]
        x = x + 0.03 * eta / (np.abs(eta).max() + 1e-9)
        out.append((x - x.min()) / (x.max() - x.min() + 1e-12))
    return np.array(out)


_CACHE = {}
def _load_optionc():
    if "X" not in _CACHE:
        _CACHE["X"] = np.load("data/synthetic_optionc/X_full.npy")
        _CACHE["y"] = np.load("data/synthetic_optionc/y.npy")
    return _CACHE["X"], _CACHE["y"]


# ---------------------------------------------------------------------------- null v2 (audit fix)
# v1 nulls were deterministic templates + POST-HOC AR(1) noise at a FIXED 0.03 amplitude, generated
# on the 500-grid (so downsampling to 100 points destroyed the noise persistence), and the 'logistic'
# null's (k, t0) did not overlap the transcritical simulations' effective logistic parameters. Any of
# these lets noise TEXTURE or parameter mismatch identify the null without dynamics (audit, 2026-07).
# v2 (default): the CANONICAL stable twin (see _make_null_v2). Each null is a template-following SDE
# path built on the classes' native 15-80-step grid, integrated through the same Euler-Maruyama
# recursion with sigma drawn from the class palette [0.005..0.20] and AR(1) colour phi ~ U[0, 0.9],
# with the same validity gates (including the visible-plateau rule); the 'logistic' null's (k, t0)
# are sampled from logistic fits to the actual TC simulations. Only the dynamics differ from TC.
# NULL_VERSION = 1 reproduces the old post-hoc-noise behaviour for the construction ladder.
NULL_VERSION = 2

# Null-construction LADDER (each stage closes one non-dynamical loophole; all selectable through
# load_benchmark(null_version=...) so the ladder is fully reproducible):
#   1            post-hoc stationary noise on templates (original)      -> texture channel open
#   "walk"       integrated noise, no restoring force, minimal gates    -> params/texture closed
#   "walk_gated" integrated walk + endpoint rise/saturation gates       -> broken curves removed
#   2            stable twin (OU-tracking, class-matched scale + gates) -> CANONICAL: dynamics only
LADDER = [1, "walk", "walk_gated", 2]
LADDER_LABEL = {1: "post-hoc\nnoise", "walk": "integrated\nwalk", "walk_gated": "walk +\nendpoint gates",
                2: "stable twin\n(canonical)"}


def _to100(Xs):
    L = Xs.shape[1]
    g, g100 = np.linspace(0, 1, L), np.linspace(0, 1, 100)
    Xr = np.stack([PchipInterpolator(g, c)(g100) for c in Xs])
    return (Xr - Xr.min(1, keepdims=True)) / (np.ptp(Xr, 1, keepdims=True) + 1e-12)


def _tc_param_pool(n_ref=800, seed=54321):
    """(k, t0) pool from logistic fits to actual TC simulations on the normalised 100-grid."""
    if "tcp" in _CACHE: return _CACHE["tcp"]
    from scipy.optimize import curve_fit
    X, y = _load_optionc(); rng = np.random.default_rng(seed)
    Xn = _to100(X[rng.choice(np.where(y == 1)[0], n_ref, replace=False)])
    t = np.linspace(0, 1, 100)
    ks, t0s = [], []
    def logi(x, t0, k): return 1.0 / (1.0 + np.exp(-k * (x - t0)))
    for xr in Xn:
        try:
            p, _ = curve_fit(logi, t, xr, p0=[0.3, 20], maxfev=2000,
                             bounds=([0.0, 1.0], [1.0, 200.0]))
            t0s.append(p[0]); ks.append(p[1])
        except Exception:
            continue
    _CACHE["tcp"] = (np.array(ks), np.array(t0s))
    return _CACHE["tcp"]


_NOISE_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]   # the SN/TC generator's palette


def _make_null_v2(kind, n, rng, variant="stable"):
    """Clean nulls built by the SAME stochastic mechanism as the SN/TC classes: a template-following
    drift INTEGRATED with the identical Euler-Maruyama noise recursion (sigma from the class palette,
    AR(1) colour phi ~ U[0, 0.9], native 15-80-step grid, then PCHIP to 500 like the classes). The
    noise accumulates exactly as in the class simulations, so texture cannot identify the null; there
    is NO state feedback (the drift follows the template, independent of x), so there is no critical
    slowing down by construction. The 'logistic' null's (k, t0) are sampled from logistic fits to the
    actual TC simulations, closing the parameter-mismatch channel."""
    ks, t0s = _tc_param_pool()
    out = []
    while len(out) < n:
        n_steps = int(rng.integers(15, 80))
        t_end = rng.uniform(5.0, 20.0)
        tn = np.linspace(0, 1, n_steps)                       # normalised time for the template
        ki = (["linear", "exponential", "ramp_expsat", "sigmoid", "logistic"][rng.integers(5)]
              if kind == "mixed" else kind)                    # open-set: heterogeneous shape per series,
                                                               # INCLUDING the TC-matched logistic
        if ki == "linear":
            tpl = tn.copy()
        elif ki == "exponential":
            k = rng.uniform(3, 8); tpl = (1 - np.exp(-k * tn)) / (1 - np.exp(-k))
        elif ki == "ramp_expsat":
            tb = rng.uniform(0.3, 0.6); xb = rng.uniform(0.4, 0.7); k = rng.uniform(5, 12)
            tpl = np.where(tn < tb, (tn / tb) * xb, xb + (1 - xb) * (1 - np.exp(-k * (tn - tb))))
        elif ki == "sigmoid":
            k = rng.uniform(8, 20); t0 = rng.uniform(0.3, 0.7); tpl = 1 / (1 + np.exp(-k * (tn - t0)))
        else:  # 'logistic': parameters from fits to the ACTUAL TC simulations
            j = rng.integers(len(ks))
            k = ks[j] * rng.uniform(0.9, 1.1)
            t0 = np.clip(t0s[j] + rng.normal(0, 0.02), 0.02, 0.98)
            tpl = 1 / (1 + np.exp(-k * (tn - t0)))
        if variant in ("walk", "walk_gated"):
            # ladder stages: integrated noise WITHOUT a restoring force (unit-height template)
            sigma = rng.choice(_NOISE_LEVELS); phi = rng.uniform(0.0, 0.9)
            dt = t_end / (n_steps - 1)
            x = np.zeros(n_steps); x[0] = tpl[0]; eta = 0.0
            for i in range(1, n_steps):
                eta = phi * eta + rng.normal(0.0, 1.0)
                x[i] = x[i - 1] + (tpl[i] - tpl[i - 1]) + sigma * eta * np.sqrt(dt)
            if not np.isfinite(x).all(): continue
            if variant == "walk":
                if x.max() - x.min() < 0.05: continue
            else:  # walk_gated: endpoint rise/saturation gates (unit template target)
                if x[-1] - x[0] < 0.3: continue
                if x[-1] < 0.7 or x[-1] > 1.6: continue
                if x[-1] - x.min() < 0.7 * (x.max() - x.min()): continue
            xn = (x - x.min()) / (x.max() - x.min())
            g500 = np.linspace(0, 1, 500)
            out.append(PchipInterpolator(tn, xn)(g500))
            continue

        # The null is TC's STABLE-PHASE TWIN: draw the same (mu, a2) as the transcritical sampler,
        # scale the template to the same physical height x_ref = mu/a2, relax toward it at the same
        # constant rate lam = mu (a stable, non-bifurcating process; no critical slowing down since
        # lam never approaches zero and there is no unstable-origin departure), drive it with the
        # same raw-unit noise palette, apply the same raw-unit validity gates, then normalise. This
        # matches amplitude scale, relaxation scale, and noise mechanism simultaneously; earlier
        # versions missed one of the three (post-hoc noise; no restoring force; unit-height template).
        mu = rng.uniform(0.3, 4.0); a2 = rng.uniform(0.8, 2.0)
        x_ref = mu / a2; lam = mu
        sigma = rng.choice(_NOISE_LEVELS)
        phi = rng.uniform(0.0, 0.9)
        dt = t_end / (n_steps - 1)
        tplS = x_ref * tpl
        x = np.zeros(n_steps); x[0] = tplS[0]; eta = 0.0
        for i in range(1, n_steps):
            eta = phi * eta + rng.normal(0.0, 1.0)
            x[i] = tplS[i] + (x[i - 1] - tplS[i - 1]) * np.exp(-lam * dt) + sigma * eta * np.sqrt(dt)
        # validity gates MIRRORING the SN/TC generator, in raw units, INCLUDING the visible-plateau
        # gate (last 15% of samples within 5% of the maximum) that the class generator applies; the
        # classes' surviving realisations have small late-time noise by this selection, so the null
        # must face the same filter.
        if not np.isfinite(x).all(): continue
        if x[-1] - x[0] < 0.3 * x_ref: continue
        if x[-1] < 0.7 * x_ref or x[-1] > 1.6 * x_ref: continue
        if x[-1] - x.min() < 0.7 * (x.max() - x.min()): continue
        n_tail = max(3, int(0.15 * n_steps)); xmax = float(x.max())
        if xmax < 1e-8 or np.any(np.abs(x[-n_tail:] - xmax) / xmax > 0.05): continue
        xn = (x - x.min()) / (x.max() - x.min())
        g500 = np.linspace(0, 1, 500)
        out.append(PchipInterpolator(tn, xn)(g500))            # same native->500 resampling as classes
    return np.array(out)


def load_benchmark(null_kind=CANONICAL_NULL, n_per=1000, seed=0, null_version=None):
    X, y = _load_optionc(); rng = np.random.default_rng(seed); L = X.shape[1]
    nv = NULL_VERSION if null_version is None else null_version
    sn = X[rng.choice(np.where(y == 0)[0], n_per, replace=False)]
    tc = X[rng.choice(np.where(y == 1)[0], n_per, replace=False)]
    sn100, tc100 = _to100(sn), _to100(tc)
    if nv == 2:
        nu100 = _to100(_make_null_v2(null_kind, n_per, rng))   # canonical stable twin (incl. 'mixed')
    elif nv in ("walk", "walk_gated"):
        nu100 = _to100(_make_null_v2(null_kind, n_per, rng, variant=nv))   # ladder stages
    elif null_kind == "mixed":
        nu100 = _to100(X[rng.choice(np.where(y == 2)[0], n_per, replace=False)])  # v1 open-set (npy)
    else:
        nu100 = _to100(make_null(null_kind, n_per, L, rng))
    Xn = np.vstack([sn100, tc100, nu100])
    ys = np.r_[np.zeros(n_per), np.ones(n_per), 2 * np.ones(n_per)].astype(int)
    return Xn.astype(np.float32), ys
