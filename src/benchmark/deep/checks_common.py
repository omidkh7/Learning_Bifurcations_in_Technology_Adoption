#!/usr/bin/env python3
"""
checks_common.py — shared generation utilities for the consistency checks.

Re-implements the §71 wide-parameter SN/TC generation (same ODEs, same wide ranges, same
validity gates as matched_inflection_experiment.gen_pool) but exposes the knobs the checks
need and that gen_pool hides:

  * the per-curve process-noise amplitude sigma (check 3: sigma confound)
  * a deterministic mode (sigma = 0) plus post-hoc OBSERVATION noise added at the annual
    sampling resolution (check 2: measurement-noise realism)
  * the raw pre-interpolation trajectory and its sampling length

Design note on what "observation noise" means here. Process noise enters the SDE and feeds
back through the dynamics, so its variance grows as the system approaches the fold: that is
critical slowing down, the signal the FeatMLP actually uses. Observation noise is added to
the finished deterministic solution, so it has constant variance and no CSD by
construction. Real reported adoption series carry the second kind, not the first (§72d/§73).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from Synthetic_Data_Gen import (saddle_node_rhs, transcritical_rhs, integrate_sde,
                                normalize_series, pchip_to_500, is_valid_trajectory,
                                has_visible_plateau)

T = 500
TGRID = np.linspace(0, 1, T)
SIGMA_CHOICES = (0.005, 0.01, 0.02, 0.05)      # the standard pool used by gen_pool


def t50_of(x):
    x = (x - x.min()) / (x.max() - x.min() + 1e-9)
    return float(TGRID[np.argmax(x >= 0.5)]) if (x >= 0.5).any() else 1.0


def draw_params(label, rng):
    """Same wide parameter draws as matched_inflection_experiment.gen_pool."""
    n_steps = int(rng.integers(20, 80))
    t_end = rng.uniform(5.0, 20.0)
    tt = np.linspace(0, t_end, n_steps)
    phi = rng.uniform(0.0, 0.9)
    if label == 0:                                  # saddle-node, wide sweep speed
        r = rng.uniform(0.05, 0.35)
        a3 = rng.uniform(-0.05, 0.05)
        speed = rng.uniform(0.1, 9.0)
        mu_sweep = np.linspace(-speed * r ** 2, r ** 2, n_steps)
        x0 = -np.sqrt(abs(-speed * r ** 2)) * rng.uniform(0.9, 1.1)
        return dict(kind="sn", tt=tt, phi=phi, a3=a3, mu_sweep=mu_sweep, x0=x0, xref=r)
    mu = rng.uniform(0.1, 6.0)                      # transcritical, wide rate
    a2 = rng.uniform(0.8, 2.0)
    a3 = rng.uniform(-0.5, 0.5)
    x0 = rng.uniform(0.005, 0.05)
    return dict(kind="tc", tt=tt, phi=phi, mu=mu, a2=a2, a3=a3, x0=x0, xref=mu / a2)


def integrate(p, sigma, rng):
    if p["kind"] == "sn":
        return integrate_sde(saddle_node_rhs, p["x0"], p["tt"], {"a2": 1.0, "a3": p["a3"]},
                             sigma, rng, phi_ar=p["phi"], mu_sweep=p["mu_sweep"])
    return integrate_sde(transcritical_rhs, p["x0"], p["tt"],
                         {"mu": p["mu"], "a2": p["a2"], "a3": p["a3"]},
                         sigma, rng, phi_ar=p["phi"], mu_sweep=None)


def valid(p, x):
    if not is_valid_trajectory(x, p["xref"]):
        return False
    if p["kind"] == "sn" and x.max() < 4.0:
        return False
    return True


def add_observation_noise(x_raw, amp, rng, phi=0.0):
    """Add constant-variance noise to a FINISHED trajectory at its own sampling resolution.

    No feedback through the dynamics, so no critical slowing down: variance is flat in time
    by construction. amp is relative to the normalised [0, 1] range.
    """
    xn = normalize_series(x_raw)
    n = len(xn)
    e = rng.normal(0, 1, n)
    if phi > 0:
        eta = np.zeros(n)
        for i in range(1, n):
            eta[i] = phi * eta[i - 1] + e[i]
        e = eta / (np.abs(eta).max() + 1e-9)
    return xn + amp * e


def to_grid(x_raw, grid=T):
    """Normalise then PCHIP-interpolate to the target grid (the real-data pipeline shape)."""
    xn = normalize_series(x_raw)
    if grid == T:
        return pchip_to_500(xn).astype(np.float32)
    src = np.linspace(0, 1, len(xn))
    xs = PchipInterpolator(src, xn, extrapolate=False)(np.linspace(0, 1, grid))
    b = np.isnan(xs)
    if b.any():
        xs[b] = np.interp(np.where(b)[0], np.where(~b)[0], xs[~b])
    xs = np.clip(xs, 0, 1)
    return ((xs - xs.min()) / (xs.max() - xs.min() + 1e-12)).astype(np.float32)


def gen_pool_meta(label, n, seed, sigma_choices=SIGMA_CHOICES, obs_noise=None,
                  n_obs=None, grid=T, phi_obs=0.0):
    """Generate n valid curves of `label` (0=SN, 1=TC), returning curves + metadata.

    sigma_choices : pool of PROCESS-noise amplitudes to draw from (set (0.0,) for
                    deterministic trajectories).
    obs_noise     : if not None, (lo, hi) range for post-hoc OBSERVATION-noise amplitude,
                    drawn per curve and added after integration.
    n_obs         : if set, subsample the trajectory to this many evenly spaced
                    "annual" points before interpolating to the grid (real-data sparsity).
    grid          : output grid length (500 synthetic default, 100 = the real-data grid).

    Returns X (n, grid), t50 (n,), sigma (n,), obs_amp (n,), nobs (n,).
    """
    rng = np.random.default_rng(seed)
    X, ts, sig, oamp, nob = [], [], [], [], []
    while len(X) < n:
        p = draw_params(label, rng)
        s = float(rng.choice(sigma_choices))
        x = integrate(p, s, rng)
        if not valid(p, x):
            continue
        a = 0.0
        if obs_noise is not None:
            a = float(rng.uniform(*obs_noise))
            x = add_observation_noise(x, a, rng, phi=phi_obs)
        if n_obs is not None and n_obs < len(x):
            idx = np.linspace(0, len(x) - 1, n_obs).round().astype(int)
            x = np.asarray(x)[idx]
        xg = to_grid(x, grid)
        if grid == T and not has_visible_plateau(xg):
            continue
        X.append(xg)
        ts.append(t50_of(xg)); sig.append(s); oamp.append(a); nob.append(len(x))
    return (np.array(X), np.array(ts), np.array(sig), np.array(oamp), np.array(nob))


def load_mixed_null(n, seed, grid=T):
    """The SI's heterogeneous Option-C null (class 2 of data/synthetic_optionc)."""
    X = np.load(os.path.join(ROOT, "data/curated/synthetic_optionc/X_full.npy"))
    y = np.load(os.path.join(ROOT, "data/curated/synthetic_optionc/y.npy"))
    idx = np.where(y == 2)[0]
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, min(n, len(idx)), replace=False)
    Xs = X[pick]
    if grid != T:
        Xs = np.stack([to_grid(c, grid) for c in Xs])
    Xs = (Xs - Xs.min(1, keepdims=True)) / (np.ptp(Xs, 1, keepdims=True) + 1e-12)
    Xs = Xs.astype(np.float32)
    return Xs, np.array([t50_of(c) for c in Xs])
