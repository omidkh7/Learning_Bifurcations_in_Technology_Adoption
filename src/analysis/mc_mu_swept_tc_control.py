#!/usr/bin/env python3
"""
ANALYSIS ONLY (Mahdi review 1.3). The mu-observed experiment gives SN a swept mu(t) and TC a
CONSTANT mu, so 3 of the 7 mu-features are pure "did the drive move" indicators and the x+mu arm
could separate the classes by sweep-detection rather than by the scaling structure the experiment
claims to probe. Control: rerun with TC ALSO given a rising mu(t) (kept strictly positive, i.e.
within the bifurcating regime), so "did mu move" no longer distinguishes the classes. If the x+mu
arm stays high on SN-vs-TC-swept, it is reading scaling structure; if it collapses toward the
x-only level, it was reading the sweep.

Writes nothing the manuscript uses; prints a table. Canonical noise regime, N per class configurable.
Run from repo root.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from mc_mu_observed import (SIGMAS, sample_saddle_node, sample_transcritical, integrate_sde,
                            saddle_node_rhs, transcritical_rhs, typing_arm, N_PER as _N)

N_PER = 3000
REGIME = "canonical"


def _gates(x, x_ref):
    if not np.isfinite(x).all() or np.abs(x).max() > 9: return False
    if x[-1] - x[0] < 0.3 * abs(x_ref): return False
    if x[-1] < 0.7 * x_ref or x[-1] > 1.6 * x_ref: return False
    return True


def gen_one(label, rng, tc_swept):
    """label 0 = saddle-node (swept, as in the paper); label 1 = transcritical, either the paper's
    CONSTANT mu (tc_swept=False) or a rising positive sweep 0.3*mu -> mu (tc_swept=True)."""
    n = int(rng.integers(15, 80)); t_end = rng.uniform(5.0, 20.0); t = np.linspace(0, t_end, n)
    sigma = rng.choice(SIGMAS[REGIME]); phi = rng.uniform(0.0, 0.9)
    if label == 0:
        params, x0, x_ref, mu_sweep = sample_saddle_node(rng, n, t_end)
        x = integrate_sde(saddle_node_rhs, x0, t, params, sigma, rng, phi, mu_sweep)
        mu = mu_sweep.copy()
    else:
        params, x0, x_ref, _ = sample_transcritical(rng, n, t_end)
        if tc_swept:
            mu_sweep = np.linspace(0.3 * params["mu"], params["mu"], n)   # rising, strictly > 0
            x = integrate_sde(transcritical_rhs, x0, t, params, sigma, rng, phi, mu_sweep)
            mu = mu_sweep.copy()
        else:
            x = integrate_sde(transcritical_rhs, x0, t, params, sigma, rng, phi, None)
            mu = np.full(n, params["mu"])
    if not _gates(x, x_ref): return None
    return dict(t=t, x=x, mu=mu, y=label)


def gen(tc_swept, seed):
    rng = np.random.default_rng(seed); out = []
    for label in (0, 1):
        k = 0; tries = 0
        while k < N_PER and tries < N_PER * 50:
            tries += 1
            s = gen_one(label, rng, tc_swept)
            if s is not None: out.append(s); k += 1
    return out


def main():
    print(f"swept-mu-TC control (canonical regime, N_PER={N_PER})\n" + "=" * 70, flush=True)
    for tc_swept, name in [(False, "BASELINE  SN(swept) vs TC(constant)  [paper]"),
                           (True,  "CONTROL   SN(swept) vs TC(swept, rising mu>0)")]:
        series = gen(tc_swept, seed=0)
        n = len(series)
        print(f"\n### {name}   (N={n})", flush=True)
        print(f"    {'featset':8s} {'skew-t(unsup)':>14s} {'oracle':>9s}")
        res = {}
        for with_mu, fn, abl in [(False, "x only", ()), (True, "x + mu", ()),
                                 (True, "x+mu(scal)", (2, 6))]:
            accs, _ = typing_arm(series, seed=0, with_mu=with_mu, ablate_mu=abl)
            res[fn] = accs
            print(f"    {fn:11s} {100*accs['skew-t']:11.1f}% {100*accs['oracle']:8.1f}%", flush=True)
        gain_u = 100 * (res["x + mu"]["skew-t"] - res["x only"]["skew-t"])
        gain_o = 100 * (res["x + mu"]["oracle"] - res["x only"]["oracle"])
        gain_o_scal = 100 * (res["x+mu(scal)"]["oracle"] - res["x only"]["oracle"])
        print(f"    --> mu gain: unsup {gain_u:+.1f} pp, oracle {gain_o:+.1f} pp; "
              f"scaling-only (ablate sign feats 2,6) oracle {gain_o_scal:+.1f} pp", flush=True)
    print("\n" + "=" * 70)
    print("Read: if the CONTROL x+mu gain is close to the BASELINE gain, the mu arm reads scaling")
    print("structure, not just the sweep. If it collapses to ~0, it was sweep-detection. The")
    print("scaling-only column ablates the two features (fraction-of-time-mu<0, signed-deepest-mu) that")
    print("separate a swept SN from a positive-swept TC by construction; a positive CONTROL scaling-only")
    print("gain shows the oracle reads the fold's scaling structure, not merely the sign of the sweep.")


if __name__ == "__main__":
    main()
