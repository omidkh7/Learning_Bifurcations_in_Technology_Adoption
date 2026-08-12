#!/usr/bin/env python3
"""
check3_sigma_confound.py — is the FeatMLP using the process-noise amplitude sigma as a
class cue?

In the headline generator both classes draw sigma from the same pool
{0.005, 0.01, 0.02, 0.05}, so sigma is not confounded with class BY CONSTRUCTION. But the
model leans on noise-structure features, and SN and TC trajectories have different
amplitudes, so the REALISED normalised noise could still differ systematically and act as
a cue. This checks it directly.

Two arms, both 2-class SN vs TC, all sets t50-matched:

  ARM 1 (our model). Train on the standard set (sigma drawn from the same pool for both
  classes). Test on three sets: sigma-matched control; SN-low/TC-high; SN-high/TC-low
  (swapped). If accuracy is stable across the swap, sigma is not a cue.

  ARM 2 (positive control). Deliberately train on a CONFOUNDED set (SN always low sigma,
  TC always high sigma), then test on the swapped set. This shows what sigma-exploitation
  actually looks like: a confounded model should collapse BELOW chance when the cue is
  inverted. Without this arm, a stable Arm 1 is not interpretable.

Writes runs/checks/check3_sigma_confound.json + fig_check3.png
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from checks_common import gen_pool_meta, SIGMA_CHOICES
from features import build_features46
from models import FeatMLP
from check2_noise_realism import match2, fit, probs
import train as T

CRUNS = os.path.join(ROOT, "results", "benchmark", "checks")
os.makedirs(CRUNS, exist_ok=True)

N_PER = 9000                 # t50-matching retains only ~15%, so pools must be large
LOW = (0.005, 0.01)          # "low" sigma pool
HIGH = (0.02, 0.05)          # "high" sigma pool
STD = SIGMA_CHOICES          # the standard pool used by the headline generator

SETS = {
    "standard (both classes, same pool)": (STD, STD),
    "sigma-matched control (both low)":   (LOW, LOW),
    "SN low / TC high":                   (LOW, HIGH),
    "SN high / TC low (swapped)":         (HIGH, LOW),
}


def build_set(sig_sn, sig_tc, seed):
    Xsn, tsn, ssn, *_ = gen_pool_meta(0, N_PER, seed, sigma_choices=sig_sn)
    Xtc, ttc, stc, *_ = gen_pool_meta(1, N_PER, seed + 1, sigma_choices=sig_tc)
    X, y = match2(Xsn, tsn, Xtc, ttc)
    F = build_features46(X).astype(np.float32)
    return F, y


def main():
    t0 = time.time()
    data = {}
    for name, (a, b) in SETS.items():
        data[name] = build_set(a, b, 601)
        print(f"built [{name}]: N={len(data[name][1])}  ({time.time()-t0:.0f}s)")

    res = {"arm1_our_model": {}, "arm2_confounded_control": {}}

    # ---------- ARM 1: our model (trained on the standard, unconfounded set) ----------
    Fs, ys = data["standard (both classes, same pool)"]
    itr, iva, ite = T.make_splits(ys, seed=700)
    mu, sd = Fs[itr].mean(0), Fs[itr].std(0) + 1e-9
    Fsz = np.clip((Fs - mu) / sd, -8, 8).astype(np.float32)
    ours = fit(Fsz, ys, itr, iva, "checks/c3_standard")

    print("\nARM 1 — model trained on the STANDARD (unconfounded) set; chance = 50%")
    for name, (F, y) in data.items():
        Fz = np.clip((F - mu) / sd, -8, 8).astype(np.float32)
        _, _, ite_c = T.make_splits(y, seed=700)
        acc = float((probs(ours, Fz[ite_c]).argmax(1) == y[ite_c]).mean())
        res["arm1_our_model"][name] = acc
        print(f"  tested on {name:36s} {acc*100:5.1f}%")

    a_lo = res["arm1_our_model"]["SN low / TC high"]
    a_hi = res["arm1_our_model"]["SN high / TC low (swapped)"]
    res["arm1_swap_gap_pts"] = float((a_lo - a_hi) * 100)

    # ---------- ARM 2: deliberately confounded model (positive control) ----------
    Fc, yc = data["SN low / TC high"]
    itr2, iva2, ite2 = T.make_splits(yc, seed=700)
    mu2, sd2 = Fc[itr2].mean(0), Fc[itr2].std(0) + 1e-9
    Fcz = np.clip((Fc - mu2) / sd2, -8, 8).astype(np.float32)
    conf = fit(Fcz, yc, itr2, iva2, "checks/c3_confounded")

    print("\nARM 2 — POSITIVE CONTROL: model deliberately trained on SN-low/TC-high")
    for name, (F, y) in data.items():
        Fz = np.clip((F - mu2) / sd2, -8, 8).astype(np.float32)
        _, _, ite_c = T.make_splits(y, seed=700)
        acc = float((probs(conf, Fz[ite_c]).argmax(1) == y[ite_c]).mean())
        res["arm2_confounded_control"][name] = acc
        print(f"  tested on {name:36s} {acc*100:5.1f}%")

    c_lo = res["arm2_confounded_control"]["SN low / TC high"]
    c_hi = res["arm2_confounded_control"]["SN high / TC low (swapped)"]
    res["arm2_swap_gap_pts"] = float((c_lo - c_hi) * 100)

    # ---------- verdict ----------
    if abs(res["arm1_swap_gap_pts"]) < 5 and res["arm2_swap_gap_pts"] > 15:
        v = (f"PASS: our model is sigma-invariant (swap gap {res['arm1_swap_gap_pts']:+.1f} pts) "
             f"while the deliberately confounded control swings {res['arm2_swap_gap_pts']:+.1f} pts, "
             f"proving the test can detect sigma-exploitation when it exists.")
    elif abs(res["arm1_swap_gap_pts"]) >= 5:
        v = (f"CONCERN: our model shifts {res['arm1_swap_gap_pts']:+.1f} pts when the sigma "
             f"assignment is swapped, so it is partly using noise amplitude as a cue.")
    else:
        v = (f"INCONCLUSIVE: our model looks sigma-invariant "
             f"({res['arm1_swap_gap_pts']:+.1f} pts) but the confounded control only moved "
             f"{res['arm2_swap_gap_pts']:+.1f} pts, so the test may lack power.")
    res["verdict"] = v
    print("\nVERDICT:", v)
    json.dump(res, open(f"{CRUNS}/check3_sigma_confound.json", "w"), indent=2)

    # ---------- figure ----------
    names = list(SETS.keys())
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(names))
    a1 = [res["arm1_our_model"][n] * 100 for n in names]
    a2 = [res["arm2_confounded_control"][n] * 100 for n in names]
    ax.bar(x - 0.2, a1, 0.4, color="#2a9d8f", alpha=0.9, label="our model (trained unconfounded)")
    ax.bar(x + 0.2, a2, 0.4, color="#e76f51", alpha=0.9,
           label="positive control (trained SN-low/TC-high)")
    ax.axhline(50, color="k", lw=0.9, ls=":", label="chance")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=18, ha="right", fontsize=7.5)
    ax.set_ylabel("2-class accuracy (%)"); ax.set_ylim(0, 105)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("Check 3: is process-noise amplitude a class cue?", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(f"{CRUNS}/fig_check3.png", dpi=180, bbox_inches="tight")
    print(f"saved {CRUNS}/check3_sigma_confound.json + fig_check3.png ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
