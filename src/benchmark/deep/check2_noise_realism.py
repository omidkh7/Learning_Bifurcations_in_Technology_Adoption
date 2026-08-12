#!/usr/bin/env python3
"""
check2_noise_realism.py — does the FeatMLP survive REAL-DATA OBSERVATION conditions, or
does it start reading noise level as class? (the §72d / §73 worry)

DESIGN NOTE — why this is not "process noise vs observation noise".
The first version of this check tried to replace process noise with observation noise by
generating deterministic (sigma = 0) trajectories. That is ill-posed, and the failure is
itself a result (reported as `deterministic_probe` below): with sigma = 0 the saddle-node
passes its validity gate only ~0.5% of the time and its t50 collapses to a single value
(~0.82), because THE SADDLE-NODE GHOST IS ESCAPED BY NOISE. Process noise is constitutive
of the SN class, not an observation layer that can be stripped off. (TC is ~99% valid
either way — it needs no noise to leave the unstable origin.)

So the answerable question is about the OBSERVATION MODEL applied to physically valid
trajectories: real adoption series are sparse annual samples, PCHIP-interpolated and
effectively smoothed, with reporting error on top. Does the observable CSD survive that?

All conditions are PAIRED: one base set of process-noise trajectories is generated and
t50-matched ONCE, then every condition re-observes THE SAME trajectories. Identical N,
identical dynamics, only the observation changes. 2-class SN vs TC (the null is not what
§72d is about).

  A. OBSERVATION LADDER — each condition scored two ways:
       transfer = the reference FeatMLP trained on the full-resolution condition
       retrain  = a FeatMLP retrained on that condition (does ANY signal survive?)
  B. PAIRED NOISE-AMPLITUDE TEST — the same trajectories re-observed with increasing
     added observation noise. If P(SN) climbs with amplitude regardless of true class,
     the model reads noise level as SN.

Writes runs/checks/check2_noise_realism.json + fig_check2.png
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from checks_common import (draw_params, integrate, valid, add_observation_noise,
                           to_grid, t50_of, SIGMA_CHOICES)
from Synthetic_Data_Gen import has_visible_plateau, normalize_series
from features import build_features46
from models import FeatMLP
import train as T

CRUNS = os.path.join(HERE, "runs", "checks")
os.makedirs(CRUNS, exist_ok=True)

N_PER = 9000
AMPS = [0.0, 0.01, 0.02, 0.04, 0.08]
SEED = 42


# ---------------------------------------------------------------- observation models
def obs_full(x, rng):
    return to_grid(x)


def _sparse(x, n):
    idx = np.linspace(0, len(x) - 1, min(n, len(x))).round().astype(int)
    return np.asarray(x)[idx]


def obs_sparse30(x, rng):
    return to_grid(_sparse(x, 30))


def obs_sparse15(x, rng):
    return to_grid(_sparse(x, 15))


def obs_smooth30(x, rng):
    s = _sparse(x, 30)
    return to_grid(gaussian_filter1d(normalize_series(s), sigma=max(1.0, len(s) / 8)))


def obs_noisy30(x, rng):
    return to_grid(_sparse(add_observation_noise(x, 0.03, rng), 30))


def obs_reallike(x, rng):
    """Sparse annual sampling + reporting error + smoothing: the full real-data pipeline."""
    s = _sparse(add_observation_noise(x, 0.03, rng), 30)
    return to_grid(gaussian_filter1d(normalize_series(s), sigma=max(1.0, len(s) / 8)))


CONDITIONS = {
    "full resolution (train-like)": obs_full,
    "sparse 30 pts (annual)":       obs_sparse30,
    "sparse 15 pts":                obs_sparse15,
    "sparse 30 + smoothed":         obs_smooth30,
    "sparse 30 + obs noise 0.03":   obs_noisy30,
    "real-like (sparse+noise+smooth)": obs_reallike,
}


def deterministic_probe(n_try=4000, seed=3):
    """Quantify the finding that motivated the redesign: a noiseless SN barely exists."""
    out = {}
    for label, nm in [(0, "SN"), (1, "TC")]:
        for sigma in (0.02, 0.0):
            rng = np.random.default_rng(seed)
            ok = plat = 0
            ts = []
            for _ in range(n_try):
                p = draw_params(label, rng)
                x = integrate(p, sigma, rng)
                if not valid(p, x):
                    continue
                ok += 1
                xg = to_grid(x)
                if has_visible_plateau(xg):
                    plat += 1; ts.append(t50_of(xg))
            ts = np.array(ts)
            out[f"{nm}_sigma{sigma}"] = dict(
                valid_frac=ok / n_try, plateau_frac=plat / n_try,
                t50_median=float(np.median(ts)) if len(ts) else None,
                t50_min=float(ts.min()) if len(ts) else None,
                t50_max=float(ts.max()) if len(ts) else None)
            print(f"  {nm} sigma={sigma}: valid {ok/n_try*100:5.1f}%  plateau {plat/n_try*100:5.1f}%  "
                  f"t50 range [{ts.min() if len(ts) else float('nan'):.2f}, "
                  f"{ts.max() if len(ts) else float('nan'):.2f}]")
    return out


def gen_base(label, n, seed):
    """Process-noise trajectories, keeping the RAW trajectory for re-observation."""
    rng = np.random.default_rng(seed)
    raws, ts = [], []
    while len(raws) < n:
        p = draw_params(label, rng)
        s = float(rng.choice(SIGMA_CHOICES))
        x = integrate(p, s, rng)
        if not valid(p, x):
            continue
        xg = to_grid(x)
        if not has_visible_plateau(xg):
            continue
        raws.append(np.asarray(x, float)); ts.append(t50_of(xg))
    return raws, np.array(ts)


def match_idx(ta, tb, nbins=18, seed=0):
    """t50-match two classes; return the selected index arrays."""
    lo, hi = max(ta.min(), tb.min()), min(ta.max(), tb.max())
    edges = np.linspace(lo, hi, nbins + 1)
    rng = np.random.default_rng(seed)
    ia_all, ib_all = [], []
    for b in range(nbins):
        ia = np.where((ta >= edges[b]) & (ta < edges[b + 1]))[0]
        ib = np.where((tb >= edges[b]) & (tb < edges[b + 1]))[0]
        k = min(len(ia), len(ib))
        if k == 0:
            continue
        ia_all.append(rng.choice(ia, k, replace=False))
        ib_all.append(rng.choice(ib, k, replace=False))
    return np.concatenate(ia_all), np.concatenate(ib_all)


def fit2(Fz, y, itr, iva, tag):
    torch.manual_seed(0); np.random.seed(0)
    m, _, _ = T.train_model(FeatMLP(n_classes=2), tag, "mlp", 0.0,
                            (torch.from_numpy(Fz), torch.from_numpy(y)), itr, iva)
    m.eval()
    return m


def probs(model, Fz):
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(Fz).to(T.DEVICE)), 1).cpu().numpy()


def main():
    t0 = time.time()
    res = {}

    print("deterministic probe (why the sigma=0 arm is ill-posed):")
    res["deterministic_probe"] = deterministic_probe()

    print(f"\ngenerating base process-noise trajectories ({N_PER}/class) ...")
    rsn, tsn = gen_base(0, N_PER, 401)
    print(f"  SN done ({time.time()-t0:.0f}s)")
    rtc, ttc = gen_base(1, N_PER, 402)
    print(f"  TC done ({time.time()-t0:.0f}s)")
    isn, itc = match_idx(tsn, ttc, seed=0)
    base = [rsn[i] for i in isn] + [rtc[i] for i in itc]
    y = np.concatenate([np.zeros(len(isn)), np.ones(len(itc))]).astype(np.int64)
    print(f"t50-matched base: N={len(y)} (SN {len(isn)} / TC {len(itc)}) — all conditions paired")
    res["N"] = int(len(y))

    # ---------- build every condition from the SAME trajectories ----------
    data = {}
    for name, fn in CONDITIONS.items():
        rng = np.random.default_rng(SEED)
        X = np.stack([fn(x, rng) for x in base]).astype(np.float32)
        data[name] = build_features46(X).astype(np.float32)
        print(f"  built [{name}] ({time.time()-t0:.0f}s)")

    itr, iva, ite = T.make_splits(y, seed=500)
    ref_name = "full resolution (train-like)"
    Fr = data[ref_name]
    mu, sd = Fr[itr].mean(0), Fr[itr].std(0) + 1e-9
    ref = fit2(np.clip((Fr - mu) / sd, -8, 8).astype(np.float32), y, itr, iva, "checks/c2_ref")

    print("\nobservation ladder (2-class SN vs TC, t50-matched, paired; chance = 50%)")
    res["ladder"] = {}
    for name, F in data.items():
        transfer = float((probs(ref, np.clip((F - mu) / sd, -8, 8).astype(np.float32)[ite]).argmax(1)
                          == y[ite]).mean())
        mu_c, sd_c = F[itr].mean(0), F[itr].std(0) + 1e-9
        Fz_c = np.clip((F - mu_c) / sd_c, -8, 8).astype(np.float32)
        m_c = fit2(Fz_c, y, itr, iva, f"checks/c2_{abs(hash(name))%9999}")
        retrain = float((probs(m_c, Fz_c[ite]).argmax(1) == y[ite]).mean())
        res["ladder"][name] = dict(transfer=transfer, retrain=retrain)
        print(f"  {name:34s} transfer {transfer*100:5.1f}%   retrain {retrain*100:5.1f}%")

    # ---------- B. paired noise-amplitude test ----------
    print("\npaired noise test (same trajectories, rising added observation noise)")
    res["paired"] = {}
    for amp in AMPS:
        rng = np.random.default_rng(99)
        X = np.stack([to_grid(add_observation_noise(x, amp, rng) if amp > 0 else x)
                      for x in base]).astype(np.float32)
        F = np.clip((build_features46(X) - mu) / sd, -8, 8).astype(np.float32)
        P = probs(ref, F[ite])
        yt = y[ite]
        res["paired"][str(amp)] = dict(
            psn_sn=float(P[yt == 0, 0].mean()), psn_tc=float(P[yt == 1, 0].mean()),
            acc=float((P.argmax(1) == yt).mean()),
            frac_sn_called=float((P.argmax(1) == 0).mean()))
        d = res["paired"][str(amp)]
        print(f"  amp {amp:.2f}: acc {d['acc']*100:5.1f}%  mean P(SN) true-SN {d['psn_sn']:.3f} "
              f"true-TC {d['psn_tc']:.3f}  called SN {d['frac_sn_called']*100:.0f}%")

    frac = np.array([res["paired"][str(a)]["frac_sn_called"] for a in AMPS])
    rho = spearmanr(np.array(AMPS), frac).statistic
    res["paired_spearman_amp_vs_fracSN"] = float(rho)
    print(f"  Spearman(noise amplitude, fraction called SN) = {rho:+.2f}")

    json.dump(res, open(f"{CRUNS}/check2_noise_realism.json", "w"), indent=2)

    # ---------- figure ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.2))
    names = list(CONDITIONS.keys())
    tr = [res["ladder"][n]["transfer"] * 100 for n in names]
    rt = [res["ladder"][n]["retrain"] * 100 for n in names]
    yp = np.arange(len(names))
    ax1.barh(yp - 0.2, tr, 0.4, color="#e76f51", alpha=0.9, label="transfer (trained full-res)")
    ax1.barh(yp + 0.2, rt, 0.4, color="#2a9d8f", alpha=0.9, label="retrained on condition")
    ax1.axvline(50, color="k", lw=0.9, ls=":", label="chance")
    ax1.set_yticks(yp); ax1.set_yticklabels(names, fontsize=7.5); ax1.invert_yaxis()
    ax1.set_xlabel("2-class accuracy (%)"); ax1.set_xlim(40, 102)
    ax1.legend(fontsize=7, loc="lower right")
    ax1.set_title("(a) observation ladder (paired, t50-matched)", fontsize=9, loc="left")

    ax2.plot(AMPS, [res["paired"][str(a)]["psn_sn"] for a in AMPS], "o-",
             color="#e63946", label="mean P(SN) | true SN")
    ax2.plot(AMPS, [res["paired"][str(a)]["psn_tc"] for a in AMPS], "s-",
             color="#457b9d", label="mean P(SN) | true TC")
    ax2.plot(AMPS, frac, "^--", color="#6a4c93", label="fraction called SN")
    ax2.axhline(0.5, color="k", lw=0.8, ls=":")
    ax2.set_xlabel("added observation-noise amplitude"); ax2.set_ylabel("P(SN)")
    ax2.set_ylim(0, 1); ax2.legend(fontsize=7.5)
    ax2.set_title(f"(b) does noise level drive the SN call?  rho = {rho:+.2f}",
                  fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(f"{CRUNS}/fig_check2.png", dpi=180, bbox_inches="tight")
    print(f"\nsaved {CRUNS}/check2_noise_realism.json + fig_check2.png ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
