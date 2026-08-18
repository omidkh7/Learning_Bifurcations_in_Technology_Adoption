#!/usr/bin/env python3
"""
matched_inflection_experiment.py
================================
Decisive test of "does the FFT learn SN/TC DYNAMICS or just inflection POSITION?" (§70).

We regenerate SN and TC trajectories from the SAME normal-form ODEs as Synthetic_Data_Gen.py,
but (a) with WIDER parameter ranges and (b) WITHOUT the inflection-position rejection filter that
artificially separates the classes — so the two classes OVERLAP in inflection timing. We then
build a t50-MATCHED training set (equal SN/TC in every t50 bin → inflection carries ZERO class
information by construction) and retrain the FFT-Transformer.

Read-out:
  • FFT acc on the t50-matched test set, vs the t50-threshold baseline (~50% by construction).
    - FFT ≈ 50–60%  ⇒  SN/TC ≈ inflection position; the NN learns nothing deeper (confirms §70).
    - FFT ≫ 60%     ⇒  a genuine deeper feature exists (e.g. the SN pre-fold stall / CSD).
  • CONTROL: FFT acc on a NATURAL (unmatched) set — should stay high (sanity check).

Output: results/figures/fig_matched_inflection.png + console verdict.
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch

from Synthetic_Data_Gen import (saddle_node_rhs, transcritical_rhs, integrate_sde,
                                 normalize_series, pchip_to_500, is_valid_trajectory,
                                 has_visible_plateau)
# NOTE: the FFT typing evidence (train_eval, §71) uses the archived stage2_deep module; that import
# is LAZY (inside train_eval) so that the generator entry points gen_pool/t50, imported by data.py and
# the reproduction chain, do not depend on the retired FFT stack. stage2_deep is archived (§91).

T = 500; t = np.linspace(0, 1, T)
def t50(x):
    x = (x - x.min()) / (x.max() - x.min() + 1e-9)
    return float(t[np.argmax(x >= 0.5)]) if (x >= 0.5).any() else 1.0
def infl(x):
    return float(np.argmax(np.diff(x))) / (len(x) - 1)


def gen_pool(label, n, seed):
    """Generate n valid SN(0)/TC(1) curves with WIDE params, NO inflection-position reject."""
    rng = np.random.default_rng(seed)
    X, meta = [], []
    while len(X) < n:
        n_steps = int(rng.integers(20, 80)); t_end = rng.uniform(5.0, 20.0)
        tt = np.linspace(0, t_end, n_steps); sigma = rng.choice([0.005, 0.01, 0.02, 0.05])
        phi = rng.uniform(0.0, 0.9)
        if label == 0:                                   # saddle-node, WIDE speed
            r = rng.uniform(0.05, 0.35); a3 = rng.uniform(-0.05, 0.05)
            speed = rng.uniform(0.1, 9.0)                # wider than (0.3,5.0) → wider t50
            mu_sweep = np.linspace(-speed * r**2, r**2, n_steps)
            x0 = -np.sqrt(abs(-speed * r**2)) * rng.uniform(0.9, 1.1)
            x = integrate_sde(saddle_node_rhs, x0, tt, {"a2": 1.0, "a3": a3}, sigma, rng,
                              phi_ar=phi, mu_sweep=mu_sweep); xref = r
            if not is_valid_trajectory(x, xref) or x.max() < 4.0:
                continue
        else:                                            # transcritical, WIDE rate
            mu = rng.uniform(0.1, 6.0); a2 = rng.uniform(0.8, 2.0); a3 = rng.uniform(-0.5, 0.5)
            x0 = rng.uniform(0.005, 0.05)
            x = integrate_sde(transcritical_rhs, x0, tt, {"mu": mu, "a2": a2, "a3": a3}, sigma,
                              rng, phi_ar=phi, mu_sweep=None); xref = mu / a2
            if not is_valid_trajectory(x, xref):
                continue
        x5 = pchip_to_500(normalize_series(x))
        if not has_visible_plateau(x5):                  # keep ONLY the genuine validity checks
            continue
        X.append(x5.astype(np.float32)); meta.append(t50(x5))
    return np.array(X), np.array(meta)


def match_t50(Xsn, t_sn, Xtc, t_tc, nbins=18):
    """Stratified t50 matching: equal SN/TC per t50 bin → t50 uninformative."""
    lo = max(t_sn.min(), t_tc.min()); hi = min(t_sn.max(), t_tc.max())
    edges = np.linspace(lo, hi, nbins + 1)
    Xs, ys = [], []
    for b in range(nbins):
        si = np.where((t_sn >= edges[b]) & (t_sn < edges[b + 1]))[0]
        ti = np.where((t_tc >= edges[b]) & (t_tc < edges[b + 1]))[0]
        k = min(len(si), len(ti))
        if k == 0:
            continue
        rng = np.random.default_rng(b)
        for idx, X, lab in [(si, Xsn, 0), (ti, Xtc, 1)]:
            pick = rng.choice(idx, k, replace=False)
            Xs.append(X[pick]); ys.append(np.full(k, lab))
    return np.vstack(Xs), np.concatenate(ys)


def train_eval(X, y, tag, epochs=30):
    try:                                # archived FFT stack, only needed for this evidence function
        from stage2_deep import FFTTransformer, make_loaders, train_model, predict_proba
    except ImportError:
        import sys; sys.path.insert(0, "archive/stage2_fft")
        from stage2_deep import FFTTransformer, make_loaders, train_model, predict_proba
    tr, va, te, *_ = make_loaders(X, y.astype(int))
    m, _ = train_model(FFTTransformer(n_classes=2), f"FFT[{tag}]", tr, va, n_epochs=epochs, patience=6)
    # rebuild test split deterministically to score + get t50 baseline
    rng = np.random.default_rng(42); idx = []
    for c in (0, 1):
        cc = np.where(y == c)[0]; rng.shuffle(cc)
        ntr = int(len(cc) * 0.7); nva = int(len(cc) * 0.15)
        idx.append(cc[ntr + nva:])
    te_idx = np.concatenate(idx)
    P = predict_proba(m, X[te_idx]); pred = P.argmax(1)
    acc = (pred == y[te_idx]).mean()
    th = np.array([t50(x) for x in X[te_idx]])
    yte = y[te_idx]
    base = max(max(a, 1 - a) for c in np.linspace(.2, .9, 71)
               for a in [((th >= c).astype(int) == yte).mean()])   # both threshold directions
    return acc, base, th, yte, P


def main():
    print("Generating SN/TC pools (wide params, NO inflection reject)…")
    Xsn, tsn = gen_pool(0, 6000, 1); Xtc, ttc = gen_pool(1, 6000, 2)
    print(f"  natural t50:  SN median {np.median(tsn):.2f} [{tsn.min():.2f},{tsn.max():.2f}]  "
          f"TC median {np.median(ttc):.2f} [{ttc.min():.2f},{ttc.max():.2f}]")

    # CONTROL: natural (unmatched) balanced set
    n = min(len(Xsn), len(Xtc))
    Xn = np.vstack([Xsn[:n], Xtc[:n]]); yn = np.concatenate([np.zeros(n), np.ones(n)])
    acc_nat, base_nat, *_ = train_eval(Xn, yn, "natural")

    # MATCHED: equal SN/TC per t50 bin
    Xm, ym = match_t50(Xsn, tsn, Xtc, ttc)
    print(f"  matched set: N={len(ym)} (SN {int((ym==0).sum())} / TC {int((ym==1).sum())})")
    acc_m, base_m, th_m, yt_m, P_m = train_eval(Xm, ym, "t50-matched")

    print("\n" + "=" * 64)
    print("  RESULT — can the FFT separate SN/TC when inflection is matched?")
    print("=" * 64)
    print(f"  NATURAL  set:  FFT acc {acc_nat*100:5.1f}%   t50-threshold baseline {base_nat*100:5.1f}%")
    print(f"  MATCHED  set:  FFT acc {acc_m*100:5.1f}%   t50-threshold baseline {base_m*100:5.1f}%")
    drop = (acc_nat - acc_m) * 100
    print(f"\n  FFT accuracy drop natural→matched: {drop:.1f} pts")
    if acc_m < 0.62:
        print("  ⇒ VERDICT: with inflection matched, FFT collapses toward chance →")
        print("    SN/TC ≈ inflection position; the NN is NOT learning deeper dynamics.")
    elif acc_m < 0.75:
        print("  ⇒ VERDICT: partial — most signal is inflection, but a weak residual feature remains.")
    else:
        print("  ⇒ VERDICT: FFT still separates well → a genuine deeper (non-inflection) feature exists.")

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].hist(tsn, bins=40, alpha=.6, color="#E63946", label="SN")
    ax[0].hist(ttc, bins=40, alpha=.6, color="#457B9D", label="TC")
    ax[0].set_title("natural t50 (wide params, no reject)"); ax[0].set_xlabel("t50"); ax[0].legend()
    tm_sn = np.array([t50(x) for x in Xm[ym == 0]]); tm_tc = np.array([t50(x) for x in Xm[ym == 1]])
    ax[1].hist(tm_sn, bins=20, alpha=.6, color="#E63946", label="SN")
    ax[1].hist(tm_tc, bins=20, alpha=.6, color="#457B9D", label="TC")
    ax[1].set_title("MATCHED t50 (inflection uninformative)"); ax[1].set_xlabel("t50"); ax[1].legend()
    ax[2].bar([0, 1, 2, 3], [acc_nat*100, base_nat*100, acc_m*100, base_m*100],
              color=["#E63946", "#f4a261", "#457B9D", "#a8dadc"])
    ax[2].set_xticks([0, 1, 2, 3]); ax[2].set_xticklabels(["FFT\nnatural", "t50base\nnatural",
                                                           "FFT\nmatched", "t50base\nmatched"])
    ax[2].axhline(50, ls="--", c="gray"); ax[2].set_ylabel("accuracy %"); ax[2].set_ylim(40, 100)
    ax[2].set_title("FFT vs t50-baseline")
    for i, v in enumerate([acc_nat, base_nat, acc_m, base_m]):
        ax[2].annotate(f"{v*100:.0f}", (i, v*100+1), ha="center", fontweight="bold")
    fig.suptitle("Matched-inflection experiment — does the FFT learn dynamics beyond inflection position?",
                 fontweight="bold")
    plt.tight_layout(); fig.savefig("results/figures/fig_matched_inflection.png", dpi=150, bbox_inches="tight")
    print("\nSaved → results/figures/fig_matched_inflection.png")


if __name__ == "__main__":
    main()
