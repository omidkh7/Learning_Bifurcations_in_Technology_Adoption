#!/usr/bin/env python3
"""
finite_time_scaling.py
======================
Probe: can the finite-time-scaling exponent of Corral, Sardanyes & Alseda (2018) -- which
distinguishes the saddle-node (tau ~ |mu-mu_c|^-1/2) from the transcritical (tau ~ |mu-mu_c|^-1)
bifurcation -- be recovered from technology-adoption ensembles?

Key result (this script):
  * In the TRUE control parameter mu, the two exponents are clean and well separated.
  * In the only OBSERVABLE proxy (the attained level L = order parameter), BOTH collapse to
    tau ~ L^-1, because L itself scales with mu (L ~ mu^beta, beta = 1 TC / 1/2 SN) so the
    discriminating information cancels. The exponent is therefore NOT recoverable from a
    trajectory alone -- a quantitative restatement of Boettiger, Ross & Hastings (2013),
    "cannot identify mechanism from data alone." (In real adoption data the attained ceiling is
    exogenous rather than mu-set, so even this proxy channel is absent.)

Output: figures/si/figS_fts_probe.png (two panels: theory in mu, and the L confound)
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from paper_style import set_style, COL2
set_style()
from paper_figures import _tag

SI = "figures/si"; os.makedirs(SI, exist_ok=True)
SNCOL, TCCOL = "#c1121f", "#1d4e89"


# --------------------------------------------------------------- normal-form simulation
def integrate(kind, mu, T=4000.0, dt=0.01):
    """Continuous normal form. TC: x' = mu x - x^2 (attractor mu). SN: x' = mu - x^2 (attractor sqrt mu)."""
    xstar = mu if kind == "TC" else np.sqrt(mu)
    x0 = 0.01 * xstar if kind == "TC" else 0.0
    n = int(T / dt); x = np.empty(n); x[0] = x0
    for i in range(1, n):
        xv = x[i-1]
        dx = (mu * xv - xv**2) if kind == "TC" else (mu - xv**2)
        x[i] = xv + dx * dt
    return np.arange(n) * dt, x


def relax_tau(t, x, L):
    """Relaxation time from the exponential approach to the attractor: log(L - x) ~ -t/tau."""
    d = L - x; m = (x > 0.1 * L) & (x < 0.9 * L) & (d > 1e-9)
    if m.sum() < 10: return np.nan
    return -1.0 / np.polyfit(t[m], np.log(d[m]), 1)[0]


def sim_scaling():
    out = {}
    for kind in ("SN", "TC"):
        mus = np.geomspace(0.02, 1.0, 14); Ls = []; taus = []
        for mu in mus:
            t, x = integrate(kind, mu); L = x[-1]
            Ls.append(L); taus.append(relax_tau(t, x, L))
        out[kind] = (mus, np.array(Ls), np.array(taus))
    return out


# --------------------------------------------------------------- figure
def fig_probe():
    sim = sim_scaling()

    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.9), gridspec_kw=dict(wspace=0.30))

    # (a) theory: tau vs mu (true control parameter) -- clean 1/2 vs 1
    ax = axs[0]
    for kind, c in [("SN", SNCOL), ("TC", TCCOL)]:
        mus, Ls, taus = sim[kind]; b = np.polyfit(np.log(mus), np.log(taus), 1)[0]
        ax.loglog(mus, taus, "o", ms=3.5, color=c, label=f"{kind}: slope {b:+.2f}")
    ax.set_xlabel(r"control parameter $\mu-\mu_c$"); ax.set_ylabel(r"transient time $\tau$")
    ax.legend(fontsize=6.4, loc="upper right"); _tag(ax, "(a) theory, in true $\\mu$: SN $-1/2$ vs TC $-1$")

    # (b) the confound: tau vs attained level L (the only observable) -- both collapse to -1.
    #     In the normal form L = x* co-varies with mu by construction; in real data the ceiling is
    #     exogenous (market size, competitors), so this proxy channel is absent altogether. Either way
    #     the discriminating -1/2 vs -1 information does not survive in a single trajectory.
    ax = axs[1]
    for kind, c in [("SN", SNCOL), ("TC", TCCOL)]:
        mus, Ls, taus = sim[kind]; b = np.polyfit(np.log(Ls), np.log(taus), 1)[0]
        ax.loglog(Ls, taus, "o", ms=3.5, color=c, label=f"{kind}: slope {b:+.2f}")
    ax.set_xlabel(r"attained level $L$ (observable)"); ax.set_ylabel(r"transient time $\tau$")
    ax.legend(fontsize=6.4, loc="upper right"); _tag(ax, "(b) in observable $L$: both $\\to -1$ (confounded)")

    fig.savefig(f"{SI}/figS_fts_probe.png", bbox_inches="tight"); plt.close(fig)
    for kind in ("SN", "TC"):
        mus, Ls, taus = sim[kind]
        print(f"  sim {kind}: tau~mu^{np.polyfit(np.log(mus),np.log(taus),1)[0]:+.2f}  "
              f"tau~L^{np.polyfit(np.log(Ls),np.log(taus),1)[0]:+.2f}")
    print("saved figS_fts_probe.png")


if __name__ == "__main__":
    fig_probe()
