"""
Theory-Grounded Feature Extraction for Bifurcation Clustering
===============================================================
Features are derived directly from bifurcation theory and early-warning
signal (EWS) literature, not from generic time-series statistics.

Theoretical basis
-----------------
Near a bifurcation, a dynamical system experiences "critical slowing down"
(CSD): the dominant eigenvalue of the Jacobian approaches zero, meaning
the system recovers more slowly from perturbations. This manifests as:

  Saddle-node   : strongest CSD — variance and lag-1 AC both spike sharply
                  in the pre-bifurcation region. Skewness becomes positive
                  (distribution skewed toward the unstable fixed point).
                  The growth phase is convex-up (accelerating), followed by
                  a sharp inflection and rapid saturation.

  Transcritical : moderate CSD — variance and AC increase but more gradually.
                  Skewness near zero (symmetric distribution). The curve is
                  approximately symmetric around the inflection point.

  Null          : no CSD — variance and AC stay roughly constant or decline.
                  Skewness near zero or negative. The curve is concave-down
                  from the start (decelerating growth, no acceleration phase).

Feature groups
--------------
Group A — Critical Slowing Down (CSD) indicators  [8 features]
    Computed on the residuals (detrended series) in a rolling window,
    following Dakos et al. (2012, PLoS ONE) and Scheffer et al. (2009).

Group B — Inflection geometry                      [7 features]
    Timing, sharpness, and asymmetry of the inflection point.
    Directly encodes the Jacobian structure difference between types.

Group C — Phase-space topology (delay embedding)   [6 features]
    Features computed on the 2D delay embedding (x(t), x(t-τ)).
    Topological differences are most visible in phase space.

Group D — Catch22 subset                           [7 features]
    7 features from Catch22 (Lubba et al. 2019) selected for relevance
    to bifurcation dynamics: nonlinearity, entropy, autocorrelation decay.

Total: 28 features per series.

References
----------
Dakos et al. (2012). Methods for detecting early warnings of critical
  transitions in time series illustrated using simulated ecological data.
  PLoS ONE 7(7).
Scheffer et al. (2009). Early-warning signals for critical transitions.
  Nature 461, 53–59.
Lubba et al. (2019). catch22: CAnonical Time-series CHaracteristics.
  Data Mining and Knowledge Discovery 33, 1821–1852.
"""

import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

N_FEATURES_CSD    = 8
N_FEATURES_INFLEC = 7
N_FEATURES_PHASE  = 6
N_FEATURES_CATCH  = 7
N_FEATURES_SN     = 5    # Group E — saddle-node fingerprint
N_FEATURES_NULL   = 5    # Group F — transcritical / null discrimination
N_FEATURES_TOTAL  = (N_FEATURES_CSD + N_FEATURES_INFLEC +
                     N_FEATURES_PHASE + N_FEATURES_CATCH +
                     N_FEATURES_SN + N_FEATURES_NULL)

FEATURE_NAMES = [
    # Group A — CSD indicators
    "ac1_early",          # lag-1 AC in first 40% of series
    "ac1_late",           # lag-1 AC in last 40% of series
    "ac1_trend",          # slope of rolling lag-1 AC (Kendall τ)
    "var_early",          # variance in first 40%
    "var_late",           # variance in last 40%
    "var_trend",          # slope of rolling variance (Kendall τ)
    "skewness_early",     # skewness in first 40%
    "skewness_late",      # skewness in last 40%

    # Group B — Inflection geometry
    "t_inflection",       # normalised time of max dx/dt
    "inflection_sharpness",   # peak dx/dt value (normalised)
    "pre_inflec_concavity",   # mean d²x/dt² before inflection (positive = SN)
    "post_inflec_concavity",  # mean d²x/dt² after inflection
    "asymmetry_ratio",    # ratio of pre- to post-inflection duration
    "saturation_speed",   # 1/(time to reach 0.9 * max)
    "onset_speed",        # 1/(time to reach 0.1 * max)

    # Group C — Phase-space topology (delay embedding)
    "loop_area",          # area enclosed by (x(t), x(t-tau)) curve
    "phase_spread_x",     # std of x(t) in phase space
    "phase_spread_y",     # std of x(t-tau) in phase space
    "phase_corr",         # correlation between x(t) and x(t-tau)
    "phase_skew_x",       # skewness of x(t) in phase space
    "phase_kurtosis_x",   # kurtosis of x(t) in phase space

    # Group D — Catch22-inspired dynamical features
    "nonlinearity",       # Terasvirta nonlinearity statistic
    "entropy_approx",     # approximate entropy (regularity)
    "ac_decay_rate",      # rate of AC decay (fit of exp decay to AC function)
    "outlier_timing",     # mean time of above-median values
    "histogram_mode",     # mode of value histogram (bimodal = SN signature)
    "zero_crossing_rate", # rate of dx/dt sign changes
    "peak_to_rms",        # peak value / RMS (sharpness of growth event)

    # Group E — Saddle-node fingerprint (5 features)
    # Theoretical basis: for a proper saddle-node (dx/dt = mu + x²), the
    # velocity profile is a parabola peaking at the midpoint of the trajectory.
    # This "slow → fast → slow" signature is absent in transcritical (monotone
    # velocity decay from exponential phase) and null (concave-down, no accel.)
    "t_inflection_symmetry",  # 1 − 2|t_inf − 0.5|; max (1.0) for SN, low for TC/null
    "velocity_cv",            # std(|dx|)/mean(|dx|); high for slow-fast-slow (SN)
    "early_plateau_frac",     # fraction of series with x < 0.15; CSD slow phase (SN)
    "bimodality_coeff",       # Sarle's (skew²+1)/kurtosis; high when x clusters at 0 & 1
    "velocity_kurtosis",      # kurtosis of dx/dt; sharp jump → high kurtosis (SN)

    # Group F — Transcritical / Null discrimination (5 features)
    # Theoretical basis (Bury et al. 2021, PNAS):
    #   Null series = same ODE, parameter fixed far from bifurcation → trajectory
    #   stays near stable equilibrium (stationary noise, no trend).  After
    #   normalisation the null series is a noisy plateau centred around ~0.5,
    #   while SN and TC produce clear 0→1 rising curves.
    #   TC fingerprint: early exponential growth → log(x) linear in first ~30 %.
    "net_rise_score",         # (last-10% mean) − (first-10% mean); ≈+0.85 SN/TC, ≈0 null
    "residual_var_frac",      # var(Gaussian-detrended residual)/var(x); ≈1 null, ≈0 SN/TC
    "early_log_slope",        # slope of log(x+0.01) in first 30%; highest for TC (exp growth)
    "monotone_dx_frac",       # fraction of positive dx steps; >0.7 SN/TC, ≈0.5 null
    "center_fraction",        # fraction of series in [0.3, 0.7]; high null, low SN/TC
]


# ─────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────

def _safe(val: float, fallback: float = 0.0) -> float:
    if np.isnan(val) or np.isinf(val):
        return fallback
    return float(val)


def _kendall_tau_slope(y: np.ndarray) -> float:
    """
    Kendall τ correlation of y with its index — a rank-based trend measure.
    Returns value in [-1, 1]; +1 = monotone increasing trend.
    Used by Dakos et al. as the standard EWS trend metric.
    """
    from scipy.stats import kendalltau
    n = len(y)
    if n < 4:
        return 0.0
    tau, _ = kendalltau(np.arange(n), y)
    return _safe(tau)


def _lag1_ac(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of x."""
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom < 1e-12:
        return 0.0
    return _safe(np.dot(x[:-1], x[1:]) / denom)


def _lag_k_ac(x: np.ndarray, k: int) -> float:
    """Lag-k autocorrelation of x (single-pass, no circular wrap)."""
    if len(x) <= k or k < 1:
        return 0.0
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom < 1e-12:
        return 0.0
    return _safe(np.dot(x[:-k], x[k:]) / denom)


def _rolling_metric(x: np.ndarray, metric_fn, window_frac: float = 0.3,
                    n_windows: int = 10) -> np.ndarray:
    """
    Compute metric_fn in a sliding window across x.
    Returns array of n_windows values.
    """
    n      = len(x)
    w      = max(5, int(window_frac * n))
    starts = np.linspace(0, n - w, n_windows).astype(int)
    return np.array([metric_fn(x[s : s + w]) for s in starts])


# ─────────────────────────────────────────────────────────
# Group A — CSD indicators
# ─────────────────────────────────────────────────────────

def _group_a_csd(x: np.ndarray) -> np.ndarray:
    """
    8 CSD features following Dakos et al. (2012).
    x is the normalised [0,1] series of length 500.
    """
    n    = len(x)
    n40  = max(5, int(0.4 * n))

    early = x[:n40]
    late  = x[n40:]

    # Detrend with Gaussian smoothing before variance/AC
    # (standard practice in EWS literature to remove deterministic trend)
    from scipy.ndimage import gaussian_filter1d
    sigma   = max(2, n // 20)
    trend   = gaussian_filter1d(x.astype(float), sigma=sigma)
    resid   = x - trend

    resid_e = resid[:n40]
    resid_l = resid[n40:]

    ac1_e = _lag1_ac(resid_e)
    ac1_l = _lag1_ac(resid_l)

    var_e = _safe(np.var(resid_e))
    var_l = _safe(np.var(resid_l))

    # Skewness on residuals (consistent with Dakos et al. 2012 EWS protocol)
    skew_e = _safe(float(np.mean(((resid_e - resid_e.mean()) /
                                   (resid_e.std() + 1e-10)) ** 3)))
    skew_l = _safe(float(np.mean(((resid_l - resid_l.mean()) /
                                   (resid_l.std() + 1e-10)) ** 3)))

    # Rolling trends (Kendall τ)
    roll_ac  = _rolling_metric(resid, _lag1_ac)
    roll_var = _rolling_metric(resid, np.var)
    ac1_trend = _kendall_tau_slope(roll_ac)
    var_trend = _kendall_tau_slope(roll_var)

    return np.array([ac1_e, ac1_l, ac1_trend,
                     var_e, var_l, var_trend,
                     skew_e, skew_l], dtype=np.float32)


# ─────────────────────────────────────────────────────────
# Group B — Inflection geometry
# ─────────────────────────────────────────────────────────

def _group_b_inflection(x: np.ndarray, t_end: float = 1.0) -> np.ndarray:
    """7 inflection-geometry features. t_end=duration (e.g. years) → real-time speeds/positions;
    default 1.0 = normalised lifecycle time (legacy behaviour, bit-identical)."""
    n  = len(x)
    t  = np.linspace(0.0, t_end, n)
    dx = np.gradient(x.astype(float), t)
    d2 = np.gradient(dx, t)

    # Inflection point = peak of dx/dt
    inf_idx = int(np.argmax(dx))
    t_inf   = float(t[inf_idx])
    peak_dx = _safe(float(dx[inf_idx]))

    # Concavity before and after inflection
    pre  = d2[:max(1, inf_idx)]
    post = d2[inf_idx:]
    pre_conc  = _safe(float(pre.mean())  if len(pre)  > 0 else 0.0)
    post_conc = _safe(float(post.mean()) if len(post) > 0 else 0.0)

    # Asymmetry: ratio of pre-inflection to post-inflection duration
    pre_dur  = max(inf_idx, 1)
    post_dur = max(n - inf_idx, 1)
    asym     = _safe(float(pre_dur) / float(post_dur))

    # Saturation speed: 1 / time to reach 0.9 * max
    thresh_hi = 0.9 * x.max()
    idx_hi    = np.where(x >= thresh_hi)[0]
    sat_speed = _safe(1.0 / (t[idx_hi[0]] + 1e-6)) if len(idx_hi) > 0 else 0.0

    # Onset speed: 1 / time to reach 0.1 * max
    thresh_lo = 0.1 * max(x.max(), 0.1)
    idx_lo    = np.where(x >= thresh_lo)[0]
    onset_spd = _safe(1.0 / (t[idx_lo[0]] + 1e-6)) if len(idx_lo) > 0 else 0.0

    return np.array([t_inf, peak_dx, pre_conc, post_conc,
                     asym, sat_speed, onset_spd], dtype=np.float32)


# ─────────────────────────────────────────────────────────
# Group C — Phase-space topology (delay embedding)
# ─────────────────────────────────────────────────────────

def _group_c_phase(x: np.ndarray, tau: int = 10) -> np.ndarray:
    """
    6 features from the 2D delay embedding (x(t), x(t-τ)).
    τ = 10 steps on the 500-point grid ≈ 2% of the series.
    """
    if len(x) <= tau:
        return np.zeros(6, dtype=np.float32)

    xi  = x[tau:].astype(float)
    xj  = x[:-tau].astype(float)

    # Area enclosed by the phase portrait (shoelace formula)
    # Non-zero area indicates the curve is not a straight line in phase space
    # — saddle-node has a characteristic "foot" shape
    area = _safe(0.5 * abs(np.sum(xi[:-1] * xj[1:] - xi[1:] * xj[:-1])))

    spread_x   = _safe(float(xi.std()))
    spread_y   = _safe(float(xj.std()))

    # Correlation between x(t) and x(t-τ)
    if spread_x > 1e-8 and spread_y > 1e-8:
        corr = _safe(float(np.corrcoef(xi, xj)[0, 1]))
    else:
        corr = 1.0

    skew_x = _safe(float(np.mean(((xi - xi.mean()) / (xi.std() + 1e-10)) ** 3)))
    kurt_x = _safe(float(np.mean(((xi - xi.mean()) / (xi.std() + 1e-10)) ** 4)) - 3.0)

    return np.array([area, spread_x, spread_y, corr, skew_x, kurt_x],
                    dtype=np.float32)


# ─────────────────────────────────────────────────────────
# Group D — Catch22-inspired dynamical features
# ─────────────────────────────────────────────────────────

def _group_d_catch(x: np.ndarray, t_end: float = 1.0) -> np.ndarray:
    """7 catch22-inspired features selected for bifurcation relevance.
    t_end=duration → outlier_timing in real time units; ratio features unaffected."""
    n  = len(x)
    xf = x.astype(float)
    t  = np.linspace(0.0, t_end, n)

    # 1. Terasvirta nonlinearity: F-stat from nonlinear vs linear AR model
    # Simplified proxy: ratio of cubic to linear fit residuals
    try:
        p1 = np.polyfit(t, xf, 1)
        p3 = np.polyfit(t, xf, 3)
        res1 = np.sum((xf - np.polyval(p1, t)) ** 2)
        res3 = np.sum((xf - np.polyval(p3, t)) ** 2)
        nonlin = _safe(float(res1 / (res3 + 1e-12)))
    except Exception:
        nonlin = 1.0

    # 2. Approximate entropy (regularity measure)
    # Higher = more irregular (null class has smoother curves)
    try:
        m   = 2
        r   = 0.2 * xf.std()
        n_s = min(200, n)
        xs  = xf[:n_s]
        def _phi(m_):
            count = 0
            for i in range(n_s - m_):
                template = xs[i : i + m_]
                matches  = np.sum(
                    np.all(np.abs(
                        np.lib.stride_tricks.sliding_window_view(xs, m_) - template
                    ) <= r, axis=1)
                )
                if matches > 0:
                    count += np.log(matches / (n_s - m_ + 1))
            return count / (n_s - m_ + 1)
        approx_ent = _safe(abs(_phi(m) - _phi(m + 1)))
    except Exception:
        approx_ent = 0.0

    # 3. AC decay rate: fit exponential to AC(k) for k=1..20
    # Computes proper lag-k autocorrelations (not lag-1 of circular shift).
    try:
        max_lag = min(20, n // 4)
        lags    = np.arange(1, max_lag + 1)
        acs     = np.array([_lag_k_ac(xf, int(k)) for k in lags])
        acs     = np.clip(acs, 1e-6, None)
        coeffs  = np.polyfit(lags, np.log(acs + 1e-6), 1)
        ac_decay = _safe(float(-coeffs[0]))   # positive = fast decay
    except Exception:
        ac_decay = 0.0

    # 4. Outlier timing: mean normalised time of above-median values
    med      = np.median(xf)
    hi_mask  = xf > med
    out_time = _safe(float(t[hi_mask].mean()) if hi_mask.sum() > 0 else 0.5)

    # 5. Histogram mode (bin index of most frequent value)
    # Bimodal distribution is characteristic of saddle-node near the fold
    counts, edges = np.histogram(xf, bins=10)
    hist_mode     = _safe(float(edges[counts.argmax()] + (edges[1] - edges[0]) / 2))

    # 6. Zero-crossing rate of dx/dt
    dx   = np.gradient(xf, t)
    zc   = _safe(float(np.sum(np.diff(np.sign(dx)) != 0)) / max(n - 1, 1))

    # 7. Peak-to-RMS ratio
    rms       = _safe(float(np.sqrt(np.mean(xf ** 2))))
    peak2rms  = _safe(float(xf.max()) / (rms + 1e-10))

    return np.array([nonlin, approx_ent, ac_decay, out_time,
                     hist_mode, zc, peak2rms], dtype=np.float32)


# ─────────────────────────────────────────────────────────
# Group E — Saddle-node fingerprint
# ─────────────────────────────────────────────────────────

def _group_e_sn_fingerprint(x: np.ndarray) -> np.ndarray:
    """
    5 features designed to fingerprint the saddle-node slow→fast→slow
    velocity profile that is absent in transcritical and null series.

    Theory
    ------
    Proper saddle-node (dx/dt = −r² + x²) starting near x = −r:
      • Velocity is near-zero at x = −r  (unstable FP, CSD)
      • Velocity peaks at x = 0          (midpoint of trajectory)
      • Velocity returns to near-zero at x = +r  (stable FP saturation)
    → inflection of x(t) is at t ≈ 0.5  (symmetric)
    → velocity distribution is peaked and high-kurtosis
    → x values cluster near 0 and 1 (bimodal in [0,1] histogram)

    Transcritical: velocity decays monotonically after exponential phase
      → inflection early (t < 0.3), low velocity CV
    Null-SN: smooth concave-down rise, velocity is highest at t = 0
      → inflection near t = 0, very low velocity CV
    Null-TC (declining): velocity highest at t = 0, t_inflection ≈ 1
    """
    n  = len(x)
    xf = x.astype(float)
    t  = np.linspace(0.0, 1.0, n)
    dx = np.gradient(xf, t)

    # 1. t_inflection_symmetry = 1 − 2|t_inf − 0.5|
    #    SN ≈ 1.0 (inflection at midpoint); TC < 0.5 (early); null ≈ 0 (start/end)
    inf_idx   = int(np.argmax(dx))
    t_inf     = float(t[inf_idx])
    t_inf_sym = _safe(1.0 - 2.0 * abs(t_inf - 0.5))

    # 2. velocity_cv = std(|dx|) / (mean(|dx|) + ε)
    #    SN: large ratio (near-zero flanks vs. high-peak middle)
    #    TC: moderate (monotone decay from peak)
    #    null: low (roughly constant or monotone decline)
    abs_dx = np.abs(dx)
    vel_cv = _safe(float(abs_dx.std()) / (float(abs_dx.mean()) + 1e-10))

    # 3. early_plateau_frac = fraction of time steps where x < 0.15
    #    SN: high (series lingers near 0 in CSD slow phase)
    #    TC: lower (rises quickly from near-zero initial condition)
    #    null-SN: very low (velocity highest at start, rises immediately)
    early_plateau = _safe(float((xf < 0.15).mean()))

    # 4. bimodality_coeff = (skewness² + 1) / (excess_kurtosis + 3)   [Sarle 1990]
    #    SN: x values cluster near 0 and 1, sparse in middle → bimodal → higher coeff
    #    TC / null: more unimodal → lower coeff
    xmean   = xf.mean()
    xstd    = xf.std() + 1e-10
    skew    = float(np.mean(((xf - xmean) / xstd) ** 3))
    kurt    = float(np.mean(((xf - xmean) / xstd) ** 4))   # raw kurtosis
    bim_c   = _safe((skew ** 2 + 1.0) / (kurt + 1e-6))

    # 5. velocity_kurtosis = excess kurtosis of dx/dt
    #    SN: sharp localised jump → highly peaked velocity → high kurtosis
    #    TC / null: smoother velocity profiles → lower kurtosis
    dx_mean  = dx.mean()
    dx_std   = dx.std() + 1e-10
    vel_kurt = _safe(float(np.mean(((dx - dx_mean) / dx_std) ** 4)) - 3.0)

    return np.array([t_inf_sym, vel_cv, early_plateau, bim_c, vel_kurt],
                    dtype=np.float32)


# ─────────────────────────────────────────────────────────
# Group F — Transcritical / Null discrimination
# ─────────────────────────────────────────────────────────

def _group_f_tc_null(x: np.ndarray, t_end: float = 1.0) -> np.ndarray:
    """
    5 features that distinguish transcritical from null and help resolve
    the three-way SN / TC / null separation.
    t_end=duration → early_log_slope becomes a real per-year exponential growth rate.

    Theory
    ------
    After normalisation to [0, 1]:

    Null (Bury approach — near-equilibrium stationary noise):
      • Series is a noisy plateau centred around some value ∈ (0.1, 0.9)
      • net_rise_score  ≈ 0   (no net change from start to end)
      • residual_var_frac ≈ 1 (all variance is noise, no smooth trend)
      • monotone_dx_frac ≈ 0.5  (random walk — dx changes sign often)
      • center_fraction  HIGH   (values cluster around 0.5)
      • early_log_slope  ≈ 0    (no growth phase)

    Transcritical (dx/dt = mu·x − a₂·x² from near 0):
      • Series rises from 0 to 1 following a sigmoid
      • net_rise_score  ≈ +0.85 (clear rise)
      • residual_var_frac ≈ 0   (smooth trend dominates)
      • monotone_dx_frac > 0.7  (mostly positive steps)
      • center_fraction  LOW    (spends time near 0 and near 1)
      • early_log_slope  HIGH   (near-exponential early growth, log-linear)

    Saddle-Node (noise-driven escape from near x=−r):
      • Series has long flat phase near 0, then sharp jump to 1
      • net_rise_score  ≈ +0.85 (large net rise)
      • residual_var_frac ≈ low (jump dominates)
      • monotone_dx_frac > 0.7  (mostly positive steps)
      • center_fraction  LOW    (bimodal: at 0 and at 1)
      • early_log_slope  LOW    (flat phase, no early growth)
    """
    n  = len(x)
    xf = x.astype(float)
    t  = np.linspace(0.0, t_end, n)
    dx = np.gradient(xf, t)

    n10 = max(1, n // 10)
    n30 = max(2, int(0.3 * n))

    # 1. net_rise_score: mean of last 10% minus mean of first 10%
    #    Null ≈ 0 (plateau, no net trend); SN/TC ≈ +0.8 (0 → 1 rise)
    rise = _safe(float(xf[-n10:].mean()) - float(xf[:n10].mean()))

    # 2. residual_var_frac: variance of Gaussian-smoothed residuals / total variance
    #    Null ≈ 1 (all variance is high-frequency noise, no smooth trend)
    #    SN/TC ≈ 0–0.3 (most variance captured by smooth trend)
    from scipy.ndimage import gaussian_filter1d as _gf
    sigma_sm   = max(3, n // 8)
    trend      = _gf(xf, sigma=sigma_sm)
    resid      = xf - trend
    total_var  = float(np.var(xf))
    resid_frac = _safe(float(np.var(resid)) / (total_var + 1e-10))

    # 3. early_log_slope: OLS slope of log(x + 0.01) vs time in first 30%
    #    TC has near-linear log growth (dx/dt ≈ mu*x at small x) → high slope
    #    SN: long flat phase → slope ≈ 0 in first 30%
    #    Null: plateau → slope ≈ 0
    try:
        t_e   = t[:n30]
        lx_e  = np.log(xf[:n30] + 0.01)
        slope_coeffs = np.polyfit(t_e, lx_e, 1)
        log_slope    = _safe(float(slope_coeffs[0]))
    except Exception:
        log_slope = 0.0

    # 4. monotone_dx_frac: fraction of dx > 0
    #    SN/TC ≈ 0.7–0.95 (mostly rising); Null ≈ 0.5 (random direction)
    mono_frac = _safe(float((dx > 0).mean()))

    # 5. center_fraction: fraction of series with x ∈ [0.3, 0.7]
    #    Null (plateau near 0.5): HIGH (≈ 0.6–0.9)
    #    SN/TC (0→1 transition): LOW (most time near 0 or near 1)
    ctr_frac = _safe(float(((xf >= 0.3) & (xf <= 0.7)).mean()))

    return np.array([rise, resid_frac, log_slope, mono_frac, ctr_frac],
                    dtype=np.float32)


# ─────────────────────────────────────────────────────────
# Group G — Transcritical fingerprint (5 features; symmetric counterpart to the
# saddle-node fingerprint, Group E). Added so the feature space is BALANCED between
# the two bifurcation types rather than carrying an SN-only fingerprint group.
# ─────────────────────────────────────────────────────────

TCFP_NAMES = [
    "tc_exp_onset",         # R^2 of log(x) ~ linear in the early rise: TC departs the unstable origin exponentially
    "tc_percap_linearity",  # R^2 of per-capita growth g=v/x vs x: logistic (TC) gives g = mu(1 - x/K), a straight line
    "tc_velocity_symmetry", # symmetry of the velocity profile about its peak: logistic (TC) velocity is symmetric (->1)
    "tc_exp_relax",         # R^2 of log(1-x) ~ linear in t near saturation: TC relaxes exponentially (eigenvalue -mu)
    "tc_logistic_r2",       # goodness of a direct logistic-curve fit to x(t): ~1 for TC, lower for SN ghost / null
]


def _r2(y, yhat):
    ss = float(np.sum((y - y.mean()) ** 2))
    return _safe(1.0 - float(np.sum((y - yhat) ** 2)) / ss) if ss > 1e-12 else 0.0


def _group_g_tc_fingerprint(x: np.ndarray, t_end: float = 1.0) -> np.ndarray:
    """
    Transcritical (dx/dt = mu x - x^2, logistic) fingerprint, mirroring Group E for the saddle node:
      * departs the unstable fixed point at the ORIGIN exponentially  -> log(x) linear early
      * per-capita growth g = v/x is LINEAR in x  (g = mu(1 - x/K))
      * velocity profile is SYMMETRIC about its peak (inflection at t ~ 0.5)
      * approaches the attractor EXPONENTIALLY (eigenvalue -mu)        -> log(1-x) linear late
      * the whole curve is well fit by a logistic
    All of these are weak/absent for the saddle-node ghost (back-loaded, algebraic) and the null.
    """
    n = len(x); t = np.linspace(0.0, 1.0, n)
    xn = np.clip((x.astype(float) - x.min()) / (x.max() - x.min() + 1e-12), 0.0, 1.0)
    v = np.gradient(xn, t)

    # 1. exponential onset
    m = (xn > 0.02) & (xn < 0.40) & (t < 0.6)
    tc_exp = _r2(np.log(xn[m] + 1e-9), np.polyval(np.polyfit(t[m], np.log(xn[m] + 1e-9), 1), t[m])) if m.sum() >= 4 else 0.0

    # 2. per-capita linearity (logistic)
    m = (xn > 0.05) & (xn < 0.90) & (v > 1e-9)
    if m.sum() >= 5:
        g = v[m] / xn[m]; tc_percap = _r2(g, np.polyval(np.polyfit(xn[m], g, 1), xn[m]))
    else:
        tc_percap = 0.0

    # 3. velocity-profile symmetry about its peak
    ip = int(np.argmax(v)); left = float(v[:ip + 1].sum()); right = float(v[ip:].sum())
    tc_sym = _safe(1.0 - abs(left - right) / (left + right + 1e-9))

    # 4. exponential relaxation to the attractor
    d = 1.0 - xn; m = (xn > 0.60) & (xn < 0.985) & (d > 1e-6)
    tc_relax = _r2(np.log(d[m]), np.polyval(np.polyfit(t[m], np.log(d[m]), 1), t[m])) if m.sum() >= 4 else 0.0

    # 5. direct logistic-fit quality
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(lambda tt, k, t0: 1.0 / (1.0 + np.exp(-k * (tt - t0))),
                            t, xn, p0=[10.0, 0.5], maxfev=2000)
        tc_log = _r2(xn, 1.0 / (1.0 + np.exp(-popt[0] * (t - popt[1]))))
    except Exception:
        tc_log = 0.0

    return np.array([tc_exp, tc_percap, tc_sym, tc_relax, tc_log], dtype=np.float32)


# ─────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────

def extract_theory_features(
    X:       np.ndarray,     # (N, 500) normalised series
    verbose: bool = True,
    t_end=None,              # None → 1.0 (normalised lifecycle time, legacy). Scalar or (N,) array
                             # of per-series durations (years) → REAL-time features: t_inflection,
                             # outlier_timing in years; onset/saturation speed + early_log_slope per yr.
    include_tc_fingerprint: bool = False,   # append Group G (5 TC-fingerprint features) → 43-D balanced space
) -> np.ndarray:
    """
    Extract all 28 theory-grounded features for N series.

    Parameters
    ----------
    X       : (N, 500) float32 — PCHIP-normalised series
    verbose : print progress
    t_end   : optional series duration(s) for a non-normalised time axis (§88)

    Returns
    -------
    F : (N, 28) float64 — feature matrix, finite values only
    """
    N = len(X)
    ncols = N_FEATURES_TOTAL + (len(TCFP_NAMES) if include_tc_fingerprint else 0)
    F = np.zeros((N, ncols), dtype=np.float64)
    te = np.full(N, 1.0) if t_end is None else np.broadcast_to(
        np.asarray(t_end, dtype=float), (N,)) if np.ndim(t_end) == 0 else np.asarray(t_end, float)

    for i in range(N):
        x = X[i].astype(np.float64)
        try:
            fa = _group_a_csd(x)
            fb = _group_b_inflection(x, t_end=te[i])
            fc = _group_c_phase(x)
            fd = _group_d_catch(x, t_end=te[i])
            fe = _group_e_sn_fingerprint(x)          # all fractional/scale-invariant — stays normalised
            ff = _group_f_tc_null(x, t_end=te[i])
            parts = [fa, fb, fc, fd, fe, ff]
            if include_tc_fingerprint:
                parts.append(_group_g_tc_fingerprint(x, t_end=te[i]))   # Group G — balanced TC fingerprint
            F[i] = np.concatenate(parts)
        except Exception as e:
            if verbose:
                print(f"  Warning: feature extraction failed for sample {i}: {e}")

    # Replace NaN/Inf with column medians
    for j in range(ncols):
        col  = F[:, j]
        bad  = ~np.isfinite(col)
        if bad.any():
            good = col[~bad]
            med  = float(np.median(good)) if len(good) > 0 else 0.0
            F[bad, j] = med

    if verbose:
        print(f"  Theory features extracted: {N} × {ncols}")

    return F


def standardise(F: np.ndarray) -> tuple:
    """Zero-mean unit-variance standardisation. Returns (F_scaled, mu, std)."""
    mu  = F.mean(axis=0)
    std = F.std(axis=0) + 1e-8
    return (F - mu) / std, mu, std
