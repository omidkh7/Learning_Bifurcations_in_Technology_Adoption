"""
Real-World Technology Adoption — Bifurcation Clustering
=========================================================
Research question
-----------------
Can we find a clustering classifier that correctly identifies saddle-node,
transcritical, and null bifurcations in real-world technology adoption data?

Approach
--------
1.  Feature extraction (AE, Theory, ROCKET) on the real-world dataset.
2.  k-sweep k=2..8 for **seven** methods:
      • AE-GMM           — Autoencoder embeddings → PCA → GMM
      • Theory-GMM       — 38 EWS + bifurcation theory features → PCA (whiten) → GMM
      • Ensemble         — Theory + AE → PCA → GMM
      • ROCKET-GMM       — 3000 random-kernel features → PCA → GMM
      • Theory+ROCKET    — Theory + ROCKET → PCA → GMM   [Bury-inspired]
      • Theory-TMM       — 38 EWS features → PCA (whiten) → Student-t Mixture Model
                           (Peel & McLachlan 2000 EM; same space as Theory-GMM)
      • Theory-TMM(raw)  — 38 EWS features → PCA (no whiten) → Student-t Mixture
                           (Tests whether whitening blunts TMM's heavy-tail advantage)
3.  Select best k per method (silhouette criterion).
4.  Bifurcation-type labelling at k=3 (canonical) and at best k:
      Nearest-centroid assignment in theory feature space using synthetic
      class prototypes (SN / TC / null) derived from the validated benchmark.
5.  Comprehensive output: k-sweep curves, cluster trajectory bands,
    UMAP scatter, feature heatmaps, per-series JSON assignments.

Synthetic benchmark performance (on 9,000 labelled series, seed=42):
  AE-GMM           ACC=72.8%   SN=50%   TC=68.5%  Null=100%
  Theory-GMM       ACC=90.9%   SN=82%   TC=91.8%  Null=99.0%
  Ensemble         ACC=76.6%   SN=58.7% TC=71.3%  Null=99.9%
  ROCKET-GMM       ACC=95.0%   SN=100%  TC=85.4%  Null=99.6%
  Theory+ROCKET    ACC=91.8%   SN=100%  TC=75.4%  Null=99.9%
  Theory-TMM       —  (real-world only; no synthetic benchmark yet)

Usage
-----
  python unsup_real_world.py \\
      --real_data_dir data/real_world \\
      --synth_dir     data/synthetic \\
      --bundle_path   runs/unsup/benchmark/synth_features.pkl \\
      --out_dir       runs/unsup/real_world \\
      --n_kernels     2000 \\
      --seed          42
"""

import argparse
import json
import os
import pickle
import warnings
import numpy as np
from scipy.optimize import linear_sum_assignment, brentq
from scipy.special import gammaln, digamma, logsumexp
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

warnings.filterwarnings("ignore")

# ─── Bifurcation class names ───────────────────────────────────────────────────
CLASS_NAMES  = ["saddle_node", "transcritical", "null"]
CLASS_COLORS = {"saddle_node": "#E63946",
                "transcritical": "#2196F3",
                "null": "#4CAF50",
                "unknown": "#9E9E9E"}
CLUSTER_COLORS = ["#E63946", "#2196F3", "#4CAF50",
                   "#FF9800", "#9C27B0", "#795548", "#607D8B", "#00BCD4"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Feature extraction helpers
#     (mirrors unsup_theory_benchmark.py so the two pipelines stay in sync)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_gmm(Z: np.ndarray, k: int, seed: int = 42,
            n_init: int = 30) -> tuple:
    """
    Fit GMM with full → diag → tied → spherical covariance fallback.
    Returns (labels, probs, gmm_model).
    """
    Z64 = Z.astype(np.float64)
    for cov in ("full", "diag", "tied", "spherical"):
        try:
            gmm    = GaussianMixture(n_components=k, n_init=n_init,
                                     random_state=seed, covariance_type=cov,
                                     max_iter=400, reg_covar=1e-3)
            labels = gmm.fit_predict(Z64)
            probs  = gmm.predict_proba(Z64)
            return labels, probs, gmm
        except Exception:
            continue
    raise RuntimeError(f"GMM fitting failed for all covariance types (k={k}).")


# ═══════════════════════════════════════════════════════════════════════════════
# 1b.  Student-t Mixture Model  (Peel & McLachlan 2000)
# ═══════════════════════════════════════════════════════════════════════════════

class StudentTMixture:
    """
    Multivariate Student-t mixture model fitted by EM.

    Reference
    ---------
    Peel & McLachlan (2000). "Robust mixture modelling using the t distribution."
    Statistics and Computing 10: 339-348. DOI: 10.1023/A:1008981510081

    Motivation
    ----------
    EWS features (variance, AC ratio) are heavy-tailed near bifurcation, causing
    standard GMM to over-inflate cluster covariances. Each t-component has an
    additional per-observation weight  u_ij = (ν + p) / (ν + δ_ij)  that
    downweights outliers in the M-step update (δ_ij = Mahalanobis distance²).
    When ν → ∞ the model reduces exactly to GaussianMixture.

    Parameters
    ----------
    n_components : int   — number of components (same role as k in GMM)
    nu_init      : float — initial degrees of freedom (4 = moderately heavy tails)
    estimate_nu  : bool  — if True, estimate ν per component by EM fixed-point;
                           if False, keep nu_init fixed (faster, more stable)
    max_iter     : int   — maximum EM iterations per restart
    tol          : float — convergence tolerance on total log-likelihood
    n_init       : int   — number of random restarts (best LL is kept)
    reg_covar    : float — diagonal regularisation added to each Σ_j
    random_state : int
    """

    def __init__(self, n_components: int = 2, nu_init: float = 4.0,
                 estimate_nu: bool = True, max_iter: int = 200,
                 tol: float = 1e-4, n_init: int = 5,
                 reg_covar: float = 1e-3, random_state: int = 42):
        self.n_components = n_components
        self.nu_init      = float(nu_init)
        self.estimate_nu  = estimate_nu
        self.max_iter     = max_iter
        self.tol          = tol
        self.n_init       = n_init
        self.reg_covar    = reg_covar
        self.random_state = random_state
        # Fitted attributes
        self.pi_          = None   # (k,)   mixture weights
        self.mu_          = None   # (k, p) component means
        self.Sigma_       = None   # (k, p, p) scale matrices
        self.nu_          = None   # (k,)  degrees of freedom
        self.lower_bound_ = -np.inf

    # ── Public API ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "StudentTMixture":
        X   = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        best_ll, best_params = -np.inf, None
        for _ in range(self.n_init):
            seed_i = int(rng.integers(0, 2**31))
            try:
                params, ll = self._fit_single(X, seed_i)
                if ll > best_ll:
                    best_ll, best_params = ll, params
            except Exception:
                continue
        if best_params is None:
            raise RuntimeError("StudentTMixture: all restarts failed.")
        self.pi_, self.mu_, self.Sigma_, self.nu_ = best_params
        self.lower_bound_ = best_ll
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X     = np.asarray(X, dtype=np.float64)
        p     = X.shape[1]
        log_r = self._log_r(X, self.pi_, self.mu_, self.Sigma_, self.nu_, p)
        return np.exp(log_r - logsumexp(log_r, axis=1, keepdims=True))

    # ── EM internals ────────────────────────────────────────────────────────────

    def _fit_single(self, X: np.ndarray, seed: int):
        N, p = X.shape
        k    = self.n_components
        # Warm-start from GMM (much better than pure random init)
        pi, mu, Sigma = self._init_from_gmm(X, k, seed, self.reg_covar)
        nu = np.full(k, self.nu_init)
        prev_ll = -np.inf

        for _ in range(self.max_iter):
            # ── E-step: log responsibilities + precision weights ──────────────
            log_r, u, ll = self._e_step(X, pi, mu, Sigma, nu, p)
            r = np.exp(log_r - logsumexp(log_r, axis=1, keepdims=True))  # (N, k)

            # ── M-step ───────────────────────────────────────────────────────
            n_j = np.maximum(r.sum(axis=0), 1e-10)          # (k,)
            pi  = n_j / N

            for j in range(k):
                rj   = r[:, j]           # (N,)
                uj   = u[:, j]           # (N,)
                wj   = rj * uj           # effective weight
                wsum = wj.sum()
                if wsum < 1e-10:
                    continue

                # Weighted mean
                mu[j] = (wj[:, None] * X).sum(axis=0) / wsum

                # Weighted scale matrix
                diff     = X - mu[j]    # (N, p)
                Sigma[j] = (
                    (wj[:, None, None] * diff[:, :, None] * diff[:, None, :])
                    .sum(axis=0) / n_j[j]
                ) + np.eye(p) * self.reg_covar

                # ν update via fixed-point (Peel & McLachlan eq. after (9))
                if self.estimate_nu:
                    nu[j] = self._update_nu(nu[j], rj, uj, n_j[j], p)

            # ── Convergence ───────────────────────────────────────────────────
            if abs(ll - prev_ll) < self.tol * (abs(prev_ll) + 1.0):
                break
            prev_ll = ll

        return (pi.copy(), mu.copy(), Sigma.copy(), nu.copy()), prev_ll

    @staticmethod
    def _init_from_gmm(X, k, seed, reg):
        """Bootstrap from GMM for a stable warm start."""
        gmm = GaussianMixture(n_components=k, n_init=1, max_iter=80,
                              covariance_type="full", reg_covar=reg,
                              random_state=seed)
        gmm.fit(X)
        return gmm.weights_.copy(), gmm.means_.copy(), gmm.covariances_.copy()

    @staticmethod
    def _log_t_density(X, mu, Sigma, nu, p):
        """
        Log-density of t_ν(μ, Σ) for each row of X.
        Returns (log_density (N,), Mahalanobis² (N,)).
        Uses Cholesky decomposition for numerical stability.
        """
        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(Sigma + np.eye(p) * 1e-6)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        diff    = X - mu                           # (N, p)
        v       = np.linalg.solve(L, diff.T)       # (p, N)
        delta   = np.sum(v ** 2, axis=0)           # (N,) Mahalanobis²
        log_dens = (
            gammaln((nu + p) / 2.0)
            - gammaln(nu / 2.0)
            - (p / 2.0) * np.log(nu * np.pi)
            - 0.5 * log_det
            - ((nu + p) / 2.0) * np.log1p(delta / nu)
        )
        return log_dens, delta

    @classmethod
    def _log_r(cls, X, pi, mu, Sigma, nu, p):
        """Unnormalised log responsibilities (N, k)."""
        k     = len(pi)
        log_r = np.empty((len(X), k), dtype=np.float64)
        for j in range(k):
            ld, _       = cls._log_t_density(X, mu[j], Sigma[j], nu[j], p)
            log_r[:, j] = np.log(max(pi[j], 1e-300)) + ld
        return log_r

    @classmethod
    def _e_step(cls, X, pi, mu, Sigma, nu, p):
        """
        E-step: compute unnormalised log responsibilities and precision weights.
        Returns log_r (N,k), u (N,k), total log-likelihood scalar.
        Single pass per component to avoid redundant Cholesky decompositions.
        """
        k     = len(pi)
        N     = len(X)
        log_r = np.empty((N, k), dtype=np.float64)
        u     = np.empty((N, k), dtype=np.float64)
        for j in range(k):
            ld, delta   = cls._log_t_density(X, mu[j], Sigma[j], nu[j], p)
            log_r[:, j] = np.log(max(pi[j], 1e-300)) + ld
            u[:, j]     = (nu[j] + p) / (nu[j] + delta)   # precision weight
        ll = float(logsumexp(log_r, axis=1).sum())
        return log_r, u, ll

    @staticmethod
    def _update_nu(nu_old: float, r_j: np.ndarray, u_j: np.ndarray,
                   n_j: float, p: int) -> float:
        """
        Numerical solution for ν_j via brentq on the fixed-point equation
        (Peel & McLachlan 2000, eq. after eq. 9):

            -ψ(ν/2) + log(ν/2) + 1 + c + ψ((ν+p)/2) - log((ν+p)/2) = 0

        where c = (1/n_j) Σ_i r_ij [log(u_ij) - u_ij].
        Falls back to nu_old if no root is found in [0.5, 200].
        """
        safe_u = np.clip(u_j, 1e-10, None)
        c = float((r_j * (np.log(safe_u) - safe_u)).sum() / max(n_j, 1e-10))

        def f(nu_):
            return (
                -digamma(nu_ / 2.0) + np.log(nu_ / 2.0) + 1.0
                + c
                + digamma((nu_ + p) / 2.0) - np.log((nu_ + p) / 2.0)
            )

        try:
            if f(0.5) * f(200.0) < 0:
                return float(np.clip(
                    brentq(f, 0.5, 200.0, xtol=1e-3, maxiter=25), 0.5, 200.0))
        except Exception:
            pass
        return float(nu_old)


def fit_t_mixture(Z: np.ndarray, k: int, seed: int = 42,
                  n_init: int = 5) -> tuple:
    """
    Fit a Student-t mixture model with the same interface as fit_gmm.
    Returns (labels, probs, model).
    n_init is lower than GMM (5 vs 30) because each restart is slower
    due to the ν update; warm-start from GMM compensates.
    """
    Z64   = Z.astype(np.float64)
    model = StudentTMixture(
        n_components=k, nu_init=4.0, estimate_nu=True,
        max_iter=200, tol=1e-4, n_init=n_init,
        reg_covar=1e-3, random_state=seed,
    )
    labels = model.fit_predict(Z64)
    probs  = model.predict_proba(Z64)
    return labels, probs, model


# ═══════════════════════════════════════════════════════════════════════════════
# 1c.  Skew-t Mixture Model  (Lin 2010 EM)
# ═══════════════════════════════════════════════════════════════════════════════

class SkewTMixture:
    """
    Multivariate skew-t mixture via EM (Lin 2010).

    Stochastic representation for component j:
        Y = μ_j + δ_j |W| + Σ_j^{1/2} Z / sqrt(V / ν_j)
    W ~ N(0,1), Z ~ N_p(0, I), V ~ χ²(ν_j), all independent.
    δ_j ∈ ℝ^p is the skewness vector; δ_j = 0 recovers StudentTMixture.

    Reference
    ---------
    Lin, T.-I. (2010).  Robust mixture modeling using multivariate skew
    t distributions.  Statistics and Computing, 20(3), 343-356.
    """

    def __init__(self, n_components=2, nu_init=4.0, estimate_nu=True,
                 max_iter=200, tol=1e-4, n_init=5, reg_covar=1e-3,
                 random_state=42):
        self.n_components = n_components
        self.nu_init      = float(nu_init)
        self.estimate_nu  = estimate_nu
        self.max_iter     = max_iter
        self.tol          = tol
        self.n_init       = n_init
        self.reg_covar    = reg_covar
        self.random_state = random_state
        self.pi_          = None
        self.mu_          = None
        self.Sigma_       = None
        self.delta_       = None   # (k, p) skewness vectors
        self.nu_          = None
        self.lower_bound_ = -np.inf

    def fit(self, X):
        X   = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        best_ll, best_params = -np.inf, None
        for _ in range(self.n_init):
            seed_i = int(rng.integers(0, 2**31))
            try:
                params, ll = self._fit_single(X, seed_i)
                if ll > best_ll:
                    best_ll, best_params = ll, params
            except Exception:
                continue
        if best_params is None:
            raise RuntimeError("SkewTMixture: all restarts failed.")
        self.pi_, self.mu_, self.Sigma_, self.delta_, self.nu_ = best_params
        self.lower_bound_ = best_ll
        return self

    def fit_predict(self, X):
        self.fit(X); return self.predict(X)

    def predict(self, X):
        X  = np.asarray(X, dtype=np.float64)
        lr = self._log_resp(X, self.pi_, self.mu_, self.Sigma_,
                            self.delta_, self.nu_)
        return np.argmax(lr, axis=1)

    def predict_proba(self, X):
        X  = np.asarray(X, dtype=np.float64)
        lr = self._log_resp(X, self.pi_, self.mu_, self.Sigma_,
                            self.delta_, self.nu_)
        lr -= logsumexp(lr, axis=1, keepdims=True)
        return np.exp(lr)

    # ── EM ─────────────────────────────────────────────────────────────────────

    def _fit_single(self, X, seed):
        n, p = X.shape
        k    = self.n_components

        # Warm-start from TMM (good μ, Σ, ν, π); δ initialised to 0
        tmm = StudentTMixture(
            n_components=k, nu_init=self.nu_init, estimate_nu=True,
            max_iter=60, tol=1e-3, n_init=1, reg_covar=self.reg_covar,
            random_state=seed,
        )
        try:
            tmm.fit(X)
            pi    = tmm.pi_.copy()
            mu    = tmm.mu_.copy()
            Sigma = tmm.Sigma_.copy()
            nu    = tmm.nu_.copy()
        except Exception:
            rng2  = np.random.default_rng(seed)
            idx   = rng2.choice(n, k, replace=False)
            pi    = np.ones(k) / k
            mu    = X[idx].copy()
            Sigma = np.array([np.eye(p) * np.var(X) for _ in range(k)])
            nu    = np.full(k, self.nu_init)
        delta = np.zeros((k, p))   # no skewness at initialisation

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            log_r, w, E1, E2, ll = self._e_step(
                X, pi, mu, Sigma, delta, nu, p)
            if abs(ll - prev_ll) < self.tol * max(1.0, abs(ll)):
                break
            prev_ll = ll

            r = np.exp(log_r - logsumexp(log_r, axis=1, keepdims=True))

            for j in range(k):
                r_j  = r[:, j];   w_j  = w[:, j]
                e1_j = E1[:, j];  e2_j = E2[:, j]
                rw_j = r_j * w_j
                n_j  = float(r_j.sum())
                D_j  = float(rw_j.sum())
                if D_j < 1e-10:
                    continue

                # Weighted sufficient statistics
                A_j = rw_j @ X                           # (p,)
                B_j = float((rw_j * e1_j).sum())         # scalar
                C_j = float((rw_j * e2_j).sum())         # scalar
                E_j = (rw_j * e1_j) @ X                  # (p,)

                # Closed-form joint update for μ_j and δ_j
                denom = C_j * D_j - B_j ** 2
                if abs(denom) < 1e-10:
                    delta_j = delta[j]
                    mu_j    = A_j / D_j
                else:
                    delta_j = (E_j * D_j - A_j * B_j) / denom
                    mu_j    = (A_j - delta_j * B_j) / D_j

                # Σ_j: residuals after removing skewness
                z_ij = X - mu_j[None] - delta_j[None] * e1_j[:, None]
                s2_j = np.maximum(e2_j - e1_j ** 2, 0.0)

                Sig_j = (
                    (rw_j[:, None, None]
                     * z_ij[:, :, None] * z_ij[:, None, :]).sum(0)
                    + float((r_j * s2_j).sum()) * np.outer(delta_j, delta_j)
                ) / n_j
                Sig_j += self.reg_covar * np.eye(p)

                if self.estimate_nu:
                    nu[j] = StudentTMixture._update_nu(nu[j], r_j, w_j, n_j, p)

                pi[j]    = n_j / n
                mu[j]    = mu_j
                delta[j] = delta_j
                Sigma[j] = Sig_j

        return (pi, mu, Sigma, delta, nu), prev_ll

    # ── E-step & density ───────────────────────────────────────────────────────

    @classmethod
    def _e_step(cls, X, pi, mu, Sigma, delta, nu, p):
        from scipy.stats import norm as _norm, t as _t
        n, k   = len(X), len(pi)
        log_r  = np.zeros((n, k))
        w_out  = np.zeros((n, k))
        E1_out = np.zeros((n, k))
        E2_out = np.zeros((n, k))

        for j in range(k):
            try:
                L = np.linalg.cholesky(Sigma[j] + 1e-8 * np.eye(p))
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(Sigma[j] + 1e-3 * np.eye(p))

            Linv = np.linalg.inv(L)
            Ld   = Linv @ delta[j]              # Σ^{-1/2} δ,  (p,)
            sd   = float(Ld @ Ld)               # δᵀΣ⁻¹δ

            log_det_Omega = (2.0 * np.sum(np.log(np.diag(L)))
                             + np.log1p(sd))

            R    = X - mu[j]                    # (n, p)
            LR   = (Linv @ R.T).T               # L⁻¹R,  (n, p)

            # Mahalanobis w.r.t. Ω = Σ + δδᵀ  (Sherman-Morrison)
            Ld_LR   = LR @ Ld                   # (n,)
            Q_Omega = (LR * LR).sum(1) - Ld_LR ** 2 / (1.0 + sd)

            # Log multivariate t density w.r.t. Ω
            ld_t = (gammaln((nu[j] + p) / 2.0)
                    - gammaln(nu[j] / 2.0)
                    - (p / 2.0) * np.log(np.pi * nu[j])
                    - 0.5 * log_det_Omega
                    - ((nu[j] + p) / 2.0) * np.log1p(Q_Omega / nu[j]))

            # Skewness factor c = δᵀΩ⁻¹(x-μ) / sqrt(1 - δᵀΩ⁻¹δ)
            # = Ld_LR / sqrt(1+sd)
            c_ij    = Ld_LR / np.sqrt(1.0 + sd)
            t_arg   = c_ij * np.sqrt(
                (nu[j] + p) / np.maximum(nu[j] + Q_Omega, 1e-10))
            log_cdf = np.maximum(_t.logcdf(t_arg, df=nu[j] + p), -500.0)

            log_r[:, j] = (np.log(max(pi[j], 1e-300))
                           + np.log(2.0) + ld_t + log_cdf)

            w_out[:, j] = (nu[j] + p) / np.maximum(nu[j] + Q_Omega, 1e-10)

            # Truncated-normal moments for |W| (conditional on y, V ≈ w)
            # W | y, V=v ~ TN(a, b², 0, ∞)
            # a = δᵀΣ⁻¹(y-μ)/(1+sd),  b² = 1/(w·(1+sd))
            a_ij  = Ld_LR / (1.0 + sd)
            b2_ij = 1.0 / np.maximum(w_out[:, j] * (1.0 + sd), 1e-10)
            b_ij  = np.sqrt(b2_ij)
            alpha = a_ij / np.maximum(b_ij, 1e-10)

            # Mills ratio R(α) = φ(α)/Φ(α); asymptotic R(α) ≈ -α for α → -∞
            mills = np.where(
                alpha > -8.0,
                _norm.pdf(alpha) / np.maximum(_norm.cdf(alpha), 1e-300),
                -alpha)

            E1_out[:, j] = a_ij + b_ij * mills
            E2_out[:, j] = b2_ij + a_ij ** 2 + a_ij * b_ij * mills

        ll = float(logsumexp(log_r, axis=1).sum())
        return log_r, w_out, E1_out, E2_out, ll

    @classmethod
    def _log_resp(cls, X, pi, mu, Sigma, delta, nu):
        p = X.shape[1]
        lr, _, _, _, _ = cls._e_step(X, pi, mu, Sigma, delta, nu, p)
        return lr


def fit_skew_t_mixture(Z: np.ndarray, k: int, seed: int = 42,
                       n_init: int = 5) -> tuple:
    """
    Fit multivariate skew-t mixture (Lin 2010 EM). Returns (labels, probs, model).
    Warm-started from StudentTMixture; δ_j initialised to zero.
    """
    Z64   = Z.astype(np.float64)
    model = SkewTMixture(
        n_components=k, nu_init=4.0, estimate_nu=True,
        max_iter=200, tol=1e-4, n_init=n_init,
        reg_covar=1e-3, random_state=seed,
    )
    labels = model.fit_predict(Z64)
    probs  = model.predict_proba(Z64)
    return labels, probs, model


# ═══════════════════════════════════════════════════════════════════════════════
# 1e.  Theory-Informed Bayesian GMM  (MAP-EM with Normal-Wishart priors)
# ═══════════════════════════════════════════════════════════════════════════════

class TheoryBayesGMM:
    """
    MAP-EM Gaussian Mixture with per-component Normal-Wishart priors
    anchored to synthetic bifurcation class centroids.

    Theory anchor
    -------------
    Synthetic SN / TC / Null centroids (projected into the PCA feature space)
    are used as prior means for the first min(n_components, 3) components.
    This encodes the expectation that real-world clusters should lie near the
    bifurcation-type centroids derived from ODE simulation.

    MAP updates (closed form per iteration)
    ----------------------------------------
    For component k with prior mean μ₀ₖ, prior strength β₀,
    and inverse-Wishart prior Ψ₀ = ψ₀·I, df ν₀:

        Nₖ    = Σᵢ rᵢₖ
        x̄ₖ    = Σᵢ rᵢₖ xᵢ / Nₖ
        μₖ*   = (Nₖ x̄ₖ + β₀ μ₀ₖ) / (Nₖ + β₀)          ← centroid shrinkage
        Sₖ    = Σᵢ rᵢₖ (xᵢ − x̄ₖ)(xᵢ − x̄ₖ)ᵀ
        cross = β₀Nₖ/(β₀+Nₖ) · (x̄ₖ − μ₀ₖ)(x̄ₖ − μ₀ₖ)ᵀ  ← prior pull
        Σₖ*   = (Sₖ + Ψ₀ + cross + ε·I) / (Nₖ + ν₀ + p + 1)
        πₖ*   = (Nₖ + α/K) / (N + α)                     ← Dirichlet posterior

    β₀→0 reduces to standard GMM M-step; Nₖ→∞ makes data dominate.

    Parameters
    ----------
    n_components    : int
    centroids_prior : ndarray (C, p), C ≤ n_components
        Synthetic class centroids projected into PCA feature space.
        Class order: [saddle_node, transcritical, null].
        Components beyond C get the data mean as prior.
    beta0           : float — prior strength; ~10 = "10 virtual obs" from centroid
    nu0             : float or None — Wishart df (None → p+2, minimal informative)
    psi0_scale      : float — Ψ₀ = psi0_scale·I  (covariance regularisation)
    alpha           : float — Dirichlet concentration (small → sparse weights)
    max_iter, tol, n_init, reg_covar, random_state : standard EM parameters
    """

    def __init__(self, n_components, centroids_prior,
                 beta0=10.0, nu0=None, psi0_scale=0.1,
                 alpha=1.0, max_iter=300, tol=1e-4,
                 n_init=5, reg_covar=1e-3, random_state=42):
        self.n_components    = n_components
        self.centroids_prior = np.asarray(centroids_prior, dtype=np.float64)
        self.beta0           = float(beta0)
        self.nu0             = nu0
        self.psi0_scale      = float(psi0_scale)
        self.alpha           = float(alpha)
        self.max_iter        = max_iter
        self.tol             = tol
        self.n_init          = n_init
        self.reg_covar       = float(reg_covar)
        self.random_state    = random_state
        self.pi_             = None
        self.mu_             = None
        self.Sigma_          = None
        self.lower_bound_    = -np.inf

    # ── Public API ───────────────────────────────────────────────────────────────

    def fit(self, X):
        X   = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        best_ll, best_params = -np.inf, None
        for _ in range(self.n_init):
            seed_i = int(rng.integers(0, 2**31))
            try:
                params, ll = self._fit_single(X, seed_i)
                if ll > best_ll:
                    best_ll, best_params = ll, params
            except Exception:
                continue
        if best_params is None:
            raise RuntimeError("TheoryBayesGMM: all restarts failed.")
        self.pi_, self.mu_, self.Sigma_ = best_params
        self.lower_bound_ = best_ll
        return self

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        X    = np.asarray(X, dtype=np.float64)
        logr = self._log_resp(X, self.pi_, self.mu_, self.Sigma_)
        logr -= logsumexp(logr, axis=1, keepdims=True)
        return np.exp(logr)

    # ── EM internals ─────────────────────────────────────────────────────────────

    def _fit_single(self, X, seed):
        N, p = X.shape
        K    = self.n_components
        nu0  = float(self.nu0) if self.nu0 is not None else float(p + 2)
        Psi0 = np.eye(p) * self.psi0_scale

        # Build per-component prior means
        C      = len(self.centroids_prior)
        x_mean = X.mean(axis=0)
        mu0    = np.array([
            self.centroids_prior[j] if j < C else x_mean
            for j in range(K)
        ])

        pi, mu, Sigma = self._init(X, mu0, seed)
        prev_ll = -np.inf

        for _ in range(self.max_iter):
            # E-step
            logr = self._log_resp(X, pi, mu, Sigma)
            ll   = float(logsumexp(logr, axis=1).sum())
            logr -= logsumexp(logr, axis=1, keepdims=True)
            r    = np.exp(logr)  # (N, K)

            # M-step
            pi, mu, Sigma = self._m_step(X, r, mu0, nu0, Psi0, N, p)

            if abs(ll - prev_ll) < self.tol * (abs(prev_ll) + 1.0):
                break
            prev_ll = ll

        return (pi.copy(), mu.copy(), Sigma.copy()), prev_ll

    def _init(self, X, mu0, seed):
        """Initialise near theory centroids with small Gaussian noise."""
        K, p  = self.n_components, X.shape[1]
        rng   = np.random.default_rng(seed)
        noise = float(np.std(X)) * 0.1
        mu    = mu0.copy() + rng.standard_normal((K, p)) * noise
        var   = float(np.var(X, axis=0).mean())
        Sigma = np.array([np.eye(p) * (var + self.reg_covar)] * K)
        pi    = np.ones(K) / K
        return pi, mu, Sigma

    def _log_resp(self, X, pi, mu, Sigma):
        """Unnormalised log responsibilities (N, K)."""
        N, K  = len(X), self.n_components
        log_r = np.empty((N, K), dtype=np.float64)
        for j in range(K):
            log_r[:, j] = np.log(max(pi[j], 1e-300)) + self._log_gauss(X, mu[j], Sigma[j])
        return log_r

    @staticmethod
    def _log_gauss(X, mu, Sigma):
        p = X.shape[1]
        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(Sigma + np.eye(p) * 1e-6)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        diff    = X - mu
        v       = np.linalg.solve(L, diff.T)   # (p, N)
        maha    = np.sum(v ** 2, axis=0)       # (N,)
        return -0.5 * (p * np.log(2 * np.pi) + log_det + maha)

    def _m_step(self, X, r, mu0, nu0, Psi0, N, p):
        K   = self.n_components
        mu  = np.zeros((K, p))
        Sig = np.zeros((K, p, p))
        Nk  = r.sum(axis=0)  # (K,)

        for j in range(K):
            nj = float(Nk[j])
            rj = r[:, j]

            if nj < 1e-6:
                mu[j]  = mu0[j]
                Sig[j] = Psi0 + np.eye(p) * self.reg_covar
                continue

            xbar = (rj[:, None] * X).sum(0) / nj

            # MAP mean: weighted average of empirical mean and centroid prior
            mu[j] = (nj * xbar + self.beta0 * mu0[j]) / (nj + self.beta0)

            # Scatter around empirical mean
            diff = X - xbar
            S    = (rj[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(0)

            # Cross-term: bias pull toward centroid
            dp    = xbar - mu0[j]
            cross = (self.beta0 * nj) / (self.beta0 + nj) * np.outer(dp, dp)

            # MAP covariance (posterior IW mode)
            Sig[j] = (S + Psi0 + cross + np.eye(p) * self.reg_covar) / (nj + nu0 + p + 1)

        # Dirichlet posterior mean for mixing weights
        pi = (Nk + self.alpha / K) / (N + self.alpha)
        pi = np.clip(pi, 1e-300, None)
        pi /= pi.sum()
        return pi, mu, Sig


def fit_theory_bayes_gmm(Z: np.ndarray, k: int,
                          centroids_prior: np.ndarray,
                          seed: int = 42, n_init: int = 5) -> tuple:
    """
    Fit Theory-Informed Bayesian GMM (MAP-EM with Normal-Wishart priors).

    Parameters
    ----------
    Z               : (N, p) feature matrix (PCA-whitened theory features)
    k               : number of components
    centroids_prior : (C, p) synthetic class centroids in PCA feature space
                      Class order: [saddle_node, transcritical, null]
    seed, n_init    : as usual

    Returns (labels, probs, model).
    """
    Z64   = Z.astype(np.float64)
    model = TheoryBayesGMM(
        n_components    = k,
        centroids_prior = centroids_prior,
        beta0           = 10.0,
        nu0             = None,    # p + 2
        psi0_scale      = 0.1,
        alpha           = 1.0,
        max_iter        = 300,
        tol             = 1e-4,
        n_init          = n_init,
        reg_covar       = 1e-3,
        random_state    = seed,
    )
    labels = model.fit_predict(Z64)
    probs  = model.predict_proba(Z64)
    return labels, probs, model


# ── Legacy uninformed Bayesian GMM (kept for reference, not used in pipeline) ──
def fit_bayes_gmm(Z: np.ndarray, k: int, seed: int = 42,
                  n_init: int = 3) -> tuple:
    """
    [LEGACY] Fit sklearn BayesianGaussianMixture with Dirichlet-Process prior.
    Uninformative version — no theory centroid priors.
    Replaced by fit_theory_bayes_gmm in the active pipeline.
    """
    from sklearn.mixture import BayesianGaussianMixture
    Z64 = Z.astype(np.float64)
    best_ll, best_labels, best_probs, best_model = -np.inf, None, None, None
    for i in range(n_init):
        try:
            bgmm = BayesianGaussianMixture(
                n_components=k, covariance_type="full",
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=1.0 / k,
                max_iter=500, n_init=1, random_state=seed + i, reg_covar=1e-3,
            )
            bgmm.fit(Z64)
            ll = bgmm.lower_bound_
            if ll > best_ll:
                best_ll, best_labels, best_probs, best_model = (
                    ll, bgmm.predict(Z64), bgmm.predict_proba(Z64), bgmm)
        except Exception:
            continue
    if best_model is None:
        raise RuntimeError("BayesianGMM: all restarts failed.")
    return best_labels, best_probs, best_model


def rocket_features(X: np.ndarray, n_kernels: int = 2000,
                    seed: int = 42) -> np.ndarray:
    """
    ROCKET: random convolutional kernels → max + proportion-positive features.
    Uses kernel lengths [7,9,11,21,41,101] (same as benchmark) for global shape.
    Returns (N, 2*n_kernels) float32.
    """
    rng    = np.random.default_rng(seed)
    N, L   = X.shape
    out    = np.zeros((N, 2 * n_kernels), dtype=np.float32)
    k_lens = [7, 9, 11, 21, 41, 101]

    for ki in range(n_kernels):
        kl      = k_lens[ki % len(k_lens)]
        weights = rng.standard_normal(kl).astype(np.float32)
        weights -= weights.mean()
        max_dil = max(1, int(np.log2(L / kl)))
        dil     = int(rng.integers(1, max_dil + 1))
        eff_len = (kl - 1) * dil + 1
        pad     = eff_len // 2
        for i in range(N):
            x_pad = np.pad(X[i].astype(np.float32), pad, mode="reflect")
            conv  = np.zeros(L, dtype=np.float32)
            for j, w in enumerate(weights):
                start = j * dil
                conv += w * x_pad[start : start + L]
            out[i, 2 * ki]     = float(conv.max())
            out[i, 2 * ki + 1] = float((conv > 0).mean())

    return out


def _cluster_scores(Z: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute silhouette, Calinski-Harabasz, Davies-Bouldin.
    Returns NaN for degenerate configurations.
    """
    Z64  = Z.astype(np.float64)
    mask = labels >= 0
    if mask.sum() < 10:
        return {"silhouette": np.nan, "calinski": np.nan, "davies": np.nan}
    Xm, ym = Z64[mask], labels[mask]
    uni, cnt = np.unique(ym, return_counts=True)
    if len(uni) < 2 or cnt.min() < 2:
        return {"silhouette": np.nan, "calinski": np.nan, "davies": np.nan}
    try:
        sil = float(silhouette_score(Xm, ym, sample_size=min(len(Xm), 3000),
                                     random_state=42))
    except Exception:
        sil = np.nan
    try:
        ch = float(calinski_harabasz_score(Xm, ym))
    except Exception:
        ch = np.nan
    try:
        db = float(davies_bouldin_score(Xm, ym))
    except Exception:
        db = np.nan
    return {"silhouette": sil, "calinski": ch, "davies": db}


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Feature bundle builder for real-world data
# ═══════════════════════════════════════════════════════════════════════════════

class RealWorldFeatures:
    """
    Holds all feature representations for the real-world dataset.

    Attributes
    ----------
    X           : (N, 500) float32  normalised adoption series
    Z_ae        : (N, d)   float32  PCA-whitened AE embeddings
    F_theory    : (N, 38)  float64  standardised theory features (raw)
    F_white     : (N, ≤20) float32  PCA-whitened theory features
    R_white     : (N, ≤50) float32  PCA-whitened ROCKET features
    Z_ens_white : (N, ≤30) float32  PCA-whitened Theory+AE
    Z_tr_white  : (N, ≤60) float32  PCA-whitened Theory+ROCKET
    feature_names : list of str     38 theory feature names
    """

    def __init__(self):
        pass

    @classmethod
    def build(
        cls,
        X:          np.ndarray,
        lengths:    np.ndarray,
        bundle_path: str,
        n_kernels:  int  = 2000,
        latent_dim: int  = 32,
        ae_epochs:  int  = 80,
        seed:       int  = 42,
        verbose:    bool = True,
    ) -> "RealWorldFeatures":

        from unsup_theory_features import extract_theory_features, standardise, FEATURE_NAMES

        obj = cls()
        obj.X = X.astype(np.float32)

        # ── AE bundle ──────────────────────────────────────────────────────
        enc_path = bundle_path.replace(".pkl", "_encoder.pt")
        bundle   = None
        try:
            import torch  # noqa: F401
            torch_ok = True
        except ImportError:
            torch_ok = False
            if verbose:
                print("  [torch not available — AE features replaced by PCA]")

        if torch_ok and os.path.exists(bundle_path):
            try:
                from unsup_features import FeatureBundle
                bundle = FeatureBundle.load(bundle_path, latent_dim=latent_dim)
                if verbose:
                    print(f"  Loaded AE bundle: {bundle.ae_whitened.shape}")
            except Exception as e:
                if verbose:
                    print(f"  Stale AE bundle — rebuilding. ({e})")
                for p in (bundle_path, enc_path):
                    if os.path.exists(p):
                        os.remove(p)
                bundle = None

        if torch_ok and bundle is None:
            try:
                from unsup_features import FeatureBundle
                bundle = FeatureBundle.build(
                    X, lengths, latent_dim=latent_dim,
                    ae_epochs=ae_epochs, verbose=verbose,
                )
                bundle.save(bundle_path)
            except ImportError:
                if verbose:
                    print("  unsup_features not available — skipping AE bundle (PCA fallback).")
                bundle = None

        if torch_ok and bundle is not None:
            obj.Z_ae = bundle.ae_whitened.astype(np.float32)
        else:
            # PCA fallback
            _pca = PCA(n_components=min(32, X.shape[1]),
                       whiten=True, random_state=seed)
            obj.Z_ae = _pca.fit_transform(X.astype(np.float64)).astype(np.float32)

        # ── Theory features ────────────────────────────────────────────────
        if verbose:
            print("\n  Extracting theory features...")
        F_raw = extract_theory_features(X, verbose=verbose)
        obj.F_raw = F_raw.astype(np.float64)          # raw features for centroid matching
        obj.F_theory, _mu, _std = standardise(F_raw)
        obj.F_theory = obj.F_theory.astype(np.float64)

        pca_th  = PCA(n_components=min(20, obj.F_theory.shape[1]),
                      whiten=True, random_state=seed)
        obj.F_white    = pca_th.fit_transform(obj.F_theory).astype(np.float32)
        obj.pca_theory = pca_th   # stored so callers can project synthetic centroids

        obj.feature_names = FEATURE_NAMES

        # ── ROCKET features ────────────────────────────────────────────────
        if verbose:
            print(f"  Computing ROCKET features ({n_kernels} kernels)...",
                  end=" ", flush=True)
        R = rocket_features(X, n_kernels=n_kernels, seed=seed)
        if verbose:
            print("done")
        pca_r   = PCA(n_components=min(50, R.shape[1]),
                      whiten=True, random_state=seed)
        obj.R_white = pca_r.fit_transform(R.astype(np.float64)).astype(np.float32)

        # ── Ensemble: Theory + AE ──────────────────────────────────────────
        sc1  = StandardScaler()
        Z_th_ae = np.hstack([
            sc1.fit_transform(obj.Z_ae.astype(np.float64)),
            sc1.fit_transform(obj.F_theory.astype(np.float64)),
        ]).astype(np.float32)
        pca_ens = PCA(n_components=min(30, Z_th_ae.shape[1]),
                      whiten=True, random_state=seed)
        obj.Z_ens_white = pca_ens.fit_transform(
            Z_th_ae.astype(np.float64)).astype(np.float32)

        # ── Theory + ROCKET ────────────────────────────────────────────────
        sc2  = StandardScaler()
        Z_th_rk = np.hstack([
            sc2.fit_transform(obj.F_theory.astype(np.float64)),
            sc2.fit_transform(obj.R_white.astype(np.float64)),
        ]).astype(np.float32)
        pca_tr  = PCA(n_components=min(60, Z_th_rk.shape[1]),
                      whiten=True, random_state=seed)
        obj.Z_tr_white = pca_tr.fit_transform(
            Z_th_rk.astype(np.float64)).astype(np.float32)

        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  k-sweep engine
# ═══════════════════════════════════════════════════════════════════════════════

METHOD_SPECS = {
    # ── Commented-out methods (kept for reference) ──────────────────────────
    # "ae_gmm":        {"label": "AE-GMM",              "synth_acc": 0.728},
    # "ensemble_gmm":  {"label": "Ensemble (Theory+AE)", "synth_acc": 0.766},
    # "rocket_gmm":    {"label": "ROCKET-GMM",           "synth_acc": 0.950},
    # "theory_rocket": {"label": "Theory+ROCKET",        "synth_acc": 0.918},
    # ── Active methods ───────────────────────────────────────────────────────
    "theory_gmm":    {"label": "Theory-GMM",   "synth_acc": 0.909},
    "theory_t":      {"label": "Theory-TMM",   "synth_acc": 0.934},
    "theory_skewt":  {"label": "Theory-SkewT", "synth_acc": None},
    # ── Archived methods (see runs/unsup/Archive/stage1_bayes_t_raw/) ───────
    # "theory_t_raw":  {"label": "Theory-TMM (no whiten)",  "synth_acc": None},
    #   → degenerate on genuine_v2: 100% SN (labels every series the same class)
    # "theory_bayes":  {"label": "Theory-Bayes GMM",        "synth_acc": None},
    #   → systematic outlier on genuine_v2: 241 SN→TC misclassifications vs
    #     GMM+TMM+SkewT consensus; only 54.7% agreement with GMM
}


def sweep_k_for_method(Z: np.ndarray,
                        method_name: str,
                        k_range,
                        seed: int = 42,
                        n_init: int = 30,
                        verbose: bool = True,
                        fit_fn=None) -> dict:
    """
    Run a mixture model at each k in k_range.

    Parameters
    ----------
    fit_fn : callable or None
        If None, uses fit_gmm (standard GaussianMixture).
        Otherwise called as fit_fn(Z, k, seed=seed) and must return
        (labels, probs, model) — same interface as fit_gmm.
        Use this to plug in fit_t_mixture or any other fitter.

    Returns dict: k → {"labels", "probs", "gmm", "scores", "Z"}
    """
    results = {}
    label   = METHOD_SPECS.get(method_name, {}).get("label", method_name)

    if verbose:
        print(f"\n  [{label}]  k-sweep {list(k_range)}")
        print(f"  {'k':>3}  {'Silhouette':>12}  {'Calinski-H':>12}  "
              f"{'Davies-B':>10}")
        print(f"  {'-'*3}  {'-'*12}  {'-'*12}  {'-'*10}")

    for k in k_range:
        if fit_fn is not None:
            labels, probs, gmm = fit_fn(Z, k, seed=seed)
        else:
            labels, probs, gmm = fit_gmm(Z, k=k, seed=seed, n_init=n_init)
        scores = _cluster_scores(Z, labels)

        def _f(v):
            return f"{v:12.4f}" if np.isfinite(v) else "         nan"

        if verbose:
            print(f"  {k:>3}  {_f(scores['silhouette'])}  "
                  f"{_f(scores['calinski'])}  {_f(scores['davies'])}")

        results[k] = {"labels": labels, "probs": probs,
                      "gmm": gmm, "scores": scores, "Z": Z}

    return results


def best_k_by_silhouette(sweep: dict) -> int:
    """Return k with highest silhouette (NaN-safe)."""
    valid = {k: v["scores"]["silhouette"] for k, v in sweep.items()
             if np.isfinite(v["scores"]["silhouette"])}
    if not valid:
        return min(sweep.keys())
    return max(valid, key=valid.get)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Bifurcation type labelling
#     Uses synthetic class centroids (in theory feature space) as prototypes.
# ═══════════════════════════════════════════════════════════════════════════════

def compute_synthetic_centroids(synth_dir: str,
                                verbose: bool = True) -> tuple:
    """
    Compute mean theory features for each class from the synthetic dataset.

    Features are standardised using the synthetic-data mean/std so that
    centroids live in the same Z-score space.  The same (mu, std) scaler
    must be applied to real-world raw features before distance computation.

    Returns
    -------
    centroids  : {class_name → (38,) centroid vector}  in standardised space
    synth_mu   : (38,) mean used for standardisation
    synth_std  : (38,) std  used for standardisation
    """
    from unsup_theory_features import extract_theory_features, standardise

    X = np.load(os.path.join(synth_dir, "X_full.npy"))
    y = np.load(os.path.join(synth_dir, "y.npy"))

    if verbose:
        print("\n  Computing synthetic class centroids in theory feature space...")

    F_raw = extract_theory_features(X, verbose=False)
    F_std, synth_mu, synth_std = standardise(F_raw)   # fit scaler on synthetic data

    centroids = {}
    for ci, name in enumerate(CLASS_NAMES):
        mask = y == ci
        centroids[name] = F_std[mask].mean(axis=0)
        if verbose:
            print(f"    {name}: n={mask.sum()}")

    return centroids, synth_mu, synth_std


def label_clusters(
    labels:         np.ndarray,
    F_theory:       np.ndarray,
    centroids:      dict,
    k:              int,
) -> dict:
    """
    Assign each cluster a bifurcation type label by nearest centroid in
    theory feature space (1-NN on cluster mean theory features).

    Returns
    -------
    cluster_labels : dict  {cluster_id → bifurcation_type_string}
    cluster_dists  : dict  {cluster_id → {class_name → distance}}
    cluster_F_means: dict  {cluster_id → (38,) mean theory features}
    """
    cluster_labels  = {}
    cluster_dists   = {}
    cluster_F_means = {}

    for cid in range(k):
        mask = labels == cid
        if mask.sum() == 0:
            cluster_labels[cid]  = "unknown"
            cluster_dists[cid]   = {}
            cluster_F_means[cid] = np.zeros(F_theory.shape[1])
            continue

        f_mean = F_theory[mask].mean(axis=0)
        cluster_F_means[cid] = f_mean

        dists = {
            name: float(np.linalg.norm(f_mean - cent))
            for name, cent in centroids.items()
        }
        cluster_dists[cid]  = dists
        cluster_labels[cid] = min(dists, key=dists.get)

    return cluster_labels, cluster_dists, cluster_F_means


def assign_series_labels(
    labels:        np.ndarray,
    cluster_types: dict,
) -> np.ndarray:
    """Map cluster-id array to bifurcation-type string array."""
    return np.array([cluster_types.get(lbl, "unknown") for lbl in labels])


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  2-D embedding (UMAP or PCA fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def embed_2d(Z: np.ndarray, seed: int = 42) -> np.ndarray:
    """Return 2-D embedding using UMAP if available, else PCA."""
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed,
                            n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(Z.astype(np.float64))
    except ImportError:
        pca2 = PCA(n_components=2, random_state=seed)
        return pca2.fit_transform(Z.astype(np.float64))


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_k_sweep_all(all_sweeps: dict, out_dir: str):
    """One figure with 3 metric subplots × N methods."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods  = list(all_sweeps.keys())
    k_values = sorted(all_sweeps[methods[0]].keys())
    metrics  = ["silhouette", "calinski", "davies"]
    m_labels = ["Silhouette ↑", "Calinski-Harabasz ↑", "Davies-Bouldin ↓"]
    colors   = ["#E63946", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, mkey, mlabel in zip(axes, metrics, m_labels):
        for mi, (mname, sweep) in enumerate(all_sweeps.items()):
            vals = [sweep[k]["scores"][mkey] for k in k_values]
            color = colors[mi % len(colors)]
            label = METHOD_SPECS.get(mname, {}).get("label", mname)
            ax.plot(k_values, vals, marker="o", lw=2, color=color, label=label)

            # Mark best k (by silhouette)
            best = best_k_by_silhouette(sweep)
            if mkey == "silhouette":
                bval = sweep[best]["scores"]["silhouette"]
                if np.isfinite(bval):
                    ax.axvline(best, color=color, lw=0.8, ls="--", alpha=0.5)
                    ax.scatter([best], [bval], color=color, s=100, zorder=5)

        ax.set_xlabel("k", fontsize=11)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.set_xticks(k_values)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle("k-sweep — Real-World Technology Adoption Dataset\n"
                 "Vertical dashed lines mark silhouette-optimal k",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "k_sweep_all_methods.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  k-sweep plot → {path}")


def plot_cluster_trajectories(
    X:        np.ndarray,        # (N, 500) normalised series
    labels:   np.ndarray,        # (N,) cluster ids
    types:    dict,              # {cluster_id → bifurcation_type}
    k:        int,
    method:   str,
    out_dir:  str,
    tag:      str = "",
):
    """Mean ± band trajectory for each cluster."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    t      = np.linspace(0, 1, 500)
    n_cls  = len(set(labels[labels >= 0]))
    fig, axes = plt.subplots(1, n_cls, figsize=(4.5 * n_cls, 4.2),
                              squeeze=False)
    axes = axes.flatten()

    for ax, cid in zip(axes, sorted(set(labels[labels >= 0]))):
        mask  = labels == cid
        Xc    = X[mask]
        color = CLASS_COLORS.get(types.get(cid, "unknown"), "#607D8B")

        p10  = np.percentile(Xc, 10, axis=0)
        p90  = np.percentile(Xc, 90, axis=0)
        mean = Xc.mean(axis=0)
        std  = Xc.std(axis=0)

        ax.fill_between(t, p10, p90, color=color, alpha=0.18)
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.30)
        ax.plot(t, mean, color=color, lw=2.2)

        bif_type = types.get(cid, "?")
        ax.set_title(f"Cluster {cid}  (n={mask.sum()})\n→ {bif_type}",
                     fontsize=9, fontweight="bold")
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlabel("Normalised time")
        ax.set_ylabel("Adoption")
        ax.grid(True, alpha=0.2)

    m_label = METHOD_SPECS.get(method, {}).get("label", method)
    fig.suptitle(f"Cluster mean trajectories — {m_label}  (k={k}){tag}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"trajectories_{method}_k{k}{tag}.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Trajectory plot → {path}")


def plot_umap(
    Z:        np.ndarray,
    labels:   np.ndarray,
    types:    dict,
    k:        int,
    method:   str,
    out_dir:  str,
    tag:      str = "",
    seed:     int = 42,
):
    """2-D scatter (UMAP or PCA) coloured by cluster, annotated by type."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    emb = embed_2d(Z, seed=seed)

    fig, ax = plt.subplots(figsize=(9, 7))
    for cid in sorted(set(labels)):
        mask  = labels == cid
        if cid < 0:
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c="#CCCCCC", s=5, alpha=0.3, rasterized=True)
            continue
        bif   = types.get(cid, "unknown")
        color = CLASS_COLORS.get(bif, CLUSTER_COLORS[cid % len(CLUSTER_COLORS)])
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=color, s=7, alpha=0.45, rasterized=True,
                   label=f"C{cid}: {bif} (n={mask.sum()})")

    # Annotate cluster centroids
    for cid in sorted(set(labels)):
        if cid < 0:
            continue
        mask = labels == cid
        cx, cy = emb[mask, 0].mean(), emb[mask, 1].mean()
        bif = types.get(cid, "?")
        ax.annotate(f"C{cid}\n{bif}", (cx, cy),
                    fontsize=9, ha="center", va="center", fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc=CLASS_COLORS.get(bif, "#607D8B"),
                              alpha=0.85, ec="none"))

    m_label = METHOD_SPECS.get(method, {}).get("label", method)
    emb_type = "UMAP" if _umap_available() else "PCA-2D"
    ax.set_title(f"{emb_type} — {m_label}  (k={k}){tag}", fontsize=11)
    ax.legend(fontsize=8, markerscale=2, loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    path = os.path.join(out_dir, f"umap_{method}_k{k}{tag}.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  UMAP plot → {path}")


def _umap_available() -> bool:
    try:
        import umap  # noqa: F401
        return True
    except ImportError:
        return False


def plot_feature_heatmap(
    cluster_F_means: dict,
    feature_names:   list,
    cluster_types:   dict,
    method:          str,
    k:               int,
    out_dir:         str,
    tag:             str = "",
):
    """
    Heatmap of standardised mean theory features per cluster.
    Rows = clusters (labelled with bifurcation type), columns = features.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    cids   = sorted(cluster_F_means.keys())
    matrix = np.stack([cluster_F_means[c] for c in cids])   # (k, 38)

    # Standardise across clusters for visualisation
    col_std = matrix.std(axis=0) + 1e-10
    matrix  = (matrix - matrix.mean(axis=0)) / col_std

    # Group boundaries for feature groups A-F
    group_boundaries = [8, 15, 21, 28, 33, 38]
    group_labels     = ["A: CSD", "B: Inflection",
                        "C: Phase", "D: Catch22",
                        "E: SN", "F: TC/Null"]

    fig, ax = plt.subplots(figsize=(max(14, len(feature_names) * 0.48),
                                    max(3, len(cids) * 0.65 + 1.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)

    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=60, ha="right", fontsize=6)
    row_labels = [
        f"C{c}: {cluster_types.get(c, '?')}" for c in cids
    ]
    ax.set_yticks(range(len(cids)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Draw group boundary lines
    prev = 0
    for gb, gl in zip(group_boundaries, group_labels):
        if gb < len(feature_names):
            ax.axvline(gb - 0.5, color="black", lw=1.2)
            ax.text((prev + gb - 1) / 2, -1.0, gl,
                    ha="center", va="top", fontsize=6.5, color="#333")
        prev = gb

    plt.colorbar(im, ax=ax, fraction=0.04, label="Normalised feature mean")
    m_label = METHOD_SPECS.get(method, {}).get("label", method)
    ax.set_title(f"Theory feature profile per cluster — {m_label}  k={k}{tag}",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"feature_heatmap_{method}_k{k}{tag}.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Feature heatmap → {path}")


def plot_bifurcation_summary(
    series_types_by_method: dict,   # {method → per-series type array}
    tech_names:             list,
    out_dir:                str,
    k:                      int,
):
    """
    Bar chart: fraction of real-world series assigned to each bifurcation type,
    per method.  Shows which adoption curves are classified as SN / TC / null.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = list(series_types_by_method.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    width   = 0.15
    x       = np.arange(len(CLASS_NAMES))

    for mi, method in enumerate(methods):
        types = series_types_by_method[method]
        fracs = [float((types == c).mean()) for c in CLASS_NAMES]
        bars  = ax.bar(x + mi * width, fracs, width,
                       color=[CLASS_COLORS[c] for c in CLASS_NAMES],
                       alpha=0.75, edgecolor="white",
                       label=METHOD_SPECS.get(method, {}).get("label", method))
        for bar, v in zip(bars, fracs):
            if v > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_ylabel("Fraction of series")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Bifurcation type distribution in real-world dataset (k={k})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out_dir, "bifurcation_distribution.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Bifurcation distribution → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Main runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_real_world(
    real_data_dir:  str  = "data/combined",
    out_dir:        str  = "runs/unsup/real_world",
    synth_dir:      str  = "data/synthetic",
    bundle_path:    str  = "runs/unsup/benchmark/synth_features.pkl",
    n_kernels:      int  = 2000,
    ae_epochs:      int  = 80,
    latent_dim:     int  = 32,
    k_range         = range(2, 9),
    seed:           int  = 42,
    verbose:        bool = True,
) -> dict:
    """
    Full real-world bifurcation clustering pipeline.

    Returns
    -------
    report : dict  — full results (also saved as JSON)
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Load real-world data ──────────────────────────────────────────────────
    with open(os.path.join(real_data_dir, "real_world_samples.pkl"), "rb") as f:
        samples   = pickle.load(f)
    X           = np.load(os.path.join(real_data_dir, "X_full.npy"))
    lengths     = np.load(os.path.join(real_data_dir, "lengths.npy"))
    sample_ids  = [s.row_id    for s in samples]
    tech_names  = [s.tech_name for s in samples]
    N           = len(X)

    print(f"\n{'='*65}")
    print(f"  Real-World Bifurcation Clustering")
    print(f"  N = {N} adoption series")
    print(f"{'='*65}")

    # ── Feature extraction (with full-feature cache) ─────────────────────────
    rw_bundle_path  = os.path.join(out_dir, "real_features.pkl")
    all_feats_cache = os.path.join(out_dir, "real_all_features.pkl")

    if os.path.exists(all_feats_cache):
        print(f"\nLoading cached feature bundle → {all_feats_cache}")
        try:
            with open(all_feats_cache, "rb") as f:
                feats = pickle.load(f)
            print(f"  Loaded: X={feats.X.shape}, Z_ae={feats.Z_ae.shape}, "
                  f"F_white={feats.F_white.shape}, R_white={feats.R_white.shape}")
        except Exception as e:
            print(f"  Cache load failed ({e}) — recomputing...")
            feats = None
    else:
        feats = None

    if feats is None:
        print("\nExtracting features...")
        feats = RealWorldFeatures.build(
            X, lengths,
            bundle_path = rw_bundle_path,
            n_kernels   = n_kernels,
            latent_dim  = latent_dim,
            ae_epochs   = ae_epochs,
            seed        = seed,
            verbose     = verbose,
        )
        with open(all_feats_cache, "wb") as f:
            pickle.dump(feats, f)
        print(f"  Feature bundle saved → {all_feats_cache}")

    # Map method name → feature matrix
    # theory_t uses the same PCA-whitened theory features as theory_gmm
    # so the comparison is apples-to-apples (only the likelihood model differs).
    method_features = {
        # ── commented-out ──────────────────────────────────────────────────
        # "ae_gmm":        feats.Z_ae,
        # "ensemble_gmm":  feats.Z_ens_white,
        # "rocket_gmm":    feats.R_white,
        # "theory_rocket": feats.Z_tr_white,
        # "theory_t_raw":  F_pca_raw,   # archived — degenerate on genuine_v2
        # "theory_bayes":  feats.F_white, # archived — systematic outlier
        # ── active ─────────────────────────────────────────────────────────
        "theory_gmm":    feats.F_white,
        "theory_t":      feats.F_white,
        "theory_skewt":  feats.F_white,
    }

    # ── Synthetic class centroids (for type labelling) ────────────────────────
    # centroids are in synth-standardised space; apply same scaler to real-world
    centroids, synth_mu, synth_std = compute_synthetic_centroids(
        synth_dir, verbose=verbose)
    # real-world features in the same standardised space as the centroids
    F_for_labeling = (feats.F_raw - synth_mu) / (synth_std + 1e-8)

    # ── k-sweep for all methods ───────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  k-sweep  k = {list(k_range)}")

    # theory_t: Student-t mixture with 5 restarts (warm-started from GMM,
    # so fewer restarts are needed; each restart is also slower due to ν update)
    import functools
    _t_fit = functools.partial(fit_t_mixture, n_init=5)

    _tmm_methods   = {"theory_t"}
    _skewt_methods = {"theory_skewt"}

    _st_fit = functools.partial(fit_skew_t_mixture, n_init=5)

    all_sweeps   = {}
    all_best_k   = {}
    for method, Z in method_features.items():
        if method in _tmm_methods:
            fit_fn = _t_fit
        elif method in _skewt_methods:
            fit_fn = _st_fit
        else:
            fit_fn = None
        sweep = sweep_k_for_method(
            Z, method, k_range,
            seed=seed, n_init=30, verbose=verbose,
            fit_fn=fit_fn,
        )
        all_sweeps[method]  = sweep
        all_best_k[method]  = best_k_by_silhouette(sweep)

    # ── Summary of best k ─────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  {'Method':<30}  {'Best k':>6}  {'Best silhouette':>16}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*16}")
    for method, best_k in all_best_k.items():
        sil = all_sweeps[method][best_k]["scores"]["silhouette"]
        label = METHOD_SPECS.get(method, {}).get("label", method)
        print(f"  {label:<30}  {best_k:>6}  "
              f"{sil:16.4f}" if np.isfinite(sil) else
              f"  {label:<30}  {best_k:>6}       nan")

    # ── Plot k-sweep ──────────────────────────────────────────────────────────
    plot_k_sweep_all(all_sweeps, out_dir)

    # ── Cluster at best k and at k=3 (canonical) ─────────────────────────────
    report        = {}
    series_types  = {}     # {method → per-series type array} at k=3

    for method, Z in method_features.items():
        method_report = {}
        label         = METHOD_SPECS.get(method, {}).get("label", method)
        print(f"\n{'─'*65}")
        print(f"  {label}")

        for tag, k in [("_best", all_best_k[method]), ("_k3", 3)]:
            result     = all_sweeps[method][k]
            labels_arr = result["labels"]
            scores     = result["scores"]

            # Type labelling (nearest synthetic centroid in synth-standardised space)
            clu_types, clu_dists, clu_F = label_clusters(
                labels_arr, F_for_labeling, centroids, k
            )

            print(f"\n    k={k}{' [best]' if k == all_best_k[method] else ' [canonical]'}")
            print(f"    Silhouette={scores['silhouette']:.4f}  "
                  f"CH={scores['calinski']:.1f}  DB={scores['davies']:.4f}")
            print(f"    Cluster assignments:")
            for cid in sorted(clu_types):
                cnt = int((labels_arr == cid).sum())
                dist_str = "  ".join(
                    f"{n}={d:.2f}" for n, d in sorted(clu_dists[cid].items()))
                print(f"      C{cid}: {clu_types[cid]:<15} n={cnt:>4}  "
                      f"dist→[{dist_str}]")

            # Visualise
            plot_cluster_trajectories(
                X, labels_arr, clu_types, k, method, out_dir, tag=tag)
            plot_umap(Z, labels_arr, clu_types, k, method, out_dir,
                      tag=tag, seed=seed)
            plot_feature_heatmap(clu_F, feats.feature_names, clu_types,
                                 method, k, out_dir, tag=tag)
            # clu_F contains per-cluster mean features in synth-standardised space

            # Per-series type labels
            type_arr = assign_series_labels(labels_arr, clu_types)
            if tag == "_k3":
                series_types[method] = type_arr

            # Store in report
            method_report[f"k{k}"] = {
                "k":            k,
                "best_k":       all_best_k[method],
                "scores":       {mk: (float(mv) if np.isfinite(mv) else None)
                                 for mk, mv in scores.items()},
                "cluster_types": {int(c): t for c, t in clu_types.items()},
                "cluster_sizes": {int(c): int((labels_arr == c).sum())
                                  for c in range(k)},
                "cluster_distances": {
                    int(c): {n: float(d) for n, d in dct.items()}
                    for c, dct in clu_dists.items()
                },
                "series_assignments": [
                    {"id": sid, "tech": tn, "cluster": int(lbl),
                     "bifurcation_type": tp}
                    for sid, tn, lbl, tp
                    in zip(sample_ids, tech_names,
                           labels_arr.tolist(), type_arr.tolist())
                ],
            }

        report[method] = method_report

    # ── Bifurcation distribution bar chart ────────────────────────────────────
    plot_bifurcation_summary(series_types, tech_names, out_dir, k=3)

    # ── Cross-method agreement at k=3 ────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Cross-method agreement at k=3 (% of series with same type label)")
    type_matrix = np.stack([series_types[m] for m in series_types], axis=0)

    for i, mi in enumerate(series_types):
        for j, mj in enumerate(series_types):
            if j <= i:
                continue
            agree = float((type_matrix[i] == type_matrix[j]).mean())
            li = METHOD_SPECS.get(mi, {}).get("label", mi)
            lj = METHOD_SPECS.get(mj, {}).get("label", mj)
            print(f"    {li:<30} vs {lj:<30}  {agree:.1%}")

    # ── Technology-level breakdown ────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Type distribution at k=3 per method:")
    print(f"  {'Method':<30}  {'SN':>8}  {'TC':>8}  {'Null':>8}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
    for method, type_arr in series_types.items():
        sn   = float((type_arr == "saddle_node").mean())
        tc   = float((type_arr == "transcritical").mean())
        nl   = float((type_arr == "null").mean())
        lab  = METHOD_SPECS.get(method, {}).get("label", method)
        print(f"  {lab:<30}  {sn:8.1%}  {tc:8.1%}  {nl:8.1%}")

    # ── Save JSON report ───────────────────────────────────────────────────────
    class _Enc(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out_path = os.path.join(out_dir, "real_world_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, cls=_Enc)
    print(f"\n  Full report → {out_path}")

    # ── Compact assignment CSV ────────────────────────────────────────────────
    try:
        import csv
        csv_path = os.path.join(out_dir, "series_bifurcation_types.csv")
        with open(csv_path, "w", newline="") as csvf:
            writer = csv.writer(csvf)
            header = ["row_id", "tech_name"] + [
                f"{m}_k3" for m in series_types
            ]
            writer.writerow(header)
            for i, (sid, tn) in enumerate(zip(sample_ids, tech_names)):
                row = [sid, tn] + [series_types[m][i] for m in series_types]
                writer.writerow(row)
        print(f"  CSV assignments → {csv_path}")
    except Exception as e:
        print(f"  CSV write failed: {e}")

    print(f"\n  All outputs saved to: {out_dir}")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-world technology adoption bifurcation clustering"
    )
    parser.add_argument("--real_data_dir",
                        default="data/combined",
                        help="Directory with X_full.npy, real_world_samples.pkl "
                             "(canonical combined dataset lives in data/combined/)")
    parser.add_argument("--synth_dir",
                        default="data/synthetic",
                        help="Directory with validated synthetic dataset (y.npy etc.)")
    parser.add_argument("--bundle_path",
                        default="runs/unsup/benchmark/synth_features.pkl",
                        help="Path to synthetic AE bundle (reused if arch matches)")
    parser.add_argument("--out_dir",
                        default="runs/unsup/real_world")
    parser.add_argument("--n_kernels", type=int, default=2000,
                        help="ROCKET kernels (2000 is fast, 3000 is more stable)")
    parser.add_argument("--ae_epochs", type=int, default=60)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--k_max",     type=int, default=8)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    run_real_world(
        real_data_dir = args.real_data_dir,
        out_dir       = args.out_dir,
        synth_dir     = args.synth_dir,
        bundle_path   = args.bundle_path,
        n_kernels     = args.n_kernels,
        ae_epochs     = args.ae_epochs,
        latent_dim    = args.latent_dim,
        k_range       = range(2, args.k_max + 1),
        seed          = args.seed,
    )
