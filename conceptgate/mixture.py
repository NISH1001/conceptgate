"""Gaussian mixtures for spectrogram-space class densities.

A concept class is a SET of (weight, mu, Sigma) components on the joint m-dim
spectrogram: p(s | class) = sum_j pi_j N(s; mu_j, Sigma_j). One joint mixture per
class (its marginal at each tapped layer is automatically a mixture at that layer).

Division of labor: FITTING is delegated to sklearn's reference EM (seeded restarts,
hardened numerics — EM bugs are silent, and the science must not depend on a
hand-rolled M-step). STORAGE + EVALUATION live in the tiny numpy `GMM` dataclass
below (logpdf / sample / n_params), so the rest of conceptgate stays sklearn-free;
a test asserts our logpdf matches sklearn's score_samples to machine precision.
The component count J is chosen by BIC, so scarce data collapses to J=1 — which is
the single-Gaussian gate — by design.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
from sklearn.mixture import GaussianMixture

_LOG2PI = float(np.log(2.0 * np.pi))


def _logsumexp(X: np.ndarray, axis: int = -1) -> np.ndarray:
    mx = np.max(X, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(X - mx), axis=axis)) + np.squeeze(mx, axis=axis)


@dataclass
class GMM:
    """weights [J], means [J, m], covs [J, m, m]."""

    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    covariance: str = "full"   # "full" | "diag" — affects n_params (BIC) only

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[0])

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    def component_logpdf(self, X: np.ndarray) -> np.ndarray:
        """Per-component Gaussian log-density. X: [N, m] -> [N, J]."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        N, m = X.shape
        out = np.empty((N, self.n_components))
        for j in range(self.n_components):
            L = np.linalg.cholesky(self.covs[j])
            y = np.linalg.solve(L, (X - self.means[j]).T)       # [m, N] whitened
            maha = np.sum(y * y, axis=0)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            out[:, j] = -0.5 * (maha + logdet + m * _LOG2PI)
        return out

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        """Mixture log-density. X: [N, m] -> [N]."""
        lp = self.component_logpdf(X) + np.log(self.weights)[None, :]
        return _logsumexp(lp, axis=1)

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(n, self.weights / self.weights.sum())
        parts = [
            rng.multivariate_normal(self.means[j], self.covs[j], size=int(c))
            for j, c in enumerate(counts)
            if c > 0
        ]
        X = np.concatenate(parts, axis=0)
        return X[rng.permutation(n)]

    def n_params(self) -> int:
        J, m = self.n_components, self.dim
        cov_p = m if self.covariance == "diag" else m * (m + 1) // 2
        return (J - 1) + J * m + J * cov_p


def fit_gmm(
    X: np.ndarray,
    J: int,
    covariance: str = "full",
    shrinkage: float = 0.1,
    seed: int = 0,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_init: int = 3,
) -> GMM:
    """Fit a J-component GMM on X [N, m] via sklearn's EM; return our numpy GMM.

    `shrinkage` maps to sklearn's reg_covar scaled by the data's mean per-dim
    variance — a ridge toward (data-scaled) identity, the small-sample safeguard.
    Deterministic given `seed` (n_init seeded restarts happen inside sklearn).
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    reg = shrinkage * float(np.mean(X.var(axis=0))) + 1e-6
    sk = GaussianMixture(
        n_components=J,
        covariance_type=covariance,
        reg_covar=reg,
        random_state=seed,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
    ).fit(X)
    if covariance == "diag":
        covs = np.stack([np.diag(c) for c in sk.covariances_])   # [J, m] -> [J, m, m]
    else:
        covs = sk.covariances_
    return GMM(sk.weights_, sk.means_, covs, covariance)


def bic(gmm: GMM, X: np.ndarray) -> float:
    """Bayesian information criterion (lower is better)."""
    N = X.shape[0]
    return -2.0 * float(gmm.logpdf(X).sum()) + gmm.n_params() * float(np.log(N))


def select_gmm(X: np.ndarray, Js: Iterable[int] = (1, 2, 3), **fit_kwargs) -> GMM:
    """Fit each J and return the BIC-best mixture (scarce data -> J=1 by design)."""
    fits = [fit_gmm(X, J=int(J), **fit_kwargs) for J in Js]
    scores = [bic(g, X) for g in fits]
    return fits[int(np.argmin(scores))]
