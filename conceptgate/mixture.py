"""Pure-numpy Gaussian mixtures for spectrogram-space class densities.

A concept class is a SET of (weight, mu, Sigma) components on the joint m-dim
spectrogram: p(s | class) = sum_j pi_j N(s; mu_j, Sigma_j). One joint mixture per
class (its marginal at each tapped layer is automatically a mixture at that layer).
Fitting is k-means-initialized EM with covariance shrinkage for small samples; the
component count J is chosen by BIC, so scarce data collapses to J=1 — which is the
single-Gaussian gate — by design.

All inputs are numpy float arrays. No scipy / sklearn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

_LOG2PI = float(np.log(2.0 * np.pi))


def _logsumexp(X: np.ndarray, axis: int = -1) -> np.ndarray:
    mx = np.max(X, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(X - mx), axis=axis)) + np.squeeze(mx, axis=axis)


def _shrink(cov: np.ndarray, rho: float, floor: float = 1e-6) -> np.ndarray:
    """Shrink toward scaled identity: (1-rho)*cov + rho*(tr/m)*I, plus a ridge floor."""
    m = cov.shape[0]
    iso = (np.trace(cov) / m) * np.eye(m)
    return (1.0 - rho) * cov + rho * iso + floor * np.eye(m)


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
