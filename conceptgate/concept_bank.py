"""The CSG core math (pure numpy).

A concept is represented across `m` tapped layers and `d` model dims by:
  - W   : [m, d]  per-layer diff-of-means *directions* (the "microphones"), unit-norm.
  - f   : [m]     a depth *bandpass filter* that blends the per-layer projections.

Pipeline for a batch of token activations A [N, m, d]:
  spectrogram  S = <A, W>        -> [N, m]   (per-layer "loudness")
  filtered     s = S @ f         -> [N]      (one blended score per token)

All inputs are numpy float arrays. The model boundary (torch) lives in data.py / hooks.py.
"""
from __future__ import annotations

import numpy as np


def _normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def fit_directions(A_pos: np.ndarray, A_neg: np.ndarray) -> np.ndarray:
    """Per-layer diff-of-means direction.

    A_pos, A_neg: [N, m, d] activation samples (every token is a sample).
    Returns W [m, d], unit-norm per layer. This is the LDA direction under equal isotropic
    covariance, and is stable from very few prompts because each prompt yields many tokens.
    """
    mu_pos = A_pos.mean(axis=0)  # [m, d]
    mu_neg = A_neg.mean(axis=0)  # [m, d]
    return _normalize(mu_pos - mu_neg, axis=-1)


def spectrogram(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Project activations onto the per-layer concept directions.

    A: [N, m, d], W: [m, d] -> S: [N, m]   (the concept's loudness profile across depth).
    """
    return np.einsum("nmd,md->nm", A, W)


def dprime_per_layer(S_pos: np.ndarray, S_neg: np.ndarray) -> np.ndarray:
    """Discriminability d' = (mu+ - mu-) / pooled_sd, per layer. S_*: [N, m] -> [m]."""
    mu_p, mu_n = S_pos.mean(0), S_neg.mean(0)
    var = 0.5 * (S_pos.var(0, ddof=1) + S_neg.var(0, ddof=1))
    return (mu_p - mu_n) / np.sqrt(np.maximum(var, 1e-12))


def fit_bandpass(
    S_pos: np.ndarray,
    S_neg: np.ndarray,
    method: str = "fisher",
    ridge: float = 1e-2,
) -> np.ndarray:
    """Learn the depth bandpass filter f [m] (unit-norm), oriented so positives score higher.

    method:
      "best"   -> one-hot on the single most discriminative layer (the SINGLE-LAYER BASELINE).
      "diag"   -> per-layer SNR weighting f_l ∝ (mu+ - mu-)_l / var_l   (assumes layers independent).
      "fisher" -> f ∝ Sigma^{-1} (mu+ - mu-) with pooled within-class covariance Sigma (optimal
                  linear combine; accounts for correlated layer noise). Ridge-regularized.
    """
    mu_p, mu_n = S_pos.mean(0), S_neg.mean(0)
    delta = mu_p - mu_n  # [m]
    m = delta.shape[0]

    if method == "best":
        dp = dprime_per_layer(S_pos, S_neg)
        f = np.zeros(m)
        f[int(np.argmax(np.abs(dp)))] = 1.0
    elif method == "diag":
        var = 0.5 * (S_pos.var(0, ddof=1) + S_neg.var(0, ddof=1))
        f = delta / np.maximum(var, 1e-12)
    elif method == "fisher":
        Sigma = 0.5 * (np.cov(S_pos, rowvar=False) + np.cov(S_neg, rowvar=False))
        Sigma = np.atleast_2d(Sigma)
        # ridge toward identity, scaled by Sigma's magnitude, for small-sample stability
        Sigma = Sigma + ridge * (np.trace(Sigma) / m) * np.eye(m)
        f = np.linalg.solve(Sigma, delta)
    else:
        raise ValueError(f"unknown bandpass method: {method!r}")

    f = _normalize(f, axis=-1)
    if float(f @ delta) < 0:  # orient so the positive (harmful) class scores higher
        f = -f
    return f


def filtered_score(S: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Blend the spectrogram into one score per sample. S: [N, m], f: [m] -> [N]."""
    return S @ f
