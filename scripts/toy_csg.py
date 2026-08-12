"""Offline validation of the CSG core math on synthetic activations (no model needed).

We synthesize per-layer activations whose per-layer discriminability is d' = [1.6, 2.0, 0.6].
Theory says:
  - the best SINGLE layer gives d' = 2.0   -> balanced error Phi(-1.0)   ~= 15.9%
  - blending across depth (independent noise) adds in quadrature:
      d' = sqrt(1.6^2 + 2.0^2 + 0.6^2) = 2.63  -> balanced error Phi(-1.315) ~= 9.4%

This exercises fit_directions -> spectrogram -> fit_bandpass -> calibrated gate end-to-end and
checks that the depth bandpass filter empirically beats the single best layer.

Run:  uv run python scripts/toy_csg.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conceptgate import spectral as spec
from conceptgate.concept import BandpassConcept, error_at_zero


def phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def synth(rng, n, sign, U, dprime):
    """[n, m, d] activations; class mean +/- (dprime/2)*U_l along a unit dir U_l, isotropic noise=1.

    Noise is drawn independently per (sample, layer), so the per-layer noise is independent across
    depth -> the quadrature-sum prediction applies.
    """
    m, d = U.shape
    A = rng.standard_normal((n, m, d))  # isotropic noise, sigma=1 (var=1 along any unit dir)
    for layer in range(m):
        A[:, layer, :] += sign * (dprime[layer] / 2.0) * U[layer]
    return A


def main() -> int:
    rng = np.random.default_rng(0)
    m, d, n = 3, 64, 6000
    dprime_target = np.array([1.6, 2.0, 0.6])
    U = spec._normalize(rng.standard_normal((m, d)), axis=-1)  # one signal direction per layer

    A_pos = synth(rng, n, +1, U, dprime_target)
    A_neg = synth(rng, n, -1, U, dprime_target)

    # train / test split
    tr = slice(0, n // 2)
    te = slice(n // 2, n)

    print(f"target per-layer d': {dprime_target.tolist()}")
    print(f"theory: single-best d'=2.0 -> err={phi(-2.0/2)*100:.1f}%   "
          f"CSG d'={math.sqrt((dprime_target**2).sum()):.2f} -> err={phi(-math.sqrt((dprime_target**2).sum())/2)*100:.1f}%")
    print("-" * 64)
    print(f"{'filter':>8} | {'recovered per-layer d-prime':>30} | {'test err':>8}")
    print("-" * 64)

    results = {}
    for method in ["best", "diag", "fisher"]:
        g = BandpassConcept(name="toy", filter_method=method).fit(A_pos[tr], A_neg[tr])
        # balanced error at the calibrated midpoint between the two filtered-score Gaussians
        sp = g.score(A_pos[te]) - 0.5 * (g.mu_pos + g.mu_neg)
        sn = g.score(A_neg[te]) - 0.5 * (g.mu_pos + g.mu_neg)
        err = error_at_zero(sp, sn)
        results[method] = err
        dp = np.array2string(g.train_dprime, precision=2, floatmode="fixed")
        print(f"{method:>8} | {dp:>30} | {err*100:>7.1f}%")

    print("-" * 64)
    improved = results["fisher"] < results["best"] and results["diag"] < results["best"]
    print(f"depth filter beats single best layer: {improved}  "
          f"(best={results['best']*100:.1f}%  diag={results['diag']*100:.1f}%  fisher={results['fisher']*100:.1f}%)")
    # sanity bounds: single-best near 16%, fisher near 9-10%
    ok = (0.13 < results["best"] < 0.19) and (results["fisher"] < results["best"] - 0.03)
    print(f"VALIDATION {'PASS' if ok and improved else 'FAIL'}")
    return 0 if (ok and improved) else 1


if __name__ == "__main__":
    raise SystemExit(main())
