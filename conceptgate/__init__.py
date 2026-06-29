"""Concept Spectrogram Gate (CSG): a lightweight, few-shot internal guardrail adapter.

Core idea: tap a frozen model M's residual stream at several layers, build a per-concept
"spectrogram" of diff-of-means projections across depth, blend it with a tiny learned bandpass
filter, and gate on a calibrated 1-D distribution. Firing either ABORTS generation (emit a fixed
token) or REROUTES the representation (add a steering vector) so M refuses on its own.

This package is pure-numpy for the math (concept_bank, gate) and torch/transformers only at the
model boundary (hooks, data, guard).
"""

from . import concept_bank, gate  # noqa: F401

__all__ = ["concept_bank", "gate"]
