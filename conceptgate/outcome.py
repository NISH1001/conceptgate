"""OutcomeHead: a per-prompt regression read fit to a MEASURED intervention outcome -- e.g. how far
a steering write moved refusal on that prompt -- rather than to concept membership.

Same tap activations the concepts see; a ridge direction instead of a class boundary. Labels are
needed at fit time only (docs/plans/steerability-gate.md, stage 1 produces them); at inference the
prediction is one dot product on activations the gate already has. The concept still says WHAT to
write and along which direction; this says WHETHER the write is worth making on this prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OutcomeHead:
    alpha: float = 10.0
    mu: np.ndarray | None = field(default=None, repr=False)
    sd: np.ndarray | None = field(default=None, repr=False)
    w: np.ndarray | None = field(default=None, repr=False)
    b: float = 0.0
    shape: tuple[int, int] | None = None

    def fit(self, A: np.ndarray, y: np.ndarray) -> "OutcomeHead":
        """A: [n, m, d] last-token tap activations; y: [n] measured outcome per prompt."""
        from sklearn.linear_model import Ridge

        A = np.asarray(A, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if A.ndim != 3:
            raise ValueError(f"A must be [n, m, d] tap activations; got shape {A.shape}")
        if y.shape != (A.shape[0],):
            raise ValueError(f"y must have one label per row of A; got {y.shape} for n={A.shape[0]}")
        n, m, d = A.shape
        X = A.reshape(n, m * d)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        r = Ridge(alpha=self.alpha).fit((X - self.mu) / self.sd, y)
        self.w, self.b, self.shape = np.asarray(r.coef_), float(r.intercept_), (m, d)
        return self

    def predict(self, A: np.ndarray) -> np.ndarray:
        """A: [n, m, d] -> [n] predicted outcomes."""
        if self.w is None:
            raise RuntimeError("OutcomeHead.predict called before fit")
        A = np.asarray(A, dtype=np.float64)
        X = A.reshape(A.shape[0], -1)
        return ((X - self.mu) / self.sd) @ self.w + self.b

    def direction(self) -> np.ndarray:
        """The fitted direction in raw activation space, [m, d] (coef / sd), for cosines to W_raw."""
        if self.w is None:
            raise RuntimeError("OutcomeHead.direction called before fit")
        return (self.w / self.sd).reshape(self.shape)
