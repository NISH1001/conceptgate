"""The calibrated gate: turn a concept's filtered score into a fire / abstain / pass decision.

Each concept fits two 1-D Gaussians on the filtered score (harmful vs benign) and decides with a
log-likelihood ratio (LLR). A threshold `tau` slides the operating point (tune to a target FPR); an
optional abstain band suppresses decisions in the uncertain middle to control false refusals.

A GateBank runs K concepts in parallel and fires if any concept fires.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import concept_bank as cb
from . import mixture as mx

_LOG2PI = float(np.log(2.0 * np.pi))


def _gauss_logpdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-8)
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma) - 0.5 * _LOG2PI


def _fit_directions(A_pos: np.ndarray, A_neg: np.ndarray):
    """Shared few-shot front-end: standardization stats, directions, spectrograms.

    A_pos, A_neg: [N, m, d] -> (mu0, sd0, W, W_raw, S_pos, S_neg).
    """
    A_all = np.concatenate([A_pos, A_neg], axis=0)
    mu0 = A_all.mean(axis=0)                    # [m, d]
    sd0 = A_all.std(axis=0) + 1e-6              # [m, d]
    Zp = (A_pos - mu0) / sd0
    Zn = (A_neg - mu0) / sd0
    W = cb.fit_directions(Zp, Zn)
    W_raw = cb._normalize(A_pos.mean(0) - A_neg.mean(0), axis=-1)
    S_pos = cb.spectrogram(Zp, W)
    S_neg = cb.spectrogram(Zn, W)
    return mu0, sd0, W, W_raw, S_pos, S_neg


@dataclass
class ConceptGate:
    """One concept's full detector: directions + bandpass filter + calibrated Gaussian gate."""

    name: str = "concept"
    filter_method: str = "fisher"
    tau: float = 0.0                 # LLR fire threshold (>tau -> fire)
    abstain_margin: float = 0.0      # if >0, |LLR|<margin -> abstain (no decision)
    # learned params (set by fit)
    W: Optional[np.ndarray] = None        # [m, d] detection directions in STANDARDIZED space
    W_raw: Optional[np.ndarray] = None     # [m, d] diff-of-means in RAW space (used for steering)
    mu0: Optional[np.ndarray] = None       # [m, d] per-dim feature mean (standardization)
    sd0: Optional[np.ndarray] = None       # [m, d] per-dim feature std (standardization)
    f: Optional[np.ndarray] = None
    mu_pos: float = 0.0
    sigma_pos: float = 1.0
    mu_neg: float = 0.0
    sigma_neg: float = 1.0
    train_dprime: Optional[np.ndarray] = None  # per-layer d' on the fit set (for inspection)

    def fit(self, A_pos: np.ndarray, A_neg: np.ndarray) -> "ConceptGate":
        """A_pos, A_neg: [N, m, d] activation samples (use the prompt's last-token rep per prompt).

        Features are standardized per (layer, dim) using pooled fit statistics before computing
        the diff-of-means direction — this tames GPT-2's massive-activation dimensions. A separate
        raw-space direction (W_raw) is kept for reroute steering (which perturbs the raw stream).
        """
        self.mu0, self.sd0, self.W, self.W_raw, S_pos, S_neg = _fit_directions(A_pos, A_neg)
        self.f = cb.fit_bandpass(S_pos, S_neg, method=self.filter_method)
        self.train_dprime = cb.dprime_per_layer(S_pos, S_neg)
        sp = cb.filtered_score(S_pos, self.f)
        sn = cb.filtered_score(S_neg, self.f)
        self.mu_pos, self.sigma_pos = float(sp.mean()), float(sp.std(ddof=1))
        self.mu_neg, self.sigma_neg = float(sn.mean()), float(sn.std(ddof=1))
        return self

    # --- scoring ---
    def score(self, A: np.ndarray) -> np.ndarray:
        """Filtered scalar score per sample. A: [N, m, d] -> [N]."""
        Z = (A - self.mu0) / self.sd0
        return cb.filtered_score(cb.spectrogram(Z, self.W), self.f)

    def llr(self, A: np.ndarray) -> np.ndarray:
        s = self.score(A)
        return _gauss_logpdf(s, self.mu_pos, self.sigma_pos) - _gauss_logpdf(
            s, self.mu_neg, self.sigma_neg
        )

    def fire(self, A: np.ndarray) -> np.ndarray:
        """Boolean fire decision per sample (abstain counts as no-fire)."""
        return self.decide(A) > 0

    def decide(self, A: np.ndarray) -> np.ndarray:
        """+1 fire, 0 abstain, -1 pass."""
        l = self.llr(A)
        out = np.where(l > self.tau, 1, -1)
        if self.abstain_margin > 0:
            out = np.where(np.abs(l - self.tau) < self.abstain_margin, 0, out)
        return out

    # --- calibration ---
    def calibrate_threshold(self, A_neg_cal: np.ndarray, target_fpr: float = 0.05) -> float:
        """Set tau so the false-positive rate on calibration negatives ~= target_fpr."""
        l = np.sort(self.llr(A_neg_cal))
        q = float(np.clip(1.0 - target_fpr, 0.0, 1.0))
        self.tau = float(np.quantile(l, q))
        return self.tau

    def calibrate_z(self, z: float = 3.0) -> float:
        """Operating point: fire only when the filtered score exceeds the benign mean by z*sigma.

        Uses the fitted benign Gaussian directly (smoother than an empirical quantile from a few
        prompts). z=3 -> ~0.1% benign-tail FPR; still catches harmful samples whenever d' > z.
        """
        s_star = np.array([self.mu_neg + z * self.sigma_neg])
        self.tau = float(
            _gauss_logpdf(s_star, self.mu_pos, self.sigma_pos)[0]
            - _gauss_logpdf(s_star, self.mu_neg, self.sigma_neg)[0]
        )
        return self.tau


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class MixtureConceptGate:
    """One concept as a SET of (mu, Sigma) components per class on the joint spectrogram.

    Density upgrade of ConceptGate: class-conditional GMMs fitted directly in
    spectrogram space R^m (J chosen by BIC; J=1 with shared covariance recovers the
    fisher-bandpass gate). Directions, standardization, and steering (W_raw) are
    identical to ConceptGate, so GateBank / Guard work unchanged.
    """

    name: str = "concept"
    Js: Tuple[int, ...] = (1, 2, 3)
    covariance: str = "full"
    shrinkage: float = 0.1
    seed: int = 0
    tau: float = 0.0                 # LLR fire threshold (>tau -> fire)
    abstain_margin: float = 0.0      # if >0, |LLR - tau| < margin -> abstain
    # learned params (set by fit)
    W: Optional[np.ndarray] = None
    W_raw: Optional[np.ndarray] = None
    mu0: Optional[np.ndarray] = None
    sd0: Optional[np.ndarray] = None
    gmm_pos: Optional[mx.GMM] = None
    gmm_neg: Optional[mx.GMM] = None
    train_dprime: Optional[np.ndarray] = None

    def fit(self, A_pos: np.ndarray, A_neg: np.ndarray) -> "MixtureConceptGate":
        """A_pos, A_neg: [N, m, d] activation samples (last-token rep per prompt)."""
        self.mu0, self.sd0, self.W, self.W_raw, S_pos, S_neg = _fit_directions(A_pos, A_neg)
        self.train_dprime = cb.dprime_per_layer(S_pos, S_neg)
        kw = dict(covariance=self.covariance, shrinkage=self.shrinkage, seed=self.seed)
        self.gmm_pos = mx.select_gmm(S_pos, Js=self.Js, **kw)
        self.gmm_neg = mx.select_gmm(S_neg, Js=self.Js, **kw)
        return self

    # --- scoring ---
    def spectro(self, A: np.ndarray) -> np.ndarray:
        """Standardized spectrogram. A: [N, m, d] -> [N, m]."""
        Z = (A - self.mu0) / self.sd0
        return cb.spectrogram(Z, self.W)

    def llr(self, A: np.ndarray) -> np.ndarray:
        S = self.spectro(A)
        return self.gmm_pos.logpdf(S) - self.gmm_neg.logpdf(S)

    def fire(self, A: np.ndarray) -> np.ndarray:
        """Boolean fire decision per sample (abstain counts as no-fire)."""
        return self.decide(A) > 0

    def decide(self, A: np.ndarray) -> np.ndarray:
        """+1 fire, 0 abstain, -1 pass."""
        l = self.llr(A)
        out = np.where(l > self.tau, 1, -1)
        if self.abstain_margin > 0:
            out = np.where(np.abs(l - self.tau) < self.abstain_margin, 0, out)
        return out

    # --- calibration ---
    def calibrate_threshold(self, A_neg_cal: np.ndarray, target_fpr: float = 0.05) -> float:
        """Set tau so the false-positive rate on calibration negatives ~= target_fpr."""
        l = np.sort(self.llr(A_neg_cal))
        q = float(np.clip(1.0 - target_fpr, 0.0, 1.0))
        self.tau = float(np.quantile(l, q))
        return self.tau

    def calibrate_z(self, z: float = 3.0, n_samples: int = 10_000) -> float:
        """Benign-mixture quantile operating point (the mixture analogue of 'mean + z*sd').

        Draw samples from the fitted benign GMM (no model calls), evaluate their LLRs,
        and put tau at the 1 - Phi(-z) quantile: z=3 -> ~0.1% benign-tail FPR.
        """
        S = self.gmm_neg.sample(n_samples, seed=self.seed)
        l = self.gmm_pos.logpdf(S) - self.gmm_neg.logpdf(S)
        self.tau = float(np.quantile(l, 1.0 - _phi(-z)))
        return self.tau


@dataclass
class GateBank:
    """K concepts; fires if ANY concept fires (max-LLR combination)."""

    gates: List[ConceptGate] = field(default_factory=list)

    def add(self, g: ConceptGate) -> "GateBank":
        self.gates.append(g)
        return self

    def llr_max(self, A: np.ndarray) -> np.ndarray:
        if not self.gates:
            return np.full(A.shape[0], -np.inf)
        return np.max(np.stack([g.llr(A) for g in self.gates], axis=0), axis=0)

    def fire(self, A: np.ndarray) -> np.ndarray:
        if not self.gates:
            return np.zeros(A.shape[0], dtype=bool)
        return np.any(np.stack([g.fire(A) for g in self.gates], axis=0), axis=0)

    def which(self, A: np.ndarray) -> np.ndarray:
        """Index of the highest-LLR concept per sample (useful for picking a steering direction)."""
        if not self.gates:
            return np.zeros(A.shape[0], dtype=int)
        return np.argmax(np.stack([g.llr(A) for g in self.gates], axis=0), axis=0)


# ---------- metrics ----------
def recall_fpr(fire_pos: np.ndarray, fire_neg: np.ndarray) -> tuple[float, float]:
    """recall = P(fire | harmful), fpr = P(fire | benign)."""
    return float(np.mean(fire_pos)), float(np.mean(fire_neg))


def error_at_zero(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Balanced misclassification at the LLR=0 / midpoint threshold (for the toy validation)."""
    fnr = float(np.mean(scores_pos <= 0))
    fpr = float(np.mean(scores_neg > 0))
    return 0.5 * (fnr + fpr)
