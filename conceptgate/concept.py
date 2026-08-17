"""One learned concept: measure a spectrogram, produce an LLR, decide fire / pass.

A Concept MEASURES (score/llr); firing is llr > tau, where tau is a calibrated operating
point. Two variants:

  - Concept (canonical): each class is a SET of (mu, Sigma) components — class-conditional
    GMMs on the joint spectrogram, J per class chosen by BIC (J=1 on scarce data).
  - BandpassConcept (nested baseline): scalar bandpass-filtered score + two 1-D Gaussians;
    carries the best / diag / fisher filter variants the depth-fusion experiments compare.

A ConceptBank runs K concepts in parallel and fires if any concept fires. The public
facade that attaches these to a frozen model and acts on firings is ConceptGate (gate.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import mixture as mx
from . import spectral as spec

_LOG2PI = float(np.log(2.0 * np.pi))


def _gauss_logpdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-8)
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma) - 0.5 * _LOG2PI


def _fit_directions(A_pos: np.ndarray, A_neg: np.ndarray, direction: str = "diff"):
    """Shared few-shot front-end: standardization stats, directions, spectrograms.

    direction "diff" -> diff-of-means (isotropic); "logistic" -> per-layer discriminative
    (covariance-aware) detection direction. W_raw stays diff-of-means -- the steering vector,
    decoupled from the detection direction. A_pos, A_neg: [N, m, d] -> (mu0, sd0, W, W_raw,
    S_pos, S_neg).
    """
    A_all = np.concatenate([A_pos, A_neg], axis=0)
    mu0 = A_all.mean(axis=0)  # [m, d]
    sd0 = A_all.std(axis=0) + 1e-6  # [m, d]
    Zp = (A_pos - mu0) / sd0
    Zn = (A_neg - mu0) / sd0
    if direction == "logistic":
        W = spec.fit_directions_logistic(Zp, Zn)
    else:
        W = spec.fit_directions(Zp, Zn)
    W_raw = spec._normalize(A_pos.mean(0) - A_neg.mean(0), axis=-1)  # steering: diff-of-means
    S_pos = spec.spectrogram(Zp, W)
    S_neg = spec.spectrogram(Zn, W)
    return mu0, sd0, W, W_raw, S_pos, S_neg


class _GateCommon:
    """Shared decision/calibration machinery. Subclasses provide llr(), tau."""

    # runtime-set default (overwritten by calibrate); class default keeps it off the dataclass
    score_scale: float = 1.0    # robust spread of benign LLR near tau -> confidence scale

    def fire(self, A: np.ndarray) -> np.ndarray:
        """Boolean fire decision per sample (abstain counts as no-fire)."""
        return self.decide(A) > 0

    def decide(self, A: np.ndarray) -> np.ndarray:
        """+1 fire, 0 abstain, -1 pass. abstain_margin is a P(present) half-width at 0.5."""
        out = np.where(self.llr(A) > self.tau, 1, -1)
        if self.abstain_margin > 0:
            out = np.where(np.abs(self.p_present(A) - 0.5) < self.abstain_margin, 0, out)
        return out

    def p_present(self, A: np.ndarray) -> np.ndarray:
        """P(concept present): logistic of (LLR - tau) scaled by the benign spread."""
        z = (self.llr(A) - self.tau) / max(self.score_scale, 1e-8)
        return 1.0 / (1.0 + np.exp(-z))

    def calibrate_threshold(
        self, A_neg_cal: np.ndarray, target_fpr: float = 0.05
    ) -> float:
        """Set tau so the false-positive rate on calibration negatives ~= target_fpr."""
        scores = np.sort(self.llr(A_neg_cal))
        q = float(np.clip(1.0 - target_fpr, 0.0, 1.0))
        self.tau = float(np.quantile(scores, q))
        return self.tau


@dataclass
class BandpassConcept(_GateCommon):
    """Scalar-filter detector: directions + bandpass filter + two 1-D Gaussians.

    The nested baseline family (filter_method: best / diag / fisher) that the
    depth-fusion experiments compare against. The canonical gate is ConceptGate
    (class-conditional mixtures); this class is its exact scalar special case."""

    name: str = "concept"
    filter_method: str = "fisher"
    tau: float = 0.0  # LLR fire threshold (>tau -> fire)
    abstain_margin: float = 0.0  # if >0, |LLR|<margin -> abstain (no decision)
    # learned params (set by fit)
    W: np.ndarray | None = None  # [m, d] detection directions in STANDARDIZED space
    W_raw: np.ndarray | None = (
        None  # [m, d] diff-of-means in RAW space (used for steering)
    )
    mu0: np.ndarray | None = None  # [m, d] per-dim feature mean (standardization)
    sd0: np.ndarray | None = None  # [m, d] per-dim feature std (standardization)
    f: np.ndarray | None = None
    mu_pos: float = 0.0
    sigma_pos: float = 1.0
    mu_neg: float = 0.0
    sigma_neg: float = 1.0
    train_dprime: np.ndarray | None = (
        None  # per-layer d' on the fit set (for inspection)
    )

    def fit(self, A_pos: np.ndarray, A_neg: np.ndarray) -> "BandpassConcept":
        """A_pos, A_neg: [N, m, d] activation samples (use the prompt's last-token rep per prompt).

        Features are standardized per (layer, dim) using pooled fit statistics before computing
        the diff-of-means direction — this tames GPT-2's massive-activation dimensions. A separate
        raw-space direction (W_raw) is kept for reroute steering (which perturbs the raw stream).
        """
        self.mu0, self.sd0, self.W, self.W_raw, S_pos, S_neg = _fit_directions(
            A_pos, A_neg
        )
        self.f = spec.fit_bandpass(S_pos, S_neg, method=self.filter_method)
        self.train_dprime = spec.dprime_per_layer(S_pos, S_neg)
        sp = spec.filtered_score(S_pos, self.f)
        sn = spec.filtered_score(S_neg, self.f)
        self.mu_pos, self.sigma_pos = float(sp.mean()), float(sp.std(ddof=1))
        self.mu_neg, self.sigma_neg = float(sn.mean()), float(sn.std(ddof=1))
        return self

    # --- scoring ---
    def score(self, A: np.ndarray) -> np.ndarray:
        """Filtered scalar score per sample. A: [N, m, d] -> [N]."""
        Z = (A - self.mu0) / self.sd0
        return spec.filtered_score(spec.spectrogram(Z, self.W), self.f)

    def llr(self, A: np.ndarray) -> np.ndarray:
        s = self.score(A)
        return _gauss_logpdf(s, self.mu_pos, self.sigma_pos) - _gauss_logpdf(
            s, self.mu_neg, self.sigma_neg
        )

    # --- calibration ---
    def calibrate_z(self, z: float = 3.0, margin: float = 0.0) -> float:
        """Operating point: fire only when the filtered score exceeds the benign mean by z*sigma.

        Uses the fitted benign Gaussian directly (smoother than an empirical quantile from a few
        prompts). z=3 -> ~0.1% benign-tail FPR; still catches harmful samples whenever d' > z.
        `margin` is a P(present) half-width around 0.5 for the abstain band; 0 disables it.
        """
        s_star = np.array([self.mu_neg + z * self.sigma_neg])
        self.tau = float(
            _gauss_logpdf(s_star, self.mu_pos, self.sigma_pos)[0]
            - _gauss_logpdf(s_star, self.mu_neg, self.sigma_neg)[0]
        )
        sb = np.random.default_rng(0).normal(self.mu_neg, self.sigma_neg, 4000)
        sc = self.llr_from_score(sb)
        lo = float(np.quantile(sc, _phi(max(z - 1.0, 0.0))))  # local spread near tau
        self.score_scale = max(self.tau - lo, 1e-6)
        self.abstain_margin = float(margin)  # P(present) half-width around 0.5
        return self.tau

    def llr_from_score(self, s: np.ndarray) -> np.ndarray:
        return _gauss_logpdf(s, self.mu_pos, self.sigma_pos) - _gauss_logpdf(
            s, self.mu_neg, self.sigma_neg
        )


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class Concept(_GateCommon):
    """The canonical learned unit: a concept class is a SET of (mu, Sigma) components on
    the joint spectrogram.

    Class-conditional GMMs fitted directly in spectrogram space R^m (J per class
    chosen by BIC; J=1 with shared covariance recovers the fisher bandpass, so
    BandpassConcept is the nested special case kept as the experimental
    baseline). Directions, standardization, and steering (W_raw) are shared."""

    name: str = "concept"
    Js: tuple[int, ...] = (1, 2, 3)
    covariance: str = "full"
    shrinkage: float = 0.1
    seed: int = 0
    direction: str = "diff"  # detection direction: "diff" (diff-of-means) | "logistic"
    tau: float = 0.0  # LLR fire threshold (>tau -> fire)
    abstain_margin: float = 0.0  # if >0, |LLR - tau| < margin -> abstain
    # learned params (set by fit)
    W: np.ndarray | None = None
    W_raw: np.ndarray | None = None
    mu0: np.ndarray | None = None
    sd0: np.ndarray | None = None
    gmm_pos: mx.GMM | None = None
    gmm_neg: mx.GMM | None = None
    train_dprime: np.ndarray | None = None

    def fit(self, A_pos: np.ndarray, A_neg: np.ndarray) -> "Concept":
        """A_pos, A_neg: [N, m, d] activation samples (last-token rep per prompt)."""
        self.mu0, self.sd0, self.W, self.W_raw, S_pos, S_neg = _fit_directions(
            A_pos, A_neg, direction=self.direction
        )
        self.train_dprime = spec.dprime_per_layer(S_pos, S_neg)
        kw = dict(covariance=self.covariance, shrinkage=self.shrinkage, seed=self.seed)
        self.gmm_pos = mx.select_gmm(S_pos, Js=self.Js, **kw)
        self.gmm_neg = mx.select_gmm(S_neg, Js=self.Js, **kw)
        return self

    # --- scoring ---
    def spectro(self, A: np.ndarray) -> np.ndarray:
        """Standardized spectrogram. A: [N, m, d] -> [N, m]."""
        Z = (A - self.mu0) / self.sd0
        return spec.spectrogram(Z, self.W)

    def llr(self, A: np.ndarray) -> np.ndarray:
        S = self.spectro(A)
        return self.gmm_pos.logpdf(S) - self.gmm_neg.logpdf(S)

    # --- calibration ---
    def calibrate_z(
        self, z: float = 3.0, n_samples: int = 10_000, margin: float = 0.0
    ) -> float:
        """Set tau to a false-alarm target, expressed as 'z sigma above benign'.

        The benign class is modeled as a Gaussian mixture (gmm_neg). We draw n_samples
        from that fitted benign density, score their LLRs, and put tau at the
        (1 - Phi(-z)) quantile -- i.e. only that fraction of benign inputs would fire.
        z=3 -> ~0.1% benign false-alarm rate. (This replaces the plain 'mean + z*std'
        rule, which assumes a single Gaussian; here the benign side may be multi-modal.)

        The same benign scores set score_scale (local spread near tau, for p_present);
        margin sets the abstain band |p_present - 0.5| < margin (0 disables it).
        """
        S = self.gmm_neg.sample(n_samples, seed=self.seed)
        scores = self.gmm_pos.logpdf(S) - self.gmm_neg.logpdf(S)
        self.tau = float(np.quantile(scores, 1.0 - _phi(-z)))
        # confidence scale = benign spread LOCAL to tau (one z-step toward the bulk), not
        # the full benign IQR -- the heavy LLR tails would otherwise dwarf the real margin
        lo = float(np.quantile(scores, _phi(max(z - 1.0, 0.0))))
        self.score_scale = max(self.tau - lo, 1e-6)
        self.abstain_margin = float(margin)  # P(present) half-width around 0.5
        return self.tau


@dataclass
class ConceptBank:
    """K concepts; fires if ANY concept fires (max-LLR combination)."""

    gates: list[_GateCommon] = field(default_factory=list)

    def add(self, g: _GateCommon) -> ConceptBank:
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
