# Mixture ConceptGate — Design Spec

**Date:** 2026-08-10
**Status:** approved design, pre-implementation
**Scope:** upgrade the per-concept density model from one Gaussian per class to a
mixture (set) of Gaussians per class, modeled directly on the m-dim spectrogram;
validate by redoing the toy problem with three scenarios.

**The approved formulation (user's words, made precise):**
`concept -> CG -> layer a, layer b, ... -> Set((mu, Sigma))` — a concept class is a
**set of (mu, Sigma) components**, not one Gaussian. One **joint** mixture is fit over
the spectrogram `s = (s_a, s_b, ...) in R^m`, NOT a separate mixture per layer glued
together afterwards: the joint mixture's *marginal* at each layer is automatically a
mixture at that layer (so "each layer has some mixture" holds), while cross-layer
correlations are captured — which per-layer-then-combine would lose.

---

## 1. Motivation

The current `ConceptGate` models each class of a concept with a **single** univariate
Gaussian on the depth-blended score `S = f·s`:

- harmful: `N(mu_pos, sigma_pos)`; benign: `N(mu_neg, sigma_neg)` (4 scalars, `gate.py`).

Two problems:

1. **Concept classes are not one blob.** "Benign" is really many modes (chit-chat,
   chemistry homework, code, …). A single Gaussian smears them into an inflated
   `sigma_neg`, which pushes the z-calibrated threshold out (recall loss) and
   miscalibrates the LLR between modes (FPR loss).
2. **The bandpass filter itself assumes unimodality.** `f` (fisher) is fit from pooled
   within-class covariance of the spectrogram. If a class is multimodal, that covariance
   is contaminated by between-mode spread, so the *projection itself* is misestimated.
   Fixing the density after a wrong projection (a GMM on `S`) cannot recover what the
   projection destroyed. In the worst case (benign modes flanking harmful along the
   discriminative axis) **no** single linear filter separates the classes at all.

## 2. What does NOT change (the ConceptGate identity)

- Adapter pattern: base model **M stays frozen**; the sidekick G latches onto m
  residual-stream tap points; removing hooks returns stock M.
- Few-shot fitting: per-layer **diff-of-means directions** `W`, `W_raw` (means only,
  ~10 examples, no backprop). Standardization `mu0/sd0` unchanged.
- The spectrogram `s in R^m` (per-latch-point loudness) unchanged.
- Steering / reroute: inject `-alpha * W_raw` at the latch points, unchanged.
- `GateBank` (K concepts, any-fires union, `which` via max-LLR) unchanged — it is
  duck-typed on `llr/fire/decide`.
- Abort mode, abstain band, FPR-quantile threshold calibration unchanged.

## 3. The change: class-conditional GMMs on the spectrogram

A concept becomes a **set of (weight, mu, Sigma) components per class**, modeled
directly in spectrogram space `R^m` (no separate scalar bandpass step):

```
p(s | class c) = sum_j pi_cj * N(s; mu_cj, Sigma_cj)     j = 1..J_c, c in {+, -}

gate:  LLR(s) = log p(s|+) - log p(s|-)  >  tau          (abstain band as before)
```

Each `mu_cj in R^m` is a **profile across the latch points** — one mode of how the
concept "sounds" across depth. `Sigma_cj` captures how the latch points co-vary
within that mode.

### 3.1 Continuity with the current architecture (nothing is lost)

- With `J=1` per class and a **shared** covariance `Sigma_s`, the LLR is affine in `s`
  with normal vector `Sigma_s^{-1}(mu_+ - mu_-)` — **exactly the fisher bandpass**
  direction. The current CSG is the one-component special case.
- With `J=1` and per-class covariances, the model is QDA in spectrogram space — a mild,
  strictly-more-general variant (quadratic boundary). On near-homoscedastic data it
  matches fisher closely (regression scenario tolerance: within ~1 point of error).
- With `J>1`, the effective filter becomes **input-dependent**: the LLR's local
  decision direction is a responsibility-weighted blend of per-component matched
  filters. "The bandpass filter accounts for multiple distributions itself."

### 3.2 Fitting

- **EM** with k-means init on the fit-set spectrograms, per class.
- **Small-sample safeguards:**
  - covariance shrinkage: `Sigma_j <- (1-rho)*Sigma_j + rho*(tr(Sigma_j)/m)*I`
    (default `rho=0.1`), plus the existing ridge floor;
  - optional `covariance="diag"` mode for the strictest few-shot regimes;
  - deterministic seeding (k-means init seeded; report variance across seeds in evals).
- **Model selection:** `J_c` chosen by **BIC over J in {1, 2, 3}** per class.
  With ~10 last-token samples per class, BIC collapses to `J=1` (today's model) —
  intended behavior, not a failure. Per-token sampling (hundreds of samples from the
  same ~10 prompts) can support real multi-mode fits.

### 3.3 Calibration

- `calibrate_threshold(A_neg_cal, target_fpr)` — **unchanged** (quantile of LLR on
  calibration negatives; valid for any LLR).
- `calibrate_z(z)` — the benign class no longer has a single (mu, sigma), so "z sigma
  above the benign mean" is replaced by the equivalent **benign-mixture quantile**:
  draw ~10k samples from the fitted benign GMM (no model calls), evaluate their LLRs,
  and set `tau` to the `1 - Phi(-z)` empirical quantile (z=3 -> ~99.87th percentile,
  the same ~0.1% benign-tail FPR as before).

## 4. Code plan

New module `conceptgate/mixture.py`. Division of labor (decided during implementation,
2026-08-10): **fitting delegates to sklearn's `GaussianMixture`** (reference EM —
hand-rolled EM risks silently-wrong fits corrupting research conclusions; the repo
already carries far heavier deps), while **storage + evaluation stay in a tiny numpy
`GMM` dataclass** so the rest of conceptgate is sklearn-free; a unit test pins our
`logpdf` to sklearn's `score_samples`. Spec §3.2's shrinkage maps to sklearn's
`reg_covar` scaled by the data's mean per-dim variance:

- `GMM` dataclass: `weights [J]`, `means [J, m]`, `covs [J, m, m]`;
  `logpdf(X) -> [N]` via logsumexp; `sample(n, seed)`.
- `fit_gmm(X, J, covariance="full"|"diag", shrinkage, seed) -> GMM` (k-means init + EM).
- `select_gmm(X, Js=(1,2,3), ...) -> GMM` (BIC selection).

`conceptgate/gate.py`:

- New dataclass `MixtureConceptGate` alongside `ConceptGate` (the validated baseline
  class stays untouched as the nested comparator):
  - `fit(A_pos, A_neg)`: standardize -> `W`, `W_raw` (reuse existing code paths) ->
    spectrograms -> `gmm_pos`, `gmm_neg` via `select_gmm`.
  - no `score(A)` (there is no single blended scalar anymore); `llr(A)` is the
    primary output, and `fire/decide/calibrate_threshold` keep the same signatures
    and semantics as `ConceptGate`.
  - `calibrate_z(z)` per §3.3.
- `GateBank` accepts both gate types (duck typing; no change).

`scripts/toy_csg_mixture.py` — the redone toy (see §5).

Docs: after validation, `math.md` gains the mixture formulation (§5b: set-of-(mu,Sigma),
EM/BIC, continuity result) and `concepts.md` updates the mental model ("each mode is
its own microphone profile"); honest-novelty table gains classic GDA/QDA + Mahalanobis
OOD citations. README status updated.

## 5. Toy validation (the deliverable of this round)

`scripts/toy_csg_mixture.py`, three scenarios, seeded, PASS/FAIL printed like
`toy_csg.py`. Methods compared in each: `best` (single layer), `fisher` (current CSG),
`mixture` (new, BIC-selected). Metrics: balanced error at the natural threshold, AUC,
selected `J` per class. Ground-truth Bayes error computed numerically from each
generator (the true densities are known) as the reference floor.

1. **Regression** — the existing unimodal generator from `toy_csg.py`
   (`d' = [1.6, 2.0, 0.6]`). Expect: BIC selects `J=1`; mixture error within
   **1 point** of fisher (~9.4%); fisher beats best (~16%) as before.
   *Proves the upgrade loses nothing.*
2. **Bimodal benign** — benign = two clusters in spectrogram space, one sitting near
   the harmful region on a subset of layers; harmful unimodal. Expect: fisher degrades
   (inflated pooled covariance); mixture error <= fisher error and within **2 points**
   of the Bayes floor.
3. **Kill shot (no linear filter works)** — benign modes flanking harmful along the
   discriminative axis (XOR-like across depth): construct so the fisher/best error is
   >= **35%** (near chance) while the Bayes floor is low. Expect: mixture within
   **3 points** of Bayes. *One plot justifies the redesign.*

Each scenario also reports results across >= 5 seeds (mean ± std) to keep the few-shot
variance story honest.

## 6. Out of scope (parked, in order)

1. **Signal-processor restructure / early short-circuit** (the agreed *next* step
   after this round): per-tap sequential gating so CG can abort the base model's
   forward pass partway (only layers up to the firing tap are computed). CG becomes a
   bandpass-filter-like processor that sits on the base model — intercepts, emits
   scores, short-circuits, steers. Requires per-stage prefix densities; the mixture
   machinery built here plugs in per stage.
2. **Facet directions** (per-component diff-of-means `w` — mixtures in direction
   space; per-facet steering).
3. **Head-vs-CSG identity experiment** on GPT-2 (classification head baselines,
   n-sweep, calibration/OOD/steering columns).
4. **P1 instruct model** (Gemma-2-2B-it) — unchanged on the roadmap.
5. Mixtures in raw `R^d` activation space (data-hungry; subsumed by facets).

## 7. Success criteria

- All three toy scenarios PASS their stated tolerances.
- `MixtureConceptGate` drop-in works with `GateBank` and existing calibration API.
- `ConceptGate` (single-Gaussian) results unchanged (it is not modified).
- GPT-2 smoke (`p0_smoke.py`) still passes with `ConceptGate`; optional: a smoke run
  with `MixtureConceptGate` showing BIC->J=1 collapse on 10-shot last-token data.

## 8. Risks & honest caveats

- **EM in R^m with tiny n**: mitigated by shrinkage, diag option, BIC guard; expected
  (and acceptable) outcome on strict 10-shot last-token data is `J=1`.
- **On unimodal real data the mixture changes nothing** — that is a valid finding and
  keeps the single-Gaussian gate as the honest default.
- **Toy scenarios are constructed** — they demonstrate capability, not real-world
  prevalence of multimodality; the real-data question moves to the identity experiment
  and P1.
