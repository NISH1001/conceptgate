# conceptgate — The Mathematics

> The rigorous companion to [`concepts.md`](./concepts.md). Every step here maps to code in
> `conceptgate/spectral.py` and `conceptgate/concept.py`; the code map is in §11.
>
> GitHub renders the `$…$` / `$$…$$` LaTeX below. In a plain editor the source is still readable.

---

## 1. Setup & notation

- **M**: a frozen causal LM with residual-stream width $d$. We tap a set of **block layers**
  $\mathcal{L}=\{\ell_1,\dots,\ell_m\}$ (0-based; block $\ell$'s output residual stream is
  `hidden_states[ℓ+1]`).
- For one token, its activation across the tapped layers is $a \in \mathbb{R}^{m\times d}$, with rows
  $a_\ell \in \mathbb{R}^d$.
- A **concept** has two classes: positive (concept present, "$+$") and negative ("$-$").
- Fitting uses one representation per prompt — the **last token's** activation (it has attended to the
  whole prompt, so it summarizes intent; per-token fitting dilutes the signal with shared boilerplate
  tokens — see §10). Let $\mathcal{A}^+=\{a^{(i)}\}$, $\mathcal{A}^-=\{a^{(j)}\}$ be the fit sets.

We write $\bar{x}$ for a sample mean, $\mathrm{Var}(\cdot)$ for sample variance (ddof=1),
$\lVert\cdot\rVert$ for the Euclidean norm, $\odot$ / division for elementwise ops.

---

## 2. Standardization (tame GPT-2's massive activations)

Residual streams have a few outlier dimensions of huge magnitude that dominate raw dot products. We
standardize per $(\ell,\text{dim})$ using **pooled fit statistics**
$\mathcal{A}=\mathcal{A}^+\cup\mathcal{A}^-$:

$$
\mu_0=\operatorname{mean}_{a\in\mathcal{A}}(a)\in\mathbb{R}^{m\times d},\qquad
\sigma_0=\operatorname{std}_{a\in\mathcal{A}}(a)+\epsilon\in\mathbb{R}^{m\times d},
$$

$$
z = (a-\mu_0)\oslash\sigma_0 \quad(\epsilon=10^{-6}).
$$

All detection math operates on standardized $z$. (Steering, §9, uses **raw** space.)

---

## 3. Diff-of-means direction (per layer)

For each layer $\ell$, the concept's **signature** is the unit vector along the difference of class
means in standardized space:

$$
\boxed{\,w_\ell=\frac{\bar z^{+}_\ell-\bar z^{-}_\ell}{\lVert \bar z^{+}_\ell-\bar z^{-}_\ell\rVert}\,}\in\mathbb{R}^d .
$$

**Why this is principled (not a hack).** Model each class as Gaussian with shared covariance
$\Sigma$. The Bayes-optimal (LDA) decision direction is $\Sigma^{-1}(\mu^+-\mu^-)$. Under
(approximately) isotropic within-class covariance $\Sigma\propto I$ — which standardization pushes
toward — this collapses to $\mu^+-\mu^-$, i.e. **diff-of-means is the optimal linear direction**.

**Why it's few-shot stable.** $w_\ell$ depends only on two *mean* vectors. With ~10 prompts and the
per-token sampling available, the means are well-estimated even though individual activations are
noisy (it is estimating a first moment, not fitting $d$ free parameters).

---

## 4. The concept spectrogram

Project a (standardized) sample onto each layer's signature to get a **loudness per layer**:

$$
s_\ell = w_\ell\cdot z_\ell,\qquad
\mathbf{s}=(s_1,\dots,s_m)\in\mathbb{R}^m .
$$

$\mathbf{s}$ is the concept's profile across depth — its "spectrogram". Per-layer discriminability:

$$
d'_\ell=\frac{\bar s^{+}_\ell-\bar s^{-}_\ell}{\sqrt{\tfrac12\!\left(\mathrm{Var}(s^{+}_\ell)+\mathrm{Var}(s^{-}_\ell)\right)}} .
$$

---

## 5. The depth bandpass filter

We blend the spectrogram into a single score with a filter $f\in\mathbb{R}^m$:

$$
\boxed{\,S=f\cdot\mathbf{s}=\sum_{\ell=1}^{m} f_\ell\, s_\ell\,}.
$$

Three ways to choose $f$ (then normalize to unit norm and orient so $f\cdot(\bar{\mathbf s}^+-\bar{\mathbf s}^-)>0$):

| method | formula | meaning |
|---|---|---|
| `best` | $f=e_{\,\arg\max_\ell\lvert d'_\ell\rvert}$ | one-hot on the single most discriminative layer — **the baseline** |
| `diag` | $f_\ell \propto \dfrac{\bar s^{+}_\ell-\bar s^{-}_\ell}{\tfrac12(\mathrm{Var}(s^+_\ell)+\mathrm{Var}(s^-_\ell))}$ | per-layer SNR weighting (assumes layers independent) |
| `fisher` | $f \propto \Sigma_{\mathbf s}^{-1}(\bar{\mathbf s}^{+}-\bar{\mathbf s}^{-})$ | optimal linear combine; accounts for correlated layers |

where $\Sigma_{\mathbf s}$ is the pooled within-class covariance of the $m$-vector $\mathbf s$,
ridge-regularized for small samples:
$\Sigma_{\mathbf s}\leftarrow\Sigma_{\mathbf s}+\lambda\,\tfrac{\operatorname{tr}\Sigma_{\mathbf s}}{m}I$
(default $\lambda=10^{-2}$).

`best` is a **nested special case** of `fisher`/`diag` (a one-hot $f$), so comparing them directly
answers "does using depth help?".

---

## 5b. Mixture densities: a concept class as a set of (μ, Σ)

A single Gaussian per class (§7) assumes each class is one blob. Realistically a class
is a **set of modes** ("benign" = chit-chat, homework, code, …). We therefore model each
class directly on the joint spectrogram $\mathbf{s}\in\mathbb{R}^m$ as a Gaussian mixture:

$$
p(\mathbf{s}\mid c)=\sum_{j=1}^{J_c}\pi_{cj}\,\mathcal{N}(\mathbf{s};\mu_{cj},\Sigma_{cj}),
\qquad c\in\{+,-\},
$$

and gate on $\mathrm{LLR}(\mathbf{s})=\log p(\mathbf{s}\mid +)-\log p(\mathbf{s}\mid -)>\tau$.

**Why joint, not per-layer:** each $\mu_{cj}$ is a *profile across depth*; the joint
mixture's marginal at any layer is automatically a mixture at that layer, while
cross-layer correlations are kept (a per-layer-then-combine scheme loses them, and a
"mixture of per-layer mixtures" is not a coherent density).

**Relation to §5/§7 (nesting):** with $J=1$ per class and shared covariance
$\Sigma_\mathbf{s}$, the LLR is affine in $\mathbf{s}$ with normal vector
$\Sigma_\mathbf{s}^{-1}(\bar{\mathbf{s}}^+-\bar{\mathbf{s}}^-)$ — exactly the `fisher`
bandpass of §5. With $J>1$ the effective filter becomes **input-dependent**: locally a
responsibility-weighted blend of per-component matched filters.

**Fitting:** sklearn's reference EM (seeded restarts; hand-rolled EM risks silently-wrong
fits), with the small-sample shrinkage mapped to `reg_covar` scaled by the data's mean
per-dim variance. Storage/evaluation (logpdf, sampling) is a tiny numpy `GMM` dataclass
pinned to sklearn's `score_samples` by a unit test.

**Model selection (BIC).** The set size $J_c$ is chosen per class by the Bayesian
Information Criterion over $J\in\{1,2,3\}$:

$$
\mathrm{BIC}(g)= -2\sum_{i}\log p_g(\mathbf{s}_i) \;+\; k_g\,\ln N,
\qquad
k_g=(J-1)+Jm+J\tfrac{m(m+1)}{2},
$$

lower is better. The $-2\log$-likelihood term rewards fit; the $k_g\ln N$ term charges
for capacity, which is what prevents a larger mixture from winning by memorizing noise
(a $J{+}1$ mixture always fits the training set at least as well as $J$). Worked
accounting: at $m{=}5$, one extra full-covariance component costs $\Delta k=1+5+15=21$
parameters. With $N{=}12$ (10-shot GPT-2) its rent is $21\ln 12\approx 52$ nats of
log-likelihood — unreachable from 12 samples, so BIC returns $J=1$ and the mixture gate
**collapses exactly to the §7 single-Gaussian gate**. With $N{=}8000$ (toy) a genuinely
bimodal class pays the rent immediately and $J=2$ is selected. This makes "one
distribution or many?" a question the data answers, with a safe few-shot default.

**Calibration:** the FPR-quantile rule (§7) is unchanged. The $z$-based rule becomes a
benign-mixture quantile: draw ~10k samples from $p(\mathbf{s}\mid-)$, set $\tau$ at the
$1-\Phi(-z)$ quantile of their LLRs (z=3 → ~0.1% benign-tail FPR).

**Validation** (`scripts/toy_csg_mixture.py`, 5 seeds): (S1) on unimodal data BIC picks
$J=1$ and the mixture reproduces `fisher` exactly (9.2%, Bayes 9.1%); (S2) bimodal
benign: mixture sits on the Bayes floor; (S3) benign modes *flanking* harmful on the
discriminative axis — **no linear filter can separate** (`fisher` 38.8%, AUC 0.60) while
the mixture LLR recovers it (7.1%, AUC 0.98, Bayes floor 5.8%). On real GPT-2
activations (`scripts/mixture_gpt2_check.py`, weapons concept, 12+12 prompts): BIC
collapses to $J=(1,1)$, held-out recall 1.00 / FPR 0.00, LLR rank agreement 0.986 with
the single-Gaussian gate (the two coincide at $J=1$, as §5b's nesting requires); no
multimodality detectable on this (deliberately easy) concept, which is the honest
few-shot expectation.

**Cost:** $J\,(m + \tfrac{m(m+1)}{2} + 1)$ numbers per class per concept — for $m=5$,
$J\le3$: ≤ 63 extra numbers. Code: `conceptgate/mixture.py` (GMM/EM/BIC),
`Concept` in `conceptgate/concept.py` (the scalar special case is `BandpassConcept`).

---

## 6. Why depth fusion wins (the key result)

Model the per-layer score as signal + independent noise:

$$
s_\ell = a_\ell\,y + n_\ell,\qquad y\in\{+1,-1\},\qquad n_\ell\sim\mathcal{N}(0,\sigma_\ell^2)\ \text{independent}.
$$

The class means at layer $\ell$ are $\pm a_\ell$, so $d'_\ell = 2a_\ell/\sigma_\ell$ (in
pooled-σ units; the constant cancels in the ratio below). The **matched filter** $f_\ell\propto
a_\ell/\sigma_\ell^2$ maximizes the post-blend discriminability, and because the noises are
independent, the discriminabilities **add in quadrature**:

$$
\boxed{\,d'_{\text{comb}}=\sqrt{\textstyle\sum_\ell (d'_\ell)^2}\,}\;\ge\;\max_\ell d'_\ell .
$$

At the optimal (equal-prior) threshold, each-class error of two equal-variance Gaussians separated by
$d'$ is

$$
\mathrm{err}=\Phi\!\left(-\tfrac{d'}{2}\right),\qquad \Phi=\text{standard normal CDF}.
$$

**Worked example** (the one in `scripts/toy_csg.py`): per-layer $d'=[1.6,\,2.0,\,0.6]$.

$$
\text{single best: } d'=2.0\Rightarrow \Phi(-1.0)=15.9\%,\qquad
\text{fused: } d'=\sqrt{1.6^2+2.0^2+0.6^2}=2.63\Rightarrow \Phi(-1.315)=9.4\%.
$$

The learned filter is $f\propto[1.6,2.0,0.6]$ — a **bandpass centered on the middle layer**. Empirically
(synthetic data, seeded) the recovered $d'$ is $[1.62,2.04,0.64]$ and test error drops $16.1\%\to9.4\%$,
matching theory. **Caveat:** the gain assumes (near-)independent per-layer noise; correlated layers
give a smaller gain, which `fisher` (using $\Sigma_{\mathbf s}^{-1}$) handles better than `diag`.

---

## 7. The calibrated Gaussian gate

Fit two 1-D Gaussians on the filtered score: $S\mid+\sim\mathcal{N}(\mu^+,\sigma^+)$,
$S\mid-\sim\mathcal{N}(\mu^-,\sigma^-)$. Decide with the **log-likelihood ratio**

$$
\mathrm{LLR}(S)=\log\mathcal{N}(S;\mu^+,\sigma^+)-\log\mathcal{N}(S;\mu^-,\sigma^-).
$$

- **Fire** iff $\mathrm{LLR}(S)>\tau$.
- **Abstain** (no decision) iff $\lvert \mathrm{LLR}(S)-\tau\rvert<\text{margin}$ — the uncertain
  middle band that controls false refusals.

**Threshold calibration** (`concept.py`):

- *FPR target*: $\tau=\operatorname{quantile}_{1-\mathrm{FPR}}\big(\{\mathrm{LLR}(S):S\in\text{neg}\}\big)$.
- *$z$-based* (used in P0): fire iff the score exceeds the benign mean by $z$ std, i.e. set
  $\tau=\mathrm{LLR}\!\big(\mu^- + z\,\sigma^-\big)$. With $z=3$ this is a ~0.1% benign-tail FPR, and it
  still catches harmful samples whenever $d'>z$. (Monotonicity of $S\mapsto\mathrm{LLR}(S)$ when
  $\sigma^+\approx\sigma^-$ makes this a clean score threshold.)

---

## 8. Combining K concepts

A bank of $K$ concepts fires if **any** fires (a union / max-LLR rule):

$$
\mathrm{fire}(a)=\bigvee_{k=1}^{K}\big[\mathrm{LLR}_k(S_k)>\tau_k\big],\qquad
\text{which}(a)=\arg\max_k \mathrm{LLR}_k(S_k).
$$

`which` selects the concept whose steering direction is used in reroute (§9).

---

## 9. Reroute (steering) math

Detection runs in standardized space, but the reroute hook perturbs the **raw** residual stream, so we
keep a raw-space direction per layer:

$$
w^{\text{raw}}_\ell=\frac{\bar a^{+}_\ell-\bar a^{-}_\ell}{\lVert\bar a^{+}_\ell-\bar a^{-}_\ell\rVert}\in\mathbb{R}^d .
$$

When concept $k$ fires, at each tapped layer $\ell$ we modify the stream:

$$
\boxed{\,a_\ell \leftarrow a_\ell - \alpha\, w^{\text{raw}}_{k,\ell}\,}.
$$

$-\alpha$ steers **away** (suppress / refuse, the guardrail use); $+\alpha$ steers **toward**
(amplify). $\alpha$ is tuned on a calibration split; too large degrades fluency.

---

## 10. Why ~10 prompts suffice (few-shot)

1. **Means, not parameters.** $w_\ell$, $f$, and the Gaussians are all first/second moments — they
   need *samples*, not a fitted high-dim network.
2. **Tokens multiply data.** A prompt of $T$ tokens yields $T$ activation vectors; ~10 prompts give a
   few hundred per class (used for per-token variants).
3. **Last-token rep avoids dilution.** Shared boilerplate tokens ("How do I …") appear in both classes
   with opposite labels and crush $d'$ if you fit on every token. The prompt's last-token activation
   carries the aggregate intent, so fitting on it gives strong separation from few prompts (on GPT-2:
   $d'\approx3.3$–$4.0$ from 12+12 prompts).

---

## 11. Parameter count, cost, and code map

**Parameters** (per concept, $m$ layers, width $d$): directions $m\,d$ + raw directions $m\,d$ +
standardization $2md$ + filter $m$ + Gaussians $4$. For GPT-2 ($m=5,d=768$): $\approx 4\times3840+5+4
\approx 1.5\times10^4$ numbers. $K$ concepts scale linearly. Far below the ≪1M target.

**Cost.** Fit = a few matrix means + one $m\times m$ solve = milliseconds (no backprop, no GPU
training). Inference = $m$ dot products of size $d$ + a length-$m$ blend per gated token = negligible
vs one transformer forward. Abort *saves* compute (skips remaining decoding).

**Code map:**

| Math | Code |
|---|---|
| standardization, $w_\ell$, spectrogram, $f$ (best/diag/fisher), $d'$ | `conceptgate/spectral.py` |
| `BandpassConcept` (std + $w$ + $w^{\text{raw}}$ + $f$ + 1-D Gaussians; §5/§7), `Concept` (§5b mixture gate, canonical), `ConceptBank` (§8), metrics | `conceptgate/concept.py` |
| GMM density, EM fit, BIC selection (§5b) | `conceptgate/mixture.py` |
| truncated forward (run only blocks 0..max tap) | `conceptgate/taps.py` |
| reroute steering hooks, model-agnostic block access | `conceptgate/hooks.py` |
| activation extraction (last-token / per-token) | `conceptgate/data.py` |
| `ConceptGate` facade + `ConceptAction` strategies (abort / steer / emit) | `conceptgate/gate.py`, `conceptgate/actions.py` |
| §6 worked example (offline) | `scripts/toy_csg.py` |
| §5b validation (regression / bimodal / kill-shot) | `scripts/toy_csg_mixture.py` |
| GPT-2 end-to-end | `scripts/p0_smoke.py` |

---

## 12. Symbol glossary

| symbol | meaning |
|---|---|
| $d$ | residual-stream width of M |
| $\mathcal{L},m$ | tapped block layers; their count |
| $a_\ell,z_\ell$ | raw / standardized activation at layer $\ell$ |
| $\mu_0,\sigma_0$ | pooled per-dim standardization mean / std |
| $w_\ell,\,w^{\text{raw}}_\ell$ | diff-of-means direction (standardized / raw) |
| $\mathbf{s},S$ | spectrogram (per-layer scores) / filtered score |
| $f$ | depth bandpass filter |
| $d'_\ell,d'_{\text{comb}}$ | per-layer / fused discriminability |
| $\mu^\pm,\sigma^\pm$ | Gaussian params of $S$ for each class |
| $\tau,\text{margin}$ | LLR fire threshold / abstain band half-width |
| $\alpha$ | reroute steering strength |
| $\Phi$ | standard normal CDF |
