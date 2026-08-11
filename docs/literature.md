# conceptgate — Literature & Honest Positioning

> **Purpose.** A standing record of prior art, so novelty claims are made against a
> known landscape rather than in a vacuum. The project's stance (unchanged from the
> docs): **every individual mechanism here already exists**; the contribution, if any,
> is a *specific combination* plus an *empirical result*, never a single mechanism.
>
> **⚠️ Verification status.** Section 2–3 entries were gathered by an automated
> literature sweep on **2026-08-11** and are **NOT yet hand-verified**. arXiv IDs,
> dates, and author lists MAY be wrong or hallucinated. **Verify every citation
> (open the abstract) before using it in any paper or public claim.** Items explicitly
> flagged "from memory" by the sweep are doubly suspect. Assistant knowledge cutoff is
> Jan 2026; anything dated later came from web search and is unconfirmed.

---

## 1. Mechanisms we use — prior art (baselines, not contributions)

Each row is a component of conceptgate whose *mechanism* is established. We cite these
as baselines and build on them; none is claimed as novel.

| Piece of conceptgate | Closest established work |
|---|---|
| probe on residual stream → score | Representation Engineering; linear probes (Zou et al. 2023) |
| diff-of-means direction; steer by adding it | Contrastive Activation Addition / CAA (Rimsky et al. 2023) |
| short-circuit / reroute harmful representations | **Circuit Breakers** (Zou et al. 2024) — closest steering analogue |
| class-conditional Gaussian → OOD/score gate | Mahalanobis OOD (Lee et al. 2018) |
| generative classifier (GDA/GMM) on features | classical GDA/QDA; deep-features density (e.g. DDU — *verify*) |
| "K concepts, each a feature/distribution" | Sparse Autoencoders / monosemantic features (Anthropic) |
| external input/output guard (reads text) | Llama Guard, ShieldGemma, Constitutional Classifiers |
| halt the forward pass early | early-exit / conditional compute (CALM) |
| residual stream as a communication channel | Transformer Circuits framework (Elhage et al. 2021) |

**Consequence for claims.** "A single-layer probe that aborts" is **not novel**. Both
"depth fusion helped" and "one layer sufficed" are legitimate, publishable *findings* —
the design keeps the single-layer probe as a *nested baseline* so the comparison is clean.

---

## 2. Gap map — where a claim might still be open

From the 2026-08-11 sweep. Verdicts: TAKEN / PARTIALLY OPEN / OPEN. **Unverified.**

### Claim 1 (flagship) — Sequential testing across DEPTH for concept action
Accumulate concept evidence layer-by-layer within one forward pass; abort early
(short-circuit remaining layers) under a calibrated error rate; report an
**error-vs-compute (layers-run) frontier**.

- **Verdict: PARTIALLY OPEN.** Every ingredient exists separately, but the sweep found
  no work combining SPRT/alpha-spending accumulation of a *safety/concept* decision
  *over transformer depth* with calibrated FPR and a compute–error frontier.
- **Must disarm (verify all):** EEG-Defender (early-exit jailbreak stop, ~2408.11308);
  Fast-yet-Safe early-exit risk control (~2405.20915, NeurIPS 2024); Confident Adaptive
  Transformers / CAT (~2104.08803); ConSol SPRT-over-samples (~2503.17587); SafeSwitch
  depth-cost note (~2502.01042); **WaldBoost** (SPRT over a classifier cascade, CVPR
  2005 — *from memory, verify*) — the obvious "this is WaldBoost on layers" objection.
- **Surviving delta:** anytime-valid gating of a concept decision over *depth* with
  calibrated FPR and an explicit compute–error frontier. Frame as a novel *composition*.

### Claim 2 — Change-point / onset detection over GENERATION TIME
CUSUM-style calibrated detection of drift-toward-a-concept across generated tokens.

- **Verdict: TAKEN (in substance).** Sweep found calibrated CUSUM for hallucination
  onset (~2606.12476) and hidden-state trajectory jailbreak defense (TrajGuard
  ~2604.07727; Kelp ~2510.09694; streaming probes ~2606.10487). Crowded.
- **Action:** drop as a standalone claim, or absorb into Claim 1 as one unified
  "detection over the depth×time grid" — which no single paper offers.

### Claim 3 — Closed-loop (feedback) activation steering
- **Verdict: TAKEN. Drop.** PID steering (~2510.04309, ICLR 2026 poster); LQR/optimal
  control of toxicity/refusal steering (~2604.19018); adaptive-strength steering
  (several 2025–2026). No standalone novelty.

### Claim 4 — Few-shot class-conditional GMMs (BIC) on multi-layer projections
- **Verdict: PARTIALLY OPEN.** Single-Gaussian/Mahalanobis on activations is saturated
  (MOOD benchmark ~2605.21602; contrastive Mahalanobis ~2512.12069; kNNGuard few-shot
  ~2607.02072). No BIC-selected GMM-on-multi-layer-projections with calibrated LLR found.
- **Landmine for the "concepts are multimodal" finding — must position against:**
  *The Geometry of Refusal / Concept Cones* (~2502.17420, ICML 2025); *More to Refusal
  than a Single Direction* (~2602.02132); *Not All Features Are Linear* (~2405.14860 —
  *verify*). Our multimodality result must be framed as density-level *confirmation* of
  this line, not a discovery. Benchmark against MOOD's Mahalanobis baseline.

### Claim 5 (framing) — "network as signal carrier / probe stack as receiver"
- **Verdict: OPEN as terminology, WEAK as a claim.** No term-collision on "concept
  spectrogram." But framing earns zero novelty credit unless it *buys* a technical
  result (e.g. the SPRT-over-depth math falling out of the receiver view). Related and
  actively growing: Concept Allocation Zone (~2605.24856); spectral-geometry of the
  residual stream (~2605.14258); matched-filter views of CNNs.

---

## 3. Landmine papers — cite regardless of direction

- **Obfuscated Activations Bypass Latent-Space Defenses** (Bailey et al., ~2412.09565,
  ICLR 2026 — *verify*): adversarially drives harmfulness-probe recall 100%→0%, attacks
  probes/SAEs/Gaussian-OOD — i.e. *our entire gate class*. An adversarial-robustness
  caveat is **mandatory**.
- **False Sense of Security** (~2509.03888 — *verify*): probing-based malicious-input
  detection fails to generalize.
- **Online Safety Monitoring for LLMs** (~2607.02510 — *verify*): finds simple
  thresholding competitive with sequential-testing monitors — must rebut or accommodate.

---

## 4. Tooling / substrate — NOT method, do not reinvent, do not claim

The plumbing (read/edit activations, partial/truncated forward) is commoditized. We
build on the standard PyTorch `register_forward_hook` API; these frameworks wrap the same
API at scale. The **stop-hook / truncated forward** is standard tooling — never a claim.

| Tool | What it is | Relation to us |
|---|---|---|
| **nnsight** (ndif-team) | tracing framework; read/edit/partial-run internals, incl. remote execution of huge models | most active/popular; a *paradigm* (trace context, proxies). Overkill for "tap 5 layers + stop"; adopt only if the project grows into heavy multi-model / remote interp |
| **baukit** (Bau lab) | tiny utilities; `Trace(..., stop=True)` = the exact truncated-forward trick | closest-fit; either depend on it or transparently re-derive the ~15-line hook |
| **TransformerLens** | hook points everywhere, run-with-cache | interp-standard substrate |
| **pyvene / pyreft** (Stanford) | representation intervention as a library | overlaps the *steering* mechanism |
| **repeng** | control vectors from contrastive pairs | overlaps our *direction + steering* — even part of the method has a library |

**Decision (current):** write the stop-hook transparently in our own `hooks.py` (it sits
on the `get_blocks()` architecture-abstraction we already own), with a comment citing
nnsight/baukit as the standard tools. Rationale: the repo's value is legibility (pure-
numpy core, readable end to end); a tracing DSL fights that for a one-line need. Pick a
dependency by *fit-to-need and cost-to-legibility*, not by star count.

---

## 5. Honest positioning (the one paragraph to remember)

conceptgate introduces **no novel mechanism**. Probes, diff-of-means directions,
activation steering, Gaussian/GMM density scoring, circuit-breaker reroute, and
forward-hook truncation are all prior art or standard tooling. The only defensible
contribution is a **specific combination** — few-shot, calibrated, sequential concept
gating that lets a *frozen* model detect and act (abort / short-circuit / steer) from its
*own middle layers*, running a fraction of itself — and, above all, the **empirical
result**: the measured error-vs-compute frontier and whether depth-fusion / mixtures beat
their nested baselines on a real model. If those measurements are strong, the finding is
publishable regardless of the machinery being classical. If BIC always picks J=1 and one
layer always suffices, *that* is also a valid, honest finding.

**Open risks to keep visible:** (a) obfuscated-activation attacks defeat this whole gate
class; (b) probe-based detection may not generalize off-distribution; (c) simple
thresholding may match sequential testing — the compute-saving, not raw accuracy, is then
the claim. None of these are hidden; all must be reported.
