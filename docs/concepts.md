# conceptgate — Concepts & Ideas

> The conceptual companion to [`math.md`](./math.md). This file explains *what* CSG is and *why* it
> works, in words and pictures. The math file makes every claim here precise.

---

## 1. The problem

Given a **frozen** language model **M** that already generates tokens, we want a tiny **sidekick G**
that:

- **listens** to M's internal computation (its residual stream),
- decides, cheaply, whether a target **concept** is present (e.g. "weapon-making intent"),
- and if so, **acts** — either **halts** generation or **steers** it away —

while being **few-shot** (≈10 examples per concept), **lightweight** (≪1M params), **attachable**
to any model, and **bidirectional** (works on the input prompt *and* on each generated token).

The crucial reframing: G is **concept-agnostic**. Nothing in it knows "harmful" from "about cats".
Guardrailing is just the first application of a general **concept detector + steerer**.

---

## 2. Mental model: a hallway of microphones

Picture M as a **hallway** that a "thought" travels down. Each transformer block is a stretch of
hallway; between blocks runs the **residual stream** — a vector the model reads and writes as it
thinks. We bolt **microphones** (probes) to the wall at several points (layers).

- Each mic is **tuned to one concept's signature** — we teach it by whispering ~10 examples of the
  concept and ~10 non-examples past it; it learns "the concept sounds like *this* minus *that*".
- Each mic outputs a **loudness** — a single number for "how much of the concept do I hear here?"
- The concept is **mushy early** in the hallway (meaning not formed), **crisp in the middle**,
  **blurry late** (turning into next-token prediction). So the mics' readings form a **profile across
  depth** — a "concept spectrogram".
- A **bandpass filter** decides how much to **trust each mic** and blends them into one reliable
  score — trusting the clear mics, suppressing the muffled/noisy ones. Blending several independent
  noisy mics beats trusting any single one (like averaging witnesses).
- A **bell-curve gate** turns that blended score into fire / abstain / pass.
- On fire, we either **cut the call** (abort) or **inject a counter-signal** so the conversation
  bends elsewhere (reroute).

This whole story is faithful to the math: "tuned signature" = diff-of-means direction, "loudness" =
dot product, "spectrogram" = projections across layers, "bandpass filter" = a learned weight over
depth (a *matched filter*), "bell-curve gate" = a calibrated Gaussian likelihood-ratio test.

---

## 3. The pipeline in one picture

```
              ┌───────────────────────── M (frozen) ─────────────────────────┐
  input text →│ block ℓ1 → block ℓ2 → … → block ℓm → … → final → logits      │→ token
              └────┬──────────┬───────────────┬───────────────────────────────┘
                   │ a_ℓ1     │ a_ℓ2          │ a_ℓm          (residual stream taps)
                   ▼          ▼               ▼
   per concept k:  sℓ = wₖ,ℓ · standardize(aℓ)        ← "loudness" per layer  (the spectrogram s∈ℝ^m)
                   Sₖ = fₖ · s                          ← bandpass blend over depth (one score)
                   fire if  LLRₖ(Sₖ) > τₖ               ← calibrated Gaussian gate (+ abstain band)
                   ─────────────────────────────────────────────────
                   ANY concept fires  →  ABORT  (stop / emit EOS or [GUARDRAILED])
                                      →  REROUTE (add −α·wₖ to the stream; M derails on its own)
```

G's entire state per concept is: `m` direction vectors `w` (size `d` each), an `m`-vector filter `f`,
standardization stats, and four Gaussian scalars. For GPT-2 (m=5, d=768) that's ~8k numbers.

---

## 4. What's novel — and what isn't (no hype)

Every *mechanism* below already exists. We cite them as **baselines**, not contributions.

| Piece of CSG | Closest prior art |
|---|---|
| probe on residual stream → score | Representation Engineering, linear probes (Zou et al. 2023) |
| diff-of-means direction; steering by adding it | Contrastive Activation Addition / CAA (Rimsky et al. 2023) |
| short-circuit / reroute harmful representations | **Circuit Breakers** (Zou et al. 2024) — closest |
| "K concepts, each a feature/distribution" | Sparse Autoencoders / monosemantic features (Anthropic) |
| Gaussian/Mahalanobis score → out-of-dist gate | OOD detection (Lee et al. 2018) |
| external input/output guard | Llama Guard / Constitutional Classifiers / ShieldGemma (these read *text*) |
| halt the forward pass early | early-exit / conditional compute (CALM) |

**The genuinely new, simple core** is the one thing nearly all of the above skip: **depth**. Almost
every probe/steering method commits to a *single* layer. CSG instead treats the concept's projection
*across all tapped layers* as a 1-D signal and learns a tiny **bandpass filter over depth** (a matched
filter), gated by a **calibrated per-concept distribution**, packaged as one **few-shot, dual-mode,
attachable** module. The contribution is that combination + the head-to-head test of whether
depth-fusion beats the universal single-layer habit (see [`math.md` §6](./math.md)).

Honest stance: a plain single-layer probe-that-aborts is *not* novel. "Depth helped" and "one layer
sufficed" are **both** valid, publishable findings — the design makes the single-layer probe a
*nested baseline* so we measure it cleanly.

---

## 5. Concept-agnostic: detect AND steer *any* concept

Because a concept is just `(direction w, distribution)`, the same machinery is a general primitive:

- **Detect**: does the concept fire? (guardrail, monitoring, routing, classification)
- **Steer toward** it: add `+α·w` → amplify the concept (tone, topic, persona, "golden-gate" style)
- **Steer away** from it: add `−α·w` → suppress/refuse (the guardrail use)

So G is really a **conditional, interpretable concept-steering adapter** — a "mixing board" of
human-named concept faders, each independently dialable, that only acts when its detector fires.

It is **LoRA-like only in spirit** (lightweight, attachable, train-once-snap-on). Mechanically it is
very different:

| | LoRA | conceptgate (G) |
|---|---|---|
| Changes | model **weights** (ΔW = BA) | **activations** (`a + αw`) |
| Trained by | backprop, GPU, 1000s of examples | subtract two means, ~10 examples, no backprop |
| Applies | always, every token | **conditionally** — only when the gate fires |
| Interpretability | black-box weight delta | one human-named direction per concept |
| Power | arbitrary nonlinear skills | one linear nudge per concept (cheaper, weaker) |

A single linear direction can't match LoRA for complex *skills* (e.g. "be good at SQL"). Its sweet
spot is **concept-level** nudges.

### 5b. A concept class is a set of modes

One Gaussian per class says "benign sounds like *one* hum". Really each class is a
**set of sound profiles** — benign chit-chat, benign homework, benign code — so the
gate now models each class as a *set of (μ, Σ) components* over the depth profile
(a Gaussian mixture on the spectrogram; the set size is picked by the data via BIC,
and ~10-shot data collapses to one component — the original gate). Two payoffs:
sharper thresholds when a class is multi-modal, and an *input-dependent* bandpass —
each mode contributes its own filter, weighted by how well it explains the sample.
The kill-shot case: if benign modes *flank* harmful on the filter axis, **no** single
linear filter separates them, but the mixture LLR does (`scripts/toy_csg_mixture.py`:
fisher 38.8% error / AUC 0.60 vs mixture 7.1% / AUC 0.98, Bayes floor 5.8%).

---

## 6. The two action modes — and the `G(M) → Tg` attach vision

When the gate fires:

- **Abort** — stop decoding; emit a fixed marker / EOS. Cheapest (you skip the rest of the net); a
  hard gate. The "short-circuit" lives in the **control loop**, not M's tensor graph.
- **Reroute** — add `−α·w` to the residual stream via a forward hook; M keeps emitting its **own**
  tokens, bent away from the concept (circuit-breaker style). Here G *does* reach into M's graph.

**Attach vision (`G(M) → Tg`).** The cleanest deployment makes G a *transparent attachment*: you call
M's **own** `model.generate(...)`, and hooks inside M's forward (a) capture the taps and (b) rewrite
the final-position logits onto EOS / a guardrail token when G fires. Then M's *own* output stream
yields the token — and removing the hooks gives back **stock M**. "M works with or without G" =
hooks present or not. (Implementation note: you cannot literally "skip to the final layer and inject a
token" — the unembedding reads the *final* residual stream — so abort forces the token at the logits,
which standard `generate()` then samples and stops on.)

---

## 7. Honest caveats & limits

- **No detector is 100%.** The real goal is high-recall + low false-refusal + jailbreak-robust +
  cheap, with a **tunable operating point** — not "100% reliable".
- **Linear directions miss non-linearly-separable concepts** → mitigated by layer search and an MLP
  ablation (variant B).
- **Reroute can degrade benign quality** if `α` is too large → measure false-refusal & quality.
- **Few-shot robustness depends on the diversity** of the ~10 prompts → report variance across seeds.
- **Generated text drifts** out of the clean-prompt distribution (e.g. degenerate repetition can
  nudge a benign sequence upward) → control with the operating point (`z·σ` above the benign mean).
- **GPT-2 can't generate coherent harmful content**, so it only demonstrates *input-side* gating
  cleanly. Output-side science needs a modern instruct model (P1).

---

## 8. Roadmap

- **P0 (done):** mechanism wired + validated. Offline: depth filter cuts error ~16% → ~9%. GPT-2:
  held-out recall 1.00 / FPR 0.00; abort emits `[GUARDRAILED]`; reroute steers.
- **P0.5 (done):** density upgrade — class-conditional Gaussian *mixtures* on the joint
  spectrogram (BIC-selected, J=1 collapse on 10-shot). Toy: regression preserved;
  bimodal-benign and no-linear-filter scenarios recovered near the Bayes floor.
- **P1:** small instruct model (Gemma-2-2B-it / Qwen2.5). Single-best-layer baseline; few-shot
  recall/FPR/PR curves; output-side gating that's actually meaningful.
- **P2:** CSG depth filter vs single-layer (A) vs MLP (B); abort-vs-reroute comparison; `attach_guard`
  so `model.generate()` is guardrail-native.
- **P3:** jailbreak robustness vs a text-classifier guard; multiple concepts K; calibration / abstain.
