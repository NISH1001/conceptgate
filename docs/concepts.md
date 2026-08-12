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
(With mixture densities, §5b, the four scalars become a small *set* of (μ, Σ) profiles per class —
tens of numbers — and the single filter `f` is absorbed into the mixture's decision rule.)

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

**Back to the hallway.** The simplest gate assumes each class makes *one* hum: "benign
sounds like this, harmful sounds like that" — one Gaussian per class. But listen to real
benign traffic in the hallway: chit-chat, chemistry homework, code questions — each a
*different voice* with its own loudness profile across the microphones. Squeezing many
voices into one hum smears the class into a fat, vague blob: thresholds get pushed out
(missed detections) and the space *between* the voices gets mislabeled (false alarms).

So each class keeps a small **library of sound profiles** — a `Set((μ, Σ))`.
Each profile is one voice: its expected loudness at every microphone (μ, a profile
*across depth*) and how those loudnesses co-vary (Σ). Formally the library is a
**Gaussian mixture model (GMM)** on the spectrogram, one mixture per class:

```
taps → loudness per mic:  s = (s₁ … s_m)          ← the spectrogram (unchanged)
     → which class's profile LIBRARY explains s better?
         p(s | harmful) = π₁·N(μ₁,Σ₁) + π₂·N(μ₂,Σ₂) + …   ← GMM, J_pos profiles
         p(s | benign ) = π₁·N(μ₁,Σ₁) + π₂·N(μ₂,Σ₂) + …   ← GMM, J_neg profiles
     → fire if  log p(s|harmful) − log p(s|benign) > τ     ← same LLR gate as before
```

**Who decides how many profiles (this is BIC).** A bigger library *always* fits the
training data at least as well — extra Gaussians will happily memorize noise. So the
set size J can't be chosen by fit alone. The **Bayesian Information Criterion** scores
each candidate as `BIC = −2·log-likelihood + k·ln(N)` (k = parameter count, N = sample
count; lower wins): every parameter pays **rent**, and an extra profile is admitted only
if the fit it buys exceeds the rent it costs. Consequences you can see in our runs:
- **Few-shot (N≈12, GPT-2):** a second full-covariance profile over 5 mics costs ~21
  parameters → rent ≈ 21·ln 12 ≈ 52; twelve samples never earn that → **J=1**: the
  library holds a single profile per class, which *is* the single-Gaussian gate — the
  J=1 special case.
- **Data-rich with real clusters (toy, N=8000):** the second benign profile pays for
  itself instantly → **J=2**, and the gate exploits it.

**Why it matters — the kill-shot example.** Put two benign voices *on either side* of
harmful along the filter axis (benign at −2 and +2, harmful at 0). A bandpass filter
score is one number with one threshold — no threshold separates "middle" from "both
sides", so the P0 gate is stuck near chance: **38.8% error, AUC 0.60**. The profile
library isn't confused at all — s is near a benign profile on the left, near a benign
profile on the right, and near the harmful profile in the middle: **7.1% error,
AUC 0.98** (true optimum 5.8%). Run it: `uv run python scripts/toy_csg_mixture.py`.

**And on the real hallway (GPT-2, weapons concept).** Fit on 12 harmful + 12 benign
prompts, tested on held-out prompts: BIC keeps **J=1 per class** (twelve samples can't
pay for more — correct behavior, not failure), held-out **recall 1.00 / FPR 0.00**, and
the mixture gate's rankings agree with the single-Gaussian gate at **0.986** — the two
gates are interchangeable in this regime, as the math says they must be. Run it: `uv run python scripts/mixture_gpt2_check.py`. Honest caveat:
bombs-vs-cookies is an *easy* concept; whether real concept classes are genuinely
multi-modal (so J>1 wins on real data) is exactly the P1 question, to be tested with
near-boundary benign prompts (chemistry homework, military history) on a larger model.

**Where the GMM lives in the code:** `conceptgate/mixture.py` fits the library
(sklearn's reference EM + BIC selection; a tiny numpy `GMM` dataclass stores and
evaluates it), and `ConceptGate` in `conceptgate/gate.py` is the gate built on it —
directions, standardization, steering (`W_raw`), and `GateBank`/`Guard` compatibility
shared with `BandpassConceptGate`, the scalar-filter special case kept as the
experimental baseline (filter variants best/diag/fisher). Math: [`math.md` §5b](./math.md).

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
- **P0.5 (done):** mixture densities — class-conditional Gaussian *mixtures* on the joint
  spectrogram (BIC-selected, J=1 on 10-shot). Toy: single-Gaussian results reproduced;
  bimodal-benign and no-linear-filter scenarios recovered near the Bayes floor.
- **Facade (done):** `ConceptGate.from_pretrained(model, layers)` — one object that learns a
  concept few-shot, MEASURES via a **truncated forward** (only blocks `0..max(tap)` run), and
  ACTS via an injected `ConceptAction` strategy (`Abort` shipped; `Steer`/`Emit` scaffolded).
  The gate stays thin: it measures and drives; actions are loosely coupled through a
  `FireContext`. Truncated detection on GPT-2 is ~46% faster than the full forward, activations
  bit-identical — a harmful prompt is caught having run a fraction of M and never generating.
  (Saving is for detect/abort; generation still needs the full forward.)
- **P1:** small instruct model (Gemma-2-2B-it / Qwen2.5). Single-best-layer baseline; few-shot
  recall/FPR/PR curves; output-side gating that's actually meaningful.
- **P2:** CSG depth filter vs single-layer (A) vs MLP (B); abort-vs-reroute comparison; `attach_guard`
  so `model.generate()` is guardrail-native.
- **P3:** jailbreak robustness vs a text-classifier guard; multiple concepts K; calibration / abstain.
