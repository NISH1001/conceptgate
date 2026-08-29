# Concept Spectrogram Gate (CSG)

A lightweight, few-shot **internal concept adapter**. Given a frozen model **M**, a tiny sidekick
**G** taps M's residual stream at several layers, builds a per-concept *spectrogram* of diff-of-means
projections across depth, blends it with a learned **bandpass filter**, and gates on a calibrated
distribution. When a concept fires, G can **act** — emit a score, **abort** generation, or **reroute**
the representation (add a steering vector).

It's **concept-agnostic** — give it ~10 examples of *any* concept (a topic, a tone, an intent) and it
detects it and can steer toward/away from it. Guardrailing (catch a policy-violating prompt, refuse)
is just one application.

## The idea in one picture

Picture M as a **hallway** a "thought" travels down. We bolt **microphones** (probes) at several
layers; each is tuned to a concept's **signature** and reports a **loudness**. The loudness across
depth is a **spectrogram**; a tiny learned **bandpass filter** blends the layers (trusting the clear
ones), and a **bell-curve gate** decides fire / abstain / pass. On fire: **abort** (stop / emit a token) or
**reroute** (steer the stream so M derails).

```
            ┌──────────────── M (frozen) ────────────────┐
input text →│ … → block ℓ → … → final → logits           │→ token
            └──────┬─────────────────────────────────────┘
                   ▼  tap residual stream at layers ℓ1…ℓm
  per concept:  sℓ = wₖ,ℓ · standardize(aℓ)     loudness per layer  (spectrogram s ∈ ℝᵐ)
                S  = fₖ · s                       bandpass blend → one score
                fire if LLRₖ(S) > τₖ              calibrated Gaussian gate
                   │
                   ├─ ABORT   → stop / emit EOS or [GUARDRAILED]
                   └─ REROUTE → add −α·wₖ to the stream  → M refuses on its own
```

The whole novelty in one line: **don't pick one layer — read the concept's loudness across depth and
blend it with a learned bandpass filter.** On synthetic data this cuts error ~16% → ~9% vs the best
single layer (discriminabilities add in quadrature: `d' = √Σ d'ℓ²`).

## Docs
- [`docs/concepts.md`](docs/concepts.md) — full conceptual write-up: intuition, prior art & honest
  novelty, the concept-agnostic generality (detect + steer), modes, the `G(M)→Tg` attach vision,
  caveats, roadmap.
- [`docs/math.md`](docs/math.md) — the rigorous mathematics: standardization, diff-of-means (LDA
  optimality), spectrogram, the three bandpass filters, the depth-fusion / quadrature result, the
  calibrated LLR gate, reroute steering, few-shot justification, parameter count, and a code map.
- Design doc / plan: `../../.claude/plans/i-have-this-vague-encapsulated-reef.md`.

## Use

It's **concept-agnostic** — teach it *any* concept from ~10 examples per side. A guardrail is
just one application (make the concept the thing to catch).

```python
from conceptgate import ConceptGate, LoadMode
from conceptgate.actions import Abort

cg = ConceptGate.from_pretrained("gpt2", layers=[4, 6, 8])   # attach to any frozen HF model

cg.learn("cooking",
         positives=["How long do I boil pasta?", "What temperature to bake bread?", "How do I dice an onion?"],
         negatives=["What's the capital of France?", "How do I center a div in CSS?", "Who won the game?"])
cg.calibrate(z=3.0, margin=0.1)                      # sets tau (operating point) + an abstain band

v = cg.check("What's the best way to sear a steak?")
# -> Verdict(fired=…, concept="cooking", p_present=…, abstained=…, score=…, tau=…)
#    (few-shot is honest: with only ~3 examples a borderline prompt may come back abstained)
```

Each `check` returns a **Verdict** — a three-way call `fired` / `abstained` / pass, plus a calibrated
`p_present`. `margin` in `calibrate` sets the abstain band (`|p_present − 0.5| < margin`, a probability
half-width; `margin=0` keeps a plain boolean). `run` reports the same Verdict inside its `RunResult`.

**Several concepts at once** — fires if any fires, and tells you which:
```python
cg.learn("cooking", positives=[...], negatives=[...])
cg.learn("travel",  positives=[...], negatives=[...])
cg.calibrate(z=3.0)
cg.check("Cheapest month to fly to Tokyo?").concept  # -> "travel"
```

**As a guardrail** (one application) — the concept is the thing to catch; inject an action:
```python
cg.learn("policy_violation", positives=[...], negatives=[...])
cg.calibrate(z=3.0, margin=0.1)                      # operating point (higher z = stricter) + abstain band
cg.run(prompt, action=Abort())                       # aborts + emits a marker when it fires
```
`ConceptGate` measures + orchestrates; actions are injected strategies (`Abort`, `Steer`, or your
own via the `ConceptAction` protocol). `check` / input-side `run` use a **truncated forward** (only
blocks `0..max(layers)` run), so a firing input is caught having run only a fraction of M.

**Steer the generation** (the write side) — the same concept direction that *reads* can also
*write*. Added back into the residual stream while M generates, it bends the output **toward** the
concept (positive strength) or **away** (negative). This is the real differentiator over a
classifier: it *changes* behavior, few-shot, no training.
```python
from conceptgate.actions import Steer, Trigger

cg.learn("food", positives=[...], negatives=[...])
cg.run(prompt, action=Steer(concept="food", fraction=0.06, when=Trigger.ALWAYS))
```
Magnitude is a **`fraction` of the residual norm** by default (model-agnostic — ~0.03–0.10 is the
coherent range, higher tips into gibberish); the sign picks toward/away. Pass `strength=` for an
absolute value instead. `concept=` names which learned concept to steer along (default: the
detected one), and raises if that name wasn't learned. Steering needs the concept **learned** (it
uses the diff-of-means direction `W_raw`) — not calibrated.

**`Trigger` — when an action acts** (shared by every action): `FIRE` (only on a confident fire),
`FIRE_OR_UNSURE` (also on an abstain), `ALWAYS` (unconditional, regardless of the verdict). So
`Abort(when=Trigger.FIRE)` blocks when a concept fires; `Steer(when=Trigger.ALWAYS)` steers every
generation. Verdict-gated triggers (`FIRE` / `FIRE_OR_UNSURE`) need `calibrate`; `ALWAYS` does not.

**`Emit` — a soft redirect** (vs `Abort`'s hard stop): seed a fixed opening into the completion
when the trigger fires, then let M continue from it in its own voice — open with a refusal on a
jailbreak fire instead of a bracketed marker.
```python
from conceptgate.actions import Emit

cg.run(prompt, action=Emit(text="\nI'm sorry, but I can't help with that.", when=Trigger.FIRE))
```
Where `Abort` appends its marker *after* decoding halts, `Emit` prepends the text to the completion
and generation goes on, conditioned on it. (Include a leading space or newline in `text` for a
clean break from the prompt.)

**Detection direction** (opt-in): `cg.learn(..., direction=Direction.LOGISTIC)` fits a per-layer
discriminative direction instead of the default diff-of-means — it closes the detection gap to a
linear SVM (largest gains on weaker models), while the steering direction stays diff-of-means.
Default is `Direction.DIFF_OF_MEANS`.

**Two optimization knobs** (independent — compose freely):
```python
# memory: how much of the model's WEIGHTS to load
ConceptGate.from_pretrained("gpt2", layers=[4,6,8], load=LoadMode.FULL)        # default; can generate
ConceptGate.from_pretrained("gpt2", layers=[4,6,8], load=LoadMode.UP_TO_TAPS)  # blocks 0..max only; detect-only

# compute: how activations are extracted during learn
cg.learn("cooking", positives=[...], negatives=[...], batch_size=1)   # loop, least memory (default)
cg.learn("cooking", positives=[...], negatives=[...], batch_size=32)  # padded batch, faster
```
`LoadMode.UP_TO_TAPS` never materializes the tail (later blocks, final norm, lm_head) — for
an 8B model tapped early that's ~6 GB instead of 16 GB — but it can't generate (`run()` that
would generate raises a clear error; use `check()`). Use it as a normal object or scoped:
```python
with ConceptGate.from_pretrained("gpt2", layers=[4,6,8], load=LoadMode.UP_TO_TAPS) as cg:
    cg.learn(...); cg.check(...)       # model freed automatically on exit (or call cg.unload())
```

**Debug logging** — `debug=True` turns on loguru logs of the gate's decisions (silent by default):
```python
cg = ConceptGate.from_pretrained("gpt2", layers=[4,6,8], debug=True)
# DEBUG conceptgate.gate:learn     - learn 'cooking': 12+12 prompts, J=(1,1)
# DEBUG conceptgate.gate:_verdict  - verdict@0: {'cooking': (-9.73, -1)} -> fired=False concept=cooking p=0.02 abstain=False
```

## Layout
```
conceptgate/
  __init__.py       # public surface: ConceptGate, Abort/Steer/Emit/Trigger, Direction, Verdict
  gate.py           # ConceptGate — facade: from_pretrained/learn/calibrate/check/run
  concept.py        # Concept (mixture LLR unit) + BandpassConcept baseline + ConceptBank
  actions.py        # ConceptAction protocol, FireContext, Decision, Abort, Steer, Emit, Trigger  (strategy layer)
  taps.py           # TapForward — the signal listener: truncated forward (or full=True), batching
  spectral.py       # diff-of-means directions, spectrogram, bandpass filter (pure numpy)
  mixture.py        # Set((mu,Sigma)) per class: GMM (sklearn EM + BIC) on the spectrogram
  hooks.py          # model-agnostic block access + steering forward hooks
  data.py           # load concept prompt sets (JSON)
data/concepts/      # tiny labeled prompt sets (~10/class)
scripts/
  demo_marimo.py    # interactive marimo walkthrough: model/tap picker, detect, steer, tap-depth vs cost
  demo.py           # end-to-end facade demo on GPT-2: learn / calibrate / check / run + speedup
  toy_csg.py        # OFFLINE math check (no model): depth filter beats single layer
  toy_csg_mixture.py # mixture validation: regression / bimodal benign / kill-shot
  mixture_gpt2_check.py    # mixture on real GPT-2 activations
```

## Run (always via `uv run`)
```bash
uv sync                              # build .venv from pyproject (torch, transformers, numpy, sklearn)
uv run python scripts/toy_csg.py     # offline math check  -> VALIDATION PASS (err 16% -> 9%)
uv run python scripts/toy_csg_mixture.py       # mixture densities -> MIXTURE VALIDATION PASS
uv run python scripts/mixture_gpt2_check.py    # mixture on real GPT-2 acts -> PASS
uv run python scripts/demo.py        # facade end-to-end + speedup -> DEMO: PASS
uv run pytest tests/ -q              # unit tests
```

## Status
- **P0 (done):** mechanism wired + validated on GPT-2. Offline math: depth bandpass cuts error
  ~16% → ~9%. On GPT-2: held-out detection recall 1.00 / FPR 0.00; abort emits `[GUARDRAILED]`;
  reroute changes the continuation. (GPT-2 can't generate coherent harmful text, so it only
  demonstrates input-side cleanly — the science is P1.)
- **P0.5 (done):** mixture densities — a concept class is a `Set((mu, Sigma))` (GMM on the
  joint spectrogram, BIC-selected J; 10-shot data yields J=1, the single-Gaussian gate).
  Where no linear filter separates (flanking benign modes: fisher 38.8% err / AUC 0.60),
  the mixture LLR recovers the Bayes floor (7.1% / AUC 0.98).
- **Facade (done):** `ConceptGate.from_pretrained(model, layers)` — one object that learns
  few-shot, `check`s via a **truncated forward** (only blocks `0..max tap`), and `run`s an
  injected `ConceptAction` strategy (`Abort` shipped). On GPT-2, truncated detection is
  ~46% faster than the full forward, activations bit-identical. The saving is for
  detect/abort; generation still needs the full forward.
- **Loading (done):** `LoadMode.UP_TO_TAPS` loads only blocks `0..max(tap)` (base model,
  `num_hidden_layers=max+1`) — the tail is never materialized, so a large model tapped early
  loads a fraction of its weights (validated bit-identical at the taps). `batch_size` on
  `learn` is the memory↔compute dial for extraction. `ConceptGate` is usable as an object or
  a context manager (`unload()` frees the weights). `LoadMode.STREAM` (accelerate offload)
  reserved for later.
- **Steer (done):** the write side — `run(action=Steer(...))` adds a concept's diff-of-means
  direction back into the residual stream during generation, steering the output toward/away
  (validated coherent on Qwen2.5-0.5B; magnitude-sensitive). `Steer(concept=...)` picks among many
  learned concepts. Every action shares a `Trigger` (`FIRE` / `FIRE_OR_UNSURE` / `ALWAYS`), and
  `run` is the single driver for detect-and-act and unconditional steering alike.
- **Emit (done):** the soft redirect — `run(action=Emit(text=...))` seeds a fixed opening into the
  completion when the trigger fires (`-> ForceToken`), then M continues from it; unlike `Abort`'s
  post-stop marker, it steers the output by conditioning generation on the forced prefix. Next:
  mid-generation token forcing and the sequential per-tap early exit.
- **P1 (next):** Gemma-2-2B-it. Single-best-layer baseline (A); measure few-shot recall/FPR/PR.
- **P2:** CSG depth filter vs A vs MLP (B) ablation; abort vs reroute comparison.
- **P3:** jailbreak robustness vs a text-classifier guard; multi-concept K.
