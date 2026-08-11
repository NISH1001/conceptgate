# Concept Spectrogram Gate (CSG)

A lightweight, few-shot **internal guardrail adapter**. Given a frozen model **M**, a tiny sidekick
**G** taps M's residual stream at several layers, builds a per-concept *spectrogram* of diff-of-means
projections across depth, blends it with a learned **bandpass filter**, and gates on a calibrated
distribution. When it fires it either **aborts** generation (emit `[GUARDRAILED]`) or **reroutes**
the representation (add a steering vector) so M refuses on its own.

It's **concept-agnostic** — give it ~10 examples of *any* concept and it can detect it and steer
toward/away from it. Guardrailing is just the first use.

## The idea in one picture

Picture M as a **hallway** a "thought" travels down. We bolt **microphones** (probes) at several
layers; each is tuned to a concept's **signature** and reports a **loudness**. The loudness across
depth is a **spectrogram**; a tiny learned **bandpass filter** blends the layers (trusting the clear
ones), and a **bell-curve gate** decides fire / pass. On fire: **abort** (stop / emit a token) or
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
```python
from conceptgate import ConceptGate
from conceptgate.actions import Abort

cg = ConceptGate.from_pretrained("gpt2", layers=[4, 6, 8])   # attach to any frozen HF model
cg.learn("weapons", positives=[...], negatives=[...])         # few-shot (~10/class)
cg.calibrate(z=3.0)                                           # sets tau (operating point)

cg.check(prompt)                    # -> Verdict(fired, concept, score) — TRUNCATED forward
cg.run(prompt, action=Abort())      # strategy decides; the gate drives + executes
```
`ConceptGate` measures + orchestrates; actions are injected strategies (`Abort`, and your
own via the `ConceptAction` protocol). `check`/input-side `run` use a **truncated forward**
(only blocks `0..max(layers)` run), so a harmful prompt is caught having run a fraction of M.

## Layout
```
conceptgate/
  __init__.py       # public surface: ConceptGate, actions, Verdict
  gate.py           # ConceptGate — facade: from_pretrained/learn/calibrate/check/run
  concept.py        # Concept (mixture LLR unit) + BandpassConcept baseline + ConceptBank
  actions.py        # ConceptAction protocol, FireContext, Decision, Abort  (strategy layer)
  taps.py           # TapForward — truncated forward (run only blocks 0..max tap)
  concept_bank.py   # diff-of-means directions, spectrogram (pure numpy)
  mixture.py        # Set((mu,Sigma)) per class: GMM (sklearn EM + BIC) on the spectrogram
  hooks.py          # model-agnostic block access + steering (reroute) forward hooks
  data.py           # load concept prompt sets + extract activations
  guard.py          # legacy generation loop (reroute demo; folds into ConceptGate.run round 2)
data/concepts/      # tiny labeled prompt sets (~10/class)
scripts/
  toy_csg.py        # OFFLINE math check (no model): depth filter beats single layer
  toy_csg_mixture.py # mixture validation: regression / bimodal benign / kill-shot
  mixture_gpt2_check.py    # mixture on real GPT-2 activations
  truncated_forward_bench.py  # truncated vs full forward wall-clock on GPT-2
  p0_smoke.py       # end-to-end on GPT-2: detection + abort + reroute
```

## Run (always via `uv run`)
```bash
uv sync                              # build .venv from pyproject (torch, transformers, numpy, sklearn)
uv run python scripts/toy_csg.py     # offline math check  -> VALIDATION PASS (err 16% -> 9%)
uv run python scripts/toy_csg_mixture.py       # mixture densities -> MIXTURE VALIDATION PASS
uv run python scripts/mixture_gpt2_check.py    # mixture on real GPT-2 acts -> PASS
uv run python scripts/truncated_forward_bench.py  # truncated forward -> BENCH: OK (~46% faster)
uv run python scripts/p0_smoke.py    # GPT-2 end-to-end     -> P0 SMOKE: PASS
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
  detect/abort; generation still needs the full forward. Next: `Steer`/`Emit` actions and
  the sequential per-tap early exit (the novel lever; see docs/superpowers/specs/).
- **P1 (next):** Gemma-2-2B-it. Single-best-layer baseline (A); measure few-shot recall/FPR/PR.
- **P2:** CSG depth filter vs A vs MLP (B) ablation; abort vs reroute comparison.
- **P3:** jailbreak robustness vs a text-classifier guard; multi-concept K.
