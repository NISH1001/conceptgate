# Concept Spectrogram Gate (CSG)

A lightweight, few-shot **internal guardrail adapter**. Given a frozen model **M**, a tiny sidekick
**G** taps M's residual stream at several layers, builds a per-concept *spectrogram* of diff-of-means
projections across depth, blends it with a learned **bandpass filter**, and gates on a calibrated
distribution. When it fires it either **aborts** generation (emit `[GUARDRAILED]`) or **reroutes**
the representation (add a steering vector) so M refuses on its own.

> The core idea, math, and prior-art are in `../../.claude/plans/i-have-this-vague-encapsulated-reef.md`.
> Mental model: microphones bolted down a hallway, each tuned to a concept's "signature", blended by
> a bandpass filter that trusts the clearest layers.

## Layout
```
conceptgate/
  concept_bank.py   # diff-of-means directions, spectrogram, closed-form bandpass filter (pure numpy)
  gate.py           # standardized fit, Gaussian LLR gate, abstain band, K-concept GateBank, metrics
  hooks.py          # model-agnostic block access + steering (reroute) forward hooks
  data.py           # load concept prompt sets + extract per-token / last-token activations
  guard.py          # Guard: generation loop with input/output-side gating, abort + reroute modes
data/concepts/      # tiny labeled prompt sets (~10/class)
scripts/
  toy_csg.py        # OFFLINE validation of the math (no model): depth filter beats single layer
  p0_smoke.py       # end-to-end on GPT-2: detection + abort + reroute
```

## Run (always via `uv run`)
```bash
uv sync                              # build .venv from pyproject (torch, transformers, numpy, matplotlib)
uv run python scripts/toy_csg.py     # offline math check  -> VALIDATION PASS (err 16% -> 9%)
uv run python scripts/p0_smoke.py    # GPT-2 end-to-end     -> P0 SMOKE: PASS
```

## Status
- **P0 (done):** mechanism wired + validated on GPT-2. Offline math: depth bandpass cuts error
  ~16% → ~9%. On GPT-2: held-out detection recall 1.00 / FPR 0.00; abort emits `[GUARDRAILED]`;
  reroute changes the continuation. (GPT-2 can't generate coherent harmful text, so it only
  demonstrates input-side cleanly — the science is P1.)
- **P1 (next):** Gemma-2-2B-it. Single-best-layer baseline (A); measure few-shot recall/FPR/PR.
- **P2:** CSG depth filter vs A vs MLP (B) ablation; abort vs reroute comparison.
- **P3:** jailbreak robustness vs a text-classifier guard; multi-concept K.
