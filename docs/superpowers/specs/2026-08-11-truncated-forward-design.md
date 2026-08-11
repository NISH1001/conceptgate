# Truncated Forward (`taps.py`) — Design Spec

**Date:** 2026-08-11
**Status:** approved scope ("just the truncated forward"), pre-implementation
**Round:** 1 of the ConceptAct track. Round 2 (`ConceptAct` action layer + protocols)
builds on this; see §6.

---

## 1. Motivation

conceptgate currently reads activations via `model(..., output_hidden_states=True)`
(`data.py`, `guard.py`) — which computes **all** blocks and the unembedding even though
the gate only needs blocks up to the deepest tap. The project's thesis ("a frozen model
can refuse from its own middle layers, running a fraction of itself") is therefore not
yet realized in code: the architecture permits the saving, the implementation doesn't.

Round 1 closes exactly that gap: a reader that runs blocks `0 … max(layers)` and stops.

**Positioning (docs/literature.md §4):** the stop-hook mechanism is standard tooling
(nnsight, baukit's `Trace(stop=True)` do the same at scale). We re-derive it
transparently (~15 lines on PyTorch's public hook API) for legibility, cite them in a
comment, and never claim the mechanism. The *contribution* this enables is the measured
error-vs-compute story, later.

## 2. Scope

**In:** one new module `conceptgate/taps.py`; unit tests; a benchmark script.
**Out (round 2+):** `ConceptAct` / action layer, `protocols.py` (no second consumer yet —
deliberately deferred to avoid premature indirection), `guard.py` refactor, generation
under truncation, memory-saving (not loading tail weights), sequential per-tap
early-exit (SPRT) — this round truncates at the *deepest* tap only.

## 3. Design

### 3.1 `conceptgate/taps.py` — `TapForward`

```python
class _StopForward(Exception):
    """Sentinel raised by the last tap's hook to halt the forward pass."""

class TapForward:
    """Read residual-stream taps while running ONLY blocks 0..max(layers).

    Mechanism: a capture hook on each tapped block; the hook on the LAST tapped
    block raises _StopForward after capturing, so no later block ever executes.
    (Standard tooling — cf. nnsight, baukit Trace(stop=True); re-derived here
    transparently on torch's register_forward_hook.)
    """
    def __init__(self, model, layers: list[int]): ...
    def read(self, tok, prompts: list[str], device="cpu", last_only=True
             ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (A, counts) exactly like data.extract_token_activations —
        same shapes [N, m, d], same values — but truncated at max(layers)."""
```

Implementation notes:
- Builds on `hooks.get_blocks(model)` (already model-agnostic: GPT-2 / Llama / Gemma /
  Qwen / NeoX).
- Hooks are registered per `read()` call and removed in `finally` — no persistent state,
  re-entrant, never leaks hooks on exception.
- Captured tensor = the block's output residual stream (tuple-aware, same convention as
  `SteeringHooks`), i.e. identical to `hidden_states[L+1]`.
- `last_only=True` keeps the last token's row (the position that attended to the whole
  prompt); `False` keeps all rows — mirroring `data.py` semantics.
- No changes to `data.py`, `gate.py`, `guard.py` this round. `data.extract_token_activations`
  remains the reference implementation the tests compare against.

### 3.2 Tests (`tests/test_taps.py`)

1. **Equivalence:** `TapForward.read()` output is `np.allclose` (tight tolerance) to
   `data.extract_token_activations()` at the tapped layers, for both `last_only` modes,
   on GPT-2 with 2–3 prompts.
2. **Truncation proof:** a probe hook on block `max(layers)+1` asserts that block is
   **never executed** during `TapForward.read()`, and **is** executed during the full
   forward.
3. **Hook hygiene:** after `read()` (including when an inner error is forced), the
   model has no lingering hooks; two consecutive `read()` calls give identical results.
4. **End-to-end:** a `ConceptGate` fit from `TapForward` activations reproduces the
   fit from full-forward activations (same recall/FPR on the weapons held-out set).

### 3.3 Benchmark (`scripts/truncated_forward_bench.py`)

Prints, for GPT-2 with the weapons taps (max tap = block 8 of 12):
- layers run: `9/12`, layer-fraction saved: `25%`;
- wall-clock per prompt, truncated vs full (median over the 24 fit prompts, warmed up);
- the extrapolation table for deeper models (taps at 30% depth of a 40-block model →
  ~67% of block compute skipped) — labeled as arithmetic, not measurement.
PASS criterion: truncated ≤ full wall-clock, equivalence spot-check passes.

## 4. Why not nnsight/baukit as a dependency

Fit-to-need vs cost-to-legibility (recorded in `docs/literature.md` §4): the need is
"tap m layers + stop once"; the repo's value is end-to-end readability. A tracing DSL
adds a paradigm for a one-liner. Decision can be revisited if the project moves to
heavy multi-model / remote interpretability.

## 5. Success criteria

- All new tests pass; existing suite + all validation scripts stay green.
- Benchmark shows a real wall-clock reduction on GPT-2 and prints the honest caveat
  that the saving applies to **detect/abort**, not to generating a response.

## 6. Round 2 preview (not in scope, recorded for continuity)

`act.py` (`ConceptAct` wrapper; `Abort`/`Steer`/`Emit` actions absorbing `guard.py`) and
`protocols.py` (`Detector`, `Action`, `Reader`) — introduced only when `ConceptAct`
gives the protocols their second consumer. Sequential per-tap early exit (evidence
accumulated tap-by-tap, abort before even reaching the deepest tap) is the research
flagship on top of this foundation.
