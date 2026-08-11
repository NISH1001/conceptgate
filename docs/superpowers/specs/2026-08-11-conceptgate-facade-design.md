# ConceptGate Facade + Strategy Actions — Design Spec

**Date:** 2026-08-11
**Status:** approved design, pre-implementation
**Supersedes:** `2026-08-11-truncated-forward-design.md` (that round-1 grew into this).

---

## 1. Goal

One public object you attach to any frozen transformer that **learns concepts few-shot,
measures cheaply (truncated forward), and acts via injectable strategies** — with the
gate staying thin (it never grows when you add an action).

```python
from conceptgate import ConceptGate
from conceptgate.actions import Abort, Steer, Emit

cg = ConceptGate.from_pretrained("gpt2", layers=[4, 6, 8])
cg.learn("weapons", positives=[...], negatives=[...])
cg.calibrate(target_fpr=0.001)
cg.check(prompt)                      # Verdict — truncated forward
cg.run(prompt, action=Abort())        # strategy decides; cg drives
```

## 2. Roles (three loosely-coupled layers)

- **`Concept`** — one learned unit: directions `W`, densities (`Set((μ,Σ))`), threshold `τ`.
  MEASURES only: `score(A) -> LLR`. No firing, no action. (This is today's mixture
  `ConceptGate` class, renamed.)
- **`ConceptGate`** — the facade: holds a frozen model, tap layers, and a dict of
  `Concept`s. `from_pretrained / learn / calibrate / check / run`. ORCHESTRATES:
  drives the loop, assembles `FireContext`, executes the `Decision`. Zero action logic,
  zero density math, zero forward plumbing.
- **`ConceptAction`** (Protocol) — a Strategy: `on_fire(ctx: FireContext) -> Decision`.
  DECIDES. Never sees cg internals — only `FireContext`. Built-ins `Abort`/`Steer`/`Emit`;
  users write their own by implementing `on_fire`.

Firing is NOT intrinsic to a Concept: measuring (LLR) is intrinsic; firing is
`score > τ`, a decision using the operating point `τ`. `τ` is **calibrated** (to a target
FPR the user picks), not learned like `W`/densities.

## 3. The seam (why cg stays thin)

When a concept fires during `run`, cg builds a narrow context and calls the strategy:

```python
@dataclass
class FireContext:
    verdict: Verdict          # which concept, score, decode step
    concept: Concept          # the firing concept (W_raw for steering)
    tok:     Any              # tokenizer (for markers)
    seq:     Any              # tokens so far (torch tensor)
    steer:   SteeringHooks    # handle to inject a vector

class Decision: ...           # STOP | CONTINUE | InjectSteer(vec) | ForceToken(id)

class ConceptAction(Protocol):
    def on_fire(self, ctx: FireContext) -> Decision: ...
```

The action returns a `Decision`; **cg executes it** (STOP → stop decoding + marker;
InjectSteer → turn on hooks; etc.). Actions are pure (no side effects, return a value) →
independently testable. Adding an action never touches cg.

## 4. Module structure (end state)

```
conceptgate/
  __init__.py      public API: ConceptGate, Verdict, actions
  gate.py          ConceptGate facade (from_pretrained, learn, calibrate, check, run/_drive)
  concept.py       Concept (learned unit) + BandpassConcept (scalar baseline) + ConceptBank
  actions.py       ConceptAction protocol, FireContext, Decision, Abort, Steer, Emit
  taps.py          TapForward — truncated forward (run only blocks 0..max tap)
  hooks.py         get_blocks, SteeringHooks           (exists)
  mixture.py       GMM density + BIC                    (exists)
  concept_bank.py  spectrogram math                     (exists, unchanged)
  data.py          activation extraction                (exists)
```

Renames from current `gate.py`: `ConceptGate` (mixture detector) → `Concept`;
`BandpassConceptGate` → `BandpassConcept`; `GateBank` → `ConceptBank`. Moved into
`concept.py`. `gate.py` becomes the facade only.

## 5. Truncated forward — `TapForward` (the resource-constraint lever)

```python
class TapForward:
    def __init__(self, model, layers): ...
    def read(self, tok, prompts, device="cpu", last_only=True) -> tuple[np.ndarray, np.ndarray]:
        """Same (A, counts) as data.extract_token_activations — same shapes/values —
        but the forward STOPS at max(layers) via a stop-hook (later blocks never run)."""
```

Mechanism: capture hook per tapped block; the hook on `max(layers)` raises a sentinel
after capturing; `model(ids)` runs inside `try/except`, hooks removed in `finally`.
Standard tooling (cf. nnsight, baukit `Trace(stop=True)`; `docs/literature.md §4`) —
re-derived transparently on `torch.register_forward_hook`, cited, never claimed.

## 6. Round 1 scope — thin-but-whole slice

**Build now:**
1. `taps.py` `TapForward` + tests (equivalence to full forward, truncation proof, hygiene).
2. Rename detectors → `concept.py` (`Concept`, `BandpassConcept`, `ConceptBank`); update
   all callers (tests, scripts, docs, illustration) so the suite stays green.
3. `actions.py`: `ConceptAction`, `FireContext`, `Decision`, and **`Abort`** only.
4. `gate.py` `ConceptGate` facade: `from_pretrained`, `learn`, `calibrate`, `check`
   (via `TapForward`), `run` (drive loop, `Abort` works). Absorbs `guard.py`'s loop.
5. `__init__.py` curated public surface.
6. `scripts/truncated_forward_bench.py` — real wall-clock truncated vs full on GPT-2.

**Defer to round 2+:** `Steer`/`Emit` actions, sequential per-tap early exit (SPRT),
memory-saving (not loading tail weights), the `Reader` protocol (introduced when a second
reader consumes it).

**Keep:** `guard.py` stays until `ConceptGate.run` replaces its callers, then removed in
the same round (P0 smoke migrated to the facade).

## 7. Success criteria

- `from conceptgate import ConceptGate; ConceptGate.from_pretrained("gpt2", layers=[...])`
  works end to end: learn → calibrate → check → run(Abort) on the weapons concept,
  reproducing P0 detection (held-out recall/FPR) and the abort marker.
- `taps.py` output is `np.allclose` to the full-forward extractor; a block after the last
  tap is provably never executed.
- Benchmark shows truncated ≤ full wall-clock, with the honest caveat printed (saving is
  for detect/abort, not for generating a response).
- Full test suite + `toy_csg.py` + `toy_csg_mixture.py` stay green after the rename.

## 8. Non-goals / honest notes

- Novelty lives in the measured early-exit frontier, not the facade or the detector
  (`docs/literature.md §5`). This round builds the object; the measurement is the science.
- Obfuscated-activation attacks defeat this whole gate class — a caveat to carry, not
  solved here.
