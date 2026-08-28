# Plan: unify interventions under `run` + strategy (Steer refactor)

Branch: `feature/steer`.  Date: 2026-08-18.

## State (what's already done)
- `Steer` action built and **validated on 3 models** (GPT-2 124M / Qwen-0.5B / SmolLM2-1.7B):
  steering toward/away a few-shot `food` concept shifts generation coherently; sweet spot
  ~**3–10% of the residual-stream norm** (absolute norms span 16→207), collapses past ~25%.
- Current (to-be-changed) surface: `Steer(strength, on_unsure)` -> `InjectSteer`; a separate
  `cg.generate(prompt, concept, strength)` for *unconditional* steering; `run` dispatches the
  action **only** when the input fires/abstains.
- Existing enums are plain `Enum` with string values (3.10-safe; project is `requires-python
  >=3.10,<3.13`, so **no `StrEnum`** — it's 3.11+): `LoadMode` (FULL/UP_TO_TAPS),
  `Direction` (DIFF_OF_MEANS/LOGISTIC), normalized via `Enum(x)`.

## Goal
Everything is `run(action=...)` + the strategy pattern. Drop `generate`. One general
`Trigger` enum for *when* an action acts. `run` always-dispatches the action; the action owns
the policy.

## Design (3 layers)
`Trigger` (when) × `Action` (what: Abort/Steer) -> `Decision` (Stop/Continue/InjectSteer/
ForceToken) -> `run` executes.

- `class Trigger(Enum)`: `FIRE="fire"`, `FIRE_OR_UNSURE="fire_or_unsure"`, `ALWAYS="always"`.
  Plain Enum (3.10-safe); normalize via `Trigger(x)` (accepts member or string).
- Actions take `when: Trigger | str = Trigger.FIRE` (replaces the `on_unsure` bool). Shared
  helper `_triggered(when, verdict) -> bool`.
- `run` dispatches the action on **every** input verdict (fire/abstain/pass), then executes:
  `Stop` -> halt+emit, `InjectSteer` -> install `SteeringHooks` + generate steered, `Continue`
  -> normal. Abort's compute-win still holds (Stop on fire skips the full model).
- Remove `cg.generate`. Unconditional steering = `Steer(when=Trigger.ALWAYS)`.
- Fast gen path in `run` when `check_output=False` (use `model.generate` instead of the manual
  token loop) so steering isn't slowed.
- Plain generation with no concept is NOT ConceptGate's job (use `cg.model.generate`).

## Steps
1. `actions.py`: add `Trigger` enum + `_triggered(when, v)`. `Abort`/`Steer` take `when`
   (drop `on_unsure`), normalize via `Trigger(when)`. Export `Trigger`.
2. `gate.py`: `run` always-dispatches; handle Stop/InjectSteer/Continue; add the fast
   `model.generate` path when `check_output=False`. Remove `generate` method.
3. `__init__.py`: export `Trigger`.
4. Tests: update `test_steer.py` / `test_actions.py` to `when=Trigger...`; add an e2e
   `run(Steer(when=ALWAYS))` check. Keep suite green.
5. Marimo notebook: steering section — pick concept + strength slider + toward/away via
   `run(Steer(when=Trigger.ALWAYS))`.
6. Validate on GPT-2 (fast), commit on branch.

## Target e2e API
```python
from conceptgate import ConceptGate, Direction
from conceptgate.actions import Abort, Steer, Trigger

cg = ConceptGate.from_pretrained("gpt2", layers=[4,6,8])
cg.learn("food", pos, neg, direction=Direction.LOGISTIC)
cg.calibrate(z=2.0, margin=0.1)

cg.check(prompt)                                              # detect only
cg.run(prompt, action=Abort(when=Trigger.FIRE_OR_UNSURE))     # block when present
cg.run(prompt, action=Steer(strength=-1.0, when=Trigger.FIRE))    # reactive steer away
cg.run(prompt, action=Steer(strength=1.0, when=Trigger.ALWAYS))   # always steer toward
```

## Deferred (not this PR)
- Auto-scale `strength` by the residual norm (pass a fraction, model-agnostic).
- `Emit`/`ForceToken` action, `Monitor`/`Log`, `Router` composite.
