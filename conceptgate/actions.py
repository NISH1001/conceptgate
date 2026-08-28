"""Actions: what to do about a concept. Strategy pattern.

A ConceptAction is a policy. On every verdict during ConceptGate.run, the gate builds a
FireContext (a narrow view of the verdict) and calls action.decide(ctx); the action returns
a Decision the gate executes. Actions never touch the gate's internals -- only the
FireContext -- so adding an action never changes the gate. WHEN an action acts (fire /
fire-or-unsure / always) is a general Trigger the action carries; WHAT it does is the action.

Shipped: Abort (-> Stop) and Steer (-> InjectSteer). Emit (-> ForceToken) is scaffolded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


# ---- Verdict: the result of measuring a prompt against the bank ----
@dataclass
class Verdict:
    fired: bool
    concept: str | None = None      # name of the firing (max-LLR) concept
    score: float = 0.0              # the firing LLR
    step: int = 0                   # decode step (0 = the prompt itself)
    tau: float = 0.0                # the concept's calibrated threshold
    margin: float = 0.0             # score - tau (signed distance to the boundary)
    p_present: float = 0.5          # P(concept present): logistic of the margin
    abstained: bool = False         # inside the unsure band around tau (no decision)


# ---- Decision: what the gate should do (a Command the driver executes) ----
@dataclass(frozen=True)
class Stop:
    """Halt decoding; optionally append `emit` text."""
    emit: str | None = None


@dataclass(frozen=True)
class Continue:
    """Do nothing; keep generating."""


@dataclass(frozen=True)
class InjectSteer:
    """Add per-layer steering vectors to the residual stream, then keep generating."""
    deltas: dict = field(default_factory=dict)   # {layer_index: 1-D vector}


@dataclass(frozen=True)
class ForceToken:
    """Force the next token id. (scaffolded, not yet wired)"""
    token_id: int = 0


Decision = Stop | Continue | InjectSteer | ForceToken
STOP = Stop()
CONTINUE = Continue()


# ---- FireContext: the narrow view handed to an action (never the gate itself) ----
@dataclass
class FireContext:
    verdict: Verdict          # which concept fired, its score, the step
    concept: Any              # the detected Concept object (the verdict's attributed concept)
    layers: list[int]         # tapped block indices
    concepts: Any = None      # {name: Concept} -- all learned, for actions that name one
    tok: Any = None           # tokenizer (for markers / decoding)
    seq: Any = None           # token ids generated so far
    step: int = 0             # decode step (0 = prompt)


# ---- the Strategy protocol: any object with decide() IS a ConceptAction ----
@runtime_checkable
class ConceptAction(Protocol):
    def decide(self, ctx: FireContext) -> Decision: ...


# ---- Trigger: WHEN an action acts (a general policy, shared across actions) ----
class Trigger(Enum):
    """When an action acts on a verdict (accepts the member or its string value)."""

    FIRE = "fire"                      # only on a confident fire
    FIRE_OR_UNSURE = "fire_or_unsure"  # ... or on an abstain (fail-closed-ish)
    ALWAYS = "always"                  # unconditionally, regardless of the verdict


def _triggered(when: Trigger | str, v: Verdict) -> bool:
    when = Trigger(when)
    if when is Trigger.ALWAYS:
        return True
    if when is Trigger.FIRE_OR_UNSURE:
        return v.fired or v.abstained
    return v.fired


# ---- built-in actions ----
@dataclass
class Abort:
    """Halt generation and emit a fixed marker when the trigger fires (default: on a fire)."""
    marker: str = "[GUARDRAILED]"
    when: Trigger | str = Trigger.FIRE

    def decide(self, ctx: FireContext) -> Decision:
        return Stop(emit=self.marker) if _triggered(self.when, ctx.verdict) else Continue()


@dataclass
class Steer:
    """Nudge generation along a concept's steering direction (W_raw): strength > 0 steers
    TOWARD the concept, < 0 AWAY. `concept` names which learned concept to steer along
    (default: the detected one). `when` controls it -- ALWAYS steers unconditionally, FIRE /
    FIRE_OR_UNSURE only when the concept is flagged. The gate installs the returned InjectSteer
    as forward hooks for the whole generation."""
    strength: float = 8.0
    when: Trigger | str = Trigger.FIRE
    concept: str | None = None

    def decide(self, ctx: FireContext) -> Decision:
        if _triggered(self.when, ctx.verdict):
            c = ctx.concepts[self.concept] if self.concept else ctx.concept
            W = c.W_raw  # [m, d] per-layer diff-of-means (raw space)
            deltas = {ctx.layers[i]: self.strength * W[i] for i in range(len(ctx.layers))}
            return InjectSteer(deltas=deltas)
        return Continue()
