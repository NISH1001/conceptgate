"""Actions: what to do when a concept fires. Strategy pattern.

A ConceptAction is a policy. When a concept is flagged (fires OR abstains) during
ConceptGate.run, the gate builds a FireContext (a narrow view of the verdict) and calls
action.decide(ctx). The action returns a Decision; the gate executes it. Actions never
touch the gate's internals -- only the FireContext -- so adding an action never changes
the gate. The fire-vs-unsure policy lives in the action (e.g. Abort(on_unsure=True)).

Round 1 ships Abort. Steer / Emit and their Decisions (InjectSteer, ForceToken) are defined
here for the seam but wired into the driver in a later round.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    """Add per-layer steering vectors to the residual stream, then continue. (round 2)"""
    deltas: dict = field(default_factory=dict)   # {layer_index: 1-D vector}


@dataclass(frozen=True)
class ForceToken:
    """Force the next token id. (round 2)"""
    token_id: int = 0


Decision = Stop | Continue | InjectSteer | ForceToken
STOP = Stop()
CONTINUE = Continue()


# ---- FireContext: the narrow view handed to an action (never the gate itself) ----
@dataclass
class FireContext:
    verdict: Verdict          # which concept fired, its score, the step
    concept: Any              # the firing Concept object (carries W_raw for steering)
    layers: list[int]         # tapped block indices
    tok: Any = None           # tokenizer (for markers / decoding)
    seq: Any = None           # token ids generated so far
    step: int = 0             # decode step (0 = prompt)


# ---- the Strategy protocol: any object with decide() IS a ConceptAction ----
@runtime_checkable
class ConceptAction(Protocol):
    def decide(self, ctx: FireContext) -> Decision: ...


# ---- built-in actions ----
@dataclass
class Abort:
    """Short-circuit: stop decoding and emit a fixed marker. With on_unsure=True it also
    stops on an abstain (unsure) verdict, not only a confident fire -- the fire-vs-unsure
    policy lives here in the action, not in run()."""
    marker: str = "[GUARDRAILED]"
    on_unsure: bool = False

    def decide(self, ctx: FireContext) -> Decision:
        v = ctx.verdict
        if v.fired or (self.on_unsure and v.abstained):
            return Stop(emit=self.marker)
        return Continue()


@dataclass
class Steer:
    """Nudge generation along the concept's steering direction (W_raw): strength > 0 steers
    TOWARD the concept, < 0 AWAY. Acts on a flagged verdict; on_unsure also steers on abstain.
    The gate installs the returned InjectSteer as forward hooks for the whole generation."""
    strength: float = 8.0
    on_unsure: bool = False

    def decide(self, ctx: FireContext) -> Decision:
        v = ctx.verdict
        if v.fired or (self.on_unsure and v.abstained):
            W = ctx.concept.W_raw  # [m, d] per-layer diff-of-means (raw space)
            deltas = {ctx.layers[i]: self.strength * W[i] for i in range(len(ctx.layers))}
            return InjectSteer(deltas=deltas)
        return Continue()
