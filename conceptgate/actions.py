"""Actions: what to do when a concept fires. Strategy pattern.

A ConceptAction is a policy. When a concept fires during ConceptGate.run, the gate builds
a FireContext (a narrow view of the firing) and calls action.on_fire(ctx). The action
returns a Decision; the gate executes it. Actions never touch the gate's internals -- only
the FireContext -- so adding an action never changes the gate.

Round 1 ships Abort. Steer / Emit and their Decisions (InjectSteer, ForceToken) are defined
here for the seam but wired into the driver in a later round.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Union, runtime_checkable


# ---- Verdict: the result of measuring a prompt against the bank ----
@dataclass
class Verdict:
    fired: bool
    concept: Optional[str] = None   # name of the firing (max-LLR) concept
    score: float = 0.0              # the firing LLR
    step: int = 0                   # decode step (0 = the prompt itself)


# ---- Decision: what the gate should do (a Command the driver executes) ----
@dataclass(frozen=True)
class Stop:
    """Halt decoding; optionally append `emit` text."""
    emit: Optional[str] = None


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


Decision = Union[Stop, Continue, InjectSteer, ForceToken]
STOP = Stop()
CONTINUE = Continue()


# ---- FireContext: the narrow view handed to an action (never the gate itself) ----
@dataclass
class FireContext:
    verdict: Verdict          # which concept fired, its score, the step
    concept: Any              # the firing Concept object (carries W_raw for steering)
    layers: List[int]         # tapped block indices
    tok: Any = None           # tokenizer (for markers / decoding)
    seq: Any = None           # token ids generated so far
    step: int = 0             # decode step (0 = prompt)


# ---- the Strategy protocol: any object with on_fire IS a ConceptAction ----
@runtime_checkable
class ConceptAction(Protocol):
    def on_fire(self, ctx: FireContext) -> Decision: ...


# ---- built-in actions ----
@dataclass
class Abort:
    """Short-circuit: stop decoding immediately and emit a fixed marker."""
    marker: str = "[GUARDRAILED]"

    def on_fire(self, ctx: FireContext) -> Decision:
        return Stop(emit=self.marker)
