import numpy as np
import pytest

from conceptgate.actions import Continue, FireContext, InjectSteer, Steer, Trigger, Verdict


class _DummyConcept:
    def __init__(self, val=1.0):
        self.W_raw = np.full((3, 4), val)  # 3 tapped layers, dim 4


def _ctx(fired=True, abstained=False):
    detected = _DummyConcept(1.0)
    food = _DummyConcept(2.0)              # a different, NAMED concept
    return FireContext(
        verdict=Verdict(fired=fired, abstained=abstained, concept="c", score=5.0),
        concept=detected,
        layers=[4, 6, 8],
        concepts={"c": detected, "food": food},
    )


def test_steer_returns_scaled_injectsteer_on_fire():
    d = Steer(strength=2.0).decide(_ctx(fired=True))
    assert isinstance(d, InjectSteer)
    assert set(d.deltas) == {4, 6, 8}                       # keyed by tapped block index
    assert np.allclose(d.deltas[4], 2.0 * np.ones(4))       # strength * W_raw[layer]


def test_steer_passes_when_not_flagged():
    assert isinstance(Steer().decide(_ctx(fired=False)), Continue)


def test_steer_fire_or_unsure_acts_on_abstain():
    assert isinstance(Steer(when=Trigger.FIRE_OR_UNSURE).decide(_ctx(fired=False, abstained=True)), InjectSteer)
    assert isinstance(Steer(when=Trigger.FIRE).decide(_ctx(fired=False, abstained=True)), Continue)


def test_steer_always_acts_on_pass():
    # unconditional steering: acts even when nothing fired (replaces the old generate())
    assert isinstance(Steer(when=Trigger.ALWAYS).decide(_ctx(fired=False, abstained=False)), InjectSteer)


def test_steer_named_concept_uses_that_direction():
    # concept="food" steers along that concept's W_raw (2), not the detected concept's (1)
    d = Steer(strength=1.0, concept="food").decide(_ctx(fired=True))
    assert np.allclose(d.deltas[4], np.full(4, 2.0))


def test_steer_unlearned_concept_raises_with_available_names():
    with pytest.raises(KeyError, match="not a learned concept"):
        Steer(concept="missing").decide(_ctx(fired=True))
