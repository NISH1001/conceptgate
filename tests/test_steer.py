import numpy as np

from conceptgate.actions import Continue, FireContext, InjectSteer, Steer, Trigger, Verdict


class _DummyConcept:
    W_raw = np.ones((3, 4))  # 3 tapped layers, dim 4, unit rows


def _ctx(fired=True, abstained=False):
    return FireContext(
        verdict=Verdict(fired=fired, abstained=abstained, concept="c", score=5.0),
        concept=_DummyConcept(),
        layers=[4, 6, 8],
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
