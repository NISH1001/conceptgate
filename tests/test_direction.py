import numpy as np
import pytest

from conceptgate import Direction
from conceptgate.concept import Concept
from conceptgate.spectral import fit_directions_logistic


def _synth():
    """[N, m, d] pos/neg separable on dim 0 (no model needed)."""
    rng = np.random.default_rng(0)
    Ap = rng.standard_normal((20, 3, 16))
    Ap[:, :, 0] += 3.0
    An = rng.standard_normal((20, 3, 16))
    return Ap, An


def test_direction_enum_normalizes():
    assert Direction("logistic") is Direction.LOGISTIC
    assert Direction(Direction.LOGISTIC) is Direction.LOGISTIC
    assert Direction("diff") is Direction.DIFF_OF_MEANS


def test_direction_rejects_unknown():
    with pytest.raises(ValueError):
        Direction("bogus")


def test_logistic_direction_shape_and_unit_norm():
    Ap, An = _synth()
    W = fit_directions_logistic(Ap, An)
    assert W.shape == (3, 16)
    assert np.allclose(np.linalg.norm(W, axis=-1), 1.0)


@pytest.mark.parametrize("d", [Direction.LOGISTIC, "logistic", Direction.DIFF_OF_MEANS, "diff"])
def test_concept_accepts_enum_and_str(d):
    Ap, An = _synth()
    c = Concept(name="t", direction=d).fit(Ap, An)
    assert c.llr(Ap).mean() > c.llr(An).mean()          # separates the synthetic classes
    # steering direction stays diff-of-means regardless of the detection direction
    assert c.W_raw is not None and c.W_raw.shape == (3, 16)
