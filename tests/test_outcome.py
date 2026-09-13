import numpy as np
import pytest

from conceptgate.outcome import OutcomeHead


def _planted(n=200, m=3, d=16, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, m, d))
    w = rng.normal(size=(m, d))
    y = (A * w).sum((1, 2)) + 0.1 * rng.normal(size=n)
    return A, y, w


def test_fit_predict_shapes_and_recovery():
    A, y, w = _planted()
    h = OutcomeHead().fit(A[:150], y[:150])
    p = h.predict(A[150:])
    assert p.shape == (50,)
    assert np.corrcoef(p, y[150:])[0, 1] > 0.95
    assert h.shape == (3, 16) and h.direction().shape == (3, 16)
    cos = np.sum(h.direction() * w) / (np.linalg.norm(h.direction()) * np.linalg.norm(w))
    assert cos > 0.9


def test_predict_before_fit_and_bad_shapes_raise():
    with pytest.raises(RuntimeError):
        OutcomeHead().predict(np.zeros((2, 3, 4)))
    with pytest.raises(ValueError):
        OutcomeHead().fit(np.zeros((5, 12)), np.zeros(5))
    with pytest.raises(ValueError):
        OutcomeHead().fit(np.zeros((5, 3, 4)), np.zeros(4))


def test_standardization_is_applied_at_predict_time():
    A, y, _ = _planted()
    h = OutcomeHead().fit(A, y)
    assert np.allclose(h.predict(A[:1]), ((A[:1].reshape(1, -1) - h.mu) / h.sd) @ h.w + h.b)


def test_predicted_and_both_triggers():
    from conceptgate.actions import Both, Continue, FireContext, InjectSteer, Predicted, Steer, Trigger, Verdict

    class _C:
        W_raw = np.ones((3, 4))

    def ctx(fired, dose):
        return FireContext(verdict=Verdict(fired=fired, abstained=False, concept="c", score=1.0,
                                           outcomes={"dose": dose}),
                           concept=_C(), layers=[4, 6, 8], concepts={"c": _C()})

    s = Steer(strength=1.0, when=Predicted("dose", 0.5))
    assert isinstance(s.decide(ctx(False, 0.7)), InjectSteer)      # presence irrelevant
    assert isinstance(s.decide(ctx(True, 0.2)), Continue)
    assert isinstance(Steer(strength=1.0, when=Predicted("dose", 0.5, above=False)).decide(ctx(True, 0.2)), InjectSteer)
    b = Steer(strength=1.0, when=Both(Trigger.FIRE, Predicted("dose", 0.5)))
    assert isinstance(b.decide(ctx(True, 0.7)), InjectSteer)
    assert isinstance(b.decide(ctx(False, 0.7)), Continue)
    assert isinstance(b.decide(ctx(True, 0.2)), Continue)
    with pytest.raises(KeyError, match="not a learned outcome"):
        Steer(when=Predicted("nope", 0.5)).decide(ctx(True, 0.7))
    assert isinstance(Steer(strength=1.0, when="always").decide(ctx(False, 0.0)), InjectSteer)   # strings still work


def test_learn_outcome_round_trip_on_gpt2():
    from conceptgate import ConceptGate

    cg = ConceptGate.from_pretrained("gpt2", layers=[4, 6, 8], device="cpu")
    prompts = ["How do I boil pasta?", "What is the capital of France?", "Explain gravity briefly.",
               "Write a haiku about rain.", "How do I center a div?", "Who won the 1998 world cup?"]
    y = [0.9, 0.1, 0.5, 0.7, 0.2, 0.3]
    cg.learn_outcome("dose", prompts, y)
    assert "dose" in cg.outcomes
    v = cg.check("How long should I steam broccoli?")
    assert "dose" in v.outcomes and np.isfinite(v.outcomes["dose"])
    fit_pred = cg.outcomes["dose"].predict(
        cg._taps.read(cg.tok, [cg._format(p) for p in prompts], cg.device, last_only=True)[0])
    assert np.corrcoef(fit_pred, y)[0, 1] > 0.9      # 6 points in 2304 dims: fits its own labels
    with pytest.raises(ValueError, match="labels"):
        cg.learn_outcome("bad", prompts, y[:3])
    cg.unload()


def test_outcome_head_reproduces_the_analysis_script_on_measured_data():
    """Cross-implementation check: the library head and scripts/analyze_behaviour_dose.py's ridge must
    agree on the REAL Qwen behavioural doses. Skipped when the activation cache (gitignored) is absent."""
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts = os.path.join(root, "scripts")
    res = os.path.join(scripts, "behaviour_dose_results__Qwen__Qwen2.5-0.5B-Instruct.json")
    npy = os.path.join(scripts, "behaviour_dose_acts__Qwen__Qwen2.5-0.5B-Instruct.npy")
    if not (os.path.exists(res) and os.path.exists(npy)):
        pytest.skip("behaviour-dose run artifacts not present (the .npy cache is gitignored)")
    sys.path.insert(0, scripts)
    import analyze_behaviour_dose as A
    from sklearn.model_selection import GroupKFold

    A.HERE = scripts
    meta, acts, _ = A.load("Qwen/Qwen2.5-0.5B-Instruct", out_dir=scripts)
    rows, k = meta["rows"], int(meta["k"])
    kind = np.array([r["kind"] for r in rows])
    atk = kind != "benign"
    P = A.rates(rows, "clf", list(range(k)))
    y = A.dose(P)[0][atk]
    X = acts[atk]
    g = A.groups_for(kind[atk])

    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=7).split(X, y, g):
        pred[te] = OutcomeHead(alpha=10.0).fit(X[tr], y[tr]).predict(X[te])
    lib = A.sp(pred, y)
    ana = A.cv(A._zscore(X.reshape(len(y), -1))[0], y, 0, groups=g, n_splits=7)[0]
    assert abs(lib - ana) < 0.02, f"library {lib:+.4f} vs analysis {ana:+.4f}"
    assert lib > 0.5, f"the measured result should reproduce, got {lib:+.4f}"
