from conceptgate.actions import (
    STOP,
    Abort,
    ConceptAction,
    Continue,
    FireContext,
    Stop,
    Trigger,
    Verdict,
)


def _ctx():
    return FireContext(verdict=Verdict(fired=True, concept="weapons", score=5.0), concept=None, layers=[4, 6, 8])


def _ctx_abstain():
    return FireContext(
        verdict=Verdict(fired=False, abstained=True, concept="weapons", score=0.1),
        concept=None, layers=[4, 6, 8],
    )


def test_abort_returns_stop_with_marker():
    d = Abort(marker="[X]").decide(_ctx())
    assert isinstance(d, Stop)
    assert d.emit == "[X]"


def test_abort_default_marker():
    assert Abort().decide(_ctx()).emit == "[GUARDRAILED]"


def test_abort_default_passes_on_abstain():
    assert isinstance(Abort().decide(_ctx_abstain()), Continue)


def test_abort_fire_or_unsure_stops_on_abstain():
    d = Abort(when=Trigger.FIRE_OR_UNSURE).decide(_ctx_abstain())
    assert isinstance(d, Stop) and d.emit == "[GUARDRAILED]"


def test_abort_always_stops_even_on_pass():
    ctx = FireContext(verdict=Verdict(fired=False, abstained=False), concept=None, layers=[4, 6, 8])
    assert isinstance(Abort(when="always").decide(ctx), Stop)   # trigger accepts a str too


def test_abort_satisfies_protocol():
    assert isinstance(Abort(), ConceptAction)


def test_custom_action_satisfies_protocol_structurally():
    class LogAndAbort:
        def decide(self, ctx):
            return STOP

    assert isinstance(LogAndAbort(), ConceptAction)


def test_non_action_fails_protocol():
    class NotAnAction:
        def something_else(self):
            pass

    assert not isinstance(NotAnAction(), ConceptAction)
