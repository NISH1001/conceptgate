from conceptgate.actions import (
    STOP,
    Abort,
    ConceptAction,
    Continue,
    FireContext,
    Stop,
    Verdict,
)


def _ctx():
    return FireContext(verdict=Verdict(fired=True, concept="weapons", score=5.0), concept=None, layers=[4, 6, 8])


def test_abort_returns_stop_with_marker():
    d = Abort(marker="[X]").on_fire(_ctx())
    assert isinstance(d, Stop)
    assert d.emit == "[X]"


def test_abort_default_marker():
    assert Abort().on_fire(_ctx()).emit == "[GUARDRAILED]"


def test_abort_satisfies_protocol():
    assert isinstance(Abort(), ConceptAction)


def test_custom_action_satisfies_protocol_structurally():
    class LogAndAbort:
        def on_fire(self, ctx):
            return STOP

    assert isinstance(LogAndAbort(), ConceptAction)


def test_non_action_fails_protocol():
    class NotAnAction:
        def something_else(self):
            pass

    assert not isinstance(NotAnAction(), ConceptAction)
