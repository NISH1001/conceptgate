import types

from conceptgate.actions import Continue, Emit, FireContext, ForceToken, Trigger, Verdict


class _Tok:
    # stand-in tokenizer: one id per whitespace token (its character length), no specials
    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[len(w) for w in text.split()])


def _ctx(fired=True, abstained=False):
    return FireContext(
        verdict=Verdict(fired=fired, abstained=abstained, concept="c", score=5.0),
        concept=None, layers=[4, 6, 8], tok=_Tok(),
    )


def test_emit_forces_tokenized_ids_on_fire():
    d = Emit(text="no way", when=Trigger.FIRE).decide(_ctx(fired=True))
    assert isinstance(d, ForceToken)
    assert d.token_ids == (2, 3)                       # len("no"), len("way")


def test_emit_passes_when_not_flagged():
    assert isinstance(Emit().decide(_ctx(fired=False)), Continue)


def test_emit_fire_or_unsure_acts_on_abstain():
    d = Emit(when=Trigger.FIRE_OR_UNSURE).decide(_ctx(fired=False, abstained=True))
    assert isinstance(d, ForceToken)
    assert isinstance(Emit(when=Trigger.FIRE).decide(_ctx(fired=False, abstained=True)), Continue)


def test_emit_always_acts_on_pass():
    assert isinstance(Emit(when=Trigger.ALWAYS).decide(_ctx(fired=False)), ForceToken)
