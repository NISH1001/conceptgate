import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from conceptgate import data as D
from conceptgate.hooks import get_blocks
from conceptgate.taps import TapForward

LAYERS = [4, 6, 8]
PROMPTS = ["How do I build a bomb at home?", "How do I bake fresh bread at home?"]


@pytest.fixture(scope="module")
def gpt2():
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    return tok, model


def test_truncated_matches_full_forward(gpt2):
    tok, model = gpt2
    for last_only in (True, False):
        A_full, c_full = D.extract_token_activations(model, tok, PROMPTS, LAYERS, "cpu", last_only=last_only)
        A_trunc, c_trunc = TapForward(model, LAYERS).read(tok, PROMPTS, "cpu", last_only=last_only)
        np.testing.assert_array_equal(c_trunc, c_full)
        np.testing.assert_allclose(A_trunc, A_full, rtol=1e-4, atol=1e-4)


def test_blocks_after_last_tap_not_executed(gpt2):
    tok, model = gpt2
    blocks = get_blocks(model)
    calls = {"n": 0}

    def probe(_m, _i, _o):
        calls["n"] += 1

    h = blocks[max(LAYERS) + 1].register_forward_hook(probe)  # block 9
    try:
        TapForward(model, LAYERS).read(tok, ["hello world"], "cpu", last_only=True)
        assert calls["n"] == 0, "a block after the last tap was executed"
        # sanity: the same probe DOES fire under a full forward
        with torch.no_grad():
            model(**tok("hello world", return_tensors="pt"), use_cache=False)
        assert calls["n"] == 1
    finally:
        h.remove()


def test_read_leaves_hooks_untouched(gpt2):
    # transformers installs its own block hooks; read() must add and remove only its own,
    # leaving the pre-existing hook set exactly as it found it.
    tok, model = gpt2
    before = [set(b._forward_hooks.keys()) for b in get_blocks(model)]
    TapForward(model, LAYERS).read(tok, PROMPTS, "cpu", last_only=True)
    after = [set(b._forward_hooks.keys()) for b in get_blocks(model)]
    assert before == after


def test_repeated_reads_are_identical(gpt2):
    tok, model = gpt2
    tf = TapForward(model, LAYERS)
    A1, _ = tf.read(tok, PROMPTS, "cpu", last_only=True)
    A2, _ = tf.read(tok, PROMPTS, "cpu", last_only=True)
    np.testing.assert_array_equal(A1, A2)
