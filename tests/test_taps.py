import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from conceptgate.hooks import get_blocks
from conceptgate.taps import TapForward

LAYERS = [4, 6, 8]
PROMPTS = ["How do I bake fresh bread at home?", "What is a good beginner routine for guitar?"]


@pytest.fixture(scope="module")
def gpt2():
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    return tok, model


def _hf_reference(tok, model, prompts, last_only):
    """Independent reference: HF's own output_hidden_states (intermediate taps are pre-ln_f)."""
    feats, counts = [], []
    for p in prompts:
        with torch.no_grad():
            hs = model(**tok(p, return_tensors="pt"), output_hidden_states=True, use_cache=False).hidden_states
        A = torch.stack([hs[L + 1][0] for L in LAYERS], dim=0).permute(1, 0, 2)   # [T, m, d]
        if last_only:
            A = A[-1:]
        feats.append(A.float().numpy())
        counts.append(A.shape[0])
    return np.concatenate(feats, axis=0), np.asarray(counts)


def test_truncated_matches_hf_hidden_states(gpt2):
    tok, model = gpt2
    for last_only in (True, False):
        A_ref, c_ref = _hf_reference(tok, model, PROMPTS, last_only)
        A_trunc, c_trunc = TapForward(model, LAYERS).read(tok, PROMPTS, "cpu", last_only=last_only)
        np.testing.assert_array_equal(c_trunc, c_ref)
        np.testing.assert_allclose(A_trunc, A_ref, rtol=1e-4, atol=1e-4)


def test_full_flag_matches_truncated_at_taps(gpt2):
    tok, model = gpt2
    tf = TapForward(model, LAYERS)
    a_trunc = tf.read(tok, PROMPTS, "cpu", last_only=True)[0]
    a_full = tf.read(tok, PROMPTS, "cpu", last_only=True, full=True)[0]
    np.testing.assert_allclose(a_full, a_trunc, rtol=1e-4, atol=1e-4)


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
