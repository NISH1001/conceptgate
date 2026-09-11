import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import eval_behaviour_dose as B  # noqa: E402


def test_random_direction_is_unit_rows_and_deterministic():
    W = np.ones((3, 8))
    U1, U2 = B.random_direction(W, 0), B.random_direction(W, 0)
    assert U1.shape == (3, 8)
    assert np.allclose(np.linalg.norm(U1, axis=-1), 1.0)
    assert np.allclose(U1, U2)
    assert not np.allclose(U1, B.random_direction(W, 1))


def test_arm_deltas_signs_and_sources():
    W = np.ones((2, 4))
    U = 2 * np.ones((2, 4))
    L = [4, 6]
    assert B.arm_deltas("none", W, U, L, 3.0) is None
    d = B.arm_deltas("minus", W, U, L, 3.0)
    assert set(d) == {4, 6} and np.allclose(d[4], -3.0 * np.ones(4))
    assert np.allclose(B.arm_deltas("plus", W, U, L, 3.0)[6], 3.0 * np.ones(4))
    assert np.allclose(B.arm_deltas("rand_minus", W, U, L, 3.0)[4], -6.0 * np.ones(4))
    assert np.allclose(B.arm_deltas("rand_plus", W, U, L, 3.0)[6], 6.0 * np.ones(4))


def test_subsample_balanced_keeps_every_kind():
    prompts = [("template", f"t{i}") for i in range(10)] + [("short", f"s{i}") for i in range(4)] \
        + [("request", "r0")] + [("benign", f"b{i}") for i in range(5)]
    out = B.subsample_balanced(prompts, 8)
    kinds = [k for k, _ in out]
    assert len(out) == 8
    assert set(kinds) == {"template", "short", "request", "benign"}
    assert out[0] == ("template", "t0") and ("request", "r0") in out
    assert B.subsample_balanced(prompts, None) == prompts
    assert B.subsample_balanced(prompts, 100) == prompts


def test_sample_seed_is_distinct_per_prompt_and_arm():
    s = {B.sample_seed(0, i, a) for i in range(5) for a in range(5)}
    assert len(s) == 25
    assert B.sample_seed(0, 3, 2) == B.sample_seed(0, 3, 2)


def test_rejection_classifier_orders_refusal_above_compliance():
    clf = B.RejectionClassifier(device="cpu")
    p = clf.score(["I'm sorry, but I can't help with that request.",
                   "Sure! Here are the steps you need to follow to get started.",
                   ""])
    assert p[0] > 0.5 > p[1]
    assert p[2] == 0.0


def test_bucket_pad_rounds_up_to_multiple():
    assert B.bucket_pad(293, 32) == 27 and (293 + 27) % 32 == 0
    assert B.bucket_pad(64, 32) == 0
    assert B.bucket_pad(1, 32) == 31
    assert B.bucket_pad(100, None) == 0 and B.bucket_pad(100, 0) == 0


def test_run_tag_suffixes_nonzero_seeds():
    assert B.run_tag("Qwen/Qwen2.5-0.5B-Instruct", 0) == "Qwen__Qwen2.5-0.5B-Instruct"
    assert B.run_tag("Qwen/Qwen2.5-0.5B-Instruct", 1) == "Qwen__Qwen2.5-0.5B-Instruct__seed1"


def test_gumbel_sampler_matches_softmax_frequencies():
    import torch
    torch.manual_seed(0)
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -3.0]])
    T = 0.7
    target = torch.softmax(logits / T, -1)[0]
    n = 40000
    proc = B.GumbelSampler(T, torch.Generator().manual_seed(1))
    draws = proc(None, logits.repeat(n, 1)).argmax(-1)
    freq = torch.bincount(draws, minlength=5).float() / n
    assert torch.allclose(freq, target, atol=0.01), (freq, target)
    assert proc(None, logits).shape == logits.shape
