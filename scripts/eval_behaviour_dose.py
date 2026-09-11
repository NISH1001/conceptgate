"""Stage 1 of docs/plans/steerability-gate.md: a reliability-tested per-prompt behavioural dose.

For every prompt in the steerability set and every arm (none, +/-alpha along the few-shot
jailbreak direction, +/-alpha along a random direction of matched norm) sample K continuations
at a fixed temperature, score each for refusal with two instruments (the repo's lexicon and a
rejection classifier), and store everything -- texts included -- so the analysis never
regenerates. Also records the first-token refusal log-odds per arm (the report's proxy; the
measure is Logit-Gap Steering's per-prompt margin, 2506.24056, in basket-sum form) and the
unsteered tap activations, so stage 2 can predict the behavioural dose from the prompt alone.

Run:  uv run --with datasets python scripts/eval_behaviour_dose.py                    # Qwen, K=16
      uv run --with datasets python scripts/eval_behaviour_dose.py --quick --no-classifier
      uv run --with datasets python scripts/eval_behaviour_dose.py --models google/gemma-2-2b-it
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_gate as E  # noqa: E402  (prompt set, few-shot examples, lexicon, token baskets)
from conceptgate import ConceptGate  # noqa: E402
from conceptgate.concept import Direction  # noqa: E402

ARMS = ("none", "minus", "plus", "rand_minus", "rand_plus")
CLASSIFIER = "protectai/distilroberta-base-rejection-v1"   # label index 1 = rejection
N_TEMPLATES, N_BENIGN = 120, 48


def tag_of(model: str) -> str:
    return model.replace("/", "__")


def random_direction(W_raw: np.ndarray, seed: int) -> np.ndarray:
    """Unit rows from the same generator as scaleup_eval, so the floor is comparable."""
    rr = np.random.default_rng([7000, seed])
    U = rr.normal(size=W_raw.shape)
    return U / np.linalg.norm(U, axis=-1, keepdims=True)


def arm_deltas(arm: str, W_raw: np.ndarray, U: np.ndarray, layers, mag: float):
    """Per-layer residual deltas for an arm; None for the unsteered arm."""
    if arm == "none":
        return None
    sign = -1.0 if arm.endswith("minus") else 1.0
    D = U if arm.startswith("rand") else W_raw
    return {int(layers[j]): sign * mag * D[j] for j in range(len(layers))}


def subsample_balanced(prompts, n):
    """First n prompts taken round-robin over kinds, so every kind stays represented."""
    if n is None or n >= len(prompts):
        return list(prompts)
    by: dict[str, list] = {}
    for kp in prompts:
        by.setdefault(kp[0], []).append(kp)
    kinds, out, i = list(by), [], 0
    while len(out) < n and any(by.values()):
        k = kinds[i % len(kinds)]
        if by[k]:
            out.append(by[k].pop(0))
        i += 1
    return out


def sample_seed(seed: int, prompt_index: int, arm_index: int) -> int:
    """SeedSequence, not addition: additive seeds collided across seeds once already."""
    return int(np.random.SeedSequence([seed, prompt_index, arm_index]).generate_state(1)[0])
