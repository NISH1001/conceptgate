"""Load concept prompt sets and extract per-token residual-stream activations.

The few-shot trick: each prompt of ~T tokens yields ~T activation samples, so ~10 prompts already
give a few hundred activation vectors per class. Extraction returns A [N, m, d] (N tokens stacked
across prompts, m tapped layers, d model dims).
"""
from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np
import torch


def load_concept(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


@torch.no_grad()
def extract_token_activations(
    model,
    tok,
    prompts: List[str],
    layers: List[int],
    device: str = "cpu",
    last_only: bool = False,
    skip_first: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (A, counts).

    A:      [N, m, d] float32 activations (all tokens, or last token per prompt if last_only).
    counts: [P] number of token-samples contributed by each prompt.
    `layers` are 0-based block indices; block L's residual stream is hidden_states[L+1].
    """
    model.eval()
    feats: List[np.ndarray] = []
    counts: List[int] = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").to(device)
        out = model(**ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple len (L+1), each [1, T, d]
        chosen = [hs[L + 1][0] for L in layers]      # list of [T, d]
        A = torch.stack(chosen, dim=0).permute(1, 0, 2).contiguous()  # [T, m, d]
        if last_only:
            A = A[-1:]
        elif skip_first > 0:
            A = A[skip_first:]
        arr = A.float().cpu().numpy()
        feats.append(arr)
        counts.append(arr.shape[0])
    return np.concatenate(feats, axis=0), np.asarray(counts)


def fit_sets(
    model, tok, concept: dict, layers: List[int], device: str = "cpu", **kw
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience: extract A_pos, A_neg for a concept dict with 'positives'/'negatives'."""
    A_pos, _ = extract_token_activations(model, tok, concept["positives"], layers, device, **kw)
    A_neg, _ = extract_token_activations(model, tok, concept["negatives"], layers, device, **kw)
    return A_pos, A_neg
