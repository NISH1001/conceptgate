"""Truncated forward: read residual-stream taps while running only blocks 0..max(layers).

The stop-hook pattern (capture the deepest tap's output, then raise to halt the forward
pass) is standard interpretability tooling -- cf. nnsight, baukit's Trace(stop=True). It
is re-derived here transparently on torch's public register_forward_hook API to keep the
package dependency-light and legible; the mechanism is not a contribution.
"""
from __future__ import annotations

import numpy as np
import torch

from .hooks import get_blocks


class _StopForward(Exception):
    """Sentinel raised by the deepest tap's hook to halt the forward pass early."""


def _block_output(out):
    """A transformer block returns a tuple (hidden_states, ...) or a bare tensor."""
    return out[0] if isinstance(out, tuple) else out


class TapForward:
    """Read taps at ``layers`` while running only blocks ``0..max(layers)``.

    Produces the same ``(A, counts)`` as :func:`data.extract_token_activations` -- same
    shapes and (up to float noise) same values -- but never computes the blocks after the
    deepest tap. ``layers`` are 0-based block indices; block L's output residual stream is
    ``hidden_states[L+1]``, which is exactly what a forward hook on block L captures.
    """

    def __init__(self, model, layers: list[int]):
        self.model = model
        self.layers = list(layers)
        self.max_layer = max(layers)
        self.blocks = get_blocks(model)

    def _hook(self, captured: dict, idx: int, stop: bool):
        def hook(_module, _inp, out):
            captured[idx] = _block_output(out)[0].detach()  # [T, d] (batch item 0)
            if stop:
                raise _StopForward

        return hook

    @torch.no_grad()
    def read(
        self,
        tok,
        prompts: list[str],
        device: str = "cpu",
        last_only: bool = False,
        skip_first: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        features: list[np.ndarray] = []
        counts: list[int] = []
        for p in prompts:
            captured: dict = {}
            handles = [
                self.blocks[L].register_forward_hook(
                    self._hook(captured, L, stop=(L == self.max_layer))
                )
                for L in self.layers
            ]
            try:
                ids = tok(p, return_tensors="pt").to(device)
                try:
                    self.model(**ids, use_cache=False)
                except _StopForward:
                    pass
            finally:
                for h in handles:
                    h.remove()
            chosen = [captured[L] for L in self.layers]                    # each [T, d]
            A = torch.stack(chosen, dim=0).permute(1, 0, 2).contiguous()   # [T, m, d]
            if last_only:
                A = A[-1:]
            elif skip_first > 0:
                A = A[skip_first:]
            arr = A.float().cpu().numpy()
            features.append(arr)
            counts.append(arr.shape[0])
        return np.concatenate(features, axis=0), np.asarray(counts)
