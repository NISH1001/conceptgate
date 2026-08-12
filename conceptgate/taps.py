"""Truncated forward: read residual-stream taps while running only blocks 0..max(layers).

The stop-hook pattern (capture the deepest tap's output, then raise to halt the forward
pass) is standard interpretability tooling -- cf. nnsight, baukit's Trace(stop=True). It
is re-derived here transparently on torch's public register_forward_hook API to keep the
package dependency-light and legible; the mechanism is not a contribution.

`batch_size` is the memory<->compute dial for extraction: 1 runs one prompt at a time
(least activation memory), larger stacks prompts into one padded forward (faster). The
single-prompt forward is already fully vectorized; batching only amortizes per-forward
overhead. Read taps via these hooks, NEVER output_hidden_states: on a truncated model the
deepest tap is the final block, so hidden_states would apply the final layernorm to it.
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

    Returns ``(A, counts)`` where ``A`` is ``[N, m, d]`` activations (last-token rep per
    prompt, or every token if not ``last_only``) and ``counts`` is per-prompt sample counts.
    ``layers`` are 0-based block indices; a forward hook on block L captures its raw output
    residual stream (equal to ``hidden_states[L+1]`` on a full model, minus the final
    layernorm) -- so we read via hooks, never output_hidden_states.
    """

    def __init__(self, model, layers: list[int]):
        self.model = model
        self.layers = list(layers)
        self.max_layer = max(layers)
        self.blocks = get_blocks(model)

    def _hook(self, captured: dict, idx: int, stop: bool):
        def hook(_module, _inp, out):
            captured[idx] = _block_output(out).detach()   # [B, T, d]
            if stop:
                raise _StopForward

        return hook

    def _run(self, enc, device: str, full: bool = False) -> dict:
        """Forward with fresh tap hooks; returns {layer: [B, T, d]}.

        Stops at the deepest tap (truncated) unless full=True, which runs every block --
        the full-forward baseline (same taps, more compute). Used for benchmarking /
        validating the truncated path.
        """
        captured: dict = {}
        handles = [
            self.blocks[L].register_forward_hook(
                self._hook(captured, L, stop=(not full and L == self.max_layer))
            )
            for L in self.layers
        ]
        try:
            try:
                self.model(**enc.to(device), use_cache=False)
            except _StopForward:
                pass
        finally:
            for h in handles:
                h.remove()
        return captured

    @torch.no_grad()
    def read(
        self,
        tok,
        prompts: list[str],
        device: str = "cpu",
        last_only: bool = False,
        skip_first: int = 0,
        batch_size: int = 1,
        full: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract tap activations for `prompts`. Truncated (stop at max tap) by default;
        full=True runs every block -- the full-forward baseline, same taps."""
        self.model.eval()
        if batch_size and batch_size > 1:
            if not last_only:
                raise ValueError(
                    "batch_size > 1 requires last_only=True "
                    "(per-token extraction is not batched); use batch_size=1"
                )
            return self._read_batched(tok, prompts, device, batch_size, full)
        return self._read_loop(tok, prompts, device, last_only, skip_first, full)

    def _read_loop(self, tok, prompts, device, last_only, skip_first, full=False):
        features: list[np.ndarray] = []
        counts: list[int] = []
        for p in prompts:
            cap = self._run(tok(p, return_tensors="pt"), device, full)
            chosen = [cap[L][0] for L in self.layers]                    # each [T, d]
            A = torch.stack(chosen, dim=0).permute(1, 0, 2).contiguous()  # [T, m, d]
            if last_only:
                A = A[-1:]
            elif skip_first > 0:
                A = A[skip_first:]
            arr = A.float().cpu().numpy()
            features.append(arr)
            counts.append(arr.shape[0])
        return np.concatenate(features, axis=0), np.asarray(counts)

    def _read_batched(self, tok, prompts, device, batch_size, full=False):
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        features: list[np.ndarray] = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True)
            cap = self._run(enc, device, full)
            # last real token per prompt (robust to left/right padding): the position
            # where the mask's cumulative sum first hits its max.
            last = enc["attention_mask"].cumsum(1).argmax(1).tolist()    # [B]
            for b in range(len(chunk)):
                acts = [cap[L][b, last[b]] for L in self.layers]         # each [d]
                A = torch.stack(acts, dim=0)                             # [m, d]
                features.append(A.float().cpu().numpy()[None, ...])      # [1, m, d]
        return np.concatenate(features, axis=0), np.asarray([1] * len(prompts))
