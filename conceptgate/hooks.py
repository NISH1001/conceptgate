"""Model-agnostic access to a frozen model's residual stream.

Reading activations uses `output_hidden_states=True` (clean, works for any HF causal LM).
Writing (reroute) uses forward hooks on the transformer *block* modules that ADD a steering
vector to the block's output residual stream.

Layer convention (used everywhere): `layers` are 0-based BLOCK indices. Block L's output residual
stream is `hidden_states[L + 1]` (index 0 is the embedding output). A steering hook on block L
therefore perturbs exactly that hidden state and everything downstream of it.
"""
from __future__ import annotations

import torch


def get_blocks(model) -> list[torch.nn.Module]:
    """Return the list of transformer block modules, across common architectures."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)                      # GPT-2 family
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)                       # Llama / Gemma / Qwen / Mistral
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)                    # GPT-NeoX / Pythia
    raise ValueError(
        f"Unknown architecture {type(model).__name__}; add its block path to get_blocks()."
    )


def num_layers(model) -> int:
    return len(get_blocks(model))


class SteeringHooks:
    """Add per-layer steering vectors to block outputs during the forward pass (for reroute).

    Usage:
        steer = SteeringHooks(model).register([4, 5, 6])
        steer.set_deltas({4: vec4, 5: vec5})   # vec_* are torch tensors shaped [d]
        steer.enabled = True
        ... model(...) ...                      # deltas added to each listed block's output
        steer.clear(); steer.remove()
    """

    def __init__(self, model):
        self.blocks = get_blocks(model)
        self.deltas: dict[int, torch.Tensor] = {}
        self.handles: list = []
        self.enabled: bool = False

    def _make_hook(self, idx: int):
        def hook(_module, _inp, out):
            if not self.enabled or idx not in self.deltas:
                return out
            delta = self.deltas[idx]
            if isinstance(out, tuple):
                hs = out[0]
                return (hs + delta.to(hs.dtype).to(hs.device),) + tuple(out[1:])
            return out + delta.to(out.dtype).to(out.device)

        return hook

    def register(self, layers: list[int]) -> SteeringHooks:
        for idx in layers:
            self.handles.append(self.blocks[idx].register_forward_hook(self._make_hook(idx)))
        return self

    def set_deltas(self, deltas: dict[int, torch.Tensor]) -> None:
        self.deltas = deltas

    def clear(self) -> None:
        self.deltas = {}
        self.enabled = False

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self) -> "SteeringHooks":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
