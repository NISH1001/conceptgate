"""Attach a CSG gate to a frozen model and generate under guardrail.

Two configurable modes when the gate fires:
  - "abort":   stop decoding and return a fixed [GUARDRAILED] marker (cheapest; a hard gate).
  - "reroute": turn on steering hooks that add  -alpha * w  at the tapped layers, so M's own
               continuation is bent away from the concept (circuit-breaker style).

The gate is checked input-side (on the prompt's last-token activations, before any generation) and
output-side (on each newly generated token's activations).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .concept import ConceptBank
from .hooks import SteeringHooks

GUARDRAILED = "[GUARDRAILED]"


@dataclass
class GenResult:
    text: str
    fired: bool = False
    stage: Optional[str] = None        # "input" | "output" | None
    concept: Optional[int] = None      # index of firing concept in the bank
    n_new: int = 0                     # number of new tokens produced before stopping
    fire_step: Optional[int] = None    # decode step at which it fired (0 = on the prompt)


class Guard:
    def __init__(self, model, tok, gate_bank: ConceptBank, layers: List[int], device: str = "cpu"):
        self.model = model
        self.tok = tok
        self.gb = gate_bank
        self.layers = layers
        self.device = device
        self.steer = SteeringHooks(model).register(layers)

    # -- read last-token activations [1, m, d] from a forward output --
    def _last_act(self, hidden_states) -> np.ndarray:
        chosen = [hidden_states[L + 1][0, -1] for L in self.layers]  # each [d]
        A = torch.stack(chosen, dim=0)                               # [m, d]
        return A.float().cpu().numpy()[None, ...]                    # [1, m, d]

    def _steer_deltas(self, concept_idx: int, alpha: float) -> Dict[int, torch.Tensor]:
        """-alpha * w_l for the firing concept at each tapped layer."""
        g = self.gb.gates[concept_idx]
        deltas = {}
        for i, L in enumerate(self.layers):
            # steer in RAW activation space (W_raw), since the hook perturbs the raw residual stream
            w = torch.from_numpy(np.asarray(g.W_raw[i], dtype=np.float32))
            deltas[L] = (-alpha) * w
        return deltas

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        mode: str = "abort",
        max_new_tokens: int = 30,
        alpha: float = 10.0,
        check_input: bool = True,
        greedy: bool = True,
    ) -> GenResult:
        self.steer.clear()
        ids = self.tok(prompt, return_tensors="pt").to(self.device).input_ids
        n_prompt = ids.shape[1]
        seq = ids
        res = GenResult(text="")
        steering_on = False

        for step in range(max_new_tokens + 1):
            out = self.model(input_ids=seq, output_hidden_states=True, use_cache=False)
            A = self._last_act(out.hidden_states)
            fired = bool(self.gb.fire(A)[0])
            stage = "input" if step == 0 else "output"

            if fired and (stage == "output" or check_input):
                if not res.fired:
                    res.fired = True
                    res.stage = stage
                    res.concept = int(self.gb.which(A)[0])
                    res.fire_step = step
                if mode == "abort":
                    gen = self.tok.decode(seq[0, n_prompt:]) if step > 0 else ""
                    res.text = (prompt + gen).rstrip() + " " + GUARDRAILED
                    res.n_new = step
                    return res
                if mode == "reroute" and not steering_on:
                    self.steer.set_deltas(self._steer_deltas(res.concept, alpha))
                    self.steer.enabled = True
                    steering_on = True
                    # re-forward with steering active to get steered logits for THIS step
                    out = self.model(input_ids=seq, output_hidden_states=True, use_cache=False)

            if step == max_new_tokens:
                break
            logits = out.logits[0, -1]
            nxt = int(torch.argmax(logits)) if greedy else int(torch.multinomial(torch.softmax(logits, -1), 1))
            seq = torch.cat([seq, torch.tensor([[nxt]], device=self.device)], dim=1)
            if self.tok.eos_token_id is not None and nxt == self.tok.eos_token_id:
                break

        self.steer.clear()
        res.text = self.tok.decode(seq[0])
        res.n_new = seq.shape[1] - n_prompt
        return res
