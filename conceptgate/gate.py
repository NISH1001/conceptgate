"""ConceptGate: the public facade.

Attach to a frozen transformer, learn concepts few-shot, MEASURE cheaply (truncated
forward), and ACT via injected strategies. ConceptGate measures and orchestrates; it holds
no action logic (strategies live in actions.py) and no density math (concept.py / mixture.py).
Firing a concept is ``llr > tau`` where tau is a calibrated operating point.

    cg = ConceptGate.from_pretrained("gpt2", layers=[4, 6, 8])
    cg.learn("weapons", positives=[...], negatives=[...])
    cg.calibrate(z=3.0)
    cg.check(prompt)                    # Verdict, truncated forward
    cg.run(prompt, action=Abort())      # strategy decides; cg drives + executes
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from . import actions as A
from .concept import Concept, ConceptBank
from .taps import TapForward


@dataclass
class RunResult:
    text: str
    verdict: A.Verdict
    n_new: int = 0
    aborted: bool = False


class ConceptGate:
    def __init__(self, model, tok, layers: List[int], device: str = "cpu"):
        self.model = model.eval()
        self.tok = tok
        self.layers = list(layers)
        self.device = device
        self.bank = ConceptBank()
        self._concepts: Dict[str, Concept] = {}
        self._taps = TapForward(model, self.layers)

    @classmethod
    def from_pretrained(cls, name: str, layers: List[int], device: str = None) -> "ConceptGate":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
        return cls(model, tok, layers, device)

    # ---- learn ----
    def learn(self, name: str, positives, negatives, **concept_kw) -> "ConceptGate":
        """Few-shot: fit a Concept from prompt sets (last-token rep per prompt)."""
        A_pos = self._taps.read(self.tok, list(positives), self.device, last_only=True)[0]
        A_neg = self._taps.read(self.tok, list(negatives), self.device, last_only=True)[0]
        c = Concept(name=name, **concept_kw).fit(A_pos, A_neg)
        self._concepts[name] = c
        self.bank.add(c)
        return self

    # ---- calibrate (sets tau = the operating point on every concept) ----
    def calibrate(self, z: float = 3.0) -> "ConceptGate":
        for c in self.bank.gates:
            c.calibrate_z(z)
        return self

    # ---- measure ----
    def _verdict(self, A_last: np.ndarray, step: int = 0) -> A.Verdict:
        if not self.bank.gates:
            return A.Verdict(fired=False, step=step)
        idx = int(self.bank.which(A_last)[0])
        c = self.bank.gates[idx]
        return A.Verdict(
            fired=bool(self.bank.fire(A_last)[0]),
            concept=c.name,
            score=float(c.llr(A_last)[0]),
            step=step,
        )

    def check(self, prompt: str) -> A.Verdict:
        """Detection only, via truncated forward (runs only blocks 0..max(layers))."""
        A_last = self._taps.read(self.tok, [prompt], self.device, last_only=True)[0]
        return self._verdict(A_last, step=0)

    # ---- act ----
    def _context(self, v: A.Verdict, seq) -> A.FireContext:
        return A.FireContext(
            verdict=v, concept=self._concepts.get(v.concept), layers=self.layers,
            tok=self.tok, seq=seq, step=v.step,
        )

    def _decide(self, action, v: A.Verdict, seq):
        """Ask the action; return a Decision. Only Stop/Continue are wired this round."""
        d = action.on_fire(self._context(v, seq))
        if isinstance(d, (A.Stop, A.Continue)):
            return d
        raise NotImplementedError(f"{type(d).__name__} decisions land in a later round")

    def _last_act(self, hidden_states) -> np.ndarray:
        chosen = [hidden_states[L + 1][0, -1] for L in self.layers]   # each [d]
        return torch.stack(chosen, dim=0).float().cpu().numpy()[None, ...]   # [1, m, d]

    @torch.no_grad()
    def run(self, prompt: str, action, max_new_tokens: int = 20,
            check_output: bool = True) -> RunResult:
        """Drive M under an action. Input-side check is truncated (cheap); if it fires and
        the action stops, M's full forward and generation are never run -- the compute win.
        Otherwise generate (full forward) with per-token output-side checks."""
        ids = self.tok(prompt, return_tensors="pt").to(self.device).input_ids

        # input-side: cheap truncated check; abort here skips the full model entirely
        v = self.check(prompt)
        if v.fired:
            d = self._decide(action, v, ids)
            if isinstance(d, A.Stop):
                text = prompt + (" " + d.emit if d.emit else "")
                return RunResult(text=text.rstrip(), verdict=v, n_new=0, aborted=True)

        # passed input-side -> generate with full forward, output-side checks per token
        seq = ids
        n_prompt = ids.shape[1]
        for step in range(1, max_new_tokens + 1):
            out = self.model(input_ids=seq, output_hidden_states=True, use_cache=False)
            if check_output:
                vo = self._verdict(self._last_act(out.hidden_states), step=step)
                if vo.fired:
                    d = self._decide(action, vo, seq)
                    if isinstance(d, A.Stop):
                        gen = self.tok.decode(seq[0, n_prompt:])
                        text = (prompt + gen).rstrip() + (" " + d.emit if d.emit else "")
                        return RunResult(text=text, verdict=vo, n_new=step - 1, aborted=True)
            nxt = int(torch.argmax(out.logits[0, -1]))
            seq = torch.cat([seq, torch.tensor([[nxt]], device=self.device)], dim=1)
            if self.tok.eos_token_id is not None and nxt == self.tok.eos_token_id:
                break
        return RunResult(text=self.tok.decode(seq[0]), verdict=v, n_new=seq.shape[1] - n_prompt)
