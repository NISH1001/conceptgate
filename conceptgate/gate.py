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
from enum import Enum

import numpy as np
import torch

from . import actions as A
from .concept import Concept
from .taps import TapForward


class LoadMode(Enum):
    """How much of the model to load into memory.

    FULL loads the whole model (can learn / check / generate). UP_TO_TAPS loads only
    embeddings + blocks 0..max(tap) via a truncated config, so the tail is never
    materialized -- least memory, but detect-only (no generation).
    """

    FULL = "full"
    UP_TO_TAPS = "up_to_taps"


@dataclass
class RunResult:
    text: str
    verdict: A.Verdict
    n_new: int = 0
    aborted: bool = False


class ConceptGate:
    def __init__(self, model, tok, layers: list[int], device: str = "cpu",
                 detect_only: bool = False):
        self.model = model.eval()
        self.tok = tok
        self.layers: list[int] = list(layers)
        self.device = device
        self.detect_only = detect_only           # loaded up-to-taps -> cannot generate
        self.concepts: dict[str, Concept] = {}   # public: name -> learned Concept
        self._taps = TapForward(model, self.layers)

    @classmethod
    def from_pretrained(cls, name: str, layers: list[int],
                        load: LoadMode | str = LoadMode.FULL,
                        device: str | None = None) -> ConceptGate:
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        load = LoadMode(load)   # normalize: accepts LoadMode or its string value
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        tok = AutoTokenizer.from_pretrained(name)
        if load is LoadMode.UP_TO_TAPS:
            # base model, truncated to max(tap)+1 blocks: the tail (later blocks, final
            # norm, lm_head) is never materialized. Read taps via hooks, never
            # output_hidden_states (which would ln_f the last block).
            model = AutoModel.from_pretrained(name, num_hidden_layers=max(layers) + 1)
            detect_only = True
        else:
            model = AutoModelForCausalLM.from_pretrained(name)
            detect_only = False
        model = model.to(device).eval()
        return cls(model, tok, layers, device, detect_only=detect_only)

    # ---- learn ----
    def learn(self, name: str, positives: list[str], negatives: list[str],
              batch_size: int = 1, **concept_kw) -> ConceptGate:
        """Few-shot: fit a Concept from prompt sets (last-token rep per prompt).

        batch_size is the memory<->compute dial for extraction: 1 runs one prompt at a
        time (least activation memory); larger stacks prompts into one padded forward
        (faster). The per-prompt forward is already vectorized -- batching amortizes
        per-forward overhead, mostly a GPU / many-prompts win.
        """
        read = lambda ps: self._taps.read(  # noqa: E731
            self.tok, list(ps), self.device, last_only=True, batch_size=batch_size)[0]
        self.concepts[name] = Concept(name=name, **concept_kw).fit(read(positives), read(negatives))
        return self

    # ---- calibrate (sets tau = the operating point on every concept) ----
    def calibrate(self, z: float = 3.0) -> ConceptGate:
        for c in self.concepts.values():
            c.calibrate_z(z)
        return self

    # ---- measure ----
    def _verdict(self, A_last: np.ndarray, step: int = 0) -> A.Verdict:
        """Fire if ANY concept fires; attribute to the highest-LLR (firing) concept."""
        if not self.concepts:
            return A.Verdict(fired=False, step=step)
        scored = {n: (float(c.llr(A_last)[0]), bool(c.fire(A_last)[0]))
                  for n, c in self.concepts.items()}
        firing = [n for n, (_, fired) in scored.items() if fired]
        name = max(firing or scored, key=lambda n: scored[n][0])
        return A.Verdict(fired=bool(firing), concept=name, score=scored[name][0], step=step)

    def check(self, prompt: str) -> A.Verdict:
        """Detection only, via truncated forward (runs only blocks 0..max(layers))."""
        A_last = self._taps.read(self.tok, [prompt], self.device, last_only=True)[0]
        return self._verdict(A_last, step=0)

    # ---- act ----
    def _context(self, v: A.Verdict, seq) -> A.FireContext:
        return A.FireContext(
            verdict=v, concept=self.concepts.get(v.concept), layers=self.layers,
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

        # reaching here means the prompt passed input-side and we would generate
        if self.detect_only:
            raise RuntimeError(
                "prompt passed input-side and this gate is detect-only "
                "(load=LoadMode.UP_TO_TAPS): it has no lm_head and cannot generate. "
                "Use check() for detection, or reload with load=LoadMode.FULL to generate."
            )

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

    # ---- lifecycle: use as a normal object (call unload() when done) or via `with` ----
    def unload(self) -> None:
        """Free the model weights. Learned concepts (tiny) are kept."""
        self.model = None
        self._taps = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> ConceptGate:
        return self

    def __exit__(self, *exc) -> None:
        self.unload()
