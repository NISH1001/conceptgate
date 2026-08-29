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
from loguru import logger

from .actions import (
    ConceptAction,
    Continue,
    Decision,
    FireContext,
    ForceToken,
    InjectSteer,
    Stop,
    Verdict,
)
from .concept import Concept
from .hooks import SteeringHooks
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
    verdict: Verdict
    n_new: int = 0
    aborted: bool = False


class ConceptGate:
    def __init__(
        self,
        model,
        tok,
        layers: list[int],
        device: str = "cpu",
        detect_only: bool = False,
        debug: bool = False,
    ):
        self.model = model.eval()
        self.tok = tok
        self.layers: list[int] = list(layers)
        self.device = device
        self.detect_only = detect_only  # loaded up-to-taps -> cannot generate
        self.debug = debug
        if debug:
            logger.enable("conceptgate")  # loguru: turn on this package's debug logs
        self.concepts: dict[str, Concept] = {}  # public: name -> learned Concept
        self._taps = TapForward(model, self.layers)

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        layers: list[int],
        load: LoadMode | str = LoadMode.FULL,
        device: str | None = None,
        debug: bool = False,
    ) -> ConceptGate:
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        load = LoadMode(load)  # normalize: accepts LoadMode or its string value
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
        gate = cls(model, tok, layers, device, detect_only=detect_only, debug=debug)
        logger.debug(
            "loaded {} on {}: load={}, taps={}, blocks_run<={}, detect_only={}",
            name,
            device,
            load.value,
            list(layers),
            max(layers),
            detect_only,
        )
        return gate

    # ---- learn ----
    def learn(
        self,
        name: str,
        positives: list[str],
        negatives: list[str],
        batch_size: int = 1,
        **concept_kw,
    ) -> ConceptGate:
        """Few-shot: fit a Concept from prompt sets (last-token rep per prompt).

        batch_size is the memory<->compute dial for extraction: 1 runs one prompt at a
        time (least activation memory); larger stacks prompts into one padded forward
        (faster). The per-prompt forward is already vectorized -- batching amortizes
        per-forward overhead, mostly a GPU / many-prompts win.
        """
        read = lambda ps: self._taps.read(  # noqa: E731
            self.tok, list(ps), self.device, last_only=True, batch_size=batch_size
        )[0]
        c = Concept(name=name, **concept_kw).fit(read(positives), read(negatives))
        self.concepts[name] = c
        logger.debug(
            "learn {!r}: {}+{} prompts, J=({},{})",
            name,
            len(positives),
            len(negatives),
            c.gmm_pos.n_components,
            c.gmm_neg.n_components,
        )
        return self

    # ---- calibrate (sets tau = the operating point on every concept) ----
    def calibrate(self, z: float = 3.0, margin: float = 0.0) -> ConceptGate:
        for c in self.concepts.values():
            c.calibrate_z(z, margin=margin)
        logger.debug(
            "calibrate z={} margin={}: tau={}",
            z,
            margin,
            {n: round(c.tau, 2) for n, c in self.concepts.items()},
        )
        return self

    # ---- measure ----
    def _verdict(self, A_last: np.ndarray, step: int = 0) -> Verdict:
        """Fire if ANY concept fires; attribute to the highest-LLR (firing) concept.

        Also reports uncertainty on the attributed concept: p_present (calibrated) and an
        abstain flag (score inside the concept's unsure band around tau)."""
        resid_norm = float(np.linalg.norm(A_last[0], axis=1).mean())  # per-tap L2, averaged
        if not self.concepts:
            return Verdict(fired=False, step=step, resid_norm=resid_norm)
        scored = {
            n: (float(c.llr(A_last)[0]), int(c.decide(A_last)[0]))
            for n, c in self.concepts.items()
        }
        firing = [n for n, (_, d) in scored.items() if d > 0]
        name = max(firing or scored, key=lambda n: scored[n][0])
        c = self.concepts[name]
        llr, dec = scored[name]
        v = Verdict(
            fired=bool(firing),
            concept=name,
            score=llr,
            step=step,
            tau=float(c.tau),
            margin=float(llr - c.tau),
            p_present=float(c.p_present(A_last)[0]),
            abstained=(not firing) and dec == 0,
            resid_norm=resid_norm,
        )
        logger.debug(
            "verdict@{}: {} -> fired={} concept={} p={:.2f} abstain={}",
            step,
            {n: (round(s, 2), d) for n, (s, d) in scored.items()},
            v.fired,
            v.concept,
            round(v.p_present, 2),
            v.abstained,
        )
        return v

    def check(self, prompt: str) -> Verdict:
        """Detection only, via truncated forward (runs only blocks 0..max(layers))."""
        A_last = self._taps.read(self.tok, [prompt], self.device, last_only=True)[0]
        return self._verdict(A_last, step=0)

    # ---- act ----
    def _context(self, v: Verdict, seq) -> FireContext:
        return FireContext(
            verdict=v,
            concept=self.concepts.get(v.concept),
            layers=self.layers,
            concepts=self.concepts,
            tok=self.tok,
            seq=seq,
            step=v.step,
        )

    def _dispatch(self, action: ConceptAction, v: Verdict, seq) -> Decision:
        """Hand the verdict to the action; return its Decision (Stop / Continue / InjectSteer)."""
        d = action.decide(self._context(v, seq))
        if isinstance(d, (Stop, Continue, InjectSteer, ForceToken)):
            return d
        raise NotImplementedError(f"{type(d).__name__} decisions land in a later round")

    def _steer_hooks(self, deltas: dict) -> SteeringHooks:
        """Install forward hooks that add per-layer steering vectors to the residual stream."""
        h = SteeringHooks(self.model).register(self.layers)
        h.set_deltas({
            int(k): torch.as_tensor(np.asarray(vec, dtype=np.float32), device=self.device)
            for k, vec in deltas.items()
        })
        h.enabled = True
        return h

    def _last_act(self, hidden_states) -> np.ndarray:
        chosen = [hidden_states[L + 1][0, -1] for L in self.layers]  # each [d]
        return torch.stack(chosen, dim=0).float().cpu().numpy()[None, ...]  # [1, m, d]

    @torch.no_grad()
    def run(
        self,
        prompt: str,
        action: ConceptAction,
        max_new_tokens: int = 20,
        check_output: bool = True,
    ) -> RunResult:
        """Drive M under an action. The action is asked on the input verdict (via a cheap
        truncated check) and returns a Decision: Stop -> halt + marker (the full model is
        skipped), InjectSteer -> steer the whole generation, Continue -> generate normally.
        With check_output the completion is monitored per token (the manual loop); otherwise
        (and always when steering) generation uses the fast KV-cached path."""
        if not isinstance(action, ConceptAction):
            raise TypeError(
                f"action must be a ConceptAction (have a .decide(ctx) method); "
                f"got {type(action).__name__}"
            )
        ids = self.tok(prompt, return_tensors="pt").to(self.device).input_ids

        # ask the action on the input verdict (cheap truncated check); it owns when + what
        v = self.check(prompt)
        d = self._dispatch(action, v, ids)
        if isinstance(d, Stop):
            text = prompt + (" " + d.emit if d.emit else "")
            return RunResult(text=text.rstrip(), verdict=v, n_new=0, aborted=True)
        steer = self._steer_hooks(d.deltas) if isinstance(d, InjectSteer) and d.deltas else None
        forced = list(d.token_ids) if isinstance(d, ForceToken) else []

        if self.detect_only:
            raise RuntimeError(
                "this gate is detect-only (load=LoadMode.UP_TO_TAPS): it has no lm_head and "
                "cannot generate. Use check() for detection, or reload with load=LoadMode.FULL."
            )

        n_prompt = ids.shape[1]
        if forced:  # seed the completion; M continues FROM the forced tokens
            ids = torch.cat([ids, torch.tensor([forced], device=self.device)], dim=1)
        try:
            # fast path: KV-cached generate (steering hooks nudge every token; a forced prefix
            # is already appended to `ids`, so M just continues from it)
            if steer is not None or forced or not check_output:
                out = self.model.generate(
                    ids, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tok.eos_token_id,
                )
                return RunResult(text=self.tok.decode(out[0]), verdict=v,
                                 n_new=out.shape[1] - n_prompt)
            # monitored path: manual loop with an output-side check per token
            seq = ids
            for step in range(1, max_new_tokens + 1):
                out = self.model(input_ids=seq, output_hidden_states=True, use_cache=False)
                vo = self._verdict(self._last_act(out.hidden_states), step=step)
                if vo.fired or vo.abstained:
                    do = self._dispatch(action, vo, seq)
                    if isinstance(do, Stop):
                        gen = self.tok.decode(seq[0, n_prompt:])
                        text = (prompt + gen).rstrip() + (" " + do.emit if do.emit else "")
                        return RunResult(text=text, verdict=vo, n_new=step - 1, aborted=True)
                nxt = int(torch.argmax(out.logits[0, -1]))
                seq = torch.cat([seq, torch.tensor([[nxt]], device=self.device)], dim=1)
                if self.tok.eos_token_id is not None and nxt == self.tok.eos_token_id:
                    break
            return RunResult(text=self.tok.decode(seq[0]), verdict=v,
                             n_new=seq.shape[1] - n_prompt)
        finally:
            if steer is not None:
                steer.remove()

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
