"""Steering measurement: does adding ±alpha·w_c to the residual stream move generation toward/away a
concept, and at what cost to fluency? Sweeps the steering fraction and scores each greedy generation
three ways -- the concept's own detector LLR (internal effectiveness), a concept-keyword lexicon
hit-rate (independent semantic shift), and perplexity under the base model (fluency) -- turning the
demoed steering (report Fig 5) into a measured dose-response with an effective window.

Run:  uv run python scripts/eval_steering.py                       # Qwen-0.5B + gpt2
      uv run python scripts/eval_steering.py --quick               # Qwen, 1 concept, coarse (smoke)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F

from conceptgate import ConceptGate, Steer, Trigger

# neutral negatives shared by every concept (topic-free everyday sentences)
NEG_TOPIC = [
    "The meeting is scheduled for Monday morning.", "She wrote a letter to her friend.",
    "The bank approved the loan yesterday.", "He parked the car outside the office.",
    "The quarterly report is due next week.", "They watched a long movie last night.",
]

# topical concepts: few-shot positives + an independent keyword lexicon (stem-matched)
CONCEPTS = {
    "food": {
        "pos": ["I love cooking fresh pasta for dinner.", "This recipe needs garlic and basil.",
                "The restaurant served a delicious soup.", "She baked warm bread in the kitchen.",
                "We shared a tasty meal together.", "Add a pinch of salt to the sauce."],
        "lex": ["food", "eat", "ate", "cook", "meal", "recipe", "kitchen", "dinner", "lunch",
                "breakfast", "bread", "sauce", "tast", "delicio", "pasta", "soup", "salt", "garlic",
                "cheese", "veget", "fruit", "dish", "restaurant", "chef", "bake", "flavor", "sweet",
                "spice", "snack", "drink", "coffee", "tea", "hungry", "plate", "dessert"],
    },
    "nature": {
        "pos": ["We hiked through the quiet green forest.", "The river flowed past the mountains.",
                "Birds were singing in the tall trees.", "The trail led up to a waterfall.",
                "Wildflowers bloomed across the meadow.", "The sunset glowed over the ocean."],
        "lex": ["forest", "tree", "river", "mountain", "bird", "trail", "waterfall", "flower",
                "meadow", "sunset", "ocean", "sea", "wild", "leaf", "leaves", "sky", "lake",
                "valley", "hill", "garden", "grass", "cliff", "beach", "wood", "nature", "petal",
                "stream", "canyon", "meadow", "sunrise", "moss", "fern"],
    },
    "technology": {
        "pos": ["I upgraded my laptop with a faster processor.", "The new app syncs data across devices.",
                "She wrote code to automate the task.", "The robot navigated using its sensors.",
                "We debugged the software all afternoon.", "The phone screen is bright and sharp."],
        "lex": ["comput", "software", "code", "coding", "program", "internet", "app", "device",
                "phone", "screen", "data", "digital", "algorithm", "machine", "robot", "tech",
                "network", "server", "laptop", "keyboard", "chip", "electron", "online", "website",
                "download", "gadget", "processor", "sensor", "debug", "pixel", "battery"],
    },
}

PROMPTS = [
    "The best part of the day was when",
    "I opened the door and",
    "She looked around and saw",
    "Later that afternoon, we",
    "The first thing I noticed was",
]
# magnitude as a fraction of the residual norm; ~0.03-0.10 is the coherent range, higher breaks it
FRACTIONS = [-0.16, -0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10, 0.16]
MAX_NEW = 40


def _words(text):
    return [w.strip(".,!?;:'\"()[]").lower() for w in text.split() if w.strip(".,!?;:'\"()[]")]


def _lex_hit(text, lex):
    """Fraction of content words that stem-match a concept keyword -- an independent semantic signal
    (does not use the steering direction, so it is not circular with the concept LLR)."""
    ws = _words(text)
    if not ws:
        return 0.0
    hits = sum(1 for w in ws if any(w.startswith(t) or t in w for t in lex))
    return hits / len(ws)


def _ppl(cg, prompt, cont):
    """Perplexity of the continuation given the prompt, under the (unsteered) base model."""
    if not cont.strip():
        return float("nan")
    ids = cg.tok(prompt + " " + cont, return_tensors="pt").input_ids.to(cg.device)
    plen = cg.tok(prompt, return_tensors="pt").input_ids.shape[1]
    if ids.shape[1] <= plen + 1:
        return float("nan")
    with torch.no_grad():
        logits = cg.model(input_ids=ids).logits
    sl = logits[:, plen - 1:-1, :].reshape(-1, logits.shape[-1])
    lb = ids[:, plen:].reshape(-1)
    loss = F.cross_entropy(sl, lb)
    return float(math.exp(min(loss.item(), 20)))


def _read(cg, texts):
    return cg._taps.read(cg.tok, texts, cg.device, last_only=True)[0]


def steering_eval(model, taps, device, fractions=FRACTIONS, max_new=MAX_NEW, concepts=None):
    concepts = concepts or list(CONCEPTS)
    print(f"\n### STEER {model}  taps {'/'.join(map(str, taps))}  concepts {concepts}  "
          f"frac {fractions[0]}..{fractions[-1]}  max_new {max_new}", flush=True)
    cg = ConceptGate.from_pretrained(model, layers=taps, device=device)
    for nm in concepts:
        cg.learn(nm, CONCEPTS[nm]["pos"], NEG_TOPIC)
    resid = round(float(cg.check(PROMPTS[0]).resid_norm), 1)

    out = {}
    for nm in concepts:
        c = cg.concepts[nm]
        lex = CONCEPTS[nm]["lex"]
        curve = {"frac": fractions, "llr": [], "lex": [], "ppl": [], "example": {}}
        for f in fractions:
            llrs, lexs, ppls, ex = [], [], [], ""
            for pi, p in enumerate(PROMPTS):
                r = cg.run(p, action=Steer(fraction=f, concept=nm, when=Trigger.ALWAYS),
                           max_new_tokens=max_new, check_output=False)
                cont = r.continuation.strip().replace("\n", " ")   # completion stems: no chat template by design
                if pi == 0:
                    ex = cont[:130]
                if cont:
                    llrs.append(float(c.llr(_read(cg, [cont]))[0]))
                    lexs.append(_lex_hit(cont, lex))
                    ppls.append(_ppl(cg, p, cont))
            curve["llr"].append(round(float(np.mean(llrs)), 2) if llrs else None)
            curve["lex"].append(round(float(np.nanmean(lexs)) * 100, 1) if lexs else None)
            curve["ppl"].append(round(float(np.nanmean(ppls)), 1) if ppls else None)
            curve["example"][str(f)] = ex
            print(f"  {nm:11s} {f:+.2f}: llr {curve['llr'][-1]:+7.1f}  lex {curve['lex'][-1]:5.1f}%  "
                  f"ppl {curve['ppl'][-1]:7.1f}   | {ex[:56]}", flush=True)
        out[nm] = curve
    cg.unload()
    return {"model": model, "taps": taps, "resid_norm": resid, "prompts": PROMPTS,
            "max_new_tokens": max_new, "fractions": fractions, "concepts": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen/Qwen2.5-0.5B-Instruct,gpt2")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--out", default="scripts/eval_steering_results.json")
    ap.add_argument("--quick", action="store_true", help="Qwen only, one concept, coarse sweep (smoke)")
    a = ap.parse_args()

    taps_by = {"gpt2": [4, 6, 8], "Qwen/Qwen2.5-0.5B-Instruct": [8, 12, 16],
               "google/gemma-2-2b-it": [9, 13, 17]}
    models = [m.strip() for m in a.models.split(",")]
    fracs, concepts = FRACTIONS, None
    if a.quick:
        models, fracs, concepts = ["Qwen/Qwen2.5-0.5B-Instruct"], [-0.10, 0.0, 0.10], ["food"]

    t0 = time.time()
    results = []
    for m in models:
        taps = taps_by.get(m, [8, 12, 16])
        try:
            results.append(steering_eval(m, taps, a.device, fractions=fracs,
                                          max_new=a.max_new, concepts=concepts))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  SKIP {m}: {type(e).__name__}: {e}", flush=True)
        with open(a.out, "w") as f:
            json.dump({"results": results}, f, indent=1)
    print(f"\ndone in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
