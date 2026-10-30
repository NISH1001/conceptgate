"""Rebuild behaviour_dose_acts__<tag>.npy (gitignored) from a stored behaviour-dose results file.

The activations are the unsteered last-token taps of each stored prompt, with the chat template, at the
taps and dtype the run recorded. No generation. Check the rebuild with
`analyze_behaviour_dose.py --stage 2 --no-save`: it must reproduce the stored stage-2 numbers.

Run:  uv run python scripts/rebuild_dose_acts.py --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from conceptgate import ConceptGate

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    tag = a.model.replace("/", "__")
    meta = json.load(open(os.path.join(HERE, f"behaviour_dose_results__{tag}.json")))
    dtype = str(meta["dtype"]).replace("torch.", "")
    cg = ConceptGate.from_pretrained(a.model, layers=list(meta["taps"]), device=a.device,
                                     chat_template=True, dtype=dtype)
    acts = [cg._taps.read(cg.tok, [cg._format(r["prompt"])], cg.device, last_only=True)[0][0]
            for r in meta["rows"]]
    out = os.path.join(HERE, f"behaviour_dose_acts__{tag}.npy")
    np.save(out, np.array(acts, dtype=np.float32))
    print(f"{len(acts)} prompts, taps {meta['taps']}, {dtype} -> {out}")


if __name__ == "__main__":
    main()
