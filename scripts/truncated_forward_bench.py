"""Benchmark: truncated forward vs full forward on GPT-2.

Shows the compute saving from running only blocks 0..max(tap). HONEST CAVEAT: the saving is
for DETECTION / ABORT -- you decide from the middle and stop. Generating a response still
needs the full forward per token, so it does not benefit.

Run:  uv run python scripts/truncated_forward_bench.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from conceptgate import data as D  # noqa: E402
from conceptgate.hooks import num_layers  # noqa: E402
from conceptgate.taps import TapForward  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _median_ms(fn, reps=5) -> float:
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000.0


def main() -> int:
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    concept = D.load_concept(os.path.join(ROOT, "data/concepts/weapons.json"))
    layers = concept["layers_gpt2"]
    prompts = concept["positives"] + concept["negatives"]
    total, max_tap = num_layers(model), max(layers)

    tf = TapForward(model, layers)
    full = lambda: D.extract_token_activations(model, tok, prompts, layers, "cpu", last_only=True)  # noqa: E731
    trunc = lambda: tf.read(tok, prompts, "cpu", last_only=True)  # noqa: E731

    A_full, A_tr = full()[0], trunc()[0]
    equiv = bool(np.allclose(A_full, A_tr, rtol=1e-4, atol=1e-4))

    full(); trunc()  # warm up
    t_full, t_trunc = _median_ms(full), _median_ms(trunc)

    skipped = total - (max_tap + 1)
    print(f"model=gpt2  blocks={total}  taps={layers}  (deepest tap = block {max_tap})")
    print(f"blocks run: {max_tap + 1}/{total}   ({100 * skipped / total:.0f}% of blocks skipped)")
    print(f"wall-clock over {len(prompts)} prompts (median of 5):")
    print(f"   full forward:      {t_full:6.1f} ms")
    print(f"   truncated forward: {t_trunc:6.1f} ms   ({100 * (1 - t_trunc / t_full):.0f}% faster)")
    print(f"activations identical at the taps: {equiv}")

    print("\nextrapolation (arithmetic, not measured) — blocks run if you tap up to block T:")
    for depth, tap in [(12, 8), (24, 8), (40, 12), (80, 20)]:
        print(f"   {depth:>2}-block model, tap up to {tap:>2}: run {tap + 1}/{depth} "
              f"= {100 * (tap + 1) / depth:.0f}% of blocks ({100 * (depth - tap - 1) / depth:.0f}% skipped)")

    print("\ncaveat: the saving is for DETECT/ABORT; generating a response still needs the full forward.")
    ok = equiv and t_trunc <= t_full
    print("BENCH:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
