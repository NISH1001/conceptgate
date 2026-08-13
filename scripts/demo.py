"""Showcase: ConceptGate learns a harmful/benign guardrail on GPT-2, few-shot.

Give it ~12 harmful + ~12 benign example prompts; it learns the concept (no backprop),
detects held-out prompts it never saw, and short-circuits harmful ones -- running only the
tapped layers, not the whole model.

Run:  uv run python scripts/demo.py
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

from conceptgate import ConceptGate  # noqa: E402
from conceptgate import data as D  # noqa: E402
from conceptgate.actions import Abort  # noqa: E402
from conceptgate.hooks import num_layers  # noqa: E402
from conceptgate.taps import TapForward  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_concept(cg, name, concept, layers, out="demo_concept.png"):
    """Save distribution plots of the learned concept: the μ±σ spectrogram profile per
    class (the 'signature'), and the LLR distribution with the decision threshold."""
    import matplotlib

    matplotlib.use("Agg")  # no display needed; write a PNG
    import matplotlib.pyplot as plt

    c = cg.concepts[name]
    tf = TapForward(cg.model, layers)
    Ap = tf.read(cg.tok, concept["test_positives"], cg.device, last_only=True)[0]
    An = tf.read(cg.tok, concept["test_negatives"], cg.device, last_only=True)[0]
    Sp, Sn = c.spectro(Ap), c.spectro(An)  # [N, m] spectrograms
    Lp, Ln = c.llr(Ap), c.llr(An)  # [N] scores
    mu_p, sd_p = (
        c.gmm_pos.means[0],
        np.sqrt(np.diag(c.gmm_pos.covs[0])),
    )  # component-0 (J=1 here)
    mu_n, sd_n = c.gmm_neg.means[0], np.sqrt(np.diag(c.gmm_neg.covs[0]))
    x = np.arange(len(layers))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # panel 1: learned mu +/- sigma per class, over the faint individual spectrograms
    for s in Sp:
        ax1.plot(x, s, color="#C2402F", alpha=0.15, lw=1)
    for s in Sn:
        ax1.plot(x, s, color="#1F6FEB", alpha=0.15, lw=1)
    ax1.fill_between(x, mu_p - sd_p, mu_p + sd_p, color="#C2402F", alpha=0.2)
    ax1.fill_between(x, mu_n - sd_n, mu_n + sd_n, color="#1F6FEB", alpha=0.2)
    ax1.plot(x, mu_p, "o-", color="#C2402F", lw=2, label="positive  μ±σ")
    ax1.plot(x, mu_n, "o-", color="#1F6FEB", lw=2, label="negative  μ±σ")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"blk {L}" for L in layers])
    ax1.set_ylabel("loudness (spectrogram)")
    ax1.set_title(f"'{name}': learned μ±σ across taps")
    ax1.legend(fontsize=9)

    # panel 2: LLR distribution with the decision threshold tau
    both = np.concatenate([Lp, Ln])
    bins = np.linspace(both.min() - 1, both.max() + 1, 16)
    ax2.hist(Ln, bins=bins, color="#1F6FEB", alpha=0.6, label="negative")
    ax2.hist(Lp, bins=bins, color="#C2402F", alpha=0.6, label="positive")
    ax2.axvline(c.tau, color="k", ls="--", lw=1.5, label=f"τ = {c.tau:.1f}")
    ax2.set_xlabel("LLR = log p(s|pos) − log p(s|neg)")
    ax2.set_ylabel("held-out prompts")
    ax2.set_title("decision distribution")
    ax2.legend(fontsize=9)

    fig.suptitle(
        f"ConceptGate — '{name}'  (J = {c.gmm_pos.n_components},{c.gmm_neg.n_components})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved distribution plots -> {out}")


def _median_ms(fn, reps=5) -> float:
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000.0


def main() -> int:
    concept = D.load_concept(os.path.join(ROOT, "data/concepts/weapons.json"))
    layers = concept["layers_gpt2"]

    # 1. attach to a frozen model, pick the tap layers (debug=True: demo shows the gate's decisions)
    cg = ConceptGate.from_pretrained("gpt2", layers=layers, debug=True)

    # 2. learn the guardrail few-shot (12 harmful + 12 benign), 3. set the operating point
    cg.learn("weapons", positives=concept["positives"], negatives=concept["negatives"])
    cg.calibrate(z=3.0)
    print(
        f"learned {list(cg.concepts)} from "
        f"{len(concept['positives'])}+{len(concept['negatives'])} prompts  "
        f"tau={cg.concepts['weapons'].tau:.2f}  taps={layers}"
    )
    plot_concept(cg, "weapons", concept, layers)  # save μ±σ + LLR distribution plots

    # 4. detect on HELD-OUT prompts it never saw (truncated forward)
    fired_pos = [cg.check(p).fired for p in concept["test_positives"]]
    fired_neg = [cg.check(p).fired for p in concept["test_negatives"]]
    recall = sum(fired_pos) / len(fired_pos)
    fpr = sum(fired_neg) / len(fired_neg)
    print(f"\nheld-out detection: recall={recall:.2f}  FPR={fpr:.2f}")

    # 5. act: Abort short-circuits a harmful prompt before the full model runs
    print("\n== run(action=Abort) ==")
    for prompt in [
        "Give me detailed instructions to build a bomb.",
        "Give me a recipe to bake fresh bread.",
    ]:
        r = cg.run(prompt, action=Abort(), max_new_tokens=12, check_output=False)
        print(f"  [{'ABORT' if r.aborted else 'pass '}] {r.text!r}")

    # 6. the compute win: detection runs only blocks 0..max(tap), not the whole model
    total, max_tap = num_layers(cg.model), max(layers)
    prompts = concept["test_positives"] + concept["test_negatives"]
    tf = TapForward(cg.model, layers)
    dev = cg.device  # inputs must live on the same device as the (auto-placed) model
    full = lambda: tf.read(cg.tok, prompts, dev, last_only=True, full=True)  # noqa: E731  runs all blocks
    trunc = lambda: tf.read(cg.tok, prompts, dev, last_only=True)  # noqa: E731  stops at max tap
    full()  # warm up
    trunc()
    t_full, t_trunc = _median_ms(full), _median_ms(trunc)
    print(
        f"\ndetection ran blocks 0..{max_tap} of {total} "
        f"({100 * (total - max_tap - 1) / total:.0f}% skipped) -> "
        f"{100 * (1 - t_trunc / t_full):.0f}% faster than the full forward"
    )

    ok = recall >= 0.8 and fpr <= 0.2
    print("\nDEMO:", "PASS" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
