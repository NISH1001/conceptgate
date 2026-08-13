"""Interactive ConceptGate demo (marimo).

Reactive widgets over a learned concept: drag the calibration slider to watch the
threshold move across the LLR distribution, and type a prompt to see where it lands.

Run:  uv run marimo edit scripts/demo_marimo.py     # editable
  or  uv run marimo run  scripts/demo_marimo.py     # app view
"""
import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, os, plt


@app.cell
def _(os):
    # loads once (no reactive deps) — model + few-shot concept
    from conceptgate import ConceptGate
    from conceptgate import data as D
    from conceptgate.taps import TapForward

    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        root = os.getcwd()
    concept = D.load_concept(os.path.join(root, "data/concepts/weapons.json"))
    layers = concept["layers_gpt2"]

    cg = ConceptGate.from_pretrained("gpt2", layers=layers)
    cg.learn("weapons", concept["positives"], concept["negatives"])
    tf = TapForward(cg.model, layers)
    return cg, concept, layers, tf


@app.cell
def _(mo):
    z = mo.ui.slider(1.0, 6.0, value=3.0, step=0.5, label="calibration z (strictness)")
    prompt = mo.ui.text(
        value="What's the best way to sear a steak?",
        label="test a prompt",
        full_width=True,
    )
    mo.vstack([z, prompt])
    return prompt, z


@app.cell
def _(cg, concept, layers, mo, np, plt, prompt, tf, z):
    cg.calibrate(z=z.value)                         # reactive: re-runs when the slider moves
    c = cg.concepts["weapons"]

    Ap = tf.read(cg.tok, concept["test_positives"], cg.device, last_only=True)[0]
    An = tf.read(cg.tok, concept["test_negatives"], cg.device, last_only=True)[0]
    Lp, Ln = c.llr(Ap), c.llr(An)
    v = cg.check(prompt.value)                      # reactive: re-runs when the text changes

    x = np.arange(len(layers))
    mu_p, sd_p = c.gmm_pos.means[0], np.sqrt(np.diag(c.gmm_pos.covs[0]))
    mu_n, sd_n = c.gmm_neg.means[0], np.sqrt(np.diag(c.gmm_neg.covs[0]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.fill_between(x, mu_p - sd_p, mu_p + sd_p, color="#C2402F", alpha=0.2)
    ax1.fill_between(x, mu_n - sd_n, mu_n + sd_n, color="#1F6FEB", alpha=0.2)
    ax1.plot(x, mu_p, "o-", color="#C2402F", lw=2, label="positive μ±σ")
    ax1.plot(x, mu_n, "o-", color="#1F6FEB", lw=2, label="negative μ±σ")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"blk {L}" for L in layers])
    ax1.set_ylabel("loudness (spectrogram)")
    ax1.set_title("learned μ±σ across taps")
    ax1.legend(fontsize=8)

    both = np.concatenate([Lp, Ln, [v.score]])
    bins = np.linspace(both.min() - 1, both.max() + 1, 16)
    ax2.hist(Ln, bins=bins, color="#1F6FEB", alpha=0.6, label="negative")
    ax2.hist(Lp, bins=bins, color="#C2402F", alpha=0.6, label="positive")
    ax2.axvline(c.tau, color="k", ls="--", lw=1.5, label=f"τ = {c.tau:.1f}")
    ax2.axvline(v.score, color="green", lw=2, label=f"your prompt ({v.score:.1f})")
    ax2.set_xlabel("LLR = log p(s|pos) − log p(s|neg)")
    ax2.set_title("decision distribution")
    ax2.legend(fontsize=8)
    fig.tight_layout()

    verdict = mo.md(
        f"**{prompt.value!r}** → {'🔴 **FIRED**' if v.fired else '🟢 passed'}  ·  "
        f"score = **{v.score:.2f}**  ·  τ = {c.tau:.2f}"
    )
    mo.vstack([verdict, fig])
    return


if __name__ == "__main__":
    app.run()
