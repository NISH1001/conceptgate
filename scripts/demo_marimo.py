"""ConceptGate — interactive walkthrough (marimo).

An educational, reactive demo: pick a concept, watch it learn from a handful of examples,
calibrate the threshold, and test your own prompts — with live μ±σ, decision, and a
per-prompt loudness heatmap.

Run:  uv run marimo edit scripts/demo_marimo.py     # editable
  or  uv run marimo run  scripts/demo_marimo.py     # app view
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import matplotlib.pyplot as plt
    import numpy as np

    import marimo as mo

    return alt, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # ConceptGate — an interactive walkthrough

    **ConceptGate** attaches to a *frozen* language model and learns to recognize a
    **concept** from just a handful of examples — no fine-tuning, no gradients.

    It works by *listening*: at a few chosen layers it taps the model's residual stream
    (the vector the model passes forward as it "thinks"), and measures how strongly each
    example resonates with a direction learned as **mean(positive) − mean(negative)**.
    That gives a **loudness** per layer — a *spectrogram* of the concept across depth —
    which two small Gaussian mixtures (positive vs. negative) turn into a calibrated
    score. A concept is detected when the score crosses a threshold **τ**.

    It's concept-agnostic: the same machinery learns *cooking*, *travel*, or a safety
    guardrail. Pick one below and follow the four steps.
    """)
    return


@app.cell
def _():
    # loads once (no reactive deps): a frozen GPT-2 + the tap layers
    from conceptgate import ConceptGate
    from conceptgate.actions import Abort
    from conceptgate.taps import TapForward

    LAYERS = [4, 5, 6, 7, 8]
    cg = ConceptGate.from_pretrained("gpt2", layers=LAYERS)
    tf = TapForward(cg.model, LAYERS)
    return Abort, cg, tf


@app.cell
def _():
    # a few benign example concepts (positive = concept present, negative = concept absent)
    PRESETS = {
        "cooking": {
            "positives": [
                "How long do I boil pasta?",
                "What temperature should I bake sourdough at?",
                "How do I dice an onion without crying?",
                "What can I substitute for buttermilk?",
                "How do I know when a steak is medium-rare?",
                "Best way to caramelize onions slowly?",
                "How much salt for a pot of pasta water?",
            ],
            "negatives": [
                "What is the capital of France?",
                "How do I center a div in CSS?",
                "Who won the 2018 World Cup?",
                "Explain recursion in one sentence.",
                "What's the exchange rate for the yen?",
                "How does a car engine work?",
                "Summarize the plot of Hamlet.",
            ],
        },
        "travel": {
            "positives": [
                "Cheapest month to fly to Tokyo?",
                "Do I need a visa to visit Vietnam?",
                "Best neighborhood to stay in Lisbon?",
                "How early should I get to the airport?",
                "Is the train from Rome to Florence scenic?",
                "What's a good carry-on packing list?",
                "Which side of the road do they drive on in Ireland?",
            ],
            "negatives": [
                "How do I refactor this Python function?",
                "What's a healthy breakfast recipe?",
                "Explain how photosynthesis works.",
                "Who painted the Mona Lisa?",
                "What's the derivative of sin(x)?",
                "How do I set up a git remote?",
                "Recommend a good sci-fi novel.",
            ],
        },
        "programming": {
            "positives": [
                "How do I reverse a list in Python?",
                "What's the difference between a tuple and a list?",
                "How do I set up a virtual environment?",
                "Why is my for-loop off by one?",
                "How do I catch an exception in Java?",
                "What does 'git rebase' actually do?",
                "How do I make an HTTP request in JavaScript?",
            ],
            "negatives": [
                "What's the best way to sear a steak?",
                "Cheapest month to fly to Tokyo?",
                "Who won the 2018 World Cup?",
                "Explain how photosynthesis works.",
                "What's the capital of Australia?",
                "Recommend a good hike near Seattle.",
                "How do I train a puppy to sit?",
            ],
        },
    }
    return (PRESETS,)


@app.cell
def _(PRESETS, mo):
    concept_pick = mo.ui.dropdown(
        options=list(PRESETS), value="cooking", label="**1. Pick a concept**"
    )
    concept_pick
    return (concept_pick,)


@app.cell
def _(PRESETS, concept_pick, mo):
    data = PRESETS[concept_pick.value]
    pos_md = "\n".join(f"- {p}" for p in data["positives"])
    neg_md = "\n".join(f"- {p}" for p in data["negatives"])
    mo.hstack(
        [
            mo.md(f"**positives** (concept present)\n\n{pos_md}"),
            mo.md(f"**negatives** (concept absent)\n\n{neg_md}"),
        ],
        widths=[1, 1],
    )
    return (data,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Learn + calibrate

    **Learning** is a subtraction, not a training loop: average the positive examples'
    activations, subtract the average of the negatives, and you have the concept's
    direction — plus a small Gaussian per class. It takes milliseconds.

    **Calibration** sets the threshold **τ**. The slider is the *strictness*: higher
    `z` pushes τ up so fewer things fire (fewer false alarms), lower `z` makes it more
    eager. Slide it and watch τ move on the decision plot below.
    """)
    return


@app.cell
def _(mo):
    z = mo.ui.slider(1.0, 6.0, value=3.0, step=0.5, label="calibration z (strictness)")
    z
    return (z,)


@app.cell
def _(cg, concept_pick, data):
    # re-learn only when the concept changes (learning re-extracts activations)
    name = concept_pick.value
    cg.concepts.clear()
    cg.learn(name, data["positives"], data["negatives"])
    return (name,)


@app.cell
def _(cg, name, z):
    # calibrate is cheap (no model) — re-runs when the slider moves
    cg.calibrate(z=z.value)
    concept = cg.concepts[name]
    return (concept,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Inspect what it learned

    **Left — the signature.** Each class's *mean* loudness across the tapped blocks,
    with its ±σ spread. Positive rides high, negative rides low — that gap is the whole
    trick. **Right — the decision.** The score distribution for the examples, with the
    τ line; anything to its right fires.
    """)
    return


@app.cell
def _(cg, concept, data, np, plt, tf):
    Ap = tf.read(cg.tok, data["positives"], cg.device, last_only=True)[0]
    An = tf.read(cg.tok, data["negatives"], cg.device, last_only=True)[0]
    Sp, Sn = concept.spectro(Ap), concept.spectro(An)
    Lp, Ln = concept.llr(Ap), concept.llr(An)

    x = np.arange(len(cg.layers))
    mu_p, sd_p = concept.gmm_pos.means[0], np.sqrt(np.diag(concept.gmm_pos.covs[0]))
    mu_n, sd_n = concept.gmm_neg.means[0], np.sqrt(np.diag(concept.gmm_neg.covs[0]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.fill_between(x, mu_p - sd_p, mu_p + sd_p, color="#C2402F", alpha=0.2)
    ax1.fill_between(x, mu_n - sd_n, mu_n + sd_n, color="#1F6FEB", alpha=0.2)
    ax1.plot(x, mu_p, "o-", color="#C2402F", lw=2, label="positive μ±σ")
    ax1.plot(x, mu_n, "o-", color="#1F6FEB", lw=2, label="negative μ±σ")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"blk {layer}" for layer in cg.layers])
    ax1.set_ylabel("loudness (spectrogram)")
    ax1.set_title("learned μ±σ across taps")
    ax1.legend(fontsize=8)

    both = np.concatenate([Lp, Ln])
    bins = np.linspace(both.min() - 1, both.max() + 1, 14)
    ax2.hist(Ln, bins=bins, color="#1F6FEB", alpha=0.6, label="negative")
    ax2.hist(Lp, bins=bins, color="#C2402F", alpha=0.6, label="positive")
    ax2.axvline(concept.tau, color="k", ls="--", lw=1.5, label=f"τ = {concept.tau:.1f}")
    ax2.set_xlabel("LLR (log positive − log negative)")
    ax2.set_title("decision distribution")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig
    return Sn, Sp


@app.cell
def _(Sn, Sp, alt, cg, data, mo, np):
    # per-prompt loudness heatmap — hover a cell to read the full prompt
    S_all = np.vstack([Sp, Sn])
    prompts = data["positives"] + data["negatives"]
    kinds = ["positive"] * len(Sp) + ["negative"] * len(Sn)
    records = [
        {
            "label": f"{'＋' if kinds[i] == 'positive' else '－'} {prompts[i][:34]}",
            "prompt": prompts[i],
            "kind": kinds[i],
            "block": f"blk {layer}",
            "loudness": round(float(S_all[i, j]), 2),
        }
        for i in range(len(prompts))
        for j, layer in enumerate(cg.layers)
    ]
    heat = (
        alt.Chart(alt.Data(values=records))
        .mark_rect()
        .encode(
            x=alt.X("block:O", title="tapped block"),
            y=alt.Y("label:N", sort=None, title=None),
            color=alt.Color(
                "loudness:Q",
                scale=alt.Scale(scheme="redblue", reverse=True, domainMid=0),
                title="loudness",
            ),
            tooltip=["prompt:N", "kind:N", "block:O", "loudness:Q"],
        )
        .properties(width=360, height=320, title="per-prompt loudness (hover for the prompt)")
    )
    mo.ui.altair_chart(heat)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Test it — and act

    Type any prompt. ConceptGate reads its activations (running only up to the deepest
    tap), scores it, and fires if the score beats τ. In a real guardrail you'd wire an
    **action** to a firing — here `run(Abort())` short-circuits and emits a marker.
    """)
    return


@app.cell
def _(mo):
    prompt = mo.ui.text(
        value="What's the best way to sear a steak?",
        label="your prompt",
        full_width=True,
    )
    prompt
    return (prompt,)


@app.cell
def _(Abort, cg, concept, mo, prompt):
    v = cg.check(prompt.value)
    r = cg.run(prompt.value, action=Abort(), max_new_tokens=10, check_output=False)
    verdict = "🔴 **FIRED**" if v.fired else "🟢 passed"
    mo.md(
        f"""
        **{prompt.value!r}**

        - verdict: {verdict} · score **{v.score:.2f}** vs τ **{concept.tau:.2f}**
        - `run(Abort())` → `{r.text!r}`
        """
    )
    return


if __name__ == "__main__":
    app.run()
