"""ConceptGate — interactive walkthrough (marimo).

Create your own concept (edit the example lists), watch it learn, calibrate the
threshold, and test prompts — with live μ±σ, a decision plot that marks your prompt,
and a per-prompt loudness heatmap.

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
    **concept** from a handful of examples — no fine-tuning, no gradients.

    It *listens*: at a few chosen layers it taps the model's residual stream and measures
    how strongly each example resonates with a direction learned as
    **mean(positive) − mean(negative)**. That gives a **loudness** per layer — a
    *spectrogram* across depth — which two small Gaussians (positive vs. negative) turn
    into a calibrated score. A concept fires when the score beats a threshold **τ**.

    The same machinery learns *any* concept. Define your own below.
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
    # starting points — pick one to LOAD its samples into the editable boxes, then edit
    PRESETS = {
        "cooking": {
            "positives": [
                "How long do I boil pasta?",
                "What temperature should I bake sourdough at?",
                "How do I dice an onion without crying?",
                "What can I substitute for buttermilk?",
                "How do I know when a steak is medium-rare?",
                "Best way to caramelize onions slowly?",
            ],
            "negatives": [
                "What is the capital of France?",
                "How do I center a div in CSS?",
                "Who won the 2018 World Cup?",
                "Explain recursion in one sentence.",
                "What's the exchange rate for the yen?",
                "How does a car engine work?",
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
            ],
            "negatives": [
                "How do I refactor this Python function?",
                "What's a healthy breakfast recipe?",
                "Explain how photosynthesis works.",
                "Who painted the Mona Lisa?",
                "What's the derivative of sin(x)?",
                "How do I set up a git remote?",
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
            ],
            "negatives": [
                "What's the best way to sear a steak?",
                "Cheapest month to fly to Tokyo?",
                "Who won the 2018 World Cup?",
                "Explain how photosynthesis works.",
                "What's the capital of Australia?",
                "Recommend a good hike near Seattle.",
            ],
        },
    }
    return (PRESETS,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Define a concept

    Pick a starting concept, then **edit the lists** — add, remove, or rewrite the
    examples (one per line). *Positives* are prompts where the concept is present;
    *negatives* are everyday prompts where it is absent. ~5–10 varied examples per side
    works well; the direction is a mean, so diversity beats quantity.
    """)
    return


@app.cell
def _(PRESETS, mo):
    preset_pick = mo.ui.dropdown(
        options=list(PRESETS), value="cooking", label="load a starting concept"
    )
    preset_pick
    return (preset_pick,)


@app.cell
def _(PRESETS, mo, preset_pick):
    # recreated (with fresh defaults) whenever the dropdown changes; editable afterwards
    d = PRESETS[preset_pick.value]
    name_in = mo.ui.text(value=preset_pick.value, label="concept name", full_width=True)
    pos_in = mo.ui.text_area(
        value="\n".join(d["positives"]), label="positive examples (one per line)",
        full_width=True, rows=7,
    )
    neg_in = mo.ui.text_area(
        value="\n".join(d["negatives"]), label="negative examples (one per line)",
        full_width=True, rows=7,
    )
    mo.vstack([name_in, mo.hstack([pos_in, neg_in], widths=[1, 1])])
    return name_in, neg_in, pos_in


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Learn + calibrate

    Learning is a subtraction (mean of positives − mean of negatives), plus a small
    Gaussian per class — milliseconds, no gradients. **Calibration** sets the threshold
    **τ**; the slider is the *strictness* — higher `z` fires less (fewer false alarms).
    """)
    return


@app.cell
def _(mo):
    z = mo.ui.slider(1.0, 6.0, value=3.0, step=0.5, label="calibration z (strictness)")
    z
    return (z,)


@app.cell
def _(cg, mo, name_in, neg_in, pos_in):
    # re-learns when the example lists change (learning re-extracts activations)
    positives = [ln.strip() for ln in pos_in.value.splitlines() if ln.strip()]
    negatives = [ln.strip() for ln in neg_in.value.splitlines() if ln.strip()]
    mo.stop(
        len(positives) < 2 or len(negatives) < 2,
        mo.md("➕ add at least **2** positive and **2** negative examples above."),
    )
    nm = name_in.value.strip() or "concept"
    cg.concepts.clear()
    cg.learn(nm, positives, negatives)
    return negatives, nm, positives


@app.cell
def _(cg, nm, z):
    cg.calibrate(z=z.value)   # cheap (no model) — re-runs when the slider moves
    concept = cg.concepts[nm]
    return (concept,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Test a prompt

    Type any prompt. Its score is drawn as a green line on the decision plot — you can
    see exactly where it lands relative to the examples and τ. `run(Abort())` shows the
    guardrail action: short-circuit and emit a marker when it fires.
    """)
    return


@app.cell
def _(mo):
    prompt = mo.ui.text(
        value="What's the best way to sear a steak?", label="your prompt", full_width=True
    )
    prompt
    return (prompt,)


@app.cell
def _(cg, concept, negatives, np, plt, positives, prompt, tf):
    Ap = tf.read(cg.tok, positives, cg.device, last_only=True)[0]
    An = tf.read(cg.tok, negatives, cg.device, last_only=True)[0]
    Sp, Sn = concept.spectro(Ap), concept.spectro(An)
    Lp, Ln = concept.llr(Ap), concept.llr(An)
    v = cg.check(prompt.value)

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

    lo = min(float(np.concatenate([Lp, Ln]).min()), v.score) - 1
    hi = max(float(np.concatenate([Lp, Ln]).max()), v.score) + 1
    bins = np.linspace(lo, hi, 14)
    ax2.hist(Ln, bins=bins, color="#1F6FEB", alpha=0.6, label="negative")
    ax2.hist(Lp, bins=bins, color="#C2402F", alpha=0.6, label="positive")
    ax2.axvline(concept.tau, color="k", ls="--", lw=1.5, label=f"τ = {concept.tau:.1f}")
    ax2.axvline(v.score, color="green", lw=2.5, label=f"your prompt ({v.score:.1f})")
    ax2.set_xlabel("LLR (log positive − log negative)")
    ax2.set_title("decision — your prompt in green")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig
    return Sn, Sp, v


@app.cell
def _(Abort, cg, concept, mo, prompt, v):
    r = cg.run(prompt.value, action=Abort(), max_new_tokens=10, check_output=False)
    verdict = "🔴 **FIRED**" if v.fired else "🟢 passed"
    mo.md(f"""
    **{prompt.value!r}** → {verdict} · score **{v.score:.2f}** vs τ **{concept.tau:.2f}**

    `run(Abort())` → `{r.text!r}`
    """)
    return


@app.cell
def _(Sn, Sp, alt, cg, mo, negatives, np, positives):
    # per-prompt loudness heatmap — hover a cell to read the full prompt
    S_all = np.vstack([Sp, Sn])
    prompts = positives + negatives
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
        .properties(width=360, height=330, title="per-prompt loudness (hover for the prompt)")
    )
    mo.ui.altair_chart(heat)
    return


if __name__ == "__main__":
    app.run()
