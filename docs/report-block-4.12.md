# Draft §4.12 for the technical report — behavioural steerability

**Status:** draft for the report session to adapt. Prose is written in the report's voice; the report's own
markup (the `cg-mono` table wrappers, `<span class="cite">` citations, `sref` cross-links) still needs adding.
Every number here comes from `scripts/behaviour_dose_analysis.json` via `docs/evaluation.md` §10.
The gemma paragraph is now filled from the completed K = 32 run; this draft is complete.

Cross-links this section needs: §4.10 (the formatting confound), §4.11 (the first-token proxy and its
withdrawal), §5.5 (what would be new). New citation required: **Logit-Gap Steering**, Li & Liu,
arXiv:2506.24056 — it defines the first-token refusal–affirmation logit gap as a per-prompt safety margin and
is therefore the *source* of the measure §4.10–4.11 use, not a near-miss. Also: **Forecasting Side Effects of
Activation Steering**, arXiv:2608.11227, which predicts cross-behaviour side effects from unsteered
representations at the behaviour level, the nearest work to what follows.

---

## 4.12 Per-prompt steerability, measured on behaviour

§4.11 asked whether a read of the prompt could predict how far a steering write would move the model, found
that it could on a first-token proxy, and then withdrew the claim at the prompt level: the two behavioural
measures brought in to validate the proxy agreed with each other at
-0.01, leaving nothing stable to validate against. That conclusion was
correct about the evidence and wrong about the cause. Neither validating instrument had its own reliability
measured. One was a single greedy continuation scored by a refusal lexicon, which takes three distinct values
per prompt; the other was the log-probability of three canned refusals against three canned compliances. Two
unreliable instruments disagreeing says the instruments are noisy, not that the quantity is absent.

So we built an instrument and measured it first. For each of the same 212 prompts and each of five arms —
unsteered, ±α along the few-shot jailbreak direction, and ±α along a random direction of matched norm — we
sample **sixteen** continuations at temperature 0.7 and score each one twice, with the refusal lexicon and
with an off-the-shelf rejection classifier. A prompt's **dose** is half the difference between its refusal
rate under +α and under −α. The first-token proxy of §4.11 is recorded on the same forward passes, so the two
can be compared per prompt rather than in aggregate.

**The per-prompt behavioural dose is measurable.** Split-half reliability over attacks is
+0.66 by the lexicon and +0.70 by the
classifier, which Spearman–Brown puts at +0.80 and
+0.82 for the full sixteen samples; the two independent scorers agree on the
per-prompt dose at +0.88. Against that instrument the first-token proxy tracks
behaviour at +0.62, above the 0.5 bar §4.11 had set and failed at
+0.43 and +0.48 with the greedy
measures. The quantity was always there. The earlier measurement was not sensitive enough to see it.

**It is predictable from the prompt, before any generation.** A ridge on the three tapped activations predicts
the dose of a held-out prompt at Spearman +0.58 under cross-validation grouped by
harmful request, against a permutation null through the identical pipeline of
-0.015 ± 0.089 — **6.7 standard deviations** above chance. The
gate ConceptGate already computes reaches only |0.30| on the same target, and with
the opposite sign: the more confident the concept read, the *less* the write moves that prompt.

Three controls decide what this is not. It is not prompt length: the taps decode token count almost perfectly
(+0.90), yet length correlates with the dose at
-0.12 and the length-residualised dose is still predicted at
+0.57. It is not prompt family: the taps separate templates from short attacks at
+0.77, yet the dose is predicted *within* templates alone
(+0.52) and within short and bare requests alone
(+0.63). And it is not refusal disposition: the unsteered refusal rate is more
predictable still (+0.81) but is unrelated to the dose
(+0.03). Nor is it an artefact of our analysis choices — all three scorers give
the same verdict (grouped CV +0.54 to +0.60, null z
6.0–6.5, their out-of-fold predictions agreeing at
+0.87), and the ridge penalty may vary a thousandfold for a swing of
+0.58 to +0.63.

**Most of the signal lives in directions the system already has.** Three raw projections onto the concept
direction predict the dose at +0.51, against the calibrated gate's
|0.30|. The information is inside ConceptGate's own read; the gate discards most
of it in the course of turning a spectrogram into a fire/abstain/pass decision.

**Used as a gate, the prediction selects where writing works.** Every arm below is a selection over the same
per-prompt refusal gain of the +α arm, with predictions taken out of fold, so this restates §4.12's prediction
as an operating characteristic rather than adding independent evidence; the size-matched random null is the
test that has teeth.

| coverage of attacks | refusal gained per write | size-matched random | P(random ≥ gate) | benign prompts written |
|---|---|---|---|---|
| 10% | +0.242 | +0.099 | 0.002 | 2% |
| 25% | +0.216 | +0.096 | 0.000 | 10% |
| 50% | +0.143 | +0.098 | 0.008 | 23% |
| 75% | +0.129 | +0.097 | 0.000 | 52% |
| 90% | +0.108 | +0.097 | 0.014 | 85% |

The gate beats its null at every operating point, and the gain per write climbs monotonically as coverage
tightens against a flat random baseline. That monotonicity is the substance: a lucky split would win at one
threshold, whereas a genuine ranking of prompts by responsiveness improves as you keep only its top. For
comparison, the concept gate at its own operating point writes to 154 of 164 attacks and to **every** benign
prompt in the set, gaining +0.090 per write.

**The second model does not replicate, and says something specific.** On gemma-2-2b the same procedure
fails, and it fails for a reason visible in the raw rates: at the same write magnitude that moves Qwen's
refusal from 0.35 to 0.55 to 0.65 across the −α, unsteered and +α arms, gemma moves
0.46 → 0.51 → 0.53. Its mean absolute dose is
0.069 against Qwen's 0.163, and only
41% of its doses are even positive. With the intervention barely perturbing the
model there is little per-prompt variance for anything to predict, and the prediction duly fails: grouped
cross-validation of +0.078 against a permutation null of
-0.001 ± 0.089, which is
0.9 standard deviations — no effect.

We can be more definite than "underpowered", because the experiment was run at two sample sizes. Sixteen
samples per arm gave a non-significant hint of +0.20 (z 2.2). Doubling to
thirty-two samples makes the *target* more reliable, and attenuation theory says that should make an
attenuated signal *clearer*: the hint implied a true correlation of 0.238, which
should have surfaced as +0.215. It fell to
+0.078 instead. A signal masked by measurement noise gets sharper when you
measure better; noise does not. The supporting pattern agrees — cross-family transfer collapsed from
+0.12 / +0.09 to +0.03 / -0.02,
and the fitted direction's split-half self-consistency fell to
0.04–0.05. There is no direction there to find.

Two things guard against reading this as a fact about gemma's representations rather than about the write.
The concept direction moves refusal 1.5–1.8× as far as a random direction of
matched norm on **both** models, so gemma's direction is not less special than Qwen's; its effect is smaller.
And gemma's gate LLR carries no dose information (-0.05), which independently
reproduces the magnitude sweep earlier in this section, where the gate's correlation with the dose replicated
on Qwen and SmolLM2 but never on gemma. The right follow-up is therefore a sweep of the write magnitude on
that model, not more samples at a magnitude which does not move it.

One methodological note that generalises. Our pre-registered reliability gate was a split-half correlation,
and **that statistic is not comparable across sample sizes**: at thirty-two samples it compares two
sixteen-sample halves, so gemma "passed" it at K = 32 (0.69) purely because the
statistic got easier, having failed it at K = 16 (0.53). We predicted the exact
value in advance (0.694 predicted against 0.692
observed) for this reason. Anyone gating a decision on measurement reliability should compare the
Spearman–Brown-corrected full-sample figure, or split-halves at matched half-size, and not the raw statistic.

**What this changes about the system.** The gate was built to answer *is the concept present?*, and on
correctly formatted attacks it answers yes almost always, which is precisely why §4.10 found it selecting
nothing. The question a write actually needs answered is *how far will this prompt move?* That is a different
quantity, readable from the same taps and the same forward pass, and a few dozen labelled prompts are enough
to fit it — eight already give +0.33, sixty-four give
+0.58. If there is a next version of this system, this is what its gate should be
fit to.

**Limits.** One concept, one write magnitude, 164 attacks, one concept fit per model, and correlational
throughout. The selection table inherits the noise of a one-arm quantity whose own reliability is only
+0.32, which attenuates those gaps but cannot manufacture the ordering, since the
predictions are out of fold. The labels cost two extra forward passes per prompt plus the sampling, which is
the real price of fitting a gate this way.
