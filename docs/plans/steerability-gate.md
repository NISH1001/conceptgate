# Plan: steerability-fitted gate

Date: 2026-09-11. Branch: `feature/steerability-gate`. Status: spec, approved in design review; no code yet.

## Why

The report (§4.11, §5.5) shows that a first-token refusal log-odds *dose* — how far a fixed write along
the jailbreak direction moves the model on a prompt — is linearly decodable from that prompt's tap
activations far above a permutation null (Qwen2.5-0.5B +0.81; gemma-2-2b +0.61 at α=0.12; SmolLM2-1.7B
+0.74). The behavioural version of that claim was withdrawn because the two instruments used to validate
it agreed with each other at −0.01. Neither instrument's own reliability was ever measured: the generated
lever was one greedy sample per arm scored by a lexicon (three possible values per prompt), and the
teacher-forced lever used three canned refusals against three canned compliances. A disagreement between
two unmeasured instruments says the instruments are noisy, not that the quantity is absent.

The report's own next step (§5.5): "many more prompts, several magnitudes so each prompt's dose is a fitted
slope rather than a two-point difference, and a judged or classifier-scored outcome instead of a token
basket … If there is a next version of this system, that is what its gate should be fit to." This plan
builds exactly that, with a stop rule at every stage.

**Prior art checked (2026-09-11, third search).** No paper predicts a per-prompt steering dose from the
unsteered prompt. Near-misses to cite: Billa 2604.15557 (which *layer* succeeds), ASTEER 2606.11599
(post-steering states, 3-class, 1.4M generations), Braun 2505.22637 (dataset-level), CRH 2605.01844
(qualitative geometry), and **Forecasting Side Effects of Activation Steering 2608.11227** (Aug 2026:
cross-behaviour side-effect matrix over 67 behaviours from unsteered representations — behaviour-level,
not prompt-level). Self-estimated P(already published) ≈ 1/3, unchanged.

## Question

Can a read of a prompt's tap activations predict, before any generation, how much a fixed steering write
will move the model's refusal **behaviour** on that prompt — and does gating the write on that prediction
beat gating on concept presence?

## Stop rules (fixed before running)

| Stage | Go if | Otherwise |
|---|---|---|
| 1 Instrument | split-half reliability of the behavioural dose ≥ 0.6 **and** lexicon-vs-classifier dose agreement ≥ 0.6, on Qwen | double K once (16 → 32); if still below, stop; report as a methodology negative extending the report's §4.11 finding |
| 2 Prediction | out-of-fold Spearman ≥ 4 sd above the permutation null **and** above the gate's LLR Spearman, on Qwen **and** gemma | report as proxy-only; no library change |
| 3 Gate | outcome-gated write beats a size-matched random subset (permutation null over subsets) and the bottom-half complement on refusal shift per write, with benign collateral no worse than the concept gate | report the negative; keep the head as a library feature only if stage 2 passed |

Thresholds are Spearman ρ throughout. "sd above the null" uses the sd of the 300-draw permutation
distribution.

## Stage 1 — a reliable per-prompt behavioural dose

**Data.** The steerability prompt set, unchanged (`eval_gate._steer_prompts(120, 48)`): 120 held-out
jackhhao jailbreak templates each carrying a harmful request, 32 short framed attacks, 12 bare requests
(164 attacks), 48 real benign. Prompt kinds are recorded per row.

**Model and concept.** Qwen2.5-0.5B-Instruct first, taps from `taps_for()` (8/12/16), chat template ON,
`dtype` as the checkpoint declares (bfloat16; recorded). Concept `jailbreak` fit with seed 0 exactly as
`steerability_eval` does: `N_SHOT` from `FIT_POS`/`FIT_NEG`, `Direction.LOGISTIC`, `calibrate(z=2.0)`.
Write direction `W_raw`; random direction `U` from `default_rng([7000, seed])` with unit rows, as in
`scaleup_eval`. Magnitude `α · resid_norm(prompt)` with α = 0.08. Per prompt we also store the gate's
verdict (`llr`, `fired`, `p_present`, `resid_norm`) and the first-token log-odds for every arm (one extra
forward each, same token baskets as `logit_eval`) so the proxy is re-tested on the same run.

**Arms (5).** `none`, `minus` (−α·W_raw), `plus` (+α·W_raw), `rand_minus`, `rand_plus`. Deltas are
installed with `cg._steer_hooks(deltas)` exactly as the logit harness does; hooks broadcast over the
batch.

**Sampling.** For each (prompt, arm): one call to `cg.model.generate(ids, do_sample=True,
temperature=0.7, top_p=1.0, num_return_sequences=K, max_new_tokens=40, pad_token_id=eos)` with
K = 16. Seed: `torch.manual_seed` from `np.random.SeedSequence([seed, prompt_index, arm_index])` (the
repo learned that additive seeds collide). MPS sampling is not guaranteed bit-reproducible across
runs; every sampled continuation is stored in the results file so all downstream analysis is
reproducible from the JSON without regeneration.

**Scoring (two instruments per continuation).**
- `lex`: `eval_gate.is_refusal(text)` ∈ {0, 1}, the report's lexicon, unchanged.
- `clf`: `protectai/distilroberta-base-rejection-v1` (text-classification, DistilRoBERTa 82M,
  label index 1 = rejection; archived, so a fixed instrument). Input is the continuation only,
  truncated to 512 tokens, batched. Store both the hard label and P(rejection).

**Definitions.** For prompt p, arm a, instrument s:
`P_s(p,a) = (1/K) Σ_k s(g_{p,a,k})`.
Behavioural dose `D_s(p) = ½ (P_s(p, plus) − P_s(p, minus))`; random dose `D_s^rand(p)` likewise from
the random arms. With the template on, `plus` raises refusal, so D > 0 means the write is a safety lever
on that prompt. The first-token proxy lever is `½(Δ_minus − Δ_plus)` of the log-odds, as in the report;
its sign convention is opposite to D, so correlations with it are reported on −proxy.

**Reliability metrics (over the 164 attacks; benign reported separately).**
- split-half `ρ_sh = Spearman(D_odd, D_even)` using odd/even sample indices; Spearman–Brown
  `ρ_K = 2ρ_sh / (1 + ρ_sh)` as the full-K estimate;
- inter-instrument `Spearman(D_lex, D_clf)`;
- baseline reliability `Spearman(P0_odd, P0_even)` for the unsteered rate;
- proxy check `Spearman(D_s, −proxy lever)` per instrument — the direct re-test of §4.11;
- dose vs random dose `mean|D| / mean|D^rand|`, the perturbation floor;
- the same for benign prompts, to see whether benign prompts move as much as attacks (collateral).

**Outputs.** `scripts/eval_behaviour_dose.py` → `scripts/behaviour_dose_results.json`
(`{model, taps, alpha, K, temperature, seed, dtype, classifier, prompt_kinds, rows:[{kind, prompt,
llr, fired, p_present, resid_norm, logit:{arm: value}, samples:{arm: [text]*K}, lex:{arm: [0/1]*K},
clf:{arm: [p]*K}}]}`), plus the per-prompt tap activations to `scripts/behaviour_dose_acts__<tag>.npy`
(`[n, m, d]`, float32; `.npy` is gitignored) and `W_raw` to `scripts/behaviour_dose_wraw__<tag>.npy`.
Analysis in `scripts/analyze_behaviour_dose.py` prints the reliability table and writes
`scripts/behaviour_dose_analysis.json`; it never regenerates.

**Flags.** `--models` (comma list; default Qwen), `--k` (16), `--alpha` (0.08), `--temperature` (0.7),
`--seed` (0), `--dtype`, `--device` (mps), `--max-prompts` (subsample, keeps kind balance),
`--quick` (8 prompts, K=2, 8 new tokens), `--no-classifier` (offline smoke: lexicon only).

**Cost.** Qwen: 212 × 5 = 1,060 generate calls of 16 × 40 tokens ≈ 35 min on the M4; classifier over
17k short texts ≈ 2 min. gemma-2-2b: several hours; run unattended with `--models google/gemma-2-2b-it`.
Text storage ≈ 3–4 MB per model.

## Stage 2 — predict the behavioural dose from the prompt

Runs only if stage 1 passes. Reuses the pipeline of `analyze_steerability.py` / `steerability_controls.py`
(same `_cv`, same Ridge configuration, same standardization) with `y = D_s` for the instrument that
scored higher on split-half reliability (the other reported alongside).

**Features.** `X` = tap activations, z-scored per tap with fit-fold statistics, flattened to `m·d`.

**Evaluation.**
- 5-fold CV, and 5-fold **grouped by harmful request** (the primary number);
- permutation null: 300 draws of shuffled `y` through the identical pipeline; report mean, sd, z;
- learning curve n ∈ {8, 16, 32, 64, all} labelled prompts;
- cross-family transfer: fit on templates → test short+bare, and the reverse;
- **baselines:** the gate's LLR (`Spearman(llr, D)`), a 3-parameter ridge on the concept's per-tap
  projections (the spectrogram scalars), and predicted `D^rand` (generic predictability floor);
- **disposition control:** ridge on `P0` (unsteered refusal rate) and `Spearman(P0, D)`, so a "this
  prompt refuses anyway" account is ruled in or out;
- direction stability: `|cos|` between the fitted direction and `W_raw`, and split-half
  self-consistency of the fitted direction — reported with the report's caveat that at this n the
  latter is a sampling-variability figure, not an estimability ceiling.

**Models.** Qwen, then gemma (its own stage 1 run), then SmolLM2-1.7B if wanted. Go/no-go is on Qwen
and gemma.

**Outputs.** Numbers into `scripts/behaviour_dose_analysis.json`; tables into `docs/evaluation.md` §10.

## Stage 3 — the outcome-fitted trigger

Runs only if stage 2 passes. Two parts: a library feature and its evaluation.

**Library.**
- `conceptgate/outcome.py`: `OutcomeHead` — `fit(A: [n, m, d], y: [n]) -> self` stores per-tap
  mean/sd and a ridge `(w: [m·d], b)`; `predict(A) -> [n]`. Pure numpy + sklearn `Ridge`, same alpha
  convention as the analysis scripts. Optional `features="concept"` variant: fit on a named concept's
  per-tap projections (3 scalars) instead of the raw taps — kept if stage 2 shows it ties the full head.
- `ConceptGate.learn_outcome(name, prompts, y, *, features="taps", concept=None, batch_size=1)`: reads
  the taps for `prompts` with the existing `TapForward.read(..., last_only=True)`, fits an
  `OutcomeHead`, stores it in `self.outcomes[name]`. Labels are needed at fit time only.
- `Verdict.outcomes: dict[str, float]` (default empty), filled in `_verdict` from the same `A_last` the
  concepts see, so `check(prompt).outcomes["dose"]` is the predicted dose at no extra forward.
- `actions.Predicted(outcome: str, threshold: float, above: bool = True)`: accepted wherever `when:` is.
  `_triggered` widens to `Trigger | str | Predicted`; a `Predicted` fires when
  `v.outcomes[outcome] >= threshold` (or `<=` if `above=False`); a missing name raises `KeyError` with
  the learned names. The `Trigger` enum is untouched.
- `actions.Both(a, b)` — fires when both triggers fire — so the deployable rule *presence AND
  steerability* is `Steer(concept="jailbreak", when=Both(Trigger.FIRE, Predicted("dose", τ)))`. Small,
  and it is the one composition specific to this system (two reads deciding one write).
- `Steer`, `Abort`, `Emit`: `when` type widened; behaviour otherwise unchanged. Exported from
  `conceptgate/__init__.py`.

**Evaluation (`analyze_behaviour_dose.py --gate`, no new generation).** Stage 1 already sampled the
`plus` arm for every prompt, so every gating arm is a *selection* over the same per-prompt
`ΔP_plus(p) = P(p, plus) − P(p, none)`; under independent prompts an arm's value is the mean of
`ΔP_plus` over the prompts it writes. Predictions are **out-of-fold** (grouped 5-fold) so no prompt is
gated by a head that saw its label. Arms at 50% coverage of attacks: `blanket` (all), `concept_gate`
(fires; ≈97%, so ≈ blanket), `outcome_gate` (top half by predicted D), `random` (500 random halves →
null distribution), `anti_outcome` (bottom half), `both` (fires AND top half). Metrics: mean `ΔP_plus`
per written attack (refusal points per write), total refusal gained over all attacks, fraction of
benign prompts written and their mean `|ΔP_plus|` (over-refusal collateral). Honest framing, stated in
the doc: this is the stage-2 prediction expressed as an operating gain and adds no independent
evidence; the random-halves null is the only genuine test (§6's identity caveat applies to
blanket = gate + anti-gate).

## Files

| Path | Change |
|---|---|
| `scripts/eval_behaviour_dose.py` | new: stage 1 generation + scoring harness |
| `scripts/analyze_behaviour_dose.py` | new: stage 1 reliability, stage 2 prediction, stage 3 selection arms |
| `scripts/behaviour_dose_results.json`, `behaviour_dose_analysis.json` | new results (committed) |
| `scripts/behaviour_dose_acts__*.npy`, `behaviour_dose_wraw__*.npy` | new caches (gitignored) |
| `conceptgate/outcome.py` | new: `OutcomeHead` |
| `conceptgate/gate.py` | `learn_outcome`, `self.outcomes`, `Verdict.outcomes` filled in `_verdict` |
| `conceptgate/actions.py` | `Predicted`, `Both`, `_triggered` widened, `when` types widened, `Verdict.outcomes` field |
| `conceptgate/__init__.py` | export `Predicted`, `Both`, `OutcomeHead` |
| `tests/test_outcome.py` | new |
| `docs/evaluation.md` | §10 stage tables + verdict |
| `docs/literature.md` | add 2608.11227 and the classifier as an instrument |
| `README.md` | one paragraph under Steer: outcome-fitted trigger |

## Tests

- `OutcomeHead`: shapes; exact recovery of a planted linear direction on synthetic `[n, m, d]` data
  (ρ ≈ 1 out of fold); standardization applied at predict time.
- `Predicted` / `Both`: `_triggered` truth table on hand-built `Verdict`s; `KeyError` message lists
  learned names; `Trigger` members and strings still work.
- `learn_outcome` round trip on gpt2 (already used by the test suite): 6 prompts, synthetic `y`,
  `check(...).outcomes["dose"]` present and finite; `Steer(when=Predicted(...))` installs hooks only
  when the prediction crosses the threshold.
- Harness smoke: `uv run --with datasets python scripts/eval_behaviour_dose.py --quick --no-classifier`
  completes and writes a well-formed results file; analysis runs on it.

## Docs and reporting

`docs/evaluation.md` §10 records each stage's table, the stop-rule outcome, and the verdict in the
existing voice (negatives as prominent as positives). The tech report gets a §4.12 only after stage 2's
result is known, on the site repo, with `render-figures-before-pushing` applied to any figure.

## Not claimed / out of scope

- No security claim; obfuscated-activation attacks (Bailey 2412.09565) apply to the head as to any
  probe.
- No mechanism novelty: ridge on activations is a probe. The candidate contribution is the *question*
  (per-prompt behavioural dose from the unsteered prompt), the *measurement* (a reliability-tested
  instrument), and the *composition* (presence AND steerability deciding one write).
- One concept (jailbreak), one α for the behavioural instrument. An α sweep of the instrument
  (fitted slope per prompt) is a stretch item, not a gate.
- An LLM judge is out of scope for now; sampled texts are stored so one can be added later.
- Correlational throughout; effective replicates for the concept fit remain one per model.
