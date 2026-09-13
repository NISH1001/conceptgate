# ConceptGate evaluation — working notes

**Goal:** establish that ConceptGate has *value* — specifically that it **learns concepts
efficiently** (the title claim). "Efficient" = matches strong baselines on the target metric at a
**fraction of the memory + compute**, few-shot and closed-form, *and* is steerable (a classifier
is not). In-distribution detection AUC alone is a commodity and does NOT establish value.

**Where this lives:** all of it is merged to `main`. Harnesses: `scripts/eval_detection.py` (detection,
efficiency, multi-concept scaling, OOD), `scripts/eval_steering.py` (steering dose-response), and
`scripts/eval_gate.py` (gate-conditioned steering, read/write cosine, decoupling). Run everything with
`uv run --with datasets python ...`.

**Data (public, cached):** `jackhhao/jailbreak-classification` (jailbreak *templates* vs benign;
train/test 1044/262), `PKU-Alignment/BeaverTails` (harmful *requests* vs safe). Prompts capped at
1200 chars. Few-shot pool = 64/class from train; test = official split.

**Detectors (all on the SAME tapped activations unless noted):**
- `conceptgate` — ConceptGate depth-bandpass, **diff-of-means** direction (default, weak mode).
- `conceptgate_log` — ConceptGate, **`Direction.LOGISTIC`** per-layer direction (the strong mode).
- `best_layer` — single-best-layer diff-of-means (single-layer baseline A).
- `logistic` / `svm_linear` / `svm_rbf` — vanilla LR / linear SVM / RBF SVM on the FULL flattened
  tap activations (~3 taps × d dims).
- `--fullprobe`: CG (3 taps) vs a linear probe on the **frozen full model, swept over ALL layers**
  (best) + last-layer (naive head). [running / to record]

**Metric:** AUC (threshold-free, primary) + recall@fixed-FPR. 3 resample seeds, mean reported.

---

## Results so far

### 1. In-distribution detection (N/class = 4→32, AUC)

| Model (N=32) | CG-diff | **CG-log** | LR | SVM-lin | SVM-rbf | detect | trunc↑ |
|---|---|---|---|---|---|---|---|
| gpt2 (taps 4/6/8) | 0.895 | 0.947 | 0.946 | 0.947 | 0.898 | 31ms | 1.75× |
| Qwen2.5-0.5B (8/12/16) | 0.948 | 0.983 | 0.988 | 0.990 | 0.969 | 82ms | 1.73× |
| SmolLM2-1.7B (8/12/16) | 0.931 | 0.982 | 0.982 | 0.985 | 0.960 | 424ms | 1.48× |
| gemma-2-2b-it (9/13/17) | 0.953 | 0.979 | 0.980 | 0.982 | 0.959 | 548ms | 2.10× |

Findings:
- **CG-logistic ≈ LR ≈ SVM-linear** (all within ~0.01 AUC at N≥16). Given the same features, no
  discriminative classifier meaningfully beats CG's 3-scalar direction.
- **SVM-rbf is worse than linear** everywhere → the concept is **linearly separable**; a non-linear
  kernel overfits. This supports CG's premise (concept = a linear direction).
- **CG-diff (default) is the weak mode** — ~3–5 AUC pts behind CG-log at N≥16. Diff-of-means is
  covariance-blind (optimal only under isotropic covariance); real activations are correlated. The
  gap is ~0 at N=4 (can't estimate covariance from 4 samples) and grows with N. **ALWAYS use
  `Direction.LOGISTIC` when comparing detection to a classifier.**
- Learn is closed-form (~0.2–14.5 ms). Truncated forward is 1.5–2.1× cheaper than full.

### 2. Cross-distribution transfer (train → test, AUC, N=32)

| | jack→jack | jack→beaver | beaver→jack | beaver→beaver |
|---|---|---|---|---|
| Qwen CG-log | 0.984 | **0.430** | **0.507** | 0.680 |
| Qwen LR | 0.985 | 0.448 | 0.465 | 0.687 |
| gemma CG-log | 0.977 | **0.520** | **0.356** | 0.720 |
| gemma LR | 0.975 | 0.518 | 0.411 | 0.731 |

Findings:
- **NULL for a CG generalization edge.** Both CG and LR collapse to ~chance cross-distribution;
  CG is not more robust (drops ~0.45–0.55 for both).
- **Why:** jackhhao (jailbreak *framing*) and BeaverTails (harmful *content*) are **different
  concepts**, not the same concept in different clothes. This tests concept *transfer* (fails for
  everyone), not robustness. **The clean test is within-concept:** BeaverTails **category holdout**
  (train on some harm categories, test on held-out ones). NOT yet run.
- Note: BeaverTails in-dist AUC is only ~0.68–0.73 (harmful requests are subtler than DAN templates).

### 3. Efficiency frontier — CG tap configs vs full-model linear probe (`--efficiency`)

AUC × memory (universal) × compute (**wall-time, Apple M4 / MPS**, N=32, 3 seeds). Linear probe =
full model, final-layer head fit on the same N examples. fwd = per-prompt forward wall-time.

**Qwen-0.5B (494M)** — probe: AUC 0.982 @ 96.1 ms (100% depth/weights)
| CG config | depth | weights | AUC-log | fwd ms | vs probe |
|---|---|---|---|---|---|
| 1tap@25% | 29% | 49% | 0.968 | 28.7 | 3.3× faster, −0.014 |
| 1tap@55% | 58% | 70% | 0.978 | 55.2 | 1.7× faster, −0.004 |
| 1tap@85% | 88% | 91% | 0.982 | 83.4 | matches |
| 3tap | 71% | 79% | 0.973 | 67.1 | 1.4× faster |
| 5tap | 75% | 82% | 0.979 | 71.4 | 1.3× faster |

**gemma-2-2b (2.66B)** — probe: AUC 0.987 @ 546.9 ms (100%)
| CG config | depth | weights | AUC-log | fwd ms | vs probe |
|---|---|---|---|---|---|
| 1tap@25% | 27% | 43% | 0.948 | 143.6 | 3.8× faster, −0.039 |
| 1tap@40% | 42% | 55% | 0.974 | 226.0 | 2.4× faster, −0.013 |
| 3tap | 69% | 76% | 0.979 | 391.7 | 1.4× faster, −0.008 |
| 5tap | 73% | 79% | 0.977 | 401.3 | 1.4× faster |

Findings:
- **CG's single-concept efficiency is NOT a CG advantage — it is the truncated forward.** A
  **depth-matched probe** (logistic regression on the SAME tapped activations, same truncated forward)
  EQUALS CG-logistic at every depth (Qwen 0.968=0.968 … 0.982=0.982; gemma 0.948=0.948 … 0.974=0.974)
  and on multi-tap slightly BEATS CG's bandpass fusion (Qwen 3tap 0.973 vs 0.978). So the "2–4× less
  compute" is only vs the FULL-MODEL probe; a fair depth-matched baseline gets CG's number at CG's
  compute. Do NOT frame single-concept efficiency as a ConceptGate contribution. [added `auc_probe_tap`
  to `bench_efficiency`/`bench_fullprobe`]
- **A single early tap suffices** (Qwen 1tap@40% 0.970; gemma 1tap@40% 0.974): the concept is linearly
  readable at ~⅖ depth — a fact about the MODEL (how early the abstraction forms), shared by any probe.
- **Depth fusion does not transfer**: 3/5-tap ≈ best single tap, and the depth-matched probe on the
  concatenated taps beats the bandpass fusion. The synthetic §4.1 win is on data matched to its own
  assumptions.
- CG-diff is the weaker mode (~0.92–0.96); logistic is the one to report — and CG-logistic on one tap
  IS logistic regression on that tap.
- Learned params: CG's full read+write state ≈ 4md (detect + steer directions + standardization),
  ~11K Qwen / ~28K gemma — an order of magnitude above the probe's d, both kilobytes; the reviewer
  correctly flagged the earlier m·d count as omitting the steering vector.

### 4. Multi-concept scaling — the amortization argument (`--scaling`)

Cost of a **K-concept bank** over BeaverTails' 14 harm categories (`PKU-Alignment/BeaverTails`),
N=32/class, 3 seeds, Apple M4 / MPS. CG taps deeper here (50/70/85% depth) because harm *content* is read
later than jailbreak *framing*. Three ways to build the bank: **CG** (a closed-form direction per concept
on the frozen taps), **linear probe** (a trained logistic head per concept on the frozen final layer),
**LoRA** (a rank-8 adapter fine-tuned per concept). LoRA measured on 3 categories (animal/child/controversial).

**Mean per-category AUC (14 categories):**

| Model | CG bank (taps) | probe bank @ final | **probe bank @ CG taps** | LoRA (3 cats) |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.832 | 0.855 | **0.840** | 0.685 |
| gemma-2-2b | 0.881 | 0.874 | **0.880** | 0.814 |

**The fair probe bank is the third column** (`scripts/eval_probe_tap_bank.py`, fit from the scaling
run's cached activations; reproduces the stored CG and probe@final numbers to 4 decimals). The original
comparison put the probe heads on the FINAL layer (100% of the backbone) while CG ran 91% — a depth
advantage CG did not get. Heads on CG's own taps at identical cost: **tie** on gemma (−0.001, ahead on
8/14 categories), **edge** CG on Qwen (+0.008, ahead on 10/14). Exactly what the single-concept §3
result predicted. On Qwen the final layer genuinely reads harm content better than the mid taps (+0.015);
on gemma it does not. On gemma CG *edges* the final-layer probe; on Qwen it trails it by 0.02. On the 3 LoRA categories CG scores 0.889 (Qwen)
/ 0.913 (gemma) vs LoRA's 0.685 / 0.814 — few-shot fine-tuning is both the slowest and the weakest.

**Per-concept cost (marginal, to add one concept) and whole-bank totals:**

| per concept | CG | probe | LoRA |
|---|---|---|---|
| learn (Qwen / gemma) | 6.1 / 11.2 ms | 1.6 / 2.5 ms | 16.7 / 125.8 s |
| params | 2.7–6.9 K | 0.9–2.3 K | 0.54–1.6 M |
| inference over all K | 1 shared fwd | 1 shared fwd | 1 fwd **each** |

Whole 14-concept bank — build: CG ~8 s (Qwen) / ~46 s (gemma) vs LoRA ~3.9 / ~29 min (**30× / 38×**);
inference for all 14: CG 11 / 65 ms (flat in K) vs LoRA 174 / 937 ms (linear in K); memory: CG 38K / 97K
params vs LoRA 7.6M / 22.4M.

Findings:
- **Cost is flat/shallow in K for a training-free bank, linear (and steep) for per-concept fine-tuning.**
  One truncated forward reads all K concepts; each concept is a closed-form fit. LoRA needs a training run
  and a separate forward per adapter.
- **The cheap bank is accurate** — matches (Qwen) or beats (gemma) the trained probe per category, and
  far exceeds few-shot LoRA (a randomly-initialized head has too little signal in 2N examples).
- **Honest scope:** a *linear-probe bank* shares CG's amortization (both are training-free latent banks).
  What is specific to CG is that the same K directions also **steer** (the read/write duality). So the
  amortization result separates training-free latent banks from *fine-tuning*, not from a probe bank.

### 5. Within-concept generalization — leave-one-category-out (`--ood`)

Fixes the confound in the cross-distribution test (§2): instead of jailbreak-*framing* vs harmful-*content*
(different concepts), this holds the **concept fixed** (harmfulness) and varies the surface **category**.
For each of the 14 BeaverTails categories c, learn the harmful direction from the OTHER 13 categories
(+ a shared benign pool, N=32/class) and score the held-out category c; compare to the in-distribution
reference that trains on c itself. Same N, same test set. CG (logistic taps) vs full-model linear probe,
3 seeds, Qwen + gemma. Reuses the scaling activation cache.

| Model | CG in→OOD (drop) | probe in→OOD (drop) |
|---|---|---|
| Qwen2.5-0.5B | 0.827→**0.647** (0.181) | 0.847→0.610 (0.237) |
| gemma-2-2b | 0.868→**0.616** (0.251) | 0.866→0.610 (0.256) |

Findings:
- **Partial generalization.** A harmfulness direction trained on 13 categories detects an unseen 14th at
  ~0.61–0.65 AUC — above chance, well below in-distribution (~0.83–0.87). Not a collapse (unlike the
  confounded cross-dist null in §2), not free transfer either.
- **CG generalizes at least as well as the probe** — a smaller drop on Qwen (0.181 vs 0.237; OOD 0.647 vs
  0.610) and tied on gemma (0.251 vs 0.256). The mid-layer tapped direction is at least as
  category-transferable as the final-layer head, so the OOD story is not a CG weakness.
- **"Harm" is not monolithic.** Per-category OOD ranges widely: violence / self-harm / drugs / terrorism /
  financial transfer well (~0.74–0.78 Qwen / 0.67–0.75 gemma), while controversial-politics (~0.39–0.47, below chance) and
  discrimination (~0.39) barely transfer — those categories read differently in the residual stream.

This is the clean generalization test the cross-distribution transfer could not give.

---

### 6. Gate-conditioned steering — SUPERSEDED by §8 (these numbers ran WITHOUT the chat template)

The steering rule is plain CAA: direction + magnitude, no spectrogram, no depth filter, no threshold.
So the read side's contribution to the write side is **not** the direction — it's deciding *when* to
write (`Steer(when=Trigger.FIRE)`). That is the only operation no probe and no external guard can do,
and it had no measurement. Qwen-0.5B, taps 8/12/16, frac −0.08, 32 held-out jailbreak + 32 benign
prompts, concept fit 8+8 from short override framings, 3 few-shot resamples:

Gate fires on **54.2%** of held-out jailbreak prompts, 10.4% of in-register benign. Five arms, because
three cannot separate *which* prompts get the write from *how many* do:

| arm | writes | jb refusal | Δ vs none (paired) | benign ppl | benign byte-identical |
|---|---|---|---|---|---|
| no steer | 0% | 46.9 ± 0.0% | — | 1.98 | 100% |
| always | 100% | 49.0 ± 3.0% | +2.1 ± 2.9 | 2.17 | **4.2 ± 1.5%** |
| **gate** | 54% | **55.2 ± 1.5%** | **+8.3 ± 1.5** | 2.03 | **89.6 ± 9.0%** |
| random (size-matched) | 54% | 47.9 ± 3.9% | +1.0 ± 3.9 | 2.01 | 89.6 ± 9.0% |
| anti-gate (complement) | 46% | 40.6 ± 2.5% | **−6.2 ± 2.6** | 2.12 | 14.6 ± 7.8% |

**Gating wins on BOTH axes**, and the two control arms give the mechanism:
- **Selection, not dosage.** Random-54% adds +1.0 vs gate's +8.3; paired by seed gate − random =
  **+7.3 ± 5.3**, non-negative in all 3 seeds. Writing to half the prompts is worth nothing; writing to
  *that* half is.
- **Writing where the concept is ABSENT actively suppresses refusal** (anti-gate −6.2, negative in all
  3 seeds). So blanket steering's flat result is a real gain cancelling a real loss.
- Two parameters fit from gate + anti-gate (+15.3 pts where it registers, −13.5 where it doesn't). CAUTION: "blanket = gate + antigate − none" is an exact IDENTITY under greedy decoding (fired/passed partition the set, prompts are generated independently) — it holds to the decimal every seed and is NOT a test. The only genuine check is the random arm (predicted +1.1, measured +1.0), and with sd 3.9 that agreement is luck. Say "consistent with additivity", never "reproduces".

Precision: 32 prompts → 1 prompt = 3.1 pts, so +8.3 is 2.7 prompts and −6.2 is 2.0. sd is across 3
few-shot resamples, not prompts; error bars overlap. What carries it is that the ordering repeats every
seed and four arms fit one two-parameter account. A decisive version needs hundreds of prompts.

**Sign-flip arms (`--signflip`): the off-target loss is PERTURBATION, and the gate finds where the write is a lever.**
Same masks, write +α instead of −α:

| Δ refusal (pts) | write −α (away) | write +α (toward) |
|---|---|---|
| gate fires (54%) | **+8.3** ± 1.5 | **−11.5** ± 5.3 |
| gate passes (46%) | **−6.2** ± 2.6 | **−12.5** ± 4.4 |

Every cell keeps its sign in all 3 seeds. Fired row: sign flips → the direction is a causal LEVER there.
Passed row: both signs hurt → the direction is NOISE there. So the gate is locating the prompts on which
the write is *interpretable at all* — that is why selection beats dosage. A reviewer proposed a
"directional" account of the −6.2 (−α pushes passed attacks toward benign → comply); it predicts +α on
passed prompts would HELP. It hurt more (−12.5). Not supported. Deployment lesson is about UNGATED CAA:
blanket safety steering degrades safety wherever the concept isn't registered, in either direction. A
gated system leaves those prompts untouched, so a gate that misses an attack does not assist it.
+α arms are the noisiest (sd 5.3/4.4); sizes loose, signs not. Benign ppl under +α on passed prompts: 2.47
(vs 2.12 for −α, 1.98 unsteered).

Two honest limits: (1) small effect sizes, 0.5B model, one magnitude, 32 prompts — establishes gated >
blanket, NOT that this is a deployable defense. (2) **The gate is only sharp in the register it was fit
in.** Fit the same concept from the dataset's long DAN templates instead → fires on **0.0%** of short
framed requests (all 3 seeds): it learned length, not intent. And the short-framing-fitted concept fires
on **91.7%** of *real* out-of-register benign prompts — catastrophic FPR in deployment.

**BeaverTails bank steering — NULL on behaviour** (`--beavertails`). The 14-concept bank had only been
measured as a *detector* while we claimed each entry also steers. Same three arms, 5 harm categories,
direction fit from 32 harmful prompts/cat vs the shared benign pool, 12 held-out prompts/cat:

| category | fire harmful/benign | refusal none→always→gate | benign untouched always/gate |
|---|---|---|---|
| violence, incitement | 50%/8% | 8.3 → 0.0 → 8.3 | 0%/92% |
| drug abuse, weapons | 58%/25% | 0.0 → 0.0 → 0.0 | 0%/75% |
| financial/property crime | 67%/25% | 25.0 → 0.0 → 16.7 | 0%/75% |
| privacy violation | 92%/33% | 8.3 → 25.0 → 25.0 | 0%/67% |
| hate speech | 58%/42% | 41.7 → 33.3 → 33.3 | 0%/58% |
| **mean** | **65%/27%** | **16.7 → 11.7 → 16.7** | **0%/73%** |

Collateral finding replicates (gate 73% untouched vs blanket 0%) and blanket steering again *hurts*
(16.7 → 11.7). But gating no longer improves anything — refusal flat at 16.7%. **Steering away from a
harm TOPIC does not make the model decline.** Contrast the jailbreak concept (+8.3 pts): "jailbreak
framing" is about request *intent/register* and sits near the safety-tuned refusal behaviour; "violence"
is about content *topic*, so steering changes what the text is about, not whether the model complies.
These directions also gate far more leakily (65 vs 27 firing, against 54 vs 10 for jailbreak) — same
partial-generalization limit as §5. Caveats: refusal scored by explicit-decline lexicon (a harm-content
reduction short of refusing wouldn't register); one magnitude, one 0.5B model.

So: "each bank entry also steers" ⇒ each entry gives a free write *direction*; demonstrated behavioural
control exists only for topical concepts (eval_steering.py) and jailbreak framing (§6).

### 7. Read/write cosine — the duality is real but not identity (`--cosine`)

Mean |cos| between the detection direction mapped to raw space (`W/sd0`, renormalized) and the raw
steering direction `W_raw`, over 3 taps × 4 concepts:

| mode | gpt2 (d=768) | Qwen-0.5B (d=896) | gemma-2-2b (d=2304) |
|---|---|---|---|
| diff-of-means | 0.79 ± 0.06 | 0.65 ± 0.07 | 0.75 ± 0.07 |
| logistic | 0.68 ± 0.11 | 0.59 ± 0.09 | 0.71 ± 0.09 |
| logistic · jailbreak from 32 real prompts/class | 0.52 | 0.45 | 0.57 |

It is **not one number** — it depends on mode, concept, and model (range 0.45–0.83). Logistic is always
lower than diff-of-means (logistic rotates away from the class-mean difference by design — better
detector, worse proxy for the steering vector). Chance is 1/√d = 0.036/0.033/0.021, so every value is
13–36σ above chance *and* 38–63° off identity. Quote the range, not a point.

**`--decouple` refutes the "GPT-2 is decoupled" hypothesis.** The worry: if the write direction sits
~60° off the read direction, the detector would fail to register its own steered output on *any* model,
so §4.6's GPT-2 observation might be a geometric artifact rather than a capability ceiling. Tested by
steering, re-reading, and projecting onto both directions: the detection score moves *with* the steering
(gpt2 LLR +21, Qwen +48), and GPT-2 is NOT the worst-aligned model in any mode — Qwen is, every time (N32-logistic: Qwen 0.45 < gpt2 0.52 < gemma 0.57), and Qwen's read tracks its write fine. Were decoupling the cause of a weak read it would show on the worst-aligned model, and it does not. (An earlier version of this note said gpt2 was the *highest* in every mode — false: gemma is higher in 2 of 3 rows.) The capability-ceiling reading stands.

### 8. CORRECTED RUNS (chat template on) — §6 and §7 above are SUPERSEDED

An independent code audit (2026-09-03) found `apply_chat_template` was never called: every steering/gate
number in §6–7 came from Qwen2.5-0.5B-**Instruct** run as a raw completer. `ConceptGate.from_pretrained(...,
chat_template=True)` now formats prompts as a user turn + generation prompt for reading AND generation.
Re-run (`--gate --signflip --beavertails --logit`, gate in `Direction.LOGISTIC`, n=32+32, 3 seeds):

| arm (template on) | writes | jb refusal | Δ vs none | benign untouched |
|---|---|---|---|---|
| none | 0% | **93.8%** | — | 100% |
| always −α | 100% | 71.9% | **−21.9 ± 5.1** | 0% |
| gate −α | 97% | 75.0% | −18.8 ± 4.4 | 82% |
| random −α (same size) | 97% | 71.9% | −21.9 ± 5.1 | 82% |
| antigate −α | 3% | 90.6% | −3.1 ± 2.6 | 18% |
| gate +α | 97% | 96.9% | +3.1 | 82% |
| antigate +α | 3% | 93.8% | +0.0 (benign over-refusal 7.3%) | 18% |

- **Baseline refusal 93.8%** (was 46.9%) → two prompts of headroom; a refusal RATE is saturated.
- **THE SIGN REVERSES.** −α ("away", the old "safety write") LOWERS refusal by 22 pts. +α raises it (at
  ceiling on attacks; 0→7% over-refusal on benign). The old +8.3 was the raw-completer regime's artifact.
- **Gate fires on 96.9±2.6% of attacks** (was 54%) → gate ≈ random ≈ blanket; NOTHING to select. The
  "selection not dosage" result is gone. Gate still confines benign collateral (82% vs 0%) = CAST Table 3.
- Register: fires 17.7% in-register benign, **95.8%** real out-of-register benign. Template-fit concept
  (`fire_only`) now fires 77%/34% on short attacks/benign (was 0/0).
- **BeaverTails corrected:** refusal 80.0 → 65.0 (blanket) → 68.3 (gate); gates fire 42–100% on SAFE
  prompts; safe untouched 3% vs 33%. Same reversal; harm-topic lever weaker (15 vs 22 pts) — Zhao 2507.11878.

**`--logit` (continuous first-token refusal logit, one forward/arm, random-direction floor of matched norm):**

| Δ refusal logit | base | concept −α | +α | lever | random −α | +α | lever |
|---|---|---|---|---|---|---|---|
| fired attacks (n=93) | +2.94 | −1.63 | +0.89 | **−1.26** | −0.56 | +0.28 | −0.42 |
| benign (n=96) | −2.21 | −0.77 | +1.21 | **−0.99** | −0.21 | +0.33 | −0.27 |

The few-shot direction is a **lever everywhere** (same sign on attacks and benign, fired or passed), **~3×
a random direction** of the same norm (|lever| 1.26 vs 0.45; concept > random on 91% of prompts). No
lever/perturbation dissociation. = Arditi's refusal direction recovered few-shot + Safety Pitfalls. NOT new.
The random direction is not inert (Rogue Scalpel).

**Prior art the paper had missed:** CAST 2409.05907 (conditional steering), DSAS 2512.03661 (few-shot
logistic gate), Arditi 2406.11717, Rogue Scalpel 2509.22067, Safety Pitfalls 2603.24543, AlphaSteer
2506.07022 (CAST detector fires on ~all math), Zhao 2507.11878, Billa 2604.15557, Residual Paving
2605.20262, LAD 2604.28129. Report refs 15–24.

**Open question (running, `--steerability`):** does the few-shot concept read predict PER PROMPT how much
a write moves the model? Needs attacks that succeed (template + request), continuous outcome, random floor,
and an outcome-fitted gate vs the concept gate. Two lit checks found nobody has done it. Result pending.

**Lessons:** audit the CODE path, not just the paper (7 reviews missed it; 1 code read found it); never
report a binary outcome at a ceiling; always include a matched-norm random direction.

### 9. STEERABILITY PREDICTION — the one thing that looks new (`--steerability`)

Different question from §6–8. Not "is the concept present?" (on formatted attacks: always yes, useless for
the write) but **"how far will a write move THIS prompt?"** Needs attacks with headroom, so: 120 held-out
jailbreak templates + a harmful request each, 32 short framings, 12 bare requests (164 attacks, 15% lean
comply), 48 real benign. Outcome = first-token refusal logit; 5 forwards/prompt (none, ±α concept, ±α
random of matched norm); no generation. **Lever** = ½(Δ₋α − Δ₊α), the sign-reversible half. 3 concept
resamples (prompts fixed, so spread = sensitivity to the 8 fitting examples).

| predictor of the per-prompt lever | Spearman |
|---|---|
| gate LLR (the concept read), all attacks | +0.51 ± 0.11 |
| gate LLR, within the 120 templates only | +0.40 ± 0.18 |
| **ridge fit to the outcome, 5-fold CV** | **+0.81 ± 0.03** |
| same, folds grouped by harmful request | +0.79 ± 0.02 |
| same, templates only, grouped | +0.70 ± 0.07 |
| fit on templates → test short+bare | +0.88 ± 0.02 |
| fit on short+bare → test templates | +0.72 ± 0.03 |

**The outcome direction is NOT the concept direction:** mean |cos| 0.09 vs chance 1/√d = 0.033 (~85° apart).
**Nor a refusal-disposition direction:** a ridge on the *unsteered* logit hits +0.86 CV but sits |cos| 0.07
from the outcome direction. **Projecting BOTH out still gives +0.81 ± 0.04.**

Confounds ruled out: residual norm (norm→lever +0.20; per-unit-write CV +0.81), baseline refusal
(−0.01; residualized CV +0.79), distance to boundary (|base|→|lever| −0.01). Concept lever 1.28 ± 0.02 vs
random-direction 0.51 ± 0.20 (~2.5× the perturbation floor).

Two lit checks found no prior art: closest is Billa 2604.15557 (aggregate, one sign, no gate), ASTEER
2606.11599 (post-steering states, 1.4M generations, GBDT), Braun 2505.22637 (dataset-level). Self-estimated
P(already published) ≈ 1/3. **Limits:** one model, one concept, one α, 164 attacks, ridge in 2688 dims on
n=164 (grouped CV + cross-family transfer are the reason to believe it), labels cost 2 forwards/prompt,
correlational.

**Next if pursued:** several α, several concepts, 2B/7B, and a held-out MODEL not just a held-out fold.
Report §4.11 + Figure 16. Numbers all from `scripts/analyze_steerability.py`.

### 10. BEHAVIOURAL STEERABILITY — a reliability-tested instrument, then prediction (`eval_behaviour_dose.py`)

**Why.** §9's behavioural validation used two instruments whose reliability was never measured: one greedy
sample per arm scored by the lexicon (three possible values per prompt) and 3+3 canned continuations,
teacher-forced. Their −0.01 agreement said the instruments were noisy, not that the per-prompt quantity was
absent. Spec: `docs/plans/steerability-gate.md` (stop rules fixed before running).

**Setup.** The §9 prompt set unchanged: 120 held-out jailbreak templates + a harmful request, 32 short framed
attacks, 12 bare requests (164 attacks), 48 real benign. Concept `jailbreak` fit as §9 (seed 0, 8+8,
`Direction.LOGISTIC`), taps at 33/50/67% depth, **chat template on**, α = 0.08 of the residual norm. Five arms:
none, ±α along the concept, ±α along a random unit direction of matched norm. **K = 16 sampled continuations**
per (prompt, arm) at T = 0.7 (top_p 1, top_k 0), 40 tokens, in **float32** (bfloat16 decodes 3.5× slower on
MPS; the dtype is recorded in the results). Two scorers per continuation: the repo refusal lexicon (`lex`) and
`protectai/distilroberta-base-rejection-v1` (`clf`, label 1 = rejection, hard label at 0.5; the mean probability
is reported as `clf_soft`). **Dose** D = ½(P_refuse(+α) − P_refuse(−α)); the first-token proxy of §8–9 (the
measure is Logit-Gap Steering's per-prompt margin, 2506.24056, in basket-sum form) is recorded per arm on the
same run. All 16,960 continuations are stored in `scripts/behaviour_dose_results__<model>.json`; the analysis
(`scripts/analyze_behaviour_dose.py`) never regenerates. Engineering notes that cost a night: keep a generate
call ≤ 32 rows on long prompts (one 80-row call took 163 s and drove the MPS allocator to 17 GB; 32-row chunks
of the same work took 30 s), release the MPS cache per prompt, left-pad to 32-token buckets, and restart the
process every 10 prompts (`scripts/run_behaviour_dose.sh`) because the MPS graph cache grows with every new
shape and is never freed.

**Stage 1 — is the per-prompt behavioural dose measurable?** Pre-registered: split-half reliability ≥ 0.6 AND
lexicon-vs-classifier agreement ≥ 0.6. Split-half = Spearman between D from odd and from even samples, over
attacks; SB = Spearman–Brown estimate at the full K.

| model | instrument | split-half (K/2 halves) | SB(K=16) | P₊−P₀ split-half | unsteered-rate split-half | lex↔clf | proxy → dose | gate LLR → dose | dose / random | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | lex | **+0.66** | +0.80 | +0.33 | +0.76 | **+0.88** | +0.63 | −0.33 | 1.47 | **GO** |
| Qwen2.5-0.5B | clf | **+0.70** | +0.82 | +0.32 | +0.87 | | +0.61 | −0.30 | 1.67 | |
| Qwen2.5-0.5B | clf_soft (aux.) | +0.72 | +0.83 | +0.32 | +0.87 | | +0.62 | −0.31 | 1.67 | |
| gemma-2-2b | lex | +0.53 | +0.69 | +0.33 | +0.93 | **+0.68** | +0.47 | −0.16 | 1.48 | **NO-GO** at K=16 |
| gemma-2-2b | clf | +0.53 | +0.69 | +0.52 | +0.92 | | +0.41 | −0.06 | 1.72 | → K=32 running |

Refusal rates on attacks (clf): Qwen none 0.55, −α 0.35, +α 0.65, random −α 0.46, random +α 0.63; on benign
0.17, 0.14, 0.18, 0.17, 0.20; 87% of attack doses positive; mean |D| attacks 0.16–0.18, benign 0.06. **gemma**
(bf16): attacks none 0.51, −α 0.46, +α 0.53, random 0.50 / 0.50; benign 0.00–0.03 in every arm; only 38% of
doses positive; mean |D| attacks 0.07, benign 0.00. The write barely moves gemma at α = 0.08 — a quarter of
Qwen's swing — so at K = 16 the per-prompt dose sits at the sampling-noise floor even though the unsteered rate
itself is highly reliable (0.92) and the two instruments agree (0.68). Pre-registered fallback: a second
sampling seed of the SAME intervention (concept fit unchanged) to reach K = 32; Spearman–Brown predicts ~0.69.

Findings:
- **The per-prompt behavioural dose is measurable at K = 16** (0.66–0.70 raw, ~0.8 at full K), and two
  independent scorers agree on it (+0.88). §9's −0.01 was the instruments.
- **The first-token proxy tracks the sampled behavioural dose at +0.62** across 164 attacks — above the 0.5 bar
  §9's validation had set and failed (+0.43 / +0.48 with the greedy instruments). Per-prompt, the proxy is a
  usable but lossy stand-in for behaviour.
- **The gate's confidence anti-correlates with the dose (−0.30):** the surer the concept read, the less the write
  moves the prompt. Sign-consistent with §9 once conventions are aligned (there, a high LLR went with a lever
  nearer zero).
- **In behaviour the concept direction is only 1.5–1.7× a random direction of matched norm** (2.5× on the
  proxy). A random write at 8% of the residual norm raises attack refusal from 0.54 to 0.62 on its own —
  Rogue Scalpel in a sampled measure.
- The one-arm gain P₊ − P₀ is far less reliable (+0.32) than the two-arm dose; stage 3 below inherits that noise.

**Stage 2 — is the dose predictable from the unsteered prompt?** Pre-registered: grouped-CV Spearman ≥ 4 sd
above a 300-draw permutation null from the identical pipeline AND above |gate LLR → dose|, on Qwen AND gemma.
Ridge (α = 10) on the 3 tapped activations, z-scored, target = `clf` dose (the more reliable instrument), folds
grouped by harmful request (7 folds).

| model | ridge → dose, grouped CV (plain) | permutation null mean ± sd (z) | gate LLR | 3 concept projections | random-direction dose | unsteered rate (Spearman rate↔dose) | learning curve 8 / 16 / 32 / 64 | templates→other / reverse | verdict |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | **+0.58** (+0.60) | −0.015 ± 0.089 (**z 6.7**) | −0.30 | +0.51 | +0.23 | +0.81 (+0.03) | +0.33 / +0.46 / +0.52 / +0.58 | +0.52 / +0.43 | **GO** |
| gemma-2-2b (K=16, stage 1 failed) | +0.20 (+0.32) | −0.007 ± 0.092 (z 2.2) | −0.06 | +0.06 | −0.02 | +0.82 (−0.21) | +0.20 / +0.24 / +0.24 / +0.29 | +0.12 / +0.09 | NO-GO (provisional) |

Direction geometry (Qwen): |cos| to `W_raw` 0.04–0.12 per tap (chance 0.033); split-half self-consistency of
the fitted direction 0.10–0.22 — as in §9, too low to support any geometry claim.

**Robustness of the Qwen result to the two analysis choices that could have manufactured it** (the target's
scorer, and the ridge penalty). Neither is load-bearing:

| target instrument | grouped CV | plain CV | null z | \|gate LLR\| | 3 concept proj. | random-dir. dose | gate |
|---|---|---|---|---|---|---|---|
| lex (lexicon) | +0.54 | +0.50 | 6.0 | 0.33 | +0.52 | +0.21 | GO |
| clf (pre-registered) | +0.58 | +0.60 | 6.3 | 0.30 | +0.51 | +0.23 | GO |
| clf_soft (mean probability) | +0.60 | +0.61 | 6.5 | 0.31 | +0.53 | +0.22 | GO |

**Two surface confounds, both ruled out** — and these are strong tests, because the taps decode each confound
*easily*, so a dose that was secretly one of them would have shown up as a much higher CV, not a lower one:

| threat | is the dose it? | does the prediction survive? |
|---|---|---|
| **prompt length** — taps decode token count at CV **+0.90** | Spearman(length, dose) = **−0.12** | length-residualized dose still CV **+0.57** (raw +0.58) |
| **prompt family** — taps decode template-vs-other at CV **+0.77** | — | within templates only (n = 120) CV **+0.52**; within short+bare only (n = 44) CV **+0.63** |

So the ridge is not reading length, and it is not sorting prompt families: it predicts the dose *within* a
single family, on both families separately.

The two scorers' out-of-fold predictions agree at **+0.87**, so the ridge is decoding the same per-prompt
quantity whichever instrument labels it. Ridge penalty (clf target, grouped CV): α = 1 → +0.58, 10 → +0.58,
100 → +0.60, 1000 → +0.63 — flat to slightly *better* under heavier regularization, as expected at n = 164
against 2688 features, so the result is not a low-penalty overfit. Stage 3 repeated against the lexicon
target reproduces as well: outcome gate **+0.143** per write vs random halves +0.102 ± 0.017
(P = 0.016), anti-outcome +0.060, and the outcome gate writes to **10%** of benign prompts against the
concept gate's 100%.

Findings (Qwen):
- **The behavioural dose is predictable from the prompt's activations before any generation**, 6.7 sd above the
  null, at roughly the level the proxy result led one to expect once attenuated by the instrument's own
  reliability (+0.58 against a ceiling of ~0.8).
- **Most of the signal again lives in the concept direction:** three concept projections predict the dose at
  +0.51 where the calibrated LLR reaches only |0.30| — the gate is a lossy readout of information its own
  directions carry.
- **It is not disposition:** the unsteered refusal rate is even more predictable (+0.81) but is unrelated to the
  dose (+0.03). It is also more predictable than a random direction's dose (+0.23), so predictability per se is
  not the finding; the excess is.
- Eight labelled prompts already give +0.33; 64 give +0.58. The direction transfers across prompt families
  (+0.52 / +0.43).

**Stage 3 — the gate as selection** (Qwen only until gemma is in; predictions OUT OF FOLD, 50% coverage,
threshold = median OOF prediction; every arm is a selection over the same per-prompt gain ΔP₊ = P₊ − P₀ of the
sampled +α arm, so this adds no evidence beyond stage 2 — the random-halves null is the only genuine test):

| arm | attacks written | refusal gained per write | total | benign written | benign mean \|ΔP₊\| |
|---|---|---|---|---|---|
| blanket | 164 | +0.097 | +15.9 | 100% | 0.092 |
| concept gate (`Trigger.FIRE`) | 154 | +0.090 | +13.8 | **100%** | 0.092 |
| **outcome gate** (top half by predicted dose) | 82 | **+0.143** | +11.7 | **23%** | 0.074 |
| anti-outcome (bottom half) | 82 | +0.052 | +4.3 | 77% | 0.098 |
| both (fires AND top half) | 74 | +0.133 | +9.9 | 23% | 0.074 |
| random halves (500) | 82 | +0.097 ± 0.016 | | | P(random ≥ outcome) = **0.002** |

**The 50% operating point is not cherry-picked, and the predictor produces a ranking rather than a lucky
split.** Sweeping coverage (out-of-fold predictions; random null = 500 size-matched draws at each point):

| coverage | n written | dP per write (clf) | random mean | P(random ≥ gate) | benign written |
|---|---|---|---|---|---|
| 10% | 16 | **+0.242** | +0.099 | 0.002 | 2% |
| 25% | 41 | **+0.216** | +0.096 | 0.000 | 10% |
| 50% | 82 | **+0.143** | +0.098 | 0.008 | 23% |
| 75% | 123 | +0.129 | +0.097 | 0.000 | 52% |
| 90% | 148 | +0.108 | +0.097 | 0.014 | 85% |

The gate beats its size-matched null at **every** operating point (P ≤ 0.014), and the gain per write rises
monotonically as coverage tightens — 0.108 at 90% up to 0.242 at 10%, against a flat random baseline of
~0.098. That monotonicity is the real evidence: a lucky split would beat the null at one threshold, whereas an
actual ranking of prompts by responsiveness gets steadily better as you keep only the top of it. The lexicon
target reproduces the whole sweep (+0.246 → +0.114, P ≤ 0.020 throughout).

The outcome gate gets 73% of the blanket write's total refusal gain with half the writes, while the concept
gate writes to every benign prompt in this set (the §8 register problem) and the outcome gate to 23%. Caveat:
ΔP₊ is the noisy one-arm quantity (reliability +0.32); noise attenuates these gaps but cannot create the
selection effect, which is out of fold.

**A defect in this section's own stop rule, and what the K = 32 rerun will therefore show** (written
2026-09-12 20:05, *before* that run finished — check it against the result below rather than the other way
round). The stage 1 gate is "split-half ≥ 0.6", and split-half compares a K/2-sample estimate against another
K/2-sample estimate. **That statistic is not comparable across K.** At K = 16 it measures the reliability of an
8-sample dose; at K = 32 it measures the reliability of a *16*-sample dose, which for gemma is already known
from the K = 16 run via Spearman–Brown: 0.69. So gemma will very likely "pass" stage 1 at K = 32 —
mechanically, because the statistic got easier, not because the model's dose became more measurable. The
comparable quantities across K are the **Spearman–Brown-corrected full-K reliability** (gemma: 0.69 at K = 16
→ ~0.82 at K = 32; Qwen: 0.82 at K = 16) or a split-half at matched half-size.

Stage 2 is the test that is not fooled this way, because a more reliable target attenuates the correlation
less. Correcting gemma's K = 16 result for the reliability gain gives a **prediction: grouped CV ≈ +0.22,
z ≈ 2.4** — still far below the z ≥ 4 bar, which needs CV ≈ +0.36. Prediction, therefore: **stage 1 flips to
GO, stage 2 stays NO-GO**, and if that is what happens, the flip must not be reported as "gemma's dose became
measurable at K = 32".

**Status:** Qwen GO / GO / positive, and robust to the scorer and the ridge penalty (table above). gemma-2-2b
at K = 16: stage 1 NO-GO (dose reliability 0.53), stage 2 NO-GO (z 2.2) on that unreliable target; the
pre-registered K = 32 rerun (second sampling seed, same concept fit) started 2026-09-12 15:40 and is slowed by
this machine sleeping. The library change (`learn_outcome`, `Predicted`, `Both`) is built and tested on the
unmerged branch `wip/outcome-head` and lands only if gemma passes **stage 2** at K = 32 — a stage 1 flip alone
does not qualify, for the reason just given. On the evidence in hand the claim is **one model**.

## Verdict (honest)

Detection accuracy is a **commodity** — CG-logistic *ties* LR/SVM in-dist and shows no
cross-distribution edge (one confounded test). **Single-concept efficiency is NOT a CG contribution**:
a depth-matched probe (LR on the same taps, same truncated forward) equals CG-logistic at every depth,
so the "2–4× less compute" is only vs the full-model probe — the saving is the truncation, generic to any
latent probe. Do not frame it as CG's.

The efficiency claim that survives is at the **bank** level, and only vs *fine-tuning*: as a training-free
bank CG's cost is **flat/shallow in K** across a 14-category safety taxonomy (each concept a closed-form
fit in ms/kilobytes, all K read in one forward) where per-concept LoRA is **steep-linear** (30–38× the
whole-bank build, a separate forward each, lower few-shot accuracy). Honest caveat: a linear-probe bank
shares this amortization too — the **only** thing unique to CG is the read/write duality (steering).

Sharper still, post-§9: the composition has no novel *method*, but asking a different question of the same
harness produced one candidate finding — per-prompt steerability is linearly decodable from the prompt, and
from a direction ~85° off the concept's. That is what to build on, not the gate.

Post-§6: even *steering* is not unique — the write rule is CAA, which needs none of CG's
machinery. What is unique is **gate-conditioned** steering: using the calibrated read to decide *when*
to write. That is measured (§6) and it beats blanket steering on both suppression and collateral. Frame
the contribution there, not on steering per se.

Corrections worth remembering: (1) diff-of-means mode under-sells CG — always use `Direction.LOGISTIC`
when comparing to a classifier. (2) The cross-dist test was confounded by concept mismatch (jailbreak
framing vs harmful content). (3) The linear probe is a *trained* baseline (frozen backbone + head fit),
not few-shot; give both methods the same N examples but let each use its natural machinery (CG taps +
closed-form; probe full model + trained head).

---

## Plan (remaining)

1. **LoRA row** ✅ — done as the fine-tune anchor in the scaling eval (§4): 3 real per-concept fits
   per model; CG's no-gradient + shared-forward advantage widens with K exactly as expected. A full
   14-category LoRA sweep would tighten the mean but the shape is already clear.
2. **Clean within-concept OOD** ✅ — BeaverTails leave-one-category-out (§5 above): partial
   generalization, CG at least as robust as the probe.
3. **Steering / duality eval** ✅ — dose-response measured (`scripts/eval_steering.py`), and the
   read/write geometry measured (`--cosine`): related but NOT identical, 0.45–0.83 depending on mode,
   concept, and model.
4. **Gate-conditioned steering** ✅ (§6 above) — the one operation only the composition can do, and it
   wins on both axes. This is the differentiator; steering alone is CAA.
5. **BeaverTails bank steering** ✅ — measured, and it is a **NULL on behaviour** (§6): the bank's harm
   directions gate (leakily) and gating still confines the write, but steering away from a harm topic
   does not make the model decline. Report it as a negative.
6. **Remaining:** a non-guardrail (topical/science) dataset to show the method is not safety-specific;
   a content-level harm score (not just a refusal lexicon) and more than one steering magnitude before
   treating the BeaverTails null as general.

Status: in-dist detection ✅ · cross-dist ✅ (null, confounded) · **efficiency frontier ✅** ·
**multi-concept scaling ✅** · **within-concept OOD ✅** · **steering dose-response ✅** ·
**gate-conditioned steering ✅** · **read/write cosine ✅** · **BeaverTails bank steering ✅ (null)** ·
non-guardrail dataset — pending.

## Re-run commands
```bash
# in-dist ladder (CG vs LR/SVM on shared taps)
uv run --with datasets python scripts/eval_detection.py \
  --models gpt2,Qwen/Qwen2.5-0.5B-Instruct,HuggingFaceTB/SmolLM2-1.7B-Instruct,google/gemma-2-2b-it
# cross-distribution transfer
uv run --with datasets python scripts/eval_detection.py --cross \
  --models Qwen/Qwen2.5-0.5B-Instruct,google/gemma-2-2b-it
# CG vs full-model all-layer linear probe
uv run --with datasets python scripts/eval_detection.py --fullprobe \
  --models Qwen/Qwen2.5-0.5B-Instruct,google/gemma-2-2b-it
# multi-concept scaling (cost-vs-K over BeaverTails' 14 harm categories, + LoRA anchor)
uv run --with datasets --with peft --with transformers python scripts/eval_detection.py --scaling \
  --models Qwen/Qwen2.5-0.5B-Instruct,google/gemma-2-2b-it --ns 32 --seeds 0,1,2 \
  --lora-cats 3 --tap-fracs 0.5,0.7,0.85
# within-concept OOD (leave-one-category-out over BeaverTails; reuses the scaling cache)
uv run --with datasets python scripts/eval_detection.py --ood \
  --models Qwen/Qwen2.5-0.5B-Instruct,google/gemma-2-2b-it --ns 32 --seeds 0,1,2 --tap-fracs 0.5,0.7,0.85
# gate-conditioned steering (3 arms) + read/write cosine sweep + the decoupling test
uv run --with datasets python scripts/eval_gate.py            # all three
uv run --with datasets python scripts/eval_gate.py --cosine   # just the cosine table
```
Results: `scripts/eval_detection_results.json`, `eval_crossdist_results.json`, `eval_fullprobe_results.json`,
`eval_scaling_results.json`, `eval_ood_results.json`, `eval_gate_results.json`.
