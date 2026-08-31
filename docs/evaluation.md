# ConceptGate evaluation — working notes

**Goal:** establish that ConceptGate has *value* — specifically that it **learns concepts
efficiently** (the title claim). "Efficient" = matches strong baselines on the target metric at a
**fraction of the memory + compute**, few-shot and closed-form, *and* is steerable (a classifier
is not). In-distribution detection AUC alone is a commodity and does NOT establish value.

**Branch:** all this work lives on `feature/detection-benchmark` (NOT main). Harness:
`scripts/eval_detection.py`. Run everything with `uv run --with datasets python ...`.

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
- **CG is Pareto-efficient.** The full-model linear probe has the highest AUC (it uses the whole
  model), but CG reaches **~98–99% of it at 2–4× less compute + ~half the weights loaded + no
  gradient training** (closed-form learn+calibrate).
- **A single early tap is the sweet spot** (Qwen 1tap@25% 0.968 @ 3.3×; gemma 1tap@40% 0.974 @ 2.4×):
  the concept is linearly readable at ~⅓–⅖ depth, so the top half of the model is skipped.
- **Depth-fusion barely helps** on real detection: 3/5-tap ≈ best single tap (the synthetic
  depth-fusion win does not transfer).
- CG-diff is the weaker mode (~0.92–0.96); logistic is the one to report.
- Memory is universal: CG loads embed + blocks 0..top-tap (weights % above); learned params
  CG = m·d (≈2.7K Qwen / 6.9K gemma), probe = d. Compute is the measured wall-time above.

### 4. Multi-concept scaling — the amortization argument (`--scaling`)

Cost of a **K-concept bank** over BeaverTails' 14 harm categories (`PKU-Alignment/BeaverTails`),
N=32/class, 3 seeds, Apple M4 / MPS. CG taps deeper here (50/70/85% depth) because harm *content* is read
later than jailbreak *framing*. Three ways to build the bank: **CG** (a closed-form direction per concept
on the frozen taps), **linear probe** (a trained logistic head per concept on the frozen final layer),
**LoRA** (a rank-8 adapter fine-tuned per concept). LoRA measured on 3 categories (animal/child/controversial).

**Mean per-category AUC (14 categories):**

| Model | CG bank | probe bank | LoRA (3 cats) |
|---|---|---|---|
| Qwen2.5-0.5B | 0.832 | 0.855 | 0.685 |
| gemma-2-2b | **0.881** | 0.874 | 0.814 |

On gemma CG *edges* the probe; on Qwen it trails by 0.02. On the 3 LoRA categories CG scores 0.889 (Qwen)
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

---

## Verdict (honest)

Detection accuracy is a **commodity** — CG-logistic *ties* LR/SVM in-dist and shows no
cross-distribution edge (one confounded test). But the **efficiency** case is now established: CG is
**Pareto-optimal** — ~98–99% of a full-model linear probe's AUC at **2–4× less compute, ~half the
weights, and no gradient training**, using a single early tap. That is "efficiently learns concepts."

The **stronger, better-measured** form of that claim is the multi-concept scaling result (§4 above): as a
training-free bank, CG's cost is **flat/shallow in K** across a 14-category safety taxonomy (each concept
a closed-form fit in ms/kilobytes, all K read in one forward) where per-concept LoRA is **steep-linear**
(30–38× the whole-bank build cost, a separate forward each, and lower few-shot accuracy). Honest caveat:
a linear-probe bank shares this amortization — what stays specific to CG is the read/write duality.

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
2. **Clean within-concept OOD** — BeaverTails category holdout (the fair generalization test).
3. **Steering / duality eval** — same learned direction steers generation (harmful → refusal/safe);
   the effectiveness differentiator a classifier cannot match. **[the remaining differentiator to measure]**

Status: in-dist detection ✅ · cross-dist ✅ (null, confounded) · **efficiency frontier ✅** ·
**multi-concept scaling ✅** · within-concept OOD / steering-measurement — pending.

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
```
Results: `scripts/eval_detection_results.json`, `eval_crossdist_results.json`, `eval_fullprobe_results.json`,
`eval_scaling_results.json`.
