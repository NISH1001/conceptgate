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

### 3. CG (3 taps) vs full-model linear probe (`--fullprobe`, AUC)

| | N=4 | N=8 | N=16 | N=32 |
|---|---|---|---|---|
| Qwen CG-log (3 taps) | 0.869 | 0.941 | 0.944 | 0.987 |
| Qwen linprobe-best (all 24 L) | 0.926 | 0.976 | 0.970 | 0.991 |
| Qwen linprobe-last | 0.921 | 0.969 | 0.967 | 0.982 |
| gemma CG-log (3 taps) | 0.814 | 0.939 | 0.926 | 0.982 |
| gemma linprobe-best (all 26 L) | 0.914 | 0.962 | 0.959 | 0.986 |
| gemma linprobe-last | 0.897 | 0.960 | 0.959 | 0.984 |

Findings (AUC dimension only — the point is the OTHER dimensions):
- The full-model probe (sweeps ALL layers) **beats CG's 3 taps at low N** (~0.05–0.10 at N≤16) and
  **ties at N=32** (~0.004). So on accuracy alone, CG does not win — a probe with the whole frozen
  model is a touch better, converging to a tie with more data.
- Even the naive **last-layer** probe is strong (0.92–0.98) — jailbreak-template detection is easy.
- **CG's case is therefore efficiency, not accuracy:** it reaches ~probe accuracy using a
  **truncated forward to ~2/3 depth + 3 taps + no layer sweep + closed-form learn**, vs the probe's
  **full forward + all-layer sweep + LR fit per layer**. That gap must be quantified (memory +
  compute) to make the "efficient" claim — see Plan #1. Accuracy parity at N=32 + big compute/memory
  savings would be the value statement.
- (CG-log numbers here differ slightly from §1 — independent random subsamples/seeds; N=16 dip is
  seed noise across 3 seeds.)

---

## Verdict so far (honest)

Detection is a **commodity**: with its logistic direction CG *matches* LR/SVM in-dist (never beats
them) and shows **no** generalization edge in the one cross-test run. So detection numbers keep
confirming commodity-ness, not value. The value case must come from **(a) efficiency** — same
metric at far less memory/compute (the `--fullprobe` + memory/compute comparison, and vs LoRA) —
and **(b) steering/duality**, the thing a classifier structurally cannot do (not yet measured).

Two prior corrections worth remembering: (1) I first ran CG in diff-of-means mode and wrongly
concluded it "loses to LR" — it does not; `Direction.LOGISTIC` ties LR/SVM. (2) The cross-dist test
was confounded by concept mismatch.

---

## Plan (the comprehensive eval that establishes value)

1. **CG vs linear-probe vs LoRA — ALL dimensions**, not just AUC:
   - **target metric** (detection AUC / recall@FPR),
   - **memory** (weights loaded: CG loads only up to max tap; probe/LoRA need the full model; +
     learned-param count: CG = m·d direction, probe = per-layer head, LoRA = rank·(…) adapter),
   - **compute** (CG truncated forward to max tap vs full forward for probe/LoRA; learn cost:
     CG closed-form vs probe LR-fit vs LoRA training steps),
   - **sample efficiency** (the N-curve).
   This is the table that backs "efficiently learn." LoRA = the fine-tuning end of the adaptation
   spectrum; needs a small training loop (PEFT) to fit a concept adapter for classification.
2. **Clean within-concept OOD** — BeaverTails category holdout (the fair generalization test).
3. **Steering / duality eval** — on gemma-2, show the same learned direction steers generation
   (harmful continuation → refusal/safe), which the probe/LoRA-classifier cannot do. The real
   differentiator.

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
```
Results: `scripts/eval_detection_results.json`, `eval_crossdist_results.json`, `eval_fullprobe_results.json`.
