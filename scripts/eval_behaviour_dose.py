"""Stage 1 of docs/plans/steerability-gate.md: a reliability-tested per-prompt behavioural dose.

For every prompt in the steerability set and every arm (none, +/-alpha along the few-shot
jailbreak direction, +/-alpha along a random direction of matched norm) sample K continuations
at a fixed temperature, score each for refusal with two instruments (the repo's lexicon and a
rejection classifier), and store everything -- texts included -- so the analysis never
regenerates. Also records the first-token refusal log-odds per arm (the report's proxy; the
measure is Logit-Gap Steering's per-prompt margin, 2506.24056, in basket-sum form) and the
unsteered tap activations, so stage 2 can predict the behavioural dose from the prompt alone.

MPS notes (M4, 16 GB, torch 2.12): float32 decodes ~3.5x faster than the checkpoint's bfloat16; keep a
generate call at <= 32 rows on long prompts -- on a 293-token prompt one 80-row call took 163 s and drove
the caching allocator to 17 GB (current_allocated stayed at 1.9 GB), while 32-row chunks of the same work
took 30 s in total; the cache is released after every prompt. Chunking changes which random draws are
realised, not their distribution: every row is an independent sample from P(text | prompt, arm), so the
split-half statistic is unaffected. Sampled texts are stored, so analysis never depends on regeneration.

Run:  uv run --with datasets python scripts/eval_behaviour_dose.py --dtype float32 --max-rows 32   # Qwen
      uv run --with datasets python scripts/eval_behaviour_dose.py --quick --no-classifier
      uv run --with datasets python scripts/eval_behaviour_dose.py --models google/gemma-2-2b-it
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_gate as E  # noqa: E402  (prompt set, few-shot examples, lexicon, token baskets)
from conceptgate import ConceptGate  # noqa: E402
from conceptgate.concept import Direction  # noqa: E402

ARMS = ("none", "minus", "plus", "rand_minus", "rand_plus")
CLASSIFIER = "protectai/distilroberta-base-rejection-v1"   # label index 1 = rejection
N_TEMPLATES, N_BENIGN = 120, 48


def tag_of(model: str) -> str:
    return model.replace("/", "__")


def random_direction(W_raw: np.ndarray, seed: int) -> np.ndarray:
    """Unit rows from the same generator as scaleup_eval, so the floor is comparable."""
    rr = np.random.default_rng([7000, seed])
    U = rr.normal(size=W_raw.shape)
    return U / np.linalg.norm(U, axis=-1, keepdims=True)


def arm_deltas(arm: str, W_raw: np.ndarray, U: np.ndarray, layers, mag: float):
    """Per-layer residual deltas for an arm; None for the unsteered arm."""
    if arm == "none":
        return None
    sign = -1.0 if arm.endswith("minus") else 1.0
    D = U if arm.startswith("rand") else W_raw
    return {int(layers[j]): sign * mag * D[j] for j in range(len(layers))}


def subsample_balanced(prompts, n):
    """First n prompts taken round-robin over kinds, so every kind stays represented."""
    if n is None or n >= len(prompts):
        return list(prompts)
    by: dict[str, list] = {}
    for kp in prompts:
        by.setdefault(kp[0], []).append(kp)
    kinds, out, i = list(by), [], 0
    while len(out) < n and any(by.values()):
        k = kinds[i % len(kinds)]
        if by[k]:
            out.append(by[k].pop(0))
        i += 1
    return out


def sample_seed(seed: int, prompt_index: int, arm_index: int) -> int:
    """SeedSequence, not addition: additive seeds collided across seeds once already."""
    return int(np.random.SeedSequence([seed, prompt_index, arm_index]).generate_state(1)[0])


def build_gate(model, taps, device, dtype, seed):
    cg = ConceptGate.from_pretrained(model, layers=list(taps), device=device, chat_template=True,
                                     dtype=dtype or None)
    rng = np.random.default_rng(seed)
    cg.learn("jailbreak",
             [E.FIT_POS[i] for i in rng.permutation(len(E.FIT_POS))[:E.N_SHOT]],
             [E.FIT_NEG[i] for i in rng.permutation(len(E.FIT_NEG))[:E.N_SHOT]],
             direction=Direction.LOGISTIC)
    cg.calibrate(z=2.0)
    return cg


def first_token_logodds(cg, prompt, deltas, R, C) -> float:
    """The report's proxy: logsumexp over refusal-opening ids minus compliance-opening ids."""
    h = cg._steer_hooks(deltas) if deltas else None
    try:
        ids = cg.tok(cg._format(prompt), return_tensors="pt").input_ids.to(cg.device)
        with torch.no_grad():
            lg = cg.model(input_ids=ids).logits[0, -1].float()
        return float(torch.logsumexp(lg[R], 0) - torch.logsumexp(lg[C], 0))
    finally:
        if h is not None:
            h.remove()


def sample_continuations(cg, prompt, deltas, k, temperature, max_new, seed) -> list[str]:
    """K sampled continuations under the arm's hooks, one batched generate call. Explicit
    sampling kwargs override the checkpoint's generation_config (Qwen ships top_p=0.8,
    top_k=20, repetition_penalty=1.05)."""
    h = cg._steer_hooks(deltas) if deltas else None
    try:
        ids = cg.tok(cg._format(prompt), return_tensors="pt").input_ids.to(cg.device)
        torch.manual_seed(seed)
        pad = cg.tok.pad_token_id if cg.tok.pad_token_id is not None else cg.tok.eos_token_id
        with torch.no_grad():
            out = cg.model.generate(ids, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                                    repetition_penalty=1.0, num_return_sequences=k,
                                    max_new_tokens=max_new, pad_token_id=pad)
        n = ids.shape[1]
        return [cg.tok.decode(o[n:], skip_special_tokens=True).strip() for o in out]
    finally:
        if h is not None:
            h.remove()


def bucket_pad(length: int, pad_to: int | None) -> int:
    """How many left-pad tokens bring `length` up to the next multiple of `pad_to` (0 if pad_to is None).
    Padding prompts to a few bucket lengths lets the MPS backend reuse compiled graphs across prompts
    instead of compiling a new one per sequence length."""
    if not pad_to or pad_to <= 0:
        return 0
    return (-length) % pad_to


def sample_all_arms(cg, prompt, deltas_by_arm, k, temperature, max_new, seed, max_rows=None,
                    pad_to=None) -> dict[str, list[str]]:
    """All arms in ONE generate call: rows [a*k:(a+1)*k] carry arm a's delta as a per-row [n_arms*k, 1, d]
    tensor (the steering hook broadcasts it over positions; the unsteered arm adds zeros). On MPS the
    per-step overhead dominates, so one 80-row call beats five 16-row calls."""
    all_arms = list(deltas_by_arm)
    per_call = max(1, (max_rows or len(all_arms) * k) // k)      # arms per generate call
    chunks = [all_arms[i:i + per_call] for i in range(0, len(all_arms), per_call)]
    layers = list(cg.layers)
    d = next(v for v in deltas_by_arm.values() if v is not None)
    ids = cg.tok(cg._format(prompt), return_tensors="pt").input_ids
    pad = cg.tok.pad_token_id if cg.tok.pad_token_id is not None else cg.tok.eos_token_id
    npad = bucket_pad(ids.shape[1], pad_to)
    attn = torch.ones_like(ids)
    if npad:
        ids = torch.cat([torch.full((1, npad), pad, dtype=ids.dtype), ids], 1)
        attn = torch.cat([torch.zeros((1, npad), dtype=attn.dtype), attn], 1)
    ids, attn = ids.to(cg.device), attn.to(cg.device)
    n = ids.shape[1]
    result = {}
    for ci, arms in enumerate(chunks):
        per_layer = {}
        for L in layers:
            rows = []
            for a in arms:
                vec = np.zeros_like(d[L]) if deltas_by_arm[a] is None else deltas_by_arm[a][L]
                rows += [vec] * k
            per_layer[L] = np.stack(rows)[:, None, :]
        h = cg._steer_hooks(per_layer)
        try:
            torch.manual_seed(seed + ci)
            with torch.no_grad():
                out = cg.model.generate(ids, attention_mask=attn, do_sample=True, temperature=temperature,
                                        top_p=1.0, top_k=0, repetition_penalty=1.0,
                                        num_return_sequences=len(arms) * k, max_new_tokens=max_new,
                                        pad_token_id=pad)
            texts = [cg.tok.decode(o[n:], skip_special_tokens=True).strip() for o in out]
            result.update({a: texts[i * k:(i + 1) * k] for i, a in enumerate(arms)})
        finally:
            h.remove()
            del out
            del_cache(cg.device)
    return result


def del_cache(device):
    """Return the caching allocator's blocks after each prompt: a 40-step, 80-row generate leaves the MPS
    allocator holding many differently-sized freed blocks, and two GPU processes on a 16 GB machine swap."""
    if str(device).startswith("mps") and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def mem_mb(device) -> float:
    if str(device).startswith("mps") and hasattr(torch, "mps"):
        return torch.mps.driver_allocated_memory() / 2**20
    if str(device).startswith("cuda"):
        return torch.cuda.memory_reserved() / 2**20
    return float("nan")


class RejectionClassifier:
    """protectai/distilroberta-base-rejection-v1 (DistilRoBERTa, 82M; archived, so a fixed
    instrument): P(label 1 = rejection) per continuation, continuation only, truncated to 512."""

    def __init__(self, name: str = CLASSIFIER, device: str = "cpu", batch_size: int = 64):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.name, self.device, self.bs = name, device, batch_size
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()

    @torch.no_grad()
    def score(self, texts: list[str]) -> list[float]:
        """Empty text scores 0.0: nothing was said, so nothing was refused (counted in n_empty)."""
        out = [0.0] * len(texts)
        idx = [i for i, t in enumerate(texts) if t.strip()]
        for s in range(0, len(idx), self.bs):
            b = idx[s:s + self.bs]
            enc = self.tok([texts[i] for i in b], return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(self.device)
            p = torch.softmax(self.model(**enc).logits.float(), -1)[:, 1].cpu().tolist()
            for i, pi in zip(b, p):
                out[i] = float(pi)
        return out


def run(model, device, *, k=16, alpha=0.08, temperature=0.7, max_new=40, seed=0, dtype="",
        max_prompts=None, use_classifier=True, clf_device="cpu", out_dir=HERE, resume=False,
        arm_batch=True, max_rows=None, pad_to=None):
    from eval_detection import taps_for
    taps, n_layers = taps_for(model)
    tag = tag_of(model)
    out_path = os.path.join(out_dir, f"behaviour_dose_results__{tag}.json")
    acts_path = os.path.join(out_dir, f"behaviour_dose_acts__{tag}.npy")
    wraw_path = os.path.join(out_dir, f"behaviour_dose_wraw__{tag}.npy")
    part_path, part_acts = out_path + ".partial", acts_path + ".partial.npy"

    prompts = subsample_balanced(E._steer_prompts(N_TEMPLATES, N_BENIGN)[0], max_prompts)
    print(f"\n{'=' * 92}\n### BEHAVIOUR DOSE {model}  taps {taps}/{n_layers}  K={k} T={temperature} "
          f"alpha={alpha} seed={seed}  {len(prompts)} prompts x {len(ARMS)} arms\n{'=' * 92}", flush=True)
    cg = build_gate(model, taps, device, dtype, seed)
    R, C = E._first_ids(cg.tok, E.REFUSAL_FIRST), E._first_ids(cg.tok, E.COMPLY_FIRST)
    W_raw = np.asarray(cg.concepts["jailbreak"].W_raw)
    U = random_direction(W_raw, seed)
    L = list(cg.layers)

    rows, acts = [], []
    if resume and os.path.exists(part_path) and os.path.exists(part_acts):
        rows = json.load(open(part_path))["rows"]
        acts = list(np.load(part_acts))
        assert len(rows) == len(acts), "partial files disagree; delete them and rerun"
        print(f"  resuming after {len(rows)} prompts", flush=True)
    start, t0 = len(rows), time.time()
    for i, (kind, pr) in enumerate(prompts):
        if i < start:
            continue
        v = cg.check(pr)
        mag = alpha * float(v.resid_norm)
        row = {"kind": kind, "prompt": pr, "llr": float(v.score), "fired": bool(v.fired),
               "p_present": float(v.p_present), "resid_norm": float(v.resid_norm),
               "logit": {}, "samples": {}, "lex": {}}
        deltas = {arm: arm_deltas(arm, W_raw, U, L, mag) for arm in ARMS}
        for arm in ARMS:
            row["logit"][arm] = first_token_logodds(cg, pr, deltas[arm], R, C)
        if arm_batch:
            samples = sample_all_arms(cg, pr, deltas, k, temperature, max_new, sample_seed(seed, i, 0),
                                      max_rows=max_rows, pad_to=pad_to)
        else:
            samples = {arm: sample_continuations(cg, pr, deltas[arm], k, temperature, max_new,
                                                 sample_seed(seed, i, ai)) for ai, arm in enumerate(ARMS)}
        for arm in ARMS:
            row["samples"][arm] = samples[arm]
            row["lex"][arm] = [int(E.is_refusal(t)) for t in samples[arm]]
        acts.append(cg._taps.read(cg.tok, [cg._format(pr)], cg.device, last_only=True)[0][0])
        rows.append(row)
        done = i + 1 - start
        if (i + 1) % 10 == 0 or i + 1 == len(prompts):
            el = time.time() - t0
            eta = el / done * (len(prompts) - i - 1)
            print(f"    {i + 1}/{len(prompts)}  {el / 60:.1f} min elapsed, ~{eta / 60:.1f} min left, "
                  f"gpu {mem_mb(cg.device):.0f} MB", flush=True)
            json.dump({"rows": rows}, open(part_path, "w"))
            np.save(part_acts, np.array(acts, dtype=np.float32))

    meta = {"model": model, "taps": L, "n_layers": n_layers, "alpha": alpha, "k": k,
            "temperature": temperature, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
            "max_new_tokens": max_new, "seed": seed, "dtype": str(cg.model.dtype), "chat_template": True,
            "direction": "logistic", "n_shot": E.N_SHOT, "arms": list(ARMS),
            "refusal_lexicon": E.REFUSAL, "refusal_first": E.REFUSAL_FIRST, "comply_first": E.COMPLY_FIRST,
            "classifier": CLASSIFIER if use_classifier else None,
            "prompt_kinds": [kk for kk, _ in prompts], "n_prompts": len(prompts)}
    meta["arm_batch"] = bool(arm_batch)
    meta["max_rows"] = max_rows
    meta["pad_to"] = pad_to
    cg.unload()
    meta["n_empty"] = int(sum(1 for r in rows for a in ARMS for t in r["samples"][a] if not t.strip()))
    meta["rows"] = rows
    json.dump(meta, open(out_path, "w"), indent=0)
    np.save(acts_path, np.array(acts, dtype=np.float32))
    np.save(wraw_path, W_raw.astype(np.float32))
    for p_ in (part_path, part_acts):
        if os.path.exists(p_):
            os.remove(p_)
    print(f"  -> {out_path}\n  -> {acts_path}\n  -> {wraw_path}", flush=True)
    if use_classifier:
        classify_file(out_path, clf_device)
    return out_path


def classify_file(out_path, clf_device="cpu"):
    """Second instrument as a separate pass over a saved results file (safe to rerun)."""
    meta = json.load(open(out_path))
    rows = meta["rows"]
    clf = RejectionClassifier(device=clf_device)
    flat = [(ri, arm) for ri in range(len(rows)) for arm in ARMS]
    texts = [t for ri, arm in flat for t in rows[ri]["samples"][arm]]
    print(f"  classifier {clf.name} over {len(texts)} continuations", flush=True)
    p, j = clf.score(texts), 0
    for ri, arm in flat:
        n_ = len(rows[ri]["samples"][arm])
        rows[ri].setdefault("clf", {})[arm] = [round(x, 4) for x in p[j:j + n_]]
        j += n_
    meta["classifier"] = clf.name
    json.dump(meta, open(out_path, "w"), indent=0)
    print(f"  -> {out_path} (with clf)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen/Qwen2.5-0.5B-Instruct", help="comma-separated")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.08)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="", help="e.g. bfloat16; default = the checkpoint's own")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--clf-device", default="cpu")
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--no-classifier", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from the .partial files")
    ap.add_argument("--no-arm-batch", action="store_true", help="one generate call per arm instead of one per prompt")
    ap.add_argument("--classify-only", action="store_true", help="only run the classifier over saved results")
    ap.add_argument("--max-rows", type=int, default=None, help="cap rows per generate call (arms are chunked)")
    ap.add_argument("--pad-to", type=int, default=None, help="left-pad prompts to a multiple of N tokens (shape reuse on MPS)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--quick", action="store_true", help="8 prompts, K=2, 8 new tokens -> scripts/quick/")
    a = ap.parse_args()
    if a.quick:
        a.max_prompts, a.k, a.max_new = 8, 2, 8
        a.out_dir = a.out_dir or os.path.join(HERE, "quick")
    a.out_dir = a.out_dir or HERE
    os.makedirs(a.out_dir, exist_ok=True)
    for m in a.models.split(","):
        if a.classify_only:
            classify_file(os.path.join(a.out_dir, f"behaviour_dose_results__{tag_of(m.strip())}.json"), a.clf_device)
            continue
        run(m.strip(), a.device, k=a.k, alpha=a.alpha, temperature=a.temperature, max_new=a.max_new,
            seed=a.seed, dtype=a.dtype, max_prompts=a.max_prompts, use_classifier=not a.no_classifier,
            clf_device=a.clf_device, out_dir=a.out_dir, resume=a.resume, arm_batch=not a.no_arm_batch,
            max_rows=a.max_rows, pad_to=a.pad_to)


if __name__ == "__main__":
    main()
