"""P1/P2 detection benchmark: ConceptGate vs baselines, few-shot, across a model ladder.

The thesis this measures: detection accuracy is a commodity (ConceptGate should TIE trained
baselines on AUC), but concept learning wins on SAMPLE EFFICIENCY and COST -- a usable detector
from a handful of examples, closed-form, no training. So we report AUC/PR-AUC (threshold-free),
recall/FPR at a z=2 operating point, a sample-efficiency curve (perf vs #examples), and learn +
detect cost -- for three detectors sharing the SAME tapped activations:

  ConceptGate   few-shot depth-bandpass gate (BandpassConcept = the gate at few-shot, J=1)
  best-layer(A) diff-of-means at the single most discriminative tap (the single-layer baseline)
  logistic(B)   L2-logistic probe on the concatenated per-tap activations (a trained probe)

Data: jackhhao/jailbreak-classification (public, balanced, official train/test split). The few-shot
examples are drawn from a seeded train pool; every detector is scored on the untouched test split.

Run:  uv run --with datasets python scripts/eval_detection.py            # full ladder
      uv run --with datasets python scripts/eval_detection.py --quick    # Qwen-0.5B smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # everything is cached; never hit the network

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from conceptgate import ConceptGate
from conceptgate.concept import BandpassConcept

LADDER = [
    "gpt2",                                  # 124M -- the "model is the ceiling" floor
    "Qwen/Qwen2.5-0.5B-Instruct",            # 0.5B -- already known to separate jailbreak cleanly
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",   # 1.7B
    "google/gemma-2-2b-it",                  # 2.6B
    "google/gemma-3-4b-it",                  # 4B (multimodal; skipped if it won't load as text)
]
DATASET = "jackhhao/jailbreak-classification"
CHAR_CAP = 1200          # prompts are long DAN templates; cap keeps the persona start, bounds cost
POOL_PER_CLASS = 64      # few-shot pool drawn from train; test is the untouched official split


# ----------------------------- data -----------------------------
def load_data(seed=0):
    from datasets import load_dataset
    ds = load_dataset(DATASET)

    def rows(split):
        return [(r["prompt"].strip()[:CHAR_CAP], 1 if r["type"] == "jailbreak" else 0)
                for r in ds[split]]

    rng = np.random.default_rng(seed)
    train, test = rows("train"), rows("test")
    pos = [p for p, y in train if y == 1]
    neg = [p for p, y in train if y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    pool_txt = pos[:POOL_PER_CLASS] + neg[:POOL_PER_CLASS]
    pool_y = np.r_[np.ones(POOL_PER_CLASS), np.zeros(POOL_PER_CLASS)].astype(int)
    test_txt = [p for p, _ in test]
    test_y = np.array([y for _, y in test])
    return pool_txt, pool_y, test_txt, test_y


# --------------------------- taps -------------------------------
def taps_for(model):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model)
    n = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    if n is None and getattr(cfg, "text_config", None) is not None:
        tc = cfg.text_config
        n = getattr(tc, "num_hidden_layers", None) or getattr(tc, "n_layer", None)
    if not n:
        raise ValueError(f"cannot determine layer count for {model}")
    return sorted({max(1, round(n * f)) for f in (0.33, 0.50, 0.67)}), n


# ------------------------ detectors -----------------------------
def _metrics(s_te, y_te):
    """Threshold-free: AUC, PR-AUC, and recall at fixed FPR (5% and 1%) from the test ROC.

    recall@FPR is the standard guardrail operating point -- it doesn't depend on the noisy
    few-shot threshold calibration (which does not transfer from ~8 benign examples), and it
    is read the same way for every detector, so the comparison is fair.
    """
    fpr, tpr, _ = roc_curve(y_te, s_te)

    def recall_at(target):
        ok = fpr <= target
        return float(tpr[ok].max()) if ok.any() else 0.0

    return {
        "auc": float(roc_auc_score(y_te, s_te)),
        "ap": float(average_precision_score(y_te, s_te)),
        "r_at_5": recall_at(0.05),
        "r_at_1": recall_at(0.01),
    }


def det_conceptgate(Ap, An, Ate, yte):
    t = time.perf_counter()
    c = BandpassConcept().fit(Ap, An)
    learn_ms = (time.perf_counter() - t) * 1e3
    m = _metrics(c.llr(Ate), yte)
    m["learn_ms"] = learn_ms
    return m


def det_best_layer(Ap, An, Ate, yte):
    X = np.concatenate([Ap, An], 0)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Zp, Zn, Zte = (Ap - mu) / sd, (An - mu) / sd, (Ate - mu) / sd
    best_L, best_dp, best_w = 0, -1.0, None
    for L in range(Ap.shape[1]):
        w = Zp[:, L].mean(0) - Zn[:, L].mean(0)
        w /= np.linalg.norm(w) + 1e-8
        sp, sn = Zp[:, L] @ w, Zn[:, L] @ w
        dp = abs(sp.mean() - sn.mean()) / (0.5 * (sp.std() + sn.std()) + 1e-8)
        if dp > best_dp:
            best_dp, best_L, best_w = dp, L, w
    return _metrics(Zte[:, best_L] @ best_w, yte)


def det_logistic(Ap, An, Ate, yte):
    Xtr = np.concatenate([Ap, An], 0).reshape(len(Ap) + len(An), -1)
    ytr = np.r_[np.ones(len(Ap)), np.zeros(len(An))]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    s_te = clf.decision_function((Ate.reshape(len(Ate), -1) - mu) / sd)
    return _metrics(s_te, yte)


DETECTORS = {"conceptgate": det_conceptgate, "best_layer": det_best_layer, "logistic": det_logistic}


# --------------------------- run --------------------------------
def load_model(model, taps, device):
    try:
        return ConceptGate.from_pretrained(model, layers=taps, device=device)
    except Exception as e:
        if device != "cpu":
            print(f"   {device} load failed ({type(e).__name__}: {e}); retrying on cpu", flush=True)
            return ConceptGate.from_pretrained(model, layers=taps, device="cpu")
        raise


def extract(cg, prompts, bs=16):
    return cg._taps.read(cg.tok, prompts, cg.device, last_only=True, batch_size=bs)[0]


def bench_model(model, Ns, seeds, device, time_full=True):
    taps, n_layers = taps_for(model)
    print(f"\n### {model}  (taps {'/'.join(map(str, taps))} of {n_layers}, device {device})", flush=True)
    cg = load_model(model, taps, device)
    dev = cg.device

    pool_txt, pool_y, test_txt, test_y = load_data(seed=0)
    t = time.perf_counter()
    pool_A = extract(cg, pool_txt)
    test_A = extract(cg, test_txt)
    extract_s = time.perf_counter() - t
    detect_ms = extract_s / (len(pool_txt) + len(test_txt)) * 1e3
    pos_idx = np.where(pool_y == 1)[0]
    neg_idx = np.where(pool_y == 0)[0]

    # cost: truncated (default) vs full forward on a small sample
    speedup = None
    if time_full:
        sub = test_txt[:24]
        t = time.perf_counter(); extract(cg, sub, bs=8); t_tr = time.perf_counter() - t
        t = time.perf_counter(); cg._taps.read(cg.tok, sub, dev, last_only=True, batch_size=8, full=True)
        t_full = time.perf_counter() - t
        speedup = t_full / max(t_tr, 1e-6)

    rows = {}
    for N in Ns:
        rows[N] = {}
        for name, fn in DETECTORS.items():
            accs = []
            for s in seeds:
                rng = np.random.default_rng(1000 + s)
                ip = rng.choice(pos_idx, N, replace=False)
                ineg = rng.choice(neg_idx, N, replace=False)
                try:
                    accs.append(fn(pool_A[ip], pool_A[ineg], test_A, test_y))
                except Exception as e:
                    print(f"     {name} N={N} seed={s} failed: {type(e).__name__}: {e}", flush=True)
            if accs:
                rows[N][name] = {k: float(np.mean([a[k] for a in accs]))
                                 for k in ("auc", "ap", "r_at_5", "r_at_1")}
                rows[N][name]["auc_std"] = float(np.std([a["auc"] for a in accs]))
                if name == "conceptgate":
                    rows[N][name]["learn_ms"] = float(np.mean([a.get("learn_ms", 0.0) for a in accs]))

    cg.unload()
    return {"model": model, "taps": taps, "n_layers": n_layers, "device": str(dev),
            "detect_ms_per_prompt": detect_ms, "trunc_vs_full_speedup": speedup, "rows": rows}


def print_table(res):
    first = res["rows"][next(iter(res["rows"]))].get("conceptgate", {}) if res["rows"] else {}
    print(f"\n{res['model']}  |  detect {res['detect_ms_per_prompt']:.1f} ms/prompt"
          + (f"  |  trunc {res['trunc_vs_full_speedup']:.2f}x vs full" if res["trunc_vs_full_speedup"] else "")
          + (f"  |  CG learn {first['learn_ms']:.1f} ms" if first.get("learn_ms") else ""))
    print(f"  {'N/cls':>5} | {'ConceptGate':>13} {'best-layer(A)':>13} {'logistic(B)':>13}   "
          f"(cell = AUC  recall@5%FPR)")
    for N, r in res["rows"].items():
        def cell(n):
            d = r.get(n)
            return f"{d['auc']:.3f} {d['r_at_5']:.2f}" if d else "    --    "
        print(f"  {N:>5} | {cell('conceptgate'):>13} {cell('best_layer'):>13} {cell('logistic'):>13}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ladder", help="'ladder', 'quick', or comma-separated names")
    ap.add_argument("--ns", default="4,8,16,32", help="few-shot sizes per class")
    ap.add_argument("--seeds", default="0,1,2", help="resample seeds to average over")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="scripts/eval_detection_results.json")
    ap.add_argument("--no-full-timing", action="store_true")
    ap.add_argument("--quick", action="store_true", help="Qwen-0.5B, N=8, one seed (smoke test)")
    a = ap.parse_args()

    if a.quick:
        a.models = "quick"
    if a.models == "ladder":
        models = LADDER
    elif a.models == "quick":
        models, a.ns, a.seeds = ["Qwen/Qwen2.5-0.5B-Instruct"], "8", "0"
    else:
        models = [m.strip() for m in a.models.split(",")]
    Ns = [int(x) for x in a.ns.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]

    print(f"benchmark: {len(models)} model(s), N/cls={Ns}, seeds={seeds}, device={a.device}", flush=True)
    results = []
    t0 = time.time()
    for model in models:
        try:
            res = bench_model(model, Ns, seeds, a.device, time_full=not a.no_full_timing)
            results.append(res)
            print_table(res)
        except Exception as e:
            print(f"  SKIP {model}: {type(e).__name__}: {e}", flush=True)
        with open(a.out, "w") as f:
            json.dump({"dataset": DATASET, "results": results}, f, indent=1)
    print(f"\ndone in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
