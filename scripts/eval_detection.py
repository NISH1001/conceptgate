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
from sklearn.svm import SVC, LinearSVC

from conceptgate import ConceptGate
from conceptgate import spectral as spec
from conceptgate.concept import BandpassConcept, Direction, _fit_directions

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
# Cross-distribution sources for the generalization axis. Each yields (prompt, label) with
# label 1 = "a prompt a guardrail should flag" (jailbreak / unsafe), 0 = benign. jackhhao is
# jailbreak *templates*; BeaverTails is plainly-asked *harmful requests* -- a real shift.
TEST_PER_CLASS = 150   # cap on the held-out test set per class (jackhhao test is smaller, kept whole)


def _source_rows(source):
    from datasets import load_dataset
    if source == "jackhhao":
        ds = load_dataset("jackhhao/jailbreak-classification")
        conv = lambda sp: [(r["prompt"].strip()[:CHAR_CAP], 1 if r["type"] == "jailbreak" else 0)
                           for r in ds[sp]]
        return conv("train"), conv("test")
    if source == "beavertails":
        ds = load_dataset("PKU-Alignment/BeaverTails")

        def conv(sp):  # BeaverTails repeats each prompt across responses -> dedup
            seen, out = set(), []
            for r in ds[sp]:
                p = r["prompt"].strip()
                if p in seen:
                    continue
                seen.add(p)
                out.append((p[:CHAR_CAP], 0 if str(r["is_safe"]) == "True" else 1))
            return out
        return conv("30k_train"), conv("30k_test")
    raise ValueError(f"unknown source {source}")


def load_source(source, seed=0):
    """(pool_txt, pool_y, test_txt, test_y) for one dataset: a balanced few-shot pool from train,
    a balanced held-out test set from the official test split."""
    rng = np.random.default_rng(seed)
    train, test = _source_rows(source)

    def balanced(rows, per_class):
        pos = [p for p, y in rows if y == 1]
        neg = [p for p, y in rows if y == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        k = min(per_class, len(pos), len(neg))
        txt = pos[:k] + neg[:k]
        return txt, np.r_[np.ones(k), np.zeros(k)].astype(int)

    pool_txt, pool_y = balanced(train, POOL_PER_CLASS)
    # jackhhao test is small + fixed -> keep whole; larger sets get a balanced cap
    if source == "jackhhao":
        test_txt = [p for p, _ in test]
        test_y = np.array([y for _, y in test])
    else:
        test_txt, test_y = balanced(test, TEST_PER_CLASS)
    return pool_txt, pool_y, test_txt, test_y


def load_data(seed=0):
    return load_source("jackhhao", seed)


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


def _fit_cg(Ap, An, direction):
    """Build a depth-bandpass gate on cached activations with a chosen detection direction.
    Replicates BandpassConcept.fit but lets us pick DIFF_OF_MEANS vs LOGISTIC (the fit method
    itself hardcodes diff-of-means)."""
    c = BandpassConcept()
    c.mu0, c.sd0, c.W, c.W_raw, S_pos, S_neg = _fit_directions(Ap, An, direction)
    c.f = spec.fit_bandpass(S_pos, S_neg, method=c.filter_method)
    sp, sn = spec.filtered_score(S_pos, c.f), spec.filtered_score(S_neg, c.f)
    c.mu_pos, c.sigma_pos = float(sp.mean()), float(sp.std(ddof=1))
    c.mu_neg, c.sigma_neg = float(sn.mean()), float(sn.std(ddof=1))
    return c


def det_conceptgate(Ap, An, Ate, yte):
    t = time.perf_counter()
    c = _fit_cg(Ap, An, Direction.DIFF_OF_MEANS)
    learn_ms = (time.perf_counter() - t) * 1e3
    m = _metrics(c.llr(Ate), yte)
    m["learn_ms"] = learn_ms
    return m


def det_conceptgate_log(Ap, An, Ate, yte):
    """ConceptGate with the per-layer LOGISTIC (discriminative, covariance-aware) direction --
    the fair, strongest-mode comparison against the joint logistic probe."""
    return _metrics(_fit_cg(Ap, An, Direction.LOGISTIC).llr(Ate), yte)


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


def _flat_clf(Ap, An, Ate, yte, clf):
    """Fit a discriminative classifier on the FULL flattened+standardized activation (all taps,
    all dims jointly) and score the test set -- the strong 'detection is a commodity' baselines."""
    Xtr = np.concatenate([Ap, An], 0).reshape(len(Ap) + len(An), -1)
    ytr = np.r_[np.ones(len(Ap)), np.zeros(len(An))]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf.fit((Xtr - mu) / sd, ytr)
    return _metrics(clf.decision_function((Ate.reshape(len(Ate), -1) - mu) / sd), yte)


def det_logistic(Ap, An, Ate, yte):   # vanilla L2-logistic regression
    return _flat_clf(Ap, An, Ate, yte, LogisticRegression(max_iter=2000, C=1.0))


def det_svm_linear(Ap, An, Ate, yte):
    return _flat_clf(Ap, An, Ate, yte, LinearSVC(C=1.0, max_iter=5000))


def det_svm_rbf(Ap, An, Ate, yte):
    return _flat_clf(Ap, An, Ate, yte, SVC(kernel="rbf", C=1.0))


DETECTORS = {"conceptgate": det_conceptgate, "conceptgate_log": det_conceptgate_log,
             "best_layer": det_best_layer, "logistic": det_logistic,
             "svm_linear": det_svm_linear, "svm_rbf": det_svm_rbf}


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
    print(f"  {'N':>3} | {'CG-diff':>8} {'CG-log':>8} {'LR':>8} {'SVM-lin':>8} {'SVM-rbf':>8} "
          f"{'best-L':>8}   (AUC)")
    for N, r in res["rows"].items():
        def cell(n):
            d = r.get(n)
            return f"{d['auc']:.3f}" if d else "  -- "
        print(f"  {N:>3} | {cell('conceptgate'):>8} {cell('conceptgate_log'):>8} {cell('logistic'):>8} "
              f"{cell('svm_linear'):>8} {cell('svm_rbf'):>8} {cell('best_layer'):>8}")


def bench_cross(model, sources, N, seeds, device):
    """Transfer matrix: learn a concept on each source's few-shot pool, test on every source's
    held-out split. The diagonal is in-distribution; off-diagonal is cross-distribution. The
    value question: does CG-logistic drop LESS than the full probe when train != test?"""
    taps, n_layers = taps_for(model)
    print(f"\n### CROSS-DIST {model}  (taps {'/'.join(map(str, taps))} of {n_layers}, N={N})", flush=True)
    cg = load_model(model, taps, device)
    data = {}
    for s in sources:
        pool_txt, pool_y, test_txt, test_y = load_source(s, 0)
        data[s] = {"poolA": extract(cg, pool_txt), "pool_y": pool_y,
                   "testA": extract(cg, test_txt), "test_y": test_y}
        print(f"   extracted {s}: pool {len(pool_txt)}, test {len(test_txt)}", flush=True)
    cg.unload()

    matrix = {}
    for tr in sources:
        pos = np.where(data[tr]["pool_y"] == 1)[0]
        neg = np.where(data[tr]["pool_y"] == 0)[0]
        for te in sources:
            cell = {}
            for name, fn in DETECTORS.items():
                aucs = []
                for sd in seeds:
                    rng = np.random.default_rng(2000 + sd)
                    ip = rng.choice(pos, N, replace=False)
                    ineg = rng.choice(neg, N, replace=False)
                    try:
                        aucs.append(fn(data[tr]["poolA"][ip], data[tr]["poolA"][ineg],
                                       data[te]["testA"], data[te]["test_y"])["auc"])
                    except Exception as e:
                        print(f"     {name} {tr}->{te} sd={sd} failed: {type(e).__name__}: {e}", flush=True)
                if aucs:
                    cell[name] = float(np.mean(aucs))
            matrix[f"{tr}->{te}"] = cell
    return {"model": model, "n_layers": n_layers, "taps": taps, "N": N,
            "sources": sources, "matrix": matrix}


def print_cross(res):
    print(f"\n{res['model']}  cross-distribution AUC (N={res['N']}/class)")
    dets = [("conceptgate_log", "CG-logistic"), ("conceptgate", "CG-diff"),
            ("logistic", "probe"), ("best_layer", "best-layer")]
    for key, cell in res["matrix"].items():
        tag = "  (in-dist)" if key.split("->")[0] == key.split("->")[1] else "  (CROSS)"
        nums = "  ".join(f"{lbl} {cell.get(k, float('nan')):.3f}" for k, lbl in dets)
        print(f"  {key:>26} | {nums}{tag}")
    # the headline: generalization drop (in-dist diagonal minus the mean cross for the same train)
    print("  --- generalization drop (in-dist AUC - cross AUC, same train source; smaller = more robust) ---")
    for tr in res["sources"]:
        ind = res["matrix"].get(f"{tr}->{tr}", {})
        cross = [res["matrix"][f"{tr}->{te}"] for te in res["sources"] if te != tr]
        for k, lbl in [("conceptgate_log", "CG-logistic"), ("logistic", "probe")]:
            if ind.get(k) is not None and cross:
                cx = np.mean([c[k] for c in cross if c.get(k) is not None])
                print(f"    train={tr:>12} {lbl:>11}: in-dist {ind[k]:.3f} -> cross {cx:.3f}  (drop {ind[k]-cx:+.3f})")


def _lr_auc(Xp, Xn, Xte, yte):
    """AUC of an L2-logistic probe on a single-layer feature block [N, d]."""
    Xtr = np.concatenate([Xp, Xn], 0)
    ytr = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=1.0).fit((Xtr - mu) / sd, ytr)
    return float(roc_auc_score(yte, clf.decision_function((Xte - mu) / sd)))


def bench_fullprobe(model, Ns, seeds, device):
    """The realistic baseline: CG (few mid-taps, few-shot, closed-form) vs a linear probe on the
    FROZEN FULL model -- swept over ALL layers (the strong standard probe) and on the last layer
    (the naive classification head). Same N for every method, so it's a fair matched comparison."""
    taps, n_layers = taps_for(model)
    all_layers = list(range(n_layers))
    print(f"\n### FULLPROBE {model}  ({n_layers} layers; CG taps {taps} vs full-model linear probe)", flush=True)
    cg = load_model(model, all_layers, device)     # load FULL model, tap every layer
    pool_txt, pool_y, test_txt, test_y = load_data(seed=0)
    poolA = extract(cg, pool_txt)                   # [N, n_layers, d]
    testA = extract(cg, test_txt)
    cg.unload()
    pos, neg = np.where(pool_y == 1)[0], np.where(pool_y == 0)[0]

    rows = {}
    for N in Ns:
        acc = {"cg_log": [], "linprobe_best": [], "linprobe_last": []}
        for sd in seeds:
            rng = np.random.default_rng(3000 + sd)
            ip = rng.choice(pos, N, replace=False)
            ineg = rng.choice(neg, N, replace=False)
            c = _fit_cg(poolA[ip][:, taps, :], poolA[ineg][:, taps, :], Direction.LOGISTIC)
            acc["cg_log"].append(_metrics(c.llr(testA[:, taps, :]), test_y)["auc"])
            acc["linprobe_best"].append(max(
                _lr_auc(poolA[ip][:, L, :], poolA[ineg][:, L, :], testA[:, L, :], test_y)
                for L in all_layers))
            acc["linprobe_last"].append(
                _lr_auc(poolA[ip][:, -1, :], poolA[ineg][:, -1, :], testA[:, -1, :], test_y))
        rows[N] = {k: float(np.mean(v)) for k, v in acc.items() if v}

    print(f"  {'N':>3} | {'CG-logistic (3 taps)':>20} {'linprobe-best (all L)':>22} {'linprobe-last':>14}   (AUC)")
    for N, r in rows.items():
        print(f"  {N:>3} | {r.get('cg_log', 0):>20.3f} {r.get('linprobe_best', 0):>22.3f} "
              f"{r.get('linprobe_last', 0):>14.3f}")
    return {"model": model, "n_layers": n_layers, "taps": taps, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ladder", help="'ladder', 'quick', or comma-separated names")
    ap.add_argument("--ns", default="4,8,16,32", help="few-shot sizes per class")
    ap.add_argument("--seeds", default="0,1,2", help="resample seeds to average over")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="scripts/eval_detection_results.json")
    ap.add_argument("--no-full-timing", action="store_true")
    ap.add_argument("--quick", action="store_true", help="Qwen-0.5B, N=8, one seed (smoke test)")
    ap.add_argument("--cross", action="store_true", help="cross-distribution transfer matrix")
    ap.add_argument("--sources", default="jackhhao,beavertails", help="datasets for --cross")
    ap.add_argument("--fullprobe", action="store_true", help="CG vs full-model all-layer linear probe")
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

    if a.fullprobe:
        out = "scripts/eval_fullprobe_results.json" if a.out == "scripts/eval_detection_results.json" else a.out
        print(f"fullprobe: {len(models)} model(s), N/cls={Ns}, seeds={seeds}", flush=True)
        results = []
        for model in models:
            try:
                results.append(bench_fullprobe(model, Ns, seeds, a.device))
            except Exception as e:
                print(f"  SKIP {model}: {type(e).__name__}: {e}", flush=True)
            with open(out, "w") as f:
                json.dump({"results": results}, f, indent=1)
        print(f"\ndone -> {out}", flush=True)
        return

    if a.cross:
        sources = [s.strip() for s in a.sources.split(",")]
        out = "scripts/eval_crossdist_results.json" if a.out == "scripts/eval_detection_results.json" else a.out
        print(f"cross-dist: {len(models)} model(s), sources={sources}, N={Ns[-1]}, seeds={seeds}", flush=True)
        results = []
        for model in models:
            try:
                res = bench_cross(model, sources, Ns[-1], seeds, a.device)
                results.append(res)
                print_cross(res)
            except Exception as e:
                print(f"  SKIP {model}: {type(e).__name__}: {e}", flush=True)
            with open(out, "w") as f:
                json.dump({"sources": sources, "results": results}, f, indent=1)
        print(f"\ndone -> {out}", flush=True)
        return

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
