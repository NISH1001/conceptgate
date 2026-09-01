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
    """Test AUC of an L2-logistic probe fit on a feature block ([N, d] or [N, m, d] -> flattened)."""
    Xp, Xn, Xte = Xp.reshape(len(Xp), -1), Xn.reshape(len(Xn), -1), Xte.reshape(len(Xte), -1)
    Xtr = np.concatenate([Xp, Xn], 0)
    ytr = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=1.0).fit((Xtr - mu) / sd, ytr)
    return float(roc_auc_score(yte, clf.decision_function((Xte - mu) / sd)))


def _svm_auc(Xp, Xn, Xte, yte):
    """Test AUC of a linear-SVM probe (frozen base + linear-SVC head) on a feature block."""
    Xp, Xn, Xte = Xp.reshape(len(Xp), -1), Xn.reshape(len(Xn), -1), Xte.reshape(len(Xte), -1)
    Xtr = np.concatenate([Xp, Xn], 0)
    ytr = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LinearSVC(C=1.0, max_iter=5000).fit((Xtr - mu) / sd, ytr)
    return float(roc_auc_score(yte, clf.decision_function((Xte - mu) / sd)))


def _fullprobe_extract(cg, prompts, model, split):
    """Last-token activations [N, len(layers), d], cached to disk (keyed by the exact layer set so
    different tap configs never collide) so re-runs are instant."""
    sig = "-".join(map(str, cg._taps.layers))
    path = f"/tmp/cg_fullprobe__{model.replace('/', '__')}__{split}__L{sig}.npy"
    if os.path.exists(path):
        A = np.load(path)
        if A.shape[0] == len(prompts):
            return A
    A = extract(cg, prompts)
    np.save(path, A)
    return A


POOL_BIG_PER_CLASS = 256   # big balanced pool so N sweeps from few-shot up to a fully-trained head


def _jackhhao_probe_split(n_per_class):
    """Balanced training set (up to n_per_class/class from the jackhhao TRAIN split) + official test."""
    from datasets import load_dataset
    ds = load_dataset("jackhhao/jailbreak-classification")
    rng = np.random.default_rng(0)
    tr = [(r["prompt"].strip()[:CHAR_CAP], 1 if r["type"] == "jailbreak" else 0) for r in ds["train"]]
    pos = [p for p, y in tr if y == 1]
    neg = [p for p, y in tr if y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n_per_class, len(pos), len(neg))
    train_txt = pos[:k] + neg[:k]
    train_y = np.r_[np.ones(k), np.zeros(k)].astype(int)
    test_txt = [r["prompt"].strip()[:CHAR_CAP] for r in ds["test"]]
    test_y = np.array([1 if r["type"] == "jailbreak" else 0 for r in ds["test"]])
    return train_txt, train_y, test_txt, test_y


def bench_fullprobe(model, Ns, seeds, device):
    """Exactly the same data for both -- the SAME N few-shot examples, the same held-out test set.
    The only differences are the method and how much of the model each uses:
      linear probe: the FULL model (final-layer representation) + a linear head fit on the N examples.
      CG:           only UP TO its taps (~2/3 depth) + a closed-form direction on the N examples.
    Both scored on the same test split. Shows CG matching a full-model linear probe while touching far
    less of the model and doing no gradient training (its taps are ~depth_fraction of the layers)."""
    taps, n_layers = taps_for(model)
    fin, top_tap = n_layers - 1, max(taps)
    layers = sorted(set(taps + [fin]))               # final layer (for the probe) forces a full forward
    fin_i, tap_i = layers.index(fin), [layers.index(t) for t in taps]
    depth = (top_tap + 1) / n_layers
    print(f"\n### FULLPROBE {model}  (same N examples; CG up-to-tap {top_tap}/{n_layers}={depth:.0%} "
          f"vs linear probe on the full model)", flush=True)
    cg = load_model(model, layers, device)
    pool_txt, pool_y, test_txt, test_y = _jackhhao_probe_split(POOL_BIG_PER_CLASS)
    A_pool = _fullprobe_extract(cg, pool_txt, model, f"pool{POOL_BIG_PER_CLASS}")
    A_te = _fullprobe_extract(cg, test_txt, model, "test")
    cg.unload()
    pos_i, neg_i = np.where(pool_y == 1)[0], np.where(pool_y == 0)[0]

    rows = {}
    for N in Ns:
        if N > min(len(pos_i), len(neg_i)):
            continue
        cgd, cgl, lp, sv, lpt = [], [], [], [], []
        for sd in seeds:
            r = np.random.default_rng(3000 + sd)
            ip, ineg = r.choice(pos_i, N, replace=False), r.choice(neg_i, N, replace=False)
            Xp, Xn, Xt = A_pool[ip][:, tap_i, :], A_pool[ineg][:, tap_i, :], A_te[:, tap_i, :]
            Fp, Fn, Ft = A_pool[ip][:, fin_i, :], A_pool[ineg][:, fin_i, :], A_te[:, fin_i, :]
            cgd.append(_metrics(_fit_cg(Xp, Xn, Direction.DIFF_OF_MEANS).llr(Xt), test_y)["auc"])
            cgl.append(_metrics(_fit_cg(Xp, Xn, Direction.LOGISTIC).llr(Xt), test_y)["auc"])
            lp.append(_lr_auc(Fp, Fn, Ft, test_y))       # probe on the FINAL layer (full forward)
            lpt.append(_lr_auc(Xp, Xn, Xt, test_y))      # DEPTH-MATCHED probe: same taps as CG, same compute
            sv.append(_svm_auc(Fp, Fn, Ft, test_y))
        sd_ = lambda a: round(float(np.std(a)), 3)
        rows[N] = {"cg_diff": float(np.mean(cgd)), "cg_logistic": float(np.mean(cgl)),
                   "linprobe": float(np.mean(lp)), "linprobe_svm": float(np.mean(sv)),
                   "linprobe_tap": float(np.mean(lpt)),
                   "std": {"cg_logistic": sd_(cgl), "linprobe": sd_(lp), "linprobe_tap": sd_(lpt)}}

    print(f"  {'N':>4} | {'CG-log':>8} {'probe@tap':>10} {'probe@final':>12} {'SVM@final':>10}   (AUC; probe@tap = depth-matched)")
    for N, r in rows.items():
        print(f"  {N:>4} | {r['cg_logistic']:>8.3f} {r['linprobe_tap']:>10.3f} "
              f"{r['linprobe']:>12.3f} {r['linprobe_svm']:>10.3f}", flush=True)
    return {"model": model, "n_layers": n_layers, "taps": taps, "top_tap": top_tap,
            "depth_fraction": round(depth, 3), "rows": rows}


def _chip():
    import subprocess
    for key in ("machdep.cpu.brand_string", "hw.model"):
        try:
            return subprocess.check_output(["sysctl", "-n", key]).decode().strip()
        except Exception:
            continue
    return "unknown"


def _model_params(model):
    """Universal (device-independent) parameter accounting from the config."""
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model)
    tc = getattr(c, "text_config", None)
    g = lambda k, dflt=None: getattr(c, k, None) or (getattr(tc, k, None) if tc else None) or dflt
    n, d, V, I = g("num_hidden_layers"), g("hidden_size"), g("vocab_size", 0), g("intermediate_size")
    H = g("num_attention_heads", 1)
    KV = g("num_key_value_heads", H)
    hd = d // H
    per_layer = (d * d + 2 * d * KV * hd + d * d) + 3 * d * (I or 4 * d)   # GQA attn + gated MLP
    embed = V * d
    return {"n": n, "d": d, "per_layer": per_layer, "embed": embed, "total": embed + n * per_layer}


def _tap_configs(n):
    cfgs = [(f"1tap@{int(f * 100)}%", [max(1, round(n * f))]) for f in (0.25, 0.40, 0.55, 0.70, 0.85)]
    cfgs.append(("3tap", sorted({max(1, round(n * f)) for f in (0.33, 0.50, 0.67)})))
    cfgs.append(("5tap", sorted({max(1, round(n * f)) for f in (0.30, 0.40, 0.50, 0.60, 0.70)})))
    return cfgs


def bench_efficiency(model, Ns, seeds, device):
    """Full comprehensive eval: AUC x memory (universal) x compute (wall-time on this machine).
    Sweeps CG tap configs (1/3/5 taps x depth, both directions) vs a full-model linear probe."""
    from conceptgate.taps import TapForward
    mp = _model_params(model)
    n, d = mp["n"], mp["d"]
    configs = _tap_configs(n)
    fin = n - 1
    union = sorted(set(sum([c[1] for c in configs], [])) | {fin})
    idx = {L: i for i, L in enumerate(union)}
    N = Ns[-1]
    print(f"\n### EFFICIENCY {model}  ({n} layers, d={d}, ~{mp['total'] / 1e6:.0f}M params) N={N}", flush=True)
    cg = load_model(model, union, device)
    pool_txt, pool_y, test_txt, test_y = _jackhhao_probe_split(POOL_BIG_PER_CLASS)
    A_pool = _fullprobe_extract(cg, pool_txt, model, f"effpool{POOL_BIG_PER_CLASS}")
    A_te = _fullprobe_extract(cg, test_txt, model, "efftest")
    pos_i, neg_i = np.where(pool_y == 1)[0], np.where(pool_y == 0)[0]

    def _auc(li, fn):
        vals = []
        for sd in seeds:
            r = np.random.default_rng(3000 + sd)
            ip, ineg = r.choice(pos_i, N, replace=False), r.choice(neg_i, N, replace=False)
            vals.append(fn(A_pool[ip][:, li, :], A_pool[ineg][:, li, :], A_te[:, li, :]))
        return float(np.mean(vals)), float(np.std(vals))

    cg_auc = lambda li, dr: _auc(li, lambda p, q, t: _metrics(_fit_cg(p, q, dr).llr(t), test_y)["auc"])
    pr_auc = lambda li: _auc(li, lambda p, q, t: _lr_auc(p, q, t, test_y))

    # compute (wall-time): per-prompt truncated forward at each distinct top-tap depth (+ final)
    depths = sorted(set(max(c[1]) for c in configs) | {fin})
    sample = test_txt[:32]
    wt = {}
    for L in depths:
        tf = TapForward(cg.model, [L])
        tf.read(cg.tok, sample[:8], cg.device, last_only=True, batch_size=8)   # warm up
        ts = []
        for _ in range(5):
            t = time.perf_counter()
            tf.read(cg.tok, sample, cg.device, last_only=True, batch_size=8)
            ts.append((time.perf_counter() - t) / len(sample) * 1e3)
        wt[L] = (float(np.mean(ts)), float(np.std(ts)))
    cg.unload()

    rows = []
    for name, layers in configs:
        top = max(layers)
        loaded = mp["embed"] + (top + 1) * mp["per_layer"]
        li = [idx[L] for L in layers]
        al, sl = cg_auc(li, Direction.LOGISTIC)
        ad, sd_ = cg_auc(li, Direction.DIFF_OF_MEANS)
        apt, spt = pr_auc(li)   # DEPTH-MATCHED probe: LR on the SAME taps -> same truncated forward as CG
        m_ = len(layers)
        rows.append({"config": name, "layers": layers, "top": top, "depth_frac": round((top + 1) / n, 2),
                     "weights_frac": round(loaded / mp["total"], 2),
                     "learned_params": 4 * m_ * d + m_,   # full stored state: detect+steer dirs (2md) + standardization (2md) + filter (m)
                     "learned_params_detect": 3 * m_ * d + m_,  # detection-only state (no steering vector)
                     "auc_log": round(al, 3), "auc_log_std": round(sl, 3), "auc_diff": round(ad, 3),
                     "auc_probe_tap": round(apt, 3), "auc_probe_tap_std": round(spt, 3),
                     "fwd_ms": round(wt[top][0], 1), "fwd_ms_std": round(wt[top][1], 1)})
    apr, spr = pr_auc([idx[fin]])
    probe = {"config": "probe@final", "depth_frac": 1.0, "weights_frac": 1.0, "learned_params": d + 1,
             "auc": round(apr, 3), "auc_std": round(spr, 3),
             "fwd_ms": round(wt[fin][0], 1), "fwd_ms_std": round(wt[fin][1], 1)}

    print(f"  {'config':>11} {'depth':>6} {'weights':>8} {'CG-log':>13} {'probe@tap':>13} {'AUC-diff':>9} "
          f"{'fwd ms/prompt':>15}", flush=True)
    for r in rows:
        print(f"  {r['config']:>11} {r['depth_frac']:>5.0%} {r['weights_frac']:>7.0%} "
              f"{r['auc_log']:>8.3f}±{r['auc_log_std']:.2f} {r['auc_probe_tap']:>8.3f}±{r['auc_probe_tap_std']:.2f} "
              f"{r['auc_diff']:>9.3f} {r['fwd_ms']:>9.1f}±{r['fwd_ms_std']:.1f}", flush=True)
    print(f"  {'probe@final':>11} {1.0:>5.0%} {1.0:>7.0%} {apr:>8.3f}±{spr:.2f} {'--':>13} {'--':>9} "
          f"{probe['fwd_ms']:>9.1f}±{probe['fwd_ms_std']:.1f}   (full-model reference)", flush=True)
    return {"model": model, "n_layers": n, "d": d, "total_params_M": round(mp["total"] / 1e6),
            "N": N, "seeds": len(seeds), "configs": rows, "linear_probe": probe}


# ------------------ multi-concept scaling (the amortization eval) ------------------
# The title claim is "efficiently learn concepts" (plural). Detection for ONE concept is a
# commodity; the defensible novelty is amortization across K concepts. A guardrail like GLiGuard
# (a GLiNER2 encoder fully fine-tuned on WildGuardTrain to host a fixed 14-harm + 11-jailbreak
# taxonomy) must be RETRAINED to change its taxonomy. ConceptGate is a training-free concept bank:
# one truncated forward reads all K concepts; adding a concept is one closed-form direction (no
# gradient run). So cost is ~flat in K. We measure that against a linear-probe bank (a trained head
# per concept on the frozen full model) and a LoRA bank (a fine-tune per concept) over the 14
# BeaverTails harm categories -- the same *kind* of taxonomy a real guardrail hosts.
BT_SAFE_POOL = 256     # shared benign pool (negatives for every concept)
BT_POS_PER_CAT = 64    # few-shot pool per category; the fit draws n_fit<=this per class
BT_TEST_CAP = 120      # per-class test cap (rare categories use whatever they have)


def _beavertails_concepts(seed=0):
    """Prompt-level BeaverTails splits: cats, {cat: train_pos}, train_safe, {cat: test_pos}, test_safe.
    A prompt is positive for category c if ANY of its unsafe responses tags c; the safe pool is
    prompts with no unsafe response at all -- one clean benign set shared by every concept."""
    from collections import defaultdict

    from datasets import load_dataset
    ds = load_dataset("PKU-Alignment/BeaverTails")
    rng = np.random.default_rng(seed)

    def build(split):
        pos_cats, ever_unsafe, allp = defaultdict(set), set(), set()
        for r in ds[split]:
            p = r["prompt"].strip()[:CHAR_CAP]
            allp.add(p)
            if not r["is_safe"]:
                ever_unsafe.add(p)
                for c, v in r["category"].items():
                    if v:
                        pos_cats[p].add(c)
        safe = [p for p in allp if p not in ever_unsafe]
        cats = sorted({c for s in pos_cats.values() for c in s})
        per = {c: [p for p, cs in pos_cats.items() if c in cs] for c in cats}
        return per, safe

    trc, trsafe = build("30k_train")
    tec, tesafe = build("30k_test")
    for lst in list(trc.values()) + list(tec.values()) + [trsafe, tesafe]:
        rng.shuffle(lst)
    cats = sorted(trc.keys())
    return cats, trc, trsafe, tec, tesafe


def _lora_fit_auc(model_name, tr_txt, tr_y, te_txt, te_y, device, epochs=8, r=8, lr=1e-4, bs=16, dtype=None):
    """One real LoRA fine-tune (frozen base + low-rank adapters + a classification head), the
    fine-tuning end of the spectrum. Returns (test AUC, train wall-ms, trainable params)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2,
                                                              dtype=dtype or torch.float32)
    base.config.pad_token_id = tok.pad_token_id
    tgt = ["q_proj", "v_proj"] if any("q_proj" in n for n, _ in base.named_modules()) else ["c_attn"]
    peftm = get_peft_model(base, LoraConfig(task_type="SEQ_CLS", r=r, lora_alpha=2 * r,
                                            lora_dropout=0.0, target_modules=tgt)).to(device)
    trainable = int(sum(p.numel() for p in peftm.parameters() if p.requires_grad))
    opt = torch.optim.AdamW([p for p in peftm.parameters() if p.requires_grad], lr=lr)
    enc = tok(tr_txt, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    y = torch.tensor(np.asarray(tr_y), dtype=torch.long, device=device)
    peftm.train()
    t = time.perf_counter()
    for _ in range(epochs):
        perm = torch.randperm(len(tr_y), device=device)
        for i in range(0, len(tr_y), bs):
            j = perm[i:i + bs]
            out = peftm(input_ids=enc["input_ids"][j], attention_mask=enc["attention_mask"][j], labels=y[j])
            out.loss.backward()
            opt.step()
            opt.zero_grad()
    if str(device).startswith(("mps", "cuda")):
        getattr(torch, str(device).split(":")[0]).synchronize()
    train_ms = (time.perf_counter() - t) * 1e3
    peftm.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(te_txt), 16):
            e = tok(te_txt[i:i + 16], padding=True, truncation=True, max_length=256,
                    return_tensors="pt").to(device)
            lg = peftm(**e).logits
            scores.append((lg[:, 1] - lg[:, 0]).float().cpu().numpy())
    auc = float(roc_auc_score(te_y, np.concatenate(scores)))
    del peftm, base
    import gc
    gc.collect()
    return auc, train_ms, trainable


def bench_scaling(model, device, n_fit=32, seeds=(0, 1, 2), lora_cats=0, tap_fracs=None):
    """Cost-vs-K over BeaverTails' 14 harm categories for three concept banks:
      CG      -- truncated forward (shared) + a closed-form direction per concept (no gradient)
      probe   -- full forward (shared)      + a trained logistic head per concept
      LoRA    -- an independent fine-tune per concept (no sharing at all)
    Build wall-time, per-prompt inference wall-time, memory (params) and mean per-category AUC.
    lora_cats>0 runs that many real LoRA fits to anchor the fine-tune line (extrapolated linearly)."""
    from conceptgate.taps import TapForward
    cats, trc, trsafe, tec, tesafe = _beavertails_concepts()
    K = len(cats)
    if tap_fracs:
        _, n_layers = taps_for(model)
        taps = sorted({max(1, min(n_layers - 1, round(n_layers * f))) for f in tap_fracs})
    else:
        taps, n_layers = taps_for(model)
    fin = n_layers - 1
    layers = sorted(set(taps + [fin]))
    tap_i = [layers.index(t) for t in taps]
    fin_i = layers.index(fin)
    top_tap = max(taps)
    mp = _model_params(model)
    d, m = mp["d"], len(taps)
    print(f"\n### SCALING {model}  ({n_layers} layers, d={d}, ~{mp['total'] / 1e6:.0f}M) "
          f"K={K} concepts, n_fit={n_fit}/class, taps {'/'.join(map(str, taps))} (top {top_tap})", flush=True)

    # ---- one shared extraction over the pooled prompts (the amortized cost) ----
    trsafe_pool = trsafe[:BT_SAFE_POOL]
    trpos = {c: trc[c][:BT_POS_PER_CAT] for c in cats}
    train_list = list(dict.fromkeys(trsafe_pool + [p for c in cats for p in trpos[c]]))
    tesafe_pool = tesafe[:BT_TEST_CAP]
    tepos = {c: tec[c][:BT_TEST_CAP] for c in cats}
    test_list = list(dict.fromkeys(tesafe_pool + [p for c in cats for p in tepos[c]]))
    cg = load_model(model, layers, device)
    A_tr = _fullprobe_extract(cg, train_list, model, f"btsc_train{len(train_list)}")
    A_te = _fullprobe_extract(cg, test_list, model, f"btsc_test{len(test_list)}")
    itr = {p: i for i, p in enumerate(train_list)}
    ite = {p: i for i, p in enumerate(test_list)}
    safe_tr = np.array([itr[p] for p in trsafe_pool])
    safe_te = np.array([ite[p] for p in tesafe_pool])
    pos_tr = {c: np.array([itr[p] for p in trpos[c]]) for c in cats}
    pos_te = {c: np.array([ite[p] for p in tepos[c]]) for c in cats}

    # ---- per-prompt forward wall-time: truncated (CG) vs full (probe/LoRA) ----
    sample = test_list[:32]
    def _fwd_ms(L):
        tf = TapForward(cg.model, [L])
        tf.read(cg.tok, sample[:8], cg.device, last_only=True, batch_size=8)  # warm
        ts = []
        for _ in range(5):
            t = time.perf_counter()
            tf.read(cg.tok, sample, cg.device, last_only=True, batch_size=8)
            ts.append((time.perf_counter() - t) / len(sample) * 1e3)
        return float(np.mean(ts)), float(np.std(ts))
    fwd_trunc = _fwd_ms(top_tap)
    fwd_full = _fwd_ms(fin)

    # ---- per-concept fit + AUC (CG closed-form vs probe logistic head) ----
    cg_auc, pr_auc, cg_fit_ms, pr_fit_ms, cg_read_us, pr_head_us = {}, {}, [], [], [], []
    for c in cats:
        yte = np.r_[np.ones(len(pos_te[c])), np.zeros(len(safe_te))].astype(int)
        Xte_tap = np.concatenate([A_te[pos_te[c]][:, tap_i, :], A_te[safe_te][:, tap_i, :]], 0)
        Xte_fin = np.concatenate([A_te[pos_te[c]][:, fin_i, :], A_te[safe_te][:, fin_i, :]], 0)
        ca, pa = [], []
        for s in seeds:
            rng = np.random.default_rng(4000 + s)
            npos = min(n_fit, len(pos_tr[c]))
            ip = rng.choice(pos_tr[c], npos, replace=False)
            ineg = rng.choice(safe_tr, n_fit, replace=False)
            # CG: closed-form direction on the taps
            t = time.perf_counter()
            gate = _fit_cg(A_tr[ip][:, tap_i, :], A_tr[ineg][:, tap_i, :], Direction.LOGISTIC)
            cg_fit_ms.append((time.perf_counter() - t) * 1e3)
            t = time.perf_counter()
            s_cg = gate.llr(Xte_tap)
            cg_read_us.append((time.perf_counter() - t) / len(Xte_tap) * 1e6)
            ca.append(float(roc_auc_score(yte, s_cg)))
            # probe: trained logistic head on the final layer
            Xp, Xn = A_tr[ip][:, fin_i, :], A_tr[ineg][:, fin_i, :]
            Xtr = np.concatenate([Xp, Xn], 0)
            ytr = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            t = time.perf_counter()
            clf = LogisticRegression(max_iter=2000, C=1.0).fit((Xtr - mu) / sd, ytr)
            pr_fit_ms.append((time.perf_counter() - t) * 1e3)
            t = time.perf_counter()
            s_pr = clf.decision_function((Xte_fin - mu) / sd)
            pr_head_us.append((time.perf_counter() - t) / len(Xte_fin) * 1e6)
            pa.append(float(roc_auc_score(yte, s_pr)))
        cg_auc[c], pr_auc[c] = float(np.mean(ca)), float(np.mean(pa))
        print(f"  {c:48s} CG {cg_auc[c]:.3f}  probe {pr_auc[c]:.3f}", flush=True)
    cg.unload()

    # ---- LoRA anchor: a few real fine-tunes (the fine-tune end of the spectrum) ----
    import torch
    lora_dtype = torch.bfloat16 if mp["total"] > 1e9 else torch.float32   # bf16 for big models (memory)
    lora = {"auc": {}, "train_ms": [], "trainable": None}
    for c in cats[:lora_cats]:
        rng = np.random.default_rng(4000)
        npos = min(n_fit, len(pos_tr[c]))
        ip = rng.choice(pos_tr[c], npos, replace=False)
        ineg = rng.choice(safe_tr, n_fit, replace=False)
        tr_txt = [train_list[i] for i in ip] + [train_list[i] for i in ineg]
        tr_y = np.r_[np.ones(len(ip)), np.zeros(len(ineg))].astype(int)
        te_txt = [test_list[i] for i in pos_te[c]] + [test_list[i] for i in safe_te]
        te_y = np.r_[np.ones(len(pos_te[c])), np.zeros(len(safe_te))].astype(int)
        try:
            au, tms, trn = _lora_fit_auc(model, tr_txt, tr_y, te_txt, te_y, device, dtype=lora_dtype)
            lora["auc"][c], lora["trainable"] = au, trn
            lora["train_ms"].append(tms)
            print(f"  [LoRA] {c:44s} AUC {au:.3f}  train {tms / 1e3:.1f}s  ({trn / 1e3:.0f}K params)", flush=True)
        except Exception as e:
            print(f"  [LoRA] {c}: {type(e).__name__}: {e}", flush=True)

    n_pool = len(train_list)
    mean_cg, mean_pr = float(np.mean(list(cg_auc.values()))), float(np.mean(list(pr_auc.values())))
    lora_ms = float(np.mean(lora["train_ms"])) if lora["train_ms"] else None
    print(f"  mean AUC over {K} concepts: CG {mean_cg:.3f} | probe {mean_pr:.3f}"
          + (f" | LoRA(subset) {np.mean(list(lora['auc'].values())):.3f}" if lora["auc"] else ""), flush=True)
    print(f"  fwd/prompt: trunc {fwd_trunc[0]:.1f}ms  full {fwd_full[0]:.1f}ms | "
          f"per-concept fit: CG {np.mean(cg_fit_ms):.2f}ms  probe {np.mean(pr_fit_ms):.1f}ms"
          + (f"  LoRA {lora_ms / 1e3:.1f}s" if lora_ms else ""), flush=True)

    return {
        "model": model, "n_layers": n_layers, "d": d, "total_params_M": round(mp["total"] / 1e6),
        "K": K, "categories": cats, "n_fit": n_fit, "n_pool": n_pool, "seeds": len(seeds),
        "taps": taps, "top_tap": top_tap, "depth_frac": round((top_tap + 1) / n_layers, 3),
        "weights_frac_cg": round((mp["embed"] + (top_tap + 1) * mp["per_layer"]) / mp["total"], 3),
        "auc": {"cg": cg_auc, "probe": pr_auc, "lora": lora["auc"], "mean_cg": mean_cg, "mean_probe": mean_pr},
        "build": {
            "fwd_trunc_ms": round(fwd_trunc[0], 2), "fwd_full_ms": round(fwd_full[0], 2),
            "cg_extract_ms": round(n_pool * fwd_trunc[0], 1), "probe_extract_ms": round(n_pool * fwd_full[0], 1),
            "cg_fit_ms": round(float(np.mean(cg_fit_ms)), 3), "probe_fit_ms": round(float(np.mean(pr_fit_ms)), 2),
            "lora_train_ms": round(lora_ms, 1) if lora_ms else None, "lora_n_measured": len(lora["train_ms"]),
        },
        "infer": {  # per prompt, scoring against all K concepts
            "cg_fwd_ms": round(fwd_trunc[0], 2), "cg_read_us": round(float(np.mean(cg_read_us)), 2),
            "probe_fwd_ms": round(fwd_full[0], 2), "probe_head_us": round(float(np.mean(pr_head_us)), 2),
            "lora_fwd_ms": round(fwd_full[0], 2),  # one full forward PER adapter -> x K
        },
        "memory": {  # learned params per concept (universal, device-independent)
            "cg_per_concept": m * d, "probe_per_concept": d + 1, "lora_per_concept": lora["trainable"],
            "backbone_full": mp["total"], "backbone_cg": mp["embed"] + (top_tap + 1) * mp["per_layer"],
        },
    }


def bench_ood(model, device, n_fit=32, seeds=(0, 1, 2), tap_fracs=None):
    """Within-concept (harmfulness) generalization: leave-one-category-out over BeaverTails' 14 harm
    categories. For each category c, learn the harmful direction from the OTHER 13 categories (+ a shared
    benign pool) and score the held-out category c; compare to the in-distribution reference that trains on
    c itself. Same N, same test set -- the only variable is whether the test category was seen. This is the
    clean generalization test the confounded cross-distribution transfer (jailbreak framing vs harmful
    content, section 2) could not give. Detectors: ConceptGate (logistic taps) vs the full-model probe."""
    cats, trc, trsafe, tec, tesafe = _beavertails_concepts()
    if tap_fracs:
        _, n_layers = taps_for(model)
        taps = sorted({max(1, min(n_layers - 1, round(n_layers * f))) for f in tap_fracs})
    else:
        taps, n_layers = taps_for(model)
    fin = n_layers - 1
    layers = sorted(set(taps + [fin]))
    tap_i = [layers.index(t) for t in taps]
    fin_i = layers.index(fin)
    print(f"\n### OOD {model}  ({n_layers} layers, taps {'/'.join(map(str, taps))}) "
          f"leave-one-category-out over {len(cats)} categories, N={n_fit}/class", flush=True)

    # same pooled prompts + cache key as the scaling eval (activations are a cache hit if that ran)
    trsafe_pool = trsafe[:BT_SAFE_POOL]
    trpos = {c: trc[c][:BT_POS_PER_CAT] for c in cats}
    train_list = list(dict.fromkeys(trsafe_pool + [p for c in cats for p in trpos[c]]))
    tesafe_pool = tesafe[:BT_TEST_CAP]
    tepos = {c: tec[c][:BT_TEST_CAP] for c in cats}
    test_list = list(dict.fromkeys(tesafe_pool + [p for c in cats for p in tepos[c]]))
    cg = load_model(model, layers, device)
    A_tr = _fullprobe_extract(cg, train_list, model, f"btsc_train{len(train_list)}")
    A_te = _fullprobe_extract(cg, test_list, model, f"btsc_test{len(test_list)}")
    cg.unload()
    itr = {p: i for i, p in enumerate(train_list)}
    ite = {p: i for i, p in enumerate(test_list)}
    safe_tr = np.array([itr[p] for p in trsafe_pool])
    safe_te = np.array([ite[p] for p in tesafe_pool])
    pos_tr = {c: np.array([itr[p] for p in trpos[c]]) for c in cats}
    pos_te = {c: np.array([ite[p] for p in tepos[c]]) for c in cats}

    rows = {}
    for c in cats:
        others = np.concatenate([pos_tr[x] for x in cats if x != c])
        yte = np.r_[np.ones(len(pos_te[c])), np.zeros(len(safe_te))].astype(int)
        Xte_tap = np.concatenate([A_te[pos_te[c]][:, tap_i, :], A_te[safe_te][:, tap_i, :]], 0)
        Xte_fin = np.concatenate([A_te[pos_te[c]][:, fin_i, :], A_te[safe_te][:, fin_i, :]], 0)
        cin, cood, pin, pood = [], [], [], []
        for s in seeds:
            rng = np.random.default_rng(5000 + s)
            ineg = rng.choice(safe_tr, n_fit, replace=False)
            ip_in = rng.choice(pos_tr[c], min(n_fit, len(pos_tr[c])), replace=False)   # train on c
            ip_ood = rng.choice(others, n_fit, replace=False)                           # train on the rest
            cin.append(roc_auc_score(yte, _fit_cg(A_tr[ip_in][:, tap_i, :], A_tr[ineg][:, tap_i, :],
                                                  Direction.LOGISTIC).llr(Xte_tap)))
            cood.append(roc_auc_score(yte, _fit_cg(A_tr[ip_ood][:, tap_i, :], A_tr[ineg][:, tap_i, :],
                                                   Direction.LOGISTIC).llr(Xte_tap)))
            pin.append(_lr_auc(A_tr[ip_in][:, fin_i, :], A_tr[ineg][:, fin_i, :], Xte_fin, yte))
            pood.append(_lr_auc(A_tr[ip_ood][:, fin_i, :], A_tr[ineg][:, fin_i, :], Xte_fin, yte))
        rows[c] = {"cg_in": float(np.mean(cin)), "cg_ood": float(np.mean(cood)),
                   "pr_in": float(np.mean(pin)), "pr_ood": float(np.mean(pood))}
        print(f"  {c:48s} CG {rows[c]['cg_in']:.3f}->{rows[c]['cg_ood']:.3f}  "
              f"probe {rows[c]['pr_in']:.3f}->{rows[c]['pr_ood']:.3f}", flush=True)

    _m = lambda k: float(np.mean([r[k] for r in rows.values()]))
    summ = {k: _m(k) for k in ("cg_in", "cg_ood", "pr_in", "pr_ood")}
    summ["cg_drop"] = round(summ["cg_in"] - summ["cg_ood"], 3)
    summ["pr_drop"] = round(summ["pr_in"] - summ["pr_ood"], 3)
    print(f"  MEAN  CG {summ['cg_in']:.3f}->{summ['cg_ood']:.3f} (drop {summ['cg_drop']:.3f}) | "
          f"probe {summ['pr_in']:.3f}->{summ['pr_ood']:.3f} (drop {summ['pr_drop']:.3f})", flush=True)
    return {"model": model, "n_layers": n_layers, "taps": taps, "n_fit": n_fit, "seeds": len(seeds),
            "categories": cats, "rows": rows, "summary": summ}


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
    ap.add_argument("--efficiency", action="store_true", help="AUC x memory x wall-time, CG tap-config sweep")
    ap.add_argument("--scaling", action="store_true", help="multi-concept cost-vs-K over BeaverTails categories")
    ap.add_argument("--lora-cats", type=int, default=0, help="# real LoRA fits to anchor the fine-tune line")
    ap.add_argument("--tap-fracs", default="", help="scaling CG tap depths, e.g. 0.4,0.6,0.8 (blank=default 0.33/0.5/0.67)")
    ap.add_argument("--ood", action="store_true", help="within-concept OOD: leave-one-category-out over BeaverTails")
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

    if a.ood:
        out = "scripts/eval_ood_results.json" if a.out == "scripts/eval_detection_results.json" else a.out
        Nf = [int(x) for x in a.ns.split(",")][-1]
        tfr = [float(x) for x in a.tap_fracs.split(",")] if a.tap_fracs else None
        print(f"ood: {len(models)} model(s), leave-one-category-out, N={Nf}/class, seeds={seeds}", flush=True)
        results = []
        for model in models:
            try:
                results.append(bench_ood(model, a.device, n_fit=Nf, seeds=seeds, tap_fracs=tfr))
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  SKIP {model}: {type(e).__name__}: {e}", flush=True)
            with open(out, "w") as f:
                json.dump({"results": results}, f, indent=1)
        print(f"\ndone -> {out}", flush=True)
        return

    if a.scaling:
        out = "scripts/eval_scaling_results.json" if a.out == "scripts/eval_detection_results.json" else a.out
        chip = _chip()
        Nf = [int(x) for x in a.ns.split(",")][-1]
        print(f"scaling: {len(models)} model(s), n_fit={Nf}/class, seeds={seeds}, lora_cats={a.lora_cats} | "
              f"wall-time on: {chip} (device {a.device})", flush=True)
        tfr = [float(x) for x in a.tap_fracs.split(",")] if a.tap_fracs else None
        results = []
        for model in models:
            try:
                results.append(bench_scaling(model, a.device, n_fit=Nf, seeds=seeds,
                                             lora_cats=a.lora_cats, tap_fracs=tfr))
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  SKIP {model}: {type(e).__name__}: {e}", flush=True)
            with open(out, "w") as f:
                json.dump({"machine": chip, "device": a.device, "results": results}, f, indent=1)
        print(f"\ndone -> {out}", flush=True)
        return

    if a.efficiency:
        out = "scripts/eval_efficiency_results.json" if a.out == "scripts/eval_detection_results.json" else a.out
        chip = _chip()
        print(f"efficiency: {len(models)} model(s), N={[int(x) for x in a.ns.split(',')][-1]}, "
              f"seeds={seeds} | wall-time on: {chip} (device {a.device})", flush=True)
        results = []
        for model in models:
            try:
                results.append(bench_efficiency(model, Ns, seeds, a.device))
            except Exception as e:
                print(f"  SKIP {model}: {type(e).__name__}: {e}", flush=True)
            with open(out, "w") as f:
                json.dump({"machine": chip, "device": a.device, "results": results}, f, indent=1)
        print(f"\ndone -> {out}", flush=True)
        return

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
