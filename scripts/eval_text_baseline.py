"""Surface-text baseline and train/test duplicate check on jackhhao/jailbreak-classification.

Does the jailbreak benchmark need activations at all? A TF-IDF + logistic-regression detector reads only
the prompt text. It uses the same few-shot protocol as eval_detection.py (same pool, same seeded draws of
N per class, same 262-prompt test split), so its AUC sits next to the ConceptGate and probe rows. It also
fits on the full training split, as an upper reference. The duplicate check counts test prompts whose
nearest training prompt (character 5-gram TF-IDF cosine) is above a threshold.

Run:  uv run --with datasets python scripts/eval_text_baseline.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_detection import _source_rows, load_source  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval_text_baseline_results.json")
NS = [4, 8, 16, 32]
SEEDS = [0, 1, 2]
DUP_THRESHOLDS = [0.8, 0.9, 0.95]

VECTORIZERS = {
    "word_1-2gram": lambda: TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
    "char_3-5gram": lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True),
}


def auc_tfidf(make_vec, train_txt, train_y, test_txt, test_y) -> float:
    vec = make_vec()
    Xtr = vec.fit_transform(train_txt)
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, train_y)
    return float(roc_auc_score(test_y, clf.decision_function(vec.transform(test_txt))))


def main():
    pool_txt, pool_y, test_txt, test_y = load_source("jackhhao", seed=0)
    pos_idx, neg_idx = np.where(pool_y == 1)[0], np.where(pool_y == 0)[0]
    out = {"n_test": len(test_txt), "n_test_pos": int(test_y.sum()), "few_shot": {}, "full_train": {}}

    for name, make_vec in VECTORIZERS.items():
        out["few_shot"][name] = {}
        for N in NS:
            aucs = []
            for s in SEEDS:   # the draw eval_detection.py uses for every detector
                rng = np.random.default_rng(1000 + s)
                ip = rng.choice(pos_idx, N, replace=False)
                ineg = rng.choice(neg_idx, N, replace=False)
                idx = np.r_[ip, ineg]
                aucs.append(auc_tfidf(make_vec, [pool_txt[i] for i in idx], pool_y[idx], test_txt, test_y))
            out["few_shot"][name][N] = {"mean": float(np.mean(aucs)), "sd": float(np.std(aucs)), "per_seed": aucs}
            print(f"{name:14s} N={N:2d}  AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}", flush=True)

    train, test = _source_rows("jackhhao")
    tr_txt, tr_y = [p for p, _ in train], np.array([y for _, y in train])
    for name, make_vec in VECTORIZERS.items():
        a = auc_tfidf(make_vec, tr_txt, tr_y, test_txt, test_y)
        out["full_train"][name] = a
        print(f"{name:14s} full train ({len(tr_txt)})  AUC {a:.3f}", flush=True)

    # near-duplicates: nearest training prompt per test prompt
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(5, 5)).fit(tr_txt + test_txt)
    sim = (vec.transform(test_txt) @ vec.transform(tr_txt).T).toarray()
    nn = sim.max(axis=1)
    exact = sum(t in set(tr_txt) for t in test_txt)
    out["duplicates"] = {"exact": int(exact), "nn_cos_median": float(np.median(nn))}
    for th in DUP_THRESHOLDS:
        m = nn >= th
        out["duplicates"][f">={th}"] = {"all": int(m.sum()), "jailbreak": int((m & (test_y == 1)).sum()),
                                       "benign": int((m & (test_y == 0)).sum())}
    print("duplicates:", json.dumps(out["duplicates"]), flush=True)
    # does the few-shot result survive dropping near-duplicate test prompts?
    keep = nn < 0.9
    out["few_shot_dedup_0.9"] = {}
    kt, ky = [t for t, k in zip(test_txt, keep) if k], test_y[keep]
    for name, make_vec in VECTORIZERS.items():
        aucs = []
        for s in SEEDS:
            rng = np.random.default_rng(1000 + s)
            idx = np.r_[rng.choice(pos_idx, 32, replace=False), rng.choice(neg_idx, 32, replace=False)]
            aucs.append(auc_tfidf(make_vec, [pool_txt[i] for i in idx], pool_y[idx], kt, ky))
        out["few_shot_dedup_0.9"][name] = {"n_test": int(keep.sum()), "mean": float(np.mean(aucs)), "sd": float(np.std(aucs))}
        print(f"{name:14s} N=32 on {keep.sum()} non-duplicate test prompts  AUC {np.mean(aucs):.3f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
