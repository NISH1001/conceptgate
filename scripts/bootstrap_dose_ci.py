"""Bootstrap confidence intervals for the behavioural-dose predictors (Qwen2.5-0.5B, classifier scorer).

The permutation z says the grouped out-of-fold Spearman is far from zero. It does not say how precise it
is. This keeps the out-of-fold predictions of stage 2 fixed and resamples the evaluation set two ways:
prompts (i.i.d.), and harmful-request groups (cluster bootstrap: the 12 request groups plus the short and
bare sets), which respects the dependence that grouped folds exist for. Needs the activations; rebuild
them with rebuild_dose_acts.py if they are missing.

Run:  uv run python scripts/bootstrap_dose_ci.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_behaviour_dose as A  # noqa: E402

MODEL, INSTRUMENT, N_BOOT, SEED = "Qwen/Qwen2.5-0.5B-Instruct", "clf", 10_000, 0
OUT = os.path.join(A.HERE, "bootstrap_dose_ci.json")


def sp(a, b) -> float:
    return float(spearmanr(a, b).correlation)


def ci(stat, groups, rng):
    """Percentile 95% intervals of stat(idx) under prompt and cluster resampling."""
    n = len(groups)
    prompt = np.array([stat(rng.integers(0, n, n)) for _ in range(N_BOOT)])
    members = [np.where(groups == g)[0] for g in np.unique(groups)]
    cluster = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(members), len(members))
        cluster.append(stat(np.concatenate([members[i] for i in pick])))
    q = lambda v: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]  # noqa: E731
    return {"prompt": q(prompt), "cluster": q(np.array(cluster))}


def main():
    meta, acts, wraw = A.load(MODEL)
    rows, k = meta["rows"], int(meta["k"])
    kind = np.array([r["kind"] for r in rows])
    atk = kind != "benign"
    y = A.dose(A.rates(rows, INSTRUMENT, list(range(k))))[0][atk]
    llr = np.array([r["llr"] for r in rows])[atk]
    Xs = A._zscore(acts[atk].reshape(int(atk.sum()), -1))[0]
    g = A.groups_for(kind[atk])
    ng = min(7, len(set(g)))
    rho, pred = A.cv(Xs, y, 0, groups=g, n_splits=ng)
    m = acts.shape[1]
    proj = np.stack([acts[atk][:, i, :] @ wraw[i] for i in range(m)], 1)
    rho_p, pred_p = A.cv(A._zscore(proj)[0], y, 0, groups=g, n_splits=ng)
    rng = np.random.default_rng(SEED)
    out = {"model": MODEL, "instrument": INSTRUMENT, "n": int(atk.sum()), "n_groups": int(len(set(g))),
           "n_boot": N_BOOT}
    for name, point, stat in [
        ("ridge_grouped", rho, lambda i: sp(pred[i], y[i])),
        ("concept_projections", rho_p, lambda i: sp(pred_p[i], y[i])),
        ("gate_llr", sp(llr, y), lambda i: sp(llr[i], y[i])),
        ("ridge_minus_projections", rho - rho_p, lambda i: sp(pred[i], y[i]) - sp(pred_p[i], y[i])),
    ]:
        out[name] = {"point": float(point), **ci(stat, g, rng)}
        c = out[name]
        print(f"{name:24s} {point:+.3f}  prompt 95% [{c['prompt'][0]:+.3f}, {c['prompt'][1]:+.3f}]"
              f"  cluster 95% [{c['cluster'][0]:+.3f}, {c['cluster'][1]:+.3f}]", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
