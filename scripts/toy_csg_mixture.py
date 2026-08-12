"""Offline validation of the mixture densities on synthetic activations (no model).

Three scenarios, each over >=5 seeds (spec 2026-08-10, section 5). Classes are
mixtures of isotropic Gaussians in activation space; the mode is sampled PER SAMPLE
and shared across layers, so the joint spectrogram has one component per mode.

  S1 regression    : unimodal both classes (the toy_csg.py setup, d'=[1.6,2.0,0.6]).
                     PASS: BIC picks J=1 on both classes; |err_mix - err_fisher| <= 1pt.
  S2 bimodal benign: benign = far mode + near-boundary mode. PASS: err_mix <= err_fisher
                     and err_mix <= bayes + 2pt.
  S3 kill shot     : benign modes FLANK harmful on the discriminative axis -> no single
                     linear filter separates. PASS: err_fisher >= 35%; err_mix <= bayes + 3pt.

Run:  uv run python scripts/toy_csg_mixture.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conceptgate import concept_bank as cb
from conceptgate.concept import BandpassConcept, Concept, error_at_zero

SEEDS = [0, 1, 2, 3, 4]
N_TRAIN = 8000   # per class (S3's near-cancelling class means need this for clean directions)
N_TEST = 3000


def synth_modes(rng, n, U, gains, modes, weights):
    """[n, m, d] activations. Mode sampled per SAMPLE (shared across layers);
    layer-l mean = modes[j] * gains[l] * U[l]; isotropic unit noise."""
    m, d = U.shape
    idx = rng.choice(len(modes), size=n, p=weights)
    A = rng.standard_normal((n, m, d))
    off = np.asarray(modes, dtype=float)[idx]                    # [n]
    for layer in range(m):
        A[:, layer, :] += (off * gains[layer])[:, None] * U[layer][None, :]
    return A


def true_logpdf(A, U, gains, modes, weights):
    """Exact class log-density of the generator (for the Bayes floor). A: [n, m, d].
    Constant terms are dropped (they cancel in the LLR)."""
    n = A.shape[0]
    parts = []
    for j, mode in enumerate(modes):
        maha = np.zeros(n)
        for layer in range(A.shape[1]):
            diff = A[:, layer, :] - mode * gains[layer] * U[layer][None, :]
            maha += np.sum(diff * diff, axis=1)
        parts.append(np.log(weights[j]) - 0.5 * maha)
    P = np.stack(parts, axis=1)
    top = P.max(axis=1, keepdims=True)
    return np.log(np.exp(P - top).sum(axis=1)) + top[:, 0]


def bayes_error(A_pos, A_neg, U, gains, spec_pos, spec_neg):
    def llr(A):
        return true_logpdf(A, U, gains, *spec_pos) - true_logpdf(A, U, gains, *spec_neg)

    return error_at_zero(llr(A_pos), llr(A_neg))


def auc(sp, sn):
    """Rank-based ROC AUC."""
    x = np.concatenate([sp, sn])
    r = x.argsort().argsort() + 1.0
    return float((r[: len(sp)].sum() - len(sp) * (len(sp) + 1) / 2) / (len(sp) * len(sn)))


SCENARIOS = {
    "S1 regression": dict(
        gains=[0.8, 1.0, 0.3],                     # d'/2 of toy_csg.py
        pos=dict(modes=[+1.0], weights=[1.0]),
        neg=dict(modes=[-1.0], weights=[1.0]),
    ),
    "S2 bimodal benign": dict(
        gains=[1.0, 1.2, 0.6],
        pos=dict(modes=[+1.0], weights=[1.0]),
        neg=dict(modes=[-1.0, +0.45], weights=[0.7, 0.3]),
    ),
    "S3 kill shot": dict(
        gains=[1.0, 1.0, 1.0],
        pos=dict(modes=[0.0], weights=[1.0]),
        # near-symmetric flanking: the class-mean difference is small, so the
        # diff-of-means direction needs the larger N_TRAIN to be estimated cleanly
        neg=dict(modes=[-2.0, +2.0], weights=[0.6, 0.4]),
    ),
}


def run_scenario(name, cfg):
    m, d = 3, 64
    errs = {"best": [], "fisher": [], "mixture": [], "bayes": []}
    aucs = {"fisher": [], "mixture": []}
    Jsel = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        U = cb._normalize(rng.standard_normal((m, d)), axis=-1)
        gains = np.asarray(cfg["gains"])
        spec_p = (cfg["pos"]["modes"], cfg["pos"]["weights"])
        spec_n = (cfg["neg"]["modes"], cfg["neg"]["weights"])
        Ap = synth_modes(rng, N_TRAIN + N_TEST, U, gains, *spec_p)
        An = synth_modes(rng, N_TRAIN + N_TEST, U, gains, *spec_n)
        tr, te = slice(0, N_TRAIN), slice(N_TRAIN, N_TRAIN + N_TEST)

        for method in ["best", "fisher"]:
            g = BandpassConcept(name=name, filter_method=method).fit(Ap[tr], An[tr])
            sp = g.score(Ap[te]) - 0.5 * (g.mu_pos + g.mu_neg)
            sn = g.score(An[te]) - 0.5 * (g.mu_pos + g.mu_neg)
            errs[method].append(error_at_zero(sp, sn))
            if method == "fisher":
                aucs["fisher"].append(auc(sp, sn))

        mg = Concept(name=name).fit(Ap[tr], An[tr])
        lp, ln = mg.llr(Ap[te]), mg.llr(An[te])
        errs["mixture"].append(error_at_zero(lp, ln))
        aucs["mixture"].append(auc(lp, ln))
        Jsel.append((mg.gmm_pos.n_components, mg.gmm_neg.n_components))

        errs["bayes"].append(bayes_error(Ap[te], An[te], U, gains, spec_p, spec_n))

    mean = {k: float(np.mean(v)) for k, v in errs.items()}
    std = {k: float(np.std(v)) for k, v in errs.items()}
    print(f"\n== {name} ==   selected J (pos,neg) per seed: {Jsel}")
    for k in ["best", "fisher", "mixture", "bayes"]:
        extra = f"   auc={np.mean(aucs[k]):.3f}" if k in aucs else ""
        print(f"  {k:>8}: err={mean[k]*100:5.1f}% ± {std[k]*100:3.1f}{extra}")
    return mean, Jsel


def main() -> int:
    checks = []
    m1, j1 = run_scenario("S1 regression", SCENARIOS["S1 regression"])
    checks.append(("S1: BIC picks J=1 both classes", all(t == (1, 1) for t in j1)))
    checks.append(("S1: |mix - fisher| <= 1pt", abs(m1["mixture"] - m1["fisher"]) <= 0.010))

    m2, _ = run_scenario("S2 bimodal benign", SCENARIOS["S2 bimodal benign"])
    checks.append(("S2: mix <= fisher", m2["mixture"] <= m2["fisher"] + 0.002))
    checks.append(("S2: mix <= bayes + 2pt", m2["mixture"] <= m2["bayes"] + 0.020))

    m3, _ = run_scenario("S3 kill shot", SCENARIOS["S3 kill shot"])
    checks.append(("S3: fisher >= 35% (linear fails)", m3["fisher"] >= 0.35))
    checks.append(("S3: mix <= bayes + 3pt", m3["mixture"] <= m3["bayes"] + 0.030))

    print("\n" + "-" * 56)
    ok = True
    for label, passed in checks:
        ok &= passed
        print(f"  [{'ok' if passed else 'XX'}] {label}")
    print(f"MIXTURE VALIDATION {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
