import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import analyze_behaviour_dose as A  # noqa: E402


def _rows(n=40, k=8, seed=0, flat=False):
    """Synthetic rows: each attack has a true dose t in [0,1]; plus-arm refusal ~ t, minus ~ 0.
    `flat=True` sets every arm's refusal probability to 0.5, so the dose is pure sampling noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        kind = "benign" if i % 8 == 7 else "template"
        t = rng.uniform(0, 1)
        row = {"kind": kind, "llr": float(t * 10), "fired": True, "logit": {}, "lex": {}, "clf": {}}
        for a in A.ARMS:
            p = {"none": 0.5 * t, "minus": 0.0, "plus": t, "rand_minus": 0.4, "rand_plus": 0.5}[a]
            if flat:
                p = 0.5
            lex = (rng.uniform(size=k) < p).astype(int).tolist()
            row["lex"][a] = lex
            row["clf"][a] = [0.9 if v else 0.1 for v in lex]          # agrees with lex exactly
            row["logit"][a] = {"none": 0.0, "minus": -2 * t, "plus": 2 * t, "rand_minus": -0.1, "rand_plus": 0.1}[a]
        rows.append(row)
    return rows


def test_rates_and_dose_use_only_requested_samples():
    rows = _rows(k=4)
    P = A.rates(rows, "lex", [0, 2])
    assert set(P) == set(A.ARMS) and P["plus"].shape == (40,)
    assert P["plus"][0] == np.mean([rows[0]["lex"]["plus"][0], rows[0]["lex"]["plus"][2]])
    D, Dr = A.dose(P)
    assert np.allclose(D, 0.5 * (P["plus"] - P["minus"]))
    assert np.allclose(Dr, 0.5 * (P["rand_plus"] - P["rand_minus"]))


def test_clf_rates_threshold_at_half():
    rows = _rows(k=4)
    assert np.allclose(A.rates(rows, "clf", [0, 1, 2, 3])["plus"], A.rates(rows, "lex", [0, 1, 2, 3])["plus"])


def test_spearman_brown_and_proxy_sign():
    assert abs(A.spearman_brown(0.5) - 2 / 3) < 1e-9
    rows = _rows()
    lever = A.proxy_lever(rows)
    assert lever[0] > 0
    assert np.isclose(lever[0], (rows[0]["logit"]["plus"] - rows[0]["logit"]["minus"]) / 2)


def test_stage1_gate_passes_on_clean_synthetic_and_fails_on_noise():
    meta = {"k": 16, "rows": _rows(n=120, k=16)}
    out = A.stage1(meta)
    s = out["instruments"]["lex"]
    assert s["split_half"] > 0.7 and out["inter_instrument"] > 0.99 and out["gate"]["go"]
    assert s["proxy_check"] > 0.8 and s["dose_ratio_attacks"] > 2
    noisy = A.stage1({"k": 16, "rows": _rows(n=120, k=16, flat=True)})
    assert abs(noisy["instruments"]["lex"]["split_half"]) < 0.3
    assert not noisy["gate"]["go"]


def _planted(n=160, m=3, d=16, seed=0):
    """Activations with a planted direction whose projection IS the dose; benign rows appended."""
    rng = np.random.default_rng(seed)
    acts = rng.normal(size=(n, m, d))
    w = rng.normal(size=(m, d))
    w /= np.linalg.norm(w)
    t = (acts * w).sum((1, 2))
    t = (t - t.min()) / (t.max() - t.min())
    rows = []
    for i in range(n):
        kind = "benign" if i >= n - 24 else ("template" if i < 120 else "short")
        row = {"kind": kind, "llr": float(rng.normal()), "fired": True, "logit": {}, "lex": {}, "clf": {}}
        for a in A.ARMS:
            p = {"none": 0.3, "minus": 0.0, "plus": t[i], "rand_minus": 0.3, "rand_plus": 0.3}[a]
            lex = (rng.uniform(size=16) < p).astype(int).tolist()
            row["lex"][a] = lex
            row["clf"][a] = [0.9 if v else 0.1 for v in lex]
            row["logit"][a] = 0.0
        rows.append(row)
    wraw = rng.normal(size=(m, d))
    return {"k": 16, "rows": rows}, acts, wraw


def test_groups_cycle_requests_and_isolate_other_kinds():
    ka = np.array(["template"] * 24 + ["short"] * 3 + ["request"] * 2)
    g = A.groups_for(ka)
    assert list(g[:12]) == list(range(12)) and list(g[12:24]) == list(range(12))
    assert set(g[24:27]) == {12} and set(g[27:]) == {13}


def test_stage2_recovers_a_planted_direction_and_beats_null():
    meta, acts, wraw = _planted()
    out = A.stage2(meta, acts, wraw, "lex", seed=0, nperm=30)
    assert out["cv_grouped"] > 0.6
    assert out["null"]["z"] > 4
    assert out["cv_grouped"] > out["llr_spearman"]
    assert out["gate"]["go"]
    assert len(out["pred_oof_grouped"]) == 136 and len(out["pred_benign"]) == 24


def test_stage3_selection_prefers_predicted_movers():
    meta, acts, wraw = _planted()
    o2 = A.stage2(meta, acts, wraw, "lex", seed=0, nperm=10)
    o3 = A.stage3(meta, o2, "lex", n_random=50)
    arms = o3["arms"]
    assert set(arms) == {"blanket", "concept_gate", "outcome_gate", "anti_outcome", "both"}
    assert arms["outcome_gate"]["mean_dP_written"] > arms["anti_outcome"]["mean_dP_written"]
    assert arms["outcome_gate"]["mean_dP_written"] > o3["random"]["mean"]
    assert o3["random"]["p_ge_outcome"] < 0.1
    assert 0 <= arms["outcome_gate"]["benign_written_frac"] <= 1


def test_clf_soft_uses_probabilities_and_merge_doubles_k():
    rows = _rows(n=16, k=4)
    soft = A.rates(rows, "clf_soft", [0, 1, 2, 3])["plus"]
    hard = A.rates(rows, "clf", [0, 1, 2, 3])["plus"]
    assert np.allclose(soft, 0.1 + 0.8 * hard)      # each sample scores 0.9 (refusal) or 0.1 (not)
    meta_a = {"k": 4, "seed": 0, "model": "m", "alpha": 0.08, "temperature": 0.7, "max_new_tokens": 8,
              "rows": [dict(r, prompt=f"p{i}", samples={a: ["x"] * 4 for a in A.ARMS}) for i, r in enumerate(rows)]}
    meta_b = {"k": 4, "seed": 0, "sample_seed": 1, "model": "m", "alpha": 0.08, "temperature": 0.7, "max_new_tokens": 8,
              "rows": [dict(r, prompt=f"p{i}", samples={a: ["y"] * 4 for a in A.ARMS}) for i, r in enumerate(_rows(n=16, k=4, seed=1))]}
    merged = A.merge_runs(meta_a, meta_b)
    assert merged["k"] == 8 and merged["merged_sample_seeds"] == [0, 1]
    assert len(merged["rows"][0]["lex"]["plus"]) == 8 and merged["rows"][0]["samples"]["plus"] == ["x"] * 4 + ["y"] * 4
    import pytest
    with pytest.raises(ValueError, match="same prompts"):
        A.merge_runs(meta_a, dict(meta_b, rows=meta_b["rows"][::-1]))
