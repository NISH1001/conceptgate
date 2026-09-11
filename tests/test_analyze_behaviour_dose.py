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
