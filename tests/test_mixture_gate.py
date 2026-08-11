import numpy as np

from conceptgate import concept_bank as cb
from conceptgate.concept import BandpassConcept, Concept, ConceptBank, error_at_zero


def _synth(rng, n, sign, U, dprime):
    m, d = U.shape
    A = rng.standard_normal((n, m, d))
    for l in range(m):
        A[:, l, :] += sign * (dprime[l] / 2.0) * U[l]
    return A


def _toy_data(seed=0, n=2000, m=3, d=32):
    rng = np.random.default_rng(seed)
    dprime = np.array([1.6, 2.0, 0.6])
    U = cb._normalize(rng.standard_normal((m, d)), axis=-1)
    return _synth(rng, n, +1, U, dprime), _synth(rng, n, -1, U, dprime)


def test_fit_and_llr_separate_classes():
    A_pos, A_neg = _toy_data()
    g = Concept(name="toy").fit(A_pos[:1000], A_neg[:1000])
    lp = g.llr(A_pos[1000:])
    ln = g.llr(A_neg[1000:])
    assert lp.mean() > ln.mean() + 2.0


def test_unimodal_data_selects_one_component_and_matches_fisher():
    # continuity (spec 3.1): J=1 collapse, error within 1.5 points of the fisher BandpassConcept
    A_pos, A_neg = _toy_data()
    tr, te = slice(0, 1000), slice(1000, 2000)
    mg = Concept(name="toy").fit(A_pos[tr], A_neg[tr])
    assert mg.gmm_pos.n_components == 1
    assert mg.gmm_neg.n_components == 1
    err_mix = error_at_zero(mg.llr(A_pos[te]), mg.llr(A_neg[te]))
    fg = BandpassConcept(name="toy", filter_method="fisher").fit(A_pos[tr], A_neg[tr])
    sp = fg.score(A_pos[te]) - 0.5 * (fg.mu_pos + fg.mu_neg)
    sn = fg.score(A_neg[te]) - 0.5 * (fg.mu_pos + fg.mu_neg)
    err_fisher = error_at_zero(sp, sn)
    assert abs(err_mix - err_fisher) < 0.015


def test_duck_type_contract_for_bank_and_guard():
    A_pos, A_neg = _toy_data(n=400)
    g = Concept(name="toy").fit(A_pos, A_neg)
    assert g.W_raw is not None and g.W_raw.shape == g.W.shape
    bank = ConceptBank().add(g)
    fired = bank.fire(A_pos[:8])
    assert fired.shape == (8,) and fired.dtype == bool
    assert bank.which(A_pos[:8]).shape == (8,)


def test_calibrate_threshold_hits_target_fpr():
    A_pos, A_neg = _toy_data(n=3000)
    g = Concept(name="toy").fit(A_pos[:1000], A_neg[:1000])
    g.calibrate_threshold(A_neg[1000:2000], target_fpr=0.05)
    fpr = float(np.mean(g.fire(A_neg[2000:])))
    assert fpr < 0.10


def test_calibrate_z_gives_low_benign_fire_rate():
    A_pos, A_neg = _toy_data(n=2000)
    g = Concept(name="toy").fit(A_pos[:1000], A_neg[:1000])
    tau = g.calibrate_z(z=3.0)
    assert np.isfinite(tau)
    assert float(np.mean(g.fire(A_neg[1000:]))) < 0.02
