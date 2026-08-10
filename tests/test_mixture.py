import numpy as np

from conceptgate.mixture import GMM, fit_gmm


def _single_gauss_logpdf(x, mu, cov):
    m = len(mu)
    diff = x - mu
    inv = np.linalg.inv(cov)
    logdet = np.linalg.slogdet(cov)[1]
    return -0.5 * (diff @ inv @ diff + logdet + m * np.log(2 * np.pi))


def test_logpdf_matches_hand_computed_single_gaussian():
    mu = np.array([1.0, -2.0])
    cov = np.array([[2.0, 0.3], [0.3, 0.5]])
    g = GMM(weights=np.array([1.0]), means=mu[None], covs=cov[None])
    X = np.array([[0.0, 0.0], [1.0, -2.0], [3.0, 1.0]])
    want = np.array([_single_gauss_logpdf(x, mu, cov) for x in X])
    np.testing.assert_allclose(g.logpdf(X), want, rtol=1e-10)


def test_logpdf_two_components_is_logsumexp_of_parts():
    means = np.array([[0.0, 0.0], [4.0, 4.0]])
    covs = np.stack([np.eye(2), 2.0 * np.eye(2)])
    w = np.array([0.3, 0.7])
    g = GMM(weights=w, means=means, covs=covs)
    X = np.array([[1.0, 1.0], [4.0, 3.0]])
    parts = np.stack(
        [np.array([_single_gauss_logpdf(x, means[j], covs[j]) for x in X]) for j in range(2)],
        axis=1,
    )
    want = np.log(np.sum(np.exp(parts) * w[None, :], axis=1))
    np.testing.assert_allclose(g.logpdf(X), want, rtol=1e-10)


def test_sample_moments_match_params():
    means = np.array([[0.0, 0.0], [10.0, 10.0]])
    covs = np.stack([np.eye(2), np.eye(2)])
    g = GMM(weights=np.array([0.5, 0.5]), means=means, covs=covs)
    X = g.sample(20000, seed=0)
    assert X.shape == (20000, 2)
    np.testing.assert_allclose(X.mean(0), [5.0, 5.0], atol=0.15)


def test_n_params_full_and_diag():
    g = GMM(weights=np.ones(2) / 2, means=np.zeros((2, 3)), covs=np.stack([np.eye(3)] * 2))
    assert g.n_params() == 1 + 2 * 3 + 2 * 6      # (J-1) + J*m + J*m(m+1)/2
    g.covariance = "diag"
    assert g.n_params() == 1 + 2 * 3 + 2 * 3


def _two_blob_data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[-3.0, 0.0], scale=1.0, size=(n // 2, 2))
    b = rng.normal(loc=[+3.0, 1.0], scale=1.0, size=(n // 2, 2))
    return np.concatenate([a, b])


def test_fit_gmm_recovers_two_separated_clusters():
    X = _two_blob_data()
    g = fit_gmm(X, J=2, seed=0)
    order = np.argsort(g.means[:, 0])
    np.testing.assert_allclose(g.means[order][0], [-3.0, 0.0], atol=0.3)
    np.testing.assert_allclose(g.means[order][1], [+3.0, 1.0], atol=0.3)
    np.testing.assert_allclose(np.sort(g.weights), [0.5, 0.5], atol=0.05)


def test_fit_gmm_improves_loglik_over_single_gaussian_on_bimodal():
    X = _two_blob_data()
    ll1 = fit_gmm(X, J=1, seed=0).logpdf(X).sum()
    ll2 = fit_gmm(X, J=2, seed=0).logpdf(X).sum()
    assert ll2 > ll1 + 50.0


def test_fit_gmm_is_deterministic_given_seed():
    X = _two_blob_data()
    g1 = fit_gmm(X, J=2, seed=7)
    g2 = fit_gmm(X, J=2, seed=7)
    np.testing.assert_allclose(g1.means, g2.means)


def test_fit_gmm_tiny_sample_does_not_crash():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 3))          # N < what J=3 wants
    g = fit_gmm(X, J=3, seed=0)
    assert np.all(np.isfinite(g.logpdf(X)))


def test_gmm_logpdf_matches_sklearn_score_samples():
    # our numpy evaluation layer must agree with sklearn's, given identical params
    from sklearn.mixture import GaussianMixture

    X = _two_blob_data()
    sk = GaussianMixture(n_components=2, random_state=0).fit(X)
    g = GMM(weights=sk.weights_, means=sk.means_, covs=sk.covariances_)
    np.testing.assert_allclose(g.logpdf(X), sk.score_samples(X), rtol=1e-10)
