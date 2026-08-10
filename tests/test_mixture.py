import numpy as np

from conceptgate.mixture import GMM


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
