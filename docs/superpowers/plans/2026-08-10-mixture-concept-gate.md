# Mixture ConceptGate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the per-concept density model from one Gaussian per class to a BIC-selected mixture `Set((mu, Sigma))` per class, fit jointly on the m-dim spectrogram, validated by a redone 3-scenario toy problem.

**Architecture:** New pure-numpy module `conceptgate/mixture.py` (GMM + EM + BIC selection). New `MixtureConceptGate` in `gate.py` sharing the diff-of-means front-end with `ConceptGate` (which stays behaviorally unchanged). New `scripts/toy_csg_mixture.py` proving: (1) no regression on unimodal data, (2) wins on bimodal benign, (3) wins where no linear filter can separate. Docs updated.

**Tech Stack:** Python >=3.10, numpy only inside `conceptgate/` (NO scipy/sklearn), pytest (dev dep) for unit tests, `uv` for env/run.

## Global Constraints

- Pure numpy in `conceptgate/` — no scipy, no sklearn (spec §4: "pure numpy, mirrors concept_bank.py style").
- `ConceptGate` results must remain unchanged; `scripts/toy_csg.py` must still print `VALIDATION PASS` after every task (spec §7).
- The repo is `package = false` (pyproject): scripts use the `sys.path.insert(0, ...)` pattern; tests need `tests/conftest.py` doing the same.
- Everything seeded/deterministic: EM/k-means take explicit `seed`; toy scenarios report mean±std over >= 5 seeds (spec §5).
- Run everything via `uv run` from the repo root.
- Duck-type contract for `GateBank`/`Guard` compatibility: gate exposes `llr(A)`, `fire(A)`, `decide(A)`, `W_raw`, `tau`, `abstain_margin` (see `guard.py:56`, `gate.py:126-140`).

---

### Task 1: Test infra + GMM density core (`GMM` dataclass: logpdf, sample)

**Files:**
- Create: `conceptgate/mixture.py`
- Create: `tests/conftest.py`
- Test: `tests/test_mixture.py`
- Modify: `pyproject.toml` (via `uv add --dev pytest`)

**Interfaces:**
- Produces: `GMM(weights [J], means [J,m], covs [J,m,m], covariance="full")` with `.logpdf(X [N,m]) -> [N]`, `.component_logpdf(X) -> [N,J]`, `.sample(n, seed) -> [n,m]`, `.n_components`, `.dim`, `.n_params()`; module helpers `_logsumexp(X, axis)`, `_shrink(cov, rho, floor)`.

- [ ] **Step 1: Add pytest dev dependency**

Run: `uv add --dev pytest`
Expected: pyproject gains `[dependency-groups] dev = ["pytest>=..."]`, `uv.lock` updated.

- [ ] **Step 2: Write conftest + failing density tests**

`tests/conftest.py`:
```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

`tests/test_mixture.py`:
```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_mixture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'conceptgate.mixture'`

- [ ] **Step 4: Implement the density core**

`conceptgate/mixture.py`:
```python
"""Pure-numpy Gaussian mixtures for spectrogram-space class densities.

A concept class is a SET of (weight, mu, Sigma) components on the joint m-dim
spectrogram: p(s | class) = sum_j pi_j N(s; mu_j, Sigma_j). One joint mixture per
class (its marginal at each tapped layer is automatically a mixture at that layer).
Fitting is k-means-initialized EM with covariance shrinkage for small samples; the
component count J is chosen by BIC, so scarce data collapses to J=1 — which is the
single-Gaussian gate — by design.

All inputs are numpy float arrays. No scipy / sklearn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

_LOG2PI = float(np.log(2.0 * np.pi))


def _logsumexp(X: np.ndarray, axis: int = -1) -> np.ndarray:
    mx = np.max(X, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(X - mx), axis=axis)) + np.squeeze(mx, axis=axis)


def _shrink(cov: np.ndarray, rho: float, floor: float = 1e-6) -> np.ndarray:
    """Shrink toward scaled identity: (1-rho)*cov + rho*(tr/m)*I, plus a ridge floor."""
    m = cov.shape[0]
    iso = (np.trace(cov) / m) * np.eye(m)
    return (1.0 - rho) * cov + rho * iso + floor * np.eye(m)


@dataclass
class GMM:
    """weights [J], means [J, m], covs [J, m, m]."""

    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    covariance: str = "full"   # "full" | "diag" — affects n_params (BIC) only

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[0])

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    def component_logpdf(self, X: np.ndarray) -> np.ndarray:
        """Per-component Gaussian log-density. X: [N, m] -> [N, J]."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        N, m = X.shape
        out = np.empty((N, self.n_components))
        for j in range(self.n_components):
            L = np.linalg.cholesky(self.covs[j])
            y = np.linalg.solve(L, (X - self.means[j]).T)       # [m, N] whitened
            maha = np.sum(y * y, axis=0)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            out[:, j] = -0.5 * (maha + logdet + m * _LOG2PI)
        return out

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        """Mixture log-density. X: [N, m] -> [N]."""
        lp = self.component_logpdf(X) + np.log(self.weights)[None, :]
        return _logsumexp(lp, axis=1)

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(n, self.weights / self.weights.sum())
        parts = [
            rng.multivariate_normal(self.means[j], self.covs[j], size=int(c))
            for j, c in enumerate(counts)
            if c > 0
        ]
        X = np.concatenate(parts, axis=0)
        return X[rng.permutation(n)]

    def n_params(self) -> int:
        J, m = self.n_components, self.dim
        cov_p = m if self.covariance == "diag" else m * (m + 1) // 2
        return (J - 1) + J * m + J * cov_p
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_mixture.py -v`
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/ conceptgate/mixture.py
git commit -m "feat(mixture): GMM density core (logpdf/sample/n_params) + pytest infra"
```

---

### Task 2: EM fitting (`fit_gmm`) with k-means init and shrinkage

**Files:**
- Modify: `conceptgate/mixture.py` (append)
- Test: `tests/test_mixture.py` (append)

**Interfaces:**
- Consumes: `GMM`, `_shrink`, `_logsumexp` from Task 1.
- Produces: `fit_gmm(X [N,m], J, covariance="full", shrinkage=0.1, seed=0, max_iter=200, tol=1e-6, n_init=3) -> GMM`; private `_kmeans(X, J, rng, iters=50) -> (centers [J,m], labels [N])`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_mixture.py`:
```python
from conceptgate.mixture import fit_gmm


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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_mixture.py -v -k fit_gmm`
Expected: FAIL with `ImportError: cannot import name 'fit_gmm'`

- [ ] **Step 3: Implement `_kmeans` + `fit_gmm`**

Append to `conceptgate/mixture.py`:
```python
def _kmeans(X: np.ndarray, J: int, rng: np.random.Generator, iters: int = 50):
    """Seeded k-means++ + Lloyd's. Returns (centers [J, m], labels [N])."""
    N = X.shape[0]
    centers = [X[rng.integers(N)]]
    for _ in range(1, J):
        d2 = np.min(np.stack([np.sum((X - c) ** 2, axis=1) for c in centers]), axis=0)
        p = d2 / max(float(d2.sum()), 1e-12)
        centers.append(X[rng.choice(N, p=p)])
    C = np.stack(centers)
    lab = np.zeros(N, dtype=int)
    for _ in range(iters):
        lab = np.argmin(np.sum((X[:, None, :] - C[None]) ** 2, axis=-1), axis=1)
        newC = np.stack(
            [X[lab == j].mean(0) if np.any(lab == j) else X[rng.integers(N)] for j in range(J)]
        )
        if np.allclose(newC, C):
            break
        C = newC
    return C, lab


def fit_gmm(
    X: np.ndarray,
    J: int,
    covariance: str = "full",
    shrinkage: float = 0.1,
    seed: int = 0,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_init: int = 3,
) -> GMM:
    """EM fit of a J-component GMM on X [N, m]; best of n_init seeded restarts."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    N, m = X.shape
    best_ll, best = -np.inf, None
    for init in range(n_init):
        rng = np.random.default_rng(seed + 1000 * init)
        C, lab = _kmeans(X, J, rng)
        w = np.array([max(float((lab == j).mean()), 1e-3) for j in range(J)])
        w = w / w.sum()
        mu = C.copy()
        cov = np.empty((J, m, m))
        for j in range(J):
            pts = X[lab == j]
            S = np.atleast_2d(np.cov(pts, rowvar=False)) if pts.shape[0] > 1 else np.eye(m)
            if covariance == "diag":
                S = np.diag(np.diag(S))
            cov[j] = _shrink(S, shrinkage)
        gmm = GMM(w, mu, cov, covariance)
        prev = -np.inf
        for _ in range(max_iter):
            lp = gmm.component_logpdf(X) + np.log(gmm.weights)[None, :]     # [N, J]
            norm = _logsumexp(lp, axis=1)                                   # [N]
            ll = float(norm.sum())
            R = np.exp(lp - norm[:, None])                                  # responsibilities
            # empty-component rescue: give the least-explained point to the dead component
            Nj = R.sum(0)
            for j in range(J):
                if Nj[j] < 1e-6:
                    k = int(np.argmin(norm))
                    R[k, :] = 0.0
                    R[k, j] = 1.0
            Nj = R.sum(0)
            w = Nj / float(N)
            mu = (R.T @ X) / Nj[:, None]
            for j in range(J):
                D = X - mu[j]
                S = (R[:, j][:, None] * D).T @ D / Nj[j]
                if covariance == "diag":
                    S = np.diag(np.diag(S))
                cov[j] = _shrink(S, shrinkage)
            gmm = GMM(w, mu, cov, covariance)
            if abs(ll - prev) < tol * max(1.0, abs(prev)):
                break
            prev = ll
        ll = float(gmm.logpdf(X).sum())
        if ll > best_ll:
            best_ll, best = ll, gmm
    return best
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mixture.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add conceptgate/mixture.py tests/test_mixture.py
git commit -m "feat(mixture): seeded k-means-init EM with shrinkage and empty-component rescue"
```

---

### Task 3: Model selection (`bic`, `select_gmm`)

**Files:**
- Modify: `conceptgate/mixture.py` (append)
- Test: `tests/test_mixture.py` (append)

**Interfaces:**
- Consumes: `GMM.n_params()`, `GMM.logpdf`, `fit_gmm`.
- Produces: `bic(gmm, X) -> float`; `select_gmm(X, Js=(1,2,3), **fit_kwargs) -> GMM` (BIC-minimizing).

- [ ] **Step 1: Write failing tests**

Append to `tests/test_mixture.py`:
```python
from conceptgate.mixture import bic, select_gmm


def test_select_gmm_picks_one_component_on_unimodal_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 2))
    assert select_gmm(X, seed=0).n_components == 1


def test_select_gmm_picks_two_components_on_bimodal_data():
    X = _two_blob_data(n=500)
    assert select_gmm(X, seed=0).n_components == 2


def test_select_gmm_collapses_to_one_on_scarce_samples():
    # few-shot guard (spec 3.2): ~10 samples must NOT support a multi-component fit
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 3))
    assert select_gmm(X, seed=0).n_components == 1


def test_bic_penalizes_parameters():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 2))
    g1 = fit_gmm(X, J=1, seed=0)
    g3 = fit_gmm(X, J=3, seed=0)
    assert bic(g1, X) < bic(g3, X)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_mixture.py -v -k "select or bic"`
Expected: FAIL with `ImportError: cannot import name 'bic'`

- [ ] **Step 3: Implement**

Append to `conceptgate/mixture.py`:
```python
def bic(gmm: GMM, X: np.ndarray) -> float:
    """Bayesian information criterion (lower is better)."""
    N = X.shape[0]
    return -2.0 * float(gmm.logpdf(X).sum()) + gmm.n_params() * float(np.log(N))


def select_gmm(X: np.ndarray, Js: Iterable[int] = (1, 2, 3), **fit_kwargs) -> GMM:
    """Fit each J and return the BIC-best mixture (scarce data -> J=1 by design)."""
    fits = [fit_gmm(X, J=int(J), **fit_kwargs) for J in Js]
    scores = [bic(g, X) for g in fits]
    return fits[int(np.argmin(scores))]
```

- [ ] **Step 4: Run full test file**

Run: `uv run pytest tests/test_mixture.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add conceptgate/mixture.py tests/test_mixture.py
git commit -m "feat(mixture): BIC model selection with few-shot collapse to J=1"
```

---

### Task 4: Refactor shared fit front-end in `gate.py` (behavior-preserving)

**Files:**
- Modify: `conceptgate/gate.py:46-68` (`ConceptGate.fit`)

**Interfaces:**
- Produces: module-level `_fit_directions(A_pos, A_neg) -> (mu0, sd0, W, W_raw, S_pos, S_neg)` — the standardize → diff-of-means → spectrogram front-end shared by both gate classes. `ConceptGate.fit` delegates to it; fitted attributes and results are bit-identical.

- [ ] **Step 1: Record baseline**

Run: `uv run python scripts/toy_csg.py`
Expected: `VALIDATION PASS` (note the printed err percentages: best ~16.1%, fisher ~9.4%).

- [ ] **Step 2: Extract the helper**

In `conceptgate/gate.py`, add above `class ConceptGate` (after `_gauss_logpdf`):
```python
def _fit_directions(A_pos: np.ndarray, A_neg: np.ndarray):
    """Shared few-shot front-end: standardization stats, directions, spectrograms.

    A_pos, A_neg: [N, m, d] -> (mu0, sd0, W, W_raw, S_pos, S_neg).
    """
    A_all = np.concatenate([A_pos, A_neg], axis=0)
    mu0 = A_all.mean(axis=0)                    # [m, d]
    sd0 = A_all.std(axis=0) + 1e-6              # [m, d]
    Zp = (A_pos - mu0) / sd0
    Zn = (A_neg - mu0) / sd0
    W = cb.fit_directions(Zp, Zn)
    W_raw = cb._normalize(A_pos.mean(0) - A_neg.mean(0), axis=-1)
    S_pos = cb.spectrogram(Zp, W)
    S_neg = cb.spectrogram(Zn, W)
    return mu0, sd0, W, W_raw, S_pos, S_neg
```

Replace the body of `ConceptGate.fit` (keep its docstring) so it reads:
```python
        self.mu0, self.sd0, self.W, self.W_raw, S_pos, S_neg = _fit_directions(A_pos, A_neg)
        self.f = cb.fit_bandpass(S_pos, S_neg, method=self.filter_method)
        self.train_dprime = cb.dprime_per_layer(S_pos, S_neg)
        sp = cb.filtered_score(S_pos, self.f)
        sn = cb.filtered_score(S_neg, self.f)
        self.mu_pos, self.sigma_pos = float(sp.mean()), float(sp.std(ddof=1))
        self.mu_neg, self.sigma_neg = float(sn.mean()), float(sn.std(ddof=1))
        return self
```

- [ ] **Step 3: Verify no behavior change**

Run: `uv run python scripts/toy_csg.py && uv run pytest tests/ -v`
Expected: `VALIDATION PASS` with identical err percentages to Step 1; all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add conceptgate/gate.py
git commit -m "refactor(gate): extract shared _fit_directions front-end (behavior-preserving)"
```

---

### Task 5: `MixtureConceptGate`

**Files:**
- Modify: `conceptgate/gate.py` (append after `ConceptGate`, before `GateBank`)
- Test: `tests/test_mixture_gate.py`

**Interfaces:**
- Consumes: `_fit_directions` (Task 4), `mixture.select_gmm` / `GMM` (Tasks 1-3).
- Produces: `MixtureConceptGate(name, Js=(1,2,3), covariance="full", shrinkage=0.1, seed=0, tau=0.0, abstain_margin=0.0)` with `.fit(A_pos, A_neg)`, `.spectro(A) -> [N,m]`, `.llr(A) -> [N]`, `.decide(A)`, `.fire(A)`, `.calibrate_threshold(A_neg_cal, target_fpr)`, `.calibrate_z(z, n_samples=10_000)`, attributes `W, W_raw, mu0, sd0, gmm_pos, gmm_neg, train_dprime`. Satisfies the `GateBank`/`Guard` duck-type contract (`llr`, `fire`, `decide`, `W_raw`).

- [ ] **Step 1: Write failing tests**

`tests/test_mixture_gate.py`:
```python
import numpy as np

from conceptgate import concept_bank as cb
from conceptgate.gate import ConceptGate, GateBank, MixtureConceptGate, error_at_zero


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
    g = MixtureConceptGate(name="toy").fit(A_pos[:1000], A_neg[:1000])
    lp = g.llr(A_pos[1000:])
    ln = g.llr(A_neg[1000:])
    assert lp.mean() > ln.mean() + 2.0


def test_unimodal_data_selects_one_component_and_matches_fisher():
    # continuity (spec 3.1): J=1 collapse, error within 1.5 points of fisher ConceptGate
    A_pos, A_neg = _toy_data()
    tr, te = slice(0, 1000), slice(1000, 2000)
    mg = MixtureConceptGate(name="toy").fit(A_pos[tr], A_neg[tr])
    assert mg.gmm_pos.n_components == 1
    assert mg.gmm_neg.n_components == 1
    err_mix = error_at_zero(mg.llr(A_pos[te]), mg.llr(A_neg[te]))
    fg = ConceptGate(name="toy", filter_method="fisher").fit(A_pos[tr], A_neg[tr])
    sp = fg.score(A_pos[te]) - 0.5 * (fg.mu_pos + fg.mu_neg)
    sn = fg.score(A_neg[te]) - 0.5 * (fg.mu_pos + fg.mu_neg)
    err_fisher = error_at_zero(sp, sn)
    assert abs(err_mix - err_fisher) < 0.015


def test_duck_type_contract_for_bank_and_guard():
    A_pos, A_neg = _toy_data(n=400)
    g = MixtureConceptGate(name="toy").fit(A_pos, A_neg)
    assert g.W_raw is not None and g.W_raw.shape == g.W.shape
    bank = GateBank().add(g)
    fired = bank.fire(A_pos[:8])
    assert fired.shape == (8,) and fired.dtype == bool
    assert bank.which(A_pos[:8]).shape == (8,)


def test_calibrate_threshold_hits_target_fpr():
    A_pos, A_neg = _toy_data(n=3000)
    g = MixtureConceptGate(name="toy").fit(A_pos[:1000], A_neg[:1000])
    g.calibrate_threshold(A_neg[1000:2000], target_fpr=0.05)
    fpr = float(np.mean(g.fire(A_neg[2000:])))
    assert fpr < 0.10


def test_calibrate_z_gives_low_benign_fire_rate():
    A_pos, A_neg = _toy_data(n=2000)
    g = MixtureConceptGate(name="toy").fit(A_pos[:1000], A_neg[:1000])
    tau = g.calibrate_z(z=3.0)
    assert np.isfinite(tau)
    assert float(np.mean(g.fire(A_neg[1000:]))) < 0.02
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_mixture_gate.py -v`
Expected: FAIL with `ImportError: cannot import name 'MixtureConceptGate'`

- [ ] **Step 3: Implement**

In `conceptgate/gate.py`: add imports `import math` (top, stdlib block) and `from . import mixture as mx`; append after `ConceptGate`:
```python
def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class MixtureConceptGate:
    """One concept as a SET of (mu, Sigma) components per class on the joint spectrogram.

    Density upgrade of ConceptGate: class-conditional GMMs fitted directly in
    spectrogram space R^m (J chosen by BIC; J=1 with shared covariance recovers the
    fisher-bandpass gate). Directions, standardization, and steering (W_raw) are
    identical to ConceptGate, so GateBank / Guard work unchanged.
    """

    name: str = "concept"
    Js: tuple = (1, 2, 3)
    covariance: str = "full"
    shrinkage: float = 0.1
    seed: int = 0
    tau: float = 0.0                 # LLR fire threshold (>tau -> fire)
    abstain_margin: float = 0.0      # if >0, |LLR - tau| < margin -> abstain
    # learned params (set by fit)
    W: Optional[np.ndarray] = None
    W_raw: Optional[np.ndarray] = None
    mu0: Optional[np.ndarray] = None
    sd0: Optional[np.ndarray] = None
    gmm_pos: Optional[mx.GMM] = None
    gmm_neg: Optional[mx.GMM] = None
    train_dprime: Optional[np.ndarray] = None

    def fit(self, A_pos: np.ndarray, A_neg: np.ndarray) -> "MixtureConceptGate":
        """A_pos, A_neg: [N, m, d] activation samples (last-token rep per prompt)."""
        self.mu0, self.sd0, self.W, self.W_raw, S_pos, S_neg = _fit_directions(A_pos, A_neg)
        self.train_dprime = cb.dprime_per_layer(S_pos, S_neg)
        kw = dict(covariance=self.covariance, shrinkage=self.shrinkage, seed=self.seed)
        self.gmm_pos = mx.select_gmm(S_pos, Js=self.Js, **kw)
        self.gmm_neg = mx.select_gmm(S_neg, Js=self.Js, **kw)
        return self

    # --- scoring ---
    def spectro(self, A: np.ndarray) -> np.ndarray:
        """Standardized spectrogram. A: [N, m, d] -> [N, m]."""
        Z = (A - self.mu0) / self.sd0
        return cb.spectrogram(Z, self.W)

    def llr(self, A: np.ndarray) -> np.ndarray:
        S = self.spectro(A)
        return self.gmm_pos.logpdf(S) - self.gmm_neg.logpdf(S)

    def fire(self, A: np.ndarray) -> np.ndarray:
        return self.decide(A) > 0

    def decide(self, A: np.ndarray) -> np.ndarray:
        """+1 fire, 0 abstain, -1 pass."""
        l = self.llr(A)
        out = np.where(l > self.tau, 1, -1)
        if self.abstain_margin > 0:
            out = np.where(np.abs(l - self.tau) < self.abstain_margin, 0, out)
        return out

    # --- calibration ---
    def calibrate_threshold(self, A_neg_cal: np.ndarray, target_fpr: float = 0.05) -> float:
        """Set tau so the false-positive rate on calibration negatives ~= target_fpr."""
        l = np.sort(self.llr(A_neg_cal))
        q = float(np.clip(1.0 - target_fpr, 0.0, 1.0))
        self.tau = float(np.quantile(l, q))
        return self.tau

    def calibrate_z(self, z: float = 3.0, n_samples: int = 10_000) -> float:
        """Benign-mixture quantile operating point (the mixture analogue of 'mean + z*sd').

        Draw samples from the fitted benign GMM (no model calls), evaluate their LLRs,
        and put tau at the 1 - Phi(-z) quantile: z=3 -> ~0.1% benign-tail FPR.
        """
        S = self.gmm_neg.sample(n_samples, seed=self.seed)
        l = self.gmm_pos.logpdf(S) - self.gmm_neg.logpdf(S)
        self.tau = float(np.quantile(l, 1.0 - _phi(-z)))
        return self.tau
```

- [ ] **Step 4: Run all tests + regression script**

Run: `uv run pytest tests/ -v && uv run python scripts/toy_csg.py`
Expected: all PASS; `VALIDATION PASS`.

- [ ] **Step 5: Commit**

```bash
git add conceptgate/gate.py tests/test_mixture_gate.py
git commit -m "feat(gate): MixtureConceptGate — Set((mu,Sigma)) per class on the joint spectrogram"
```

---

### Task 6: Toy validation redo (`scripts/toy_csg_mixture.py`)

**Files:**
- Create: `scripts/toy_csg_mixture.py`

**Interfaces:**
- Consumes: `ConceptGate` (methods `best`/`fisher`), `MixtureConceptGate`, `error_at_zero`, `cb._normalize`.
- Produces: a runnable validation script printing per-scenario tables and a final `MIXTURE VALIDATION PASS/FAIL`; exit code 0/1.

- [ ] **Step 1: Write the script**

`scripts/toy_csg_mixture.py`:
```python
"""Offline validation of the mixture upgrade on synthetic activations (no model).

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
from conceptgate.gate import ConceptGate, MixtureConceptGate, error_at_zero

SEEDS = [0, 1, 2, 3, 4]
N_TRAIN = 3000   # per class
N_TEST = 3000


def synth_modes(rng, n, U, gains, modes, weights):
    """[n, m, d] activations. Mode sampled per SAMPLE (shared across layers);
    layer-l mean = modes[j] * gains[l] * U[l]; isotropic unit noise."""
    m, d = U.shape
    idx = rng.choice(len(modes), size=n, p=weights)
    A = rng.standard_normal((n, m, d))
    off = np.asarray(modes, dtype=float)[idx]                    # [n]
    for l in range(m):
        A[:, l, :] += (off * gains[l])[:, None] * U[l][None, :]
    return A


def true_logpdf(A, U, gains, modes, weights):
    """Exact class log-density of the generator (for the Bayes floor). A: [n, m, d]."""
    n, m, d = A.shape
    parts = []
    for j, mode in enumerate(modes):
        maha = np.zeros(n)
        for l in range(m):
            diff = A[:, l, :] - mode * gains[l] * U[l][None, :]
            maha += np.sum(diff * diff, axis=1)
        parts.append(np.log(weights[j]) - 0.5 * maha)            # const dropped (cancels in LLR)
    P = np.stack(parts, axis=1)
    mx_ = P.max(axis=1, keepdims=True)
    return np.log(np.exp(P - mx_).sum(axis=1)) + mx_[:, 0]


def bayes_error(A_pos, A_neg, U, gains, spec_pos, spec_neg):
    lp = lambda A: true_logpdf(A, U, gains, *spec_pos) - true_logpdf(A, U, gains, *spec_neg)
    return error_at_zero(lp(A_pos), lp(A_neg))


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
            g = ConceptGate(name=name, filter_method=method).fit(Ap[tr], An[tr])
            sp = g.score(Ap[te]) - 0.5 * (g.mu_pos + g.mu_neg)
            sn = g.score(An[te]) - 0.5 * (g.mu_pos + g.mu_neg)
            errs[method].append(error_at_zero(sp, sn))
            if method == "fisher":
                aucs["fisher"].append(auc(sp, sn))

        mg = MixtureConceptGate(name=name).fit(Ap[tr], An[tr])
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
```

- [ ] **Step 2: Run it**

Run: `uv run python scripts/toy_csg_mixture.py`
Expected: `MIXTURE VALIDATION PASS`. If a tolerance check fails, inspect the printed
table: S2/S3 constructions may need mode/weight nudges (e.g. S2 near-mode 0.45 -> 0.5,
S3 weights 0.6/0.4 -> 0.65/0.35) — adjust the SCENARIOS dict only, never the checks.

- [ ] **Step 3: Confirm old validations still pass**

Run: `uv run python scripts/toy_csg.py && uv run pytest tests/ -q`
Expected: `VALIDATION PASS`; all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/toy_csg_mixture.py
git commit -m "feat(toy): 3-scenario mixture validation — regression, bimodal benign, kill shot"
```

---

### Task 7: Documentation update + final verification

**Files:**
- Modify: `docs/math.md` (insert new section after §5, renumber not required — use "5b")
- Modify: `docs/concepts.md` (extend §5 area with the mode story; update §8 roadmap)
- Modify: `README.md` (layout, run, status)

**Interfaces:**
- Consumes: final APIs from Tasks 1-6 (names must match exactly: `mixture.py`, `GMM`, `fit_gmm`, `select_gmm`, `bic`, `MixtureConceptGate`, `toy_csg_mixture.py`).

- [ ] **Step 1: math.md — add §5b after §5 (before §6)**

Insert:
```markdown
## 5b. The mixture upgrade: a concept class as a set of (μ, Σ)

A single Gaussian per class (§7) assumes each class is one blob. Realistically a class
is a **set of modes** ("benign" = chit-chat, homework, code, …). We therefore model each
class directly on the joint spectrogram $\mathbf{s}\in\mathbb{R}^m$ as a Gaussian mixture:

$$
p(\mathbf{s}\mid c)=\sum_{j=1}^{J_c}\pi_{cj}\,\mathcal{N}(\mathbf{s};\mu_{cj},\Sigma_{cj}),
\qquad c\in\{+,-\},
$$

and gate on $\mathrm{LLR}(\mathbf{s})=\log p(\mathbf{s}\mid +)-\log p(\mathbf{s}\mid -)>\tau$.

**Why joint, not per-layer:** each $\mu_{cj}$ is a *profile across depth*; the joint
mixture's marginal at any layer is automatically a mixture at that layer, while
cross-layer correlations are kept (a per-layer-then-combine scheme loses them, and a
"mixture of per-layer mixtures" is not a coherent density).

**Continuity (nothing is lost):** with $J=1$ per class and shared covariance
$\Sigma_\mathbf{s}$, the LLR is affine in $\mathbf{s}$ with normal vector
$\Sigma_\mathbf{s}^{-1}(\bar{\mathbf{s}}^+-\bar{\mathbf{s}}^-)$ — exactly the `fisher`
bandpass of §5. With $J>1$ the effective filter becomes **input-dependent**: locally a
responsibility-weighted blend of per-component matched filters.

**Fitting:** seeded k-means-init EM with covariance shrinkage
$\Sigma\leftarrow(1-\rho)\Sigma+\rho\,\tfrac{\operatorname{tr}\Sigma}{m}I$
(default $\rho=0.1$); $J_c$ chosen by **BIC** over $\{1,2,3\}$ — scarce (10-shot) data
collapses to $J=1$, i.e. the §7 gate, by design.

**Calibration:** the FPR-quantile rule (§7) is unchanged. The $z$-based rule becomes a
benign-mixture quantile: draw ~10k samples from $p(\mathbf{s}\mid-)$, set $\tau$ at the
$1-\Phi(-z)$ quantile of their LLRs (z=3 → ~0.1% benign-tail FPR).

**Cost:** $J\,(m + \tfrac{m(m+1)}{2} + 1)$ numbers per class per concept — for $m=5$,
$J\le3$: ≤ 63 extra numbers. Code: `conceptgate/mixture.py` (GMM/EM/BIC),
`MixtureConceptGate` in `conceptgate/gate.py`, validation in `scripts/toy_csg_mixture.py`.
```

Also append to the §11 code-map table:
```markdown
| GMM density, EM fit, BIC selection (§5b) | `conceptgate/mixture.py` |
| `MixtureConceptGate` (§5b gate) | `conceptgate/gate.py` |
| §5b validation (regression / bimodal / kill-shot) | `scripts/toy_csg_mixture.py` |
```

- [ ] **Step 2: concepts.md — extend the mental model and roadmap**

After §5's mixing-board paragraph (before §6), insert:
```markdown
### 5b. A concept class is a set of modes

One Gaussian per class says "benign sounds like *one* hum". Really each class is a
**set of sound profiles** — benign chit-chat, benign homework, benign code — so the
gate now models each class as a *set of (μ, Σ) components* over the depth profile
(a Gaussian mixture on the spectrogram; the set size is picked by the data via BIC,
and ~10-shot data collapses to one component — the original gate). Two payoffs:
sharper thresholds when a class is multi-modal, and an *input-dependent* bandpass —
each mode contributes its own filter, weighted by how well it explains the sample.
The kill-shot case: if benign modes *flank* harmful on the filter axis, **no** single
linear filter separates them, but the mixture LLR does (`scripts/toy_csg_mixture.py`).
```

In §8 roadmap, mark the mixture work done by inserting after the P0 bullet:
```markdown
- **P0.5 (done):** density upgrade — class-conditional Gaussian *mixtures* on the joint
  spectrogram (BIC-selected, J=1 collapse on 10-shot). Toy: regression preserved;
  bimodal-benign and no-linear-filter scenarios recovered near the Bayes floor.
```

- [ ] **Step 3: README.md — layout, run, status**

In the Layout block, after the `concept_bank.py` line add:
```
  mixture.py        # Set((mu,Sigma)) per class: pure-numpy GMM (EM + BIC) on the spectrogram
```
and under `scripts/` after `toy_csg.py`:
```
  toy_csg_mixture.py # mixture validation: regression / bimodal benign / kill-shot scenarios
```
In the Run block add:
```bash
uv run python scripts/toy_csg_mixture.py  # mixture upgrade    -> MIXTURE VALIDATION PASS
uv run pytest tests/ -q                   # unit tests
```
In Status, after the P0 bullet add:
```markdown
- **P0.5 (done):** mixture upgrade — a concept class is a `Set((mu, Sigma))` (GMM on the
  joint spectrogram, BIC-selected J, 10-shot collapses to J=1). Regression toy preserved;
  where no linear filter separates (flanking benign modes), the mixture LLR recovers the
  Bayes floor. Next: signal-processor restructure (early short-circuit; see spec).
```

- [ ] **Step 4: Final full verification**

Run: `uv run pytest tests/ -q && uv run python scripts/toy_csg.py && uv run python scripts/toy_csg_mixture.py`
Expected: all tests PASS; `VALIDATION PASS`; `MIXTURE VALIDATION PASS`.

Run: `uv run python scripts/p0_smoke.py` (spec §7: ConceptGate end-to-end unchanged;
needs the cached GPT-2 weights — note the result, don't block on a download failure)
Expected: `P0 SMOKE: PASS`

- [ ] **Step 5: Commit**

```bash
git add docs/math.md docs/concepts.md README.md
git commit -m "docs: mixture upgrade — Set((mu,Sigma)) per class (math 5b, concepts 5b, README P0.5)"
```
