"""Tests for batch dose-response fitting (CPU and GPU)."""

import numpy as np
import pytest

from pystatsbio.doseresponse import fit_drm, fit_drm_batch, ll4

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_data():
    """Generate K=10 compounds with known EC50s."""
    np.random.seed(42)
    K, N = 10, 8
    dose = np.tile(
        [0, 0.01, 0.1, 1, 10, 100, 1000, 10000], (K, 1)
    ).astype(float)

    ec50_true = np.logspace(-1, 2, K)  # 0.1 to 100
    response = np.zeros((K, N))
    for i in range(K):
        response[i] = ll4(dose[i], 10, 90, ec50_true[i], 1.5) + np.random.normal(0, 2, N)

    return dose, response, ec50_true


# ---------------------------------------------------------------------------
# CPU batch
# ---------------------------------------------------------------------------

class TestBatchCPU:
    """CPU batch fitting."""

    def test_n_compounds(self, batch_data):
        dose, response, _ = batch_data
        r = fit_drm_batch(dose, response, backend="cpu")
        assert r.n_compounds == dose.shape[0]

    def test_all_converged(self, batch_data):
        dose, response, _ = batch_data
        r = fit_drm_batch(dose, response, backend="cpu")
        assert r.converged.sum() == r.n_compounds

    def test_ec50_recovery(self, batch_data):
        """EC50 values should be within 50% of true values."""
        dose, response, ec50_true = batch_data
        r = fit_drm_batch(dose, response, backend="cpu")
        for i in range(r.n_compounds):
            if r.converged[i]:
                assert r.ec50[i] == pytest.approx(ec50_true[i], rel=0.5)

    def test_matches_sequential(self, batch_data):
        """Batch CPU should match sequential fit_drm per compound."""
        dose, response, _ = batch_data
        r_batch = fit_drm_batch(dose, response, backend="cpu")

        for i in range(r_batch.n_compounds):
            r_single = fit_drm(dose[i], response[i])
            assert r_batch.ec50[i] == pytest.approx(r_single.params.ec50, rel=0.01)
            assert r_batch.hill[i] == pytest.approx(r_single.params.hill, rel=0.01)

    def test_rss_positive(self, batch_data):
        dose, response, _ = batch_data
        r = fit_drm_batch(dose, response, backend="cpu")
        assert np.all(r.rss > 0)

    def test_output_shapes(self, batch_data):
        dose, response, _ = batch_data
        K = dose.shape[0]
        r = fit_drm_batch(dose, response, backend="cpu")
        assert r.ec50.shape == (K,)
        assert r.hill.shape == (K,)
        assert r.top.shape == (K,)
        assert r.bottom.shape == (K,)
        assert r.converged.shape == (K,)
        assert r.rss.shape == (K,)


# ---------------------------------------------------------------------------
# GPU batch (skip if no torch)
# ---------------------------------------------------------------------------

class TestBatchGPU:
    """GPU batch fitting (requires torch)."""

    @pytest.fixture(autouse=True)
    def requires_torch(self):
        pytest.importorskip("torch")

    def test_gpu_runs(self, batch_data):
        """GPU batch should run without error (even on CPU-torch)."""
        dose, response, _ = batch_data
        # Force GPU path — will use CPU-torch if no real GPU
        r = fit_drm_batch(dose, response, backend="gpu")
        assert r.n_compounds == dose.shape[0]

    def test_gpu_convergence(self, batch_data):
        dose, response, _ = batch_data
        r = fit_drm_batch(dose, response, backend="gpu")
        # Most should converge
        assert r.converged.sum() >= r.n_compounds // 2

    def test_gpu_ec50_reasonable(self, batch_data):
        """GPU EC50s should be in the right ballpark."""
        dose, response, ec50_true = batch_data
        r = fit_drm_batch(dose, response, backend="gpu")
        for i in range(r.n_compounds):
            if r.converged[i]:
                assert r.ec50[i] == pytest.approx(ec50_true[i], rel=1.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestBatchValidation:
    """Input validation for batch fitting."""

    def test_not_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            fit_drm_batch(np.array([1, 2, 3]), np.array([1, 2, 3]))

    def test_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            fit_drm_batch(np.ones((3, 4)), np.ones((3, 5)))

    def test_unsupported_model(self):
        with pytest.raises(ValueError, match="LL.4"):
            fit_drm_batch(np.ones((3, 5)), np.ones((3, 5)), model="LL.5")
