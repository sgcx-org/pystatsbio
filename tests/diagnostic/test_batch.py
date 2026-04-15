"""Tests for batch AUC computation (CPU and GPU)."""

import numpy as np
import pytest

from pystatsbio.diagnostic import BatchAUCResult, batch_auc, roc

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_data():
    """Batch biomarker panel: 5 good markers + 5 random markers."""
    np.random.seed(42)
    n = 200
    M = 10
    response = np.array([0] * 100 + [1] * 100)
    predictors = np.random.randn(n, M)
    # Make first 5 discriminative
    for i in range(5):
        predictors[100:, i] += 2.0 + i * 0.5
    return response, predictors


# ---------------------------------------------------------------------------
# CPU batch
# ---------------------------------------------------------------------------

class TestBatchCPU:
    """CPU batch AUC."""

    def test_returns_result(self, batch_data):
        response, predictors = batch_data
        r = batch_auc(response, predictors, backend="cpu")
        assert isinstance(r, BatchAUCResult)

    def test_n_markers(self, batch_data):
        response, predictors = batch_data
        r = batch_auc(response, predictors, backend="cpu")
        assert r.n_markers == predictors.shape[1]

    def test_shapes(self, batch_data):
        response, predictors = batch_data
        r = batch_auc(response, predictors, backend="cpu")
        assert r.auc.shape == (10,)
        assert r.se.shape == (10,)

    def test_discriminative_markers_high_auc(self, batch_data):
        """First 5 markers should have AUC > 0.8."""
        response, predictors = batch_data
        r = batch_auc(response, predictors, backend="cpu")
        assert np.all(r.auc[:5] > 0.8)

    def test_random_markers_near_half(self, batch_data):
        """Last 5 markers should have AUC near 0.5."""
        response, predictors = batch_data
        r = batch_auc(response, predictors, backend="cpu")
        assert np.all(np.abs(r.auc[5:] - 0.5) < 0.15)

    def test_se_positive(self, batch_data):
        response, predictors = batch_data
        r = batch_auc(response, predictors, backend="cpu")
        assert np.all(r.se > 0)

    def test_auc_bounded(self, batch_data):
        response, predictors = batch_data
        r = batch_auc(response, predictors, backend="cpu")
        assert np.all(r.auc >= 0)
        assert np.all(r.auc <= 1)

    def test_matches_single_roc(self, batch_data):
        """Batch AUC should match single roc(direction='<') for each marker.

        batch_auc always uses direction='<' (higher = positive), so we
        must compare against roc() with a fixed direction, not 'auto'.
        """
        response, predictors = batch_data
        r_batch = batch_auc(response, predictors, backend="cpu")
        for m in range(r_batch.n_markers):
            r_single = roc(response, predictors[:, m], direction="<")
            assert r_batch.auc[m] == pytest.approx(r_single.auc, abs=1e-10)
            assert r_batch.se[m] == pytest.approx(r_single.auc_se, abs=1e-10)


# ---------------------------------------------------------------------------
# GPU batch
# ---------------------------------------------------------------------------

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class TestBatchGPU:
    """GPU batch AUC (CUDA only — MPS scatter_add_ is too slow)."""

    @pytest.fixture(autouse=True)
    def requires_cuda(self):
        if not _has_cuda():
            pytest.skip("batch_auc GPU requires CUDA (MPS not supported)")

    def test_gpu_runs(self, batch_data):
        response, predictors = batch_data
        r = batch_auc(response, predictors, backend="gpu")
        assert r.n_markers == predictors.shape[1]

    def test_gpu_matches_cpu(self, batch_data):
        """GPU results should match CPU results."""
        response, predictors = batch_data
        r_cpu = batch_auc(response, predictors, backend="cpu")
        r_gpu = batch_auc(response, predictors, backend="gpu")
        assert np.allclose(r_cpu.auc, r_gpu.auc, atol=0.01)
        assert np.allclose(r_cpu.se, r_gpu.se, atol=0.01)

    def test_gpu_auc_bounded(self, batch_data):
        response, predictors = batch_data
        r = batch_auc(response, predictors, backend="gpu")
        assert np.all(r.auc >= 0)
        assert np.all(r.auc <= 1)


class TestBatchMPSRejected:
    """MPS should raise RuntimeError, not silently crawl."""

    def test_mps_raises(self, batch_data):
        torch = pytest.importorskip("torch")
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("MPS not available")
        response, predictors = batch_data
        with pytest.raises(RuntimeError, match="not supported on MPS"):
            batch_auc(response, predictors, backend="gpu")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestBatchValidation:
    """Input validation."""

    def test_response_not_1d(self):
        with pytest.raises(ValueError, match="1-D"):
            batch_auc(np.array([[0, 1]]), np.array([[1.0, 2.0]]))

    def test_predictors_not_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            batch_auc(np.array([0, 1]), np.array([1.0, 2.0]))

    def test_mismatched_rows(self):
        with pytest.raises(ValueError, match="length"):
            batch_auc(np.array([0, 1, 1]), np.ones((4, 3)))

    def test_non_binary_response(self):
        with pytest.raises(ValueError, match="binary"):
            batch_auc(np.array([0, 1, 2, 3]), np.ones((4, 2)))
