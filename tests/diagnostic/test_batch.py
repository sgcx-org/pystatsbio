"""Tests for batch AUC computation (CPU and GPU)."""

import numpy as np
import pytest

from pystatsbio.diagnostic import BatchAUCSolution, batch_auc, roc

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
        assert isinstance(r, BatchAUCSolution)

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


def _has_mps() -> bool:
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        return False


def _has_supported_gpu() -> bool:
    """CUDA, or MPS with the torch >= 2.13 native Metal kernels."""
    from pystatsbio.diagnostic._batch import _mps_native_kernels

    return _has_cuda() or (_has_mps() and _mps_native_kernels())


class TestBatchGPU:
    """GPU batch AUC (CUDA, or MPS with torch >= 2.13)."""

    @pytest.fixture(autouse=True)
    def requires_gpu(self):
        if not _has_supported_gpu():
            pytest.skip("batch_auc GPU requires CUDA, or MPS with torch >= 2.13")

    def test_gpu_runs(self, batch_data):
        response, predictors = batch_data
        r = batch_auc(response, predictors, backend="gpu")
        assert r.n_markers == predictors.shape[1]

    def test_gpu_matches_cpu_at_gpu_fp32_tier(self, batch_data):
        """GPU fp32 results must match the CPU fp64 reference at GPU_FP32."""
        from pystatistics.core.compute.tolerances import GPU_FP32

        response, predictors = batch_data
        r_cpu = batch_auc(response, predictors, backend="cpu")
        r_gpu = batch_auc(response, predictors, backend="gpu")
        assert np.allclose(r_cpu.auc, r_gpu.auc, rtol=GPU_FP32.rtol, atol=GPU_FP32.atol)
        assert np.allclose(r_cpu.se, r_gpu.se, rtol=GPU_FP32.rtol, atol=GPU_FP32.atol)

    def test_gpu_matches_cpu_with_ties(self):
        """Tied values stress the midrank tie-group scatter; must still match."""
        from pystatistics.core.compute.tolerances import GPU_FP32

        rng = np.random.default_rng(7)
        n, m = 300, 50
        response = np.zeros(n, dtype=np.intp)
        response[: n // 2] = 1
        rng.shuffle(response)
        # Coarsely discretized values: many ties per column
        predictors = np.round(rng.standard_normal((n, m)) * 2) / 2
        predictors[response == 1, : m // 5] += 1.0

        r_cpu = batch_auc(response, predictors, backend="cpu")
        r_gpu = batch_auc(response, predictors, backend="gpu")
        assert np.allclose(r_cpu.auc, r_gpu.auc, rtol=GPU_FP32.rtol, atol=GPU_FP32.atol)
        assert np.allclose(r_cpu.se, r_gpu.se, rtol=GPU_FP32.rtol, atol=GPU_FP32.atol)

    def test_gpu_auc_bounded(self, batch_data):
        response, predictors = batch_data
        r = batch_auc(response, predictors, backend="gpu")
        assert np.all(r.auc >= 0)
        assert np.all(r.auc <= 1)


class TestBatchMPS:
    """MPS gating: supported on torch >= 2.13, loud failure below."""

    def test_old_torch_raises(self, batch_data):
        """On torch < 2.13, MPS must raise RuntimeError, not silently crawl."""
        from pystatsbio.diagnostic._batch import _mps_native_kernels

        if not _has_mps():
            pytest.skip("MPS not available")
        if _has_cuda():
            pytest.skip("CUDA present — backend='gpu' resolves to CUDA")
        if _mps_native_kernels():
            pytest.skip("torch >= 2.13 — MPS is supported")
        response, predictors = batch_data
        with pytest.raises(RuntimeError, match="requires torch >= 2.13"):
            batch_auc(response, predictors, backend="gpu")

    def test_new_torch_runs_on_mps(self, batch_data):
        """On torch >= 2.13, backend='gpu' runs on MPS and reports it."""
        from pystatsbio.diagnostic._batch import _mps_native_kernels

        if not _has_mps():
            pytest.skip("MPS not available")
        if _has_cuda():
            pytest.skip("CUDA present — backend='gpu' resolves to CUDA")
        if not _mps_native_kernels():
            pytest.skip("torch < 2.13 — MPS not supported")
        response, predictors = batch_data
        r = batch_auc(response, predictors, backend="gpu")
        assert "mps" in r.backend_name

    def test_auto_routes_by_torch_version(self, batch_data):
        """'auto' picks MPS on torch >= 2.13, CPU below (MPS-only machines)."""
        from pystatsbio.diagnostic._batch import _mps_native_kernels

        if not _has_mps():
            pytest.skip("MPS not available")
        if _has_cuda():
            pytest.skip("CUDA present — 'auto' resolves to CUDA")
        response, predictors = batch_data
        r = batch_auc(response, predictors, backend="auto")
        if _mps_native_kernels():
            assert "mps" in r.backend_name
        else:
            assert r.backend_name == "cpu"


class TestMPSNativeKernelsPredicate:
    """Version parsing in the torch >= 2.13 predicate (machine-independent)."""

    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("2.12.1", False),
            ("2.13.0", True),
            ("2.13.0+cpu", True),
            ("2.14.0a0+git1234ab", True),
            ("3.0.0", True),
            ("2.9.0", False),
            ("garbage", False),
        ],
    )
    def test_versions(self, monkeypatch, version, expected):
        torch = pytest.importorskip("torch")
        from pystatsbio.diagnostic import _batch

        monkeypatch.setattr(torch, "__version__", version)
        assert _batch._mps_native_kernels() is expected


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
