"""Tests for DeLong test comparing two correlated ROC curves."""

import numpy as np
import pytest

from pystatsbio.diagnostic import ROCTestResult, roc, roc_test


@pytest.fixture
def two_markers():
    """Two markers on the same subjects: one strong, one weak."""
    np.random.seed(42)
    n = 200
    response = np.array([0] * 100 + [1] * 100)
    # Strong marker
    pred1 = np.concatenate([
        np.random.normal(3, 1, 100),
        np.random.normal(6, 1, 100),
    ])
    # Weak marker
    pred2 = np.concatenate([
        np.random.normal(3, 1, 100),
        np.random.normal(5, 1.5, 100),
    ])
    return response, pred1, pred2


class TestROCTest:
    """DeLong test for two correlated ROC curves."""

    def test_returns_roc_test_result(self, two_markers):
        response, pred1, pred2 = two_markers
        r1 = roc(response, pred1)
        r2 = roc(response, pred2)
        t = roc_test(r1, r2, predictor1=pred1, predictor2=pred2, response=response)
        assert isinstance(t, ROCTestResult)

    def test_significant_difference(self, two_markers):
        """Strong vs weak marker should give small p-value."""
        response, pred1, pred2 = two_markers
        r1 = roc(response, pred1)
        r2 = roc(response, pred2)
        t = roc_test(r1, r2, predictor1=pred1, predictor2=pred2, response=response)
        assert t.p_value < 0.01

    def test_same_marker_no_difference(self, two_markers):
        """Same marker compared to itself → p ≈ 1."""
        response, pred1, _ = two_markers
        r1 = roc(response, pred1)
        t = roc_test(r1, r1, predictor1=pred1, predictor2=pred1, response=response)
        assert t.p_value > 0.9
        assert abs(t.auc_diff) < 1e-10

    def test_auc_diff_sign(self, two_markers):
        """auc_diff = auc1 - auc2, and strong marker has higher AUC."""
        response, pred1, pred2 = two_markers
        r1 = roc(response, pred1)
        r2 = roc(response, pred2)
        t = roc_test(r1, r2, predictor1=pred1, predictor2=pred2, response=response)
        assert t.auc_diff > 0  # r1 is stronger
        assert t.auc_diff == pytest.approx(t.auc1 - t.auc2, abs=1e-10)

    def test_method_is_delong(self, two_markers):
        response, pred1, pred2 = two_markers
        r1 = roc(response, pred1)
        r2 = roc(response, pred2)
        t = roc_test(r1, r2, predictor1=pred1, predictor2=pred2, response=response)
        assert t.method == "delong"

    def test_requires_predictor_data(self, two_markers):
        response, pred1, pred2 = two_markers
        r1 = roc(response, pred1)
        r2 = roc(response, pred2)
        with pytest.raises(ValueError, match="required"):
            roc_test(r1, r2)

    def test_invalid_method(self, two_markers):
        response, pred1, pred2 = two_markers
        r1 = roc(response, pred1)
        r2 = roc(response, pred2)
        with pytest.raises(ValueError, match="delong"):
            roc_test(r1, r2, predictor1=pred1, predictor2=pred2,
                     response=response, method="bootstrap")

    def test_summary(self, two_markers):
        response, pred1, pred2 = two_markers
        r1 = roc(response, pred1)
        r2 = roc(response, pred2)
        t = roc_test(r1, r2, predictor1=pred1, predictor2=pred2, response=response)
        s = t.summary()
        assert "DeLong" in s
        assert "p-value" in s
