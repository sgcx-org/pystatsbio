"""Tests for optimal cutoff selection."""

import numpy as np
import pytest

from pystatsbio.diagnostic import CutoffResult, optimal_cutoff, roc


@pytest.fixture
def roc_result():
    """ROC result for optimal cutoff tests."""
    np.random.seed(42)
    response = np.array([0] * 100 + [1] * 100)
    predictor = np.concatenate([
        np.random.normal(3, 1, 100),
        np.random.normal(6, 1, 100),
    ])
    return roc(response, predictor)


class TestYouden:
    """Youden index method."""

    def test_returns_cutoff_result(self, roc_result):
        c = optimal_cutoff(roc_result)
        assert isinstance(c, CutoffResult)

    def test_method_is_youden(self, roc_result):
        c = optimal_cutoff(roc_result, method="youden")
        assert c.method == "youden"

    def test_youden_criterion_positive(self, roc_result):
        """For a discriminative marker, Youden J > 0."""
        c = optimal_cutoff(roc_result)
        assert c.criterion_value > 0.5

    def test_sensitivity_specificity_bounded(self, roc_result):
        c = optimal_cutoff(roc_result)
        assert 0 <= c.sensitivity <= 1
        assert 0 <= c.specificity <= 1

    def test_youden_maximises_j(self, roc_result):
        """The optimal J should be the max of TPR - FPR."""
        c = optimal_cutoff(roc_result)
        finite = np.isfinite(roc_result.thresholds)
        j_all = roc_result.tpr[finite] - roc_result.fpr[finite]
        assert c.criterion_value == pytest.approx(j_all.max(), abs=1e-10)


class TestClosestToTopLeft:
    """Closest-to-top-left method."""

    def test_method_name(self, roc_result):
        c = optimal_cutoff(roc_result, method="closest_topleft")
        assert c.method == "closest_topleft"

    def test_distance_small(self, roc_result):
        """Distance should be small for a good marker."""
        c = optimal_cutoff(roc_result, method="closest_topleft")
        assert c.criterion_value < 0.2

    def test_distance_nonnegative(self, roc_result):
        c = optimal_cutoff(roc_result, method="closest_topleft")
        assert c.criterion_value >= 0


class TestCostBased:
    """Cost-based method."""

    def test_method_name(self, roc_result):
        c = optimal_cutoff(roc_result, method="cost")
        assert c.method == "cost"

    def test_equal_cost_similar_to_youden(self, roc_result):
        """With equal costs and balanced prevalence, cost ≈ youden."""
        c_y = optimal_cutoff(roc_result, method="youden")
        c_c = optimal_cutoff(roc_result, method="cost", prevalence=0.5)
        # Cutoffs should be similar (not necessarily identical)
        assert abs(c_y.cutoff - c_c.cutoff) < 1.0

    def test_high_fn_cost_favours_sensitivity(self, roc_result):
        """High FN cost should push toward higher sensitivity."""
        c_balanced = optimal_cutoff(roc_result, method="cost",
                                    cost_fp=1, cost_fn=1)
        c_fn_heavy = optimal_cutoff(roc_result, method="cost",
                                    cost_fp=1, cost_fn=10)
        assert c_fn_heavy.sensitivity >= c_balanced.sensitivity - 0.01


class TestCutoffValidation:
    """Input validation."""

    def test_invalid_method(self, roc_result):
        with pytest.raises(ValueError, match="method"):
            optimal_cutoff(roc_result, method="invalid")
