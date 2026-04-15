"""Tests for ROC analysis with DeLong confidence intervals."""

import numpy as np
import pytest

from pystatsbio.diagnostic import ROCResult, roc

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def well_separated():
    """Well-separated cases and controls → high AUC."""
    np.random.seed(42)
    response = np.array([0] * 100 + [1] * 100)
    predictor = np.concatenate([
        np.random.normal(3, 1, 100),
        np.random.normal(6, 1, 100),
    ])
    return response, predictor


@pytest.fixture
def moderate_separation():
    """Moderate separation → mid AUC."""
    np.random.seed(123)
    response = np.array([0] * 80 + [1] * 80)
    predictor = np.concatenate([
        np.random.normal(5, 2, 80),
        np.random.normal(7, 2, 80),
    ])
    return response, predictor


# ---------------------------------------------------------------------------
# Basic ROC
# ---------------------------------------------------------------------------

class TestROCBasic:
    """Basic ROC curve properties."""

    def test_returns_roc_result(self, well_separated):
        r = roc(*well_separated)
        assert isinstance(r, ROCResult)

    def test_auc_high_for_separation(self, well_separated):
        r = roc(*well_separated)
        assert r.auc > 0.95

    def test_auc_moderate(self, moderate_separation):
        r = roc(*moderate_separation)
        assert 0.5 < r.auc < 0.95

    def test_auc_bounded(self, well_separated):
        r = roc(*well_separated)
        assert 0.0 <= r.auc <= 1.0

    def test_se_positive(self, well_separated):
        r = roc(*well_separated)
        assert r.auc_se > 0

    def test_ci_contains_auc(self, well_separated):
        r = roc(*well_separated)
        assert r.auc_ci_lower <= r.auc <= r.auc_ci_upper

    def test_ci_in_01(self, well_separated):
        r = roc(*well_separated)
        assert 0.0 <= r.auc_ci_lower <= 1.0
        assert 0.0 <= r.auc_ci_upper <= 1.0

    def test_counts(self, well_separated):
        r = roc(*well_separated)
        assert r.n_positive == 100
        assert r.n_negative == 100

    def test_roc_curve_endpoints(self, well_separated):
        """ROC curve should start at (0,0) and end at (1,1)."""
        r = roc(*well_separated)
        assert r.fpr[0] == 0.0 and r.tpr[0] == 0.0
        assert r.fpr[-1] == 1.0 and r.tpr[-1] == 1.0

    def test_fpr_tpr_monotonic(self, well_separated):
        """TPR and FPR should be non-decreasing along the curve."""
        r = roc(*well_separated)
        assert np.all(np.diff(r.fpr) >= 0)
        assert np.all(np.diff(r.tpr) >= 0)


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------

class TestROCDirection:
    """Direction auto-detection and manual control."""

    def test_auto_direction_lt(self, well_separated):
        """Cases > controls → direction '<'."""
        r = roc(*well_separated)
        assert r.direction == "<"

    def test_auto_direction_gt(self, well_separated):
        """Negate predictor: controls > cases → direction '>'."""
        response, predictor = well_separated
        r = roc(response, -predictor)
        assert r.direction == ">"

    def test_auto_auc_geq_half(self, well_separated):
        """Auto direction should always give AUC >= 0.5."""
        r = roc(*well_separated)
        assert r.auc >= 0.5

    def test_forced_direction(self, well_separated):
        response, predictor = well_separated
        r = roc(response, predictor, direction="<")
        assert r.direction == "<"

    def test_invalid_direction(self, well_separated):
        with pytest.raises(ValueError, match="direction"):
            roc(*well_separated, direction="up")


# ---------------------------------------------------------------------------
# DeLong CI
# ---------------------------------------------------------------------------

class TestDeLongCI:
    """DeLong CI properties."""

    def test_wider_at_lower_conf(self, well_separated):
        """99% CI should be wider than 90% CI."""
        r_90 = roc(*well_separated, conf_level=0.90)
        r_99 = roc(*well_separated, conf_level=0.99)
        width_90 = r_90.auc_ci_upper - r_90.auc_ci_lower
        width_99 = r_99.auc_ci_upper - r_99.auc_ci_lower
        assert width_99 > width_90

    def test_se_decreases_with_n(self):
        """Larger samples should give smaller SE."""
        np.random.seed(0)
        response_sm = np.array([0] * 30 + [1] * 30)
        pred_sm = np.concatenate([
            np.random.normal(3, 1, 30), np.random.normal(6, 1, 30),
        ])
        response_lg = np.array([0] * 300 + [1] * 300)
        pred_lg = np.concatenate([
            np.random.normal(3, 1, 300), np.random.normal(6, 1, 300),
        ])
        r_sm = roc(response_sm, pred_sm)
        r_lg = roc(response_lg, pred_lg)
        assert r_lg.auc_se < r_sm.auc_se


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestROCSummary:
    """Summary string."""

    def test_summary_has_auc(self, well_separated):
        r = roc(*well_separated)
        s = r.summary()
        assert "AUC" in s
        assert "DeLong" in s


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestROCValidation:
    """Input validation."""

    def test_non_binary_response(self):
        with pytest.raises(ValueError, match="binary"):
            roc(np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]))

    def test_mismatched_length(self):
        with pytest.raises(ValueError, match="same length"):
            roc(np.array([0, 1]), np.array([1.0, 2.0, 3.0]))

    def test_all_zeros_response(self):
        with pytest.raises(ValueError, match="binary"):
            roc(np.array([0, 0, 0]), np.array([1.0, 2.0, 3.0]))

    def test_invalid_conf_level(self, well_separated):
        with pytest.raises(ValueError, match="conf_level"):
            roc(*well_separated, conf_level=1.5)


# ---------------------------------------------------------------------------
# Random predictor (AUC ≈ 0.5)
# ---------------------------------------------------------------------------

class TestROCRandom:
    """Random predictor should give AUC near 0.5."""

    def test_random_auc_near_half(self):
        np.random.seed(99)
        n = 500
        response = np.array([0] * n + [1] * n)
        predictor = np.random.randn(2 * n)
        r = roc(response, predictor)
        assert 0.4 < r.auc < 0.6

    def test_random_ci_contains_half(self):
        np.random.seed(99)
        n = 500
        response = np.array([0] * n + [1] * n)
        predictor = np.random.randn(2 * n)
        r = roc(response, predictor)
        assert r.auc_ci_lower <= 0.5 <= r.auc_ci_upper
