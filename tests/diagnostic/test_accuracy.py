"""Tests for diagnostic accuracy metrics."""

import numpy as np
import pytest

from pystatsbio.diagnostic import DiagnosticResult, diagnostic_accuracy


@pytest.fixture
def biomarker_data():
    """Biomarker data with known sensitivity/specificity."""
    np.random.seed(42)
    response = np.array([0] * 100 + [1] * 100)
    predictor = np.concatenate([
        np.random.normal(3, 1, 100),
        np.random.normal(6, 1, 100),
    ])
    return response, predictor


# ---------------------------------------------------------------------------
# Basic
# ---------------------------------------------------------------------------

class TestDiagnosticAccuracy:
    """Basic diagnostic accuracy tests."""

    def test_returns_result(self, biomarker_data):
        r = diagnostic_accuracy(*biomarker_data, cutoff=4.5)
        assert isinstance(r, DiagnosticResult)

    def test_sensitivity_range(self, biomarker_data):
        r = diagnostic_accuracy(*biomarker_data, cutoff=4.5)
        assert 0 <= r.sensitivity <= 1

    def test_specificity_range(self, biomarker_data):
        r = diagnostic_accuracy(*biomarker_data, cutoff=4.5)
        assert 0 <= r.specificity <= 1

    def test_high_sensitivity_at_low_cutoff(self, biomarker_data):
        """Very low cutoff classifies almost everyone positive → high sensitivity."""
        r = diagnostic_accuracy(*biomarker_data, cutoff=0.0)
        assert r.sensitivity == 1.0

    def test_high_specificity_at_high_cutoff(self, biomarker_data):
        """Very high cutoff classifies almost nobody positive → high specificity."""
        r = diagnostic_accuracy(*biomarker_data, cutoff=10.0)
        assert r.specificity == 1.0

    def test_good_cutoff_gives_high_both(self, biomarker_data):
        """At a good cutoff, both sens and spec should be high."""
        r = diagnostic_accuracy(*biomarker_data, cutoff=4.5)
        assert r.sensitivity > 0.9
        assert r.specificity > 0.9


# ---------------------------------------------------------------------------
# CIs
# ---------------------------------------------------------------------------

class TestCIs:
    """Confidence interval properties."""

    def test_ci_contains_point(self, biomarker_data):
        r = diagnostic_accuracy(*biomarker_data, cutoff=4.5)
        assert r.sensitivity_ci[0] <= r.sensitivity <= r.sensitivity_ci[1]
        assert r.specificity_ci[0] <= r.specificity <= r.specificity_ci[1]

    def test_ci_in_01(self, biomarker_data):
        r = diagnostic_accuracy(*biomarker_data, cutoff=4.5)
        assert 0 <= r.sensitivity_ci[0] <= 1
        assert 0 <= r.sensitivity_ci[1] <= 1
        assert 0 <= r.specificity_ci[0] <= 1
        assert 0 <= r.specificity_ci[1] <= 1

    def test_wilson_ci(self, biomarker_data):
        r = diagnostic_accuracy(*biomarker_data, cutoff=4.5, ci_method="wilson")
        assert r.method == "wilson"
        assert r.sensitivity_ci[0] <= r.sensitivity <= r.sensitivity_ci[1]

    def test_invalid_ci_method(self, biomarker_data):
        with pytest.raises(ValueError, match="ci_method"):
            diagnostic_accuracy(*biomarker_data, cutoff=4.5, ci_method="wald")


# ---------------------------------------------------------------------------
# Predictive values
# ---------------------------------------------------------------------------

class TestPredictiveValues:
    """PPV, NPV, and prevalence adjustment."""

    def test_ppv_npv_in_01(self, biomarker_data):
        r = diagnostic_accuracy(*biomarker_data, cutoff=4.5)
        assert 0 <= r.ppv <= 1
        assert 0 <= r.npv <= 1

    def test_prevalence_adjustment(self, biomarker_data):
        """Higher prevalence should increase PPV."""
        r_lo = diagnostic_accuracy(*biomarker_data, cutoff=4.5, prevalence=0.1)
        r_hi = diagnostic_accuracy(*biomarker_data, cutoff=4.5, prevalence=0.5)
        assert r_hi.ppv > r_lo.ppv

    def test_default_prevalence_is_sample(self, biomarker_data):
        r = diagnostic_accuracy(*biomarker_data, cutoff=4.5)
        assert r.prevalence == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Likelihood ratios
# ---------------------------------------------------------------------------

class TestLikelihoodRatios:
    """LR+ and LR−."""

    def test_lr_positive_gt_1(self, biomarker_data):
        """Good test should have LR+ > 1."""
        r = diagnostic_accuracy(*biomarker_data, cutoff=4.5)
        assert r.lr_positive > 1

    def test_lr_negative_lt_1(self, biomarker_data):
        """Good test should have LR− < 1."""
        r = diagnostic_accuracy(*biomarker_data, cutoff=4.5)
        assert r.lr_negative < 1

    def test_lr_formula(self, biomarker_data):
        """LR+ = sens / (1-spec), LR- = (1-sens) / spec."""
        r = diagnostic_accuracy(*biomarker_data, cutoff=4.5)
        assert r.lr_positive == pytest.approx(
            r.sensitivity / (1 - r.specificity), rel=0.01
        )
        assert r.lr_negative == pytest.approx(
            (1 - r.sensitivity) / r.specificity, rel=0.01
        )


# ---------------------------------------------------------------------------
# DOR
# ---------------------------------------------------------------------------

class TestDOR:
    """Diagnostic odds ratio."""

    def test_dor_positive(self, biomarker_data):
        r = diagnostic_accuracy(*biomarker_data, cutoff=4.5)
        assert r.dor > 0

    def test_dor_ci_contains_estimate(self, biomarker_data):
        r = diagnostic_accuracy(*biomarker_data, cutoff=4.5)
        assert r.dor_ci[0] <= r.dor <= r.dor_ci[1]

    def test_dor_equals_lr_ratio(self, biomarker_data):
        """DOR should approximately equal LR+ / LR−."""
        r = diagnostic_accuracy(*biomarker_data, cutoff=4.5)
        assert r.dor == pytest.approx(r.lr_positive / r.lr_negative, rel=0.05)


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------

class TestDirection:
    """Direction parameter."""

    def test_direction_gt(self, biomarker_data):
        """direction='>' should swap the classification rule."""
        response, predictor = biomarker_data
        r = diagnostic_accuracy(response, -predictor, cutoff=-4.5, direction=">")
        assert r.sensitivity > 0.9

    def test_invalid_direction(self, biomarker_data):
        with pytest.raises(ValueError, match="direction"):
            diagnostic_accuracy(*biomarker_data, cutoff=4.5, direction="auto")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    """Summary output."""

    def test_summary_contains_metrics(self, biomarker_data):
        r = diagnostic_accuracy(*biomarker_data, cutoff=4.5)
        s = r.summary()
        assert "Sensitivity" in s
        assert "Specificity" in s
        assert "PPV" in s
        assert "DOR" in s


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestAccuracyValidation:
    """Input validation."""

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            diagnostic_accuracy(np.array([0, 1]), np.array([1.0, 2.0, 3.0]), cutoff=1.5)

    def test_invalid_conf_level(self, biomarker_data):
        with pytest.raises(ValueError, match="conf_level"):
            diagnostic_accuracy(*biomarker_data, cutoff=4.5, conf_level=0.0)
