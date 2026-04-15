"""Tests for diagnostic/_common.py: ROCResult.summary() and DiagnosticResult.summary()."""

import numpy as np
import pytest

from pystatsbio.diagnostic import diagnostic_accuracy, roc

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def roc_result():
    np.random.seed(7)
    scores = np.concatenate([
        np.random.normal(0.3, 0.1, 50),
        np.random.normal(0.7, 0.1, 50),
    ])
    labels = np.array([0] * 50 + [1] * 50)
    return roc(labels, scores)


@pytest.fixture
def diag_result():
    np.random.seed(7)
    scores = np.concatenate([
        np.random.normal(0.3, 0.1, 50),
        np.random.normal(0.7, 0.1, 50),
    ])
    labels = np.array([0] * 50 + [1] * 50)
    return diagnostic_accuracy(labels, scores, cutoff=0.5)


# ---------------------------------------------------------------------------
# ROCResult
# ---------------------------------------------------------------------------

class TestROCResult:
    """ROCResult fields and summary()."""

    def test_auc_in_range(self, roc_result):
        assert 0.0 <= roc_result.auc <= 1.0

    def test_auc_above_chance(self, roc_result):
        assert roc_result.auc > 0.5

    def test_ci_bounds_ordered(self, roc_result):
        assert roc_result.auc_ci_lower < roc_result.auc < roc_result.auc_ci_upper

    def test_tpr_fpr_same_length(self, roc_result):
        assert len(roc_result.tpr) == len(roc_result.fpr)

    def test_tpr_starts_at_zero(self, roc_result):
        assert roc_result.tpr[0] == pytest.approx(0.0, abs=1e-9)

    def test_fpr_starts_at_zero(self, roc_result):
        assert roc_result.fpr[0] == pytest.approx(0.0, abs=1e-9)

    def test_tpr_ends_at_one(self, roc_result):
        assert roc_result.tpr[-1] == pytest.approx(1.0, abs=1e-9)

    def test_fpr_ends_at_one(self, roc_result):
        assert roc_result.fpr[-1] == pytest.approx(1.0, abs=1e-9)

    def test_sample_counts(self, roc_result):
        assert roc_result.n_positive == 50
        assert roc_result.n_negative == 50

    def test_frozen(self, roc_result):
        with pytest.raises(AttributeError):
            roc_result.auc = 0.5  # type: ignore[misc]

    def test_summary_contains_auc(self, roc_result):
        s = roc_result.summary()
        assert "AUC" in s

    def test_summary_contains_n_positive(self, roc_result):
        s = roc_result.summary()
        assert "n positive" in s

    def test_summary_contains_direction(self, roc_result):
        s = roc_result.summary()
        assert "Direction" in s

    def test_summary_contains_ci(self, roc_result):
        s = roc_result.summary()
        assert "CI" in s

    def test_summary_is_string(self, roc_result):
        assert isinstance(roc_result.summary(), str)


# ---------------------------------------------------------------------------
# DiagnosticResult
# ---------------------------------------------------------------------------

class TestDiagnosticResult:
    """DiagnosticResult fields and summary()."""

    def test_sensitivity_in_range(self, diag_result):
        assert 0.0 <= diag_result.sensitivity <= 1.0

    def test_specificity_in_range(self, diag_result):
        assert 0.0 <= diag_result.specificity <= 1.0

    def test_ppv_in_range(self, diag_result):
        assert 0.0 <= diag_result.ppv <= 1.0

    def test_npv_in_range(self, diag_result):
        assert 0.0 <= diag_result.npv <= 1.0

    def test_lr_positive_non_negative(self, diag_result):
        assert diag_result.lr_positive >= 0.0

    def test_dor_non_negative(self, diag_result):
        assert diag_result.dor >= 0.0

    def test_ci_sensitivity_ordered(self, diag_result):
        lo, hi = diag_result.sensitivity_ci
        assert lo <= diag_result.sensitivity <= hi

    def test_ci_specificity_ordered(self, diag_result):
        lo, hi = diag_result.specificity_ci
        assert lo <= diag_result.specificity <= hi

    def test_frozen(self, diag_result):
        with pytest.raises(AttributeError):
            diag_result.sensitivity = 0.5  # type: ignore[misc]

    def test_summary_contains_sensitivity(self, diag_result):
        s = diag_result.summary()
        assert "Sensitivity" in s

    def test_summary_contains_specificity(self, diag_result):
        s = diag_result.summary()
        assert "Specificity" in s

    def test_summary_contains_ppv(self, diag_result):
        s = diag_result.summary()
        assert "PPV" in s

    def test_summary_contains_cutoff(self, diag_result):
        s = diag_result.summary()
        assert "Cutoff" in s

    def test_summary_contains_dor(self, diag_result):
        s = diag_result.summary()
        assert "DOR" in s

    def test_summary_is_string(self, diag_result):
        assert isinstance(diag_result.summary(), str)

    def test_summary_contains_prevalence(self, diag_result):
        s = diag_result.summary()
        assert "Prevalence" in s
