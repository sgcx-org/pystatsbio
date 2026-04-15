"""Tests for EC50 estimation and relative potency analysis."""

import numpy as np
import pytest

from pystatsbio.doseresponse import ec50, fit_drm, ll4, relative_potency

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_ll4():
    np.random.seed(42)
    dose = np.array([0, 0.001, 0.01, 0.1, 1, 10, 100, 1000], dtype=float)
    response = ll4(dose, 10, 90, 1.0, 1.5) + np.random.normal(0, 1.5, len(dose))
    return fit_drm(dose, response, model="LL.4")


@pytest.fixture
def two_fitted_curves():
    """Two LL.4 curves with EC50 of 1.0 and 5.0."""
    np.random.seed(42)
    dose = np.array([0, 0.01, 0.1, 1, 10, 100, 1000], dtype=float)
    r1 = ll4(dose, 10, 90, 1.0, 1.5) + np.random.normal(0, 1.5, len(dose))
    r2 = ll4(dose, 10, 90, 5.0, 1.5) + np.random.normal(0, 1.5, len(dose))
    return fit_drm(dose, r1), fit_drm(dose, r2)


# ---------------------------------------------------------------------------
# EC50
# ---------------------------------------------------------------------------

class TestEC50:
    """EC50 extraction with delta method CI."""

    def test_ec50_positive(self, fitted_ll4):
        r = ec50(fitted_ll4)
        assert r.estimate > 0

    def test_ci_contains_estimate(self, fitted_ll4):
        r = ec50(fitted_ll4)
        assert r.ci_lower < r.estimate < r.ci_upper

    def test_ci_contains_true_value(self, fitted_ll4):
        r = ec50(fitted_ll4)
        # True EC50 = 1.0; CI should be in the right ballpark
        assert r.ci_lower < 1.5
        assert r.ci_upper > 0.5

    def test_se_positive(self, fitted_ll4):
        r = ec50(fitted_ll4)
        assert r.se > 0

    def test_conf_level_stored(self, fitted_ll4):
        r = ec50(fitted_ll4, conf_level=0.95)
        assert r.conf_level == 0.95

    def test_wider_ci_at_99(self, fitted_ll4):
        r95 = ec50(fitted_ll4, conf_level=0.95)
        r99 = ec50(fitted_ll4, conf_level=0.99)
        assert (r99.ci_upper - r99.ci_lower) > (r95.ci_upper - r95.ci_lower)

    def test_method_stored(self, fitted_ll4):
        r = ec50(fitted_ll4, method="delta")
        assert r.method == "delta"

    def test_invalid_conf_level(self, fitted_ll4):
        with pytest.raises(ValueError, match="conf_level"):
            ec50(fitted_ll4, conf_level=1.5)

    def test_invalid_method(self, fitted_ll4):
        with pytest.raises(ValueError, match="method"):
            ec50(fitted_ll4, method="profile")


# ---------------------------------------------------------------------------
# Relative potency
# ---------------------------------------------------------------------------

class TestRelativePotency:
    """Relative potency with Fieller's CI."""

    def test_ratio_positive(self, two_fitted_curves):
        fit1, fit2 = two_fitted_curves
        r = relative_potency(fit1, fit2)
        assert r.ratio > 0

    def test_ratio_approximately_5(self, two_fitted_curves):
        """EC50_2 / EC50_1 ≈ 5.0 / 1.0 = 5."""
        fit1, fit2 = two_fitted_curves
        r = relative_potency(fit1, fit2)
        assert r.ratio == pytest.approx(5.0, rel=0.5)

    def test_ci_contains_ratio(self, two_fitted_curves):
        fit1, fit2 = two_fitted_curves
        r = relative_potency(fit1, fit2)
        assert r.ci_lower < r.ratio < r.ci_upper

    def test_method_is_fieller(self, two_fitted_curves):
        fit1, fit2 = two_fitted_curves
        r = relative_potency(fit1, fit2)
        assert r.method == "fieller"

    def test_conf_level_stored(self, two_fitted_curves):
        fit1, fit2 = two_fitted_curves
        r = relative_potency(fit1, fit2, conf_level=0.99)
        assert r.conf_level == 0.99

    def test_invalid_conf_level(self, two_fitted_curves):
        fit1, fit2 = two_fitted_curves
        with pytest.raises(ValueError, match="conf_level"):
            relative_potency(fit1, fit2, conf_level=2.0)

    def test_inverse_ratio(self, two_fitted_curves):
        """relative_potency(fit2, fit1) should be ≈ 1/5."""
        fit1, fit2 = two_fitted_curves
        r_fwd = relative_potency(fit1, fit2)
        r_rev = relative_potency(fit2, fit1)
        assert r_rev.ratio == pytest.approx(1.0 / r_fwd.ratio, rel=0.1)
