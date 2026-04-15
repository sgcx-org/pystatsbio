"""Tests for benchmark dose (BMD) analysis."""

import numpy as np
import pytest

from pystatsbio.doseresponse import bmd, fit_drm, ll4

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_ll4():
    np.random.seed(42)
    dose = np.array([0, 0.001, 0.01, 0.1, 1, 10, 100, 1000], dtype=float)
    response = ll4(dose, 10, 90, 1.0, 1.5) + np.random.normal(0, 1.5, len(dose))
    return fit_drm(dose, response, model="LL.4")


# ---------------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------------

class TestBMDNormal:
    """BMD point estimate and CI for well-behaved fits."""

    def test_bmd_positive(self, fitted_ll4):
        r = bmd(fitted_ll4, bmr=0.10)
        assert r.bmd > 0

    def test_bmdl_less_than_bmd(self, fitted_ll4):
        r = bmd(fitted_ll4, bmr=0.10)
        assert r.bmdl < r.bmd

    def test_bmdu_greater_than_bmd(self, fitted_ll4):
        r = bmd(fitted_ll4, bmr=0.10)
        assert r.bmdu > r.bmd

    def test_bmd_10_less_than_ec50(self, fitted_ll4):
        """BMD10 (10% response) should be less than EC50 (50% response)."""
        r = bmd(fitted_ll4, bmr=0.10)
        assert r.bmd < fitted_ll4.params.ec50

    def test_higher_bmr_higher_bmd(self, fitted_ll4):
        """Higher BMR requires a higher dose to achieve."""
        r10 = bmd(fitted_ll4, bmr=0.10)
        r25 = bmd(fitted_ll4, bmr=0.25)
        assert r25.bmd > r10.bmd

    def test_additional_risk_type(self, fitted_ll4):
        r = bmd(fitted_ll4, bmr=0.10, bmr_type="additional")
        assert r.bmd > 0

    def test_bmr_stored(self, fitted_ll4):
        r = bmd(fitted_ll4, bmr=0.15)
        assert r.bmr == pytest.approx(0.15)

    def test_conf_level_stored(self, fitted_ll4):
        r = bmd(fitted_ll4, bmr=0.10, conf_level=0.90)
        assert r.conf_level == pytest.approx(0.90)

    def test_wider_ci_at_higher_conf(self, fitted_ll4):
        r90 = bmd(fitted_ll4, bmr=0.10, conf_level=0.90)
        r99 = bmd(fitted_ll4, bmr=0.10, conf_level=0.99)
        width90 = r90.bmdu - r90.bmdl
        width99 = r99.bmdu - r99.bmdl
        assert width99 > width90

    def test_method_stored(self, fitted_ll4):
        r = bmd(fitted_ll4, bmr=0.10)
        assert r.method == "delta"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestBMDEdgeCases:
    """BMD boundary and near-boundary inputs."""

    def test_bmr_just_above_zero(self, fitted_ll4):
        """bmr very close to 0 should still produce a valid result."""
        r = bmd(fitted_ll4, bmr=0.01)
        assert r.bmd > 0

    def test_bmr_just_below_one(self, fitted_ll4):
        """bmr close to 1.0 should produce a valid result."""
        r = bmd(fitted_ll4, bmr=0.49)
        assert r.bmd > 0


# ---------------------------------------------------------------------------
# Failure cases — invalid inputs must raise
# ---------------------------------------------------------------------------

class TestBMDValidation:
    """bmd() raises ValueError on invalid inputs (Rule 1: fail fast)."""

    def test_bmr_zero_raises(self, fitted_ll4):
        with pytest.raises(ValueError, match="bmr"):
            bmd(fitted_ll4, bmr=0.0)

    def test_bmr_one_raises(self, fitted_ll4):
        with pytest.raises(ValueError, match="bmr"):
            bmd(fitted_ll4, bmr=1.0)

    def test_bmr_negative_raises(self, fitted_ll4):
        with pytest.raises(ValueError, match="bmr"):
            bmd(fitted_ll4, bmr=-0.1)

    def test_bmr_above_one_raises(self, fitted_ll4):
        with pytest.raises(ValueError, match="bmr"):
            bmd(fitted_ll4, bmr=1.5)

    def test_bmr_nan_raises(self, fitted_ll4):
        with pytest.raises(ValueError, match="bmr"):
            bmd(fitted_ll4, bmr=float("nan"))

    def test_bmr_inf_raises(self, fitted_ll4):
        with pytest.raises(ValueError, match="bmr"):
            bmd(fitted_ll4, bmr=float("inf"))

    def test_invalid_bmr_type_raises(self, fitted_ll4):
        with pytest.raises(ValueError, match="bmr_type"):
            bmd(fitted_ll4, bmr_type="invalid")

    def test_invalid_method_raises(self, fitted_ll4):
        with pytest.raises(ValueError, match="method"):
            bmd(fitted_ll4, method="likelihood")

    def test_invalid_conf_level_raises(self, fitted_ll4):
        with pytest.raises(ValueError, match="conf_level"):
            bmd(fitted_ll4, conf_level=1.5)
