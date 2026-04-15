"""Tests for fit_drm (single curve fitting)."""

import numpy as np
import pytest

from pystatsbio.doseresponse import fit_drm, ll4, weibull1, weibull2

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ll4_data():
    """Clean LL.4 data with known parameters + small noise."""
    np.random.seed(42)
    dose = np.array([0, 0.001, 0.01, 0.1, 1, 10, 100, 1000], dtype=float)
    true = dict(bottom=10.0, top=90.0, ec50=1.0, hill=1.5)
    response = ll4(dose, **true) + np.random.normal(0, 2, len(dose))
    return dose, response, true


@pytest.fixture
def inhibitor_data():
    """Decreasing (inhibitor) LL.4 curve."""
    np.random.seed(123)
    dose = np.array([0, 0.01, 0.1, 1, 10, 100, 1000], dtype=float)
    true = dict(bottom=5.0, top=95.0, ec50=10.0, hill=-2.0)
    response = ll4(dose, **true) + np.random.normal(0, 3, len(dose))
    return dose, response, true


# ---------------------------------------------------------------------------
# LL.4 fitting
# ---------------------------------------------------------------------------

class TestFitDrmLL4:
    """Fitting the 4-parameter log-logistic model."""

    def test_recovers_ec50(self, ll4_data):
        """EC50 should be recovered within 50% of true value."""
        dose, response, true = ll4_data
        r = fit_drm(dose, response, model="LL.4")
        assert r.params.ec50 == pytest.approx(true["ec50"], rel=0.5)
        assert r.converged

    def test_recovers_bottom_top(self, ll4_data):
        """Bottom and top should be recovered within 10 units."""
        dose, response, true = ll4_data
        r = fit_drm(dose, response, model="LL.4")
        assert r.params.bottom == pytest.approx(true["bottom"], abs=10)
        assert r.params.top == pytest.approx(true["top"], abs=10)

    def test_converges(self, ll4_data):
        dose, response, _ = ll4_data
        r = fit_drm(dose, response)
        assert r.converged

    def test_residuals_shape(self, ll4_data):
        dose, response, _ = ll4_data
        r = fit_drm(dose, response)
        assert r.residuals.shape == (len(dose),)

    def test_se_positive(self, ll4_data):
        dose, response, _ = ll4_data
        r = fit_drm(dose, response)
        assert np.all(r.se > 0)

    def test_rss_positive(self, ll4_data):
        dose, response, _ = ll4_data
        r = fit_drm(dose, response)
        assert r.rss > 0

    def test_predict_matches_data(self, ll4_data):
        """Predicted values should be close to observed."""
        dose, response, _ = ll4_data
        r = fit_drm(dose, response)
        pred = r.predict()
        # Residuals should be small (noise was sd=2)
        assert np.max(np.abs(pred - response)) < 10

    def test_predict_at_new_doses(self, ll4_data):
        dose, response, _ = ll4_data
        r = fit_drm(dose, response)
        new_dose = np.array([0.5, 5.0, 50.0])
        pred = r.predict(new_dose)
        assert pred.shape == (3,)

    def test_dose_zero_handled(self):
        """Fitting with dose=0 should not raise."""
        dose = np.array([0, 0.1, 1, 10, 100])
        response = np.array([10, 20, 50, 80, 90])
        r = fit_drm(dose, response)
        assert r.converged

    def test_inhibitor_curve(self, inhibitor_data):
        """Decreasing dose-response should be fitted correctly."""
        dose, response, true = inhibitor_data
        r = fit_drm(dose, response, model="LL.4")
        assert r.converged
        # Hill should be negative
        assert r.params.hill < 0


class TestFitDrmSummary:
    """Summary output."""

    def test_summary_contains_fields(self, ll4_data):
        dose, response, _ = ll4_data
        r = fit_drm(dose, response)
        s = r.summary()
        assert "LL.4" in s
        assert "ec50" in s
        assert "RSS" in s
        assert "AIC" in s
        assert "Converged" in s

    def test_curve_params_summary(self, ll4_data):
        dose, response, _ = ll4_data
        r = fit_drm(dose, response)
        # CurveParams round-trip via to_array / from_array
        arr = r.params.to_array()
        assert len(arr) == 4
        cp2 = type(r.params).from_array(arr, "LL.4")
        assert cp2.ec50 == pytest.approx(r.params.ec50)


class TestFitDrmOtherModels:
    """Fitting other model types."""

    def test_ll5_converges(self):
        """LL.5 should converge on symmetric data (asymmetry ≈ 1)."""
        np.random.seed(42)
        dose = np.array([0.01, 0.1, 1, 10, 100, 1000], dtype=float)
        response = ll4(dose, 10, 90, 5.0, 1.5) + np.random.normal(0, 2, len(dose))
        r = fit_drm(dose, response, model="LL.5")
        assert r.converged
        assert r.params.asymmetry is not None

    def test_weibull1_converges(self):
        np.random.seed(42)
        dose = np.array([0.01, 0.1, 1, 10, 100, 1000], dtype=float)
        response = weibull1(dose, 10, 90, 5.0, 1.5) + np.random.normal(0, 2, len(dose))
        r = fit_drm(dose, response, model="W1.4")
        assert r.converged

    def test_weibull2_converges(self):
        np.random.seed(42)
        dose = np.array([0.01, 0.1, 1, 10, 100, 1000], dtype=float)
        response = weibull2(dose, 10, 90, 5.0, 1.5) + np.random.normal(0, 2, len(dose))
        r = fit_drm(dose, response, model="W2.4")
        assert r.converged


class TestFitDrmWithStartValues:
    """User-supplied starting values."""

    def test_custom_start(self, ll4_data):
        dose, response, true = ll4_data
        start = {"bottom": 5.0, "top": 95.0, "ec50": 2.0, "hill": 1.0}
        r = fit_drm(dose, response, start=start)
        assert r.converged

    def test_with_bounds(self, ll4_data):
        dose, response, _ = ll4_data
        r = fit_drm(dose, response, lower={"bottom": 0}, upper={"top": 100})
        assert r.converged
        assert r.params.bottom >= 0
        assert r.params.top <= 100


class TestFitDrmValidation:
    """Input validation."""

    def test_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            fit_drm(np.array([1, 2]), np.array([1, 2, 3]))

    def test_invalid_model(self):
        with pytest.raises(ValueError, match="model"):
            fit_drm(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5]), model="invalid")

    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="at least"):
            fit_drm(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), model="LL.4")

    def test_not_1d(self):
        with pytest.raises(ValueError, match="1-D"):
            fit_drm(np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]]))
