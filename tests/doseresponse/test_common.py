"""Tests for doseresponse/_common.py: CurveParams, DoseResponseResult, BatchDoseResponseResult."""

import numpy as np
import pytest

from pystatsbio.doseresponse import fit_drm, ll4
from pystatsbio.doseresponse._common import CurveParams

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_dose():
    return np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], dtype=float)


@pytest.fixture
def fitted_ll4(standard_dose):
    np.random.seed(0)
    noise = np.random.normal(0, 1.0, len(standard_dose))
    response = ll4(standard_dose, 5.0, 95.0, 1.0, 1.5) + noise
    return fit_drm(standard_dose, response, model="LL.4")


# ---------------------------------------------------------------------------
# CurveParams
# ---------------------------------------------------------------------------

class TestCurveParams:
    """CurveParams construction, predict, to_array, from_array."""

    def test_construct_ll4(self):
        p = CurveParams(bottom=5.0, top=95.0, ec50=1.0, hill=1.5, model="LL.4")
        assert p.bottom == 5.0
        assert p.ec50 == 1.0

    def test_frozen(self):
        p = CurveParams(bottom=5.0, top=95.0, ec50=1.0, hill=1.5, model="LL.4")
        with pytest.raises(AttributeError):
            p.ec50 = 2.0  # type: ignore[misc]

    def test_predict_midpoint_is_ec50(self):
        """At dose == EC50, response should equal (bottom+top)/2 for symmetric LL.4."""
        p = CurveParams(bottom=0.0, top=100.0, ec50=1.0, hill=1.0, model="LL.4")
        pred = p.predict(np.array([1.0]))
        assert pred[0] == pytest.approx(50.0, abs=0.1)

    def test_predict_monotone(self):
        """Prediction should be monotone increasing with dose for standard 4PL."""
        p = CurveParams(bottom=5.0, top=95.0, ec50=1.0, hill=1.5, model="LL.4")
        doses = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
        preds = p.predict(doses)
        assert np.all(np.diff(preds) > 0)

    def test_predict_approaches_bottom(self):
        p = CurveParams(bottom=5.0, top=95.0, ec50=1.0, hill=2.0, model="LL.4")
        pred = p.predict(np.array([1e-6]))
        assert pred[0] == pytest.approx(5.0, abs=0.1)

    def test_predict_approaches_top(self):
        p = CurveParams(bottom=5.0, top=95.0, ec50=1.0, hill=2.0, model="LL.4")
        pred = p.predict(np.array([1e6]))
        assert pred[0] == pytest.approx(95.0, abs=0.1)

    def test_to_array_ll4_length(self):
        p = CurveParams(bottom=5.0, top=95.0, ec50=1.0, hill=1.5, model="LL.4")
        arr = p.to_array()
        assert len(arr) == 4

    def test_to_array_ll5_length(self):
        p = CurveParams(bottom=5.0, top=95.0, ec50=1.0, hill=1.5, asymmetry=1.0, model="LL.5")
        arr = p.to_array()
        assert len(arr) == 5

    def test_to_array_order(self):
        """to_array should match model's param_names order."""
        p = CurveParams(bottom=5.0, top=95.0, ec50=1.0, hill=1.5, model="LL.4")
        arr = p.to_array()
        # LL.4 param order: bottom, top, ec50, hill
        assert arr[0] == pytest.approx(5.0)
        assert arr[1] == pytest.approx(95.0)
        assert arr[2] == pytest.approx(1.0)
        assert arr[3] == pytest.approx(1.5)

    def test_from_array_roundtrip(self):
        p = CurveParams(bottom=5.0, top=95.0, ec50=1.0, hill=1.5, model="LL.4")
        arr = p.to_array()
        p2 = CurveParams.from_array(arr, "LL.4")
        assert p2.bottom == pytest.approx(p.bottom)
        assert p2.top == pytest.approx(p.top)
        assert p2.ec50 == pytest.approx(p.ec50)
        assert p2.hill == pytest.approx(p.hill)

    def test_from_array_ll5_roundtrip(self):
        p = CurveParams(bottom=5.0, top=95.0, ec50=1.0, hill=1.5, asymmetry=1.2, model="LL.5")
        arr = p.to_array()
        p2 = CurveParams.from_array(arr, "LL.5")
        assert p2.asymmetry == pytest.approx(1.2)

    def test_from_array_preserves_model(self):
        arr = np.array([5.0, 95.0, 1.0, 1.5])
        p = CurveParams.from_array(arr, "LL.4")
        assert p.model == "LL.4"


# ---------------------------------------------------------------------------
# DoseResponseResult (via fit_drm)
# ---------------------------------------------------------------------------

class TestDoseResponseResult:
    """DoseResponseResult fields and summary()."""

    def test_params_type(self, fitted_ll4):
        assert isinstance(fitted_ll4.params, CurveParams)

    def test_se_length_matches_params(self, fitted_ll4):
        assert len(fitted_ll4.se) == len(fitted_ll4.params.to_array())

    def test_se_positive(self, fitted_ll4):
        assert np.all(fitted_ll4.se > 0)

    def test_rss_non_negative(self, fitted_ll4):
        assert fitted_ll4.rss >= 0.0

    def test_aic_finite(self, fitted_ll4):
        assert np.isfinite(fitted_ll4.aic)

    def test_bic_finite(self, fitted_ll4):
        assert np.isfinite(fitted_ll4.bic)

    def test_converged(self, fitted_ll4):
        assert fitted_ll4.converged is True

    def test_model_name(self, fitted_ll4):
        assert fitted_ll4.model == "LL.4"

    def test_n_obs(self, fitted_ll4, standard_dose):
        assert fitted_ll4.n_obs == len(standard_dose)

    def test_predict_default_dose(self, fitted_ll4):
        pred = fitted_ll4.predict()
        assert len(pred) == fitted_ll4.n_obs

    def test_predict_custom_dose(self, fitted_ll4):
        new_dose = np.array([0.5, 5.0])
        pred = fitted_ll4.predict(new_dose)
        assert len(pred) == 2

    def test_summary_contains_model(self, fitted_ll4):
        s = fitted_ll4.summary()
        assert "LL.4" in s

    def test_summary_contains_rss(self, fitted_ll4):
        s = fitted_ll4.summary()
        assert "RSS" in s

    def test_summary_contains_aic(self, fitted_ll4):
        s = fitted_ll4.summary()
        assert "AIC" in s

    def test_summary_contains_param_names(self, fitted_ll4):
        s = fitted_ll4.summary()
        assert "ec50" in s
        assert "hill" in s
