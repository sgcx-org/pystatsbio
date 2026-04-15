"""Tests for pk/_common.py: NCAResult fields and summary()."""

import numpy as np
import pytest

from pystatsbio.pk import nca

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nca_iv():
    """IV bolus profile with clear terminal phase."""
    time = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])
    conc = np.array([100.0, 75.0, 58.0, 34.0, 12.0, 4.5, 1.8, 0.3, 0.05])
    return nca(time, conc, dose=100.0, route="iv")


@pytest.fixture
def nca_ev():
    """Extravascular (oral) profile with absorption then elimination."""
    time = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])
    conc = np.array([0.0, 8.5, 12.1, 10.4, 7.2, 4.9, 3.1, 1.4, 0.3])
    return nca(time, conc, dose=100.0, route="ev")


@pytest.fixture
def nca_no_dose():
    """NCA without dose — CL and Vz should be None."""
    time = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    conc = np.array([50.0, 35.0, 24.0, 12.0, 3.0, 0.8, 0.1])
    return nca(time, conc)


# ---------------------------------------------------------------------------
# NCAResult fields
# ---------------------------------------------------------------------------

class TestNCAResultFields:
    """All fields of NCAResult have sensible values."""

    def test_cmax_positive(self, nca_iv):
        assert nca_iv.cmax > 0.0

    def test_tmax_non_negative(self, nca_iv):
        assert nca_iv.tmax >= 0.0

    def test_auc_last_positive(self, nca_iv):
        assert nca_iv.auc_last > 0.0

    def test_auc_inf_greater_than_last(self, nca_iv):
        assert nca_iv.auc_inf is not None
        assert nca_iv.auc_inf > nca_iv.auc_last

    def test_pct_extrap_between_0_and_100(self, nca_iv):
        assert nca_iv.auc_pct_extrap is not None
        assert 0.0 <= nca_iv.auc_pct_extrap <= 100.0

    def test_half_life_positive(self, nca_iv):
        assert nca_iv.half_life is not None
        assert nca_iv.half_life > 0.0

    def test_lambda_z_positive(self, nca_iv):
        assert nca_iv.lambda_z is not None
        assert nca_iv.lambda_z > 0.0

    def test_half_life_lambda_z_consistent(self, nca_iv):
        """t1/2 = ln(2) / lambda_z."""
        import math
        assert nca_iv.half_life == pytest.approx(math.log(2) / nca_iv.lambda_z, rel=1e-6)

    def test_clearance_positive(self, nca_iv):
        assert nca_iv.clearance is not None
        assert nca_iv.clearance > 0.0

    def test_vz_positive(self, nca_iv):
        assert nca_iv.vz is not None
        assert nca_iv.vz > 0.0

    def test_no_dose_cl_none(self, nca_no_dose):
        assert nca_no_dose.clearance is None

    def test_no_dose_vz_none(self, nca_no_dose):
        assert nca_no_dose.vz is None

    def test_route_stored(self, nca_iv):
        assert nca_iv.route == "iv"

    def test_route_ev(self, nca_ev):
        assert nca_ev.route == "ev"

    def test_auc_method_stored(self, nca_iv):
        assert nca_iv.auc_method == "linear-up/log-down"

    def test_n_terminal_positive(self, nca_iv):
        assert nca_iv.n_terminal >= 3

    def test_frozen(self, nca_iv):
        with pytest.raises(AttributeError):
            nca_iv.cmax = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NCAResult.summary()
# ---------------------------------------------------------------------------

class TestNCAResultSummary:
    """NCAResult.summary() output."""

    def test_summary_is_string(self, nca_iv):
        assert isinstance(nca_iv.summary(), str)

    def test_summary_contains_cmax(self, nca_iv):
        assert "Cmax" in nca_iv.summary()

    def test_summary_contains_tmax(self, nca_iv):
        assert "Tmax" in nca_iv.summary()

    def test_summary_contains_auc_last(self, nca_iv):
        assert "AUC(0-last)" in nca_iv.summary()

    def test_summary_contains_auc_inf(self, nca_iv):
        assert "AUC(0-inf)" in nca_iv.summary()

    def test_summary_contains_half_life(self, nca_iv):
        assert "t1/2" in nca_iv.summary()

    def test_summary_contains_clearance_iv(self, nca_iv):
        s = nca_iv.summary()
        assert "CL" in s

    def test_summary_contains_clearance_ev(self, nca_ev):
        s = nca_ev.summary()
        assert "CL/F" in s

    def test_summary_contains_route(self, nca_iv):
        assert "IV" in nca_iv.summary()

    def test_summary_no_dose_omits_cl(self, nca_no_dose):
        s = nca_no_dose.summary()
        assert "CL" not in s

    def test_summary_contains_dose(self, nca_iv):
        s = nca_iv.summary()
        assert "Dose" in s
