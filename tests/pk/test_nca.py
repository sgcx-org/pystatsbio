"""Tests for non-compartmental pharmacokinetic analysis (NCA)."""

import numpy as np
import pytest

from pystatsbio.pk import NCAResult, nca

# ---------------------------------------------------------------------------
# Fixtures — realistic PK profiles
# ---------------------------------------------------------------------------

@pytest.fixture
def oral_pk():
    """Typical oral PK profile (extravascular).

    Simulated one-compartment oral: ka=1.0, ke=0.1, V=10, F=1, Dose=100.
    C(t) = (F*Dose*ka / (V*(ka-ke))) * (exp(-ke*t) - exp(-ka*t))
    """
    time = np.array([0, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 16, 24])
    ka, ke, V, F, D = 1.0, 0.1, 10.0, 1.0, 100.0
    A = F * D * ka / (V * (ka - ke))
    concentration = A * (np.exp(-ke * time) - np.exp(-ka * time))
    concentration[concentration < 0] = 0  # numerical safety
    return time, concentration


@pytest.fixture
def iv_bolus_pk():
    """IV bolus PK profile.

    One-compartment IV: C(t) = (Dose/V) * exp(-ke*t)
    ke=0.1, V=10, Dose=100 → C0=10.
    """
    time = np.array([0, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 16, 24])
    ke, V, D = 0.1, 10.0, 100.0
    concentration = (D / V) * np.exp(-ke * time)
    return time, concentration


@pytest.fixture
def simple_triangle():
    """Simple triangle PK profile for hand-calculable AUC.

    Time: [0, 1, 2, 3]
    Conc: [0, 10, 5, 0]

    NCA convention: AUC_last goes to last measurable (non-zero) concentration.
    Last measurable is at t=2, conc=5.
    Linear AUC(0→2) = 0.5*(0+10)*1 + 0.5*(10+5)*1 = 5 + 7.5 = 12.5
    """
    time = np.array([0.0, 1.0, 2.0, 3.0])
    concentration = np.array([0.0, 10.0, 5.0, 0.0])
    return time, concentration


# ---------------------------------------------------------------------------
# Basic NCA functionality
# ---------------------------------------------------------------------------

class TestNCABasic:
    """Basic NCA output."""

    def test_returns_nca_result(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert isinstance(r, NCAResult)

    def test_cmax_positive(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.cmax > 0

    def test_cmax_equals_max(self, oral_pk):
        """Cmax should equal the maximum concentration."""
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.cmax == pytest.approx(np.max(conc), abs=1e-10)

    def test_tmax_at_peak(self, oral_pk):
        """Tmax should correspond to Cmax."""
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        idx_max = np.argmax(conc)
        assert r.tmax == pytest.approx(time[idx_max], abs=1e-10)

    def test_auc_last_positive(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.auc_last > 0

    def test_n_points(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.n_points == len(time)

    def test_route_stored(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.route == "ev"

    def test_dose_stored(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.dose == 100


# ---------------------------------------------------------------------------
# AUC methods
# ---------------------------------------------------------------------------

class TestAUCMethods:
    """AUC calculation methods."""

    def test_linear_auc_triangle(self, simple_triangle):
        """Hand-calculated linear AUC for triangle profile.

        AUC_last goes to last measurable (non-zero) concentration at t=2.
        AUC(0→2) = 0.5*(0+10)*1 + 0.5*(10+5)*1 = 12.5
        """
        time, conc = simple_triangle
        r = nca(time, conc, auc_method="linear")
        assert r.auc_last == pytest.approx(12.5, abs=1e-10)

    def test_linear_auc_positive(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, auc_method="linear")
        assert r.auc_last > 0

    def test_loglinear_auc_positive(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, auc_method="log-linear")
        assert r.auc_last > 0

    def test_luldown_auc_positive(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, auc_method="linear-up/log-down")
        assert r.auc_last > 0

    def test_luldown_is_default(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc)
        assert r.auc_method == "linear-up/log-down"

    def test_methods_give_similar_auc(self, oral_pk):
        """All three methods should give similar AUC for smooth profiles."""
        time, conc = oral_pk
        r_lin = nca(time, conc, auc_method="linear")
        r_log = nca(time, conc, auc_method="log-linear")
        r_lul = nca(time, conc, auc_method="linear-up/log-down")
        # Within 10% of each other for this smooth profile
        mean_auc = np.mean([r_lin.auc_last, r_log.auc_last, r_lul.auc_last])
        assert abs(r_lin.auc_last - mean_auc) / mean_auc < 0.10
        assert abs(r_log.auc_last - mean_auc) / mean_auc < 0.10
        assert abs(r_lul.auc_last - mean_auc) / mean_auc < 0.10

    def test_linear_auc_constant_concentration(self):
        """Constant concentration: AUC = C * T."""
        time = np.array([0, 1, 2, 3, 4])
        conc = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        r = nca(time, conc, auc_method="linear")
        assert r.auc_last == pytest.approx(20.0, abs=1e-10)

    def test_loglinear_constant_falls_back_to_linear(self):
        """Log-linear method with constant conc should fall back to linear."""
        time = np.array([0, 1, 2, 3, 4])
        conc = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        r = nca(time, conc, auc_method="log-linear")
        assert r.auc_last == pytest.approx(20.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Terminal elimination / half-life
# ---------------------------------------------------------------------------

class TestTerminalElimination:
    """Lambda_z and half-life estimation."""

    def test_lambda_z_estimated(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.lambda_z is not None
        assert r.lambda_z > 0

    def test_half_life_from_lambda_z(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.half_life is not None
        assert r.half_life == pytest.approx(np.log(2) / r.lambda_z, abs=1e-10)

    def test_lambda_z_close_to_true(self, oral_pk):
        """For this simulated profile, lambda_z ≈ ke = 0.1."""
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        # Should be close to 0.1 (within 15% — NCA estimate from finite samples)
        assert abs(r.lambda_z - 0.1) / 0.1 < 0.15

    def test_half_life_close_to_true(self, oral_pk):
        """t1/2 ≈ ln(2)/0.1 ≈ 6.93."""
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        true_half_life = np.log(2) / 0.1
        assert abs(r.half_life - true_half_life) / true_half_life < 0.15

    def test_r_squared_high(self, oral_pk):
        """Terminal phase should have good fit (R² > 0.95)."""
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.lambda_z_r_squared is not None
        assert r.lambda_z_r_squared > 0.95

    def test_n_terminal_at_least_3(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.n_terminal >= 3

    def test_iv_lambda_z(self, iv_bolus_pk):
        """IV bolus with pure monoexponential should give exact ke."""
        time, conc = iv_bolus_pk
        r = nca(time, conc, dose=100, route="iv")
        assert r.lambda_z is not None
        # Pure monoexponential — should be very accurate
        assert r.lambda_z == pytest.approx(0.1, rel=0.05)

    def test_fixed_n_points(self, oral_pk):
        """Manual specification of terminal points."""
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev", lambda_z_n_points=3)
        assert r.n_terminal == 3

    def test_triangle_no_lambda_z(self, simple_triangle):
        """Triangle profile with zero at end — lambda_z may not be estimable."""
        time, conc = simple_triangle
        r = nca(time, conc, auc_method="linear")
        # Only 1 point after Cmax with positive conc, not enough for regression
        # lambda_z could be None
        # (This tests graceful degradation)
        assert r.auc_last == pytest.approx(12.5, abs=1e-10)


# ---------------------------------------------------------------------------
# AUC extrapolation to infinity
# ---------------------------------------------------------------------------

class TestAUCInfinity:
    """AUC extrapolated to infinity."""

    def test_auc_inf_gt_auc_last(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.auc_inf is not None
        assert r.auc_inf > r.auc_last

    def test_pct_extrap_small(self, oral_pk):
        """For this profile sampled out to 24h (>3 half-lives), extrapolation should be small."""
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.auc_pct_extrap is not None
        assert r.auc_pct_extrap < 20  # Less than 20% extrapolated

    def test_pct_extrap_bounded(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.auc_pct_extrap >= 0
        assert r.auc_pct_extrap <= 100

    def test_iv_auc_inf_close_to_true(self, iv_bolus_pk):
        """IV bolus AUC_inf ≈ Dose / (ke * V) = 100 / (0.1 * 10) = 100."""
        time, conc = iv_bolus_pk
        r = nca(time, conc, dose=100, route="iv")
        true_auc = 100.0 / (0.1 * 10.0)
        assert r.auc_inf is not None
        assert r.auc_inf == pytest.approx(true_auc, rel=0.05)


# ---------------------------------------------------------------------------
# Clearance and volume of distribution
# ---------------------------------------------------------------------------

class TestDerivedParameters:
    """Dose-dependent parameters: CL and Vz."""

    def test_clearance_with_dose(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.clearance is not None
        assert r.clearance > 0

    def test_vz_with_dose(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.vz is not None
        assert r.vz > 0

    def test_no_dose_no_clearance(self, oral_pk):
        """Without dose, CL and Vz should be None."""
        time, conc = oral_pk
        r = nca(time, conc)
        assert r.clearance is None
        assert r.vz is None

    def test_clearance_formula(self, oral_pk):
        """CL = Dose / AUC_inf."""
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.clearance == pytest.approx(100.0 / r.auc_inf, rel=1e-10)

    def test_vz_formula(self, oral_pk):
        """Vz = Dose / (lambda_z * AUC_inf)."""
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.vz == pytest.approx(
            100.0 / (r.lambda_z * r.auc_inf), rel=1e-10
        )

    def test_iv_clearance_close_to_true(self, iv_bolus_pk):
        """IV bolus CL ≈ ke * V = 0.1 * 10 = 1.0."""
        time, conc = iv_bolus_pk
        r = nca(time, conc, dose=100, route="iv")
        assert r.clearance is not None
        assert r.clearance == pytest.approx(1.0, rel=0.05)

    def test_iv_vz_close_to_true(self, iv_bolus_pk):
        """IV bolus Vz ≈ V = 10."""
        time, conc = iv_bolus_pk
        r = nca(time, conc, dose=100, route="iv")
        assert r.vz is not None
        assert r.vz == pytest.approx(10.0, rel=0.05)


# ---------------------------------------------------------------------------
# IV vs EV route
# ---------------------------------------------------------------------------

class TestRoutes:
    """Route-specific behavior."""

    def test_iv_route(self, iv_bolus_pk):
        time, conc = iv_bolus_pk
        r = nca(time, conc, dose=100, route="iv")
        assert r.route == "iv"

    def test_ev_route(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.route == "ev"

    def test_iv_cmax_at_time_zero(self, iv_bolus_pk):
        """For IV bolus, Cmax should be at or near t=0."""
        time, conc = iv_bolus_pk
        r = nca(time, conc, dose=100, route="iv")
        assert r.tmax == pytest.approx(0.0, abs=1e-10)

    def test_ev_cmax_not_at_zero(self, oral_pk):
        """For oral dosing, Tmax should be > 0 (absorption phase)."""
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        assert r.tmax > 0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    """Summary output."""

    def test_summary_contains_key_params(self, oral_pk):
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        s = r.summary()
        assert "Cmax" in s
        assert "Tmax" in s
        assert "AUC" in s
        assert "t1/2" in s

    def test_summary_ev_shows_clf(self, oral_pk):
        """EV route should show CL/F, not CL."""
        time, conc = oral_pk
        r = nca(time, conc, dose=100, route="ev")
        s = r.summary()
        assert "CL/F" in s

    def test_summary_iv_shows_cl(self, iv_bolus_pk):
        """IV route should show CL, not CL/F."""
        time, conc = iv_bolus_pk
        r = nca(time, conc, dose=100, route="iv")
        s = r.summary()
        assert "CL" in s
        # Check it's not CL/F
        lines = s.split("\n")
        cl_lines = [l for l in lines if "CL" in l and "CL/F" not in l]
        assert len(cl_lines) > 0

    def test_summary_without_dose(self, oral_pk):
        """Without dose, CL/Vz should not appear."""
        time, conc = oral_pk
        r = nca(time, conc)
        s = r.summary()
        assert "CL" not in s
        assert "Vz" not in s


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_all_zero_concentration(self):
        """All-zero concentrations should return AUC=0."""
        time = np.array([0, 1, 2, 3, 4])
        conc = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        r = nca(time, conc)
        assert r.auc_last == 0.0
        assert r.cmax == 0.0
        assert r.lambda_z is None
        assert r.half_life is None

    def test_single_peak(self):
        """Profile with concentration only at one point.

        AUC_last goes to last measurable (the peak itself at t=2).
        AUC(0→2) = 0.5*(0+10)*1 + 0.5*(0+10)*1 = 5 + 5... wait:
        conc = [0, 0, 10, 0, 0] → last measurable is at t=2 (index 2).
        AUC from t=0 to t=2: 0.5*(0+0)*1 + 0.5*(0+10)*1 = 0 + 5 = 5.
        """
        time = np.array([0, 1, 2, 3, 4])
        conc = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
        r = nca(time, conc, auc_method="linear")
        assert r.cmax == 10.0
        assert r.tmax == 2.0
        assert r.auc_last == pytest.approx(5.0, abs=1e-10)

    def test_monotone_decreasing(self):
        """Pure exponential decay (like IV bolus from t=0)."""
        time = np.array([0, 1, 2, 4, 8, 12, 24])
        conc = 10.0 * np.exp(-0.1 * time)
        r = nca(time, conc, dose=100, route="iv")
        assert r.cmax == pytest.approx(10.0, abs=1e-10)
        assert r.tmax == 0.0

    def test_trailing_zeros(self):
        """Profile with BLQ trailing zeros."""
        time = np.array([0, 1, 2, 4, 8, 12, 24, 36, 48])
        conc = np.array([0, 5, 10, 8, 4, 1, 0.1, 0, 0])
        r = nca(time, conc, auc_method="linear")
        # AUC should only go to last measurable (t=24)
        assert r.auc_last > 0
        assert r.n_points == 9

    def test_unsorted_input(self):
        """Input should be auto-sorted by time."""
        time = np.array([2, 0, 1, 4, 3])
        conc = np.array([10, 0, 5, 2, 5])
        r = nca(time, conc, auc_method="linear")
        # Should produce same result as sorted
        time_sorted = np.array([0, 1, 2, 3, 4])
        conc_sorted = np.array([0, 5, 10, 5, 2])
        r2 = nca(time_sorted, conc_sorted, auc_method="linear")
        assert r.auc_last == pytest.approx(r2.auc_last, abs=1e-10)
        assert r.cmax == pytest.approx(r2.cmax, abs=1e-10)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    """Input validation."""

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            nca(np.array([0, 1, 2]), np.array([1.0, 2.0]))

    def test_too_few_points(self):
        with pytest.raises(ValueError, match="at least 3"):
            nca(np.array([0, 1]), np.array([1.0, 2.0]))

    def test_negative_time(self):
        with pytest.raises(ValueError, match="non-negative"):
            nca(np.array([-1, 0, 1]), np.array([0, 5, 3]))

    def test_negative_concentration(self):
        with pytest.raises(ValueError, match="non-negative"):
            nca(np.array([0, 1, 2]), np.array([0, -1, 3]))

    def test_invalid_route(self):
        with pytest.raises(ValueError, match="route"):
            nca(np.array([0, 1, 2]), np.array([0, 5, 3]), route="oral")

    def test_invalid_auc_method(self):
        with pytest.raises(ValueError, match="auc_method"):
            nca(np.array([0, 1, 2]), np.array([0, 5, 3]), auc_method="other")

    def test_duplicate_times(self):
        with pytest.raises(ValueError, match="Duplicate"):
            nca(np.array([0, 1, 1, 2]), np.array([0, 5, 6, 3]))

    def test_lambda_z_n_points_too_few(self, oral_pk):
        time, conc = oral_pk
        with pytest.raises(ValueError, match="lambda_z_n_points"):
            nca(time, conc, lambda_z_n_points=2)

    def test_lambda_z_n_points_too_many(self):
        """More terminal points than available."""
        time = np.array([0, 1, 2, 3, 4])
        conc = np.array([0, 10, 5, 2, 1])
        with pytest.raises(ValueError, match="exceeds"):
            nca(time, conc, lambda_z_n_points=10)


# ---------------------------------------------------------------------------
# Cross-validation with analytical results
# ---------------------------------------------------------------------------

class TestAnalytical:
    """Cross-validation against analytically computed values."""

    def test_iv_bolus_analytical_auc(self):
        """IV bolus: AUC_inf = C0/ke = (Dose/V)/ke.

        Dose=100, V=10, ke=0.2 → C0=10, AUC=50.
        Sample densely for good trapezoidal approximation.
        """
        ke = 0.2
        V = 10.0
        D = 100.0
        time = np.linspace(0, 50, 201)  # Dense sampling, 5 half-lives
        conc = (D / V) * np.exp(-ke * time)
        r = nca(time, conc, dose=D, route="iv", auc_method="linear")
        true_auc = D / (V * ke)  # 50.0
        assert r.auc_inf is not None
        assert r.auc_inf == pytest.approx(true_auc, rel=0.02)

    def test_iv_bolus_analytical_halflife(self):
        """IV bolus: t1/2 = ln(2)/ke."""
        ke = 0.2
        V = 10.0
        D = 100.0
        time = np.linspace(0, 50, 201)
        conc = (D / V) * np.exp(-ke * time)
        r = nca(time, conc, dose=D, route="iv")
        true_t12 = np.log(2) / ke
        assert r.half_life is not None
        assert r.half_life == pytest.approx(true_t12, rel=0.02)

    def test_linear_auc_exact_for_linear_segments(self):
        """Linear trapezoidal should be exact for piecewise-linear profiles.

        conc = [0, 4, 8, 4, 0] — last measurable at index 3 (conc=4, t=5).
        AUC(0→5):
        [0,1]: 0.5*(0+4)*1 = 2
        [1,3]: 0.5*(4+8)*2 = 12
        [3,5]: 0.5*(8+4)*2 = 12
        Total = 26
        """
        time = np.array([0, 1, 3, 5, 10])
        conc = np.array([0, 4, 8, 4, 0])
        r = nca(time, conc, auc_method="linear")
        assert r.auc_last == pytest.approx(26.0, abs=1e-10)

    def test_loglinear_exact_for_exponential(self):
        """Log-linear trapezoidal should be exact for exponential decay.

        For C(t) = C0 * exp(-k*t), the exact AUC from t1 to t2 is:
        (C1 - C2) / k = (C1 - C2) * (t2 - t1) / ln(C1/C2)
        which is exactly the log-linear formula.
        """
        ke = 0.3
        C0 = 10.0
        time = np.array([0, 2, 5, 8, 12, 20])
        conc = C0 * np.exp(-ke * time)
        r = nca(time, conc, auc_method="log-linear")
        # True AUC from 0 to 20: C0/ke * (1 - exp(-ke*20))
        true_auc = (C0 / ke) * (1 - np.exp(-ke * 20))
        assert r.auc_last == pytest.approx(true_auc, rel=1e-10)
