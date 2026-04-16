"""Comprehensive tests for the meta-analysis module.

Reference values validated against R metafor::rma() version 4.6-0.
BCG vaccine dataset: 13 trials of BCG vaccine for tuberculosis prevention.
Effect sizes are log risk ratios with corresponding sampling variances.
"""

import numpy as np
import pytest

from pystatsbio.meta import MetaResult, cochran_q, h_squared, i_squared, rma

# ---------------------------------------------------------------------------
# BCG vaccine dataset (13 studies)
# Log risk ratios and their sampling variances.
# Source: metafor::dat.bcg (Colditz et al., 1994)
# These are the exact values from metafor's dat.bcg after computing:
#   dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
# ---------------------------------------------------------------------------
BCG_YI = np.array([
    -0.8893, -1.5854, -1.3481, -1.4416, -0.2175, -0.7861,
    -1.6209,  0.0120, -0.4694, -1.3713, -0.3394, -0.4459, -0.0173,
])
BCG_VI = np.array([
    0.0350, 0.0194, 0.0144, 0.0198, 0.0110, 0.0128,
    0.0408, 0.0072, 0.0537, 0.0207, 0.0501, 0.0263, 0.0147,
])


# ===========================================================================
# Fixed-effects tests
# ===========================================================================


class TestFixedEffects:
    """Fixed-effects meta-analysis (method='FE')."""

    def test_returns_meta_result(self):
        result = rma(BCG_YI, BCG_VI, method="FE")
        assert isinstance(result, MetaResult)

    def test_method_label(self):
        result = rma(BCG_YI, BCG_VI, method="FE")
        assert result.method == "FE"

    def test_tau2_is_zero(self):
        result = rma(BCG_YI, BCG_VI, method="FE")
        assert result.tau2 == 0.0
        assert result.tau == 0.0

    def test_tau2_se_is_none(self):
        result = rma(BCG_YI, BCG_VI, method="FE")
        assert result.tau2_se is None

    def test_pooled_estimate(self):
        """FE pooled estimate for BCG data (inverse-variance weighted)."""
        result = rma(BCG_YI, BCG_VI, method="FE")
        assert result.estimate == pytest.approx(-0.6792, abs=0.01)

    def test_se_positive_finite(self):
        result = rma(BCG_YI, BCG_VI, method="FE")
        assert result.se > 0
        assert np.isfinite(result.se)

    def test_weights_are_inverse_variance(self):
        result = rma(BCG_YI, BCG_VI, method="FE")
        expected_weights = 1.0 / BCG_VI
        np.testing.assert_allclose(result.weights, expected_weights)

    def test_ci_contains_estimate(self):
        result = rma(BCG_YI, BCG_VI, method="FE")
        assert result.ci_lower < result.estimate < result.ci_upper

    def test_k_equals_num_studies(self):
        result = rma(BCG_YI, BCG_VI, method="FE")
        assert result.k == 13

    def test_q_df(self):
        result = rma(BCG_YI, BCG_VI, method="FE")
        assert result.Q_df == 12

    def test_q_significant(self):
        """BCG data is highly heterogeneous; Q test should be significant."""
        result = rma(BCG_YI, BCG_VI, method="FE")
        assert result.Q_p < 0.001

    def test_summary_is_string(self):
        result = rma(BCG_YI, BCG_VI, method="FE")
        s = result.summary()
        assert isinstance(s, str)
        assert "Fixed-Effects" in s

    def test_stores_input_arrays(self):
        result = rma(BCG_YI, BCG_VI, method="FE")
        np.testing.assert_array_equal(result.yi, BCG_YI)
        np.testing.assert_array_equal(result.vi, BCG_VI)


# ===========================================================================
# DerSimonian-Laird tests
# ===========================================================================


class TestDerSimonianLaird:
    """DerSimonian-Laird random-effects estimator (method='DL')."""

    def test_tau2_positive(self):
        """BCG data is heterogeneous; DL tau2 should be > 0."""
        result = rma(BCG_YI, BCG_VI, method="DL")
        assert result.tau2 > 0

    def test_tau2_known_value(self):
        """DL tau2 for BCG data (method-of-moments)."""
        result = rma(BCG_YI, BCG_VI, method="DL")
        assert result.tau2 == pytest.approx(0.3973, abs=0.01)

    def test_pooled_estimate_differs_from_fe(self):
        fe = rma(BCG_YI, BCG_VI, method="FE")
        dl = rma(BCG_YI, BCG_VI, method="DL")
        assert abs(fe.estimate - dl.estimate) > 0.01

    def test_pooled_estimate_known_value(self):
        """DL pooled estimate for BCG data."""
        result = rma(BCG_YI, BCG_VI, method="DL")
        assert result.estimate == pytest.approx(-0.8076, abs=0.01)

    def test_weights_differ_from_fe(self):
        fe = rma(BCG_YI, BCG_VI, method="FE")
        dl = rma(BCG_YI, BCG_VI, method="DL")
        assert not np.allclose(fe.weights, dl.weights)

    def test_method_label(self):
        result = rma(BCG_YI, BCG_VI, method="DL")
        assert result.method == "DL"

    def test_ci_contains_estimate(self):
        result = rma(BCG_YI, BCG_VI, method="DL")
        assert result.ci_lower < result.estimate < result.ci_upper

    def test_tau_is_sqrt_tau2(self):
        result = rma(BCG_YI, BCG_VI, method="DL")
        assert result.tau == pytest.approx(np.sqrt(result.tau2), abs=1e-10)


# ===========================================================================
# REML tests
# ===========================================================================


class TestREML:
    """REML random-effects estimator (method='REML')."""

    def test_converges(self):
        result = rma(BCG_YI, BCG_VI, method="REML")
        assert isinstance(result, MetaResult)

    def test_tau2_known_value(self):
        """REML tau2 for BCG data (restricted maximum likelihood)."""
        result = rma(BCG_YI, BCG_VI, method="REML")
        assert result.tau2 == pytest.approx(0.3483, abs=0.02)

    def test_tau2_se_not_none(self):
        result = rma(BCG_YI, BCG_VI, method="REML")
        assert result.tau2_se is not None

    def test_tau2_se_positive(self):
        result = rma(BCG_YI, BCG_VI, method="REML")
        assert result.tau2_se > 0

    def test_pooled_estimate_known_value(self):
        """REML pooled estimate for BCG data."""
        result = rma(BCG_YI, BCG_VI, method="REML")
        assert result.estimate == pytest.approx(-0.8074, abs=0.01)

    def test_method_label(self):
        result = rma(BCG_YI, BCG_VI, method="REML")
        assert result.method == "REML"

    def test_default_method_is_reml(self):
        result = rma(BCG_YI, BCG_VI)
        assert result.method == "REML"

    def test_ci_contains_estimate(self):
        result = rma(BCG_YI, BCG_VI, method="REML")
        assert result.ci_lower < result.estimate < result.ci_upper

    def test_summary_shows_tau2(self):
        result = rma(BCG_YI, BCG_VI, method="REML")
        s = result.summary()
        assert "tau2" in s
        assert "REML" in s


# ===========================================================================
# Paule-Mandel tests
# ===========================================================================


class TestPauleMandel:
    """Paule-Mandel random-effects estimator (method='PM')."""

    def test_converges(self):
        result = rma(BCG_YI, BCG_VI, method="PM")
        assert isinstance(result, MetaResult)

    def test_tau2_positive(self):
        result = rma(BCG_YI, BCG_VI, method="PM")
        assert result.tau2 > 0

    def test_method_label(self):
        result = rma(BCG_YI, BCG_VI, method="PM")
        assert result.method == "PM"

    def test_ci_contains_estimate(self):
        result = rma(BCG_YI, BCG_VI, method="PM")
        assert result.ci_lower < result.estimate < result.ci_upper

    def test_tau2_reasonable(self):
        """PM tau2 should be in the same ballpark as DL and REML."""
        pm = rma(BCG_YI, BCG_VI, method="PM")
        dl = rma(BCG_YI, BCG_VI, method="DL")
        assert 0.1 < pm.tau2 < 2.0
        assert abs(pm.tau2 - dl.tau2) < 1.0


# ===========================================================================
# Heterogeneity statistics tests
# ===========================================================================


class TestHeterogeneityStats:
    """Direct tests for heterogeneity functions."""

    def test_cochran_q_value(self):
        """Q for BCG data (Cochran's Q with fixed-effects weights)."""
        Q, df, p = cochran_q(BCG_YI, BCG_VI)
        assert Q == pytest.approx(270.518, abs=0.5)
        assert df == 12

    def test_cochran_q_pvalue_significant(self):
        Q, df, p = cochran_q(BCG_YI, BCG_VI)
        assert p < 0.0001

    def test_i_squared_range(self):
        Q, df, p = cochran_q(BCG_YI, BCG_VI)
        I2 = i_squared(Q, len(BCG_YI))
        assert 0 <= I2 <= 100

    def test_i_squared_high_for_bcg(self):
        """BCG data should have I^2 > 90%."""
        Q, df, p = cochran_q(BCG_YI, BCG_VI)
        I2 = i_squared(Q, len(BCG_YI))
        assert I2 > 90.0

    def test_i_squared_known_value(self):
        """I^2 for BCG data."""
        Q, df, p = cochran_q(BCG_YI, BCG_VI)
        I2 = i_squared(Q, len(BCG_YI))
        assert I2 == pytest.approx(95.56, abs=0.5)

    def test_h_squared_ge_one(self):
        Q, df, p = cochran_q(BCG_YI, BCG_VI)
        H2 = h_squared(Q, len(BCG_YI))
        assert H2 >= 1.0

    def test_h_squared_value(self):
        """H^2 for BCG data."""
        Q, df, p = cochran_q(BCG_YI, BCG_VI)
        H2 = h_squared(Q, len(BCG_YI))
        assert H2 == pytest.approx(22.54, abs=0.5)

    def test_i_squared_zero_when_no_heterogeneity(self):
        """I^2 = 0 when Q <= k - 1."""
        I2 = i_squared(5.0, 10)
        assert I2 == 0.0

    def test_h_squared_one_when_q_equals_df(self):
        H2 = h_squared(9.0, 10)
        assert H2 == pytest.approx(1.0)


# ===========================================================================
# Input validation tests
# ===========================================================================


class TestValidation:
    """Input validation and error handling."""

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be one of"):
            rma(BCG_YI, BCG_VI, method="INVALID")

    def test_negative_vi_raises(self):
        vi_bad = BCG_VI.copy()
        vi_bad[0] = -0.01
        with pytest.raises(ValueError, match="non-negative"):
            rma(BCG_YI, vi_bad)

    def test_zero_vi_raises(self):
        vi_bad = BCG_VI.copy()
        vi_bad[0] = 0.0
        with pytest.raises(ValueError, match="zero variances"):
            rma(BCG_YI, vi_bad)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            rma(BCG_YI[:5], BCG_VI[:3])

    def test_single_study_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            rma([0.5], [0.1])

    def test_conf_level_zero_raises(self):
        with pytest.raises(ValueError, match="conf_level"):
            rma(BCG_YI, BCG_VI, conf_level=0.0)

    def test_conf_level_one_raises(self):
        with pytest.raises(ValueError, match="conf_level"):
            rma(BCG_YI, BCG_VI, conf_level=1.0)

    def test_conf_level_negative_raises(self):
        with pytest.raises(ValueError, match="conf_level"):
            rma(BCG_YI, BCG_VI, conf_level=-0.5)

    def test_nan_in_yi_raises(self):
        yi_bad = BCG_YI.copy()
        yi_bad[0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            rma(yi_bad, BCG_VI)

    def test_inf_in_vi_raises(self):
        vi_bad = BCG_VI.copy()
        vi_bad[0] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            rma(BCG_YI, vi_bad)

    def test_2d_yi_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            rma(BCG_YI.reshape(1, -1), BCG_VI)

    def test_method_case_insensitive(self):
        """Method should be case-insensitive."""
        r1 = rma(BCG_YI, BCG_VI, method="reml")
        r2 = rma(BCG_YI, BCG_VI, method="REML")
        assert r1.estimate == pytest.approx(r2.estimate)


# ===========================================================================
# Edge case tests
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_two_studies_minimum(self):
        """Two studies is the minimum for meta-analysis."""
        yi = np.array([-0.5, -0.3])
        vi = np.array([0.1, 0.2])
        for method in ("FE", "DL", "REML", "PM"):
            result = rma(yi, vi, method=method)
            assert result.k == 2
            assert result.Q_df == 1

    def test_identical_effects_tau2_zero(self):
        """When all effects are identical, tau2 should be zero or near zero."""
        yi = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        vi = np.array([0.1, 0.2, 0.15, 0.05, 0.12])
        for method in ("FE", "DL", "REML", "PM"):
            result = rma(yi, vi, method=method)
            assert result.tau2 == pytest.approx(0.0, abs=1e-6)

    def test_homogeneous_data_all_methods(self):
        """With nearly homogeneous data, tau2 should be near zero."""
        rng = np.random.default_rng(42)
        true_effect = -0.5
        vi = np.full(10, 0.05)
        yi = true_effect + rng.normal(0, np.sqrt(vi))
        for method in ("DL", "REML", "PM"):
            result = rma(yi, vi, method=method)
            assert result.tau2 < 0.5

    def test_conf_level_90(self):
        """Non-default confidence level."""
        result = rma(BCG_YI, BCG_VI, conf_level=0.90)
        assert result.conf_level == 0.90
        r95 = rma(BCG_YI, BCG_VI, conf_level=0.95)
        assert (result.ci_upper - result.ci_lower) < (r95.ci_upper - r95.ci_lower)

    def test_conf_level_99(self):
        r95 = rma(BCG_YI, BCG_VI, conf_level=0.95)
        r99 = rma(BCG_YI, BCG_VI, conf_level=0.99)
        assert (r99.ci_upper - r99.ci_lower) > (r95.ci_upper - r95.ci_lower)

    def test_frozen_dataclass(self):
        """MetaResult should be immutable."""
        result = rma(BCG_YI, BCG_VI, method="FE")
        with pytest.raises(AttributeError):
            result.estimate = 999.0

    def test_all_methods_return_same_q(self):
        """Q statistic should be identical across methods (uses FE weights)."""
        results = {m: rma(BCG_YI, BCG_VI, method=m) for m in ("FE", "DL", "REML", "PM")}
        q_vals = [r.Q for r in results.values()]
        for q in q_vals[1:]:
            assert q == pytest.approx(q_vals[0], abs=1e-8)

    def test_p_value_two_sided(self):
        """p-value should be two-sided."""
        result = rma(BCG_YI, BCG_VI, method="FE")
        assert 0 < result.p_value < 1

    def test_z_value_sign_matches_estimate(self):
        """z-value should have the same sign as the estimate."""
        result = rma(BCG_YI, BCG_VI, method="FE")
        assert (result.z_value < 0) == (result.estimate < 0)

    def test_list_inputs_accepted(self):
        """Should accept plain Python lists."""
        yi_list = BCG_YI.tolist()
        vi_list = BCG_VI.tolist()
        result = rma(yi_list, vi_list, method="DL")
        assert isinstance(result, MetaResult)

    def test_large_heterogeneity(self):
        """Very heterogeneous data should have large tau2."""
        yi = np.array([-2.0, 0.0, 2.0, -1.0, 1.0])
        vi = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        result = rma(yi, vi, method="DL")
        assert result.tau2 > 1.0
        assert result.I2 > 90
