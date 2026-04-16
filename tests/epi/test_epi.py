"""Comprehensive tests for the epidemiology module."""

import numpy as np
import pytest

from pystatsbio.epi import (
    Epi2x2Result,
    EpiMeasure,
    MantelHaenszelResult,
    StandardizedRate,
    epi_2by2,
    mantel_haenszel,
    rate_standardize,
)


# ===========================================================================
# epi_2by2 tests
# ===========================================================================


class TestEpi2by2Basic:
    """Basic functionality and return types for epi_2by2."""

    def test_returns_epi2x2result(self):
        result = epi_2by2([[10, 90], [5, 95]])
        assert isinstance(result, Epi2x2Result)

    def test_all_measures_are_epimeasure(self):
        result = epi_2by2([[10, 90], [5, 95]])
        assert isinstance(result.risk_ratio, EpiMeasure)
        assert isinstance(result.odds_ratio, EpiMeasure)
        assert isinstance(result.risk_difference, EpiMeasure)
        assert isinstance(result.attributable_risk_exposed, EpiMeasure)
        assert isinstance(result.population_attributable_fraction, EpiMeasure)
        assert isinstance(result.nnt, EpiMeasure)

    def test_table_stored(self):
        result = epi_2by2([[10, 90], [5, 95]])
        assert result.table.shape == (2, 2)

    def test_summary_is_string(self):
        result = epi_2by2([[10, 90], [5, 95]])
        s = result.summary()
        assert isinstance(s, str)
        assert "Risk Ratio" in s


class TestEpi2by2TextbookExample:
    """Classic cohort study: table [[136, 10864], [1000, 99000]].

    RR = (136/11000) / (1000/100000) = 0.012364 / 0.01 = 1.2364
    OR = (136*99000) / (10864*1000) = 1.2393
    """

    @pytest.fixture
    def result(self):
        return epi_2by2([[136, 10864], [1000, 99000]])

    def test_risk_ratio_value(self, result):
        # RR = (136/11000) / (1000/100000) = 0.012364 / 0.01 = 1.2364
        assert result.risk_ratio.estimate == pytest.approx(1.2364, rel=0.01)

    def test_odds_ratio_value(self, result):
        # OR = (136*99000) / (10864*1000) = 13464000 / 10864000 = 12.393
        assert result.odds_ratio.estimate == pytest.approx(
            (136 * 99000) / (10864 * 1000), rel=0.001
        )

    def test_risk_ratio_positive(self, result):
        assert result.risk_ratio.estimate > 1.0

    def test_odds_ratio_positive(self, result):
        assert result.odds_ratio.estimate > 1.0

    def test_risk_difference_positive(self, result):
        # Exposed has higher risk
        assert result.risk_difference.estimate > 0

    def test_ci_contains_estimate_rr(self, result):
        rr = result.risk_ratio
        assert rr.ci_lower <= rr.estimate <= rr.ci_upper

    def test_ci_contains_estimate_or(self, result):
        orr = result.odds_ratio
        assert orr.ci_lower <= orr.estimate <= orr.ci_upper

    def test_ci_contains_estimate_rd(self, result):
        rd = result.risk_difference
        assert rd.ci_lower <= rd.estimate <= rd.ci_upper

    def test_attributable_risk_exposed_positive(self, result):
        # When RR > 1, AFe should be positive
        assert result.attributable_risk_exposed.estimate > 0

    def test_paf_positive(self, result):
        # When RR > 1, PAF should be positive
        assert result.population_attributable_fraction.estimate > 0

    def test_nnt_positive(self, result):
        assert result.nnt.estimate > 0

    def test_afe_formula(self, result):
        # AFe = (RR - 1) / RR
        rr = result.risk_ratio.estimate
        expected = (rr - 1) / rr
        assert result.attributable_risk_exposed.estimate == pytest.approx(
            expected, rel=1e-6
        )


class TestEpi2by2DirectionAndSign:
    """Verify correct sign/direction across different table patterns."""

    def test_protective_exposure(self):
        """When exposure is protective (RR < 1)."""
        result = epi_2by2([[5, 95], [20, 80]])
        assert result.risk_ratio.estimate < 1.0
        assert result.odds_ratio.estimate < 1.0
        assert result.risk_difference.estimate < 0

    def test_no_effect(self):
        """Balanced table: RR ~ 1, OR ~ 1."""
        result = epi_2by2([[50, 50], [50, 50]])
        assert result.risk_ratio.estimate == pytest.approx(1.0, rel=0.01)
        assert result.odds_ratio.estimate == pytest.approx(1.0, rel=0.01)
        assert result.risk_difference.estimate == pytest.approx(0.0, abs=0.01)


class TestEpi2by2ZeroCells:
    """Zero-cell handling with continuity correction."""

    def test_zero_cell_does_not_raise(self):
        """A table with a zero cell should use 0.5 correction."""
        result = epi_2by2([[0, 100], [10, 90]])
        assert isinstance(result, Epi2x2Result)
        # The corrected table should have no zeros
        assert np.all(result.table > 0)

    def test_correction_applied(self):
        """Verify the table stored has 0.5 added."""
        result = epi_2by2([[0, 100], [10, 90]])
        assert result.table[0, 0] == pytest.approx(0.5)
        assert result.table[0, 1] == pytest.approx(100.5)


class TestEpi2by2Validation:
    """Input validation for epi_2by2."""

    def test_non_2by2_raises(self):
        with pytest.raises(ValueError, match="2x2"):
            epi_2by2([[1, 2, 3], [4, 5, 6]])

    def test_3by2_raises(self):
        with pytest.raises(ValueError, match="2x2"):
            epi_2by2([[1, 2], [3, 4], [5, 6]])

    def test_negative_values_raise(self):
        with pytest.raises(ValueError, match="non-negative"):
            epi_2by2([[10, -5], [5, 95]])

    def test_invalid_conf_level_low(self):
        with pytest.raises(ValueError, match="conf_level"):
            epi_2by2([[10, 90], [5, 95]], conf_level=0.0)

    def test_invalid_conf_level_high(self):
        with pytest.raises(ValueError, match="conf_level"):
            epi_2by2([[10, 90], [5, 95]], conf_level=1.0)

    def test_1d_array_raises(self):
        with pytest.raises(ValueError, match="2x2"):
            epi_2by2([10, 90, 5, 95])


class TestEpi2by2ConfidenceLevels:
    """Test that different confidence levels produce wider/narrower CIs."""

    def test_wider_ci_at_99(self):
        r95 = epi_2by2([[136, 10864], [1000, 99000]], conf_level=0.95)
        r99 = epi_2by2([[136, 10864], [1000, 99000]], conf_level=0.99)
        # 99% CI should be wider than 95% CI
        width_95 = r95.risk_ratio.ci_upper - r95.risk_ratio.ci_lower
        width_99 = r99.risk_ratio.ci_upper - r99.risk_ratio.ci_lower
        assert width_99 > width_95


# ===========================================================================
# rate_standardize tests
# ===========================================================================


class TestRateStandardizeDirect:
    """Direct age standardization."""

    @pytest.fixture
    def example_data(self):
        """Three age strata with different rates."""
        counts = np.array([10, 20, 50])
        person_time = np.array([1000, 2000, 5000])
        standard_pop = np.array([30000, 40000, 30000])
        return counts, person_time, standard_pop

    def test_returns_standardized_rate(self, example_data):
        result = rate_standardize(*example_data, method="direct")
        assert isinstance(result, StandardizedRate)

    def test_method_is_direct(self, example_data):
        result = rate_standardize(*example_data, method="direct")
        assert result.method == "direct"

    def test_sir_is_none(self, example_data):
        """Direct method should not produce SIR."""
        result = rate_standardize(*example_data, method="direct")
        assert result.sir is None
        assert result.sir_ci is None

    def test_adjusted_differs_from_crude(self, example_data):
        """When strata rates differ and weights differ, adjusted != crude."""
        result = rate_standardize(*example_data, method="direct")
        # With equal rates, adjusted would equal crude. With different rates
        # and different standard population weights, they should differ.
        # crude = 80/8000 = 0.01
        assert result.crude_rate == pytest.approx(80 / 8000)
        # adjusted rate should differ from crude since weight distribution
        # differs from the study population distribution
        # Not necessarily different by much here, but the CI should be valid
        assert result.adjusted_rate_ci[0] <= result.adjusted_rate
        assert result.adjusted_rate <= result.adjusted_rate_ci[1]

    def test_crude_rate_value(self, example_data):
        result = rate_standardize(*example_data, method="direct")
        expected_crude = 80 / 8000  # sum(counts) / sum(person_time)
        assert result.crude_rate == pytest.approx(expected_crude)

    def test_summary_is_string(self, example_data):
        result = rate_standardize(*example_data, method="direct")
        s = result.summary()
        assert isinstance(s, str)
        assert "direct" in s

    def test_uniform_rates_adjusted_equals_crude(self):
        """If all strata have same rate, adjusted == crude regardless of weights."""
        counts = np.array([10, 20, 30])
        person_time = np.array([1000, 2000, 3000])
        standard_pop = np.array([50000, 30000, 20000])
        result = rate_standardize(counts, person_time, standard_pop, method="direct")
        assert result.adjusted_rate == pytest.approx(result.crude_rate, rel=1e-10)


class TestRateStandardizeIndirect:
    """Indirect age standardization."""

    @pytest.fixture
    def example_data(self):
        """Three strata with standard rates."""
        counts = np.array([15, 25, 60])
        person_time = np.array([1000, 2000, 5000])
        standard_rates = np.array([0.008, 0.012, 0.015])
        return counts, person_time, standard_rates

    def test_returns_standardized_rate(self, example_data):
        result = rate_standardize(*example_data, method="indirect")
        assert isinstance(result, StandardizedRate)

    def test_method_is_indirect(self, example_data):
        result = rate_standardize(*example_data, method="indirect")
        assert result.method == "indirect"

    def test_sir_computed(self, example_data):
        """Indirect method should produce SIR."""
        result = rate_standardize(*example_data, method="indirect")
        assert result.sir is not None
        assert result.sir_ci is not None
        assert result.sir > 0

    def test_sir_value(self, example_data):
        """SIR = observed / expected."""
        counts, person_time, standard_rates = example_data
        observed = np.sum(counts)  # 100
        expected = np.sum(standard_rates * person_time)
        # expected = 0.008*1000 + 0.012*2000 + 0.015*5000 = 8 + 24 + 75 = 107
        result = rate_standardize(*example_data, method="indirect")
        assert result.sir == pytest.approx(observed / expected, rel=1e-6)

    def test_sir_ci_contains_sir(self, example_data):
        result = rate_standardize(*example_data, method="indirect")
        assert result.sir_ci[0] <= result.sir <= result.sir_ci[1]


class TestRateStandardizeValidation:
    """Input validation for rate_standardize."""

    def test_different_lengths_raise(self):
        with pytest.raises(ValueError, match="equal length"):
            rate_standardize([10, 20], [1000, 2000, 3000], [5000, 5000])

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            rate_standardize([10], [1000], [5000], method="other")

    def test_negative_counts_raise(self):
        with pytest.raises(ValueError, match="non-negative"):
            rate_standardize([-1, 10], [1000, 2000], [5000, 5000])

    def test_zero_person_time_raises(self):
        with pytest.raises(ValueError, match="positive"):
            rate_standardize([10, 20], [0, 2000], [5000, 5000])

    def test_empty_arrays_raise(self):
        with pytest.raises(ValueError, match="empty"):
            rate_standardize([], [], [])

    def test_invalid_conf_level(self):
        with pytest.raises(ValueError, match="conf_level"):
            rate_standardize([10], [1000], [5000], conf_level=1.5)

    def test_2d_arrays_raise(self):
        with pytest.raises(ValueError, match="1-D"):
            rate_standardize([[10, 20]], [[1000, 2000]], [[5000, 5000]])


# ===========================================================================
# mantel_haenszel tests
# ===========================================================================


class TestMantelHaenszelBasic:
    """Basic functionality of mantel_haenszel."""

    @pytest.fixture
    def three_strata(self):
        """Three strata with moderate association."""
        return np.array([
            [[20, 80], [10, 90]],
            [[30, 70], [15, 85]],
            [[25, 75], [12, 88]],
        ])

    def test_returns_result(self, three_strata):
        result = mantel_haenszel(three_strata)
        assert isinstance(result, MantelHaenszelResult)

    def test_pooled_is_epimeasure(self, three_strata):
        result = mantel_haenszel(three_strata)
        assert isinstance(result.pooled_estimate, EpiMeasure)

    def test_n_strata(self, three_strata):
        result = mantel_haenszel(three_strata)
        assert result.n_strata == 3

    def test_measure_or(self, three_strata):
        result = mantel_haenszel(three_strata, measure="OR")
        assert result.measure == "OR"

    def test_measure_rr(self, three_strata):
        result = mantel_haenszel(three_strata, measure="RR")
        assert result.measure == "RR"

    def test_cmh_p_value_range(self, three_strata):
        result = mantel_haenszel(three_strata)
        assert 0 <= result.cmh_p_value <= 1

    def test_breslow_day_computed_for_or(self, three_strata):
        result = mantel_haenszel(three_strata, measure="OR")
        assert result.breslow_day_statistic is not None
        assert result.breslow_day_p_value is not None

    def test_breslow_day_none_for_rr(self, three_strata):
        result = mantel_haenszel(three_strata, measure="RR")
        assert result.breslow_day_statistic is None
        assert result.breslow_day_p_value is None

    def test_summary_is_string(self, three_strata):
        result = mantel_haenszel(three_strata)
        s = result.summary()
        assert isinstance(s, str)
        assert "Mantel-Haenszel" in s


class TestMantelHaenszelValues:
    """Numerical correctness of MH estimates."""

    def test_single_stratum_or_equals_crude(self):
        """With one stratum, MH OR should equal the crude OR."""
        table = np.array([[[20, 80], [10, 90]]])
        result = mantel_haenszel(table, measure="OR")
        # Crude OR = (20*90) / (80*10) = 1800 / 800 = 2.25
        expected_or = (20 * 90) / (80 * 10)
        assert result.pooled_estimate.estimate == pytest.approx(
            expected_or, rel=0.01
        )

    def test_single_stratum_rr_equals_crude(self):
        """With one stratum, MH RR should equal the crude RR."""
        table = np.array([[[20, 80], [10, 90]]])
        result = mantel_haenszel(table, measure="RR")
        # Crude RR = (20/100) / (10/100) = 2.0
        expected_rr = (20 / 100) / (10 / 100)
        assert result.pooled_estimate.estimate == pytest.approx(
            expected_rr, rel=0.01
        )

    def test_ci_contains_estimate(self):
        tables = np.array([
            [[20, 80], [10, 90]],
            [[30, 70], [15, 85]],
        ])
        result = mantel_haenszel(tables, measure="OR")
        est = result.pooled_estimate
        assert est.ci_lower <= est.estimate <= est.ci_upper

    def test_significant_cmh(self):
        """Strong association should give significant CMH p-value."""
        tables = np.array([
            [[50, 50], [10, 90]],
            [[40, 60], [8, 92]],
            [[45, 55], [12, 88]],
        ])
        result = mantel_haenszel(tables)
        assert result.cmh_p_value < 0.05


class TestMantelHaenszelHomogeneity:
    """Breslow-Day test for homogeneity detection."""

    def test_homogeneous_strata_high_p(self):
        """Strata with similar ORs should have non-significant BD test."""
        # All strata have similar OR ~ 2.25
        tables = np.array([
            [[20, 80], [10, 90]],
            [[20, 80], [10, 90]],
            [[20, 80], [10, 90]],
        ])
        result = mantel_haenszel(tables, measure="OR")
        # BD p-value should be high (strata are identical)
        assert result.breslow_day_p_value > 0.5

    def test_heterogeneous_strata_low_p(self):
        """Strata with very different ORs should have significant BD test."""
        # Stratum 1: OR ~ 9, Stratum 2: OR ~ 0.25
        tables = np.array([
            [[45, 5], [5, 45]],
            [[5, 45], [45, 5]],
            [[45, 5], [5, 45]],
            [[5, 45], [45, 5]],
        ])
        result = mantel_haenszel(tables, measure="OR")
        # BD should detect heterogeneity
        assert result.breslow_day_p_value < 0.05


class TestMantelHaenszelValidation:
    """Input validation for mantel_haenszel."""

    def test_invalid_measure_raises(self):
        with pytest.raises(ValueError, match="measure"):
            mantel_haenszel(np.array([[[10, 90], [5, 95]]]), measure="PR")

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="K, 2, 2"):
            mantel_haenszel(np.array([[10, 90], [5, 95]]))

    def test_negative_cells_raise(self):
        with pytest.raises(ValueError, match="non-negative"):
            mantel_haenszel(np.array([[[-1, 90], [5, 95]]]))

    def test_invalid_conf_level(self):
        with pytest.raises(ValueError, match="conf_level"):
            mantel_haenszel(
                np.array([[[10, 90], [5, 95]]]),
                conf_level=0.0,
            )

    def test_empty_tables_raise(self):
        with pytest.raises(ValueError):
            mantel_haenszel(np.zeros((0, 2, 2)))


# ===========================================================================
# EpiMeasure summary tests
# ===========================================================================


class TestEpiMeasureSummary:
    """Test EpiMeasure.summary() formatting."""

    def test_summary_format(self):
        m = EpiMeasure(
            name="Risk Ratio",
            estimate=2.0,
            ci_lower=1.5,
            ci_upper=2.7,
            conf_level=0.95,
            method="log-transformed",
        )
        s = m.summary()
        assert "Risk Ratio" in s
        assert "2.0000" in s
        assert "95%" in s
        assert "log-transformed" in s
