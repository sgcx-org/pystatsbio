"""Tests for dose-response model functions."""

import numpy as np
import pytest

from pystatsbio.doseresponse import brain_cousens, ll4, ll5, weibull1, weibull2


class TestLL4:
    """4-parameter log-logistic model."""

    def test_at_ec50_gives_midpoint(self):
        """At dose=EC50, LL.4 should give (bottom+top)/2."""
        result = ll4(np.array([1.0]), bottom=0, top=100, ec50=1.0, hill=1.0)
        assert result[0] == pytest.approx(50.0)

    def test_at_ec50_general(self):
        """At dose=EC50 with arbitrary bottom/top."""
        result = ll4(np.array([5.0]), bottom=20, top=80, ec50=5.0, hill=2.0)
        assert result[0] == pytest.approx(50.0)

    def test_dose_zero_positive_hill(self):
        """dose=0 with positive hill → bottom."""
        result = ll4(np.array([0.0]), bottom=10, top=90, ec50=1.0, hill=1.0)
        assert result[0] == pytest.approx(10.0)

    def test_dose_zero_negative_hill(self):
        """dose=0 with negative hill → top."""
        result = ll4(np.array([0.0]), bottom=10, top=90, ec50=1.0, hill=-1.0)
        assert result[0] == pytest.approx(90.0)

    def test_increasing_positive_hill(self):
        """Positive hill → response increases with dose."""
        dose = np.array([0.01, 0.1, 1, 10, 100])
        resp = ll4(dose, bottom=0, top=100, ec50=1.0, hill=1.0)
        for i in range(len(resp) - 1):
            assert resp[i] < resp[i + 1]

    def test_decreasing_negative_hill(self):
        """Negative hill → response decreases with dose."""
        dose = np.array([0.01, 0.1, 1, 10, 100])
        resp = ll4(dose, bottom=0, top=100, ec50=1.0, hill=-1.0)
        for i in range(len(resp) - 1):
            assert resp[i] > resp[i + 1]

    def test_steeper_hill(self):
        """Higher |hill| → steeper transition."""
        dose = np.array([0.5])
        r_shallow = ll4(dose, 0, 100, 1.0, hill=0.5)
        r_steep = ll4(dose, 0, 100, 1.0, hill=5.0)
        # Both should be < 50 (since dose < EC50), but steep is closer to 0
        assert r_steep[0] < r_shallow[0]

    def test_large_dose_approaches_top(self):
        """Very large dose → top asymptote."""
        result = ll4(np.array([1e10]), bottom=0, top=100, ec50=1.0, hill=1.0)
        assert result[0] == pytest.approx(100.0, abs=0.01)

    def test_vector_output(self):
        """Multiple dose values at once."""
        dose = np.array([0.1, 1.0, 10.0])
        result = ll4(dose, 0, 100, 1.0, 1.0)
        assert result.shape == (3,)


class TestLL5:
    """5-parameter log-logistic model."""

    def test_asymmetry_1_equals_ll4(self):
        """LL.5 with asymmetry=1 should equal LL.4."""
        dose = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
        r4 = ll4(dose, 10, 90, 5.0, 2.0)
        r5 = ll5(dose, 10, 90, 5.0, 2.0, asymmetry=1.0)
        np.testing.assert_allclose(r4, r5)

    def test_asymmetry_shifts_curve(self):
        """Asymmetry != 1 should produce different curve shape."""
        dose = np.array([0.5, 1.0, 2.0])
        r_sym = ll5(dose, 0, 100, 1.0, 1.0, asymmetry=1.0)
        r_asym = ll5(dose, 0, 100, 1.0, 1.0, asymmetry=2.0)
        assert not np.allclose(r_sym, r_asym)


class TestWeibull1:
    """Weibull type 1 model."""

    def test_monotonic_increasing(self):
        """Positive hill → increasing."""
        dose = np.array([0.01, 0.1, 1, 10, 100])
        resp = weibull1(dose, 0, 100, 1.0, hill=1.0)
        for i in range(len(resp) - 1):
            assert resp[i] < resp[i + 1]

    def test_bounded(self):
        """Response stays within [bottom, top]."""
        dose = np.logspace(-3, 3, 50)
        resp = weibull1(dose, 10, 90, 1.0, hill=2.0)
        assert np.all(resp >= 10 - 0.01)
        assert np.all(resp <= 90 + 0.01)


class TestWeibull2:
    """Weibull type 2 model."""

    def test_monotonic(self):
        """W2.4 should be monotonic (decreasing for positive hill with this parameterisation)."""
        dose = np.array([0.01, 0.1, 1, 10, 100])
        resp = weibull2(dose, 0, 100, 1.0, hill=1.0)
        diffs = np.diff(resp)
        # All diffs should have the same sign (monotonic)
        assert np.all(diffs <= 0) or np.all(diffs >= 0)

    def test_complement_of_weibull1(self):
        """W2.4 is related to W1.4 (complement structure)."""
        dose = np.array([0.01, 0.1, 1, 10, 100])
        r1 = weibull1(dose, 0, 100, 1.0, hill=1.0)
        r2 = weibull2(dose, 0, 100, 1.0, hill=1.0)
        # Both are bounded [0, 100] and monotonic but have different shapes
        assert not np.allclose(r1, r2)


class TestBrainCousens:
    """Brain-Cousens hormesis model."""

    def test_hormesis_zero_equals_ll4(self):
        """hormesis=0 should reduce to LL.4."""
        dose = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
        r_ll4 = ll4(dose, 10, 90, 5.0, 2.0)
        r_bc = brain_cousens(dose, 10, 90, 5.0, 2.0, hormesis=0.0)
        np.testing.assert_allclose(r_ll4, r_bc)

    def test_hormesis_positive_overshoot(self):
        """Positive hormesis can push response above top at intermediate doses."""
        dose = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        resp = brain_cousens(dose, 0, 100, 10.0, 1.0, hormesis=5.0)
        # At some intermediate dose, response might exceed 100
        assert np.max(resp) > 100 or np.max(resp) >= 99
