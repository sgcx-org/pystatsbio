"""Tests for power_prop_test and power_fisher_test."""

import math

import pytest

from pystatsbio.power import power_fisher_test, power_prop_test


class TestPowerPropTest:
    """Tests for two-proportion z-test power."""

    def test_solve_n(self):
        """Solve for n with moderate effect size."""
        h = 2 * (math.asin(math.sqrt(0.65)) - math.asin(math.sqrt(0.45)))
        r = power_prop_test(h=h, alpha=0.05, power=0.80)
        assert isinstance(r.n, int)
        assert r.n > 0

    def test_solve_power(self):
        """Solve for power."""
        r = power_prop_test(n=100, h=0.3, alpha=0.05)
        assert 0.0 < r.power < 1.0

    def test_solve_h(self):
        """Solve for effect size."""
        r = power_prop_test(n=100, alpha=0.05, power=0.80)
        assert r.effect_size > 0

    def test_roundtrip(self):
        """Round-trip: solve n, then verify power >= target."""
        r1 = power_prop_test(h=0.3, alpha=0.05, power=0.80)
        r2 = power_prop_test(n=r1.n, h=0.3, alpha=0.05)
        assert r2.power >= 0.80

    def test_larger_h_smaller_n(self):
        """Larger effect needs fewer subjects."""
        r1 = power_prop_test(h=0.3, alpha=0.05, power=0.80)
        r2 = power_prop_test(h=0.5, alpha=0.05, power=0.80)
        assert r2.n < r1.n

    def test_zero_h_error(self):
        """h=0 when solving for n should raise."""
        with pytest.raises(ValueError, match="h = 0"):
            power_prop_test(h=0.0, alpha=0.05, power=0.80)


class TestPowerFisherTest:
    """Tests for Fisher's exact test power (normal approximation)."""

    def test_solve_n(self):
        """Solve for n."""
        r = power_fisher_test(p1=0.6, p2=0.4, alpha=0.05, power=0.80)
        assert isinstance(r.n, int)
        assert r.n > 0

    def test_solve_power(self):
        """Solve for power."""
        r = power_fisher_test(n=100, p1=0.6, p2=0.4, alpha=0.05)
        assert 0.0 < r.power < 1.0

    def test_requires_p1_p2(self):
        """p1 and p2 must be provided."""
        with pytest.raises(ValueError, match="p1 and p2"):
            power_fisher_test(n=100, alpha=0.05)

    def test_invalid_p1(self):
        """p1 out of range."""
        with pytest.raises(ValueError, match="p1"):
            power_fisher_test(n=100, p1=1.5, p2=0.3, alpha=0.05)

    def test_cohens_h_computed(self):
        """Effect size should be Cohen's h."""
        r = power_fisher_test(p1=0.6, p2=0.4, alpha=0.05, power=0.80)
        expected_h = 2 * (math.asin(math.sqrt(0.6)) - math.asin(math.sqrt(0.4)))
        assert r.effect_size == pytest.approx(expected_h, rel=1e-6)
