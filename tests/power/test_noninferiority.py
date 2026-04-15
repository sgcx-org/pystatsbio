"""Tests for non-inferiority, equivalence, and superiority power functions."""

import pytest

from pystatsbio.power import (
    power_equiv_mean,
    power_noninf_mean,
    power_noninf_prop,
    power_superiority_mean,
)


class TestPowerNoninfMean:
    """Tests for non-inferiority of means."""

    def test_solve_n(self):
        r = power_noninf_mean(delta=0.0, margin=0.5, sd=1.0, alpha=0.025, power=0.80)
        assert isinstance(r.n, int) and r.n > 0

    def test_solve_power(self):
        r = power_noninf_mean(n=100, delta=0.0, margin=0.5, sd=1.0, alpha=0.025)
        assert 0.0 < r.power < 1.0

    def test_solve_delta(self):
        r = power_noninf_mean(n=100, margin=0.5, sd=1.0, alpha=0.025, power=0.80)
        assert r.effect_size is not None

    def test_roundtrip(self):
        r1 = power_noninf_mean(delta=0.0, margin=0.5, sd=1.0, alpha=0.025, power=0.80)
        r2 = power_noninf_mean(n=r1.n, delta=0.0, margin=0.5, sd=1.0, alpha=0.025)
        assert r2.power >= 0.80

    def test_larger_margin_fewer_n(self):
        """Larger NI margin means easier to demonstrate NI, fewer subjects."""
        r_small = power_noninf_mean(delta=0.0, margin=0.3, sd=1.0, alpha=0.025, power=0.80)
        r_large = power_noninf_mean(delta=0.0, margin=0.5, sd=1.0, alpha=0.025, power=0.80)
        assert r_large.n < r_small.n

    def test_negative_margin_error(self):
        with pytest.raises(ValueError, match="margin"):
            power_noninf_mean(delta=0.0, margin=-0.5, sd=1.0, alpha=0.025, power=0.80)


class TestPowerNoninfProp:
    """Tests for non-inferiority of proportions."""

    def test_solve_n(self):
        r = power_noninf_prop(p1=0.85, p2=0.85, margin=0.10, alpha=0.025, power=0.80)
        assert isinstance(r.n, int) and r.n > 0

    def test_solve_power(self):
        r = power_noninf_prop(n=200, p1=0.85, p2=0.85, margin=0.10, alpha=0.025)
        assert 0.0 < r.power < 1.0

    def test_requires_p1_p2(self):
        with pytest.raises(ValueError, match="p1 and p2"):
            power_noninf_prop(n=100, margin=0.10, alpha=0.025)


class TestPowerEquivMean:
    """Tests for equivalence (TOST) of means."""

    def test_solve_n(self):
        r = power_equiv_mean(delta=0.0, margin=0.5, sd=1.0, alpha=0.05, power=0.80)
        assert isinstance(r.n, int) and r.n > 0

    def test_solve_power(self):
        r = power_equiv_mean(n=100, delta=0.0, margin=0.5, sd=1.0, alpha=0.05)
        assert 0.0 < r.power < 1.0

    def test_roundtrip(self):
        r1 = power_equiv_mean(delta=0.0, margin=0.5, sd=1.0, alpha=0.05, power=0.80)
        r2 = power_equiv_mean(n=r1.n, delta=0.0, margin=0.5, sd=1.0, alpha=0.05)
        assert r2.power >= 0.80

    def test_nonzero_delta_less_power(self):
        """True difference > 0 reduces equivalence power."""
        r0 = power_equiv_mean(n=100, delta=0.0, margin=0.5, sd=1.0, alpha=0.05)
        r1 = power_equiv_mean(n=100, delta=0.3, margin=0.5, sd=1.0, alpha=0.05)
        assert r1.power < r0.power


class TestPowerSuperiorityMean:
    """Tests for superiority of means."""

    def test_solve_n(self):
        r = power_superiority_mean(delta=0.5, margin=0.0, sd=1.0, alpha=0.025, power=0.80)
        assert isinstance(r.n, int) and r.n > 0

    def test_solve_power(self):
        r = power_superiority_mean(n=100, delta=0.5, margin=0.0, sd=1.0, alpha=0.025)
        assert 0.0 < r.power < 1.0

    def test_roundtrip(self):
        r1 = power_superiority_mean(delta=0.5, margin=0.0, sd=1.0, alpha=0.025, power=0.80)
        r2 = power_superiority_mean(n=r1.n, delta=0.5, margin=0.0, sd=1.0, alpha=0.025)
        assert r2.power >= 0.80

    def test_margin_increases_n(self):
        """Positive margin makes test harder -> more subjects."""
        r0 = power_superiority_mean(delta=0.5, margin=0.0, sd=1.0, alpha=0.025, power=0.80)
        r1 = power_superiority_mean(delta=0.5, margin=0.1, sd=1.0, alpha=0.025, power=0.80)
        assert r1.n > r0.n
