"""Tests for power_logrank."""

import pytest

from pystatsbio.power import power_logrank


class TestPowerLogrank:
    """Tests for log-rank power calculation."""

    def test_solve_n_schoenfeld(self):
        """Solve for n with Schoenfeld formula."""
        r = power_logrank(hazard_ratio=0.7, alpha=0.05, power=0.80)
        assert isinstance(r.n, int)
        assert r.n > 0

    def test_solve_power(self):
        """Solve for power."""
        r = power_logrank(n=200, hazard_ratio=0.7, alpha=0.05)
        assert 0.0 < r.power < 1.0

    def test_solve_hr(self):
        """Solve for hazard ratio."""
        r = power_logrank(n=200, alpha=0.05, power=0.80)
        assert 0.0 < r.effect_size < 1.0  # HR < 1 by convention

    def test_roundtrip(self):
        """Round-trip: solve n, then verify power >= target."""
        r1 = power_logrank(hazard_ratio=0.7, alpha=0.05, power=0.80)
        r2 = power_logrank(n=r1.n, hazard_ratio=0.7, alpha=0.05)
        assert r2.power >= 0.80

    def test_methods_differ(self):
        """Different methods give different n."""
        r_s = power_logrank(hazard_ratio=0.7, alpha=0.05, power=0.80, method="schoenfeld")
        r_f = power_logrank(hazard_ratio=0.7, alpha=0.05, power=0.80, method="freedman")
        r_l = power_logrank(hazard_ratio=0.7, alpha=0.05, power=0.80, method="lachin_foulkes")
        # They should all produce valid results but different n
        assert r_s.n != r_f.n or r_s.n != r_l.n

    def test_censoring_increases_n(self):
        """More censoring (lower p_event) requires larger n."""
        r_full = power_logrank(hazard_ratio=0.7, alpha=0.05, power=0.80, p_event=1.0)
        r_cens = power_logrank(hazard_ratio=0.7, alpha=0.05, power=0.80, p_event=0.5)
        assert r_cens.n > r_full.n

    def test_unequal_allocation(self):
        """Unequal allocation changes n."""
        r_equal = power_logrank(hazard_ratio=0.7, alpha=0.05, power=0.80, alloc_ratio=1.0)
        r_unequal = power_logrank(hazard_ratio=0.7, alpha=0.05, power=0.80, alloc_ratio=2.0)
        assert r_equal.n != r_unequal.n

    def test_hr_equals_1_error(self):
        """HR=1 (no effect) should raise."""
        with pytest.raises(ValueError, match="hazard_ratio must be != 1"):
            power_logrank(hazard_ratio=1.0, alpha=0.05, power=0.80)

    def test_invalid_method(self):
        """Invalid method should raise."""
        with pytest.raises(ValueError, match="method"):
            power_logrank(hazard_ratio=0.7, alpha=0.05, power=0.80, method="invalid")

    def test_one_sided(self):
        """One-sided test needs fewer subjects."""
        r_two = power_logrank(hazard_ratio=0.7, alpha=0.05, power=0.80, alternative="two-sided")
        r_one = power_logrank(hazard_ratio=0.7, alpha=0.05, power=0.80, alternative="one-sided")
        assert r_one.n < r_two.n
