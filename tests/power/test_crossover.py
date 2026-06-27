"""Tests for power_crossover_be."""

import pytest

from pystatsbio.power import power_crossover_be


class TestPowerCrossoverBE:
    """Tests for 2x2 crossover bioequivalence power."""

    def test_solve_n(self):
        """Solve for n."""
        r = power_crossover_be(coef_variation=0.30, power=0.80)
        assert isinstance(r.n, int)
        assert r.n > 0
        assert r.n % 2 == 0  # Should be even

    def test_solve_power(self):
        """Solve for power."""
        r = power_crossover_be(n=40, coef_variation=0.30)
        assert 0.0 < r.power < 1.0

    def test_roundtrip(self):
        """Round-trip: solve n, then verify power >= target."""
        r1 = power_crossover_be(coef_variation=0.30, power=0.80)
        r2 = power_crossover_be(n=r1.n, coef_variation=0.30)
        assert r2.power >= 0.80

    def test_higher_cv_more_n(self):
        """Higher within-subject variability needs more subjects."""
        r_low = power_crossover_be(coef_variation=0.20, power=0.80)
        r_high = power_crossover_be(coef_variation=0.40, power=0.80)
        assert r_high.n > r_low.n

    def test_theta0_at_1_easiest(self):
        """theta0=1.0 (no bias) should give highest power / lowest n."""
        r_0 = power_crossover_be(coef_variation=0.30, theta0=1.0, power=0.80)
        r_1 = power_crossover_be(coef_variation=0.30, theta0=0.90, power=0.80)
        assert r_0.n <= r_1.n

    def test_cv_required(self):
        """coef_variation must be provided."""
        with pytest.raises(ValueError, match="coef_variation is always required"):
            power_crossover_be(power=0.80)

    def test_n_even(self):
        """Result n should always be even."""
        for cv in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
            r = power_crossover_be(coef_variation=cv, power=0.80)
            assert r.n % 2 == 0, f"n={r.n} not even for cv={cv}"

    def test_small_n_error(self):
        """n < 4 should raise."""
        with pytest.raises(ValueError, match="n must be >= 4"):
            power_crossover_be(n=2, coef_variation=0.30)
