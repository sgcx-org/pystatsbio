"""Tests for power_anova_oneway and power_anova_factorial."""

import pytest

from pystatsbio.power import power_anova_factorial, power_anova_oneway


class TestPowerAnovaOneway:
    """Tests for one-way ANOVA power."""

    def test_solve_n(self):
        """Solve for n."""
        r = power_anova_oneway(f=0.25, k=3, alpha=0.05, power=0.80)
        assert isinstance(r.n, int)
        assert r.n > 0

    def test_solve_power(self):
        """Solve for power."""
        r = power_anova_oneway(n=50, f=0.25, k=3, alpha=0.05)
        assert 0.0 < r.power < 1.0

    def test_solve_f(self):
        """Solve for Cohen's f."""
        r = power_anova_oneway(n=50, k=3, alpha=0.05, power=0.80)
        assert r.effect_size > 0

    def test_roundtrip(self):
        """Round-trip: solve n, then verify power >= target."""
        r1 = power_anova_oneway(f=0.25, k=3, alpha=0.05, power=0.80)
        r2 = power_anova_oneway(n=r1.n, f=0.25, k=3, alpha=0.05)
        assert r2.power >= 0.80

    def test_more_groups_more_n(self):
        """More groups require more subjects (per group)."""
        r3 = power_anova_oneway(f=0.25, k=3, alpha=0.05, power=0.80)
        r5 = power_anova_oneway(f=0.25, k=5, alpha=0.05, power=0.80)
        # Total subjects = n * k; per-group n might decrease but total should increase
        assert r5.n * 5 >= r3.n * 3

    def test_k_less_than_2_error(self):
        with pytest.raises(ValueError, match="k must be >= 2"):
            power_anova_oneway(f=0.25, k=1, alpha=0.05, power=0.80)


class TestPowerAnovaFactorial:
    """Tests for factorial ANOVA power."""

    def test_solve_n_interaction(self):
        """Solve for n (interaction effect)."""
        r = power_anova_factorial(f=0.25, n_levels=(2, 3), alpha=0.05, power=0.80)
        assert isinstance(r.n, int)
        assert r.n > 0

    def test_solve_n_main_effect(self):
        """Solve for n (main effect)."""
        r = power_anova_factorial(
            f=0.25, n_levels=(2, 3), alpha=0.05, power=0.80, effect="main_A",
        )
        assert isinstance(r.n, int)

    def test_solve_power(self):
        """Solve for power."""
        r = power_anova_factorial(n=30, f=0.25, n_levels=(2, 3), alpha=0.05)
        assert 0.0 < r.power < 1.0

    def test_roundtrip(self):
        """Round-trip consistency."""
        r1 = power_anova_factorial(f=0.25, n_levels=(2, 3), alpha=0.05, power=0.80)
        r2 = power_anova_factorial(n=r1.n, f=0.25, n_levels=(2, 3), alpha=0.05)
        assert r2.power >= 0.80

    def test_invalid_effect(self):
        with pytest.raises(ValueError, match="effect"):
            power_anova_factorial(
                f=0.25, n_levels=(2, 3), alpha=0.05, power=0.80, effect="invalid",
            )

    def test_invalid_n_levels(self):
        with pytest.raises(ValueError, match="at least 2 factors"):
            power_anova_factorial(f=0.25, n_levels=(3,), alpha=0.05, power=0.80)
