"""Tests for power_t_test and power_paired_t_test."""


import pytest

from pystatsbio.power import power_paired_t_test, power_t_test


class TestPowerTTestSolveN:
    """Solve for n (given d, alpha, power)."""

    def test_classic_textbook(self):
        """d=0.5, alpha=0.05, power=0.80 -> n=64 per group (two-sample)."""
        r = power_t_test(d=0.5, alpha=0.05, power=0.80)
        assert r.n == 64

    def test_small_effect(self):
        """Small effect requires larger n."""
        r = power_t_test(d=0.2, alpha=0.05, power=0.80)
        assert r.n > 300

    def test_large_effect(self):
        """Large effect requires smaller n."""
        r = power_t_test(d=0.8, alpha=0.05, power=0.80)
        assert r.n < 30

    def test_one_sample(self):
        """One-sample t-test requires fewer subjects than two-sample."""
        r_two = power_t_test(d=0.5, alpha=0.05, power=0.80, type="two.sample")
        r_one = power_t_test(d=0.5, alpha=0.05, power=0.80, type="one.sample")
        assert r_one.n < r_two.n

    def test_paired(self):
        """Paired t-test has same formula as one-sample."""
        r_one = power_t_test(d=0.5, alpha=0.05, power=0.80, type="one.sample")
        r_pair = power_t_test(d=0.5, alpha=0.05, power=0.80, type="paired")
        assert r_one.n == r_pair.n

    def test_one_sided_smaller_n(self):
        """One-sided test needs fewer subjects than two-sided."""
        r_two = power_t_test(d=0.5, alpha=0.05, power=0.80, alternative="two.sided")
        r_one = power_t_test(d=0.5, alpha=0.05, power=0.80, alternative="greater")
        assert r_one.n < r_two.n

    def test_higher_power_more_n(self):
        """Higher power requires more subjects."""
        r_80 = power_t_test(d=0.5, alpha=0.05, power=0.80)
        r_90 = power_t_test(d=0.5, alpha=0.05, power=0.90)
        assert r_90.n > r_80.n

    def test_lower_alpha_more_n(self):
        """Lower alpha requires more subjects."""
        r_05 = power_t_test(d=0.5, alpha=0.05, power=0.80)
        r_01 = power_t_test(d=0.5, alpha=0.01, power=0.80)
        assert r_01.n > r_05.n


class TestPowerTTestSolvePower:
    """Solve for power (given n, d, alpha)."""

    def test_n64_d05(self):
        """n=64, d=0.5, alpha=0.05 should give power ~ 0.80."""
        r = power_t_test(n=64, d=0.5, alpha=0.05)
        assert r.power == pytest.approx(0.80, abs=0.02)

    def test_power_increases_with_n(self):
        """Power should increase monotonically with n."""
        powers = []
        for n in [20, 40, 60, 80, 100]:
            r = power_t_test(n=n, d=0.5, alpha=0.05)
            powers.append(r.power)
        for i in range(len(powers) - 1):
            assert powers[i] < powers[i + 1]

    def test_power_bounded(self):
        """Power should be in [0, 1]."""
        r = power_t_test(n=10, d=0.1, alpha=0.05)
        assert 0.0 <= r.power <= 1.0

    def test_large_n_high_power(self):
        """Very large n should give power near 1."""
        r = power_t_test(n=10000, d=0.5, alpha=0.05)
        assert r.power > 0.999


class TestPowerTTestSolveD:
    """Solve for d (given n, alpha, power)."""

    def test_n50_power80(self):
        """n=50, power=0.80 should give d ~ 0.57."""
        r = power_t_test(n=50, alpha=0.05, power=0.80)
        assert r.effect_size == pytest.approx(0.57, abs=0.01)

    def test_larger_n_smaller_d(self):
        """Larger n should detect smaller effect size."""
        r1 = power_t_test(n=50, alpha=0.05, power=0.80)
        r2 = power_t_test(n=200, alpha=0.05, power=0.80)
        assert r2.effect_size < r1.effect_size


class TestPowerTTestRoundTrip:
    """Round-trip consistency: solve for n, then verify power >= target."""

    @pytest.mark.parametrize("d,power_target", [
        (0.2, 0.80), (0.5, 0.80), (0.8, 0.80),
        (0.3, 0.90), (0.5, 0.95),
    ])
    def test_roundtrip(self, d, power_target):
        """Solving for n then computing power should meet or exceed target."""
        r1 = power_t_test(d=d, alpha=0.05, power=power_target)
        r2 = power_t_test(n=r1.n, d=d, alpha=0.05)
        assert r2.power >= power_target


class TestPowerTTestValidation:
    """Input validation."""

    def test_all_given(self):
        with pytest.raises(ValueError, match="Exactly one"):
            power_t_test(n=50, d=0.5, alpha=0.05, power=0.80)

    def test_two_none(self):
        with pytest.raises(ValueError, match="Exactly one"):
            power_t_test(alpha=0.05)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            power_t_test(d=0.5, alpha=1.5, power=0.80)

    def test_invalid_power(self):
        with pytest.raises(ValueError, match="power"):
            power_t_test(n=50, d=0.5, alpha=0.05, power=1.5)

    def test_zero_d(self):
        with pytest.raises(ValueError, match="d = 0"):
            power_t_test(d=0.0, alpha=0.05, power=0.80)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="type"):
            power_t_test(d=0.5, alpha=0.05, power=0.80, type="invalid")

    def test_invalid_alternative(self):
        with pytest.raises(ValueError, match="alternative"):
            power_t_test(d=0.5, alpha=0.05, power=0.80, alternative="invalid")


class TestPowerTTestSummary:
    """Summary output."""

    def test_summary_contains_key_fields(self):
        r = power_t_test(d=0.5, alpha=0.05, power=0.80)
        s = r.summary()
        assert "n = " in s
        assert "effect size" in s
        assert "power" in s
        assert "alpha" in s
        assert "two.sided" in s


class TestPowerPairedTTest:
    """power_paired_t_test delegates to power_t_test(type='paired')."""

    def test_matches_t_test_paired(self):
        r1 = power_paired_t_test(d=0.5, alpha=0.05, power=0.80)
        r2 = power_t_test(d=0.5, alpha=0.05, power=0.80, type="paired")
        assert r1.n == r2.n
        assert r1.power == r2.power
