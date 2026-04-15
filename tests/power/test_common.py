"""Tests for power/_common.py: PowerResult, _check_power_args, _solve_parameter."""

import math

import pytest

from pystatsbio.power._common import PowerResult, _check_power_args, _solve_parameter

# ---------------------------------------------------------------------------
# PowerResult
# ---------------------------------------------------------------------------

class TestPowerResult:
    """PowerResult dataclass and summary()."""

    def _make(self, **kwargs):
        defaults = dict(
            n=64, power=0.80, effect_size=0.5,
            alpha=0.05, alternative="two.sided",
            method="Two-sample t-test power calculation",
        )
        defaults.update(kwargs)
        return PowerResult(**defaults)

    def test_fields_stored(self):
        r = self._make()
        assert r.n == 64
        assert r.power == pytest.approx(0.80)
        assert r.effect_size == pytest.approx(0.5)
        assert r.alpha == pytest.approx(0.05)
        assert r.alternative == "two.sided"

    def test_frozen(self):
        r = self._make()
        with pytest.raises(AttributeError):
            r.n = 100  # type: ignore[misc]

    def test_note_default_empty(self):
        r = self._make()
        assert r.note == ""

    def test_note_stored(self):
        r = self._make(note="n is per group")
        assert "per group" in r.note

    def test_summary_contains_n(self):
        r = self._make()
        assert "n = 64" in r.summary()

    def test_summary_contains_power(self):
        r = self._make()
        assert "power" in r.summary()

    def test_summary_contains_alpha(self):
        r = self._make()
        assert "alpha" in r.summary()

    def test_summary_contains_alternative(self):
        r = self._make()
        assert "two.sided" in r.summary()

    def test_summary_contains_effect(self):
        r = self._make()
        assert "effect size" in r.summary()

    def test_summary_contains_note(self):
        r = self._make(note="per group; sd = 1.0")
        assert "per group" in r.summary()

    def test_summary_none_n_omitted(self):
        r = self._make(n=None)
        s = r.summary()
        assert "n = " not in s

    def test_summary_none_effect_omitted(self):
        r = self._make(effect_size=None)
        s = r.summary()
        assert "effect size" not in s


# ---------------------------------------------------------------------------
# _check_power_args
# ---------------------------------------------------------------------------

class TestCheckPowerArgs:
    """_check_power_args returns correct solve-for and validates inputs."""

    def test_returns_n_when_n_is_none(self):
        result = _check_power_args(n=None, effect=0.5, power=0.80, alpha=0.05)
        assert result == "n"

    def test_returns_effect_when_effect_is_none(self):
        result = _check_power_args(n=64, effect=None, power=0.80, alpha=0.05)
        assert result == "effect"

    def test_returns_power_when_power_is_none(self):
        result = _check_power_args(n=64, effect=0.5, power=None, alpha=0.05)
        assert result == "power"

    def test_error_all_provided(self):
        with pytest.raises(ValueError, match="Exactly one"):
            _check_power_args(n=64, effect=0.5, power=0.80, alpha=0.05)

    def test_error_two_none(self):
        with pytest.raises(ValueError, match="Exactly one"):
            _check_power_args(n=None, effect=None, power=0.80, alpha=0.05)

    def test_error_all_none(self):
        with pytest.raises(ValueError, match="Exactly one"):
            _check_power_args(n=None, effect=None, power=None, alpha=0.05)

    def test_invalid_alpha_above_1(self):
        with pytest.raises(ValueError, match="alpha"):
            _check_power_args(n=None, effect=0.5, power=0.80, alpha=1.5)

    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha"):
            _check_power_args(n=None, effect=0.5, power=0.80, alpha=0.0)

    def test_invalid_n_less_than_2(self):
        with pytest.raises(ValueError, match="n must be"):
            _check_power_args(n=1, effect=0.5, power=None, alpha=0.05)

    def test_invalid_power_above_1(self):
        with pytest.raises(ValueError, match="power"):
            _check_power_args(n=None, effect=0.5, power=1.5, alpha=0.05)

    def test_invalid_power_zero(self):
        with pytest.raises(ValueError, match="power"):
            _check_power_args(n=None, effect=0.5, power=0.0, alpha=0.05)

    def test_invalid_effect_inf(self):
        with pytest.raises(ValueError, match="effect_size"):
            _check_power_args(n=64, effect=math.inf, power=None, alpha=0.05)

    def test_invalid_effect_nan(self):
        with pytest.raises(ValueError, match="effect_size"):
            _check_power_args(n=64, effect=float("nan"), power=None, alpha=0.05)

    def test_custom_effect_name_in_error(self):
        with pytest.raises(ValueError, match="delta"):
            _check_power_args(
                n=64, effect=float("nan"), power=None,
                alpha=0.05, effect_name="delta",
            )

    def test_n_exactly_2_is_valid(self):
        # n=2 is the boundary — should not raise
        result = _check_power_args(n=2, effect=0.5, power=None, alpha=0.05)
        assert result == "power"


# ---------------------------------------------------------------------------
# _solve_parameter
# ---------------------------------------------------------------------------

class TestSolveParameter:
    """_solve_parameter solves f(x) == target via Brent's method."""

    def test_linear_function(self):
        # f(x) = x, target = 3.0 -> x = 3.0
        x = _solve_parameter(lambda x: x, target=3.0, bracket=(0.0, 10.0))
        assert x == pytest.approx(3.0, abs=1e-8)

    def test_quadratic_function(self):
        # f(x) = x^2, target = 4.0 -> x = 2.0 (in [0, 5])
        x = _solve_parameter(lambda x: x**2, target=4.0, bracket=(0.0, 5.0))
        assert x == pytest.approx(2.0, abs=1e-8)

    def test_monotone_power_function(self):
        # Increasing function: should converge
        from scipy.stats import norm
        # Normal CDF is monotone; solve for x such that Phi(x) = 0.975
        x = _solve_parameter(
            lambda x: float(norm.cdf(x)),
            target=0.975,
            bracket=(-5.0, 5.0),
        )
        assert x == pytest.approx(1.96, abs=0.01)

    def test_invalid_bracket_raises(self):
        # f(x) = x is always positive; target = -1 is outside [0, 10]
        with pytest.raises(ValueError, match="outside achievable range"):
            _solve_parameter(lambda x: x, target=-1.0, bracket=(0.0, 10.0))

    def test_bracket_same_sign_raises(self):
        # Both endpoints give values > target
        with pytest.raises(ValueError, match="outside achievable range"):
            _solve_parameter(lambda x: x + 10, target=0.0, bracket=(1.0, 5.0))
