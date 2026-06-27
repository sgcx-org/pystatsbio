"""Power calculations for proportion tests.

Validates against: R pwr::pwr.2p.test()
"""

from __future__ import annotations

import math

from pystatistics.core.exceptions import ValidationError
from scipy.stats import norm

from pystatsbio.power._common import PowerResult, _check_power_args, _solve_parameter

_VALID_ALTERNATIVES = ("two.sided", "less", "greater")


# ---------------------------------------------------------------------------
# Internal power computation
# ---------------------------------------------------------------------------

def _prop_power(
    n: float,
    h: float,
    alpha: float,
    alternative: str,
) -> float:
    """Compute power for the two-proportion z-test using Cohen's h.

    Uses the normal approximation: the test statistic under H1 is
    approximately N(h * sqrt(n/2), 1).
    """
    z_effect = h * math.sqrt(n / 2.0)

    if alternative == "two.sided":
        z_crit = norm.ppf(1.0 - alpha / 2.0)
        pwr = float(norm.sf(z_crit - z_effect) + norm.cdf(-z_crit - z_effect))
    elif alternative == "greater":
        z_crit = norm.ppf(1.0 - alpha)
        pwr = float(norm.sf(z_crit - z_effect))
    else:  # less
        z_crit = norm.ppf(alpha)
        pwr = float(norm.cdf(z_crit - z_effect))

    return pwr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def power_prop_test(
    n: int | None = None,
    h: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    alternative: str = "two.sided",
) -> PowerResult:
    """Power calculation for two-proportion z-test (chi-squared test).

    Exactly one of ``n``, ``h``, ``power`` must be ``None``.

    Parameters
    ----------
    n : int or None
        Sample size per group.
    h : float or None
        Cohen's h effect size: ``h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))``.
    alpha : float
        Significance level (default 0.05).
    power : float or None
        Desired power.
    alternative : str
        ``'two.sided'``, ``'less'``, or ``'greater'``.

    Returns
    -------
    PowerResult

    Examples
    --------
    >>> import math
    >>> h = 2 * (math.asin(math.sqrt(0.5)) - math.asin(math.sqrt(0.3)))
    >>> r = power_prop_test(h=h, alpha=0.05, power=0.80)
    >>> r.n  # ~93
    93

    Validates against: R pwr::pwr.2p.test()
    """
    if alternative not in _VALID_ALTERNATIVES:
        raise ValidationError(
            f"alternative must be one of {_VALID_ALTERNATIVES}, got {alternative!r}"
        )

    solve_for = _check_power_args(n=n, effect=h, power=power, alpha=alpha, effect_name="h")

    # For two-sided, R uses |h|
    h_internal = h
    if h is not None and alternative == "two.sided":
        h_internal = abs(h)

    if solve_for == "power":
        assert n is not None and h_internal is not None
        result_power = _prop_power(float(n), h_internal, alpha, alternative)
        result_n = n
        result_h = h

    elif solve_for == "n":
        assert h_internal is not None and power is not None
        if h_internal == 0.0:
            raise ValidationError("Cannot solve for n when h = 0 (no effect)")
        raw_n = _solve_parameter(
            func=lambda x: _prop_power(x, h_internal, alpha, alternative),
            target=power,
            bracket=(2.0, 1e7),
        )
        result_n = math.ceil(raw_n)
        result_power = power
        result_h = h

    else:  # solve_for == "effect"
        assert n is not None and power is not None
        if alternative == "two.sided" or alternative == "greater":
            result_h = _solve_parameter(
                func=lambda x: _prop_power(float(n), x, alpha, alternative),
                target=power,
                bracket=(1e-10, 100.0),
            )
        else:  # less
            result_h = _solve_parameter(
                func=lambda x: _prop_power(float(n), x, alpha, alternative),
                target=power,
                bracket=(-100.0, -1e-10),
            )
        result_n = n
        result_power = power

    return PowerResult(
        n=result_n,
        power=result_power,
        effect_size=result_h,
        alpha=alpha,
        alternative=alternative,
        method="Difference of two proportions power calculation",
        note="n is number in *each* group",
    )


def power_fisher_test(
    n: int | None = None,
    p1: float | None = None,
    p2: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    alternative: str = "two.sided",
) -> PowerResult:
    """Power calculation for Fisher's exact test (normal approximation).

    Uses the arcsine (Cohen's h) normal approximation. For exact power
    computation via hypergeometric enumeration, a future version will
    add ``method='exact'``.

    Exactly one of ``n``, ``power`` must be ``None`` (``p1`` and ``p2``
    are always required).

    Parameters
    ----------
    n : int or None
        Sample size per group.
    p1 : float
        Probability in group 1.
    p2 : float
        Probability in group 2.
    alpha : float
        Significance level.
    power : float or None
        Desired power.
    alternative : str
        ``'two.sided'``, ``'less'``, or ``'greater'``.

    Returns
    -------
    PowerResult

    Validates against: R pwr::pwr.2p.test() (via Cohen's h)
    """
    if alternative not in _VALID_ALTERNATIVES:
        raise ValidationError(
            f"alternative must be one of {_VALID_ALTERNATIVES}, got {alternative!r}"
        )
    if p1 is None or p2 is None:
        raise ValidationError("p1 and p2 are always required")
    if not (0.0 < p1 < 1.0):
        raise ValidationError(f"p1 must be in (0, 1), got {p1}")
    if not (0.0 < p2 < 1.0):
        raise ValidationError(f"p2 must be in (0, 1), got {p2}")

    # Cohen's h from the two proportions
    h = 2.0 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))

    # Delegate to the proportion power function
    result = power_prop_test(n=n, h=h, alpha=alpha, power=power, alternative=alternative)

    return PowerResult(
        n=result.n,
        power=result.power,
        effect_size=h,
        alpha=alpha,
        alternative=alternative,
        method="Fisher's exact test power calculation (normal approximation)",
        note=f"n is number in *each* group; h = {h:.6f} (Cohen's h from p1={p1}, p2={p2})",
    )
