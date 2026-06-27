"""Power calculations for t-tests (two-sample, one-sample, paired).

Validates against: R pwr::pwr.t.test()
"""

from __future__ import annotations

import math

from pystatistics.core.exceptions import ValidationError
from scipy.stats import nct, norm
from scipy.stats import t as t_dist

from pystatsbio.power._common import PowerResult, _check_power_args, _solve_parameter

_VALID_TYPES = ("two-sample", "one-sample", "paired")
_VALID_ALTERNATIVES = ("two-sided", "less", "greater")


# ---------------------------------------------------------------------------
# Internal power computation — also used by _cluster.py
# ---------------------------------------------------------------------------

def _normal_approx_power(
    ncp: float,
    alpha: float,
    alternative: str,
) -> float:
    """Normal approximation to noncentral t power (exact as df -> inf)."""
    if alternative == "two-sided":
        z_crit = norm.ppf(1.0 - alpha / 2.0)
        return float(norm.sf(z_crit - ncp) + norm.cdf(-z_crit - ncp))
    elif alternative == "greater":
        z_crit = norm.ppf(1.0 - alpha)
        return float(norm.sf(z_crit - ncp))
    else:  # less
        z_crit = norm.ppf(alpha)
        return float(norm.cdf(z_crit - ncp))


def _t_test_power(
    n: float,
    d: float,
    alpha: float,
    alternative: str,
    type: str,
) -> float:
    """Compute t-test power for given n, d, alpha.

    Parameters
    ----------
    n : float
        Sample size (per group for two-sample, total for one-sample/paired).
        May be non-integer during root-finding.
    d : float
        Cohen's d (positive = treatment > control).
    alpha : float
        Significance level.
    alternative : str
        'two-sided', 'less', or 'greater'.
    type : str
        'two-sample', 'one-sample', or 'paired'.

    Returns
    -------
    float
        Statistical power in [0, 1].
    """
    # Noncentrality parameter and degrees of freedom
    if type == "two-sample":
        ncp = d * math.sqrt(n / 2.0)
        df = 2.0 * n - 2.0
    else:  # one-sample or paired
        ncp = d * math.sqrt(n)
        df = n - 1.0

    if df < 1.0:
        return 0.0

    # For very large df, go straight to normal approximation (exact in limit).
    if df > 1e5:
        return _normal_approx_power(ncp, alpha, alternative)

    # Power via noncentral t distribution
    if alternative == "two-sided":
        t_crit = t_dist.ppf(1.0 - alpha / 2.0, df)
        # P(|T| > t_crit) under H1
        pwr = float(nct.sf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp))
    elif alternative == "greater":
        t_crit = t_dist.ppf(1.0 - alpha, df)
        pwr = float(nct.sf(t_crit, df, ncp))
    else:  # less
        t_crit = t_dist.ppf(alpha, df)
        pwr = float(nct.cdf(t_crit, df, ncp))

    # scipy's nct can return NaN for moderate-to-large noncentrality params
    # (known issue with ncp ~10+ even at moderate df ~1000).  Fall back to
    # the normal approximation which is very accurate for df > ~30.
    if math.isnan(pwr):
        pwr = _normal_approx_power(ncp, alpha, alternative)

    return pwr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def power_t_test(
    n: int | None = None,
    effect_size: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    alternative: str = "two-sided",
    test_type: str = "two-sample",
) -> PowerResult:
    """Power calculation for t-tests.

    Exactly one of ``n``, ``effect_size``, ``power`` must be ``None`` — that
    parameter is solved for given the others.

    Parameters
    ----------
    n : int or None
        Sample size per group (two-sample) or total (one-sample/paired).
    effect_size : float or None
        Cohen's d effect size.
    alpha : float
        Significance level (default 0.05).
    power : float or None
        Desired statistical power (1 - beta).
    alternative : str
        ``'two-sided'``, ``'less'``, or ``'greater'``.
    test_type : str
        ``'two-sample'``, ``'one-sample'``, or ``'paired'``.

    Returns
    -------
    PowerResult

    Examples
    --------
    >>> r = power_t_test(effect_size=0.5, alpha=0.05, power=0.80)
    >>> r.n
    64
    >>> r = power_t_test(n=50, effect_size=0.5, alpha=0.05)
    >>> round(r.power, 3)
    0.697

    Validates against: R pwr::pwr.t.test()
    """
    # --- Validate ---
    if alternative not in _VALID_ALTERNATIVES:
        raise ValidationError(
            f"alternative must be one of {_VALID_ALTERNATIVES}, got {alternative!r}"
        )
    if test_type not in _VALID_TYPES:
        raise ValidationError(
            f"test_type must be one of {_VALID_TYPES}, got {test_type!r}"
        )

    solve_for = _check_power_args(
        n=n, effect=effect_size, power=power, alpha=alpha, effect_name="effect_size"
    )

    # For two-sided tests, R's pwr works with |effect_size|
    d_internal = effect_size
    if effect_size is not None and alternative == "two-sided":
        d_internal = abs(effect_size)

    # --- Solve ---
    if solve_for == "power":
        assert n is not None and d_internal is not None
        result_power = _t_test_power(float(n), d_internal, alpha, alternative, test_type)
        result_n = n
        result_effect_size = effect_size

    elif solve_for == "n":
        assert d_internal is not None and power is not None
        if d_internal == 0.0:
            raise ValidationError(
                "Cannot solve for n when effect_size = 0 (no effect)"
            )
        raw_n = _solve_parameter(
            func=lambda x: _t_test_power(x, d_internal, alpha, alternative, test_type),
            target=power,
            bracket=(2.0, 1e7),
        )
        result_n = math.ceil(raw_n)
        result_power = power
        result_effect_size = effect_size

    else:  # solve_for == "effect"
        assert n is not None and power is not None
        if alternative == "two-sided":
            # Solve for positive effect_size (symmetric)
            result_effect_size = _solve_parameter(
                func=lambda x: _t_test_power(float(n), x, alpha, alternative, test_type),
                target=power,
                bracket=(1e-10, 100.0),
            )
        elif alternative == "greater":
            result_effect_size = _solve_parameter(
                func=lambda x: _t_test_power(float(n), x, alpha, alternative, test_type),
                target=power,
                bracket=(1e-10, 100.0),
            )
        else:  # less
            result_effect_size = _solve_parameter(
                func=lambda x: _t_test_power(float(n), x, alpha, alternative, test_type),
                target=power,
                bracket=(-100.0, -1e-10),
            )
        result_n = n
        result_power = power

    # --- Method label ---
    type_labels = {
        "two-sample": "Two-sample",
        "one-sample": "One-sample",
        "paired": "Paired",
    }
    method = f"{type_labels[test_type]} t test power calculation"
    note = "n is number in *each* group" if test_type == "two-sample" else ""

    return PowerResult(
        n=result_n,
        power=result_power,
        effect_size=result_effect_size,
        alpha=alpha,
        alternative=alternative,
        method=method,
        note=note,
    )


def power_paired_t_test(
    n: int | None = None,
    effect_size: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    alternative: str = "two-sided",
) -> PowerResult:
    """Convenience wrapper: ``power_t_test`` with ``test_type='paired'``.

    Validates against: R pwr::pwr.t.test(type='paired')
    """
    return power_t_test(
        n=n,
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative=alternative,
        test_type="paired",
    )
