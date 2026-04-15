"""Power calculations for non-inferiority, equivalence, and superiority designs.

All four functions use the normal approximation for two-sample comparisons
with a margin-shifted hypothesis.

Validates against: R TrialSize, PowerTOST, TOSTER
"""

from __future__ import annotations

import math

from scipy.stats import norm

from pystatsbio.power._common import PowerResult, _check_power_args, _solve_parameter

# ---------------------------------------------------------------------------
# Non-inferiority for means
# ---------------------------------------------------------------------------

def _noninf_mean_power(
    n: float,
    delta: float,
    margin: float,
    sd: float,
    alpha: float,
) -> float:
    """Power for NI test of means.

    H0: delta <= -margin  vs  H1: delta > -margin
    Power = Phi((delta + margin) / SE - z_alpha)
    where SE = sd * sqrt(2/n)
    """
    se = sd * math.sqrt(2.0 / n)
    z_alpha = norm.ppf(1.0 - alpha)
    z_effect = (delta + margin) / se
    return float(norm.cdf(z_effect - z_alpha))


def power_noninf_mean(
    n: int | None = None,
    delta: float | None = None,
    margin: float = 0.0,
    sd: float = 1.0,
    alpha: float = 0.025,
    power: float | None = None,
    alternative: str = "one.sided",
) -> PowerResult:
    """Power for non-inferiority test of means.

    Tests H0: treatment - control <= -margin (treatment is inferior)
    vs   H1: treatment - control > -margin (treatment is non-inferior).

    Exactly one of ``n``, ``delta``, ``power`` must be ``None``.

    Parameters
    ----------
    n : int or None
        Sample size per group.
    delta : float or None
        True difference in means (treatment - control).
    margin : float
        Non-inferiority margin (>= 0).
    sd : float
        Common standard deviation (> 0).
    alpha : float
        One-sided significance level (default 0.025).
    power : float or None
        Desired power.
    alternative : str
        ``'one.sided'`` (standard for NI trials).

    Returns
    -------
    PowerResult

    Validates against: R TrialSize::TwoSampleMean.NIS()
    """
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    if sd <= 0:
        raise ValueError(f"sd must be > 0, got {sd}")

    solve_for = _check_power_args(
        n=n, effect=delta, power=power, alpha=alpha, effect_name="delta",
    )

    if solve_for == "power":
        assert n is not None and delta is not None
        result_power = _noninf_mean_power(float(n), delta, margin, sd, alpha)
        result_n, result_delta = n, delta

    elif solve_for == "n":
        assert delta is not None and power is not None
        raw_n = _solve_parameter(
            func=lambda x: _noninf_mean_power(x, delta, margin, sd, alpha),
            target=power,
            bracket=(2.0, 1e7),
        )
        result_n = math.ceil(raw_n)
        result_power, result_delta = power, delta

    else:  # effect (delta)
        assert n is not None and power is not None
        result_delta = _solve_parameter(
            func=lambda x: _noninf_mean_power(float(n), x, margin, sd, alpha),
            target=power,
            bracket=(-100.0 * sd, 100.0 * sd),
        )
        result_n, result_power = n, power

    return PowerResult(
        n=result_n,
        power=result_power,
        effect_size=result_delta,
        alpha=alpha,
        alternative=alternative,
        method="Non-inferiority test of means power calculation",
        note=f"n is per group; margin = {margin}; sd = {sd}",
    )


# ---------------------------------------------------------------------------
# Non-inferiority for proportions
# ---------------------------------------------------------------------------

def _noninf_prop_power(
    n: float,
    p1: float,
    p2: float,
    margin: float,
    alpha: float,
) -> float:
    """Power for NI test of proportions.

    H0: p1 - p2 <= -margin  vs  H1: p1 - p2 > -margin
    """
    delta = p1 - p2
    se = math.sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / n)
    z_alpha = norm.ppf(1.0 - alpha)
    z_effect = (delta + margin) / se
    return float(norm.cdf(z_effect - z_alpha))


def power_noninf_prop(
    n: int | None = None,
    p1: float | None = None,
    p2: float | None = None,
    margin: float = 0.0,
    alpha: float = 0.025,
    power: float | None = None,
) -> PowerResult:
    """Power for non-inferiority test of proportions.

    Exactly one of ``n``, ``power`` must be ``None`` (``p1`` and ``p2``
    are always required).

    Parameters
    ----------
    n : int or None
        Sample size per group.
    p1 : float
        Expected proportion in treatment group.
    p2 : float
        Expected proportion in control group.
    margin : float
        Non-inferiority margin (>= 0).
    alpha : float
        One-sided significance level (default 0.025).
    power : float or None
        Desired power.

    Returns
    -------
    PowerResult

    Validates against: R TrialSize::TwoSampleProportion.NIS()
    """
    if p1 is None or p2 is None:
        raise ValueError("p1 and p2 are always required")
    if not (0.0 < p1 < 1.0):
        raise ValueError(f"p1 must be in (0, 1), got {p1}")
    if not (0.0 < p2 < 1.0):
        raise ValueError(f"p2 must be in (0, 1), got {p2}")
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")

    # We solve for n or power; p1/p2 are always provided
    none_count = sum(x is None for x in (n, power))
    if none_count != 1:
        raise ValueError("Exactly one of n, power must be None")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if power is not None and not (0.0 < power < 1.0):
        raise ValueError(f"power must be in (0, 1), got {power}")
    if n is not None and n < 2:
        raise ValueError(f"n must be >= 2, got {n}")

    delta = p1 - p2

    if n is None:
        # Solve for n
        assert power is not None
        raw_n = _solve_parameter(
            func=lambda x: _noninf_prop_power(x, p1, p2, margin, alpha),
            target=power,
            bracket=(2.0, 1e7),
        )
        result_n = math.ceil(raw_n)
        result_power = power
    else:
        # Solve for power
        result_power = _noninf_prop_power(float(n), p1, p2, margin, alpha)
        result_n = n

    return PowerResult(
        n=result_n,
        power=result_power,
        effect_size=delta,
        alpha=alpha,
        alternative="one.sided",
        method="Non-inferiority test of proportions power calculation",
        note=f"n is per group; margin = {margin}; p1 = {p1}, p2 = {p2}",
    )


# ---------------------------------------------------------------------------
# Equivalence (TOST) for means
# ---------------------------------------------------------------------------

def _equiv_mean_power(
    n: float,
    delta: float,
    margin: float,
    sd: float,
    alpha: float,
) -> float:
    """Power for equivalence (TOST) of means.

    Bounds: (-margin, +margin)
    Power = Phi((delta + margin)/SE - z_alpha) + Phi((-delta + margin)/SE - z_alpha) - 1
    """
    se = sd * math.sqrt(2.0 / n)
    z_alpha = norm.ppf(1.0 - alpha)
    pwr = (
        norm.cdf((delta + margin) / se - z_alpha)
        + norm.cdf((-delta + margin) / se - z_alpha)
        - 1.0
    )
    return float(max(pwr, 0.0))


def power_equiv_mean(
    n: int | None = None,
    delta: float | None = None,
    margin: float = 0.0,
    sd: float = 1.0,
    alpha: float = 0.05,
    power: float | None = None,
) -> PowerResult:
    """Power for equivalence test (TOST) of means.

    Tests H01: delta <= -margin AND H02: delta >= +margin.
    Reject both -> equivalence.

    Exactly one of ``n``, ``delta``, ``power`` must be ``None``.

    Parameters
    ----------
    n : int or None
        Sample size per group.
    delta : float or None
        True difference in means.
    margin : float
        Equivalence margin (>= 0, symmetric bounds).
    sd : float
        Common standard deviation (> 0).
    alpha : float
        Significance level for each one-sided test (default 0.05).
    power : float or None
        Desired power.

    Returns
    -------
    PowerResult

    Validates against: R PowerTOST::power.TOST(), TOSTER
    """
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    if sd <= 0:
        raise ValueError(f"sd must be > 0, got {sd}")

    solve_for = _check_power_args(
        n=n, effect=delta, power=power, alpha=alpha, effect_name="delta",
    )

    if solve_for == "power":
        assert n is not None and delta is not None
        result_power = _equiv_mean_power(float(n), delta, margin, sd, alpha)
        result_n, result_delta = n, delta

    elif solve_for == "n":
        assert delta is not None and power is not None
        raw_n = _solve_parameter(
            func=lambda x: _equiv_mean_power(x, delta, margin, sd, alpha),
            target=power,
            bracket=(2.0, 1e7),
        )
        result_n = math.ceil(raw_n)
        result_power, result_delta = power, delta

    else:  # effect (delta)
        assert n is not None and power is not None
        # delta can be any value in (-margin, margin) for equivalence to hold
        result_delta = _solve_parameter(
            func=lambda x: _equiv_mean_power(float(n), x, margin, sd, alpha),
            target=power,
            bracket=(-margin + 1e-10, margin - 1e-10),
        )
        result_n, result_power = n, power

    return PowerResult(
        n=result_n,
        power=result_power,
        effect_size=result_delta,
        alpha=alpha,
        alternative="two.sided",
        method="Equivalence test (TOST) of means power calculation",
        note=f"n is per group; margin = {margin}; sd = {sd}",
    )


# ---------------------------------------------------------------------------
# Superiority for means
# ---------------------------------------------------------------------------

def _superiority_mean_power(
    n: float,
    delta: float,
    margin: float,
    sd: float,
    alpha: float,
) -> float:
    """Power for superiority test of means.

    H0: delta <= margin  vs  H1: delta > margin
    Power = Phi((delta - margin) / SE - z_alpha)
    """
    se = sd * math.sqrt(2.0 / n)
    z_alpha = norm.ppf(1.0 - alpha)
    z_effect = (delta - margin) / se
    return float(norm.cdf(z_effect - z_alpha))


def power_superiority_mean(
    n: int | None = None,
    delta: float | None = None,
    margin: float = 0.0,
    sd: float = 1.0,
    alpha: float = 0.025,
    power: float | None = None,
) -> PowerResult:
    """Power for superiority test of means.

    Tests H0: treatment - control <= margin (no superiority)
    vs   H1: treatment - control > margin (treatment is superior).

    Exactly one of ``n``, ``delta``, ``power`` must be ``None``.

    Parameters
    ----------
    n : int or None
        Sample size per group.
    delta : float or None
        True difference in means.
    margin : float
        Superiority margin (>= 0).
    sd : float
        Common standard deviation (> 0).
    alpha : float
        One-sided significance level (default 0.025).
    power : float or None
        Desired power.

    Returns
    -------
    PowerResult

    Validates against: R TrialSize
    """
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    if sd <= 0:
        raise ValueError(f"sd must be > 0, got {sd}")

    solve_for = _check_power_args(
        n=n, effect=delta, power=power, alpha=alpha, effect_name="delta",
    )

    if solve_for == "power":
        assert n is not None and delta is not None
        result_power = _superiority_mean_power(float(n), delta, margin, sd, alpha)
        result_n, result_delta = n, delta

    elif solve_for == "n":
        assert delta is not None and power is not None
        raw_n = _solve_parameter(
            func=lambda x: _superiority_mean_power(x, delta, margin, sd, alpha),
            target=power,
            bracket=(2.0, 1e7),
        )
        result_n = math.ceil(raw_n)
        result_power, result_delta = power, delta

    else:  # effect (delta)
        assert n is not None and power is not None
        result_delta = _solve_parameter(
            func=lambda x: _superiority_mean_power(float(n), x, margin, sd, alpha),
            target=power,
            bracket=(margin + 1e-10, margin + 100.0 * sd),
        )
        result_n, result_power = n, power

    return PowerResult(
        n=result_n,
        power=result_power,
        effect_size=result_delta,
        alpha=alpha,
        alternative="one.sided",
        method="Superiority test of means power calculation",
        note=f"n is per group; margin = {margin}; sd = {sd}",
    )
