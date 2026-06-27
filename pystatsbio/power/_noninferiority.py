"""Power calculations for non-inferiority, equivalence, and superiority designs.

All four functions use the normal approximation for two-sample comparisons
with a margin-shifted hypothesis.

Validates against: R TrialSize, PowerTOST, TOSTER
"""

from __future__ import annotations

import math

from pystatistics.core.exceptions import ValidationError
from scipy.stats import norm

from pystatsbio.power._common import PowerResult, _check_power_args, _solve_parameter

# ---------------------------------------------------------------------------
# Non-inferiority for means
# ---------------------------------------------------------------------------

def _noninf_mean_power(
    n: float,
    delta: float,
    margin: float,
    std: float,
    alpha: float,
) -> float:
    """Power for NI test of means.

    H0: delta <= -margin  vs  H1: delta > -margin
    Power = Phi((delta + margin) / SE - z_alpha)
    where SE = std * sqrt(2/n)
    """
    se = std * math.sqrt(2.0 / n)
    z_alpha = norm.ppf(1.0 - alpha)
    z_effect = (delta + margin) / se
    return float(norm.cdf(z_effect - z_alpha))


def power_noninf_mean(
    n: int | None = None,
    delta: float | None = None,
    margin: float = 0.0,
    std: float = 1.0,
    alpha: float = 0.025,
    power: float | None = None,
    alternative: str = "one-sided",
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
    std : float
        Common standard deviation (> 0).
    alpha : float
        One-sided significance level (default 0.025).
    power : float or None
        Desired power.
    alternative : str
        ``'one-sided'`` (standard for NI trials).

    Returns
    -------
    PowerResult

    Validates against: R TrialSize::TwoSampleMean.NIS()
    """
    if margin < 0:
        raise ValidationError(f"margin must be >= 0, got {margin}")
    if std <= 0:
        raise ValidationError(f"std must be > 0, got {std}")

    solve_for = _check_power_args(
        n=n, effect=delta, power=power, alpha=alpha, effect_name="delta",
    )

    if solve_for == "power":
        assert n is not None and delta is not None
        result_power = _noninf_mean_power(float(n), delta, margin, std, alpha)
        result_n, result_delta = n, delta

    elif solve_for == "n":
        assert delta is not None and power is not None
        raw_n = _solve_parameter(
            func=lambda x: _noninf_mean_power(x, delta, margin, std, alpha),
            target=power,
            bracket=(2.0, 1e7),
        )
        result_n = math.ceil(raw_n)
        result_power, result_delta = power, delta

    else:  # effect (delta)
        assert n is not None and power is not None
        result_delta = _solve_parameter(
            func=lambda x: _noninf_mean_power(float(n), x, margin, std, alpha),
            target=power,
            bracket=(-100.0 * std, 100.0 * std),
        )
        result_n, result_power = n, power

    return PowerResult(
        n=result_n,
        power=result_power,
        effect_size=result_delta,
        alpha=alpha,
        alternative=alternative,
        method="Non-inferiority test of means power calculation",
        note=f"n is per group; margin = {margin}; std = {std}",
    )


# ---------------------------------------------------------------------------
# Non-inferiority for proportions
# ---------------------------------------------------------------------------

def _noninf_prop_power(
    n: float,
    prop1: float,
    prop2: float,
    margin: float,
    alpha: float,
) -> float:
    """Power for NI test of proportions.

    H0: prop1 - prop2 <= -margin  vs  H1: prop1 - prop2 > -margin
    """
    delta = prop1 - prop2
    se = math.sqrt((prop1 * (1.0 - prop1) + prop2 * (1.0 - prop2)) / n)
    z_alpha = norm.ppf(1.0 - alpha)
    z_effect = (delta + margin) / se
    return float(norm.cdf(z_effect - z_alpha))


def power_noninf_prop(
    n: int | None = None,
    prop1: float | None = None,
    prop2: float | None = None,
    margin: float = 0.0,
    alpha: float = 0.025,
    power: float | None = None,
) -> PowerResult:
    """Power for non-inferiority test of proportions.

    Exactly one of ``n``, ``power`` must be ``None`` (``prop1`` and ``prop2``
    are always required).

    Parameters
    ----------
    n : int or None
        Sample size per group.
    prop1 : float
        Expected proportion in treatment group.
    prop2 : float
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
    if prop1 is None or prop2 is None:
        raise ValidationError("prop1 and prop2 are always required")
    if not (0.0 < prop1 < 1.0):
        raise ValidationError(f"prop1 must be in (0, 1), got {prop1}")
    if not (0.0 < prop2 < 1.0):
        raise ValidationError(f"prop2 must be in (0, 1), got {prop2}")
    if margin < 0:
        raise ValidationError(f"margin must be >= 0, got {margin}")

    # We solve for n or power; prop1/prop2 are always provided
    none_count = sum(x is None for x in (n, power))
    if none_count != 1:
        raise ValidationError("Exactly one of n, power must be None")
    if not (0.0 < alpha < 1.0):
        raise ValidationError(f"alpha must be in (0, 1), got {alpha}")
    if power is not None and not (0.0 < power < 1.0):
        raise ValidationError(f"power must be in (0, 1), got {power}")
    if n is not None and n < 2:
        raise ValidationError(f"n must be >= 2, got {n}")

    delta = prop1 - prop2

    if n is None:
        # Solve for n
        assert power is not None
        raw_n = _solve_parameter(
            func=lambda x: _noninf_prop_power(x, prop1, prop2, margin, alpha),
            target=power,
            bracket=(2.0, 1e7),
        )
        result_n = math.ceil(raw_n)
        result_power = power
    else:
        # Solve for power
        result_power = _noninf_prop_power(float(n), prop1, prop2, margin, alpha)
        result_n = n

    return PowerResult(
        n=result_n,
        power=result_power,
        effect_size=delta,
        alpha=alpha,
        alternative="one-sided",
        method="Non-inferiority test of proportions power calculation",
        note=f"n is per group; margin = {margin}; prop1 = {prop1}, prop2 = {prop2}",
    )


# ---------------------------------------------------------------------------
# Equivalence (TOST) for means
# ---------------------------------------------------------------------------

def _equiv_mean_power(
    n: float,
    delta: float,
    margin: float,
    std: float,
    alpha: float,
) -> float:
    """Power for equivalence (TOST) of means.

    Bounds: (-margin, +margin)
    Power = Phi((delta + margin)/SE - z_alpha) + Phi((-delta + margin)/SE - z_alpha) - 1
    """
    se = std * math.sqrt(2.0 / n)
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
    std: float = 1.0,
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
    std : float
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
        raise ValidationError(f"margin must be >= 0, got {margin}")
    if std <= 0:
        raise ValidationError(f"std must be > 0, got {std}")

    solve_for = _check_power_args(
        n=n, effect=delta, power=power, alpha=alpha, effect_name="delta",
    )

    if solve_for == "power":
        assert n is not None and delta is not None
        result_power = _equiv_mean_power(float(n), delta, margin, std, alpha)
        result_n, result_delta = n, delta

    elif solve_for == "n":
        assert delta is not None and power is not None
        raw_n = _solve_parameter(
            func=lambda x: _equiv_mean_power(x, delta, margin, std, alpha),
            target=power,
            bracket=(2.0, 1e7),
        )
        result_n = math.ceil(raw_n)
        result_power, result_delta = power, delta

    else:  # effect (delta)
        assert n is not None and power is not None
        # delta can be any value in (-margin, margin) for equivalence to hold
        result_delta = _solve_parameter(
            func=lambda x: _equiv_mean_power(float(n), x, margin, std, alpha),
            target=power,
            bracket=(-margin + 1e-10, margin - 1e-10),
        )
        result_n, result_power = n, power

    return PowerResult(
        n=result_n,
        power=result_power,
        effect_size=result_delta,
        alpha=alpha,
        alternative="two-sided",
        method="Equivalence test (TOST) of means power calculation",
        note=f"n is per group; margin = {margin}; std = {std}",
    )


# ---------------------------------------------------------------------------
# Superiority for means
# ---------------------------------------------------------------------------

def _superiority_mean_power(
    n: float,
    delta: float,
    margin: float,
    std: float,
    alpha: float,
) -> float:
    """Power for superiority test of means.

    H0: delta <= margin  vs  H1: delta > margin
    Power = Phi((delta - margin) / SE - z_alpha)
    """
    se = std * math.sqrt(2.0 / n)
    z_alpha = norm.ppf(1.0 - alpha)
    z_effect = (delta - margin) / se
    return float(norm.cdf(z_effect - z_alpha))


def power_superiority_mean(
    n: int | None = None,
    delta: float | None = None,
    margin: float = 0.0,
    std: float = 1.0,
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
    std : float
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
        raise ValidationError(f"margin must be >= 0, got {margin}")
    if std <= 0:
        raise ValidationError(f"std must be > 0, got {std}")

    solve_for = _check_power_args(
        n=n, effect=delta, power=power, alpha=alpha, effect_name="delta",
    )

    if solve_for == "power":
        assert n is not None and delta is not None
        result_power = _superiority_mean_power(float(n), delta, margin, std, alpha)
        result_n, result_delta = n, delta

    elif solve_for == "n":
        assert delta is not None and power is not None
        raw_n = _solve_parameter(
            func=lambda x: _superiority_mean_power(x, delta, margin, std, alpha),
            target=power,
            bracket=(2.0, 1e7),
        )
        result_n = math.ceil(raw_n)
        result_power, result_delta = power, delta

    else:  # effect (delta)
        assert n is not None and power is not None
        result_delta = _solve_parameter(
            func=lambda x: _superiority_mean_power(float(n), x, margin, std, alpha),
            target=power,
            bracket=(margin + 1e-10, margin + 100.0 * std),
        )
        result_n, result_power = n, power

    return PowerResult(
        n=result_n,
        power=result_power,
        effect_size=result_delta,
        alpha=alpha,
        alternative="one-sided",
        method="Superiority test of means power calculation",
        note=f"n is per group; margin = {margin}; std = {std}",
    )
