"""Power calculations for ANOVA designs.

Validates against: R pwr::pwr.anova.test(), pwr::pwr.f2.test()
"""

from __future__ import annotations

import math

from scipy.stats import f as f_dist
from scipy.stats import ncf

from pystatsbio.power._common import PowerResult, _check_power_args, _solve_parameter

# ---------------------------------------------------------------------------
# Internal power computation
# ---------------------------------------------------------------------------

def _anova_power(
    n: float,
    f_effect: float,
    k: int,
    alpha: float,
) -> float:
    """Compute power for one-way balanced ANOVA.

    Parameters
    ----------
    n : float
        Sample size per group (may be fractional during root-finding).
    f_effect : float
        Cohen's f effect size.
    k : int
        Number of groups.
    alpha : float
        Significance level.
    """
    df1 = k - 1
    df2 = k * (n - 1.0)

    if df2 < 1.0:
        return 0.0

    ncp = n * k * f_effect ** 2
    f_crit = f_dist.ppf(1.0 - alpha, df1, df2)
    pwr = float(ncf.sf(f_crit, df1, df2, ncp))

    # Guard against NaN for extreme ncp
    if math.isnan(pwr):
        pwr = 1.0 if ncp > 50.0 else 0.0

    return pwr


def _factorial_power(
    n: float,
    f_effect: float,
    n_levels: tuple[int, ...],
    alpha: float,
    df_num: int,
) -> float:
    """Compute power for a factorial ANOVA effect.

    Parameters
    ----------
    n : float
        Sample size per cell.
    f_effect : float
        Cohen's f for the target effect.
    n_levels : tuple of int
        Number of levels per factor.
    alpha : float
        Significance level.
    df_num : int
        Numerator degrees of freedom for the target effect.
    """
    total_cells = math.prod(n_levels)
    df_den = total_cells * (n - 1.0)

    if df_den < 1.0:
        return 0.0

    ncp = n * total_cells * f_effect ** 2
    f_crit = f_dist.ppf(1.0 - alpha, df_num, df_den)
    pwr = float(ncf.sf(f_crit, df_num, df_den, ncp))

    if math.isnan(pwr):
        pwr = 1.0 if ncp > 50.0 else 0.0

    return pwr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def power_anova_oneway(
    n: int | None = None,
    f: float | None = None,
    k: int = 2,
    alpha: float = 0.05,
    power: float | None = None,
) -> PowerResult:
    """Power calculation for one-way ANOVA (balanced design).

    Exactly one of ``n``, ``f``, ``power`` must be ``None``.

    Parameters
    ----------
    n : int or None
        Sample size per group.
    f : float or None
        Cohen's f effect size.
    k : int
        Number of groups (default 2).
    alpha : float
        Significance level (default 0.05).
    power : float or None
        Desired power.

    Returns
    -------
    PowerResult

    Examples
    --------
    >>> r = power_anova_oneway(f=0.25, k=3, alpha=0.05, power=0.80)
    >>> r.n  # per group
    52

    Validates against: R pwr::pwr.anova.test()
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    solve_for = _check_power_args(n=n, effect=f, power=power, alpha=alpha, effect_name="f")

    if solve_for == "power":
        assert n is not None and f is not None
        result_power = _anova_power(float(n), f, k, alpha)
        result_n = n
        result_f = f

    elif solve_for == "n":
        assert f is not None and power is not None
        if f == 0.0:
            raise ValueError("Cannot solve for n when f = 0 (no effect)")
        raw_n = _solve_parameter(
            func=lambda x: _anova_power(x, f, k, alpha),
            target=power,
            bracket=(2.0, 1e7),
        )
        result_n = math.ceil(raw_n)
        result_power = power
        result_f = f

    else:  # solve_for == "effect"
        assert n is not None and power is not None
        result_f = _solve_parameter(
            func=lambda x: _anova_power(float(n), x, k, alpha),
            target=power,
            bracket=(1e-10, 100.0),
        )
        result_n = n
        result_power = power

    return PowerResult(
        n=result_n,
        power=result_power,
        effect_size=result_f,
        alpha=alpha,
        alternative="one.sided",  # F-test is inherently one-sided
        method=f"Balanced one-way analysis of variance power calculation (k = {k})",
        note="n is number in each group",
    )


def power_anova_factorial(
    n: int | None = None,
    f: float | None = None,
    n_levels: tuple[int, ...] = (2, 2),
    alpha: float = 0.05,
    power: float | None = None,
    effect: str = "interaction",
) -> PowerResult:
    """Power calculation for factorial ANOVA.

    Exactly one of ``n``, ``f``, ``power`` must be ``None``.

    Parameters
    ----------
    n : int or None
        Sample size per cell.
    f : float or None
        Cohen's f effect size for the target effect.
    n_levels : tuple of int
        Number of levels for each factor, e.g. ``(2, 3)`` for a 2x3 design.
    alpha : float
        Significance level (default 0.05).
    power : float or None
        Desired power.
    effect : str
        Which effect: ``'interaction'``, ``'main_A'``, ``'main_B'``, etc.

    Returns
    -------
    PowerResult

    Examples
    --------
    >>> r = power_anova_factorial(f=0.25, n_levels=(2, 3), alpha=0.05, power=0.80)
    >>> r.n  # per cell
    36

    Validates against: R pwr::pwr.f2.test() (via df conversion)
    """
    if len(n_levels) < 2:
        raise ValueError("n_levels must have at least 2 factors")
    for i, lev in enumerate(n_levels):
        if lev < 2:
            raise ValueError(f"Factor {i} must have >= 2 levels, got {lev}")

    # Determine numerator df for the target effect
    if effect == "interaction":
        df_num = math.prod(lev - 1 for lev in n_levels)
    elif effect.startswith("main_"):
        factor_letter = effect[-1].upper()
        factor_idx = ord(factor_letter) - ord("A")
        if factor_idx < 0 or factor_idx >= len(n_levels):
            raise ValueError(
                f"Invalid effect {effect!r}: factor index {factor_idx} "
                f"out of range for {len(n_levels)} factors"
            )
        df_num = n_levels[factor_idx] - 1
    else:
        raise ValueError(
            f"effect must be 'interaction' or 'main_A', 'main_B', etc., got {effect!r}"
        )

    solve_for = _check_power_args(n=n, effect=f, power=power, alpha=alpha, effect_name="f")

    if solve_for == "power":
        assert n is not None and f is not None
        result_power = _factorial_power(float(n), f, n_levels, alpha, df_num)
        result_n = n
        result_f = f

    elif solve_for == "n":
        assert f is not None and power is not None
        if f == 0.0:
            raise ValueError("Cannot solve for n when f = 0")
        raw_n = _solve_parameter(
            func=lambda x: _factorial_power(x, f, n_levels, alpha, df_num),
            target=power,
            bracket=(2.0, 1e7),
        )
        result_n = math.ceil(raw_n)
        result_power = power
        result_f = f

    else:  # solve_for == "effect"
        assert n is not None and power is not None
        result_f = _solve_parameter(
            func=lambda x: _factorial_power(float(n), x, n_levels, alpha, df_num),
            target=power,
            bracket=(1e-10, 100.0),
        )
        result_n = n
        result_power = power

    design_str = "x".join(str(lev) for lev in n_levels)

    return PowerResult(
        n=result_n,
        power=result_power,
        effect_size=result_f,
        alpha=alpha,
        alternative="one.sided",
        method=f"Factorial ANOVA power calculation ({design_str} design, {effect})",
        note="n is number in each cell",
    )
