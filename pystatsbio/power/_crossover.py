"""Power calculations for crossover designs (bioequivalence).

Standard 2x2 crossover TOST on the log-scale for average bioequivalence.

Validates against: R PowerTOST::sampleN.TOST(), PowerTOST::power.TOST()
"""

from __future__ import annotations

import math

from scipy.stats import nct, norm
from scipy.stats import t as t_dist

from pystatsbio.power._common import PowerResult, _solve_parameter

# ---------------------------------------------------------------------------
# Internal power computation
# ---------------------------------------------------------------------------

def _crossover_be_power(
    n: float,
    cv: float,
    theta0: float,
    theta1: float,
    theta2: float,
    alpha: float,
) -> float:
    """Compute TOST power for 2x2 crossover bioequivalence.

    Parameters
    ----------
    n : float
        Total number of subjects (both sequences).
    cv : float
        Within-subject coefficient of variation.
    theta0 : float
        Assumed true ratio of geometric means.
    theta1, theta2 : float
        Lower and upper bioequivalence limits.
    alpha : float
        Overall significance level (each one-sided test at alpha/2).
    """
    # Within-subject SD on log-scale
    sigma_w = math.sqrt(math.log(1.0 + cv ** 2))

    # Standard error of the log-mean difference in a 2x2 crossover
    se = sigma_w * math.sqrt(2.0 / n)

    # Degrees of freedom for 2x2 crossover
    df = n - 2.0
    if df < 1.0:
        return 0.0

    # Critical t-value (each one-sided test at alpha/2)
    t_crit = t_dist.ppf(1.0 - alpha / 2.0, df)

    # Log-ratios
    ln_theta0 = math.log(theta0)
    ln_theta1 = math.log(theta1)
    ln_theta2 = math.log(theta2)

    # Non-centrality parameters for lower and upper TOST
    ncp_lower = (ln_theta0 - ln_theta1) / se
    ncp_upper = (ln_theta2 - ln_theta0) / se

    # Power = P(reject lower) + P(reject upper) - 1
    # P(reject lower) = P(T > t_crit | ncp_lower)
    # P(reject upper) = P(T > t_crit | ncp_upper)
    # Combined via the intersection: power_lower + power_upper - 1
    if df > 1e5:
        # Normal approximation for very large df
        z_crit = norm.ppf(1.0 - alpha / 2.0)
        power_lower = float(norm.sf(z_crit - ncp_lower))
        power_upper = float(norm.sf(z_crit - ncp_upper))
    else:
        power_lower = float(nct.sf(t_crit, df, ncp_lower))
        power_upper = float(nct.sf(t_crit, df, ncp_upper))

        # scipy nct can return NaN for moderate-to-large ncp; fall back to normal
        if math.isnan(power_lower):
            z_crit = norm.ppf(1.0 - alpha / 2.0)
            power_lower = float(norm.sf(z_crit - ncp_lower))
        if math.isnan(power_upper):
            z_crit = norm.ppf(1.0 - alpha / 2.0)
            power_upper = float(norm.sf(z_crit - ncp_upper))

    pwr = power_lower + power_upper - 1.0
    return max(pwr, 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def power_crossover_be(
    n: int | None = None,
    cv: float | None = None,
    theta1: float = 0.80,
    theta2: float = 1.25,
    theta0: float = 0.95,
    alpha: float = 0.05,
    power: float | None = None,
) -> PowerResult:
    """Power for 2x2 crossover bioequivalence study (TOST on log-scale).

    Standard average bioequivalence (ABE) design.
    Exactly one of ``n``, ``power`` must be ``None``.

    Parameters
    ----------
    n : int or None
        Total number of subjects (both sequences).
    cv : float
        Within-subject coefficient of variation (e.g., 0.30 for 30% CV).
        Always required.
    theta1 : float
        Lower bioequivalence limit (default 0.80).
    theta2 : float
        Upper bioequivalence limit (default 1.25).
    theta0 : float
        Assumed true ratio of geometric means (default 0.95).
    alpha : float
        Significance level for TOST (default 0.05, i.e. two one-sided 0.025).
    power : float or None
        Desired power.

    Returns
    -------
    PowerResult

    Examples
    --------
    >>> r = power_crossover_be(cv=0.30, power=0.80)
    >>> r.n  # total subjects
    40

    Validates against: R PowerTOST::sampleN.TOST(), PowerTOST::power.TOST()
    """
    # --- Validate ---
    if cv is None:
        raise ValueError("cv is always required")
    if cv <= 0.0:
        raise ValueError(f"cv must be > 0, got {cv}")
    if not (0.0 < theta1 < 1.0):
        raise ValueError(f"theta1 must be in (0, 1), got {theta1}")
    if theta2 <= 1.0:
        raise ValueError(f"theta2 must be > 1, got {theta2}")
    if theta0 <= 0.0:
        raise ValueError(f"theta0 must be > 0, got {theta0}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    # Check: exactly one of n, power must be None
    none_count = sum(x is None for x in (n, power))
    if none_count != 1:
        raise ValueError("Exactly one of n, power must be None")
    if power is not None and not (0.0 < power < 1.0):
        raise ValueError(f"power must be in (0, 1), got {power}")
    if n is not None and n < 4:
        raise ValueError(f"n must be >= 4 for 2x2 crossover, got {n}")

    if n is None:
        # Solve for n
        assert power is not None
        raw_n = _solve_parameter(
            func=lambda x: _crossover_be_power(x, cv, theta0, theta1, theta2, alpha),
            target=power,
            bracket=(4.0, 1e6),
        )
        # Round up to next even number (standard for crossover)
        result_n = math.ceil(raw_n)
        if result_n % 2 != 0:
            result_n += 1
        result_power = power
    else:
        # Solve for power
        result_power = _crossover_be_power(float(n), cv, theta0, theta1, theta2, alpha)
        result_n = n

    return PowerResult(
        n=result_n,
        power=result_power,
        effect_size=cv,  # report CV as the "effect size" metric
        alpha=alpha,
        alternative="two.sided",
        method="2x2 crossover bioequivalence power calculation (TOST)",
        note=(
            f"n is total subjects (both sequences); CV = {cv:.2%}; "
            f"BE limits = [{theta1}, {theta2}]; theta0 = {theta0}"
        ),
    )
