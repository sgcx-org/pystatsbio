"""Age-standardized rates via direct and indirect methods.

Direct standardization weights stratum-specific rates by a standard
population. Indirect standardization computes a standardized incidence
(or mortality) ratio by comparing observed counts to expected counts
derived from standard rates.

Validates against: R epitools::ageadjust.direct(), epitools::ageadjust.indirect()
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from pystatsbio.epi._common import StandardizedRate


def _validate_arrays(
    counts: np.ndarray,
    person_time: np.ndarray,
    standard_pop: np.ndarray,
) -> None:
    """Validate that inputs are 1-D arrays of equal length with valid values.

    Raises
    ------
    ValueError
        If arrays are not 1-D, have different lengths, or contain
        invalid values.
    """
    if counts.ndim != 1 or person_time.ndim != 1 or standard_pop.ndim != 1:
        raise ValueError("counts, person_time, and standard_pop must be 1-D arrays")

    if not (len(counts) == len(person_time) == len(standard_pop)):
        raise ValueError(
            f"counts ({len(counts)}), person_time ({len(person_time)}), "
            f"and standard_pop ({len(standard_pop)}) must have equal length"
        )

    if len(counts) == 0:
        raise ValueError("arrays must not be empty")

    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")

    if np.any(person_time <= 0):
        raise ValueError("person_time must be positive")

    if np.any(standard_pop < 0):
        raise ValueError("standard_pop must be non-negative")


def _direct_standardize(
    counts: np.ndarray,
    person_time: np.ndarray,
    standard_pop: np.ndarray,
    conf_level: float,
) -> StandardizedRate:
    """Direct age standardization.

    adjusted_rate = sum(rate_i * weight_i) / sum(weight_i)

    SE via Fay-Feuer (gamma-based) approach:
        var = sum(weight_i^2 * count_i / person_time_i^2)
        SE = sqrt(var) / sum(weight_i)

    CI: normal approximation on the adjusted rate.
    """
    rates = counts / person_time
    total_weight = np.sum(standard_pop)

    if total_weight <= 0:
        raise ValueError("sum of standard_pop must be positive for direct method")

    crude_rate = float(np.sum(counts) / np.sum(person_time))

    # Weighted adjusted rate
    adjusted_rate = float(np.sum(rates * standard_pop) / total_weight)

    # Variance: sum(w_i^2 * d_i / n_i^2) / (sum(w_i))^2
    variance = float(
        np.sum(standard_pop ** 2 * counts / person_time ** 2)
        / total_weight ** 2
    )
    se = np.sqrt(variance)

    z = stats.norm.ppf((1 + conf_level) / 2)
    ci_lower = adjusted_rate - z * se
    ci_upper = adjusted_rate + z * se

    return StandardizedRate(
        crude_rate=crude_rate,
        adjusted_rate=adjusted_rate,
        adjusted_rate_ci=(float(ci_lower), float(ci_upper)),
        conf_level=conf_level,
        method="direct",
        sir=None,
        sir_ci=None,
    )


def _indirect_standardize(
    counts: np.ndarray,
    person_time: np.ndarray,
    standard_rates: np.ndarray,
    conf_level: float,
) -> StandardizedRate:
    """Indirect age standardization.

    expected = sum(standard_rate_i * person_time_i)
    SIR = observed / expected

    CI: exact Poisson CI on observed count, divided by expected.
    The adjusted rate is SIR * crude_standard_rate.
    """
    observed = float(np.sum(counts))
    expected = float(np.sum(standard_rates * person_time))

    if expected <= 0:
        raise ValueError(
            "expected count must be positive; check that standard rates "
            "and person-time are valid"
        )

    sir = observed / expected

    crude_rate = float(np.sum(counts) / np.sum(person_time))

    # Crude rate in the standard population (weighted average of standard rates)
    # For indirect method, we use SIR * overall standard rate as adjusted rate
    overall_standard_rate = float(
        np.sum(standard_rates * person_time) / np.sum(person_time)
    )
    adjusted_rate = sir * overall_standard_rate

    # Exact Poisson CI for observed count
    alpha = 1 - conf_level
    if observed == 0:
        obs_lower = 0.0
        obs_upper = float(stats.chi2.ppf(1 - alpha / 2, 2) / 2)
    else:
        obs_lower = float(stats.chi2.ppf(alpha / 2, 2 * observed) / 2)
        obs_upper = float(stats.chi2.ppf(1 - alpha / 2, 2 * (observed + 1)) / 2)

    sir_lower = obs_lower / expected
    sir_upper = obs_upper / expected

    adj_ci_lower = sir_lower * overall_standard_rate
    adj_ci_upper = sir_upper * overall_standard_rate

    return StandardizedRate(
        crude_rate=crude_rate,
        adjusted_rate=adjusted_rate,
        adjusted_rate_ci=(float(adj_ci_lower), float(adj_ci_upper)),
        conf_level=conf_level,
        method="indirect",
        sir=float(sir),
        sir_ci=(float(sir_lower), float(sir_upper)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rate_standardize(
    counts: ArrayLike,
    person_time: ArrayLike,
    standard_pop: ArrayLike,
    *,
    method: str = "direct",
    conf_level: float = 0.95,
) -> StandardizedRate:
    """Age-standardize rates using direct or indirect method.

    Direct standardization:
        adjusted_rate = sum(rate_i * weight_i) / sum(weight_i)
        where rate_i = counts_i / person_time_i
        and weight_i = standard_pop_i

        CI via normal approximation:
        SE = sqrt(sum(weight_i^2 * count_i / person_time_i^2)) / sum(weight_i)

    Indirect standardization:
        expected = sum(standard_rate_i * person_time_i)
        SIR = observed / expected
        CI: exact Poisson CI on observed, divided by expected

        For indirect, standard_pop should be standard RATES, not populations.

    Parameters
    ----------
    counts : array-like
        Observed event counts per stratum (e.g., age group).
    person_time : array-like
        Person-time at risk per stratum.
    standard_pop : array-like
        Standard population weights (direct) or standard rates (indirect).
    method : str
        'direct' or 'indirect'.
    conf_level : float
        Confidence level for intervals. Must be in (0, 1).

    Returns
    -------
    StandardizedRate

    Raises
    ------
    ValueError
        If inputs are invalid or method is not recognized.

    Validates against: R epitools::ageadjust.direct(), epitools::ageadjust.indirect()
    """
    if method not in ("direct", "indirect"):
        raise ValueError(f"method must be 'direct' or 'indirect', got {method!r}")

    if not 0 < conf_level < 1:
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")

    c = np.asarray(counts, dtype=np.float64)
    pt = np.asarray(person_time, dtype=np.float64)
    sp = np.asarray(standard_pop, dtype=np.float64)

    _validate_arrays(c, pt, sp)

    if method == "direct":
        return _direct_standardize(c, pt, sp, conf_level)
    else:
        return _indirect_standardize(c, pt, sp, conf_level)
