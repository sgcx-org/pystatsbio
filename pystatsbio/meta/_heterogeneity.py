"""Heterogeneity statistics for meta-analysis.

Computes Cochran's Q, I-squared, and H-squared statistics to quantify
between-study heterogeneity.

Validates against: R metafor::rma()
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def cochran_q(yi: NDArray, vi: NDArray) -> tuple[float, int, float]:
    """Cochran's Q statistic for heterogeneity.

    Q = sum(w_i * (y_i - mu_FE)^2) where w_i = 1/v_i and
    mu_FE is the fixed-effects pooled estimate.
    The p-value is from a chi-squared distribution with df = k - 1.

    Parameters
    ----------
    yi : NDArray
        Effect sizes (1-D, length k >= 2).
    vi : NDArray
        Sampling variances (1-D, same length as yi, all positive).

    Returns
    -------
    tuple of (float, int, float)
        (Q statistic, degrees of freedom, p-value).
    """
    wi = 1.0 / vi
    mu_fe = np.sum(wi * yi) / np.sum(wi)
    Q = float(np.sum(wi * (yi - mu_fe) ** 2))
    df = len(yi) - 1
    p_val = float(stats.chi2.sf(Q, df))
    return Q, df, p_val


def i_squared(Q: float, k: int) -> float:
    """I-squared heterogeneity statistic.

    I^2 = max(0, (Q - (k-1)) / Q) * 100

    Measures the percentage of total variability due to between-study
    heterogeneity rather than sampling error.

    Parameters
    ----------
    Q : float
        Cochran's Q statistic.
    k : int
        Number of studies.

    Returns
    -------
    float
        I-squared as a percentage (0 to 100).
    """
    if Q <= 0.0:
        return 0.0
    return max(0.0, (Q - (k - 1)) / Q) * 100.0


def h_squared(Q: float, k: int) -> float:
    """H-squared heterogeneity statistic.

    H^2 = Q / (k - 1)

    Ratio of total variability to sampling variability.
    H^2 = 1 when there is no heterogeneity.

    Parameters
    ----------
    Q : float
        Cochran's Q statistic.
    k : int
        Number of studies.

    Returns
    -------
    float
        H-squared (>= 0).
    """
    df = k - 1
    if df <= 0:
        return 1.0
    return Q / df
