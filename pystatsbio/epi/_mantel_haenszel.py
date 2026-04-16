"""Mantel-Haenszel stratified analysis for pooled OR or RR.

Pools odds ratios or risk ratios across K strata of 2x2 tables
using the Mantel-Haenszel method, with Cochran-Mantel-Haenszel test
for conditional independence and Breslow-Day test for homogeneity.

Validates against: R stats::mantelhaen.test(), DescTools::BreslowDayTest()
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from pystatsbio.epi._common import EpiMeasure, MantelHaenszelResult


def _validate_tables(tables: np.ndarray) -> np.ndarray:
    """Validate array of 2x2 tables.

    Parameters
    ----------
    tables : ndarray
        Array of shape (K, 2, 2).

    Returns
    -------
    ndarray
        Validated float64 array.

    Raises
    ------
    ValueError
        If shape or values are invalid.
    """
    if tables.ndim != 3 or tables.shape[1] != 2 or tables.shape[2] != 2:
        raise ValueError(
            f"tables must have shape (K, 2, 2), got {tables.shape}"
        )

    if tables.shape[0] == 0:
        raise ValueError("tables must contain at least one stratum")

    if np.any(tables < 0):
        raise ValueError("all table cells must be non-negative")

    return tables.astype(np.float64)


def _mh_odds_ratio(
    tables: np.ndarray,
    z: float,
    conf_level: float,
) -> EpiMeasure:
    """Mantel-Haenszel pooled odds ratio with Robins-Breslow-Greenland CI.

    MH_OR = sum(a_i * d_i / n_i) / sum(b_i * c_i / n_i)

    Robins-Breslow-Greenland variance:
        var(ln(MH_OR)) computed from the R, S, P, Q sums.
    """
    k = tables.shape[0]
    a = tables[:, 0, 0]
    b = tables[:, 0, 1]
    c = tables[:, 1, 0]
    d = tables[:, 1, 1]
    n = a + b + c + d

    numerator = np.sum(a * d / n)
    denominator = np.sum(b * c / n)

    if denominator == 0:
        raise ValueError(
            "MH OR denominator is zero: all strata have b*c = 0"
        )

    mh_or = numerator / denominator

    # Robins-Breslow-Greenland variance estimator
    # P_i = (a_i + d_i) / n_i
    # Q_i = (b_i + c_i) / n_i
    p_i = (a + d) / n
    q_i = (b + c) / n

    r_i = a * d / n  # numerator terms
    s_i = b * c / n  # denominator terms

    sum_r = np.sum(r_i)
    sum_s = np.sum(s_i)

    # Variance of ln(MH_OR) per Robins-Breslow-Greenland (1986)
    var_ln = (
        np.sum(p_i * r_i) / (2 * sum_r ** 2)
        + np.sum(p_i * s_i + q_i * r_i) / (2 * sum_r * sum_s)
        + np.sum(q_i * s_i) / (2 * sum_s ** 2)
    )

    se_ln = np.sqrt(var_ln)
    ci_lower = np.exp(np.log(mh_or) - z * se_ln)
    ci_upper = np.exp(np.log(mh_or) + z * se_ln)

    return EpiMeasure(
        name="MH Odds Ratio",
        estimate=float(mh_or),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        conf_level=conf_level,
        method="Robins-Breslow-Greenland",
    )


def _mh_risk_ratio(
    tables: np.ndarray,
    z: float,
    conf_level: float,
) -> EpiMeasure:
    """Mantel-Haenszel pooled risk ratio with Greenland-Robins CI.

    MH_RR = sum(a_i * (c_i+d_i) / n_i) / sum(c_i * (a_i+b_i) / n_i)
    """
    a = tables[:, 0, 0]
    b = tables[:, 0, 1]
    c = tables[:, 1, 0]
    d = tables[:, 1, 1]
    n = a + b + c + d
    n1 = a + b  # exposed total per stratum
    n0 = c + d  # unexposed total per stratum

    numerator = np.sum(a * n0 / n)
    denominator = np.sum(c * n1 / n)

    if denominator == 0:
        raise ValueError(
            "MH RR denominator is zero: all strata have c*(a+b) = 0"
        )

    mh_rr = numerator / denominator

    # Greenland-Robins variance estimator for ln(MH_RR)
    sum_num = numerator
    sum_den = denominator

    # Variance terms
    var_ln = 0.0
    p_terms = np.zeros(len(a))
    for i in range(len(a)):
        ni = n[i]
        m1i = a[i] + c[i]  # total disease in stratum
        p_terms[i] = (
            m1i * n1[i] * n0[i] / ni ** 2
            - a[i] * c[i] / ni
        )

    var_ln = float(np.sum(p_terms) / (sum_num * sum_den))

    se_ln = np.sqrt(var_ln)
    ci_lower = np.exp(np.log(mh_rr) - z * se_ln)
    ci_upper = np.exp(np.log(mh_rr) + z * se_ln)

    return EpiMeasure(
        name="MH Risk Ratio",
        estimate=float(mh_rr),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        conf_level=conf_level,
        method="Greenland-Robins",
    )


def _cmh_test(tables: np.ndarray) -> tuple[float, float]:
    """Cochran-Mantel-Haenszel chi-squared test.

    chi2_CMH = (|sum(a_i - E(a_i))| - 0.5)^2 / sum(Var(a_i))

    where under independence in stratum i:
        E(a_i) = (a_i+b_i)(a_i+c_i) / n_i
        Var(a_i) = (a_i+b_i)(c_i+d_i)(a_i+c_i)(b_i+d_i) / (n_i^2 * (n_i-1))

    Returns (chi2, p_value).
    """
    a = tables[:, 0, 0]
    b = tables[:, 0, 1]
    c = tables[:, 1, 0]
    d = tables[:, 1, 1]
    n = a + b + c + d
    n1 = a + b
    n0 = c + d
    m1 = a + c
    m0 = b + d

    expected = n1 * m1 / n
    variance = n1 * n0 * m1 * m0 / (n ** 2 * (n - 1))

    # Guard against strata with n_i <= 1
    valid = n > 1
    if not np.any(valid):
        raise ValueError("all strata have n <= 1; cannot compute CMH statistic")

    sum_diff = np.sum((a - expected)[valid])
    sum_var = np.sum(variance[valid])

    if sum_var == 0:
        raise ValueError("variance sum is zero; cannot compute CMH statistic")

    # Continuity-corrected CMH
    chi2 = (abs(sum_diff) - 0.5) ** 2 / sum_var
    p_value = float(stats.chi2.sf(chi2, df=1))

    return float(chi2), p_value


def _breslow_day_test(
    tables: np.ndarray,
    mh_or: float,
) -> tuple[float | None, float | None]:
    """Breslow-Day test for homogeneity of odds ratios.

    Tests H0: common OR across strata.

    For each stratum, find the expected value of a_i under the
    common OR (MH_OR) by solving:
        a_i * d_i / (b_i * c_i) = OR_MH
    given the marginals.

    BD = sum((a_i - E(a_i|OR_MH))^2 / Var(a_i|OR_MH))
    p-value from chi-squared(df=K-1).

    Returns (None, None) if K < 2.
    """
    k = tables.shape[0]
    if k < 2:
        return None, None

    a = tables[:, 0, 0]
    b = tables[:, 0, 1]
    c = tables[:, 1, 0]
    d = tables[:, 1, 1]
    n = a + b + c + d
    n1 = a + b
    m1 = a + c

    bd_stat = 0.0

    for i in range(k):
        # Solve quadratic for expected a_i given MH OR and marginals
        # Under OR_MH with fixed marginals n1_i, m1_i, n_i:
        # OR_MH = a*(n-n1-m1+a) / ((n1-a)*(m1-a))
        # This is a quadratic in a:
        # (OR_MH - 1) * a^2 - (OR_MH*(n1+m1) + n0_m0_diff) * a + OR_MH*n1*m1 = 0
        # where the coefficients come from expanding the equation.

        n1_i = n1[i]
        m1_i = m1[i]
        n_i = n[i]

        if mh_or == 1.0:
            # Under OR=1, expected a is just (n1*m1)/n
            e_a = n1_i * m1_i / n_i
        else:
            # Coefficients of quadratic: (OR-1)*a^2 - B*a + C = 0
            coeff_a = mh_or - 1.0
            coeff_b = -(mh_or * (n1_i + m1_i) + (n_i - n1_i - m1_i))
            coeff_c = mh_or * n1_i * m1_i

            discriminant = coeff_b ** 2 - 4 * coeff_a * coeff_c
            if discriminant < 0:
                # Numerical issue; skip this stratum
                continue

            sqrt_disc = np.sqrt(discriminant)
            root1 = (-coeff_b - sqrt_disc) / (2 * coeff_a)
            root2 = (-coeff_b + sqrt_disc) / (2 * coeff_a)

            # Pick root that is within valid range [max(0, n1+m1-n), min(n1, m1)]
            lo = max(0, n1_i + m1_i - n_i)
            hi = min(n1_i, m1_i)

            if lo <= root1 <= hi:
                e_a = root1
            elif lo <= root2 <= hi:
                e_a = root2
            else:
                # Both roots out of range; pick closest
                e_a = root1 if abs(root1 - a[i]) < abs(root2 - a[i]) else root2

        # Variance under the common OR
        # 1/Var = 1/e_a + 1/(n1_i - e_a) + 1/(m1_i - e_a) + 1/(n_i - n1_i - m1_i + e_a)
        b_exp = n1_i - e_a
        c_exp = m1_i - e_a
        d_exp = n_i - n1_i - m1_i + e_a

        denom_terms = [e_a, b_exp, c_exp, d_exp]
        if any(t <= 0 for t in denom_terms):
            # Degenerate stratum, skip
            continue

        var_a = 1.0 / (1.0 / e_a + 1.0 / b_exp + 1.0 / c_exp + 1.0 / d_exp)
        bd_stat += (a[i] - e_a) ** 2 / var_a

    p_value = float(stats.chi2.sf(bd_stat, df=k - 1))

    return float(bd_stat), p_value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mantel_haenszel(
    tables: ArrayLike,
    *,
    measure: str = "OR",
    conf_level: float = 0.95,
) -> MantelHaenszelResult:
    """Mantel-Haenszel stratified analysis for pooled OR or RR.

    Pools effect measures across K strata of 2x2 tables, tests for
    conditional independence (CMH test), and tests for homogeneity
    of effect across strata (Breslow-Day test, OR only).

    Parameters
    ----------
    tables : array-like, shape (K, 2, 2)
        K strata of 2x2 tables. Each stratum follows the layout:
        [[a, b], [c, d]] where a = exposed+disease, b = exposed+no disease,
        c = unexposed+disease, d = unexposed+no disease.
    measure : str
        'OR' for odds ratio or 'RR' for risk ratio.
    conf_level : float
        Confidence level for intervals. Must be in (0, 1).

    Returns
    -------
    MantelHaenszelResult

    Raises
    ------
    ValueError
        If tables shape is wrong, measure is invalid, or computations
        are degenerate.

    Validates against: R stats::mantelhaen.test(), DescTools::BreslowDayTest()
    """
    if measure not in ("OR", "RR"):
        raise ValueError(f"measure must be 'OR' or 'RR', got {measure!r}")

    if not 0 < conf_level < 1:
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")

    tbl = np.asarray(tables, dtype=np.float64)
    tbl = _validate_tables(tbl)

    z = stats.norm.ppf((1 + conf_level) / 2)

    if measure == "OR":
        pooled = _mh_odds_ratio(tbl, z, conf_level)
    else:
        pooled = _mh_risk_ratio(tbl, z, conf_level)

    cmh_chi2, cmh_p = _cmh_test(tbl)

    # Breslow-Day test (only for OR)
    if measure == "OR":
        bd_stat, bd_p = _breslow_day_test(tbl, pooled.estimate)
    else:
        bd_stat, bd_p = None, None

    return MantelHaenszelResult(
        pooled_estimate=pooled,
        cmh_statistic=cmh_chi2,
        cmh_p_value=cmh_p,
        breslow_day_statistic=bd_stat,
        breslow_day_p_value=bd_p,
        n_strata=int(tbl.shape[0]),
        measure=measure,
    )
