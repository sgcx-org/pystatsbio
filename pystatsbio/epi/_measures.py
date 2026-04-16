"""Epidemiological measures from a 2x2 contingency table.

Computes risk ratio, odds ratio, risk difference, attributable fractions,
and number needed to treat from a 2x2 table following the layout used by
R's epiR::epi.2by2.

Validates against: R epiR::epi.2by2()
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from pystatsbio.epi._common import Epi2x2Result, EpiMeasure


def _validate_2by2(table: NDArray) -> NDArray:
    """Validate and return a float64 2x2 table, applying continuity correction if needed.

    Parameters
    ----------
    table : NDArray
        A 2x2 array of non-negative values.

    Returns
    -------
    NDArray
        The validated (and possibly corrected) table as float64.

    Raises
    ------
    ValueError
        If the table is not 2x2 or contains negative values.
    """
    if table.shape != (2, 2):
        raise ValueError(
            f"table must be 2x2, got shape {table.shape}"
        )
    if np.any(table < 0):
        raise ValueError("table must contain non-negative values")

    # Apply 0.5 continuity correction if any cell is zero
    if np.any(table == 0):
        table = table + 0.5

    return table.astype(np.float64)


def _risk_ratio(
    a: float, b: float, c: float, d: float,
    z: float, conf_level: float,
) -> EpiMeasure:
    """Risk ratio with log-transformed CI.

    RR = (a/(a+b)) / (c/(c+d))
    SE(log(RR)) = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    """
    p1 = a / (a + b)
    p2 = c / (c + d)
    rr = p1 / p2
    log_se = np.sqrt(1.0 / a - 1.0 / (a + b) + 1.0 / c - 1.0 / (c + d))
    ci_lower = np.exp(np.log(rr) - z * log_se)
    ci_upper = np.exp(np.log(rr) + z * log_se)
    return EpiMeasure(
        name="Risk Ratio",
        estimate=float(rr),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        conf_level=conf_level,
        method="log-transformed",
    )


def _odds_ratio(
    a: float, b: float, c: float, d: float,
    z: float, conf_level: float,
) -> EpiMeasure:
    """Odds ratio with Woolf (log-transformed) CI.

    OR = (a*d) / (b*c)
    SE(log(OR)) = sqrt(1/a + 1/b + 1/c + 1/d)
    """
    oratio = (a * d) / (b * c)
    log_se = np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    ci_lower = np.exp(np.log(oratio) - z * log_se)
    ci_upper = np.exp(np.log(oratio) + z * log_se)
    return EpiMeasure(
        name="Odds Ratio",
        estimate=float(oratio),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        conf_level=conf_level,
        method="Woolf (log-transformed)",
    )


def _risk_difference(
    a: float, b: float, c: float, d: float,
    z: float, conf_level: float,
) -> EpiMeasure:
    """Risk difference with Wald CI.

    RD = p1 - p2
    SE(RD) = sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    """
    n1 = a + b
    n2 = c + d
    p1 = a / n1
    p2 = c / n2
    rd = p1 - p2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_lower = rd - z * se
    ci_upper = rd + z * se
    return EpiMeasure(
        name="Risk Difference",
        estimate=float(rd),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        conf_level=conf_level,
        method="Wald",
    )


def _attributable_risk_exposed(
    rr: float, z: float, conf_level: float,
    rr_ci_lower: float, rr_ci_upper: float,
) -> EpiMeasure:
    """Attributable fraction in exposed: AFe = (RR - 1) / RR.

    CI derived from RR CI: AFe_ci = (RR_ci - 1) / RR_ci
    """
    afe = (rr - 1.0) / rr
    # Transform RR CI to AFe CI
    afe_lower = (rr_ci_lower - 1.0) / rr_ci_lower
    afe_upper = (rr_ci_upper - 1.0) / rr_ci_upper
    return EpiMeasure(
        name="Attrib. Fraction (Exposed)",
        estimate=float(afe),
        ci_lower=float(afe_lower),
        ci_upper=float(afe_upper),
        conf_level=conf_level,
        method="from RR CI",
    )


def _population_attributable_fraction(
    a: float, b: float, c: float, d: float,
    rr: float, z: float, conf_level: float,
    rr_ci_lower: float, rr_ci_upper: float,
) -> EpiMeasure:
    """Population attributable fraction via Levin's formula.

    PAF = pe * (RR - 1) / (pe * (RR - 1) + 1)
    where pe = (a+c) / (a+b+c+d) is prevalence of exposure.

    CI: Levin's formula applied to RR CI bounds.
    """
    n = a + b + c + d
    pe = (a + c) / n

    def _levin(r: float) -> float:
        return pe * (r - 1.0) / (pe * (r - 1.0) + 1.0)

    paf = _levin(rr)
    paf_lower = _levin(rr_ci_lower)
    paf_upper = _levin(rr_ci_upper)

    # Ensure lower <= upper
    if paf_lower > paf_upper:
        paf_lower, paf_upper = paf_upper, paf_lower

    return EpiMeasure(
        name="Pop. Attrib. Fraction",
        estimate=float(paf),
        ci_lower=float(paf_lower),
        ci_upper=float(paf_upper),
        conf_level=conf_level,
        method="Levin",
    )


def _nnt(
    rd: float, rd_ci_lower: float, rd_ci_upper: float,
    conf_level: float,
) -> EpiMeasure:
    """Number needed to treat: NNT = 1 / |RD|.

    CI derived by inverting RD CI. When the RD CI spans zero,
    NNT CI is set to (NNT_estimate, inf).
    """
    nnt_est = 1.0 / abs(rd) if rd != 0 else float("inf")

    # Invert the RD CI for NNT CI
    # NNT CI bounds come from RD CI, but inverted and sorted
    if rd_ci_lower > 0:
        # Both bounds positive: risk in exposed > unexposed
        nnt_upper = 1.0 / rd_ci_lower
        nnt_lower = 1.0 / rd_ci_upper
    elif rd_ci_upper < 0:
        # Both bounds negative: risk in exposed < unexposed
        nnt_upper = 1.0 / abs(rd_ci_upper)
        nnt_lower = 1.0 / abs(rd_ci_lower)
    else:
        # CI spans zero: NNT is not well-defined
        nnt_lower = nnt_est
        nnt_upper = float("inf")

    return EpiMeasure(
        name="NNT",
        estimate=float(nnt_est),
        ci_lower=float(nnt_lower),
        ci_upper=float(nnt_upper),
        conf_level=conf_level,
        method="from RD CI",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def epi_2by2(
    table: ArrayLike,
    *,
    conf_level: float = 0.95,
) -> Epi2x2Result:
    """Compute epidemiological measures from a 2x2 contingency table.

    Table layout (matching R's epiR::epi.2by2):
        [[a, b],    a = exposed+disease, b = exposed+no disease
         [c, d]]    c = unexposed+disease, d = unexposed+no disease

    Computes:
    - Risk ratio (RR) with log-transformed CI
    - Odds ratio (OR) with Woolf CI
    - Risk difference (RD) with Wald CI
    - Attributable fraction in exposed (AFe)
    - Population attributable fraction (PAF) via Levin's formula
    - Number needed to treat (NNT)

    Parameters
    ----------
    table : array-like, shape (2, 2)
        2x2 contingency table with non-negative values.
        If any cell is zero, a 0.5 continuity correction is applied.
    conf_level : float
        Confidence level for all intervals. Must be in (0, 1).

    Returns
    -------
    Epi2x2Result

    Raises
    ------
    ValueError
        If table is not 2x2, contains negative values, or conf_level
        is out of range.

    Validates against: R epiR::epi.2by2()
    """
    if not 0 < conf_level < 1:
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")

    tbl = np.asarray(table, dtype=np.float64)
    tbl = _validate_2by2(tbl)

    a, b, c, d = tbl[0, 0], tbl[0, 1], tbl[1, 0], tbl[1, 1]
    z = stats.norm.ppf((1 + conf_level) / 2)

    rr_result = _risk_ratio(a, b, c, d, z, conf_level)
    or_result = _odds_ratio(a, b, c, d, z, conf_level)
    rd_result = _risk_difference(a, b, c, d, z, conf_level)

    afe_result = _attributable_risk_exposed(
        rr_result.estimate, z, conf_level,
        rr_result.ci_lower, rr_result.ci_upper,
    )

    paf_result = _population_attributable_fraction(
        a, b, c, d,
        rr_result.estimate, z, conf_level,
        rr_result.ci_lower, rr_result.ci_upper,
    )

    nnt_result = _nnt(
        rd_result.estimate,
        rd_result.ci_lower, rd_result.ci_upper,
        conf_level,
    )

    return Epi2x2Result(
        risk_ratio=rr_result,
        odds_ratio=or_result,
        risk_difference=rd_result,
        attributable_risk_exposed=afe_result,
        population_attributable_fraction=paf_result,
        nnt=nnt_result,
        table=tbl,
    )
