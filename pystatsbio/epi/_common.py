"""Shared result types for epidemiological analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EpiMeasure:
    """A single epidemiological measure with confidence interval.

    Attributes
    ----------
    name : str
        Human-readable measure name (e.g. "Risk Ratio", "Odds Ratio").
    estimate : float
        Point estimate.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    conf_level : float
        Confidence level used (e.g. 0.95).
    method : str
        Name of the CI method used.
    """

    name: str
    estimate: float
    ci_lower: float
    ci_upper: float
    conf_level: float
    method: str

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"{self.name}: {self.estimate:.4f} "
            f"({self.conf_level:.0%} CI: {self.ci_lower:.4f}\u2013{self.ci_upper:.4f}) "
            f"[{self.method}]"
        )


@dataclass(frozen=True)
class Epi2x2Result:
    """Complete results from a 2x2 contingency table analysis.

    Attributes
    ----------
    risk_ratio : EpiMeasure
        Risk ratio (relative risk) with CI.
    odds_ratio : EpiMeasure
        Odds ratio with CI (Woolf method).
    risk_difference : EpiMeasure
        Risk difference (absolute risk reduction) with CI.
    attributable_risk_exposed : EpiMeasure
        Attributable fraction in exposed: (RR - 1) / RR.
    population_attributable_fraction : EpiMeasure
        Population attributable fraction (Levin formula).
    nnt : EpiMeasure
        Number needed to treat: 1 / |RD|.
    table : NDArray
        The original (possibly corrected) 2x2 table.
    """

    risk_ratio: EpiMeasure
    odds_ratio: EpiMeasure
    risk_difference: EpiMeasure
    attributable_risk_exposed: EpiMeasure
    population_attributable_fraction: EpiMeasure
    nnt: EpiMeasure
    table: NDArray

    def summary(self) -> str:
        """Human-readable summary of all measures."""
        lines = [
            "2x2 Epidemiological Analysis",
            "=" * 50,
            self.risk_ratio.summary(),
            self.odds_ratio.summary(),
            self.risk_difference.summary(),
            self.attributable_risk_exposed.summary(),
            self.population_attributable_fraction.summary(),
            self.nnt.summary(),
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class StandardizedRate:
    """Result from rate standardization.

    Attributes
    ----------
    crude_rate : float
        Unstandardized (crude) rate.
    adjusted_rate : float
        Standardized (adjusted) rate.
    adjusted_rate_ci : tuple of float
        Confidence interval for the adjusted rate.
    conf_level : float
        Confidence level used.
    method : str
        'direct' or 'indirect'.
    sir : float or None
        Standardized incidence/mortality ratio (indirect only).
    sir_ci : tuple of float or None
        CI for SIR (indirect only).
    """

    crude_rate: float
    adjusted_rate: float
    adjusted_rate_ci: tuple[float, float]
    conf_level: float
    method: str
    sir: float | None
    sir_ci: tuple[float, float] | None

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Rate Standardization ({self.method})",
            "=" * 50,
            f"Crude rate    : {self.crude_rate:.6f}",
            f"Adjusted rate : {self.adjusted_rate:.6f}",
            f"{self.conf_level:.0%} CI       : "
            f"[{self.adjusted_rate_ci[0]:.6f}, {self.adjusted_rate_ci[1]:.6f}]",
        ]
        if self.sir is not None and self.sir_ci is not None:
            lines.append(f"SIR           : {self.sir:.4f}")
            lines.append(
                f"SIR {self.conf_level:.0%} CI  : "
                f"[{self.sir_ci[0]:.4f}, {self.sir_ci[1]:.4f}]"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class MantelHaenszelResult:
    """Result from Mantel-Haenszel stratified analysis.

    Attributes
    ----------
    pooled_estimate : EpiMeasure
        MH pooled odds ratio or risk ratio with CI.
    cmh_statistic : float
        Cochran-Mantel-Haenszel chi-squared test statistic.
    cmh_p_value : float
        p-value for the CMH test (chi-squared df=1).
    breslow_day_statistic : float or None
        Breslow-Day homogeneity test statistic (None if < 2 strata).
    breslow_day_p_value : float or None
        p-value for Breslow-Day test (None if < 2 strata).
    n_strata : int
        Number of strata.
    measure : str
        'OR' or 'RR'.
    """

    pooled_estimate: EpiMeasure
    cmh_statistic: float
    cmh_p_value: float
    breslow_day_statistic: float | None
    breslow_day_p_value: float | None
    n_strata: int
    measure: str

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Mantel-Haenszel Stratified Analysis ({self.measure})",
            "=" * 50,
            self.pooled_estimate.summary(),
            f"CMH chi-sq    : {self.cmh_statistic:.4f}",
            f"CMH p-value   : {self.cmh_p_value:.4g}",
            f"Strata        : {self.n_strata}",
        ]
        if self.breslow_day_statistic is not None:
            lines.append(
                f"Breslow-Day   : {self.breslow_day_statistic:.4f} "
                f"(p = {self.breslow_day_p_value:.4g})"
            )
        return "\n".join(lines)
