"""Shared result types for epidemiological analysis.

Each top-level analysis has a frozen ``*Params`` payload of computed outputs
and a public ``*Solution`` return that wraps a ``core.result.Result[*Params]``
so every epi result exposes the same ``.backend_name`` / ``.timing`` /
``.warnings`` / ``.info`` metadata and Jupyter ``_repr_html_`` as the rest of
the ecosystem (``pystatsbio/CONVENTIONS.md`` B2). ``EpiMeasure`` is a nested
value object (a single estimate + CI) that appears as a field inside these
payloads; it is not itself a Solution.
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray
from pystatistics.core.result import Result, SolutionReprMixin


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
            f"({self.conf_level:.0%} CI: {self.ci_lower:.4f}–{self.ci_upper:.4f}) "
            f"[{self.method}]"
        )


@dataclass(frozen=True)
class Epi2x2Params:
    """Computed payload from a 2x2 contingency table analysis.

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


class Epi2x2Solution(SolutionReprMixin):
    """Public result of a 2x2 analysis — wraps ``Result[Epi2x2Params]``.

    Exposes every measure as a read-only property plus the uniform
    ``.backend_name`` / ``.timing`` / ``.warnings`` / ``.info`` metadata and a
    Jupyter ``_repr_html_`` (via :class:`SolutionReprMixin`).
    """

    def __init__(self, result: Result[Epi2x2Params]) -> None:
        self._result = result

    # --- Metadata (from the Result envelope) ---
    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def timing(self) -> dict[str, float] | None:
        return self._result.timing

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._result.warnings

    @property
    def info(self) -> dict:
        return self._result.info

    # --- Measures (from the payload) ---
    @property
    def risk_ratio(self) -> EpiMeasure:
        return self._result.params.risk_ratio

    @property
    def odds_ratio(self) -> EpiMeasure:
        return self._result.params.odds_ratio

    @property
    def risk_difference(self) -> EpiMeasure:
        return self._result.params.risk_difference

    @property
    def attributable_risk_exposed(self) -> EpiMeasure:
        return self._result.params.attributable_risk_exposed

    @property
    def population_attributable_fraction(self) -> EpiMeasure:
        return self._result.params.population_attributable_fraction

    @property
    def nnt(self) -> EpiMeasure:
        return self._result.params.nnt

    @property
    def table(self) -> NDArray:
        return self._result.params.table

    def summary(self) -> str:
        """Human-readable summary of all measures."""
        p = self._result.params
        lines = [
            "2x2 Epidemiological Analysis",
            "=" * 50,
            p.risk_ratio.summary(),
            p.odds_ratio.summary(),
            p.risk_difference.summary(),
            p.attributable_risk_exposed.summary(),
            p.population_attributable_fraction.summary(),
            p.nnt.summary(),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        p = self._result.params
        return (
            f"Epi2x2Solution(RR={p.risk_ratio.estimate:.4f}, "
            f"OR={p.odds_ratio.estimate:.4f})"
        )


@dataclass(frozen=True)
class StandardizedRateParams:
    """Computed payload from rate standardization.

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


class StandardizedRateSolution(SolutionReprMixin):
    """Public result of rate standardization — wraps ``Result[StandardizedRateParams]``.

    Exposes every output as a read-only property plus the uniform
    ``.backend_name`` / ``.timing`` / ``.warnings`` / ``.info`` metadata and a
    Jupyter ``_repr_html_`` (via :class:`SolutionReprMixin`).
    """

    def __init__(self, result: Result[StandardizedRateParams]) -> None:
        self._result = result

    # --- Metadata (from the Result envelope) ---
    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def timing(self) -> dict[str, float] | None:
        return self._result.timing

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._result.warnings

    @property
    def info(self) -> dict:
        return self._result.info

    # --- Outputs (from the payload) ---
    @property
    def crude_rate(self) -> float:
        return self._result.params.crude_rate

    @property
    def adjusted_rate(self) -> float:
        return self._result.params.adjusted_rate

    @property
    def adjusted_rate_ci(self) -> tuple[float, float]:
        return self._result.params.adjusted_rate_ci

    @property
    def conf_level(self) -> float:
        return self._result.params.conf_level

    @property
    def method(self) -> str:
        return self._result.params.method

    @property
    def sir(self) -> float | None:
        return self._result.params.sir

    @property
    def sir_ci(self) -> tuple[float, float] | None:
        return self._result.params.sir_ci

    def summary(self) -> str:
        """Human-readable summary."""
        p = self._result.params
        lines = [
            f"Rate Standardization ({p.method})",
            "=" * 50,
            f"Crude rate    : {p.crude_rate:.6f}",
            f"Adjusted rate : {p.adjusted_rate:.6f}",
            f"{p.conf_level:.0%} CI       : "
            f"[{p.adjusted_rate_ci[0]:.6f}, {p.adjusted_rate_ci[1]:.6f}]",
        ]
        if p.sir is not None and p.sir_ci is not None:
            lines.append(f"SIR           : {p.sir:.4f}")
            lines.append(
                f"SIR {p.conf_level:.0%} CI  : "
                f"[{p.sir_ci[0]:.4f}, {p.sir_ci[1]:.4f}]"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        p = self._result.params
        return (
            f"StandardizedRateSolution(method={p.method!r}, "
            f"adjusted_rate={p.adjusted_rate:.6f})"
        )


@dataclass(frozen=True)
class MantelHaenszelParams:
    """Computed payload from Mantel-Haenszel stratified analysis.

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
        'odds-ratio' or 'risk-ratio'.
    """

    pooled_estimate: EpiMeasure
    cmh_statistic: float
    cmh_p_value: float
    breslow_day_statistic: float | None
    breslow_day_p_value: float | None
    n_strata: int
    measure: str


class MantelHaenszelSolution(SolutionReprMixin):
    """Public result of an MH analysis — wraps ``Result[MantelHaenszelParams]``.

    Exposes every output as a read-only property plus the uniform
    ``.backend_name`` / ``.timing`` / ``.warnings`` / ``.info`` metadata and a
    Jupyter ``_repr_html_`` (via :class:`SolutionReprMixin`).
    """

    def __init__(self, result: Result[MantelHaenszelParams]) -> None:
        self._result = result

    # --- Metadata (from the Result envelope) ---
    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def timing(self) -> dict[str, float] | None:
        return self._result.timing

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._result.warnings

    @property
    def info(self) -> dict:
        return self._result.info

    # --- Outputs (from the payload) ---
    @property
    def pooled_estimate(self) -> EpiMeasure:
        return self._result.params.pooled_estimate

    @property
    def cmh_statistic(self) -> float:
        return self._result.params.cmh_statistic

    @property
    def cmh_p_value(self) -> float:
        return self._result.params.cmh_p_value

    @property
    def breslow_day_statistic(self) -> float | None:
        return self._result.params.breslow_day_statistic

    @property
    def breslow_day_p_value(self) -> float | None:
        return self._result.params.breslow_day_p_value

    @property
    def n_strata(self) -> int:
        return self._result.params.n_strata

    @property
    def measure(self) -> str:
        return self._result.params.measure

    def summary(self) -> str:
        """Human-readable summary."""
        p = self._result.params
        lines = [
            f"Mantel-Haenszel Stratified Analysis ({p.measure})",
            "=" * 50,
            p.pooled_estimate.summary(),
            f"CMH chi-sq    : {p.cmh_statistic:.4f}",
            f"CMH p-value   : {p.cmh_p_value:.4g}",
            f"Strata        : {p.n_strata}",
        ]
        if p.breslow_day_statistic is not None:
            lines.append(
                f"Breslow-Day   : {p.breslow_day_statistic:.4f} "
                f"(p = {p.breslow_day_p_value:.4g})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        p = self._result.params
        return (
            f"MantelHaenszelSolution(measure={p.measure!r}, "
            f"pooled={p.pooled_estimate.estimate:.4f}, n_strata={p.n_strata})"
        )
