"""Result types for non-compartmental pharmacokinetic analysis (NCA).

``NCAParams`` is the frozen payload of computed outputs; ``NCASolution`` is
the public return — it wraps a ``core.result.Result[NCAParams]`` so every NCA
fit exposes the same ``.backend_name`` / ``.timing`` / ``.warnings`` / ``.info``
metadata and Jupyter ``_repr_html_`` as the rest of the ecosystem
(``pystatsbio/CONVENTIONS.md`` B2).
"""

from __future__ import annotations

from dataclasses import dataclass

from pystatistics.core.result import Result, SolutionReprMixin


@dataclass(frozen=True)
class NCAParams:
    """Computed payload of a non-compartmental pharmacokinetic analysis."""

    # Primary PK parameters
    auc_last: float  # AUC from 0 to last measurable concentration
    auc_inf: float | None  # AUC extrapolated to infinity
    auc_pct_extrap: float | None  # % AUC extrapolated
    cmax: float  # peak concentration
    tmax: float  # time of peak concentration
    half_life: float | None  # terminal elimination half-life
    lambda_z: float | None  # terminal elimination rate constant
    lambda_z_r_squared: float | None  # r-squared of terminal slope regression

    # Derived parameters (require dose)
    clearance: float | None  # CL = Dose / AUC_inf (or CL/F for oral)
    vz: float | None  # Vz = Dose / (lambda_z * AUC_inf)

    # Metadata
    dose: float | None
    route: str  # 'iv' or 'ev' (extravascular)
    auc_method: str  # 'linear', 'log-linear', 'linear-up/log-down'
    n_points: int
    n_terminal: int  # number of points used for terminal slope


class NCASolution(SolutionReprMixin):
    """Public result of an NCA fit — a Solution wrapping ``Result[NCAParams]``.

    Exposes every NCA parameter as a read-only property plus the uniform
    ``.backend_name`` / ``.timing`` / ``.warnings`` / ``.info`` metadata and a
    Jupyter ``_repr_html_`` (via :class:`SolutionReprMixin`).
    """

    def __init__(self, result: Result[NCAParams]) -> None:
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

    # --- PK parameters (from the payload) ---
    @property
    def auc_last(self) -> float:
        return self._result.params.auc_last

    @property
    def auc_inf(self) -> float | None:
        return self._result.params.auc_inf

    @property
    def auc_pct_extrap(self) -> float | None:
        return self._result.params.auc_pct_extrap

    @property
    def cmax(self) -> float:
        return self._result.params.cmax

    @property
    def tmax(self) -> float:
        return self._result.params.tmax

    @property
    def half_life(self) -> float | None:
        return self._result.params.half_life

    @property
    def lambda_z(self) -> float | None:
        return self._result.params.lambda_z

    @property
    def lambda_z_r_squared(self) -> float | None:
        return self._result.params.lambda_z_r_squared

    @property
    def clearance(self) -> float | None:
        return self._result.params.clearance

    @property
    def vz(self) -> float | None:
        return self._result.params.vz

    @property
    def dose(self) -> float | None:
        return self._result.params.dose

    @property
    def route(self) -> str:
        return self._result.params.route

    @property
    def auc_method(self) -> str:
        return self._result.params.auc_method

    @property
    def n_points(self) -> int:
        return self._result.params.n_points

    @property
    def n_terminal(self) -> int:
        return self._result.params.n_terminal

    def summary(self) -> str:
        """Human-readable PK summary."""
        p = self._result.params
        lines = ["Non-Compartmental Analysis", ""]
        lines.append(f"  Route: {p.route.upper()}")
        lines.append(f"  AUC method: {p.auc_method}")
        if p.dose is not None:
            lines.append(f"  Dose: {p.dose}")
        lines.append("")
        lines.append(f"  Cmax          = {p.cmax:.4g}")
        lines.append(f"  Tmax          = {p.tmax:.4g}")
        lines.append(f"  AUC(0-last)   = {p.auc_last:.4g}")
        if p.auc_inf is not None:
            lines.append(f"  AUC(0-inf)    = {p.auc_inf:.4g}")
            lines.append(f"  %AUC extrap   = {p.auc_pct_extrap:.1f}%")
        if p.half_life is not None:
            lines.append(f"  t1/2          = {p.half_life:.4g}")
            lines.append(f"  lambda_z      = {p.lambda_z:.4g}")
            lines.append(f"  r-squared     = {p.lambda_z_r_squared:.4f}")
        if p.clearance is not None:
            label = "CL" if p.route == "iv" else "CL/F"
            lines.append(f"  {label:<14s} = {p.clearance:.4g}")
        if p.vz is not None:
            label = "Vz" if p.route == "iv" else "Vz/F"
            lines.append(f"  {label:<14s} = {p.vz:.4g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        p = self._result.params
        return (
            f"NCASolution(route={p.route!r}, cmax={p.cmax:.4g}, "
            f"auc_last={p.auc_last:.4g})"
        )
