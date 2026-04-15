"""Shared result types for pharmacokinetic analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NCAResult:
    """Result of non-compartmental pharmacokinetic analysis."""

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

    def summary(self) -> str:
        """Human-readable PK summary."""
        lines = ["Non-Compartmental Analysis", ""]
        lines.append(f"  Route: {self.route.upper()}")
        lines.append(f"  AUC method: {self.auc_method}")
        if self.dose is not None:
            lines.append(f"  Dose: {self.dose}")
        lines.append("")
        lines.append(f"  Cmax          = {self.cmax:.4g}")
        lines.append(f"  Tmax          = {self.tmax:.4g}")
        lines.append(f"  AUC(0-last)   = {self.auc_last:.4g}")
        if self.auc_inf is not None:
            lines.append(f"  AUC(0-inf)    = {self.auc_inf:.4g}")
            lines.append(f"  %AUC extrap   = {self.auc_pct_extrap:.1f}%")
        if self.half_life is not None:
            lines.append(f"  t1/2          = {self.half_life:.4g}")
            lines.append(f"  lambda_z      = {self.lambda_z:.4g}")
            lines.append(f"  r-squared     = {self.lambda_z_r_squared:.4f}")
        if self.clearance is not None:
            label = "CL" if self.route == "iv" else "CL/F"
            lines.append(f"  {label:<14s} = {self.clearance:.4g}")
        if self.vz is not None:
            label = "Vz" if self.route == "iv" else "Vz/F"
            lines.append(f"  {label:<14s} = {self.vz:.4g}")
        return "\n".join(lines)
