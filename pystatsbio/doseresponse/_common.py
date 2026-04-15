"""Shared result types for dose-response modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CurveParams:
    """Parameters of a fitted dose-response curve.

    For 4PL: bottom + (top - bottom) / (1 + (ec50/x)^hill)
    """

    bottom: float
    top: float
    ec50: float
    hill: float
    asymmetry: float | None = None  # 5PL only
    hormesis: float | None = None  # BC.5 only
    model: str = "LL.4"

    def predict(self, dose: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict response at given dose levels."""
        from pystatsbio.doseresponse._models import _MODEL_MAP

        func, param_names = _MODEL_MAP[self.model]
        kwargs = {name: getattr(self, name) for name in param_names}
        return func(dose, **kwargs)

    def to_array(self) -> NDArray[np.floating]:
        """Return parameter vector in model order (for Jacobian indexing)."""
        from pystatsbio.doseresponse._models import _MODEL_MAP

        _, param_names = _MODEL_MAP[self.model]
        return np.array([getattr(self, name) for name in param_names], dtype=np.float64)

    @staticmethod
    def from_array(params: NDArray[np.floating], model: str) -> CurveParams:
        """Construct from parameter vector and model name."""
        from pystatsbio.doseresponse._models import _MODEL_MAP

        _, param_names = _MODEL_MAP[model]
        d = dict(zip(param_names, params, strict=True))
        return CurveParams(
            bottom=d["bottom"],
            top=d["top"],
            ec50=d["ec50"],
            hill=d["hill"],
            asymmetry=d.get("asymmetry"),
            hormesis=d.get("hormesis"),
            model=model,
        )


@dataclass(frozen=True)
class DoseResponseResult:
    """Result of fitting a single dose-response curve."""

    params: CurveParams
    se: NDArray[np.floating]  # standard errors of parameters
    residuals: NDArray[np.floating]
    rss: float
    aic: float
    bic: float
    converged: bool
    n_iter: int
    model: str  # e.g., "LL.4", "LL.5", "W1.4"
    dose: NDArray[np.floating]
    response: NDArray[np.floating]
    n_obs: int
    jac: NDArray[np.floating]  # Jacobian at solution (n_obs, n_params)

    def predict(self, dose: NDArray[np.floating] | None = None) -> NDArray[np.floating]:
        """Predict response.  If *dose* is ``None``, use the fitted dose."""
        if dose is None:
            dose = self.dose
        return self.params.predict(dose)

    def summary(self) -> str:
        """Human-readable summary, similar to R drc::summary()."""
        from pystatsbio.doseresponse._models import _MODEL_MAP

        _, param_names = _MODEL_MAP[self.model]
        lines = [
            f"Dose-response model: {self.model}",
            "",
            "Parameter estimates:",
        ]

        p_arr = self.params.to_array()
        for i, name in enumerate(param_names):
            val = p_arr[i]
            se_val = self.se[i] if i < len(self.se) else float("nan")
            t_val = val / se_val if se_val > 0 and not np.isnan(se_val) else float("nan")
            lines.append(f"  {name:>12s} = {val:>12.6f}  (SE = {se_val:.6f}, t = {t_val:.3f})")

        lines.append("")
        lines.append(f"  RSS = {self.rss:.6f}")
        lines.append(f"  AIC = {self.aic:.2f}")
        lines.append(f"  BIC = {self.bic:.2f}")
        lines.append(f"  n   = {self.n_obs}")
        lines.append(f"  Converged: {self.converged}")
        return "\n".join(lines)


@dataclass(frozen=True)
class BatchDoseResponseResult:
    """Result of batch-fitting dose-response curves (HTS).

    Each array has length n_compounds.
    """

    ec50: NDArray[np.floating]
    hill: NDArray[np.floating]
    top: NDArray[np.floating]
    bottom: NDArray[np.floating]
    converged: NDArray[np.bool_]
    rss: NDArray[np.floating]
    n_compounds: int
