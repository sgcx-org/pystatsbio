"""Shared result types for dose-response modeling.

``CurveParams`` is the fitted-curve value object (bottom/top/ec50/hill) and is
*not* a Solution — it is a nested value type like a measure. The ``XParams`` /
``XSolution`` pairs below follow the ecosystem envelope: each ``XParams`` is the
frozen payload of computed outputs; each ``XSolution`` wraps a
``core.result.Result[XParams]`` so every fit exposes the same ``.backend_name``
/ ``.timing`` / ``.warnings`` / ``.info`` metadata and Jupyter ``_repr_html_``
(``pystatsbio/CONVENTIONS.md`` B2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pystatistics.core.result import Result, SolutionReprMixin


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
class DoseResponseParams:
    """Computed payload of fitting a single dose-response curve.

    The fitted-curve value object lives in ``curve`` (a :class:`CurveParams`);
    the envelope reserves ``Result.params`` for this payload, so the field is
    named ``curve`` rather than ``params``.  The public
    :class:`DoseResponseSolution` re-exposes it as ``.params`` to preserve the
    existing API (``fit_result.params.ec50`` etc.).
    """

    curve: CurveParams
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


class DoseResponseSolution(SolutionReprMixin):
    """Public result of a single dose-response fit — wraps ``Result[DoseResponseParams]``.

    Exposes every fit output as a read-only property plus the uniform
    ``.backend_name`` / ``.timing`` / ``.warnings`` / ``.info`` metadata and a
    Jupyter ``_repr_html_`` (via :class:`SolutionReprMixin`).

    ``.params`` returns the fitted :class:`CurveParams`, preserving the public
    API that callers rely on (``fit_result.params.ec50`` / ``.to_array()``).
    """

    def __init__(self, result: Result[DoseResponseParams]) -> None:
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

    # --- Fit outputs (from the payload) ---
    @property
    def params(self) -> CurveParams:
        """The fitted dose-response curve (bottom/top/ec50/hill)."""
        return self._result.params.curve

    @property
    def se(self) -> NDArray[np.floating]:
        return self._result.params.se

    @property
    def residuals(self) -> NDArray[np.floating]:
        return self._result.params.residuals

    @property
    def rss(self) -> float:
        return self._result.params.rss

    @property
    def aic(self) -> float:
        return self._result.params.aic

    @property
    def bic(self) -> float:
        return self._result.params.bic

    @property
    def converged(self) -> bool:
        return self._result.params.converged

    @property
    def n_iter(self) -> int:
        return self._result.params.n_iter

    @property
    def model(self) -> str:
        return self._result.params.model

    @property
    def dose(self) -> NDArray[np.floating]:
        return self._result.params.dose

    @property
    def response(self) -> NDArray[np.floating]:
        return self._result.params.response

    @property
    def n_obs(self) -> int:
        return self._result.params.n_obs

    @property
    def jac(self) -> NDArray[np.floating]:
        return self._result.params.jac

    def predict(self, dose: NDArray[np.floating] | None = None) -> NDArray[np.floating]:
        """Predict response.  If *dose* is ``None``, use the fitted dose."""
        p = self._result.params
        if dose is None:
            dose = p.dose
        return p.curve.predict(dose)

    def summary(self) -> str:
        """Human-readable summary, similar to R drc::summary()."""
        from pystatsbio.doseresponse._models import _MODEL_MAP

        p = self._result.params
        _, param_names = _MODEL_MAP[p.model]
        lines = [
            f"Dose-response model: {p.model}",
            "",
            "Parameter estimates:",
        ]

        p_arr = p.curve.to_array()
        for i, name in enumerate(param_names):
            val = p_arr[i]
            se_val = p.se[i] if i < len(p.se) else float("nan")
            t_val = val / se_val if se_val > 0 and not np.isnan(se_val) else float("nan")
            lines.append(f"  {name:>12s} = {val:>12.6f}  (SE = {se_val:.6f}, t = {t_val:.3f})")

        lines.append("")
        lines.append(f"  RSS = {p.rss:.6f}")
        lines.append(f"  AIC = {p.aic:.2f}")
        lines.append(f"  BIC = {p.bic:.2f}")
        lines.append(f"  n   = {p.n_obs}")
        lines.append(f"  Converged: {p.converged}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        p = self._result.params
        return (
            f"DoseResponseSolution(model={p.model!r}, ec50={p.curve.ec50:.4g}, "
            f"n_obs={p.n_obs}, converged={p.converged})"
        )


@dataclass(frozen=True)
class BatchDoseResponseParams:
    """Computed payload of batch-fitting dose-response curves (HTS).

    Each array has length n_compounds.
    """

    ec50: NDArray[np.floating]
    hill: NDArray[np.floating]
    top: NDArray[np.floating]
    bottom: NDArray[np.floating]
    converged: NDArray[np.bool_]
    rss: NDArray[np.floating]
    n_compounds: int


class BatchDoseResponseSolution(SolutionReprMixin):
    """Public result of batch dose-response fitting — wraps ``Result[BatchDoseResponseParams]``.

    Exposes every output as a read-only property plus the uniform
    ``.backend_name`` / ``.timing`` / ``.warnings`` / ``.info`` metadata and a
    Jupyter ``_repr_html_`` (via :class:`SolutionReprMixin`).
    """

    def __init__(self, result: Result[BatchDoseResponseParams]) -> None:
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

    # --- Batch outputs (from the payload) ---
    @property
    def ec50(self) -> NDArray[np.floating]:
        return self._result.params.ec50

    @property
    def hill(self) -> NDArray[np.floating]:
        return self._result.params.hill

    @property
    def top(self) -> NDArray[np.floating]:
        return self._result.params.top

    @property
    def bottom(self) -> NDArray[np.floating]:
        return self._result.params.bottom

    @property
    def converged(self) -> NDArray[np.bool_]:
        return self._result.params.converged

    @property
    def rss(self) -> NDArray[np.floating]:
        return self._result.params.rss

    @property
    def n_compounds(self) -> int:
        return self._result.params.n_compounds

    def __repr__(self) -> str:
        p = self._result.params
        n_conv = int(np.sum(p.converged))
        return (
            f"BatchDoseResponseSolution(n_compounds={p.n_compounds}, "
            f"converged={n_conv}/{p.n_compounds})"
        )
