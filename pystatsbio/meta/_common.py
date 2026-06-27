"""Result types for meta-analysis.

``MetaParams`` is the frozen payload of computed outputs; ``MetaSolution`` is
the public return — it wraps a ``core.result.Result[MetaParams]`` so every
meta-analysis exposes the same ``.backend_name`` / ``.timing`` / ``.warnings`` /
``.info`` metadata and Jupyter ``_repr_html_`` as the rest of the ecosystem
(``pystatsbio/CONVENTIONS.md`` B2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pystatistics.core.exceptions import ValidationError
from pystatistics.core.result import Result, SolutionReprMixin


@dataclass(frozen=True)
class MetaParams:
    """Computed payload of a meta-analysis.

    Attributes
    ----------
    estimate : float
        Pooled effect size.
    se : float
        Standard error of the pooled estimate.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    z_value : float
        Test statistic for the pooled estimate (estimate / se).
    p_value : float
        Two-sided p-value for the test of estimate = 0.
    tau2 : float
        Between-study variance (0 for fixed-effects).
    tau2_se : float or None
        Standard error of tau2 (method-dependent; None if unavailable).
    tau : float
        Square root of tau2.
    I2 : float
        I-squared heterogeneity statistic (percentage, 0--100).
    H2 : float
        H-squared statistic (ratio of total to sampling variability).
    Q : float
        Cochran's Q statistic for heterogeneity.
    Q_df : int
        Degrees of freedom for the Q test.
    Q_p : float
        p-value for the Q test (chi-squared distribution).
    k : int
        Number of studies.
    method : str
        Estimation method: 'FE', 'DL', 'REML', or 'PM'.
    conf_level : float
        Confidence level used for intervals.
    weights : NDArray
        Study weights used in the final pooling.
    yi : NDArray
        Input effect sizes.
    vi : NDArray
        Input sampling variances.
    """

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    z_value: float
    p_value: float
    tau2: float
    tau2_se: float | None
    tau: float
    I2: float
    H2: float
    Q: float
    Q_df: int
    Q_p: float
    k: int
    method: str
    conf_level: float
    weights: NDArray
    yi: NDArray
    vi: NDArray


class MetaSolution(SolutionReprMixin):
    """Public result of a meta-analysis — a Solution wrapping ``Result[MetaParams]``.

    Exposes every fit output as a read-only property plus the uniform
    ``.backend_name`` / ``.timing`` / ``.warnings`` / ``.info`` metadata and a
    Jupyter ``_repr_html_`` (via :class:`SolutionReprMixin`).
    """

    def __init__(self, result: Result[MetaParams]) -> None:
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
    def estimate(self) -> float:
        return self._result.params.estimate

    @property
    def se(self) -> float:
        return self._result.params.se

    @property
    def ci_lower(self) -> float:
        return self._result.params.ci_lower

    @property
    def ci_upper(self) -> float:
        return self._result.params.ci_upper

    @property
    def z_value(self) -> float:
        return self._result.params.z_value

    @property
    def p_value(self) -> float:
        return self._result.params.p_value

    @property
    def tau2(self) -> float:
        return self._result.params.tau2

    @property
    def tau2_se(self) -> float | None:
        return self._result.params.tau2_se

    @property
    def tau(self) -> float:
        return self._result.params.tau

    @property
    def I2(self) -> float:
        return self._result.params.I2

    @property
    def H2(self) -> float:
        return self._result.params.H2

    @property
    def Q(self) -> float:
        return self._result.params.Q

    @property
    def Q_df(self) -> int:
        return self._result.params.Q_df

    @property
    def Q_p(self) -> float:
        return self._result.params.Q_p

    @property
    def k(self) -> int:
        return self._result.params.k

    @property
    def method(self) -> str:
        return self._result.params.method

    @property
    def conf_level(self) -> float:
        return self._result.params.conf_level

    @property
    def weights(self) -> NDArray:
        return self._result.params.weights

    @property
    def yi(self) -> NDArray:
        return self._result.params.yi

    @property
    def vi(self) -> NDArray:
        return self._result.params.vi

    def summary(self) -> str:
        """R-style summary matching metafor::rma() output.

        Returns
        -------
        str
            Multi-line human-readable summary of the meta-analysis.
        """
        p = self._result.params
        method_labels = {
            "FE": "Fixed-Effects Model",
            "DL": "Random-Effects Model (DerSimonian-Laird)",
            "REML": "Random-Effects Model (REML)",
            "PM": "Random-Effects Model (Paule-Mandel)",
        }
        label = method_labels.get(p.method, p.method)
        lines = [
            label,
            "=" * 60,
            "",
            f"Number of studies: k = {p.k}",
            "",
        ]
        if p.method != "FE":
            lines.append(
                f"tau2 (estimated amount of total heterogeneity): "
                f"{p.tau2:.4f}"
                + (f" (SE = {p.tau2_se:.4f})" if p.tau2_se is not None else "")
            )
            lines.append(f"tau  (sqrt of tau2):                            {p.tau:.4f}")
            lines.append(f"I2   (total heterogeneity / total variability): {p.I2:.2f}%")
            lines.append(f"H2   (total variability / sampling variability): {p.H2:.2f}")
            lines.append("")

        lines.append("Test for Heterogeneity:")
        lines.append(
            f"Q(df = {p.Q_df}) = {p.Q:.4f}, p-val "
            + ("< .0001" if p.Q_p < 0.0001 else f"= {p.Q_p:.4f}")
        )
        lines.append("")
        lines.append("Model Results:")
        lines.append(
            f"estimate = {p.estimate:.4f}, se = {p.se:.4f}, "
            f"z = {p.z_value:.4f}, "
            f"p " + ("< .0001" if p.p_value < 0.0001 else f"= {p.p_value:.4f}")
        )
        ci_pct = f"{p.conf_level:.0%}"
        lines.append(
            f"{ci_pct} CI: [{p.ci_lower:.4f}, {p.ci_upper:.4f}]"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        p = self._result.params
        return (
            f"MetaSolution(method={p.method!r}, k={p.k}, "
            f"estimate={p.estimate:.4f}, tau2={p.tau2:.4f})"
        )


def validate_inputs(
    yi: ArrayLike,
    vi: ArrayLike,
    conf_level: float,
) -> tuple[NDArray, NDArray]:
    """Validate meta-analysis inputs and return float64 arrays.

    Parameters
    ----------
    yi : ArrayLike
        Effect sizes.
    vi : ArrayLike
        Sampling variances.
    conf_level : float
        Confidence level (must be in (0, 1)).

    Returns
    -------
    tuple of NDArray
        Validated (yi, vi) as float64 arrays.

    Raises
    ------
    ValidationError
        If inputs are invalid.
    """
    yi_arr = np.asarray(yi, dtype=np.float64)
    vi_arr = np.asarray(vi, dtype=np.float64)

    if yi_arr.ndim != 1:
        raise ValidationError(f"yi must be 1-D, got {yi_arr.ndim}-D")
    if vi_arr.ndim != 1:
        raise ValidationError(f"vi must be 1-D, got {vi_arr.ndim}-D")
    if yi_arr.shape[0] != vi_arr.shape[0]:
        raise ValidationError(
            f"yi and vi must have the same length, "
            f"got {yi_arr.shape[0]} and {vi_arr.shape[0]}"
        )
    if yi_arr.shape[0] < 2:
        raise ValidationError(
            f"meta-analysis requires at least 2 studies, got {yi_arr.shape[0]}"
        )
    if np.any(vi_arr < 0):
        raise ValidationError("vi must contain non-negative values")
    if np.any(vi_arr == 0):
        raise ValidationError(
            "vi contains zero variances; fixed-effects weights would be infinite"
        )
    if not (0 < conf_level < 1):
        raise ValidationError(f"conf_level must be in (0, 1), got {conf_level}")
    if np.any(np.isnan(yi_arr)) or np.any(np.isnan(vi_arr)):
        raise ValidationError("yi and vi must not contain NaN values")
    if np.any(np.isinf(yi_arr)) or np.any(np.isinf(vi_arr)):
        raise ValidationError("yi and vi must not contain infinite values")

    return yi_arr, vi_arr
