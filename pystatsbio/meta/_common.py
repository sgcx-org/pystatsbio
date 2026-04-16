"""Shared result types for meta-analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class MetaResult:
    """Result from a meta-analysis.

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

    def summary(self) -> str:
        """R-style summary matching metafor::rma() output.

        Returns
        -------
        str
            Multi-line human-readable summary of the meta-analysis.
        """
        method_labels = {
            "FE": "Fixed-Effects Model",
            "DL": "Random-Effects Model (DerSimonian-Laird)",
            "REML": "Random-Effects Model (REML)",
            "PM": "Random-Effects Model (Paule-Mandel)",
        }
        label = method_labels.get(self.method, self.method)
        lines = [
            label,
            "=" * 60,
            "",
            f"Number of studies: k = {self.k}",
            "",
        ]
        if self.method != "FE":
            lines.append(
                f"tau2 (estimated amount of total heterogeneity): "
                f"{self.tau2:.4f}"
                + (f" (SE = {self.tau2_se:.4f})" if self.tau2_se is not None else "")
            )
            lines.append(f"tau  (sqrt of tau2):                            {self.tau:.4f}")
            lines.append(f"I2   (total heterogeneity / total variability): {self.I2:.2f}%")
            lines.append(f"H2   (total variability / sampling variability): {self.H2:.2f}")
            lines.append("")

        lines.append("Test for Heterogeneity:")
        lines.append(
            f"Q(df = {self.Q_df}) = {self.Q:.4f}, p-val "
            + (f"< .0001" if self.Q_p < 0.0001 else f"= {self.Q_p:.4f}")
        )
        lines.append("")
        lines.append("Model Results:")
        lines.append(
            f"estimate = {self.estimate:.4f}, se = {self.se:.4f}, "
            f"z = {self.z_value:.4f}, "
            f"p " + (f"< .0001" if self.p_value < 0.0001 else f"= {self.p_value:.4f}")
        )
        ci_pct = f"{self.conf_level:.0%}"
        lines.append(
            f"{ci_pct} CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        )
        return "\n".join(lines)


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
    ValueError
        If inputs are invalid.
    """
    yi_arr = np.asarray(yi, dtype=np.float64)
    vi_arr = np.asarray(vi, dtype=np.float64)

    if yi_arr.ndim != 1:
        raise ValueError(f"yi must be 1-D, got {yi_arr.ndim}-D")
    if vi_arr.ndim != 1:
        raise ValueError(f"vi must be 1-D, got {vi_arr.ndim}-D")
    if yi_arr.shape[0] != vi_arr.shape[0]:
        raise ValueError(
            f"yi and vi must have the same length, "
            f"got {yi_arr.shape[0]} and {vi_arr.shape[0]}"
        )
    if yi_arr.shape[0] < 2:
        raise ValueError(
            f"meta-analysis requires at least 2 studies, got {yi_arr.shape[0]}"
        )
    if np.any(vi_arr < 0):
        raise ValueError("vi must contain non-negative values")
    if np.any(vi_arr == 0):
        raise ValueError(
            "vi contains zero variances; fixed-effects weights would be infinite"
        )
    if not (0 < conf_level < 1):
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")
    if np.any(np.isnan(yi_arr)) or np.any(np.isnan(vi_arr)):
        raise ValueError("yi and vi must not contain NaN values")
    if np.any(np.isinf(yi_arr)) or np.any(np.isinf(vi_arr)):
        raise ValueError("yi and vi must not contain infinite values")

    return yi_arr, vi_arr
