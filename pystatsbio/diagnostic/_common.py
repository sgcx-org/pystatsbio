"""Shared result types for diagnostic accuracy analysis.

Each ``XParams`` is the frozen payload of computed outputs; each ``XSolution``
is the public return — it wraps a ``core.result.Result[XParams]`` so every
diagnostic call exposes the same ``.backend_name`` / ``.timing`` / ``.warnings``
/ ``.info`` metadata and Jupyter ``_repr_html_`` as the rest of the ecosystem
(``pystatsbio/CONVENTIONS.md`` B2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pystatistics.core.result import Result, SolutionReprMixin


@dataclass(frozen=True)
class ROCParams:
    """Computed payload of ROC analysis.

    Attributes
    ----------
    thresholds : array
        Thresholds at which TPR/FPR are evaluated.  Includes ``-inf``
        and ``+inf`` so the curve always passes through (0,0) and (1,1).
    tpr : array
        True positive rate (sensitivity) at each threshold.
    fpr : array
        False positive rate (1 − specificity) at each threshold.
    auc : float
        Area under the ROC curve (Mann-Whitney U / (n1*n0)).
    auc_se : float
        DeLong standard error of the AUC.
    auc_ci_lower, auc_ci_upper : float
        Confidence interval for AUC (logit-transformed DeLong).
    conf_level : float
        Confidence level used for CI.
    n_positive, n_negative : int
        Number of positive (case) and negative (control) observations.
    direction : str
        ``'<'`` (controls < cases) or ``'>'`` (controls > cases).
    """

    thresholds: NDArray[np.floating]
    tpr: NDArray[np.floating]  # sensitivity / true positive rate
    fpr: NDArray[np.floating]  # 1 - specificity / false positive rate
    auc: float
    auc_se: float  # DeLong standard error
    auc_ci_lower: float
    auc_ci_upper: float
    conf_level: float
    n_positive: int
    n_negative: int
    direction: str  # '<' or '>'


class ROCSolution(SolutionReprMixin):
    """Public result of ROC analysis — a Solution wrapping ``Result[ROCParams]``.

    Exposes every output as a read-only property plus the uniform
    ``.backend_name`` / ``.timing`` / ``.warnings`` / ``.info`` metadata and a
    Jupyter ``_repr_html_`` (via :class:`SolutionReprMixin`).
    """

    def __init__(self, result: Result[ROCParams]) -> None:
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

    # --- ROC outputs (from the payload) ---
    @property
    def thresholds(self) -> NDArray[np.floating]:
        return self._result.params.thresholds

    @property
    def tpr(self) -> NDArray[np.floating]:
        return self._result.params.tpr

    @property
    def fpr(self) -> NDArray[np.floating]:
        return self._result.params.fpr

    @property
    def auc(self) -> float:
        return self._result.params.auc

    @property
    def auc_se(self) -> float:
        return self._result.params.auc_se

    @property
    def auc_ci_lower(self) -> float:
        return self._result.params.auc_ci_lower

    @property
    def auc_ci_upper(self) -> float:
        return self._result.params.auc_ci_upper

    @property
    def conf_level(self) -> float:
        return self._result.params.conf_level

    @property
    def n_positive(self) -> int:
        return self._result.params.n_positive

    @property
    def n_negative(self) -> int:
        return self._result.params.n_negative

    @property
    def direction(self) -> str:
        return self._result.params.direction

    def summary(self) -> str:
        """Human-readable summary."""
        p = self._result.params
        lines = [
            "ROC Analysis",
            "=" * 40,
            f"Direction   : controls {p.direction} cases",
            f"AUC         : {p.auc:.4f}",
            f"DeLong SE   : {p.auc_se:.4f}",
            f"{p.conf_level:.0%} CI      : [{p.auc_ci_lower:.4f}, {p.auc_ci_upper:.4f}]",
            f"n positive  : {p.n_positive}",
            f"n negative  : {p.n_negative}",
            f"n thresholds: {len(p.thresholds)}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        p = self._result.params
        return (
            f"ROCSolution(auc={p.auc:.4f}, direction={p.direction!r}, "
            f"n_positive={p.n_positive}, n_negative={p.n_negative})"
        )


@dataclass(frozen=True)
class DiagnosticParams:
    """Computed payload of diagnostic accuracy evaluation at a fixed cutoff.

    All CIs use the method specified in ``method`` (e.g.
    ``'clopper-pearson'`` for exact binomial CIs).
    """

    cutoff: float
    sensitivity: float
    sensitivity_ci: tuple[float, float]
    specificity: float
    specificity_ci: tuple[float, float]
    ppv: float
    npv: float
    lr_positive: float
    lr_negative: float
    dor: float  # diagnostic odds ratio
    dor_ci: tuple[float, float]
    prevalence: float
    conf_level: float
    method: str  # CI method, e.g. 'clopper-pearson'


class DiagnosticSolution(SolutionReprMixin):
    """Public result of diagnostic accuracy — wraps ``Result[DiagnosticParams]``.

    Exposes every output as a read-only property plus the uniform
    ``.backend_name`` / ``.timing`` / ``.warnings`` / ``.info`` metadata and a
    Jupyter ``_repr_html_`` (via :class:`SolutionReprMixin`).
    """

    def __init__(self, result: Result[DiagnosticParams]) -> None:
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

    # --- Diagnostic outputs (from the payload) ---
    @property
    def cutoff(self) -> float:
        return self._result.params.cutoff

    @property
    def sensitivity(self) -> float:
        return self._result.params.sensitivity

    @property
    def sensitivity_ci(self) -> tuple[float, float]:
        return self._result.params.sensitivity_ci

    @property
    def specificity(self) -> float:
        return self._result.params.specificity

    @property
    def specificity_ci(self) -> tuple[float, float]:
        return self._result.params.specificity_ci

    @property
    def ppv(self) -> float:
        return self._result.params.ppv

    @property
    def npv(self) -> float:
        return self._result.params.npv

    @property
    def lr_positive(self) -> float:
        return self._result.params.lr_positive

    @property
    def lr_negative(self) -> float:
        return self._result.params.lr_negative

    @property
    def dor(self) -> float:
        return self._result.params.dor

    @property
    def dor_ci(self) -> tuple[float, float]:
        return self._result.params.dor_ci

    @property
    def prevalence(self) -> float:
        return self._result.params.prevalence

    @property
    def conf_level(self) -> float:
        return self._result.params.conf_level

    @property
    def method(self) -> str:
        return self._result.params.method

    def summary(self) -> str:
        """Human-readable summary."""
        p = self._result.params
        lines = [
            "Diagnostic Accuracy",
            "=" * 40,
            f"Cutoff        : {p.cutoff:.4g}",
            f"Sensitivity   : {p.sensitivity:.4f}  "
            f"({p.conf_level:.0%} CI: "
            f"{p.sensitivity_ci[0]:.4f}–{p.sensitivity_ci[1]:.4f})",
            f"Specificity   : {p.specificity:.4f}  "
            f"({p.conf_level:.0%} CI: "
            f"{p.specificity_ci[0]:.4f}–{p.specificity_ci[1]:.4f})",
            f"PPV           : {p.ppv:.4f}",
            f"NPV           : {p.npv:.4f}",
            f"LR+           : {p.lr_positive:.4f}",
            f"LR−           : {p.lr_negative:.4f}",
            f"DOR           : {p.dor:.4f}  "
            f"({p.conf_level:.0%} CI: {p.dor_ci[0]:.4f}–{p.dor_ci[1]:.4f})",
            f"Prevalence    : {p.prevalence:.4f}",
            f"CI method     : {p.method}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        p = self._result.params
        return (
            f"DiagnosticSolution(cutoff={p.cutoff:.4g}, "
            f"sensitivity={p.sensitivity:.4f}, "
            f"specificity={p.specificity:.4f})"
        )
