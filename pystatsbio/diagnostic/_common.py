"""Shared result types for diagnostic accuracy analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ROCResult:
    """Result of ROC analysis.

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

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "ROC Analysis",
            "=" * 40,
            f"Direction   : controls {self.direction} cases",
            f"AUC         : {self.auc:.4f}",
            f"DeLong SE   : {self.auc_se:.4f}",
            f"{self.conf_level:.0%} CI      : [{self.auc_ci_lower:.4f}, {self.auc_ci_upper:.4f}]",
            f"n positive  : {self.n_positive}",
            f"n negative  : {self.n_negative}",
            f"n thresholds: {len(self.thresholds)}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class DiagnosticResult:
    """Result of diagnostic accuracy evaluation at a fixed cutoff.

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

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Diagnostic Accuracy",
            "=" * 40,
            f"Cutoff        : {self.cutoff:.4g}",
            f"Sensitivity   : {self.sensitivity:.4f}  "
            f"({self.conf_level:.0%} CI: "
            f"{self.sensitivity_ci[0]:.4f}–{self.sensitivity_ci[1]:.4f})",
            f"Specificity   : {self.specificity:.4f}  "
            f"({self.conf_level:.0%} CI: "
            f"{self.specificity_ci[0]:.4f}–{self.specificity_ci[1]:.4f})",
            f"PPV           : {self.ppv:.4f}",
            f"NPV           : {self.npv:.4f}",
            f"LR+           : {self.lr_positive:.4f}",
            f"LR−           : {self.lr_negative:.4f}",
            f"DOR           : {self.dor:.4f}  "
            f"({self.conf_level:.0%} CI: {self.dor_ci[0]:.4f}–{self.dor_ci[1]:.4f})",
            f"Prevalence    : {self.prevalence:.4f}",
            f"CI method     : {self.method}",
        ]
        return "\n".join(lines)
