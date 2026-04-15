"""
Diagnostic accuracy analysis for biomarker evaluation.

ROC analysis, sensitivity/specificity, predictive values, likelihood ratios,
and high-throughput batch AUC computation for biomarker panel screening.

Validates against: R packages pROC, OptimalCutpoints, epiR.
"""

from pystatsbio.diagnostic._accuracy import diagnostic_accuracy
from pystatsbio.diagnostic._batch import BatchAUCResult, batch_auc
from pystatsbio.diagnostic._common import DiagnosticResult, ROCResult
from pystatsbio.diagnostic._cutoff import CutoffResult, optimal_cutoff
from pystatsbio.diagnostic._roc import ROCTestResult, roc, roc_test

__all__ = [
    "ROCResult",
    "DiagnosticResult",
    "ROCTestResult",
    "CutoffResult",
    "BatchAUCResult",
    "roc",
    "roc_test",
    "diagnostic_accuracy",
    "optimal_cutoff",
    "batch_auc",
]
