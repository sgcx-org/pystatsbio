"""
Diagnostic accuracy analysis for biomarker evaluation.

ROC analysis, sensitivity/specificity, predictive values, likelihood ratios,
and high-throughput batch AUC computation for biomarker panel screening.

Validates against: R packages pROC, OptimalCutpoints, epiR.
"""

from pystatsbio.diagnostic._accuracy import diagnostic_accuracy
from pystatsbio.diagnostic._batch import BatchAUCParams, BatchAUCSolution, batch_auc
from pystatsbio.diagnostic._common import (
    DiagnosticParams,
    DiagnosticSolution,
    ROCParams,
    ROCSolution,
)
from pystatsbio.diagnostic._cutoff import CutoffParams, CutoffSolution, optimal_cutoff
from pystatsbio.diagnostic._roc import (
    ROCTestParams,
    ROCTestSolution,
    roc,
    roc_test,
)

__all__ = [
    "ROCParams",
    "ROCSolution",
    "DiagnosticParams",
    "DiagnosticSolution",
    "ROCTestParams",
    "ROCTestSolution",
    "CutoffParams",
    "CutoffSolution",
    "BatchAUCParams",
    "BatchAUCSolution",
    "roc",
    "roc_test",
    "diagnostic_accuracy",
    "optimal_cutoff",
    "batch_auc",
]
