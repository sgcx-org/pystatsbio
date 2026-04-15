"""ROC curve analysis with DeLong confidence intervals and comparison test.

Implements the empirical ROC curve, AUC via Mann-Whitney U, DeLong
standard errors and CIs (logit-transformed), and the DeLong test for
comparing two correlated ROC curves.

References
----------
DeLong, DeLong & Clarke-Pearson (1988). Comparing the areas under two
or more correlated receiver operating characteristic curves: a
nonparametric approach.  *Biometrics*, 44(3), 837-845.

Validates against: R ``pROC::roc()``, ``pROC::ci.auc()``,
``pROC::roc.test()``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from pystatsbio.diagnostic._common import ROCResult

# ---------------------------------------------------------------------------
# ROCTestResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ROCTestResult:
    """Result of comparing two correlated ROC curves (DeLong test)."""

    statistic: float
    p_value: float
    auc1: float
    auc2: float
    auc_diff: float
    method: str  # 'delong'

    def summary(self) -> str:
        lines = [
            "DeLong Test for Two Correlated ROC Curves",
            "=" * 45,
            f"AUC 1     : {self.auc1:.4f}",
            f"AUC 2     : {self.auc2:.4f}",
            f"Difference: {self.auc_diff:.4f}",
            f"Z         : {self.statistic:.4f}",
            f"p-value   : {self.p_value:.4g}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_roc_inputs(
    response: NDArray, predictor: NDArray
) -> tuple[NDArray, NDArray]:
    """Validate and coerce inputs for ROC analysis."""
    response = np.asarray(response, dtype=np.intp)
    predictor = np.asarray(predictor, dtype=np.float64)

    if response.ndim != 1 or predictor.ndim != 1:
        raise ValueError("response and predictor must be 1-D arrays")
    if response.shape[0] != predictor.shape[0]:
        raise ValueError(
            f"response and predictor must have the same length, "
            f"got {response.shape[0]} and {predictor.shape[0]}"
        )

    unique_labels = np.unique(response)
    if not np.array_equal(unique_labels, np.array([0, 1])):
        raise ValueError(
            f"response must be binary (0/1), got unique values {unique_labels}"
        )

    n1 = int(response.sum())
    n0 = len(response) - n1
    if n1 < 1 or n0 < 1:
        raise ValueError("Need at least one case and one control")

    return response, predictor


def _resolve_direction(
    response: NDArray, predictor: NDArray, direction: str,
) -> str:
    """Choose direction if 'auto'."""
    if direction == "auto":
        med_cases = np.median(predictor[response == 1])
        med_controls = np.median(predictor[response == 0])
        return "<" if med_controls <= med_cases else ">"
    if direction not in ("<", ">"):
        raise ValueError(f"direction must be '<', '>' or 'auto', got {direction!r}")
    return direction


def _compute_auc_and_placements(
    response: NDArray,
    predictor: NDArray,
    direction: str,
) -> tuple[float, NDArray, NDArray]:
    """Compute AUC via ranks and DeLong placement values.

    Returns (auc, V10, V01) where V10 has shape (n1,) and V01 has
    shape (n0,).
    """
    # If direction is '>', negate predictor so higher = positive
    if direction == ">":
        predictor = -predictor

    case_mask = response == 1
    n1 = int(case_mask.sum())
    n0 = len(response) - n1

    # Pooled ranks (midrank for ties)
    pooled_ranks = stats.rankdata(predictor, method="average")

    # Ranks among cases only and controls only
    case_vals = predictor[case_mask]
    ctrl_vals = predictor[~case_mask]

    case_ranks_within = stats.rankdata(case_vals, method="average")
    ctrl_ranks_within = stats.rankdata(ctrl_vals, method="average")

    # AUC = (sum of case ranks in pooled - n1*(n1+1)/2) / (n1*n0)
    sum_case_ranks = pooled_ranks[case_mask].sum()
    auc = (sum_case_ranks - n1 * (n1 + 1) / 2) / (n1 * n0)

    # Placement values via rank difference
    # V10[i] = (pooled_rank_i - within_case_rank_i) / n0
    V10 = (pooled_ranks[case_mask] - case_ranks_within) / n0

    # V01[j] = 1 - (pooled_rank_j - within_ctrl_rank_j) / n1
    # For controls, the "number of cases beaten" is:
    # V01[j] = (1 - direction-corrected placement)
    # Actually: V01[j] measures how many cases the control "loses" to.
    # V01[j] = 1 - (pooled_rank_j - within_ctrl_rank_j) / n1
    V01 = 1.0 - (pooled_ranks[~case_mask] - ctrl_ranks_within) / n1

    return float(auc), V10, V01


def _delong_variance(
    auc: float, V10: NDArray, V01: NDArray, n1: int, n0: int,
) -> float:
    """DeLong variance of AUC from placement values."""
    S10 = np.var(V10, ddof=1) if n1 > 1 else 0.0
    S01 = np.var(V01, ddof=1) if n0 > 1 else 0.0
    return S10 / n1 + S01 / n0


def _logit_ci(
    auc: float, var_auc: float, conf_level: float,
) -> tuple[float, float]:
    """AUC confidence interval on logit scale (matching pROC default)."""
    z = stats.norm.ppf((1 + conf_level) / 2)

    # Clamp AUC away from 0/1 to avoid log(0)
    auc_c = np.clip(auc, 1e-10, 1.0 - 1e-10)

    logit_auc = np.log(auc_c / (1.0 - auc_c))
    se_logit = np.sqrt(var_auc) / (auc_c * (1.0 - auc_c))

    logit_lo = logit_auc - z * se_logit
    logit_hi = logit_auc + z * se_logit

    ci_lo = 1.0 / (1.0 + np.exp(-logit_lo))
    ci_hi = 1.0 / (1.0 + np.exp(-logit_hi))

    return float(ci_lo), float(ci_hi)


# ---------------------------------------------------------------------------
# Public API — roc()
# ---------------------------------------------------------------------------

def roc(
    response: NDArray[np.integer],
    predictor: NDArray[np.floating],
    *,
    direction: str = "auto",
    conf_level: float = 0.95,
) -> ROCResult:
    """Compute empirical ROC curve with DeLong AUC confidence interval.

    Parameters
    ----------
    response : array of int
        Binary outcome (0/1).
    predictor : array of float
        Continuous predictor (biomarker value).
    direction : str
        ``'<'`` (controls < cases, higher predictor → positive),
        ``'>'`` (controls > cases, lower predictor → positive),
        or ``'auto'`` (choose direction giving AUC ≥ 0.5).
    conf_level : float
        Confidence level for AUC CI.

    Returns
    -------
    ROCResult

    Validates against: R ``pROC::roc()``, ``pROC::ci.auc()``
    """
    if not 0 < conf_level < 1:
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")

    response, predictor = _validate_roc_inputs(response, predictor)
    direction = _resolve_direction(response, predictor, direction)

    n1 = int(response.sum())
    n0 = len(response) - n1

    # AUC and placement values
    auc_val, V10, V01 = _compute_auc_and_placements(
        response, predictor, direction,
    )

    # DeLong SE and CI
    var_auc = _delong_variance(auc_val, V10, V01, n1, n0)
    se_auc = float(np.sqrt(var_auc))
    ci_lo, ci_hi = _logit_ci(auc_val, var_auc, conf_level)

    # Build empirical ROC curve (thresholds, TPR, FPR)
    thresholds, tpr, fpr = _empirical_roc_curve(
        response, predictor, direction,
    )

    return ROCResult(
        thresholds=thresholds,
        tpr=tpr,
        fpr=fpr,
        auc=auc_val,
        auc_se=se_auc,
        auc_ci_lower=ci_lo,
        auc_ci_upper=ci_hi,
        conf_level=conf_level,
        n_positive=n1,
        n_negative=n0,
        direction=direction,
    )


def _empirical_roc_curve(
    response: NDArray,
    predictor: NDArray,
    direction: str,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute empirical ROC curve points.

    Returns (thresholds, tpr, fpr) sorted from (0,0) to (1,1).
    """
    case_mask = response == 1
    n1 = int(case_mask.sum())
    n0 = len(response) - n1

    # Unique thresholds sorted descending (for direction '<')
    unique_vals = np.unique(predictor)

    if direction == "<":
        # Higher predictor = positive: classify positive if predictor >= c
        # Sort thresholds from high to low
        sorted_thresh = np.sort(unique_vals)[::-1]
        tpr_list = []
        fpr_list = []
        for c in sorted_thresh:
            tpr_list.append(np.sum(predictor[case_mask] >= c) / n1)
            fpr_list.append(np.sum(predictor[~case_mask] >= c) / n0)

        # Prepend (0, 0) — at threshold above max, nobody classified positive
        # Append (1, 1) — at threshold below min, everybody classified positive
        thresholds = np.concatenate([
            [np.inf], sorted_thresh, [-np.inf],
        ])
        tpr_arr = np.array([0.0] + tpr_list + [1.0])
        fpr_arr = np.array([0.0] + fpr_list + [1.0])
    else:
        # Lower predictor = positive: classify positive if predictor <= c
        sorted_thresh = np.sort(unique_vals)
        tpr_list = []
        fpr_list = []
        for c in sorted_thresh:
            tpr_list.append(np.sum(predictor[case_mask] <= c) / n1)
            fpr_list.append(np.sum(predictor[~case_mask] <= c) / n0)

        thresholds = np.concatenate([
            [-np.inf], sorted_thresh, [np.inf],
        ])
        tpr_arr = np.array([0.0] + tpr_list + [1.0])
        fpr_arr = np.array([0.0] + fpr_list + [1.0])

    return thresholds, tpr_arr, fpr_arr


# ---------------------------------------------------------------------------
# Public API — roc_test()
# ---------------------------------------------------------------------------

def roc_test(
    roc1: ROCResult,
    roc2: ROCResult,
    *,
    predictor1: NDArray[np.floating] | None = None,
    predictor2: NDArray[np.floating] | None = None,
    response: NDArray[np.integer] | None = None,
    method: str = "delong",
) -> ROCTestResult:
    """Compare two correlated ROC curves using DeLong's test.

    The two ROC curves must be computed on the **same** subjects (same
    response vector).  The original predictor values and shared response
    are required to compute the paired DeLong covariance.

    Parameters
    ----------
    roc1, roc2 : ROCResult
        Two ROC curves computed on the same subjects.
    predictor1, predictor2 : array of float
        Original predictor values for each marker.
    response : array of int
        Shared binary outcome.
    method : str
        ``'delong'`` (only supported method).

    Returns
    -------
    ROCTestResult

    Validates against: R ``pROC::roc.test()``
    """
    if method != "delong":
        raise ValueError(f"Only 'delong' method is supported, got {method!r}")

    if predictor1 is None or predictor2 is None or response is None:
        raise ValueError(
            "predictor1, predictor2, and response are required for DeLong test"
        )

    response = np.asarray(response, dtype=np.intp)
    predictor1 = np.asarray(predictor1, dtype=np.float64)
    predictor2 = np.asarray(predictor2, dtype=np.float64)

    n = len(response)
    if predictor1.shape[0] != n or predictor2.shape[0] != n:
        raise ValueError("predictor1, predictor2, and response must have equal length")

    n1 = int(response.sum())
    n0 = n - n1

    # Compute placement values for each marker
    auc1, V10_1, V01_1 = _compute_auc_and_placements(
        response, predictor1, roc1.direction,
    )
    auc2, V10_2, V01_2 = _compute_auc_and_placements(
        response, predictor2, roc2.direction,
    )

    # Variance of each AUC
    var1 = _delong_variance(auc1, V10_1, V01_1, n1, n0)
    var2 = _delong_variance(auc2, V10_2, V01_2, n1, n0)

    # Covariance between the two AUCs
    S10_12 = np.cov(V10_1, V10_2, ddof=1)[0, 1] if n1 > 1 else 0.0
    S01_12 = np.cov(V01_1, V01_2, ddof=1)[0, 1] if n0 > 1 else 0.0
    cov_12 = S10_12 / n1 + S01_12 / n0

    # Variance of the difference
    var_diff = var1 + var2 - 2 * cov_12
    var_diff = max(var_diff, 1e-20)  # avoid division by zero

    # Z statistic
    z_stat = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * stats.norm.sf(abs(z_stat))

    return ROCTestResult(
        statistic=float(z_stat),
        p_value=float(p_value),
        auc1=auc1,
        auc2=auc2,
        auc_diff=auc1 - auc2,
        method="delong",
    )
