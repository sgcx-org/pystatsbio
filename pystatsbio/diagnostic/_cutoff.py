"""Optimal cutoff selection for diagnostic tests.

Three methods: Youden index (maximize sensitivity + specificity − 1),
closest-to-top-left (minimize Euclidean distance to the (0,1) corner),
and cost-based (minimize weighted misclassification cost given
prevalence).

Validates against: R ``OptimalCutpoints::optimal.cutpoints()``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pystatsbio.diagnostic._common import ROCResult


@dataclass(frozen=True)
class CutoffResult:
    """Result of optimal cutoff selection."""

    cutoff: float
    sensitivity: float
    specificity: float
    method: str  # 'youden', 'closest_topleft', 'cost'
    criterion_value: float  # value of the optimization criterion


def optimal_cutoff(
    roc_result: ROCResult,
    *,
    method: str = "youden",
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    prevalence: float | None = None,
) -> CutoffResult:
    """Find optimal classification cutoff from an ROC curve.

    Parameters
    ----------
    roc_result : ROCResult
        A computed ROC curve.
    method : str
        ``'youden'`` — maximize sensitivity + specificity − 1.
        ``'closest_topleft'`` — minimize distance to ``(FPR=0, TPR=1)``.
        ``'cost'`` — minimize weighted misclassification cost.
    cost_fp, cost_fn : float
        Costs of false positives and false negatives (for ``method='cost'``).
    prevalence : float or None
        Disease prevalence (for ``method='cost'``).  Uses sample
        prevalence ``n_positive / (n_positive + n_negative)`` if
        ``None``.

    Returns
    -------
    CutoffResult

    Validates against: R ``OptimalCutpoints::optimal.cutpoints()``
    """
    valid_methods = ("youden", "closest_topleft", "cost")
    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {valid_methods}, got {method!r}"
        )

    tpr = roc_result.tpr
    fpr = roc_result.fpr
    thresholds = roc_result.thresholds

    # Exclude the boundary points (inf / -inf) from candidate set
    # because they correspond to "classify nobody" or "classify everybody"
    finite_mask = np.isfinite(thresholds)
    if finite_mask.sum() == 0:
        raise ValueError("ROC result has no finite thresholds")

    tpr_f = tpr[finite_mask]
    fpr_f = fpr[finite_mask]
    thresh_f = thresholds[finite_mask]
    spec_f = 1.0 - fpr_f

    if method == "youden":
        # J = sens + spec - 1 = TPR - FPR
        criterion = tpr_f - fpr_f
        best_idx = int(np.argmax(criterion))
        crit_val = float(criterion[best_idx])

    elif method == "closest_topleft":
        # Euclidean distance to (FPR=0, TPR=1)
        dist = np.sqrt(fpr_f ** 2 + (1.0 - tpr_f) ** 2)
        best_idx = int(np.argmin(dist))
        crit_val = float(dist[best_idx])

    else:  # cost
        if prevalence is None:
            prev = roc_result.n_positive / (
                roc_result.n_positive + roc_result.n_negative
            )
        else:
            prev = prevalence

        # Expected cost = cost_fp * FPR * (1-prev) + cost_fn * (1-TPR) * prev
        cost = cost_fp * fpr_f * (1 - prev) + cost_fn * (1 - tpr_f) * prev
        best_idx = int(np.argmin(cost))
        crit_val = float(cost[best_idx])

    return CutoffResult(
        cutoff=float(thresh_f[best_idx]),
        sensitivity=float(tpr_f[best_idx]),
        specificity=float(spec_f[best_idx]),
        method=method,
        criterion_value=crit_val,
    )
