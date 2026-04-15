"""Sensitivity, specificity, predictive values, and likelihood ratios.

Computes a comprehensive set of diagnostic accuracy metrics at a fixed
cutoff: sensitivity/specificity with exact (Clopper-Pearson) or Wilson
CIs, PPV/NPV with optional prevalence adjustment, likelihood ratios
(LR+/LR−), and diagnostic odds ratio (DOR) with log-scale CI.

Validates against: R ``epiR::epi.tests()``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from pystatsbio.diagnostic._common import DiagnosticResult

# ---------------------------------------------------------------------------
# CI helpers for binomial proportions
# ---------------------------------------------------------------------------

def _clopper_pearson_ci(
    k: int, n: int, conf_level: float,
) -> tuple[float, float]:
    """Exact Clopper-Pearson CI for binomial proportion k/n."""
    alpha = 1 - conf_level
    if k == 0:
        lo = 0.0
        hi = 1.0 - (alpha / 2) ** (1.0 / n)
    elif k == n:
        lo = (alpha / 2) ** (1.0 / n)
        hi = 1.0
    else:
        lo = float(stats.beta.ppf(alpha / 2, k, n - k + 1))
        hi = float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def _wilson_ci(
    k: int, n: int, conf_level: float,
) -> tuple[float, float]:
    """Wilson score CI for binomial proportion k/n."""
    p_hat = k / n
    z = stats.norm.ppf((1 + conf_level) / 2)
    z2 = z ** 2
    denom = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    margin = z / denom * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n ** 2))
    return float(centre - margin), float(centre + margin)


def _binomial_ci(
    k: int, n: int, conf_level: float, method: str,
) -> tuple[float, float]:
    """Dispatch to the requested CI method."""
    if method == "clopper-pearson":
        return _clopper_pearson_ci(k, n, conf_level)
    elif method == "wilson":
        return _wilson_ci(k, n, conf_level)
    else:
        raise ValueError(
            f"ci_method must be 'clopper-pearson' or 'wilson', got {method!r}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def diagnostic_accuracy(
    response: NDArray[np.integer],
    predictor: NDArray[np.floating],
    *,
    cutoff: float,
    direction: str = "<",
    prevalence: float | None = None,
    conf_level: float = 0.95,
    ci_method: str = "clopper-pearson",
) -> DiagnosticResult:
    """Compute diagnostic accuracy metrics at a fixed cutoff.

    Parameters
    ----------
    response : array of int
        Binary outcome (0/1).
    predictor : array of float
        Continuous predictor.
    cutoff : float
        Classification threshold.
    direction : str
        ``'<'`` means predictor ≥ cutoff is classified positive
        (controls < cases, higher values = disease).
        ``'>'`` means predictor ≤ cutoff is classified positive
        (controls > cases, lower values = disease).
    prevalence : float or None
        Disease prevalence for PPV/NPV adjustment via Bayes' theorem.
        If ``None``, uses sample prevalence.
    conf_level : float
        Confidence level.
    ci_method : str
        ``'clopper-pearson'`` (exact) or ``'wilson'``.

    Returns
    -------
    DiagnosticResult

    Validates against: R ``epiR::epi.tests()``
    """
    response = np.asarray(response, dtype=np.intp)
    predictor = np.asarray(predictor, dtype=np.float64)

    if response.ndim != 1 or predictor.ndim != 1:
        raise ValueError("response and predictor must be 1-D")
    if len(response) != len(predictor):
        raise ValueError("response and predictor must have equal length")
    if direction not in ("<", ">"):
        raise ValueError(f"direction must be '<' or '>', got {direction!r}")
    if not 0 < conf_level < 1:
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")

    # Classify
    predicted_pos = predictor >= cutoff if direction == "<" else predictor <= cutoff

    actual_pos = response == 1

    TP = int(np.sum(predicted_pos & actual_pos))
    FP = int(np.sum(predicted_pos & ~actual_pos))
    FN = int(np.sum(~predicted_pos & actual_pos))
    TN = int(np.sum(~predicted_pos & ~actual_pos))

    n1 = TP + FN  # total positives
    n0 = TN + FP  # total negatives

    if n1 == 0 or n0 == 0:
        raise ValueError("Need at least one case and one control")

    # Sensitivity and specificity
    sens = TP / n1
    spec = TN / n0

    sens_ci = _binomial_ci(TP, n1, conf_level, ci_method)
    spec_ci = _binomial_ci(TN, n0, conf_level, ci_method)

    # Prevalence
    if prevalence is None:
        prev = n1 / (n1 + n0)
    else:
        if not 0 < prevalence < 1:
            raise ValueError(f"prevalence must be in (0, 1), got {prevalence}")
        prev = prevalence

    # PPV / NPV (with optional prevalence adjustment via Bayes)
    ppv_denom = sens * prev + (1 - spec) * (1 - prev)
    npv_denom = (1 - sens) * prev + spec * (1 - prev)
    ppv = (sens * prev / ppv_denom) if ppv_denom > 0 else float("nan")
    npv = (spec * (1 - prev) / npv_denom) if npv_denom > 0 else float("nan")

    # Likelihood ratios
    lr_pos = sens / (1 - spec) if (1 - spec) > 0 else float("inf")
    lr_neg = (1 - sens) / spec if spec > 0 else float("inf")

    # Diagnostic odds ratio (with 0.5 Haldane correction if any cell is 0)
    z = stats.norm.ppf((1 + conf_level) / 2)
    if TP == 0 or FP == 0 or FN == 0 or TN == 0:
        TPc = TP + 0.5
        FPc = FP + 0.5
        FNc = FN + 0.5
        TNc = TN + 0.5
    else:
        TPc, FPc, FNc, TNc = float(TP), float(FP), float(FN), float(TN)

    dor = (TPc * TNc) / (FPc * FNc)
    log_dor_se = np.sqrt(1 / TPc + 1 / FPc + 1 / FNc + 1 / TNc)
    dor_lo = np.exp(np.log(dor) - z * log_dor_se)
    dor_hi = np.exp(np.log(dor) + z * log_dor_se)

    return DiagnosticResult(
        cutoff=float(cutoff),
        sensitivity=float(sens),
        sensitivity_ci=sens_ci,
        specificity=float(spec),
        specificity_ci=spec_ci,
        ppv=float(ppv),
        npv=float(npv),
        lr_positive=float(lr_pos),
        lr_negative=float(lr_neg),
        dor=float(dor),
        dor_ci=(float(dor_lo), float(dor_hi)),
        prevalence=float(prev),
        conf_level=conf_level,
        method=ci_method,
    )
