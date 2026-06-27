"""EC50/IC50 estimation and relative potency analysis.

EC50 confidence intervals via the delta method on the log scale.
Relative potency (ratio of EC50s from two independent fits) uses
Fieller's theorem.

Validates against: R drc::ED(), drc::EDcomp()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pystatistics.core.exceptions import ValidationError
from pystatistics.core.result import Result, SolutionReprMixin
from scipy.stats import norm
from scipy.stats import t as t_dist

from pystatsbio.doseresponse._common import DoseResponseSolution
from pystatsbio.doseresponse._models import _MODEL_MAP


@dataclass(frozen=True)
class EC50Params:
    """Computed payload of EC50 (or IC50) estimation with confidence interval."""

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    conf_level: float
    method: str  # 'delta'


class EC50Solution(SolutionReprMixin):
    """Public result of EC50 estimation — wraps ``Result[EC50Params]``.

    Exposes every output as a read-only property plus the uniform
    ``.backend_name`` / ``.timing`` / ``.warnings`` / ``.info`` metadata and a
    Jupyter ``_repr_html_`` (via :class:`SolutionReprMixin`).
    """

    def __init__(self, result: Result[EC50Params]) -> None:
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

    # --- EC50 outputs (from the payload) ---
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
    def conf_level(self) -> float:
        return self._result.params.conf_level

    @property
    def method(self) -> str:
        return self._result.params.method

    def __repr__(self) -> str:
        p = self._result.params
        return (
            f"EC50Solution(estimate={p.estimate:.4g}, "
            f"ci=[{p.ci_lower:.4g}, {p.ci_upper:.4g}], "
            f"conf_level={p.conf_level})"
        )


@dataclass(frozen=True)
class RelativePotencyParams:
    """Computed payload of relative potency (ratio of EC50s) with Fieller's CI."""

    ratio: float
    ci_lower: float
    ci_upper: float
    conf_level: float
    method: str  # 'fieller'


class RelativePotencySolution(SolutionReprMixin):
    """Public result of relative potency — wraps ``Result[RelativePotencyParams]``.

    Exposes every output as a read-only property plus the uniform
    ``.backend_name`` / ``.timing`` / ``.warnings`` / ``.info`` metadata and a
    Jupyter ``_repr_html_`` (via :class:`SolutionReprMixin`).
    """

    def __init__(self, result: Result[RelativePotencyParams]) -> None:
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

    # --- Relative potency outputs (from the payload) ---
    @property
    def ratio(self) -> float:
        return self._result.params.ratio

    @property
    def ci_lower(self) -> float:
        return self._result.params.ci_lower

    @property
    def ci_upper(self) -> float:
        return self._result.params.ci_upper

    @property
    def conf_level(self) -> float:
        return self._result.params.conf_level

    @property
    def method(self) -> str:
        return self._result.params.method

    def __repr__(self) -> str:
        p = self._result.params
        return (
            f"RelativePotencySolution(ratio={p.ratio:.4g}, "
            f"ci=[{p.ci_lower:.4g}, {p.ci_upper:.4g}], "
            f"conf_level={p.conf_level})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ec50(
    fit_result: DoseResponseSolution,
    *,
    conf_level: float = 0.95,
    method: str = "delta",
) -> EC50Solution:
    """Extract EC50 with confidence interval from a fitted model.

    For models where EC50 is a direct parameter (LL.4, LL.5, W1.4, W2.4,
    BC.5), the standard error comes from the parameter covariance matrix.
    The confidence interval is constructed on the log scale (since EC50 is
    positive) and back-transformed.

    Parameters
    ----------
    fit_result : DoseResponseSolution
        A fitted dose-response model.
    conf_level : float
        Confidence level (default 0.95).
    method : str
        ``'delta'`` (delta method).

    Returns
    -------
    EC50Solution

    Validates against: R drc::ED()
    """
    if not (0.0 < conf_level < 1.0):
        raise ValidationError(f"conf_level must be in (0, 1), got {conf_level}")
    if method != "delta":
        raise ValidationError(f"method must be 'delta', got {method!r}")

    _, param_names = _MODEL_MAP[fit_result.model]
    ec50_idx = param_names.index("ec50")
    ec50_val = fit_result.params.ec50
    se_ec50 = float(fit_result.se[ec50_idx])

    # Use t-distribution with residual df on the raw scale,
    # matching R drc::ED(interval="delta")
    n_params = len(fit_result.se)
    df_resid = fit_result.n_obs - n_params
    if df_resid > 0:
        t_crit = t_dist.ppf(1.0 - (1.0 - conf_level) / 2.0, df_resid)
    else:
        t_crit = norm.ppf(1.0 - (1.0 - conf_level) / 2.0)

    # Raw-scale CI: estimate ± t * SE
    if ec50_val > 0 and np.isfinite(ec50_val) and se_ec50 > 0 and not np.isnan(se_ec50):
        ci_lower = float(ec50_val - t_crit * se_ec50)
        ci_upper = float(ec50_val + t_crit * se_ec50)
    else:
        ci_lower = float("nan")
        ci_upper = float("nan")

    params = EC50Params(
        estimate=ec50_val,
        se=se_ec50,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        conf_level=conf_level,
        method=method,
    )
    result = Result(
        params=params,
        info={"conf_level": conf_level, "method": method},
        timing=None,
        backend_name="cpu",
    )
    return EC50Solution(result)


def relative_potency(
    fit1: DoseResponseSolution,
    fit2: DoseResponseSolution,
    *,
    conf_level: float = 0.95,
) -> RelativePotencySolution:
    """Relative potency: ratio of EC50s between two curves with Fieller's CI.

    Computes ``rho = EC50_2 / EC50_1`` with a confidence interval based on
    Fieller's theorem for the ratio of two independent estimates.

    Parameters
    ----------
    fit1 : DoseResponseSolution
        First fitted model (reference).
    fit2 : DoseResponseSolution
        Second fitted model (test).
    conf_level : float
        Confidence level (default 0.95).

    Returns
    -------
    RelativePotencySolution

    Validates against: R drc::compParm(), drc::EDcomp()
    """
    if not (0.0 < conf_level < 1.0):
        raise ValidationError(f"conf_level must be in (0, 1), got {conf_level}")

    _, names1 = _MODEL_MAP[fit1.model]
    _, names2 = _MODEL_MAP[fit2.model]
    e1 = fit1.params.ec50
    e2 = fit2.params.ec50
    se1 = float(fit1.se[names1.index("ec50")])
    se2 = float(fit2.se[names2.index("ec50")])

    ratio = e2 / e1 if e1 != 0 else float("nan")

    z = norm.ppf(1.0 - (1.0 - conf_level) / 2.0)
    z2 = z**2

    # Fieller's theorem for a/b where a = e2, b = e1, independent fits (cov=0)
    a, b = e2, e1
    var_a, var_b = se2**2, se1**2

    denom = b**2 - z2 * var_b
    if denom <= 0:
        # Fieller's CI undefined (denominator crosses zero → infinite CI)
        ci_lower, ci_upper = float("-inf"), float("inf")
    else:
        num_center = a * b
        D = num_center**2 - (a**2 - z2 * var_a) * (b**2 - z2 * var_b)
        if D < 0:
            ci_lower, ci_upper = float("-inf"), float("inf")
        else:
            ci_lower = float((num_center - np.sqrt(D)) / denom)
            ci_upper = float((num_center + np.sqrt(D)) / denom)

    params = RelativePotencyParams(
        ratio=ratio,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        conf_level=conf_level,
        method="fieller",
    )
    result = Result(
        params=params,
        info={"conf_level": conf_level, "method": "fieller"},
        timing=None,
        backend_name="cpu",
    )
    return RelativePotencySolution(result)
