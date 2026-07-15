"""EC50/IC50 estimation and relative potency analysis.

The EC50 (ED50) is the dose producing a response half-way between the fitted
lower and upper asymptotes. It is obtained by solving the fitted curve for that
half-maximal response — which equals the model's ``e`` location parameter only
for the symmetric LL.4 model, and differs from ``e`` for the asymmetric models
(LL.5, W1.4, W2.4, BC.5), matching R ``drc::ED(type="relative")``. The
confidence interval is a raw-scale symmetric Wald interval (estimate ± t·SE)
where SE comes from the delta method applied to the solved ED50, matching R
``drc::ED(interval="delta")``.

Relative potency (ratio of EC50s from two independent fits) uses Fieller's
theorem.

Validates against: R drc::ED(), drc::EDcomp()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pystatistics.core.exceptions import ValidationError
from pystatistics.core.result import Result, SolutionReprMixin
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.stats import t as t_dist

from pystatsbio.doseresponse._common import CurveParams, DoseResponseSolution


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

def _solve_ed50(curve: CurveParams, e_hint: float) -> float:
    """Dose at the half-maximal response (bottom+top)/2 for a fitted curve.

    Solves ``predict(x) = (bottom+top)/2`` for x on the descending/ascending
    branch. Equals the model's ``e`` parameter for the symmetric LL.4 and
    differs for the asymmetric models — matching ``drc::ED(type="relative")``.
    For non-monotone curves (BC.5 hormesis) the highest-dose crossing (the
    descending branch) is taken. Returns NaN if no crossing is found.
    """
    target = 0.5 * (curve.bottom + curve.top)

    def f(log_x: float) -> float:
        return float(curve.predict(np.array([np.exp(log_x)]))[0]) - target

    if not (np.isfinite(e_hint) and e_hint > 0):
        return float("nan")
    grid = np.log(e_hint) + np.linspace(-12.0, 12.0, 400)
    vals = np.array([f(lx) for lx in grid])
    finite = np.isfinite(vals)
    changes = np.where(np.diff(np.sign(vals)) != 0)[0]
    # keep only sign changes with finite endpoints; take the highest-dose one
    changes = [i for i in changes if finite[i] and finite[i + 1]]
    if not changes:
        return float("nan")
    i = changes[-1]
    try:
        return float(np.exp(brentq(f, grid[i], grid[i + 1])))
    except (ValueError, RuntimeError):
        return float("nan")


def _ed50_delta_se(fit_result: DoseResponseSolution, ed50: float) -> float:
    """Delta-method SE of the solved ED50, matching drc::ED(interval="delta").

    Uses the fit's parameter covariance (observed information, the same matrix
    drc uses) and the gradient of ED50 with respect to the parameters (finite
    differences, re-solving ED50 at each perturbation).
    """
    model = fit_result.model
    theta = fit_result.params.to_array()
    n_p = len(theta)
    n_obs = fit_result.n_obs
    if n_obs <= n_p or not (np.isfinite(ed50) and ed50 > 0):
        return float("nan")

    cov = fit_result.cov
    if cov is None or not np.all(np.isfinite(cov)):
        return float("nan")

    grad = np.zeros(n_p)
    for i in range(n_p):
        eps = max(1e-6, abs(theta[i]) * 1e-6)
        tp = theta.copy()
        tp[i] += eps
        ed_p = _solve_ed50(CurveParams.from_array(tp, model), fit_result.params.ec50)
        if not np.isfinite(ed_p):
            return float("nan")
        grad[i] = (ed_p - ed50) / eps

    var = float(grad @ cov @ grad)
    return float(np.sqrt(var)) if var >= 0 else float("nan")


def ec50(
    fit_result: DoseResponseSolution,
    *,
    conf_level: float = 0.95,
    method: str = "delta",
) -> EC50Solution:
    """Extract the EC50 (ED50) with confidence interval from a fitted model.

    The EC50 is the dose producing a response half-way between the fitted lower
    and upper asymptotes, obtained by solving the fitted curve — matching R
    ``drc::ED(type="relative")``. This equals the model's ``e`` location
    parameter for the symmetric LL.4 and differs for the asymmetric models
    (LL.5, W1.4, W2.4, BC.5). The confidence interval is a raw-scale symmetric
    Wald interval (estimate ± t·SE) with the SE from the delta method applied to
    the solved ED50, matching ``drc::ED(interval="delta")``.

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

    ec50_val = _solve_ed50(fit_result.params, fit_result.params.ec50)
    se_ec50 = _ed50_delta_se(fit_result, ec50_val)

    # t-distribution with residual df on the raw scale (drc::ED delta convention)
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
    """Relative potency: ratio of EC50s (ED50s) between two curves, Fieller's CI.

    Computes ``rho = ED50_2 / ED50_1`` with a confidence interval based on
    Fieller's theorem for the ratio of two independent estimates.

    Both ED50s are obtained by solving each fitted curve for its half-maximal
    response (the same quantity :func:`ec50` returns), *not* by ratioing the raw
    ``e`` location parameters. For the symmetric LL.4 — and for any two *parallel*
    curves, where the ``e``-to-ED50 factor is identical and cancels — the two are
    the same. They differ for non-parallel asymmetric fits (LL.5/W1.4/W2.4/BC.5),
    where ratioing ``e`` would not be the potency ratio at all.

    Note this function does **not** test parallelism; relative potency is only
    meaningful for curves of the same shape, and assessing that is the caller's
    responsibility.

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

    # Solved ED50s (half-maximal doses), consistent with ec50(), plus their
    # delta-method SEs — NOT the raw `e` parameters.
    e1 = _solve_ed50(fit1.params, fit1.params.ec50)
    e2 = _solve_ed50(fit2.params, fit2.params.ec50)
    se1 = _ed50_delta_se(fit1, e1)
    se2 = _ed50_delta_se(fit2, e2)

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
