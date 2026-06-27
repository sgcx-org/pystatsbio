"""Benchmark dose (BMD) analysis for toxicology.

Computes the dose at which a specified benchmark response (BMR) is
reached, with lower/upper confidence limits (BMDL/BMDU) via the
delta method.

Validates against: EPA BMDS software, R BMDL packages
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pystatistics.core.exceptions import NumericalError, ValidationError
from pystatistics.core.result import Result, SolutionReprMixin
from scipy.optimize import brentq
from scipy.stats import norm

from pystatsbio.doseresponse._common import CurveParams, DoseResponseSolution


@dataclass(frozen=True)
class BMDParams:
    """Computed payload of benchmark dose analysis."""

    bmd: float  # benchmark dose (point estimate)
    bmdl: float  # lower confidence limit
    bmdu: float  # upper confidence limit
    bmr: float  # benchmark response level
    conf_level: float
    method: str  # 'delta'


class BMDSolution(SolutionReprMixin):
    """Public result of benchmark dose analysis — wraps ``Result[BMDParams]``.

    Exposes every output as a read-only property plus the uniform
    ``.backend_name`` / ``.timing`` / ``.warnings`` / ``.info`` metadata and a
    Jupyter ``_repr_html_`` (via :class:`SolutionReprMixin`).
    """

    def __init__(self, result: Result[BMDParams]) -> None:
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

    # --- BMD outputs (from the payload) ---
    @property
    def bmd(self) -> float:
        return self._result.params.bmd

    @property
    def bmdl(self) -> float:
        return self._result.params.bmdl

    @property
    def bmdu(self) -> float:
        return self._result.params.bmdu

    @property
    def bmr(self) -> float:
        return self._result.params.bmr

    @property
    def conf_level(self) -> float:
        return self._result.params.conf_level

    @property
    def method(self) -> str:
        return self._result.params.method

    def __repr__(self) -> str:
        p = self._result.params
        return (
            f"BMDSolution(bmd={p.bmd:.4g}, "
            f"bmdl={p.bmdl:.4g}, bmdu={p.bmdu:.4g}, bmr={p.bmr})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bmd_ll4_analytical(params: CurveParams, target: float) -> float:
    """Analytical BMD for LL.4: dose = ec50 * ((top-target)/(target-bottom))^(1/hill).

    Raises
    ------
    ValueError
        If the geometry is degenerate (zero denominator, zero numerator,
        or non-positive ratio — all of which indicate the target is
        outside the achievable curve range).
    """
    c, d, e, h = params.bottom, params.top, params.ec50, params.hill
    numer = d - target
    denom = target - c
    if denom == 0 or numer == 0:
        raise ValidationError(
            f"BMD is undefined: target={target:.4g} is at the curve boundary "
            f"(bottom={c:.4g}, top={d:.4g}). Adjust bmr."
        )
    ratio = numer / denom
    if ratio <= 0:
        raise ValidationError(
            f"BMD is undefined: target={target:.4g} is outside the curve range "
            f"[{min(c, d):.4g}, {max(c, d):.4g}]."
        )
    return e * ratio ** (1.0 / h)


def _bmd_numerical(params: CurveParams, target: float) -> float:
    """Numerical BMD via root-finding in log-dose space.

    Raises
    ------
    RuntimeError
        If root-finding fails (target not crossed in log-dose range [-50, 50]).
    """
    def f(log_dose: float) -> float:
        dose_arr = np.array([np.exp(log_dose)])
        return float(params.predict(dose_arr)[0]) - target

    try:
        log_bmd = brentq(f, -50, 50, xtol=1e-12)
    except ValueError as exc:
        raise NumericalError(
            f"BMD root-finding failed: target response {target:.4g} was not "
            f"crossed in dose range [exp(-50), exp(50)]. "
            f"Check that bmr is achievable for this curve."
        ) from exc
    return float(np.exp(log_bmd))


def _bmd_from_params_array(
    params_arr: NDArray,
    model: str,
    target: float,
) -> float:
    """Compute BMD for a given parameter array (used for numerical gradient)."""
    cp = CurveParams.from_array(params_arr, model)
    if model == "LL.4":
        return _bmd_ll4_analytical(cp, target)
    return _bmd_numerical(cp, target)


def _bmd_delta_ci(
    fit_result: DoseResponseSolution,
    bmd_val: float,
    target: float,
    conf_level: float,
) -> tuple[float, float]:
    """BMDL/BMDU via delta method with numerical gradient.

    Raises
    ------
    RuntimeError
        If the parameter covariance matrix is singular (Jacobian is rank-deficient)
        or if numerical issues produce a negative variance estimate.
    """
    params_arr = fit_result.params.to_array()
    n_params = len(params_arr)
    model = fit_result.model

    # Numerical gradient of BMD w.r.t. parameters (central differences)
    eps = 1e-6
    grad = np.zeros(n_params)
    for i in range(n_params):
        p_plus = params_arr.copy()
        p_plus[i] += eps
        p_minus = params_arr.copy()
        p_minus[i] -= eps
        bmd_plus = _bmd_from_params_array(p_plus, model, target)
        bmd_minus = _bmd_from_params_array(p_minus, model, target)
        grad[i] = (bmd_plus - bmd_minus) / (2 * eps)

    # Parameter covariance matrix
    jac = fit_result.jac
    n_obs = fit_result.n_obs
    s2 = fit_result.rss / (n_obs - n_params)

    try:
        cov = np.linalg.inv(jac.T @ jac) * s2
    except np.linalg.LinAlgError as exc:
        raise NumericalError(
            "Cannot compute BMD confidence interval: the Jacobian matrix is "
            "rank-deficient. The model may be overparameterised or the fit "
            "did not converge to a stable solution."
        ) from exc

    var_bmd = float(grad @ cov @ grad)
    if var_bmd < 0:
        raise NumericalError(
            f"Cannot compute BMD confidence interval: delta-method variance "
            f"is negative ({var_bmd:.4g}), indicating numerical instability "
            f"in the covariance matrix. Try a simpler model or more data."
        )
    se_bmd = math.sqrt(var_bmd)

    z = norm.ppf(1.0 - (1.0 - conf_level) / 2.0)

    # CI on log scale (BMD is positive)
    se_log = se_bmd / bmd_val
    log_bmd = math.log(bmd_val)
    bmdl = math.exp(log_bmd - z * se_log)
    bmdu = math.exp(log_bmd + z * se_log)

    return bmdl, bmdu


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bmd(
    fit_result: DoseResponseSolution,
    *,
    bmr: float = 0.10,
    bmr_type: str = "extra",
    conf_level: float = 0.95,
    method: str = "delta",
) -> BMDSolution:
    """Compute benchmark dose (BMD) with BMDL/BMDU.

    Parameters
    ----------
    fit_result : DoseResponseSolution
        A fitted dose-response model.
    bmr : float
        Benchmark response level (default 10 % = 0.10).
    bmr_type : str
        ``'extra'`` (extra risk) or ``'additional'`` (additional risk).
    conf_level : float
        Confidence level (default 0.95).
    method : str
        ``'delta'`` (delta method).

    Returns
    -------
    BMDSolution

    Raises
    ------
    ValueError
        If inputs are invalid or the BMR target is outside the curve range.
    RuntimeError
        If numerical BMD computation or CI estimation fails.

    Notes
    -----
    For **extra risk**: the target response is
    ``top - bmr * (top - bottom)`` (i.e. a ``bmr`` fraction of the
    full range from the upper asymptote).

    For **additional risk**: the target is
    ``top - bmr * |top - bottom|``.

    Validates against: EPA BMDS software, R BMDL packages
    """
    if not math.isfinite(bmr):
        raise ValidationError(f"bmr must be finite, got {bmr}")
    if not (0.0 < bmr < 1.0):
        raise ValidationError(f"bmr must be in (0, 1), got {bmr}")
    if bmr_type not in ("extra", "additional"):
        raise ValidationError(f"bmr_type must be 'extra' or 'additional', got {bmr_type!r}")
    if not (0.0 < conf_level < 1.0):
        raise ValidationError(f"conf_level must be in (0, 1), got {conf_level}")
    if method != "delta":
        raise ValidationError(f"method must be 'delta', got {method!r}")

    p = fit_result.params
    c, d = p.bottom, p.top

    # Compute target response level
    target = d - bmr * (d - c) if bmr_type == "extra" else d - bmr * abs(d - c)

    # Pre-validate that target is strictly inside the curve range
    lo, hi = min(c, d), max(c, d)
    if not (lo < target < hi):
        raise ValidationError(
            f"BMR target {target:.4g} is outside the fitted curve range "
            f"[{lo:.4g}, {hi:.4g}] (bottom={c:.4g}, top={d:.4g}, bmr={bmr}). "
            f"Reduce bmr or check the fitted model."
        )

    # Point estimate
    if fit_result.model == "LL.4":
        bmd_val = _bmd_ll4_analytical(p, target)
    else:
        bmd_val = _bmd_numerical(p, target)

    # Confidence limits
    bmdl, bmdu = _bmd_delta_ci(fit_result, bmd_val, target, conf_level)

    params = BMDParams(
        bmd=bmd_val,
        bmdl=bmdl,
        bmdu=bmdu,
        bmr=bmr,
        conf_level=conf_level,
        method=method,
    )
    result = Result(
        params=params,
        info={"bmr": bmr, "bmr_type": bmr_type, "conf_level": conf_level, "method": method},
        timing=None,
        backend_name="cpu",
    )
    return BMDSolution(result)
