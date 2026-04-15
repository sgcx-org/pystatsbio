"""Single dose-response curve fitting via nonlinear least squares.

Uses ``scipy.optimize.least_squares`` with Trust Region Reflective (TRF)
algorithm for bounded optimisation.  EC50 is constrained to be positive.

Includes data-driven self-starting estimates so the user never has to
guess initial parameter values.

Validates against: R drc::drm()
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from pystatsbio.doseresponse._common import CurveParams, DoseResponseResult
from pystatsbio.doseresponse._models import _MODEL_MAP, VALID_MODELS

# ---------------------------------------------------------------------------
# Self-starting parameter estimation
# ---------------------------------------------------------------------------

def _interpolate_ec50(
    dose_sorted: NDArray,
    resp_sorted: NDArray,
    midpoint: float,
) -> float:
    """Find dose at which response crosses *midpoint* via linear interpolation
    on the log-dose scale.
    """
    for i in range(len(resp_sorted) - 1):
        r1, r2 = resp_sorted[i], resp_sorted[i + 1]
        if (r1 - midpoint) * (r2 - midpoint) <= 0:
            d1 = np.log(dose_sorted[i])
            d2 = np.log(dose_sorted[i + 1])
            if abs(r2 - r1) < 1e-12:
                return float(np.exp((d1 + d2) / 2.0))
            frac = (midpoint - r1) / (r2 - r1)
            return float(np.exp(d1 + frac * (d2 - d1)))

    # No crossing — geometric mean of dose range
    return float(np.exp(np.mean(np.log(dose_sorted))))


def _estimate_hill(
    dose_sorted: NDArray,
    resp_sorted: NDArray,
    bottom: float,
    top: float,
) -> float:
    """Estimate Hill coefficient via logit-linear regression."""
    span = top - bottom
    if abs(span) < 1e-12:
        return 1.0

    y_norm = np.clip((resp_sorted - bottom) / span, 0.01, 0.99)
    logit_y = np.log(y_norm / (1.0 - y_norm))
    log_dose = np.log(dose_sorted)

    if len(log_dose) < 2:
        return 1.0

    # Simple linear regression:  logit(y) ~ slope * log(dose) + intercept
    coeffs = np.polyfit(log_dose, logit_y, 1)
    hill = float(np.clip(coeffs[0], -20.0, 20.0))
    if abs(hill) < 0.05:
        hill = 1.0 if hill >= 0 else -1.0
    return hill


def _initial_params(
    dose: NDArray,
    response: NDArray,
    model: str,
) -> dict[str, float]:
    """Data-driven starting values for nonlinear fitting.

    Algorithm
    ---------
    1.  Use only dose > 0 for log-scale estimation.
    2.  bottom/top from lowest/highest dose-group means.
    3.  Direction from correlation of response with dose rank.
    4.  EC50 via linear interpolation at midpoint on log-dose scale.
    5.  Hill via logit-linear regression.
    """
    mask = dose > 0
    dose_pos = dose[mask]
    resp_pos = response[mask]

    if len(dose_pos) < 2:
        # Fallback — not enough positive-dose data
        return {
            "bottom": float(np.min(response)),
            "top": float(np.max(response)),
            "ec50": 1.0,
            "hill": 1.0,
            **({"asymmetry": 1.0} if model == "LL.5" else {}),
            **({"hormesis": 0.0} if model == "BC.5" else {}),
        }

    order = np.argsort(dose_pos)
    d_sorted = dose_pos[order]
    r_sorted = resp_pos[order]

    n_edge = max(1, len(d_sorted) // 4)
    low_resp = float(np.mean(r_sorted[:n_edge]))
    high_resp = float(np.mean(r_sorted[-n_edge:]))

    # Include dose=0 data in direction detection
    if np.any(dose == 0):
        zero_resp = float(np.mean(response[dose == 0]))
        low_resp = min(low_resp, zero_resp)
        increasing = high_resp > zero_resp
    else:
        increasing = high_resp > low_resp

    if increasing:
        bottom_est = low_resp
        top_est = high_resp
    else:
        bottom_est = high_resp
        top_est = low_resp

    # EC50 — dose at midpoint response
    mid = (bottom_est + top_est) / 2.0
    ec50_est = _interpolate_ec50(d_sorted, r_sorted, mid)

    # Hill slope
    hill_est = _estimate_hill(d_sorted, r_sorted, bottom_est, top_est)

    start: dict[str, float] = {
        "bottom": bottom_est,
        "top": top_est,
        "ec50": max(ec50_est, 1e-20),  # ensure positive
        "hill": hill_est,
    }

    if model == "LL.5":
        start["asymmetry"] = 1.0
    elif model == "BC.5":
        start["hormesis"] = 0.0

    return start


# ---------------------------------------------------------------------------
# Standard error computation
# ---------------------------------------------------------------------------

def _compute_se(
    jac: NDArray,
    rss: float,
    n_obs: int,
    n_params: int,
) -> NDArray[np.floating]:
    """Standard errors from Jacobian: ``se = sqrt(diag((J'J)^{-1} * s²))``.

    Parameters
    ----------
    jac : (n_obs, n_params)
        Jacobian of the residual vector at the solution.
    rss : float
        Residual sum of squares.
    n_obs, n_params : int
        Number of observations and parameters.
    """
    if n_obs <= n_params:
        return np.full(n_params, np.nan)

    s2 = rss / (n_obs - n_params)
    JtJ = jac.T @ jac

    try:
        cov = np.linalg.inv(JtJ) * s2
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(n_params, np.nan)

    return se


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_drm(
    dose: NDArray[np.floating],
    response: NDArray[np.floating],
    *,
    model: str = "LL.4",
    weights: NDArray[np.floating] | None = None,
    start: dict[str, float] | None = None,
    lower: dict[str, float] | None = None,
    upper: dict[str, float] | None = None,
) -> DoseResponseResult:
    """Fit a dose-response model to a single curve.

    Uses Trust Region Reflective nonlinear least squares
    (``scipy.optimize.least_squares``).

    Parameters
    ----------
    dose : array
        Dose (concentration) values.
    response : array
        Response values.
    model : str
        Model name: ``'LL.4'``, ``'LL.5'``, ``'W1.4'``, ``'W2.4'``, ``'BC.5'``.
    weights : array or None
        Optional observation weights.
    start : dict or None
        Starting values for parameters.  If ``None``, uses self-starting
        estimates derived from the data.
    lower, upper : dict or None
        Box constraints on parameters.

    Returns
    -------
    DoseResponseResult

    Examples
    --------
    >>> import numpy as np
    >>> dose = np.array([0, 0.01, 0.1, 1, 10, 100])
    >>> response = np.array([10, 12, 30, 55, 85, 92])
    >>> result = fit_drm(dose, response, model='LL.4')
    >>> round(result.params.ec50, 1)
    1.0

    Validates against: R drc::drm()
    """
    # --- Validate ---
    dose = np.asarray(dose, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)

    if dose.ndim != 1 or response.ndim != 1:
        raise ValueError("dose and response must be 1-D arrays")
    if dose.shape != response.shape:
        raise ValueError(
            f"dose and response must have same shape, got {dose.shape} and {response.shape}"
        )
    if model not in VALID_MODELS:
        raise ValueError(f"model must be one of {VALID_MODELS}, got {model!r}")

    model_func, param_names = _MODEL_MAP[model]
    n_params = len(param_names)
    n_obs = len(dose)

    if n_obs < n_params + 1:
        raise ValueError(
            f"Need at least {n_params + 1} observations for model {model}, got {n_obs}"
        )

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != dose.shape:
            raise ValueError("weights must have same shape as dose")

    # --- Starting values ---
    if start is None:
        start = _initial_params(dose, response, model)
    x0 = np.array([start[name] for name in param_names], dtype=np.float64)

    # --- Bounds ---
    lb = np.full(n_params, -np.inf)
    ub = np.full(n_params, np.inf)
    # EC50 must be positive
    ec50_idx = param_names.index("ec50")
    lb[ec50_idx] = 1e-20
    if lower is not None:
        for name, val in lower.items():
            lb[param_names.index(name)] = val
    if upper is not None:
        for name, val in upper.items():
            ub[param_names.index(name)] = val

    # Ensure starting values are within bounds
    x0 = np.clip(x0, lb + 1e-15, ub - 1e-15)

    # --- Residual function ---
    def residuals(p: NDArray) -> NDArray:
        kwargs = dict(zip(param_names, p, strict=True))
        pred = model_func(dose, **kwargs)
        r = response - pred
        if weights is not None:
            r = r * np.sqrt(weights)
        return r

    # --- Fit ---
    result = least_squares(
        residuals,
        x0,
        method="trf",
        bounds=(lb, ub),
        jac="2-point",
        max_nfev=2000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    # --- Extract ---
    popt = result.x
    res_vec = result.fun
    rss = float(np.sum(res_vec**2))
    converged = result.success
    n_iter = result.nfev

    jac = result.jac
    se = _compute_se(jac, rss, n_obs, n_params)

    # AIC / BIC
    aic = float(n_obs * np.log(rss / n_obs) + 2 * n_params)
    bic = float(n_obs * np.log(rss / n_obs) + n_params * np.log(n_obs))

    curve_params = CurveParams.from_array(popt, model)

    return DoseResponseResult(
        params=curve_params,
        se=se,
        residuals=res_vec,
        rss=rss,
        aic=aic,
        bic=bic,
        converged=converged,
        n_iter=n_iter,
        model=model,
        dose=dose,
        response=response,
        n_obs=n_obs,
        jac=jac,
    )
