"""Single dose-response curve fitting via nonlinear least squares.

Uses ``scipy.optimize.least_squares`` with Trust Region Reflective (TRF)
algorithm for bounded optimisation.  EC50 is constrained to be positive.

Includes data-driven self-starting estimates so the user never has to
guess initial parameter values.

Validates against: R drc::drm()
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from pystatistics.core.exceptions import ValidationError
from pystatistics.core.result import Result
from scipy.optimize import least_squares

from pystatsbio.doseresponse._common import (
    CurveParams,
    DoseResponseParams,
    DoseResponseSolution,
)
from pystatsbio.doseresponse._models import _JAC_LOG_MAP, _MODEL_MAP, VALID_MODELS

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

def _rss_hessian(
    rss_fn: Callable[[NDArray], float],
    popt: NDArray,
    rel_step: float = 1e-4,
) -> NDArray[np.floating]:
    """Numerical Hessian of the RSS objective at ``popt`` (central differences).

    Second-derivative central differences have error O(h^2) + O(eps_machine/h^2),
    so the step is taken relative to each parameter's own scale.
    """
    p = len(popt)
    h = rel_step * np.maximum(np.abs(popt), 1.0)
    H = np.empty((p, p), dtype=np.float64)
    for j in range(p):
        for k in range(j, p):
            ej = np.zeros(p); ej[j] = h[j]
            ek = np.zeros(p); ek[k] = h[k]
            val = (
                rss_fn(popt + ej + ek)
                - rss_fn(popt + ej - ek)
                - rss_fn(popt - ej + ek)
                + rss_fn(popt - ej - ek)
            ) / (4.0 * h[j] * h[k])
            H[j, k] = H[k, j] = val
    return H


def _compute_cov_se(
    rss_fn: Callable[[NDArray], float],
    popt: NDArray,
    jac: NDArray,
    rss: float,
    n_obs: int,
    n_params: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Parameter covariance and standard errors from the OBSERVED information.

    The covariance is ``cov = s2 * (H/2)^-1 = 2*s2*H^-1`` where ``H`` is the
    Hessian of the residual sum of squares and ``s2 = RSS/(n-p)``. Equivalently
    ``(J'J + sum_i r_i * d2r_i)^-1 * s2`` — i.e. Gauss-Newton PLUS the
    second-order curvature term.

    This is what R ``drc::drm()`` reports (its ``vcCont`` inverts the scaled RSS
    Hessian), and it is the reference this module validates against. The
    Gauss-Newton approximation ``s2*(J'J)^-1`` drops the curvature term; both are
    asymptotically valid, but on a curved model they differ materially along
    poorly-conditioned directions (up to ~11% on the Hill slope for the log-logistic
    family), so we match the reference rather than the cheaper approximation.

    Falls back to Gauss-Newton if the Hessian is singular or not usable.
    """
    if n_obs <= n_params:
        nan = np.full(n_params, np.nan)
        return np.full((n_params, n_params), np.nan), nan

    s2 = rss / (n_obs - n_params)

    try:
        H = _rss_hessian(rss_fn, popt)
        cov = 2.0 * s2 * np.linalg.inv(H)
        diag = np.diag(cov)
        if not np.all(np.isfinite(diag)) or np.any(diag < 0):
            raise np.linalg.LinAlgError("non-positive observed-information variance")
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        # Fall back to the Gauss-Newton covariance.
        try:
            cov = np.linalg.inv(jac.T @ jac) * s2
        except np.linalg.LinAlgError:
            nan = np.full(n_params, np.nan)
            return np.full((n_params, n_params), np.nan), nan

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return cov, se


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
) -> DoseResponseSolution:
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
    DoseResponseSolution

    Notes
    -----
    Parameter standard errors (``.se``) and the covariance (``.cov``) use the
    **observed-information** covariance ``s²·(½H)⁻¹`` — the inverse of the scaled
    Hessian of the residual sum of squares, ``s² = RSS/(n-p)`` — which is what R
    ``drc::drm()`` reports. This differs from the Gauss-Newton approximation
    ``s²·(JᵀJ)⁻¹`` used by ``scipy.optimize.curve_fit`` / R ``nls`` by the
    second-order curvature term; the two are asymptotically equivalent but can
    differ by ~10% on a poorly-conditioned coefficient (e.g. the Hill slope).

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
        raise ValidationError("dose and response must be 1-D arrays")
    if dose.shape != response.shape:
        raise ValidationError(
            f"dose and response must have same shape, got {dose.shape} and {response.shape}"
        )
    if model not in VALID_MODELS:
        raise ValidationError(f"model must be one of {VALID_MODELS}, got {model!r}")

    model_func, param_names = _MODEL_MAP[model]
    n_params = len(param_names)
    n_obs = len(dose)

    if n_obs < n_params + 1:
        raise ValidationError(
            f"Need at least {n_params + 1} observations for model {model}, got {n_obs}"
        )

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != dose.shape:
            raise ValidationError("weights must have same shape as dose")

    # --- Starting values ---
    start_was_none = start is None
    if start is None:
        start = _initial_params(dose, response, model)
    x0 = np.array([start[name] for name in param_names], dtype=np.float64)

    # --- Bounds ---
    ec50_idx = param_names.index("ec50")
    has_custom_bounds = lower is not None or upper is not None

    lb = np.full(n_params, -np.inf)
    ub = np.full(n_params, np.inf)
    lb[ec50_idx] = 1e-20
    if lower is not None:
        for name, val in lower.items():
            lb[param_names.index(name)] = val
    if upper is not None:
        for name, val in upper.items():
            ub[param_names.index(name)] = val

    # Ensure starting values are within bounds
    x0 = np.clip(x0, lb + 1e-15, ub - 1e-15)

    jac_log_func = _JAC_LOG_MAP.get(model)
    use_lm = (
        jac_log_func is not None
        and weights is None
        and not has_custom_bounds
    )

    def _fit_from(x0_in: NDArray):
        """Run one least-squares fit from a single starting point.

        Fast path: MINPACK LM (compiled lmder, ~8× over scipy's Python TRF
        loop) with an ec50 → log(ec50) reparameterisation that removes the
        positivity bound so method='lm' can be used. Slow path (custom bounds,
        weighted fits, or models without an analytical Jacobian): TRF.
        """
        if use_lm:
            x0_log = x0_in.copy()
            x0_log[ec50_idx] = np.log(max(x0_in[ec50_idx], 1e-20))

            def residuals_log(p: NDArray) -> NDArray:
                p_real = p.copy()
                p_real[ec50_idx] = np.exp(p[ec50_idx])
                kwargs = dict(zip(param_names, p_real, strict=True))
                return response - model_func(dose, **kwargs)

            with np.errstate(over="ignore", invalid="ignore"):
                res = least_squares(
                    residuals_log, x0_log, method="lm",
                    jac=lambda p: jac_log_func(dose, p),
                    max_nfev=200, xtol=1e-12, ftol=1e-12, gtol=1e-12,
                )
            ec50_fitted = np.exp(res.x[ec50_idx])
            res.x[ec50_idx] = ec50_fitted
            # log-space → natural-space Jacobian column: ∂r/∂ec50 = ∂r/∂log_ec50 / ec50
            if np.isfinite(ec50_fitted) and ec50_fitted > 0:
                res.jac[:, ec50_idx] /= ec50_fitted
            else:
                res.success = False
            return res

        def residuals(p: NDArray) -> NDArray:
            kwargs = dict(zip(param_names, p, strict=True))
            r = response - model_func(dose, **kwargs)
            if weights is not None:
                r = r * np.sqrt(weights)
            return r

        return least_squares(
            residuals, x0_in, method="trf", bounds=(lb, ub), jac="2-point",
            max_nfev=2000, xtol=1e-12, ftol=1e-12, gtol=1e-12,
        )

    # --- Candidate starting points ---
    # The Weibull-2 (W2.4) model has a mirror-image local optimum (swap the
    # asymptotes, negate the Hill slope). On decreasing data the data-driven
    # self-start seeds the wrong basin and converges silently to a ~14%-worse
    # RSS with swapped asymptotes. For an auto-start W2.4 fit, also try the
    # mirror start and keep whichever reaches the lower RSS — which recovers the
    # natural-label global optimum. Multistart never worsens a fit (it keeps the
    # better of the two). W1.4 is left untouched: it is the same curve family
    # with asymptotes labelled oppositely, so mirroring it would flip its labels;
    # its self-start already matches the reference.
    candidates = [x0]
    if start_was_none and model == "W2.4":
        b_i, t_i, h_i = (param_names.index("bottom"),
                         param_names.index("top"), param_names.index("hill"))
        mirror = x0.copy()
        mirror[b_i], mirror[t_i] = x0[t_i], x0[b_i]
        mirror[h_i] = -x0[h_i]
        mirror = np.clip(mirror, lb + 1e-15, ub - 1e-15)
        candidates.append(mirror)

    result = min(
        (_fit_from(c) for c in candidates),
        key=lambda r: float(np.sum(r.fun**2)),
    )

    # --- Extract ---
    popt = result.x
    res_vec = result.fun
    rss = float(np.sum(res_vec**2))
    converged = result.success
    n_iter = result.nfev

    jac = result.jac

    def _rss_of(theta: NDArray) -> float:
        """RSS at an arbitrary parameter vector (same objective the fit minimised)."""
        kwargs = dict(zip(param_names, theta, strict=True))
        r = response - model_func(dose, **kwargs)
        if weights is not None:
            r = r * np.sqrt(weights)
        return float(np.sum(r**2))

    cov, se = _compute_cov_se(_rss_of, popt, jac, rss, n_obs, n_params)

    # AIC / BIC
    aic = float(n_obs * np.log(rss / n_obs) + 2 * n_params)
    bic = float(n_obs * np.log(rss / n_obs) + n_params * np.log(n_obs))

    curve_params = CurveParams.from_array(popt, model)

    params = DoseResponseParams(
        curve=curve_params,
        se=se,
        cov=cov,
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
    result = Result(
        params=params,
        info={
            "model": model,
            "n_obs": n_obs,
            "converged": converged,
        },
        timing=None,
        backend_name="cpu",
    )
    return DoseResponseSolution(result)
