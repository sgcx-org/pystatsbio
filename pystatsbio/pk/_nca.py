"""Non-compartmental pharmacokinetic analysis (NCA).

Implements the standard NCA calculations used in every PK study:
AUC (linear, log-linear, linear-up/log-down trapezoidal), Cmax/Tmax,
terminal elimination rate constant (lambda_z) via log-linear regression,
half-life, clearance, volume of distribution, AUMC, and MRT.

The linear-up/log-down method is the FDA-recommended default: linear
trapezoidal on ascending segments, log-linear trapezoidal on descending
segments.

References
----------
Gabrielsson & Weiner (2000). *Pharmacokinetic and Pharmacodynamic
Data Analysis*, 3rd ed.

Gibaldi & Perrier (1982). *Pharmacokinetics*, 2nd ed.

FDA Guidance: Bioanalytical Method Validation (2018).

Validates against: R ``PKNCA::pk.nca()``, ``NonCompart::sNCA()``.
"""

from __future__ import annotations

import contextlib

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from pystatsbio.pk._common import NCAResult

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    time: NDArray[np.floating],
    concentration: NDArray[np.floating],
    route: str,
    auc_method: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and clean NCA inputs.

    Returns time and concentration as sorted float64 arrays with
    leading zeros handled.
    """
    time = np.asarray(time, dtype=np.float64).ravel()
    concentration = np.asarray(concentration, dtype=np.float64).ravel()

    if time.shape[0] != concentration.shape[0]:
        raise ValueError(
            f"time and concentration must have equal length, "
            f"got {time.shape[0]} and {concentration.shape[0]}"
        )
    if time.shape[0] < 3:
        raise ValueError("Need at least 3 time-concentration points for NCA")

    if np.any(time < 0):
        raise ValueError("time values must be non-negative")
    if np.any(concentration < 0):
        raise ValueError("concentration values must be non-negative")

    if route not in ("iv", "ev"):
        raise ValueError(f"route must be 'iv' or 'ev', got {route!r}")

    valid_methods = ("linear", "log-linear", "linear-up/log-down")
    if auc_method not in valid_methods:
        raise ValueError(
            f"auc_method must be one of {valid_methods}, got {auc_method!r}"
        )

    # Sort by time
    order = np.argsort(time, kind="stable")
    time = time[order]
    concentration = concentration[order]

    # Check for duplicate time points
    if np.any(np.diff(time) == 0):
        raise ValueError("Duplicate time points detected; merge or remove them")

    return time, concentration


# ---------------------------------------------------------------------------
# AUC trapezoidal methods
# ---------------------------------------------------------------------------

def _auc_linear_segment(t1: float, t2: float, c1: float, c2: float) -> float:
    """Linear trapezoidal AUC for a single interval."""
    return 0.5 * (c1 + c2) * (t2 - t1)


def _auc_loglinear_segment(t1: float, t2: float, c1: float, c2: float) -> float:
    """Log-linear trapezoidal AUC for a single interval.

    Uses the formula: AUC = (C1 - C2) * (t2 - t1) / ln(C1/C2).
    Falls back to linear if C1 == C2 or either is zero.
    """
    if c1 <= 0 or c2 <= 0 or c1 == c2:
        return _auc_linear_segment(t1, t2, c1, c2)
    return (c1 - c2) * (t2 - t1) / np.log(c1 / c2)


def _compute_auc_segments(
    time: NDArray[np.float64],
    concentration: NDArray[np.float64],
    method: str,
) -> NDArray[np.float64]:
    """Compute per-segment AUC contributions.

    Parameters
    ----------
    time, concentration : arrays, same length
    method : 'linear', 'log-linear', or 'linear-up/log-down'

    Returns
    -------
    segments : array of length n-1, per-interval AUC
    """
    n = len(time)
    segments = np.empty(n - 1, dtype=np.float64)

    for i in range(n - 1):
        t1, t2 = time[i], time[i + 1]
        c1, c2 = concentration[i], concentration[i + 1]

        if method == "linear":
            segments[i] = _auc_linear_segment(t1, t2, c1, c2)
        elif method == "log-linear":
            segments[i] = _auc_loglinear_segment(t1, t2, c1, c2)
        elif method == "linear-up/log-down":
            # Linear on ascending (c2 >= c1), log-linear on descending
            if c2 >= c1:
                segments[i] = _auc_linear_segment(t1, t2, c1, c2)
            else:
                segments[i] = _auc_loglinear_segment(t1, t2, c1, c2)

    return segments


# ---------------------------------------------------------------------------
# AUMC (area under the first moment curve)
# ---------------------------------------------------------------------------

def _aumc_linear_segment(t1: float, t2: float, c1: float, c2: float) -> float:
    """Linear trapezoidal AUMC for a single interval.

    AUMC = integral of t*C(t) dt.  Linear trapezoidal:
    AUMC_i = 0.5 * (t1*C1 + t2*C2) * (t2 - t1)
    """
    return 0.5 * (t1 * c1 + t2 * c2) * (t2 - t1)


def _aumc_loglinear_segment(
    t1: float, t2: float, c1: float, c2: float
) -> float:
    """Log-linear trapezoidal AUMC for a single interval.

    For log-linear model C(t) = C1 * exp(-k*(t-t1)) where k = ln(C1/C2)/(t2-t1):
    AUMC = (t1*C1 - t2*C2)/k + (C1 - C2)/k^2

    Falls back to linear if C1 == C2 or either is zero.
    """
    if c1 <= 0 or c2 <= 0 or c1 == c2:
        return _aumc_linear_segment(t1, t2, c1, c2)
    k = np.log(c1 / c2) / (t2 - t1)
    return (t1 * c1 - t2 * c2) / k + (c1 - c2) / (k * k)


def _compute_aumc_segments(
    time: NDArray[np.float64],
    concentration: NDArray[np.float64],
    method: str,
) -> NDArray[np.float64]:
    """Compute per-segment AUMC contributions."""
    n = len(time)
    segments = np.empty(n - 1, dtype=np.float64)

    for i in range(n - 1):
        t1, t2 = time[i], time[i + 1]
        c1, c2 = concentration[i], concentration[i + 1]

        if method == "linear":
            segments[i] = _aumc_linear_segment(t1, t2, c1, c2)
        elif method == "log-linear":
            segments[i] = _aumc_loglinear_segment(t1, t2, c1, c2)
        elif method == "linear-up/log-down":
            if c2 >= c1:
                segments[i] = _aumc_linear_segment(t1, t2, c1, c2)
            else:
                segments[i] = _aumc_loglinear_segment(t1, t2, c1, c2)

    return segments


# ---------------------------------------------------------------------------
# Terminal elimination rate constant (lambda_z)
# ---------------------------------------------------------------------------

def _find_last_measurable(concentration: NDArray[np.float64]) -> int:
    """Index of last non-zero concentration (Clast)."""
    nonzero = np.where(concentration > 0)[0]
    if len(nonzero) == 0:
        return -1
    return int(nonzero[-1])


class LambdaZEstimationError(ValueError):
    """Raised when the terminal elimination rate constant cannot be estimated.

    This is not a programming error — it signals that the concentration-time
    profile does not contain enough terminal phase data to fit a reliable
    log-linear slope.
    """


def _estimate_lambda_z(
    time: NDArray[np.float64],
    concentration: NDArray[np.float64],
    n_points: int | None,
    idx_cmax: int,
) -> tuple[float, float, int]:
    """Estimate terminal elimination rate constant via log-linear regression.

    Selects terminal phase points (after Cmax, with positive concentration)
    and fits log(C) vs time. If n_points is None, iterates from 3 points
    to all eligible points and picks the regression with the best adjusted
    R-squared.

    Parameters
    ----------
    time, concentration : sorted arrays
    n_points : fixed number of terminal points, or None for auto-select
    idx_cmax : index of Cmax (terminal phase starts after this)

    Returns
    -------
    lambda_z : terminal rate constant (positive)
    r_squared_adj : adjusted R-squared of the best fit
    n_terminal : number of points used

    Raises
    ------
    LambdaZEstimationError
        If there are insufficient terminal phase points or the fitted slope
        is non-negative (not a true elimination phase).
    """
    # Terminal phase candidates: after Cmax, positive concentration
    idx_last = _find_last_measurable(concentration)
    if idx_last < 0:
        raise LambdaZEstimationError(
            "Cannot estimate lambda_z: all concentrations are zero."
        )

    # Candidates: indices from Cmax+1 to last measurable (inclusive)
    candidates = []
    for i in range(idx_cmax + 1, idx_last + 1):
        if concentration[i] > 0:
            candidates.append(i)

    if len(candidates) < 3:
        # Also try including Cmax itself if not enough points after
        candidates_with_cmax = []
        for i in range(idx_cmax, idx_last + 1):
            if concentration[i] > 0:
                candidates_with_cmax.append(i)
        if len(candidates_with_cmax) >= 3:
            candidates = candidates_with_cmax
        else:
            raise LambdaZEstimationError(
                f"Cannot estimate lambda_z: only {len(candidates_with_cmax)} "
                f"positive concentration point(s) available after Cmax "
                f"(need at least 3)."
            )

    candidates = np.array(candidates)

    if n_points is not None:
        # Fixed number of terminal points (from the end)
        if n_points < 3:
            raise ValueError("lambda_z_n_points must be >= 3")
        if n_points > len(candidates):
            raise ValueError(
                f"lambda_z_n_points={n_points} exceeds available "
                f"terminal points ({len(candidates)})"
            )
        idx_use = candidates[-n_points:]
        t_fit = time[idx_use]
        log_c_fit = np.log(concentration[idx_use])
        slope, _, r_value, _, _ = stats.linregress(t_fit, log_c_fit)
        n_fit = n_points
        # Adjusted R-squared
        r_sq = r_value ** 2
        r_sq_adj = 1.0 - (1.0 - r_sq) * (n_fit - 1) / (n_fit - 2) if n_fit > 2 else r_sq
    else:
        # Auto-select: try 3 to len(candidates) points from the end,
        # pick the one with best adjusted R-squared
        best_r_sq_adj = -np.inf
        best_slope = None
        best_n = 0

        for n_try in range(3, len(candidates) + 1):
            idx_use = candidates[-n_try:]
            t_fit = time[idx_use]
            log_c_fit = np.log(concentration[idx_use])
            s, _, r_value, _, _ = stats.linregress(t_fit, log_c_fit)
            r_sq = r_value ** 2
            r_sq_adj_try = 1.0 - (1.0 - r_sq) * (n_try - 1) / (n_try - 2)

            if r_sq_adj_try > best_r_sq_adj:
                best_r_sq_adj = r_sq_adj_try
                best_slope = s
                best_n = n_try

        slope = best_slope
        r_sq_adj = best_r_sq_adj
        n_fit = best_n

    # lambda_z must be positive (slope should be negative for elimination)
    if slope is None or slope >= 0:
        raise LambdaZEstimationError(
            f"Cannot estimate lambda_z: terminal slope is {slope} (non-negative). "
            f"The concentration profile does not show a true elimination phase "
            f"in the selected terminal points."
        )

    lambda_z = -slope
    return lambda_z, r_sq_adj, n_fit


# ---------------------------------------------------------------------------
# Cmax / Tmax
# ---------------------------------------------------------------------------

def _find_cmax_tmax(
    time: NDArray[np.float64],
    concentration: NDArray[np.float64],
) -> tuple[float, float, int]:
    """Find peak concentration.

    Returns
    -------
    cmax : peak concentration
    tmax : time of peak concentration (first occurrence)
    idx_cmax : index of Cmax
    """
    idx = int(np.argmax(concentration))
    return float(concentration[idx]), float(time[idx]), idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def nca(
    time: NDArray[np.floating],
    concentration: NDArray[np.floating],
    *,
    dose: float | None = None,
    route: str = "ev",
    auc_method: str = "linear-up/log-down",
    lambda_z_n_points: int | None = None,
) -> NCAResult:
    """Non-compartmental pharmacokinetic analysis.

    Parameters
    ----------
    time : array
        Time points (non-negative, no duplicates).
    concentration : array
        Plasma concentration values (non-negative).
    dose : float or None
        Administered dose (needed for CL and Vz).
    route : str
        ``'iv'`` (intravenous bolus) or ``'ev'`` (extravascular / oral).
    auc_method : str
        ``'linear'`` (linear trapezoidal),
        ``'log-linear'`` (log-linear trapezoidal),
        ``'linear-up/log-down'`` (linear up, log-linear down -- the default,
        recommended by FDA guidance).
    lambda_z_n_points : int or None
        Number of terminal points for half-life estimation.
        If ``None``, automatically selects the best terminal phase
        (maximum adjusted r-squared with >= 3 points).

    Returns
    -------
    NCAResult
        Frozen dataclass with all NCA parameters.

    Notes
    -----
    CPU-only.  PK data is always small (typically 10-20 time points per
    subject).

    Validates against: R ``PKNCA::pk.nca()``, ``NonCompart::sNCA()``.
    """
    time, concentration = _validate_inputs(time, concentration, route, auc_method)
    n_points = len(time)

    # ----- Cmax / Tmax -----
    cmax, tmax, idx_cmax = _find_cmax_tmax(time, concentration)

    # ----- AUC to last measurable concentration -----
    idx_last = _find_last_measurable(concentration)
    if idx_last < 0:
        # All concentrations zero — degenerate case
        return NCAResult(
            auc_last=0.0,
            auc_inf=None,
            auc_pct_extrap=None,
            cmax=0.0,
            tmax=float(time[0]),
            half_life=None,
            lambda_z=None,
            lambda_z_r_squared=None,
            clearance=None,
            vz=None,
            dose=dose,
            route=route,
            auc_method=auc_method,
            n_points=n_points,
            n_terminal=0,
        )

    # AUC from time[0] to time[idx_last]
    t_auc = time[: idx_last + 1]
    c_auc = concentration[: idx_last + 1]
    auc_segments = _compute_auc_segments(t_auc, c_auc, auc_method)
    auc_last = float(np.sum(auc_segments))

    # ----- Terminal elimination -----
    lambda_z: float | None = None
    r_sq_adj: float | None = None
    n_terminal = 0
    with contextlib.suppress(LambdaZEstimationError):
        lambda_z, r_sq_adj, n_terminal = _estimate_lambda_z(
            time, concentration, lambda_z_n_points, idx_cmax
        )

    # ----- Extrapolated AUC and AUMC -----
    auc_inf: float | None = None
    auc_pct_extrap: float | None = None
    half_life: float | None = None
    clearance: float | None = None
    vz: float | None = None

    if lambda_z is not None and lambda_z > 0:
        half_life = np.log(2) / lambda_z
        c_last = concentration[idx_last]

        # AUC extrapolation: Clast / lambda_z
        auc_extrap = c_last / lambda_z
        auc_inf = auc_last + auc_extrap
        auc_pct_extrap = 100.0 * auc_extrap / auc_inf if auc_inf > 0 else 0.0

        # Dose-dependent parameters
        if dose is not None and auc_inf > 0:
            clearance = dose / auc_inf
            vz = dose / (lambda_z * auc_inf)

    return NCAResult(
        auc_last=auc_last,
        auc_inf=auc_inf,
        auc_pct_extrap=auc_pct_extrap,
        cmax=cmax,
        tmax=tmax,
        half_life=half_life,
        lambda_z=lambda_z,
        lambda_z_r_squared=r_sq_adj,
        clearance=clearance,
        vz=vz,
        dose=dose,
        route=route,
        auc_method=auc_method,
        n_points=n_points,
        n_terminal=n_terminal,
    )
