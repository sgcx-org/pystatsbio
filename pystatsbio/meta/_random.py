"""Random-effects meta-analysis estimators.

Implements three tau2 estimators for between-study variance:
- DerSimonian-Laird (DL): method-of-moments, closed-form
- REML: restricted maximum likelihood, iterative
- Paule-Mandel (PM): iterative generalized Q-statistic

After tau2 estimation, all methods compute the pooled estimate using
inverse-variance weights w_i* = 1 / (v_i + tau2).

Validates against: R metafor::rma(method="DL"|"REML"|"PM")
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

from pystatsbio.meta._common import MetaResult
from pystatsbio.meta._heterogeneity import cochran_q, h_squared, i_squared


def _pool_random(
    yi: NDArray,
    vi: NDArray,
    tau2: float,
    tau2_se: float | None,
    method: str,
    conf_level: float,
) -> MetaResult:
    """Pool studies using random-effects weights for a given tau2.

    Parameters
    ----------
    yi : NDArray
        Effect sizes.
    vi : NDArray
        Sampling variances.
    tau2 : float
        Between-study variance estimate.
    tau2_se : float or None
        Standard error of tau2 (None if not available).
    method : str
        Label for the estimation method.
    conf_level : float
        Confidence level.

    Returns
    -------
    MetaResult
        Random-effects meta-analysis results.
    """
    k = len(yi)
    wi_star = 1.0 / (vi + tau2)
    sum_wi_star = np.sum(wi_star)

    estimate = float(np.sum(wi_star * yi) / sum_wi_star)
    se = float(1.0 / np.sqrt(sum_wi_star))

    z_crit = stats.norm.ppf((1 + conf_level) / 2)
    ci_lower = estimate - z_crit * se
    ci_upper = estimate + z_crit * se
    z_value = estimate / se
    p_value = float(2.0 * stats.norm.sf(abs(z_value)))

    Q, Q_df, Q_p = cochran_q(yi, vi)
    I2 = i_squared(Q, k)
    H2 = h_squared(Q, k)
    tau = float(np.sqrt(tau2))

    return MetaResult(
        estimate=estimate,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        z_value=z_value,
        p_value=p_value,
        tau2=tau2,
        tau2_se=tau2_se,
        tau=tau,
        I2=I2,
        H2=H2,
        Q=Q,
        Q_df=Q_df,
        Q_p=Q_p,
        k=k,
        method=method,
        conf_level=conf_level,
        weights=wi_star,
        yi=yi,
        vi=vi,
    )


def _fit_dl(
    yi: NDArray,
    vi: NDArray,
    conf_level: float,
) -> MetaResult:
    """DerSimonian-Laird estimator for random-effects meta-analysis.

    Closed-form method-of-moments estimator:
        tau2_DL = max(0, (Q - (k-1)) / (sum(w_i) - sum(w_i^2)/sum(w_i)))
    where Q = sum(w_i * (y_i - mu_FE)^2) and w_i = 1/v_i.

    Parameters
    ----------
    yi : NDArray
        Effect sizes (already validated).
    vi : NDArray
        Sampling variances (already validated).
    conf_level : float
        Confidence level.

    Returns
    -------
    MetaResult
        DL random-effects results.
    """
    wi = 1.0 / vi
    sum_wi = np.sum(wi)
    mu_fe = float(np.sum(wi * yi) / sum_wi)

    Q = float(np.sum(wi * (yi - mu_fe) ** 2))
    k = len(yi)
    c = float(sum_wi - np.sum(wi**2) / sum_wi)

    tau2 = max(0.0, (Q - (k - 1)) / c)

    return _pool_random(yi, vi, tau2, tau2_se=None, method="DL", conf_level=conf_level)


def _reml_nll(tau2: float, yi: NDArray, vi: NDArray) -> float:
    """Negative restricted log-likelihood for REML estimation.

    l_R(tau2) = -0.5 * [k*log(2*pi) + sum(log(v_i + tau2))
                        + log(sum(1/(v_i + tau2)))
                        + sum((y_i - mu(tau2))^2 / (v_i + tau2))]

    We return the negative so that minimization finds the REML estimate.

    Parameters
    ----------
    tau2 : float
        Candidate between-study variance.
    yi : NDArray
        Effect sizes.
    vi : NDArray
        Sampling variances.

    Returns
    -------
    float
        Negative restricted log-likelihood.
    """
    k = len(yi)
    wi = 1.0 / (vi + tau2)
    sum_wi = np.sum(wi)
    mu = np.sum(wi * yi) / sum_wi

    ll = -0.5 * (
        k * np.log(2.0 * np.pi)
        + np.sum(np.log(vi + tau2))
        + np.log(sum_wi)
        + np.sum(wi * (yi - mu) ** 2)
    )
    return -float(ll)


def _reml_tau2_se(tau2: float, yi: NDArray, vi: NDArray) -> float:
    """Standard error of tau2 from observed Fisher information.

    The second derivative of the restricted log-likelihood with respect
    to tau2 is:
        d2l/d(tau2)^2 = 0.5 * [sum(w_i^2) - 2*sum(w_i^3*(y_i-mu)^2)
                                + (sum(w_i^2)/sum(w_i))^2
                                - sum(w_i^2)^2/sum(w_i)]
    where w_i = 1/(v_i + tau2).

    We compute it numerically via central differences for robustness.

    Parameters
    ----------
    tau2 : float
        REML estimate of between-study variance.
    yi : NDArray
        Effect sizes.
    vi : NDArray
        Sampling variances.

    Returns
    -------
    float
        Standard error of tau2, or NaN if information matrix is non-positive.
    """
    h = max(1e-6, tau2 * 1e-4)
    f0 = _reml_nll(tau2, yi, vi)
    fp = _reml_nll(tau2 + h, yi, vi)
    fm = _reml_nll(tau2 - h if tau2 - h > 0 else 0.0, yi, vi)
    h_actual_left = tau2 - (tau2 - h if tau2 - h > 0 else 0.0)
    h_actual_right = h

    if abs(h_actual_left - h_actual_right) < 1e-12:
        d2 = (fp - 2.0 * f0 + fm) / (h**2)
    else:
        d2 = 2.0 * (
            fp / (h_actual_right * (h_actual_right + h_actual_left))
            - f0 / (h_actual_right * h_actual_left)
            + fm / (h_actual_left * (h_actual_right + h_actual_left))
        )

    if d2 > 0:
        return float(1.0 / np.sqrt(d2))
    return float("nan")


def _fit_reml(
    yi: NDArray,
    vi: NDArray,
    conf_level: float,
) -> MetaResult:
    """REML (Restricted Maximum Likelihood) estimator.

    Maximizes the restricted log-likelihood over tau2 >= 0 using
    scipy.optimize.minimize_scalar with bounded method.

    The SE of tau2 is obtained from the observed Fisher information
    (numerical second derivative of the restricted log-likelihood).

    Parameters
    ----------
    yi : NDArray
        Effect sizes (already validated).
    vi : NDArray
        Sampling variances (already validated).
    conf_level : float
        Confidence level.

    Returns
    -------
    MetaResult
        REML random-effects results.

    Raises
    ------
    RuntimeError
        If the optimizer fails to converge.
    """
    upper_bound = max(10.0 * np.var(yi), 10.0 * np.max(vi), 100.0)

    result = optimize.minimize_scalar(
        _reml_nll,
        bounds=(0.0, upper_bound),
        args=(yi, vi),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"REML optimization failed: {result.message}")

    tau2 = float(max(0.0, result.x))
    tau2_se = _reml_tau2_se(tau2, yi, vi)

    return _pool_random(
        yi, vi, tau2, tau2_se=tau2_se, method="REML", conf_level=conf_level
    )


def _pm_objective(tau2: float, yi: NDArray, vi: NDArray, k: int) -> float:
    """Paule-Mandel objective: Q*(tau2) - (k - 1).

    The PM estimator finds tau2 such that the generalized Q statistic
    equals its expected value under the random-effects model:
        Q*(tau2) = sum(w_i* * (y_i - mu*)^2) = k - 1
    where w_i* = 1/(v_i + tau2).

    Parameters
    ----------
    tau2 : float
        Candidate between-study variance.
    yi : NDArray
        Effect sizes.
    vi : NDArray
        Sampling variances.
    k : int
        Number of studies.

    Returns
    -------
    float
        Q*(tau2) - (k - 1).
    """
    wi = 1.0 / (vi + tau2)
    mu = np.sum(wi * yi) / np.sum(wi)
    Q_star = float(np.sum(wi * (yi - mu) ** 2))
    return Q_star - (k - 1)


def _fit_pm(
    yi: NDArray,
    vi: NDArray,
    conf_level: float,
) -> MetaResult:
    """Paule-Mandel estimator for random-effects meta-analysis.

    Iteratively finds tau2 such that Q*(tau2) = k - 1, where
    Q* uses weights w_i* = 1/(v_i + tau2).

    Uses scipy.optimize.brentq for root-finding. If the Q statistic
    at tau2=0 is already <= k-1 (no evidence of heterogeneity),
    tau2 is set to 0.

    Parameters
    ----------
    yi : NDArray
        Effect sizes (already validated).
    vi : NDArray
        Sampling variances (already validated).
    conf_level : float
        Confidence level.

    Returns
    -------
    MetaResult
        PM random-effects results.

    Raises
    ------
    RuntimeError
        If the root-finding algorithm fails to converge.
    """
    k = len(yi)
    f_at_zero = _pm_objective(0.0, yi, vi, k)

    if f_at_zero <= 0.0:
        tau2 = 0.0
    else:
        upper = max(10.0 * np.var(yi), 10.0 * np.max(vi), 100.0)
        while _pm_objective(upper, yi, vi, k) > 0.0:
            upper *= 10.0
            if upper > 1e15:
                raise RuntimeError(
                    "Paule-Mandel: could not find upper bracket for tau2"
                )

        tau2 = float(
            optimize.brentq(
                _pm_objective,
                0.0,
                upper,
                args=(yi, vi, k),
                xtol=1e-10,
                maxiter=1000,
            )
        )
        tau2 = max(0.0, tau2)

    return _pool_random(yi, vi, tau2, tau2_se=None, method="PM", conf_level=conf_level)
