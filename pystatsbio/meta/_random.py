"""Random-effects meta-analysis estimators.

Implements three tau2 estimators for between-study variance:
- DerSimonian-Laird (DL): method-of-moments, closed-form
- REML: restricted maximum likelihood, iterative
- Paule-Mandel (PM): iterative generalized Q-statistic

After tau2 estimation, all methods compute the pooled estimate using
inverse-variance weights w_i* = 1 / (v_i + tau2). I2 and H2 are estimator-specific
(derived from each method's own tau2, matching metafor), not the Q-based value.

The standard error of tau2 is reported for the likelihood-based **REML** estimator
(from the expected/Fisher information). It is deliberately **not** reported for the
moment-based DerSimonian-Laird and Paule-Mandel estimators (``tau2_se`` is None):
their tau2 SEs are non-standard, estimator-specific large-sample quantities — use
``method="REML"`` if a principled SE of tau2 is needed.

Validates against: R metafor::rma(method="DL"|"REML"|"PM")
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pystatistics.core.exceptions import ConvergenceError
from pystatistics.core.result import Result
from scipy import optimize, stats

from pystatsbio.meta._common import MetaParams, MetaSolution
from pystatsbio.meta._heterogeneity import cochran_q


def _pool_random(
    yi: NDArray,
    vi: NDArray,
    tau2: float,
    tau2_se: float | None,
    method: str,
    conf_level: float,
) -> MetaSolution:
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
    MetaSolution
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

    # Cochran's Q (heterogeneity test) is estimator-independent (fixed-effect
    # weights) and reported as-is.
    Q, Q_df, Q_p = cochran_q(yi, vi)

    # I2 / H2 are estimator-specific: metafor derives them from THIS method's tau2
    # via the "typical" within-study variance s2, not from Q. For DerSimonian-Laird
    # this reduces exactly to the Q-based forms (s2 = (k-1)/c), so DL is unchanged;
    # for REML/PM it gives the estimator's own I2/H2 (the Q-based value was wrong).
    wi_fe = 1.0 / vi
    sum_wi_fe = np.sum(wi_fe)
    s2 = float((k - 1) * sum_wi_fe / (sum_wi_fe**2 - np.sum(wi_fe**2)))
    I2 = float(100.0 * tau2 / (tau2 + s2)) if (tau2 + s2) > 0 else 0.0
    H2 = float((tau2 + s2) / s2)
    tau = float(np.sqrt(tau2))

    params = MetaParams(
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
    result = Result(
        params=params,
        info={"method": method, "tau2": tau2, "k": k},
        timing=None,
        backend_name="cpu",
    )
    return MetaSolution(result)


def _fit_dl(
    yi: NDArray,
    vi: NDArray,
    conf_level: float,
) -> MetaSolution:
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
    MetaSolution
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
    """Standard error of tau2 from the REML *expected* (Fisher) information.

    The expected information for tau2 under REML is
        I(tau2) = 0.5 * [ S2 - 2*S3/S1 + (S2/S1)^2 ]
    with w_i = 1/(v_i + tau2), S1 = sum(w_i), S2 = sum(w_i^2), S3 = sum(w_i^3);
    se(tau2) = 1/sqrt(I). This is the expected-information SE that
    ``metafor::rma(method="REML")`` reports (matching it to ~1e-7), rather than
    the observed-information numerical second derivative used previously.

    Parameters
    ----------
    tau2 : float
        REML estimate of between-study variance.
    yi : NDArray
        Effect sizes (unused; kept for signature symmetry — the expected
        information does not depend on the observed y_i).
    vi : NDArray
        Sampling variances.

    Returns
    -------
    float
        Standard error of tau2, or NaN if the information is non-positive.
    """
    w = 1.0 / (vi + tau2)
    s1 = float(np.sum(w))
    s2 = float(np.sum(w**2))
    s3 = float(np.sum(w**3))
    info = 0.5 * (s2 - 2.0 * s3 / s1 + (s2 / s1) ** 2)
    if info > 0:
        return float(1.0 / np.sqrt(info))
    return float("nan")


def _fit_reml(
    yi: NDArray,
    vi: NDArray,
    conf_level: float,
) -> MetaSolution:
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
    MetaSolution
        REML random-effects results.

    Raises
    ------
    ConvergenceError
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
        raise ConvergenceError(
            f"REML optimization failed: {result.message}",
            iterations=int(getattr(result, "nfev", 0)),
            reason=str(result.message),
        )

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
) -> MetaSolution:
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
    MetaSolution
        PM random-effects results.

    Raises
    ------
    ConvergenceError
        If the root-finding algorithm fails to converge.
    """
    k = len(yi)
    f_at_zero = _pm_objective(0.0, yi, vi, k)

    if f_at_zero <= 0.0:
        tau2 = 0.0
    else:
        upper = max(10.0 * np.var(yi), 10.0 * np.max(vi), 100.0)
        n_expand = 0
        while _pm_objective(upper, yi, vi, k) > 0.0:
            upper *= 10.0
            n_expand += 1
            if upper > 1e15:
                raise ConvergenceError(
                    "Paule-Mandel: could not find upper bracket for tau2",
                    iterations=n_expand,
                    reason="objective stayed positive up to tau2=1e15",
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
