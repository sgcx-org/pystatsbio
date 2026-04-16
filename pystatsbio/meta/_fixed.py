"""Fixed-effects (inverse-variance weighted) meta-analysis.

Computes the common-effect estimate under the assumption that all studies
share a single true effect size. Weights are the inverse sampling variances.

Validates against: R metafor::rma(method="FE")
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from pystatsbio.meta._common import MetaResult
from pystatsbio.meta._heterogeneity import cochran_q, h_squared, i_squared


def _fit_fixed(
    yi: NDArray,
    vi: NDArray,
    conf_level: float,
) -> MetaResult:
    """Fixed-effects meta-analysis.

    Weights: w_i = 1 / v_i
    Pooled estimate: mu = sum(w_i * y_i) / sum(w_i)
    SE: 1 / sqrt(sum(w_i))
    tau2 = 0 by definition.

    Parameters
    ----------
    yi : NDArray
        Effect sizes (already validated).
    vi : NDArray
        Sampling variances (already validated).
    conf_level : float
        Confidence level for the interval.

    Returns
    -------
    MetaResult
        Fixed-effects meta-analysis results.
    """
    k = len(yi)
    wi = 1.0 / vi
    sum_wi = np.sum(wi)

    estimate = float(np.sum(wi * yi) / sum_wi)
    se = float(1.0 / np.sqrt(sum_wi))

    z_crit = stats.norm.ppf((1 + conf_level) / 2)
    ci_lower = estimate - z_crit * se
    ci_upper = estimate + z_crit * se
    z_value = estimate / se
    p_value = float(2.0 * stats.norm.sf(abs(z_value)))

    Q, Q_df, Q_p = cochran_q(yi, vi)
    I2 = i_squared(Q, k)
    H2 = h_squared(Q, k)

    return MetaResult(
        estimate=estimate,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        z_value=z_value,
        p_value=p_value,
        tau2=0.0,
        tau2_se=None,
        tau=0.0,
        I2=I2,
        H2=H2,
        Q=Q,
        Q_df=Q_df,
        Q_p=Q_p,
        k=k,
        method="FE",
        conf_level=conf_level,
        weights=wi,
        yi=yi,
        vi=vi,
    )
