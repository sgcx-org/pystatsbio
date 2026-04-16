"""Meta-analysis: fixed-effects and random-effects pooling of study results.

Computes inverse-variance weighted pooled estimates, between-study
heterogeneity (tau2), and heterogeneity diagnostics (Q, I2, H2).

Supports four estimation methods:
- FE:   Fixed-effects (common-effect)
- DL:   DerSimonian-Laird (random-effects, method-of-moments)
- REML: Restricted maximum likelihood (random-effects)
- PM:   Paule-Mandel (random-effects, generalized Q)

Validates against: R metafor::rma()
"""

from numpy.typing import ArrayLike

from pystatsbio.meta._common import MetaResult, validate_inputs
from pystatsbio.meta._fixed import _fit_fixed
from pystatsbio.meta._heterogeneity import cochran_q, h_squared, i_squared
from pystatsbio.meta._random import _fit_dl, _fit_pm, _fit_reml

_VALID_METHODS = {"FE", "DL", "REML", "PM"}


def rma(
    yi: ArrayLike,
    vi: ArrayLike,
    *,
    method: str = "REML",
    conf_level: float = 0.95,
) -> MetaResult:
    """Random-effects (or fixed-effects) meta-analysis.

    Matches R's metafor::rma(yi, vi, method=...) for the common case
    of pre-computed effect sizes and sampling variances.

    Parameters
    ----------
    yi : ArrayLike
        Effect sizes (e.g., log odds ratios, standardized mean differences).
    vi : ArrayLike
        Corresponding sampling variances (NOT standard errors).
    method : str
        Estimation method. One of:
        - ``'FE'``: fixed-effects (inverse-variance weighted)
        - ``'DL'``: DerSimonian-Laird (default in many older packages)
        - ``'REML'``: restricted maximum likelihood (default, recommended)
        - ``'PM'``: Paule-Mandel
    conf_level : float
        Confidence level for confidence intervals. Default 0.95.

    Returns
    -------
    MetaResult
        Frozen dataclass with pooled estimate, heterogeneity statistics,
        and study weights.

    Raises
    ------
    ValueError
        If ``method`` is not one of the valid methods, or if inputs
        fail validation (wrong shape, negative variances, etc.).
    RuntimeError
        If iterative optimization (REML, PM) fails to converge.

    Examples
    --------
    >>> import numpy as np
    >>> from pystatsbio.meta import rma
    >>> yi = np.array([-0.89, -1.59, -1.35])
    >>> vi = np.array([0.035, 0.019, 0.014])
    >>> result = rma(yi, vi, method='DL')
    >>> print(f"Pooled: {result.estimate:.3f}, tau2: {result.tau2:.4f}")
    """
    method_upper = method.upper()
    if method_upper not in _VALID_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_VALID_METHODS)}, got {method!r}"
        )

    yi_arr, vi_arr = validate_inputs(yi, vi, conf_level)

    if method_upper == "FE":
        return _fit_fixed(yi_arr, vi_arr, conf_level)
    elif method_upper == "DL":
        return _fit_dl(yi_arr, vi_arr, conf_level)
    elif method_upper == "REML":
        return _fit_reml(yi_arr, vi_arr, conf_level)
    else:
        return _fit_pm(yi_arr, vi_arr, conf_level)


__all__ = [
    "MetaResult",
    "rma",
    "cochran_q",
    "i_squared",
    "h_squared",
]
