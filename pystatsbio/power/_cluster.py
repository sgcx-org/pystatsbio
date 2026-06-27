"""Power calculations for cluster randomized trials.

Adjusts individual-level power by the design effect (DEFF).

Validates against: R clusterPower, CRTSize
"""

from __future__ import annotations

import math

from pystatistics.core.exceptions import ValidationError

from pystatsbio.power._common import PowerResult, _check_power_args, _solve_parameter
from pystatsbio.power._means import _t_test_power

# ---------------------------------------------------------------------------
# Internal power computation
# ---------------------------------------------------------------------------

def _cluster_power(
    n_clusters: float,
    cluster_size: int,
    d: float,
    icc: float,
    alpha: float,
) -> float:
    """Compute power for a cluster randomized trial (two-arm parallel).

    The design effect inflates the required sample size:
        DEFF = 1 + (m - 1) * ICC
    where m = cluster_size.

    Effective individual-level n per arm = n_clusters * cluster_size / DEFF
    Then uses the two-sample t-test power formula.
    """
    deff = 1.0 + (cluster_size - 1.0) * icc
    n_eff = n_clusters * cluster_size / deff

    if n_eff < 2.0:
        return 0.0

    # Use the two-sample t-test power with effective n
    return _t_test_power(n_eff, d, alpha, "two-sided", "two-sample")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def power_cluster(
    n_clusters: int | None = None,
    cluster_size: int | None = None,
    effect_size: float | None = None,
    icc: float = 0.05,
    alpha: float = 0.05,
    power: float | None = None,
) -> PowerResult:
    """Power calculation for cluster randomized trial (two-arm parallel design).

    Adjusts individual-level sample size by the design effect:
    ``DEFF = 1 + (m - 1) * ICC`` where ``m`` = cluster_size.

    Exactly one of ``n_clusters``, ``effect_size``, ``power`` must be ``None``.

    Parameters
    ----------
    n_clusters : int or None
        Number of clusters per arm.
    cluster_size : int
        Average number of subjects per cluster. Always required.
    effect_size : float or None
        Cohen's d effect size.
    icc : float
        Intraclass correlation coefficient (default 0.05).
    alpha : float
        Significance level (default 0.05).
    power : float or None
        Desired power.

    Returns
    -------
    PowerResult

    Examples
    --------
    >>> r = power_cluster(cluster_size=20, effect_size=0.5, icc=0.05, alpha=0.05, power=0.80)
    >>> r.n  # clusters per arm
    8

    Validates against: R clusterPower, CRTSize
    """
    # --- Validate ---
    if cluster_size is None:
        raise ValidationError("cluster_size is always required")
    if cluster_size < 2:
        raise ValidationError(f"cluster_size must be >= 2, got {cluster_size}")
    if not (0.0 <= icc <= 1.0):
        raise ValidationError(f"icc must be in [0, 1], got {icc}")

    solve_for = _check_power_args(
        n=n_clusters, effect=effect_size, power=power, alpha=alpha, effect_name="effect_size",
    )

    deff = 1.0 + (cluster_size - 1.0) * icc

    if solve_for == "power":
        assert n_clusters is not None and effect_size is not None
        result_power = _cluster_power(float(n_clusters), cluster_size, effect_size, icc, alpha)
        result_n = n_clusters
        result_effect_size = effect_size

    elif solve_for == "n":
        assert effect_size is not None and power is not None
        if effect_size == 0.0:
            raise ValidationError("Cannot solve for n_clusters when effect_size = 0")
        raw_n = _solve_parameter(
            func=lambda x: _cluster_power(x, cluster_size, effect_size, icc, alpha),
            target=power,
            bracket=(1.0, 1e6),
        )
        result_n = math.ceil(raw_n)
        result_power = power
        result_effect_size = effect_size

    else:  # solve_for == "effect" (effect_size)
        assert n_clusters is not None and power is not None
        result_effect_size = _solve_parameter(
            func=lambda x: _cluster_power(float(n_clusters), cluster_size, x, icc, alpha),
            target=power,
            bracket=(1e-10, 100.0),
        )
        result_n = n_clusters
        result_power = power

    return PowerResult(
        n=result_n,
        power=result_power,
        effect_size=result_effect_size,
        alpha=alpha,
        alternative="two-sided",
        method="Cluster randomized trial power calculation",
        note=(
            f"n is clusters per arm; cluster_size = {cluster_size}; "
            f"ICC = {icc}; DEFF = {deff:.2f}"
        ),
    )
