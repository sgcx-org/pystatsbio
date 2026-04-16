"""Sandwich (robust/Huber-White) variance estimator for GEE.

Computes both the naive (model-based) and sandwich (robust)
variance-covariance matrices for GEE regression coefficients.

The sandwich estimator V_robust = B^{-1} M B^{-1} is consistent
even if the working correlation structure is misspecified, which is
the key theoretical property that makes GEE practical.

References
----------
Liang, K.-Y. & Zeger, S. L. (1986). Longitudinal data analysis using
generalized linear models. Biometrika, 73(1), 13-22.

Zeger, S. L. & Liang, K.-Y. (1986). Longitudinal data analysis for
discrete and continuous outcomes. Biometrics, 42(1), 121-130.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.regression.families import Family

from pystatsbio.gee._correlation import CorrStructure
from pystatsbio.gee._estimating_equations import _group_by_cluster


def sandwich_variance(
    X: NDArray,
    y: NDArray,
    mu: NDArray,
    beta: NDArray,
    cluster_ids: NDArray,
    family: Family,
    corr: CorrStructure,
    phi: float,
) -> tuple[NDArray, NDArray]:
    """Compute naive and sandwich variance-covariance matrices.

    Naive (model-based):
        V_naive = B^{-1} = [sum_i D_i' W_i^{-1} D_i]^{-1}

    Sandwich (robust):
        V_robust = B^{-1} M B^{-1}
        where M = sum_i (D_i' W_i^{-1} e_i)(D_i' W_i^{-1} e_i)'
        and e_i = y_i - mu_i is the residual vector for cluster i.

    The sandwich estimator is consistent for the true variance of
    beta_hat even if the working correlation structure is misspecified.

    Parameters
    ----------
    X : NDArray
        Design matrix (n, p).
    y : NDArray
        Response vector (n,).
    mu : NDArray
        Fitted values (n,).
    beta : NDArray
        Estimated coefficients (p,).
    cluster_ids : NDArray
        Cluster identifiers (n,).
    family : Family
        GLM family (provides link and variance).
    corr : CorrStructure
        Working correlation structure.
    phi : float
        Estimated dispersion parameter.

    Returns
    -------
    tuple[NDArray, NDArray]
        (naive_vcov, robust_vcov) both of shape (p, p).
    """
    n, p = X.shape
    link = family.link

    # Group by cluster
    y_groups, X_groups, idx_groups = _group_by_cluster(y, X, cluster_ids)

    B = np.zeros((p, p))  # "bread" matrix
    M = np.zeros((p, p))  # "meat" matrix

    for y_i, X_i, idx_i in zip(y_groups, X_groups, idx_groups):
        n_i = len(y_i)
        mu_i = mu[idx_i]
        var_i = family.variance(mu_i)
        eta_i = link.link(mu_i)
        mu_eta_i = link.mu_eta(eta_i)

        # D_i = diag(dmu/deta) @ X_i
        D_i = X_i * mu_eta_i[:, None]

        # Working covariance W_i = phi * A_i^{1/2} R_i A_i^{1/2}
        A_sqrt = np.sqrt(np.maximum(var_i, 1e-10))
        R_i = corr.working_corr(n_i)
        W_i = phi * (A_sqrt[:, None] * R_i * A_sqrt[None, :])

        # W_i^{-1} with numerical stabilization
        W_i_inv = np.linalg.inv(W_i + np.eye(n_i) * 1e-10)

        # Accumulate B
        DtWinv = D_i.T @ W_i_inv
        B += DtWinv @ D_i

        # Cluster residual and its contribution to meat
        e_i = y_i - mu_i
        u_i = DtWinv @ e_i  # (p,) vector
        M += np.outer(u_i, u_i)

    # Naive: B^{-1}
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        B_inv = np.linalg.inv(B + np.eye(p) * 1e-6)

    naive_vcov = B_inv

    # Sandwich: B^{-1} M B^{-1}
    robust_vcov = B_inv @ M @ B_inv

    return naive_vcov, robust_vcov
