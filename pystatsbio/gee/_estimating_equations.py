"""Core GEE iterative fitting algorithm.

Implements the Generalized Estimating Equations algorithm of
Liang & Zeger (1986) for fitting regression models to clustered data
with a specified working correlation structure.

The algorithm iterates between:
1. Updating the working correlation parameters from Pearson residuals
2. Updating the regression coefficients via a modified score equation

References
----------
Liang, K.-Y. & Zeger, S. L. (1986). Longitudinal data analysis using
generalized linear models. Biometrika, 73(1), 13-22.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.regression.families import Family

from pystatsbio.gee._correlation import CorrStructure


def _group_by_cluster(
    y: NDArray,
    X: NDArray,
    cluster_ids: NDArray,
) -> tuple[list[NDArray], list[NDArray], list[NDArray]]:
    """Group observations by cluster_id.

    Sorts by cluster_id and returns per-cluster views of y, X,
    and the original indices within each cluster.

    Parameters
    ----------
    y : NDArray
        Response vector (n,).
    X : NDArray
        Design matrix (n, p).
    cluster_ids : NDArray
        Cluster identifiers (n,).

    Returns
    -------
    tuple
        (y_groups, X_groups, idx_groups) where each is a list of arrays,
        one per cluster, sorted by cluster_id.
    """
    order = np.argsort(cluster_ids, kind="stable")
    sorted_ids = cluster_ids[order]
    sorted_y = y[order]
    sorted_X = X[order]

    # Find breakpoints between clusters
    breaks = np.where(sorted_ids[:-1] != sorted_ids[1:])[0] + 1
    idx_splits = np.split(order, breaks)
    y_groups = np.split(sorted_y, breaks)
    X_groups = np.split(sorted_X, breaks)

    return y_groups, X_groups, idx_splits


def _initialize_beta(
    y: NDArray,
    X: NDArray,
    family: Family,
    max_irls: int = 25,
) -> NDArray:
    """Initialize beta via independence IRLS (simplified GLM fit).

    Runs a basic iteratively reweighted least squares using the family's
    variance function and link, assuming independence across all observations.

    Parameters
    ----------
    y : NDArray
        Response vector (n,).
    X : NDArray
        Design matrix (n, p).
    family : Family
        GLM family with link and variance functions.
    max_irls : int
        Maximum IRLS iterations for initialization.

    Returns
    -------
    NDArray
        Initial coefficient estimates (p,).
    """
    n, p = X.shape
    link = family.link

    # Initialize mu from the family
    mu = family.initialize(y).astype(np.float64)
    eta = link.link(mu)

    for _ in range(max_irls):
        mu_eta_deriv = link.mu_eta(eta)
        var_mu = family.variance(mu)
        # IRLS weights: w = (dmu/deta)^2 / V(mu)
        w = mu_eta_deriv**2 / np.maximum(var_mu, 1e-10)
        # Working response: z = eta + (y - mu) / (dmu/deta)
        z = eta + (y - mu) / np.maximum(mu_eta_deriv, 1e-10)

        # Weighted least squares: (X'WX)^{-1} X'Wz
        W_sqrt = np.sqrt(np.maximum(w, 1e-10))
        Xw = X * W_sqrt[:, None]
        zw = z * W_sqrt
        beta_new, _, _, _ = np.linalg.lstsq(Xw, zw, rcond=None)

        eta_new = X @ beta_new
        mu_new = link.linkinv(eta_new)

        # Check convergence
        change = np.max(np.abs(eta_new - eta)) / (np.max(np.abs(eta)) + 0.1)
        eta = eta_new
        mu = mu_new
        if change < 1e-8:
            break

    return beta_new


def _compute_cluster_quantities(
    y_i: NDArray,
    X_i: NDArray,
    beta: NDArray,
    family: Family,
    corr: CorrStructure,
    phi: float,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Compute per-cluster quantities for the GEE score equation.

    For cluster i, computes:
    - mu_i: predicted means
    - D_i: derivative matrix (dmu/deta * X)
    - W_i_inv: inverse of working covariance
    - resid_i: y_i - mu_i
    - pearson_i: Pearson residuals

    Parameters
    ----------
    y_i : NDArray
        Response for cluster i (n_i,).
    X_i : NDArray
        Design matrix for cluster i (n_i, p).
    beta : NDArray
        Current coefficient estimates (p,).
    family : Family
        GLM family.
    corr : CorrStructure
        Working correlation structure.
    phi : float
        Dispersion parameter.

    Returns
    -------
    tuple
        (mu_i, D_i, W_i_inv, resid_i, pearson_i)
    """
    link = family.link
    n_i = len(y_i)

    eta_i = X_i @ beta
    mu_i = link.linkinv(eta_i)
    var_i = family.variance(mu_i)

    # D_i = diag(dmu/deta) @ X_i
    mu_eta_i = link.mu_eta(eta_i)
    D_i = X_i * mu_eta_i[:, None]

    # Working covariance: W_i = phi * A_i^{1/2} @ R_i @ A_i^{1/2}
    # where A_i = diag(V(mu_i))
    A_sqrt = np.sqrt(np.maximum(var_i, 1e-10))
    R_i = corr.working_corr(n_i)
    W_i = phi * (A_sqrt[:, None] * R_i * A_sqrt[None, :])

    # Solve W_i^{-1} using Cholesky or general solve
    W_i_inv = np.linalg.inv(W_i + np.eye(n_i) * 1e-10)

    resid_i = y_i - mu_i
    pearson_i = resid_i / np.maximum(A_sqrt, 1e-10)

    return mu_i, D_i, W_i_inv, resid_i, pearson_i


def _fit_gee(
    y: NDArray,
    X: NDArray,
    cluster_ids: NDArray,
    family: Family,
    corr: CorrStructure,
    tol: float,
    max_iter: int,
    scale_fix: float | None,
) -> tuple[NDArray, NDArray, NDArray, float, int, bool]:
    """Core GEE fitting algorithm (Liang & Zeger, 1986).

    Algorithm:
    1. Initialize beta via independence IRLS.
    2. Repeat until convergence:
       a. Compute mu_i, V_i for each cluster.
       b. Compute Pearson residuals.
       c. Estimate dispersion phi = sum(r^2) / (N - p).
       d. Estimate correlation parameters from residuals.
       e. Form working covariance W_i for each cluster.
       f. Update beta: beta += [sum D_i' W_i^{-1} D_i]^{-1}
                                [sum D_i' W_i^{-1} (y_i - mu_i)]
       g. Check convergence: max|beta_new - beta| / (max|beta| + 0.1) < tol.

    Parameters
    ----------
    y : NDArray
        Response vector (n,).
    X : NDArray
        Design matrix (n, p).
    cluster_ids : NDArray
        Cluster identifiers (n,).
    family : Family
        GLM family (provides link, variance, initialize).
    corr : CorrStructure
        Working correlation structure (mutated during fitting).
    tol : float
        Convergence tolerance for relative coefficient change.
    max_iter : int
        Maximum number of GEE iterations.
    scale_fix : float | None
        If not None, fix dispersion at this value.

    Returns
    -------
    tuple
        (coefficients, fitted_values, residuals, scale, n_iter, converged)
        where fitted_values and residuals are in original observation order.
    """
    n, p = X.shape
    link = family.link

    # Group by cluster
    y_groups, X_groups, idx_groups = _group_by_cluster(y, X, cluster_ids)
    n_clusters = len(y_groups)

    # Initialize beta
    beta = _initialize_beta(y, X, family)
    phi = 1.0

    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        # Compute per-cluster quantities and accumulate score equation
        pearson_all: list[NDArray] = []
        B = np.zeros((p, p))  # sum of D_i' W_i^{-1} D_i
        U = np.zeros(p)  # sum of D_i' W_i^{-1} (y_i - mu_i)
        sum_r2 = 0.0

        for y_i, X_i, _ in zip(y_groups, X_groups, idx_groups):
            mu_i, D_i, W_i_inv, resid_i, pearson_i = _compute_cluster_quantities(
                y_i, X_i, beta, family, corr, phi
            )
            pearson_all.append(pearson_i)
            sum_r2 += float(np.sum(pearson_i**2))

            # Accumulate B and U
            DtWinv = D_i.T @ W_i_inv
            B += DtWinv @ D_i
            U += DtWinv @ resid_i

        # Estimate dispersion
        if scale_fix is not None:
            phi = scale_fix
        else:
            phi = max(sum_r2 / (n - p), 1e-10)

        # Estimate correlation parameters
        corr.estimate(pearson_all, phi, p)

        # Re-compute with updated correlation and phi
        B = np.zeros((p, p))
        U = np.zeros(p)
        for y_i, X_i, _ in zip(y_groups, X_groups, idx_groups):
            mu_i, D_i, W_i_inv, resid_i, pearson_i = _compute_cluster_quantities(
                y_i, X_i, beta, family, corr, phi
            )
            DtWinv = D_i.T @ W_i_inv
            B += DtWinv @ D_i
            U += DtWinv @ resid_i

        # Update beta: beta_new = beta + B^{-1} U
        try:
            delta = np.linalg.solve(B, U)
        except np.linalg.LinAlgError:
            # Singular B matrix -- add small ridge
            delta = np.linalg.solve(B + np.eye(p) * 1e-6, U)

        beta_new = beta + delta
        n_iter = iteration + 1

        # Check convergence
        rel_change = np.max(np.abs(delta)) / (np.max(np.abs(beta)) + 0.1)
        if rel_change < tol:
            converged = True
            beta = beta_new
            break

        beta = beta_new

    # Final fitted values and residuals in original order
    fitted = np.empty(n)
    residuals = np.empty(n)
    for y_i, X_i, idx_i in zip(y_groups, X_groups, idx_groups):
        eta_i = X_i @ beta
        mu_i = link.linkinv(eta_i)
        var_i = family.variance(mu_i)
        fitted[idx_i] = mu_i
        residuals[idx_i] = (y_i - mu_i) / np.maximum(np.sqrt(var_i), 1e-10)

    # Final dispersion
    if scale_fix is not None:
        phi = scale_fix
    else:
        phi = max(float(np.sum(residuals**2)) / (n - p), 1e-10)

    return beta, fitted, residuals, phi, n_iter, converged
