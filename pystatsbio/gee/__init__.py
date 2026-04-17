"""GEE: Generalized Estimating Equations for clustered/correlated data.

Fits regression models to clustered data (repeated measures on subjects,
patients within clinics, etc.) using working correlation structures and
sandwich variance estimation for valid inference even under misspecification.

Supports four working correlation structures:
  - Independence: R = I (recovers GLM with sandwich SE)
  - Exchangeable: R_ij = alpha (compound symmetry)
  - AR(1): R_ij = alpha^|i-j| (autoregressive)
  - Unstructured: R_ij estimated freely (equal cluster sizes only)

Supports four GLM families via pystatistics:
  - Gaussian (identity link)
  - Binomial (logit link)
  - Poisson (log link)
  - Gamma (inverse link)

Validates against: R geepack::geeglm()
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from pystatistics.regression.families import Family, resolve_family

from pystatsbio.gee._common import GEEResult
from pystatsbio.gee._correlation import (
    AR1Corr,
    CorrStructure,
    ExchangeableCorr,
    IndependenceCorr,
    UnstructuredCorr,
    resolve_corr,
)
from pystatsbio.gee._estimating_equations import _fit_gee
from pystatsbio.gee._sandwich import sandwich_variance


def _validate_inputs(
    y: NDArray,
    X: NDArray,
    cluster_id: NDArray,
) -> None:
    """Validate GEE inputs.

    Parameters
    ----------
    y : NDArray
        Response vector.
    X : NDArray
        Design matrix.
    cluster_id : NDArray
        Cluster identifiers.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got {y.ndim}-D")
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got {X.ndim}-D")
    if cluster_id.ndim != 1:
        raise ValueError(f"cluster_id must be 1-D, got {cluster_id.ndim}-D")

    n = y.shape[0]
    if X.shape[0] != n:
        raise ValueError(
            f"y has {n} observations but X has {X.shape[0]} rows"
        )
    if cluster_id.shape[0] != n:
        raise ValueError(
            f"y has {n} observations but cluster_id has {cluster_id.shape[0]} elements"
        )

    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values (NaN or Inf)")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values (NaN or Inf)")

    unique_clusters = np.unique(cluster_id)
    if len(unique_clusters) < 2:
        raise ValueError(
            f"GEE requires at least 2 clusters, got {len(unique_clusters)}"
        )


def gee(
    y: ArrayLike,
    X: ArrayLike,
    cluster_id: ArrayLike,
    *,
    family: str = "gaussian",
    corr_structure: str = "exchangeable",
    names: list[str] | None = None,
    scale_fix: float | None = None,
    tol: float = 1e-6,
    max_iter: int = 50,
    backend: str | None = None,
    use_fp64: bool = False,
) -> GEEResult:
    """Fit a Generalized Estimating Equations (GEE) model.

    GEE extends GLMs to handle correlated/clustered data by specifying
    a working correlation structure and using sandwich variance estimation
    for valid inference even if the working correlation is wrong.

    Parameters
    ----------
    y : ArrayLike
        Response variable (1-D array, n observations).
    X : ArrayLike
        Design matrix (n x p). Should include an intercept column
        if an intercept is desired.
    cluster_id : ArrayLike
        Cluster/subject identifiers (1-D array, n observations).
        Observations with the same cluster_id are treated as correlated.
    family : str
        GLM family: 'gaussian', 'binomial', 'poisson', or 'gamma'.
    corr_structure : str
        Working correlation: 'independence', 'exchangeable', 'ar1',
        or 'unstructured'.
    names : list[str] | None
        Optional coefficient names (length p). If None, defaults to
        'x0', 'x1', ... in output.
    scale_fix : float | None
        If not None, fix the dispersion parameter at this value
        instead of estimating it.
    tol : float
        Convergence tolerance for relative coefficient change.
    max_iter : int
        Maximum GEE iterations.

    Returns
    -------
    GEEResult
        Frozen dataclass with coefficients, robust SE, correlation
        parameters, convergence diagnostics, and summary method.

    Raises
    ------
    ValueError
        If inputs fail validation (mismatched lengths, non-finite values,
        invalid family or correlation structure, fewer than 2 clusters).

    Examples
    --------
    >>> import numpy as np
    >>> from pystatsbio.gee import gee
    >>> n_clusters, cluster_size = 50, 5
    >>> n = n_clusters * cluster_size
    >>> cluster_id = np.repeat(np.arange(n_clusters), cluster_size)
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = X @ np.array([1.0, 0.5]) + np.random.randn(n)
    >>> result = gee(y, X, cluster_id, family='gaussian')
    >>> print(result.summary())
    """
    # Convention shared with the rest of pystatistics GPU-capable
    # entry points (pca, multinom): tensor input defaults to GPU,
    # numpy input defaults to CPU. See GPU_BACKEND_CONVENTION.md.
    import sys as _sys
    _is_X_tensor = (
        "torch" in _sys.modules
        and isinstance(X, _sys.modules["torch"].Tensor)
    )

    if _is_X_tensor:
        import torch
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.ndim}-D")
        if not torch.isfinite(X).all():
            raise ValueError("X contains non-finite values (NaN or Inf)")
        if backend is None:
            backend = "gpu" if X.device.type != "cpu" else "cpu"
        if backend == "cpu":
            raise ValueError(
                "backend='cpu' was specified but X is a torch.Tensor "
                f"on device {X.device}. Either pass a numpy array / "
                "CPU DataSource to the CPU backend, or call `.to('cpu')` "
                "on the DataSource explicitly to move it back."
            )
        y_arr = (
            y.detach().cpu().numpy().astype(np.float64)
            if isinstance(y, torch.Tensor)
            else np.asarray(y, dtype=np.float64)
        )
        cluster_arr = (
            cluster_id.detach().cpu().numpy()
            if isinstance(cluster_id, torch.Tensor)
            else np.asarray(cluster_id)
        )
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1-D, got {y_arr.ndim}-D")
        if cluster_arr.ndim != 1:
            raise ValueError(
                f"cluster_id must be 1-D, got {cluster_arr.ndim}-D"
            )
        n = y_arr.shape[0]
        if X.shape[0] != n:
            raise ValueError(
                f"y has {n} observations but X has {X.shape[0]} rows"
            )
        if cluster_arr.shape[0] != n:
            raise ValueError(
                f"y has {n} observations but cluster_id has "
                f"{cluster_arr.shape[0]} elements"
            )
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y contains non-finite values (NaN or Inf)")
        if len(np.unique(cluster_arr)) < 2:
            raise ValueError(
                "GEE requires at least 2 clusters, got "
                f"{len(np.unique(cluster_arr))}"
            )
        X_arr = None
        X_for_gpu = X
    else:
        if backend is None:
            backend = "cpu"
        y_arr = np.asarray(y, dtype=np.float64)
        X_arr = np.asarray(X, dtype=np.float64)
        cluster_arr = np.asarray(cluster_id)
        _validate_inputs(y_arr, X_arr, cluster_arr)
        X_for_gpu = None

    if backend not in ("cpu", "auto", "gpu"):
        raise ValueError(
            f"backend: must be 'cpu', 'auto', or 'gpu', got {backend!r}"
        )

    fam = resolve_family(family)
    corr = resolve_corr(corr_structure)

    p = X_arr.shape[1] if X_arr is not None else X_for_gpu.shape[1]
    if names is not None:
        if len(names) != p:
            raise ValueError(
                f"names has {len(names)} elements but X has {p} columns"
            )
        names_tuple: tuple[str, ...] | None = tuple(names)
    else:
        names_tuple = None

    naive_vcov = None
    robust_vcov = None
    if X_arr is None:
        from pystatsbio.gee.backends.gpu_fit import fit_gee_gpu
        gpu_device = X_for_gpu.device.type
        (beta, fitted, residuals, phi, n_iter, converged,
         naive_vcov, robust_vcov) = fit_gee_gpu(
            y_arr, X_for_gpu, cluster_arr, fam, corr, tol, max_iter,
            scale_fix, device=gpu_device, use_fp64=use_fp64,
        )
    elif backend == "cpu":
        beta, fitted, residuals, phi, n_iter, converged = _fit_gee(
            y_arr, X_arr, cluster_arr, fam, corr, tol, max_iter, scale_fix
        )
    else:
        from pystatistics.core.compute.device import select_device
        dev = select_device("gpu" if backend == "gpu" else "auto")
        if dev.is_gpu:
            from pystatsbio.gee.backends.gpu_fit import fit_gee_gpu
            (beta, fitted, residuals, phi, n_iter, converged,
             naive_vcov, robust_vcov) = fit_gee_gpu(
                y_arr, X_arr, cluster_arr, fam, corr, tol, max_iter,
                scale_fix, device=dev.device_type, use_fp64=use_fp64,
            )
        elif backend == "gpu":
            raise RuntimeError(
                "backend='gpu' requested but no GPU is available. "
                "Install PyTorch with CUDA/MPS support or use "
                "backend='cpu'."
            )
        else:
            beta, fitted, residuals, phi, n_iter, converged = _fit_gee(
                y_arr, X_arr, cluster_arr, fam, corr, tol, max_iter,
                scale_fix,
            )

    if naive_vcov is None:
        naive_vcov, robust_vcov = sandwich_variance(
            X_arr, y_arr, fitted, beta, cluster_arr, fam, corr, phi
        )

    # Standard errors
    naive_se = np.sqrt(np.maximum(np.diag(naive_vcov), 0.0))
    robust_se = np.sqrt(np.maximum(np.diag(robust_vcov), 0.0))

    # z-values and p-values (using robust SE)
    z_values = beta / np.maximum(robust_se, 1e-10)
    p_values = 2.0 * stats.norm.sf(np.abs(z_values))

    n_clusters = len(np.unique(cluster_arr))

    return GEEResult(
        coefficients=beta,
        naive_se=naive_se,
        robust_se=robust_se,
        naive_vcov=naive_vcov,
        robust_vcov=robust_vcov,
        z_values=z_values,
        p_values=p_values,
        fitted_values=fitted,
        residuals=residuals,
        correlation_type=corr.name,
        correlation_params=corr.params,
        scale=phi,
        n_clusters=n_clusters,
        n_obs=y_arr.shape[0],
        family_name=fam.name,
        link_name=fam.link.name,
        converged=converged,
        n_iter=n_iter,
        names=names_tuple,
    )


__all__ = [
    "GEEResult",
    "gee",
    "CorrStructure",
    "IndependenceCorr",
    "ExchangeableCorr",
    "AR1Corr",
    "UnstructuredCorr",
]
