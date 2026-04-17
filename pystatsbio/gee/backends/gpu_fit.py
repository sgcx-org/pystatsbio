"""GPU backend for Generalized Estimating Equations (GEE).

The CPU path in ``_estimating_equations.py`` iterates over clusters in a
Python loop, solving an (m_i, m_i) working-covariance system per cluster
per iteration. For longitudinal datasets with hundreds of clusters this
Python-level overhead plus per-cluster BLAS call dispatch dominates the
wall time even though each individual inversion is tiny.

On GPU the clusters are independent: we can group them by cluster size
and batch their working-covariance solves into a single
``torch.linalg.solve`` call on a ``(K_s, s, s)`` tensor, then reduce the
per-cluster ``D_i' W_i^{-1} D_i`` and ``D_i' W_i^{-1} e_i`` contributions
into the (p, p) bread and (p,) score. Sandwich meat accumulates in the
same pass.

Two-tier validation (matching the rest of the library):
    CPU is validated against R geepack::geeglm().
    GPU is validated against CPU at the ``GPU_FP32`` tier (rtol = 1e-4,
    atol = 1e-5). FP64 on CUDA matches CPU to machine precision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatistics.regression.families import Family

from pystatsbio.gee._correlation import CorrStructure
from pystatsbio.gee.backends._gpu_correlation import GPUCorrelation
from pystatsbio.gee.backends._gpu_family import GPUFamilyOps, resolve_gpu_family


def _group_clusters(
    cluster_ids: NDArray,
) -> tuple[NDArray, dict[int, NDArray]]:
    """Sort rows by cluster id and group row indices by cluster size.

    Returns
    -------
    order : NDArray
        Permutation taking the original row order to the sorted order.
    size_groups : dict[int, NDArray]
        Maps cluster size ``s`` to a ``(K_s, s)`` int64 array of sorted
        row indices — row i of that array is the list of sorted-space
        rows belonging to the i-th cluster of that size.
    """
    order = np.argsort(cluster_ids, kind="stable")
    sorted_ids = cluster_ids[order]
    breaks = np.where(sorted_ids[:-1] != sorted_ids[1:])[0] + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(cluster_ids)]))
    sizes = ends - starts

    size_groups: dict[int, NDArray] = {}
    for s in np.unique(sizes):
        s_int = int(s)
        cl_idx = np.where(sizes == s_int)[0]
        row_starts = starts[cl_idx]
        row_idx = row_starts[:, None] + np.arange(s_int)[None, :]
        size_groups[s_int] = row_idx.astype(np.int64)

    return order, size_groups


def _initialize_beta_gpu(
    y_gpu: Any,
    X_gpu: Any,
    fam_ops: GPUFamilyOps,
    p: int,
    max_irls: int = 25,
) -> Any:
    """Independence-IRLS initialization on GPU.

    Mirrors the CPU ``_initialize_beta`` in ``_estimating_equations.py``
    but stays on-device so a tensor-entry fit does not have to pull X
    back across the PCIe bus just to compute starting values.
    """
    import torch

    mu = fam_ops.mu_from_y(y_gpu)
    eta = fam_ops.link_fn(mu)
    beta_new = None

    for _ in range(max_irls):
        mu_eta = fam_ops.mu_eta(eta)
        var_mu = fam_ops.variance(mu)
        w = (mu_eta * mu_eta) / torch.clamp(var_mu, min=1e-10)
        z = eta + (y_gpu - mu) / torch.clamp(mu_eta, min=1e-10)
        w_sqrt = torch.sqrt(torch.clamp(w, min=1e-10))
        Xw = X_gpu * w_sqrt.unsqueeze(1)
        zw = z * w_sqrt
        # Weighted least squares via lstsq (batched on GPU).
        sol = torch.linalg.lstsq(Xw, zw.unsqueeze(1))
        beta_new = sol.solution.squeeze(1)

        eta_new = X_gpu @ beta_new
        mu_new = fam_ops.linkinv(eta_new)
        change = (
            torch.max(torch.abs(eta_new - eta))
            / (torch.max(torch.abs(eta)) + 0.1)
        )
        eta = eta_new
        mu = mu_new
        if float(change.detach().to("cpu").item()) < 1e-8:
            break

    if beta_new is None:
        beta_new = torch.zeros(p, device=X_gpu.device, dtype=X_gpu.dtype)
    return beta_new


def _accumulate_gee_step(
    X_gpu: Any,
    y_gpu: Any,
    beta: Any,
    fam_ops: GPUFamilyOps,
    gpu_corr: GPUCorrelation,
    size_group_tensors: dict[int, Any],
    phi_gpu: Any,
    p: int,
    want_meat: bool,
) -> tuple[Any, Any, Any, dict[int, Any], Any]:
    """One GEE sweep over all size groups.

    Returns ``(B, U, M, pearson_groups, phi_new_numerator_sumr2)``.

    ``B`` is the ``p x p`` bread contribution, ``U`` the ``p`` score,
    ``M`` the ``p x p`` sandwich meat (or a zeros tensor when
    ``want_meat=False``), and ``pearson_groups`` the per-size Pearson
    residual batches used by the correlation estimator. The returned
    ``sum_r2`` (scalar tensor) is phi's numerator: sum of squared
    Pearson residuals over all observations.
    """
    import torch

    dtype = X_gpu.dtype
    device = X_gpu.device

    eta = X_gpu @ beta
    mu = fam_ops.linkinv(eta)
    var_mu = fam_ops.variance(mu)
    mu_eta_all = fam_ops.mu_eta(eta)
    A_sqrt_all = torch.sqrt(torch.clamp(var_mu, min=1e-10))
    resid_all = y_gpu - mu
    pearson_all = resid_all / torch.clamp(A_sqrt_all, min=1e-10)
    D_all = X_gpu * mu_eta_all.unsqueeze(1)

    B = torch.zeros((p, p), device=device, dtype=dtype)
    U = torch.zeros((p,), device=device, dtype=dtype)
    M = torch.zeros((p, p), device=device, dtype=dtype)
    sum_r2 = torch.zeros((), device=device, dtype=dtype)
    pearson_groups: dict[int, Any] = {}

    for s, idx_tensor in size_group_tensors.items():
        D_batch = D_all[idx_tensor]              # (K_s, s, p)
        A_sqrt_batch = A_sqrt_all[idx_tensor]    # (K_s, s)
        resid_batch = resid_all[idx_tensor]      # (K_s, s)
        pearson_batch = pearson_all[idx_tensor]  # (K_s, s)
        pearson_groups[s] = pearson_batch
        sum_r2 = sum_r2 + (pearson_batch * pearson_batch).sum()

        R_s = gpu_corr.build_R(s)  # (s, s)
        # W = phi * diag(A_sqrt) R diag(A_sqrt), all batched.
        W = (
            phi_gpu
            * A_sqrt_batch.unsqueeze(-1)
            * R_s.unsqueeze(0)
            * A_sqrt_batch.unsqueeze(-2)
        )
        # Tiny ridge to regularize against near-singular W (matches the
        # CPU path's numerical stabilizer in _compute_cluster_quantities).
        eye_s = torch.eye(s, device=device, dtype=dtype) * 1e-10
        W = W + eye_s

        # Solve W @ Z = [D_batch | resid_batch.unsqueeze(-1)] in one call.
        rhs = torch.cat([D_batch, resid_batch.unsqueeze(-1)], dim=-1)
        sol = torch.linalg.solve(W, rhs)        # (K_s, s, p + 1)
        Z = sol[..., :p]                         # (K_s, s, p) = W^{-1} D
        v = sol[..., p]                          # (K_s, s)    = W^{-1} r

        Dt = D_batch.transpose(-2, -1)           # (K_s, p, s)
        B_batch = Dt @ Z                         # (K_s, p, p)
        U_each = (Dt @ v.unsqueeze(-1)).squeeze(-1)  # (K_s, p)
        B = B + B_batch.sum(dim=0)
        U = U + U_each.sum(dim=0)
        if want_meat:
            M = M + (U_each.unsqueeze(-1) * U_each.unsqueeze(-2)).sum(dim=0)

    return B, U, M, pearson_groups, sum_r2


def _solve_update(B: Any, U: Any) -> Any:
    """Solve B @ delta = U on GPU with a small ridge fallback."""
    import torch
    try:
        return torch.linalg.solve(B, U)
    except RuntimeError:
        p = B.shape[0]
        ridge = torch.eye(p, device=B.device, dtype=B.dtype) * 1e-6
        return torch.linalg.solve(B + ridge, U)


def fit_gee_gpu(
    y: NDArray,
    X: NDArray | Any,                  # numpy array or torch.Tensor
    cluster_ids: NDArray,
    family: Family,
    corr: CorrStructure,
    tol: float,
    max_iter: int,
    scale_fix: float | None,
    device: str,
    use_fp64: bool,
) -> tuple[NDArray, NDArray, NDArray, float, int, bool, NDArray, NDArray]:
    """Fit a GEE model on GPU.

    Parameters
    ----------
    y : NDArray
        Response vector (n,).
    X : NDArray or torch.Tensor
        Design matrix (n, p). When a tensor already on ``device``, it is
        used in place and no H2D transfer happens on this call — the
        amortized DataSource path.
    cluster_ids : NDArray
        Cluster identifiers (n,). Numpy is fine — used only to build the
        per-size index groupings, a one-shot build of small integer
        tensors.
    family : Family
        Parsed GLM family. Only the ``name`` attribute is consulted on
        GPU (we dispatch to :mod:`_gpu_family`); the CPU ``Family`` ops
        are not called.
    corr : CorrStructure
        Parsed working correlation structure. Wrapped in
        :class:`GPUCorrelation` and mutated in place: at the end of the
        fit the estimated parameters are synced back so the CPU
        ``.params`` property is accurate.
    tol, max_iter, scale_fix :
        Standard GEE convergence controls (same semantics as CPU path).
    device : str
        'cuda' or 'mps'.
    use_fp64 : bool
        FP64 on CUDA yes, MPS raises (MPS has no FP64).

    Returns
    -------
    (beta, fitted, residuals, phi, n_iter, converged, naive_vcov,
     robust_vcov), all numpy arrays / scalars in original row order.
    """
    import torch

    if device == "mps" and use_fp64:
        raise RuntimeError(
            "GPU GEE: MPS does not support FP64. Use use_fp64=False "
            "or backend='cpu'."
        )

    torch_device = torch.device(device)
    dtype = torch.float64 if use_fp64 else torch.float32

    n = len(y)
    if isinstance(X, torch.Tensor):
        p = X.shape[1]
    else:
        p = X.shape[1]

    # Group clusters by size — numpy, one-shot.
    order, size_groups_np = _group_clusters(cluster_ids)

    # Move X and y to GPU in sorted order. If X is already a device
    # tensor, take a device-resident index (zero H2D for the big matrix)
    # and reuse the existing buffer.
    order_gpu = torch.as_tensor(order, device=torch_device, dtype=torch.long)
    if isinstance(X, torch.Tensor):
        X_same = X.device == torch_device and X.dtype == dtype
        X_base = X if X_same else X.to(device=torch_device, dtype=dtype)
        X_gpu = X_base.index_select(0, order_gpu)
    else:
        X_host = torch.as_tensor(X, device=torch_device, dtype=dtype)
        X_gpu = X_host.index_select(0, order_gpu)
    y_gpu = torch.as_tensor(y, device=torch_device, dtype=dtype).index_select(
        0, order_gpu,
    )

    # Per-size index tensors point into the *sorted* row order.
    size_group_tensors: dict[int, Any] = {
        s: torch.as_tensor(idx, device=torch_device, dtype=torch.long)
        for s, idx in size_groups_np.items()
    }

    fam_ops = resolve_gpu_family(family.name)
    gpu_corr = GPUCorrelation(corr, device=torch_device, dtype=dtype)

    beta = _initialize_beta_gpu(y_gpu, X_gpu, fam_ops, p)
    phi_gpu = torch.ones((), device=torch_device, dtype=dtype)

    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        # First sweep — compute phi, pearson residuals, update correlation.
        B, U, M, pearson_groups, sum_r2 = _accumulate_gee_step(
            X_gpu, y_gpu, beta, fam_ops, gpu_corr, size_group_tensors,
            phi_gpu, p, want_meat=False,
        )

        if scale_fix is not None:
            phi_gpu = torch.full(
                (), float(scale_fix), device=torch_device, dtype=dtype,
            )
        else:
            phi_gpu = torch.clamp(sum_r2 / (n - p), min=1e-10)

        gpu_corr.estimate(pearson_groups, phi_gpu, p)

        # Second sweep with updated phi and correlation. Matches the CPU
        # path's "re-compute with updated correlation and phi" step.
        B, U, _, _, _ = _accumulate_gee_step(
            X_gpu, y_gpu, beta, fam_ops, gpu_corr, size_group_tensors,
            phi_gpu, p, want_meat=False,
        )

        delta = _solve_update(B, U)
        beta_new = beta + delta
        n_iter = iteration + 1

        rel_change = (
            torch.max(torch.abs(delta))
            / (torch.max(torch.abs(beta)) + 0.1)
        )
        # Single scalar D2H per iteration to drive convergence logic.
        if float(rel_change.detach().to("cpu").item()) < tol:
            converged = True
            beta = beta_new
            break
        beta = beta_new

    # Final sandwich pass: compute B with meat on the converged beta.
    B, U, M, _, sum_r2_final = _accumulate_gee_step(
        X_gpu, y_gpu, beta, fam_ops, gpu_corr, size_group_tensors,
        phi_gpu, p, want_meat=True,
    )

    try:
        B_inv = torch.linalg.inv(B)
    except RuntimeError:
        ridge = torch.eye(p, device=torch_device, dtype=dtype) * 1e-6
        B_inv = torch.linalg.inv(B + ridge)
    naive_vcov_gpu = B_inv
    robust_vcov_gpu = B_inv @ M @ B_inv

    # Final fitted and Pearson residuals (return in original row order).
    eta_final = X_gpu @ beta
    mu_final = fam_ops.linkinv(eta_final)
    var_final = fam_ops.variance(mu_final)
    pearson_final = (y_gpu - mu_final) / torch.clamp(
        torch.sqrt(torch.clamp(var_final, min=1e-10)), min=1e-10,
    )

    # Invert the sort so we return fitted/residuals in original order.
    inv_order = torch.empty_like(order_gpu)
    inv_order[order_gpu] = torch.arange(
        n, device=torch_device, dtype=torch.long,
    )
    fitted = mu_final.index_select(0, inv_order).to(torch.float64).cpu().numpy()
    residuals = pearson_final.index_select(0, inv_order).to(
        torch.float64,
    ).cpu().numpy()
    beta_np = beta.to(torch.float64).cpu().numpy()
    naive_vcov = naive_vcov_gpu.to(torch.float64).cpu().numpy()
    robust_vcov = robust_vcov_gpu.to(torch.float64).cpu().numpy()

    if scale_fix is not None:
        phi = float(scale_fix)
    else:
        phi = max(
            float(sum_r2_final.to(torch.float64).cpu().item()) / (n - p),
            1e-10,
        )

    # Copy GPU-estimated correlation params into the CPU CorrStructure
    # so GEEResult.correlation_params reflects the fit.
    gpu_corr.sync_to_cpu()

    return (
        beta_np, fitted, residuals, phi, n_iter, converged,
        naive_vcov, robust_vcov,
    )
