"""Batch dose-response fitting for high-throughput screening (HTS).

This is the primary GPU showcase: fit thousands of 4PL curves simultaneously.
Each compound's curve fit is independent — perfect for GPU batching.

**CPU path**: loops over compounds calling :func:`fit_drm` for each.

**GPU path**: batched Levenberg-Marquardt in PyTorch.  All K compounds are
fit simultaneously using vectorised forward passes, finite-difference
Jacobians, and batched linear solves.  EC50 is parameterised on the log
scale so the optimisation is unconstrained.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pystatsbio.doseresponse._common import BatchDoseResponseResult

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

def _batch_cpu(
    dose_matrix: NDArray,
    response_matrix: NDArray,
    model: str,
    max_iter: int,
    tol: float,
) -> BatchDoseResponseResult:
    """Fit each compound sequentially on CPU via :func:`fit_drm`."""
    from pystatsbio.doseresponse._fit import fit_drm

    K = dose_matrix.shape[0]
    ec50 = np.empty(K)
    hill = np.empty(K)
    top = np.empty(K)
    bottom = np.empty(K)
    converged = np.empty(K, dtype=bool)
    rss = np.empty(K)

    for i in range(K):
        try:
            result = fit_drm(dose_matrix[i], response_matrix[i], model=model)
            ec50[i] = result.params.ec50
            hill[i] = result.params.hill
            top[i] = result.params.top
            bottom[i] = result.params.bottom
            converged[i] = result.converged
            rss[i] = result.rss
        except (ValueError, RuntimeError):
            # Fitting failed for this compound (bad data or no convergence).
            # Mark as unconverged with NaN parameters; do not swallow
            # programming errors (TypeError, AttributeError, etc.).
            ec50[i] = hill[i] = top[i] = bottom[i] = rss[i] = np.nan
            converged[i] = False

    return BatchDoseResponseResult(
        ec50=ec50, hill=hill, top=top, bottom=bottom,
        converged=converged, rss=rss, n_compounds=K,
    )


# ---------------------------------------------------------------------------
# GPU batched Levenberg-Marquardt
# ---------------------------------------------------------------------------

def _batch_gpu(
    dose_matrix: NDArray,
    response_matrix: NDArray,
    max_iter: int,
    tol: float,
) -> BatchDoseResponseResult:
    """Batched Levenberg-Marquardt for LL.4 on GPU (CUDA / MPS / CPU-torch).

    Parameters are ``[bottom, top, log_ec50, hill]`` — log-scale EC50
    keeps the optimisation unconstrained.
    """
    import torch

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # MPS (Apple Silicon) does not support float64 — use float32 there
    dtype = torch.float32 if device.type == "mps" else torch.float64

    # Adapt numerical constants for precision of chosen dtype
    is_f32 = (dtype == torch.float32)
    eps_fd = 1e-3 if is_f32 else 1e-5        # finite-difference step
    tol_eff = max(tol, 1e-5) if is_f32 else tol  # convergence tolerance
    diag_clamp = 1e-6 if is_f32 else 1e-12   # minimum diagonal damping
    lam_lo = 1e-10 if is_f32 else 1e-15      # lambda lower bound
    lam_hi = 1e10 if is_f32 else 1e15        # lambda upper bound

    K, N = dose_matrix.shape

    dose_t = torch.from_numpy(dose_matrix).to(device=device, dtype=dtype)
    resp_t = torch.from_numpy(response_matrix).to(device=device, dtype=dtype)

    # Pre-compute log-dose (handle dose=0 → -inf)
    log_dose = torch.where(
        dose_t > 0,
        torch.log(dose_t),
        torch.tensor(float("-inf"), device=device, dtype=dtype),
    )

    # ---- vectorised self-start -----------------------------------------
    theta = _batch_init_gpu(dose_t, resp_t, log_dose, device, dtype)  # (K, 4)

    lam = torch.full((K,), 1.0, device=device, dtype=dtype)  # start conservatively
    conv = torch.zeros(K, dtype=torch.bool, device=device)

    def _ll4_fwd(ld: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        """LL.4 forward pass for all K compounds.

        th[:, 0]=bottom, th[:, 1]=top, th[:, 2]=log_ec50, th[:, 3]=hill
        """
        b = th[:, 0].unsqueeze(1)
        t = th[:, 1].unsqueeze(1)
        le = th[:, 2].unsqueeze(1)
        h = th[:, 3].unsqueeze(1)
        exponent = -h * (ld - le)
        return b + (t - b) / (1.0 + torch.exp(exponent))

    for _ in range(max_iter):
        pred = _ll4_fwd(log_dose, theta)
        r = resp_t - pred
        rss_old = (r**2).sum(dim=1)

        # Batched model Jacobian via finite differences: J[k, n, p] = ∂f/∂θ_p
        J = torch.zeros(K, N, 4, device=device, dtype=dtype)
        for p in range(4):
            th_p = theta.clone()
            th_p[:, p] += eps_fd
            pred_p = _ll4_fwd(log_dose, th_p)
            J[:, :, p] = (pred_p - pred) / eps_fd

        Jt = J.transpose(1, 2)            # (K, 4, N)
        JtJ = Jt @ J                       # (K, 4, 4)
        Jtr = (Jt @ r.unsqueeze(2))        # (K, 4, 1)

        # LM damping: A = JtJ + λ * (diag(JtJ) + μ*I)
        # The identity term prevents zero damping when JtJ diagonal is tiny
        diag_JtJ = torch.diagonal(JtJ, dim1=-2, dim2=-1)  # (K, 4)
        diag_JtJ = torch.clamp(diag_JtJ, min=diag_clamp)
        # Add identity floor to ensure damping works even for near-zero columns
        mu = diag_JtJ.mean(dim=1, keepdim=True).clamp(min=1.0)  # (K, 1)
        damping = lam.unsqueeze(1) * (diag_JtJ + mu)
        A = JtJ + torch.diag_embed(damping)

        # Solve for step
        try:
            delta = torch.linalg.solve(A, Jtr).squeeze(2)  # (K, 4)
        except RuntimeError:
            lam *= 10.0
            continue

        # Trial step
        theta_new = theta + delta
        pred_new = _ll4_fwd(log_dose, theta_new)
        rss_new = ((resp_t - pred_new) ** 2).sum(dim=1)

        # Accept / reject per compound
        improved = rss_new < rss_old
        active = ~conv
        accept = improved & active

        theta = torch.where(accept.unsqueeze(1), theta_new, theta)

        # Update damping
        lam = torch.where(improved, lam * 0.1, lam * 10.0)
        lam = torch.clamp(lam, lam_lo, lam_hi)

        # Convergence
        rel_change = torch.abs(rss_new - rss_old) / (rss_old + diag_clamp)
        newly = (rel_change < tol_eff) & improved
        conv = conv | newly

        if conv.all():
            break

    # Final RSS
    pred_final = _ll4_fwd(log_dose, theta)
    rss_final = ((resp_t - pred_final) ** 2).sum(dim=1)

    ec50_out = torch.exp(theta[:, 2])

    return BatchDoseResponseResult(
        ec50=ec50_out.cpu().numpy(),
        hill=theta[:, 3].cpu().numpy(),
        top=theta[:, 1].cpu().numpy(),
        bottom=theta[:, 0].cpu().numpy(),
        converged=conv.cpu().numpy(),
        rss=rss_final.cpu().numpy(),
        n_compounds=K,
    )


def _batch_init_gpu(
    dose_t: torch.Tensor,
    resp_t: torch.Tensor,
    log_dose: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Vectorised self-starting for all K compounds.

    Returns ``(K, 4)`` tensor: ``[bottom, top, log_ec50, hill]``.
    """
    import torch

    K, N = dose_t.shape

    # Sort responses by dose for each compound
    dose_order = dose_t.argsort(dim=1)
    resp_sorted = resp_t.gather(1, dose_order)

    n_edge = max(1, N // 4)
    low_resp = resp_sorted[:, :n_edge].mean(dim=1)
    high_resp = resp_sorted[:, -n_edge:].mean(dim=1)

    # Direction
    increasing = high_resp > low_resp
    bottom = torch.where(increasing, low_resp, high_resp)
    top = torch.where(increasing, high_resp, low_resp)

    # EC50 ≈ geometric mean of positive doses
    pos_mask = dose_t > 0
    # Replace zero/neg doses with 1 for log (won't affect mean much)
    dose_safe = torch.where(pos_mask, dose_t, torch.ones_like(dose_t))
    # Mean of log over positive doses per compound
    pos_float = pos_mask.to(dtype=dtype)
    n_pos = pos_float.sum(dim=1).clamp(min=1)
    log_ec50 = (torch.log(dose_safe) * pos_float).sum(dim=1) / n_pos

    # Hill: +1 for increasing, -1 for decreasing
    hill = torch.where(
        increasing,
        torch.ones(K, device=device, dtype=dtype),
        -torch.ones(K, device=device, dtype=dtype),
    )

    return torch.stack([bottom, top, log_ec50, hill], dim=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_drm_batch(
    dose_matrix: NDArray[np.floating],
    response_matrix: NDArray[np.floating],
    *,
    model: str = "LL.4",
    backend: str = "auto",
    max_iter: int = 100,
    tol: float = 1e-8,
) -> BatchDoseResponseResult:
    """Batch-fit dose-response curves across many compounds.

    Parameters
    ----------
    dose_matrix : array, shape ``(n_compounds, n_doses)``
        Dose values for each compound.
    response_matrix : array, shape ``(n_compounds, n_doses)``
        Response values for each compound.
    model : str
        Model name (currently only ``'LL.4'`` for batch fitting).
    backend : str
        ``'cpu'``, ``'gpu'``, or ``'auto'``.  GPU uses batched
        Levenberg-Marquardt via PyTorch for massive parallelism.
    max_iter : int
        Maximum LM iterations per compound (default 100).
    tol : float
        Convergence tolerance on relative RSS change (default 1e-8).

    Returns
    -------
    BatchDoseResponseResult

    Notes
    -----
    GPU backend requires ``pip install pystatsbio[gpu]`` (PyTorch).
    On CPU, curves are fit sequentially using ``scipy.optimize``.
    On GPU, all curves are fit simultaneously using batched Jacobian
    computation and batched normal equations.
    """
    dose_matrix = np.asarray(dose_matrix, dtype=np.float64)
    response_matrix = np.asarray(response_matrix, dtype=np.float64)

    if dose_matrix.ndim != 2:
        raise ValueError(
            f"dose_matrix must be 2-D (n_compounds, n_doses), got shape {dose_matrix.shape}"
        )
    if dose_matrix.shape != response_matrix.shape:
        raise ValueError(
            f"dose_matrix and response_matrix must have same shape, "
            f"got {dose_matrix.shape} and {response_matrix.shape}"
        )
    if model != "LL.4":
        raise ValueError(f"Batch fitting currently supports only 'LL.4', got {model!r}")

    if backend == "cpu":
        return _batch_cpu(dose_matrix, response_matrix, model, max_iter, tol)

    if backend == "gpu":
        return _batch_gpu(dose_matrix, response_matrix, max_iter, tol)

    # auto — try GPU, fall back to CPU
    try:
        import torch

        has_gpu = torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if has_gpu:
            return _batch_gpu(dose_matrix, response_matrix, max_iter, tol)
    except ImportError:
        pass

    return _batch_cpu(dose_matrix, response_matrix, model, max_iter, tol)
