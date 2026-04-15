"""Batch AUC computation for high-throughput biomarker panels.

Computes AUC (Mann-Whitney U / (n1*n0)) and DeLong standard errors for
many biomarker candidates simultaneously.  The CPU path uses
``scipy.stats.rankdata`` column-wise.  The GPU path uses PyTorch's
batched ``argsort`` for ranking and a vectorised masked sum.

GPU is beneficial when ``n_markers > 100``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class BatchAUCResult:
    """Result of batch AUC computation across multiple biomarkers."""

    auc: NDArray[np.floating]  # shape (n_markers,)
    se: NDArray[np.floating]  # DeLong SE for each
    n_markers: int


# ---------------------------------------------------------------------------
# CPU path
# ---------------------------------------------------------------------------

def _batch_auc_cpu(
    response: NDArray, predictors: NDArray,
) -> BatchAUCResult:
    """Column-wise AUC + DeLong SE on CPU."""
    from scipy import stats as sp_stats

    N, M = predictors.shape
    case_mask = response == 1
    n1 = int(case_mask.sum())
    n0 = N - n1

    auc_arr = np.empty(M)
    se_arr = np.empty(M)

    for m in range(M):
        col = predictors[:, m]

        # Pooled ranks (midranks for ties)
        pooled_ranks = sp_stats.rankdata(col, method="average")

        # AUC via Mann-Whitney
        sum_case_ranks = pooled_ranks[case_mask].sum()
        auc_m = (sum_case_ranks - n1 * (n1 + 1) / 2) / (n1 * n0)

        # DeLong placement values
        case_ranks_within = sp_stats.rankdata(col[case_mask], method="average")
        ctrl_ranks_within = sp_stats.rankdata(col[~case_mask], method="average")

        V10 = (pooled_ranks[case_mask] - case_ranks_within) / n0
        V01 = 1.0 - (pooled_ranks[~case_mask] - ctrl_ranks_within) / n1

        S10 = np.var(V10, ddof=1) if n1 > 1 else 0.0
        S01 = np.var(V01, ddof=1) if n0 > 1 else 0.0
        var_auc = S10 / n1 + S01 / n0

        auc_arr[m] = auc_m
        se_arr[m] = np.sqrt(var_auc)

    return BatchAUCResult(auc=auc_arr, se=se_arr, n_markers=M)


# ---------------------------------------------------------------------------
# GPU path
# ---------------------------------------------------------------------------

def _batch_auc_gpu(
    response: NDArray, predictors: NDArray,
) -> BatchAUCResult:
    """Batched AUC + DeLong SE on GPU via PyTorch.

    Uses batched argsort for column-wise ranking, then a single
    matrix-vector product for the Mann-Whitney sum across all markers.
    """
    import torch

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # MPS uses float32, others float64
    dtype = torch.float32 if device.type == "mps" else torch.float64

    N, M = predictors.shape
    case_mask_np = response == 1
    n1 = int(case_mask_np.sum())
    n0 = N - n1

    pred_t = torch.from_numpy(predictors).to(device=device, dtype=dtype)

    # Column-wise ranking via argsort-of-argsort (handles ties with average)
    # For GPU efficiency, use the argsort approach:
    # rank[i] = position of element i in sorted order
    # For ties, we need midranks which requires more work.
    # Use argsort twice: argsort gives sorted indices, inverting gives ranks.
    sorted_indices = pred_t.argsort(dim=0)           # (N, M)
    ranks = torch.empty_like(pred_t)
    base_ranks = torch.arange(1, N + 1, device=device, dtype=dtype).unsqueeze(1).expand(N, M)
    ranks.scatter_(0, sorted_indices, base_ranks)

    # Midrank correction for ties
    # Group equal values and assign their mean rank (per-column loop)
    sorted_vals = pred_t.gather(0, sorted_indices)  # (N, M) sorted
    sorted_ranks = (
        torch.arange(1, N + 1, device=device, dtype=dtype).unsqueeze(1).expand(N, M).clone()
    )

    # Group boundaries in sorted order
    for m_idx in range(M):
        col_sorted = sorted_vals[:, m_idx]
        col_ranks = sorted_ranks[:, m_idx]
        # Find runs of equal values
        i = 0
        while i < N:
            j = i + 1
            while j < N and col_sorted[j] == col_sorted[i]:
                j += 1
            if j > i + 1:
                # Tied block [i, j): assign midrank
                midrank = (i + 1 + j) / 2.0
                col_ranks[i:j] = midrank
            i = j

    # Scatter midranks back to original positions
    pooled_ranks_gpu = torch.empty_like(pred_t)
    pooled_ranks_gpu.scatter_(0, sorted_indices, sorted_ranks)

    # AUC: sum of case ranks - n1*(n1+1)/2, divided by n1*n0
    case_mask_t = torch.from_numpy(case_mask_np).to(device=device)
    case_ranks_pooled = pooled_ranks_gpu[case_mask_t]  # (n1, M)
    sum_case_ranks = case_ranks_pooled.sum(dim=0)  # (M,)
    auc_t = (sum_case_ranks - n1 * (n1 + 1) / 2) / (n1 * n0)

    # DeLong SE: need within-group ranks for cases and controls separately
    # Rank cases among cases only
    case_pred = pred_t[case_mask_t]      # (n1, M)
    ctrl_pred = pred_t[~case_mask_t]     # (n0, M)

    case_within_ranks = _midranks_gpu(case_pred, device, dtype)  # (n1, M)
    ctrl_within_ranks = _midranks_gpu(ctrl_pred, device, dtype)  # (n0, M)

    # Placement values
    V10 = (pooled_ranks_gpu[case_mask_t] - case_within_ranks) / n0  # (n1, M)
    V01 = 1.0 - (pooled_ranks_gpu[~case_mask_t] - ctrl_within_ranks) / n1  # (n0, M)

    # Variance per marker
    auc_expanded = auc_t.unsqueeze(0)  # (1, M)
    if n1 > 1:
        S10 = ((V10 - auc_expanded) ** 2).sum(dim=0) / (n1 - 1)  # (M,)
    else:
        S10 = torch.zeros(M, device=device, dtype=dtype)
    if n0 > 1:
        S01 = ((V01 - auc_expanded) ** 2).sum(dim=0) / (n0 - 1)  # (M,)
    else:
        S01 = torch.zeros(M, device=device, dtype=dtype)

    var_auc = S10 / n1 + S01 / n0
    se_t = torch.sqrt(var_auc)

    return BatchAUCResult(
        auc=auc_t.cpu().numpy().astype(np.float64),
        se=se_t.cpu().numpy().astype(np.float64),
        n_markers=M,
    )


def _midranks_gpu(
    data: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute midranks column-wise for a (N, M) tensor on GPU."""
    import torch

    N, M = data.shape
    sorted_indices = data.argsort(dim=0)
    sorted_vals = data.gather(0, sorted_indices)
    ranks = torch.arange(1, N + 1, device=device, dtype=dtype).unsqueeze(1).expand(N, M).clone()

    # Fix ties to midranks
    for m_idx in range(M):
        col_sorted = sorted_vals[:, m_idx]
        col_ranks = ranks[:, m_idx]
        i = 0
        while i < N:
            j = i + 1
            while j < N and col_sorted[j] == col_sorted[i]:
                j += 1
            if j > i + 1:
                col_ranks[i:j] = (i + 1 + j) / 2.0
            i = j

    result = torch.empty_like(data)
    result.scatter_(0, sorted_indices, ranks)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def batch_auc(
    response: NDArray[np.integer],
    predictors: NDArray[np.floating],
    *,
    backend: str = "auto",
) -> BatchAUCResult:
    """Compute AUC for many biomarker candidates simultaneously.

    Parameters
    ----------
    response : array of int, shape ``(n_samples,)``
        Shared binary outcome (0/1).
    predictors : array of float, shape ``(n_samples, n_markers)``
        Matrix of biomarker values (one column per candidate marker).
    backend : str
        ``'cpu'``, ``'gpu'``, or ``'auto'``.

    Returns
    -------
    BatchAUCResult

    Notes
    -----
    GPU backend is beneficial when ``n_markers > 100``.  Uses rank-based
    AUC computation which is embarrassingly parallel across markers.
    DeLong standard errors are computed for each marker.
    """
    response = np.asarray(response, dtype=np.intp)
    predictors = np.asarray(predictors, dtype=np.float64)

    if response.ndim != 1:
        raise ValueError(f"response must be 1-D, got shape {response.shape}")
    if predictors.ndim != 2:
        raise ValueError(
            f"predictors must be 2-D (n_samples, n_markers), got shape {predictors.shape}"
        )
    if response.shape[0] != predictors.shape[0]:
        raise ValueError(
            f"response length {response.shape[0]} != predictors rows {predictors.shape[0]}"
        )

    unique_labels = np.unique(response)
    if not np.array_equal(unique_labels, np.array([0, 1])):
        raise ValueError(
            f"response must be binary (0/1), got unique values {unique_labels}"
        )

    if not np.all(np.isfinite(predictors)):
        raise ValueError(
            "predictors contains NaN or infinite values. "
            "Remove or impute missing values before calling batch_auc."
        )

    if backend == "cpu":
        return _batch_auc_cpu(response, predictors)

    if backend == "gpu":
        return _batch_auc_gpu(response, predictors)

    # auto — try GPU, fall back to CPU
    try:
        import torch

        has_gpu = torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if has_gpu:
            return _batch_auc_gpu(response, predictors)
    except ImportError:
        pass

    return _batch_auc_cpu(response, predictors)
