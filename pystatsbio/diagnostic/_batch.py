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

    Fully vectorized: no Python loops over markers or samples.
    Uses batched argsort for ranking and vectorized tie detection
    via diff-based boundary detection with cumsum grouping.
    """
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dtype = torch.float32 if device.type == "mps" else torch.float64

    N, M = predictors.shape
    case_mask_np = response == 1
    n1 = int(case_mask_np.sum())
    n0 = N - n1

    pred_t = torch.from_numpy(predictors).to(device=device, dtype=dtype)

    # Pooled midranks — fully vectorized across all M columns
    pooled_ranks_gpu = _midranks_vectorized(pred_t, device, dtype)

    # AUC via Mann-Whitney
    case_mask_t = torch.from_numpy(case_mask_np).to(device=device)
    sum_case_ranks = pooled_ranks_gpu[case_mask_t].sum(dim=0)  # (M,)
    auc_t = (sum_case_ranks - n1 * (n1 + 1) / 2) / (n1 * n0)

    # DeLong SE: within-group ranks for placement values
    case_pred = pred_t[case_mask_t]      # (n1, M)
    ctrl_pred = pred_t[~case_mask_t]     # (n0, M)

    case_within_ranks = _midranks_vectorized(case_pred, device, dtype)
    ctrl_within_ranks = _midranks_vectorized(ctrl_pred, device, dtype)

    V10 = (pooled_ranks_gpu[case_mask_t] - case_within_ranks) / n0
    V01 = 1.0 - (pooled_ranks_gpu[~case_mask_t] - ctrl_within_ranks) / n1

    auc_expanded = auc_t.unsqueeze(0)
    S10 = ((V10 - auc_expanded) ** 2).sum(dim=0) / max(n1 - 1, 1)
    S01 = ((V01 - auc_expanded) ** 2).sum(dim=0) / max(n0 - 1, 1)

    se_t = torch.sqrt(S10 / n1 + S01 / n0)

    return BatchAUCResult(
        auc=auc_t.cpu().numpy().astype(np.float64),
        se=se_t.cpu().numpy().astype(np.float64),
        n_markers=M,
    )


def _midranks_vectorized(
    data: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute midranks column-wise for a (N, M) tensor — fully vectorized.

    No Python loops over columns or rows. Tie detection uses diff-based
    boundary detection: consecutive equal values in sorted order form a
    tie group. Each group's midrank = mean of its positional ranks.

    Algorithm:
        1. argsort each column
        2. Detect tie boundaries via diff == 0 across sorted values
        3. Assign group IDs via cumsum of boundary flags
        4. Compute group midranks via scatter-add and scatter-count
        5. Map midranks back to original positions
    """
    import torch

    N, M = data.shape

    sorted_indices = data.argsort(dim=0)                   # (N, M)
    sorted_vals = data.gather(0, sorted_indices)            # (N, M)

    # Base ranks: 1, 2, ..., N for each column
    base_ranks = torch.arange(
        1, N + 1, device=device, dtype=dtype,
    ).unsqueeze(1).expand(N, M)

    # Detect tie boundaries: where consecutive sorted values differ
    # diff[i] = 1 if sorted_vals[i] != sorted_vals[i-1], else 0
    # First element always starts a new group
    diffs = (sorted_vals[1:] != sorted_vals[:-1]).to(dtype)  # (N-1, M)
    boundaries = torch.cat([
        torch.ones(1, M, device=device, dtype=dtype),
        diffs,
    ], dim=0)  # (N, M)

    # Group IDs via cumsum: each new boundary increments the group ID
    group_ids = boundaries.cumsum(dim=0).long()  # (N, M)
    n_groups_per_col = group_ids[-1]              # (M,) — max group ID per column

    # For each group, compute sum of ranks and count of members
    # Then midrank = sum_of_ranks / count
    # Use a flat scatter approach: flatten (N, M) into (N*M,) with offset per column
    max_groups = int(n_groups_per_col.max().item())
    col_offsets = (
        torch.arange(M, device=device).unsqueeze(0) * (max_groups + 1)
    )  # (1, M)
    flat_group_ids = (group_ids + col_offsets).reshape(-1)   # (N*M,)
    flat_ranks = base_ranks.reshape(-1)                       # (N*M,)
    total_bins = M * (max_groups + 1)

    # Sum of ranks per group and count per group
    rank_sums = torch.zeros(total_bins, device=device, dtype=dtype)
    rank_sums.scatter_add_(0, flat_group_ids, flat_ranks)
    counts = torch.zeros(total_bins, device=device, dtype=dtype)
    counts.scatter_add_(0, flat_group_ids, torch.ones_like(flat_ranks))

    # Midrank = sum / count (avoid div-by-zero for unused bins)
    counts = counts.clamp(min=1)
    midranks_per_group = rank_sums / counts  # (total_bins,)

    # Map midranks back: each element gets its group's midrank
    sorted_midranks = midranks_per_group[flat_group_ids].reshape(N, M)

    # Scatter back to original positions
    result = torch.empty_like(data)
    result.scatter_(0, sorted_indices, sorted_midranks)
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
