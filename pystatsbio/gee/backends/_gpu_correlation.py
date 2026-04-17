"""GPU working-correlation helpers for GEE.

Wraps the CPU :class:`CorrStructure` instances so that R_i matrices and
parameter updates happen on the device, against Pearson residuals that
stay in GPU memory between iterations. Only the final float (for
``GEEResult.correlation_params``) crosses the bus per iteration.

The four supported structures mirror the CPU module exactly:
  - independence: R = I, no parameters
  - exchangeable: R_ij = alpha (compound symmetry)
  - AR(1): R_ij = alpha^|i-j|
  - unstructured: free R (equal cluster sizes only)
"""

from __future__ import annotations

from typing import Any

from pystatsbio.gee._correlation import (
    AR1Corr,
    CorrStructure,
    ExchangeableCorr,
    IndependenceCorr,
    UnstructuredCorr,
)


class GPUCorrelation:
    """Device-resident mirror of a :class:`CorrStructure`.

    State is managed on the GPU between iterations; at the end of the
    fit ``sync_to_cpu`` copies the estimated parameters back into the
    wrapped CPU CorrStructure so :class:`GEEResult.correlation_params`
    is populated correctly.
    """

    def __init__(self, cpu_corr: CorrStructure, device: Any, dtype: Any) -> None:
        import torch

        self._cpu_corr = cpu_corr
        self._device = device
        self._dtype = dtype
        self._torch = torch

        # Device-resident alpha (exchangeable, ar1) or R matrix
        # (unstructured). Kept as a 0-D tensor to avoid per-iteration
        # Python-float syncs during R matrix assembly.
        self._alpha_gpu = torch.zeros((), device=device, dtype=dtype)
        self._R_unstructured: Any | None = None  # (m, m) tensor

    @property
    def name(self) -> str:
        return self._cpu_corr.name

    def build_R(self, cluster_size: int) -> Any:
        """Build (s, s) working correlation matrix on device."""
        torch = self._torch
        s = cluster_size
        if isinstance(self._cpu_corr, IndependenceCorr):
            return torch.eye(s, device=self._device, dtype=self._dtype)
        if isinstance(self._cpu_corr, ExchangeableCorr):
            if s == 1:
                return torch.ones(1, 1, device=self._device, dtype=self._dtype)
            R = torch.full(
                (s, s), 0.0, device=self._device, dtype=self._dtype,
            )
            R = R + self._alpha_gpu
            # off-diagonal is alpha; diagonal 1.
            R = R - torch.diag_embed(torch.full(
                (s,), 0.0, device=self._device, dtype=self._dtype,
            ) + self._alpha_gpu)
            R = R + torch.eye(s, device=self._device, dtype=self._dtype)
            return R
        if isinstance(self._cpu_corr, AR1Corr):
            if s == 1:
                return torch.ones(1, 1, device=self._device, dtype=self._dtype)
            idx = torch.arange(s, device=self._device, dtype=self._dtype)
            d = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
            # alpha ** d; handle alpha=0 by raising 0**0=1 safely via mask.
            # torch.pow(0, 0) returns 1 which is what we want here.
            return torch.pow(self._alpha_gpu, d)
        if isinstance(self._cpu_corr, UnstructuredCorr):
            if self._R_unstructured is not None and (
                self._R_unstructured.shape[0] == s
            ):
                return self._R_unstructured
            return torch.eye(s, device=self._device, dtype=self._dtype)
        raise RuntimeError(
            f"GPU GEE: unsupported correlation structure {self.name!r}"
        )

    def estimate(
        self,
        pearson_groups: dict[int, Any],  # {size: (K_s, s) tensor}
        phi_gpu: Any,
        n_params: int,
    ) -> None:
        """Update correlation parameters from GPU Pearson residual groups.

        ``pearson_groups`` maps cluster size s to a (K_s, s) tensor of
        Pearson residuals for clusters of that size. All tensors are on
        the same device as ``phi_gpu``.
        """
        torch = self._torch

        if isinstance(self._cpu_corr, IndependenceCorr):
            return

        if isinstance(self._cpu_corr, ExchangeableCorr):
            num = torch.zeros((), device=self._device, dtype=self._dtype)
            denom = 0.0
            for s, r_batch in pearson_groups.items():
                if s < 2:
                    continue
                # per cluster: (sum r)^2 - sum r^2, halved.
                sum_r = r_batch.sum(dim=1)
                sum_r2 = (r_batch * r_batch).sum(dim=1)
                # cross products across i<k — matches CPU formula.
                num = num + (sum_r * sum_r - sum_r2).sum() / 2.0
                denom += r_batch.shape[0] * s * (s - 1) / 2.0
            denom -= n_params
            if denom <= 0:
                self._alpha_gpu = torch.zeros(
                    (), device=self._device, dtype=self._dtype,
                )
                return
            alpha = num / (denom * phi_gpu)
            alpha = torch.clamp(alpha, min=-1.0, max=1.0)
            self._alpha_gpu = alpha
            return

        if isinstance(self._cpu_corr, AR1Corr):
            num = torch.zeros((), device=self._device, dtype=self._dtype)
            denom = 0.0
            for s, r_batch in pearson_groups.items():
                if s < 2:
                    continue
                # consecutive lag-1 products within each cluster.
                num = num + (r_batch[:, :-1] * r_batch[:, 1:]).sum()
                denom += r_batch.shape[0] * (s - 1)
            denom -= n_params
            if denom <= 0:
                self._alpha_gpu = torch.zeros(
                    (), device=self._device, dtype=self._dtype,
                )
                return
            alpha = num / (denom * phi_gpu)
            alpha = torch.clamp(alpha, min=-1.0, max=1.0)
            self._alpha_gpu = alpha
            return

        if isinstance(self._cpu_corr, UnstructuredCorr):
            # Unstructured demands equal sizes — CPU path already raises
            # if unequal, so we assume exactly one size group here.
            if len(pearson_groups) != 1:
                raise ValueError(
                    "Unstructured correlation requires all clusters to "
                    f"have the same size, got sizes: "
                    f"{sorted(pearson_groups.keys())}"
                )
            s, r_batch = next(iter(pearson_groups.items()))
            K = r_batch.shape[0]
            denom = K - n_params
            if denom <= 0:
                self._R_unstructured = torch.eye(
                    s, device=self._device, dtype=self._dtype,
                )
                return
            # R_jk = sum_i r_ij * r_ik / (denom * phi)
            corr = (r_batch.T @ r_batch) / (denom * phi_gpu)
            diag_sqrt = torch.sqrt(torch.clamp(torch.diagonal(corr), min=1e-10))
            corr = corr / (diag_sqrt.unsqueeze(0) * diag_sqrt.unsqueeze(1))
            # Force diagonal to 1.
            corr = corr - torch.diag_embed(torch.diagonal(corr))
            corr = corr + torch.eye(
                s, device=self._device, dtype=self._dtype,
            )
            self._R_unstructured = corr
            return

        raise RuntimeError(
            f"GPU GEE: unsupported correlation structure {self.name!r}"
        )

    def sync_to_cpu(self) -> None:
        """Copy GPU-estimated parameters back into the wrapped CPU object.

        Reaches into the CPU CorrStructure's private attributes because
        those are exactly the state that :attr:`CorrStructure.params`
        exposes in :class:`GEEResult`. The CorrStructure interface does
        not currently define a public setter (the CPU fit mutates the
        same private attributes in-place), so this module follows the
        same convention.
        """
        import numpy as np

        if isinstance(self._cpu_corr, (ExchangeableCorr, AR1Corr)):
            self._cpu_corr._alpha = float(
                self._alpha_gpu.detach().to("cpu").item()
            )
        elif isinstance(self._cpu_corr, UnstructuredCorr):
            if self._R_unstructured is not None:
                self._cpu_corr._corr_matrix = (
                    self._R_unstructured.detach().to("cpu").numpy().astype(np.float64)
                )
