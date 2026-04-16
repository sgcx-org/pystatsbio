"""Working correlation structures for GEE.

Implements four standard working correlation structures used in
Generalized Estimating Equations (Liang & Zeger, 1986):
  - Independence: R = I
  - Exchangeable (compound symmetry): R_ij = alpha for i != j
  - AR(1): R_ij = alpha^|i-j|
  - Unstructured: R_ij estimated freely (equal cluster sizes only)

RULE 5 EXCEPTION: These classes use mutable internal state (self._alpha,
self._corr_matrix) because the GEE algorithm iteratively updates correlation
parameters during fitting. Making them frozen dataclasses would require
reconstructing a new object at every GEE iteration, which adds complexity
without safety benefit since these objects are internal to a single fit call
and never exposed as part of the public API's frozen result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class CorrStructure(ABC):
    """Base class for GEE working correlation structures.

    Subclasses must implement:
    - ``name`` property: human-readable structure name
    - ``working_corr(cluster_size)``: build the R_i matrix
    - ``estimate(pearson_resids, phi)``: update parameters from residuals
    - ``params`` property: current parameter values as a dict
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable correlation structure name."""
        ...

    @abstractmethod
    def working_corr(self, cluster_size: int) -> NDArray:
        """Return the working correlation matrix R_i for a cluster.

        Parameters
        ----------
        cluster_size : int
            Number of observations in this cluster.

        Returns
        -------
        NDArray
            Correlation matrix of shape (cluster_size, cluster_size).
        """
        ...

    @abstractmethod
    def estimate(
        self, pearson_resids: list[NDArray], phi: float, n_params: int
    ) -> None:
        """Estimate correlation parameters from standardized Pearson residuals.

        Updates internal state in-place. Each element of ``pearson_resids``
        is a vector of Pearson residuals for one cluster.

        Parameters
        ----------
        pearson_resids : list[NDArray]
            Per-cluster Pearson residual vectors.
        phi : float
            Current dispersion estimate.
        n_params : int
            Number of regression parameters (p), used in denominator.
        """
        ...

    @property
    @abstractmethod
    def params(self) -> dict[str, float]:
        """Current parameter values as a dict."""
        ...


class IndependenceCorr(CorrStructure):
    """Independence working correlation: R = I.

    Assumes all within-cluster observations are uncorrelated.
    No parameters to estimate. This is the simplest structure and
    recovers standard GLM-like estimates (though with sandwich SE).
    """

    @property
    def name(self) -> str:
        return "independence"

    def working_corr(self, cluster_size: int) -> NDArray:
        """Return identity matrix.

        Parameters
        ----------
        cluster_size : int
            Number of observations in this cluster.

        Returns
        -------
        NDArray
            Identity matrix of shape (cluster_size, cluster_size).
        """
        return np.eye(cluster_size)

    def estimate(
        self, pearson_resids: list[NDArray], phi: float, n_params: int
    ) -> None:
        """No-op: independence has no parameters to estimate."""

    @property
    def params(self) -> dict[str, float]:
        """Empty dict: no correlation parameters."""
        return {}


class ExchangeableCorr(CorrStructure):
    """Exchangeable (compound symmetry) working correlation.

    R_ij = alpha for i != j, R_ii = 1.

    Alpha is estimated as:
        alpha = [sum_i sum_{j<k} r_ij * r_ik] / [sum_i n_i*(n_i-1)/2 - p] / phi

    where r_ij are Pearson residuals, phi is the dispersion, and p is the
    number of regression parameters.

    References
    ----------
    Liang, K.-Y. & Zeger, S. L. (1986). Biometrika, 73(1), 13-22.
    """

    def __init__(self) -> None:
        self._alpha: float = 0.0

    @property
    def name(self) -> str:
        return "exchangeable"

    def working_corr(self, cluster_size: int) -> NDArray:
        """Build exchangeable correlation matrix.

        Parameters
        ----------
        cluster_size : int
            Number of observations in this cluster.

        Returns
        -------
        NDArray
            Matrix with 1 on diagonal and alpha off-diagonal.
        """
        if cluster_size == 1:
            return np.ones((1, 1))
        R = np.full((cluster_size, cluster_size), self._alpha)
        np.fill_diagonal(R, 1.0)
        return R

    def estimate(
        self, pearson_resids: list[NDArray], phi: float, n_params: int
    ) -> None:
        """Estimate alpha from cross-products of Pearson residuals.

        Parameters
        ----------
        pearson_resids : list[NDArray]
            Per-cluster Pearson residual vectors.
        phi : float
            Current dispersion estimate.
        n_params : int
            Number of regression parameters (p).
        """
        numerator = 0.0
        denominator = 0.0
        for r in pearson_resids:
            n_i = len(r)
            if n_i < 2:
                continue
            # Sum of all cross-products r_j * r_k for j < k
            total = float(np.sum(r)) ** 2 - float(np.sum(r**2))
            numerator += total / 2.0
            denominator += n_i * (n_i - 1) / 2.0

        denominator -= n_params
        if denominator <= 0:
            self._alpha = 0.0
            return

        self._alpha = float(np.clip(numerator / (denominator * phi), -1.0, 1.0))

    @property
    def params(self) -> dict[str, float]:
        """Current alpha estimate."""
        return {"alpha": self._alpha}


class AR1Corr(CorrStructure):
    """AR(1) working correlation: R_ij = alpha^|i-j|.

    Assumes observations within a cluster are ordered (e.g., by time)
    and adjacent observations have correlation alpha.

    Alpha is estimated as:
        alpha = [sum_i sum_t r_{i,t} * r_{i,t+1}] / [sum_i (n_i - 1) - p] / phi

    References
    ----------
    Liang, K.-Y. & Zeger, S. L. (1986). Biometrika, 73(1), 13-22.
    """

    def __init__(self) -> None:
        self._alpha: float = 0.0

    @property
    def name(self) -> str:
        return "ar1"

    def working_corr(self, cluster_size: int) -> NDArray:
        """Build AR(1) correlation matrix.

        Parameters
        ----------
        cluster_size : int
            Number of observations in this cluster.

        Returns
        -------
        NDArray
            Toeplitz matrix with R_ij = alpha^|i-j|.
        """
        if cluster_size == 1:
            return np.ones((1, 1))
        idx = np.arange(cluster_size)
        return np.power(self._alpha, np.abs(idx[:, None] - idx[None, :]))

    def estimate(
        self, pearson_resids: list[NDArray], phi: float, n_params: int
    ) -> None:
        """Estimate alpha from consecutive residual products.

        Parameters
        ----------
        pearson_resids : list[NDArray]
            Per-cluster Pearson residual vectors.
        phi : float
            Current dispersion estimate.
        n_params : int
            Number of regression parameters (p).
        """
        numerator = 0.0
        denominator = 0.0
        for r in pearson_resids:
            n_i = len(r)
            if n_i < 2:
                continue
            numerator += float(np.sum(r[:-1] * r[1:]))
            denominator += n_i - 1

        denominator -= n_params
        if denominator <= 0:
            self._alpha = 0.0
            return

        self._alpha = float(np.clip(numerator / (denominator * phi), -1.0, 1.0))

    @property
    def params(self) -> dict[str, float]:
        """Current alpha estimate."""
        return {"alpha": self._alpha}


class UnstructuredCorr(CorrStructure):
    """Unstructured working correlation.

    R_jk is estimated freely for each pair (j, k). This is only feasible
    when all clusters have the same size, because each R_jk requires data
    from all clusters at positions j and k.

    R_jk = sum_i r_ij * r_ik / (K - p) / phi

    where K is the number of clusters and p is the number of parameters.

    Raises
    ------
    ValueError
        If clusters have unequal sizes during estimation.

    References
    ----------
    Liang, K.-Y. & Zeger, S. L. (1986). Biometrika, 73(1), 13-22.
    """

    def __init__(self) -> None:
        self._corr_matrix: NDArray | None = None

    @property
    def name(self) -> str:
        return "unstructured"

    def working_corr(self, cluster_size: int) -> NDArray:
        """Return the estimated unstructured correlation matrix.

        If not yet estimated, returns the identity matrix.

        Parameters
        ----------
        cluster_size : int
            Number of observations in this cluster.

        Returns
        -------
        NDArray
            Correlation matrix of shape (cluster_size, cluster_size).
        """
        if self._corr_matrix is not None and self._corr_matrix.shape[0] == cluster_size:
            return self._corr_matrix.copy()
        return np.eye(cluster_size)

    def estimate(
        self, pearson_resids: list[NDArray], phi: float, n_params: int
    ) -> None:
        """Estimate all pairwise correlations from residuals.

        Parameters
        ----------
        pearson_resids : list[NDArray]
            Per-cluster Pearson residual vectors. All must have the same length.
        phi : float
            Current dispersion estimate.
        n_params : int
            Number of regression parameters (p).

        Raises
        ------
        ValueError
            If cluster sizes are not all equal.
        """
        sizes = {len(r) for r in pearson_resids}
        if len(sizes) != 1:
            raise ValueError(
                "Unstructured correlation requires all clusters to have the "
                f"same size, got sizes: {sorted(sizes)}"
            )

        n_i = sizes.pop()
        K = len(pearson_resids)
        denom = K - n_params
        if denom <= 0:
            self._corr_matrix = np.eye(n_i)
            return

        # Stack residuals: (K, n_i)
        R_mat = np.vstack([r[np.newaxis, :] for r in pearson_resids])
        # Compute R_jk = sum_i r_ij * r_ik / (denom * phi)
        corr = (R_mat.T @ R_mat) / (denom * phi)

        # Force diagonal to 1 and ensure symmetry
        diag_sqrt = np.sqrt(np.diag(corr))
        diag_sqrt = np.maximum(diag_sqrt, 1e-10)
        corr = corr / (diag_sqrt[:, None] * diag_sqrt[None, :])
        np.fill_diagonal(corr, 1.0)

        self._corr_matrix = corr

    @property
    def params(self) -> dict[str, float]:
        """Current correlation matrix entries as a flat dict."""
        if self._corr_matrix is None:
            return {}
        result = {}
        n = self._corr_matrix.shape[0]
        for j in range(n):
            for k in range(j + 1, n):
                result[f"r_{j}_{k}"] = float(self._corr_matrix[j, k])
        return result


_CORR_STRUCTURES: dict[str, type[CorrStructure]] = {
    "independence": IndependenceCorr,
    "exchangeable": ExchangeableCorr,
    "ar1": AR1Corr,
    "unstructured": UnstructuredCorr,
}


def resolve_corr(corr_structure: str) -> CorrStructure:
    """Resolve a correlation structure name to an instance.

    Parameters
    ----------
    corr_structure : str
        One of 'independence', 'exchangeable', 'ar1', 'unstructured'.

    Returns
    -------
    CorrStructure
        A new instance of the requested correlation structure.

    Raises
    ------
    ValueError
        If the name is not recognized.
    """
    cls = _CORR_STRUCTURES.get(corr_structure.lower())
    if cls is None:
        valid = ", ".join(sorted(_CORR_STRUCTURES.keys()))
        raise ValueError(
            f"Unknown corr_structure: {corr_structure!r}. Valid: {valid}"
        )
    return cls()
