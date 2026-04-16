"""Shared result types for GEE (Generalized Estimating Equations).

Contains the frozen dataclass that holds all outputs from a GEE fit,
including coefficients, robust and naive standard errors, correlation
parameters, and convergence diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True)
class GEEResult:
    """Result from fitting a GEE model.

    Attributes
    ----------
    coefficients : NDArray
        Regression coefficients (length p).
    naive_se : NDArray
        Model-based (naive) standard errors.
    robust_se : NDArray
        Sandwich (robust) standard errors.
    naive_vcov : NDArray
        Model-based variance-covariance matrix (p x p).
    robust_vcov : NDArray
        Sandwich variance-covariance matrix (p x p).
    z_values : NDArray
        Wald z-statistics: coefficients / robust_se.
    p_values : NDArray
        Two-sided p-values from z (normal approximation).
    fitted_values : NDArray
        Predicted means (mu_hat) for all observations.
    residuals : NDArray
        Pearson residuals: (y - mu_hat) / sqrt(V(mu_hat)).
    correlation_type : str
        Working correlation structure name.
    correlation_params : dict
        Estimated correlation parameters (e.g. {'alpha': 0.35}).
    scale : float
        Estimated dispersion parameter (phi).
    n_clusters : int
        Number of clusters in the data.
    n_obs : int
        Total number of observations.
    family_name : str
        GLM family name (e.g. 'gaussian', 'binomial').
    link_name : str
        Link function name (e.g. 'identity', 'logit').
    converged : bool
        Whether the iterative algorithm converged.
    n_iter : int
        Number of iterations performed.
    names : tuple[str, ...] | None
        Coefficient names, if provided.
    """

    coefficients: NDArray
    naive_se: NDArray
    robust_se: NDArray
    naive_vcov: NDArray
    robust_vcov: NDArray
    z_values: NDArray
    p_values: NDArray
    fitted_values: NDArray
    residuals: NDArray
    correlation_type: str
    correlation_params: dict
    scale: float
    n_clusters: int
    n_obs: int
    family_name: str
    link_name: str
    converged: bool
    n_iter: int
    names: tuple[str, ...] | None

    @property
    def coef(self) -> dict[str, float]:
        """Coefficient dict (name -> value).

        Uses provided names, or 'x0', 'x1', ... if names were not given.

        Returns
        -------
        dict[str, float]
            Mapping from coefficient name to estimated value.
        """
        if self.names is not None:
            labels = self.names
        else:
            labels = tuple(f"x{i}" for i in range(len(self.coefficients)))
        return {name: float(val) for name, val in zip(labels, self.coefficients)}

    def summary(self) -> str:
        """R-style summary matching geepack::geeglm() output.

        Returns
        -------
        str
            Multi-line human-readable summary of the GEE fit.
        """
        lines = [
            "GEE Model Summary",
            "=" * 65,
            "",
            f"Family : {self.family_name}",
            f"Link   : {self.link_name}",
            f"Corr   : {self.correlation_type}",
            f"Scale  : {self.scale:.4f}",
            "",
            f"Number of clusters   : {self.n_clusters}",
            f"Number of observations: {self.n_obs}",
            f"Converged            : {self.converged} ({self.n_iter} iterations)",
            "",
        ]

        # Coefficient table
        if self.names is not None:
            labels = list(self.names)
        else:
            labels = [f"x{i}" for i in range(len(self.coefficients))]

        header = f"{'':>12s} {'Estimate':>10s} {'Naive SE':>10s} {'Robust SE':>10s} {'z':>8s} {'Pr(>|z|)':>10s}"
        lines.append("Coefficients:")
        lines.append(header)
        lines.append("-" * 65)
        for i, label in enumerate(labels):
            lines.append(
                f"{label:>12s} {self.coefficients[i]:10.4f} "
                f"{self.naive_se[i]:10.4f} {self.robust_se[i]:10.4f} "
                f"{self.z_values[i]:8.4f} "
                + (
                    f"{'< .0001':>10s}"
                    if self.p_values[i] < 0.0001
                    else f"{self.p_values[i]:10.4f}"
                )
            )

        lines.append("")
        lines.append("Estimated correlation parameters:")
        if self.correlation_params:
            for key, val in self.correlation_params.items():
                lines.append(f"  {key}: {val:.4f}")
        else:
            lines.append("  (none)")

        return "\n".join(lines)
