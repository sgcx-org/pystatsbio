"""Result types for GEE (Generalized Estimating Equations).

``GEEParams`` is the frozen payload of computed outputs; ``GEESolution`` is
the public return — it wraps a ``core.result.Result[GEEParams]`` so every GEE
fit exposes the same ``.backend_name`` / ``.timing`` / ``.warnings`` / ``.info``
metadata and Jupyter ``_repr_html_`` as the rest of the ecosystem
(``pystatsbio/CONVENTIONS.md`` B2).
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray
from pystatistics.core.result import Result, SolutionReprMixin


@dataclass(frozen=True)
class GEEParams:
    """Computed payload of a GEE fit.

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


class GEESolution(SolutionReprMixin):
    """Public result of a GEE fit — a Solution wrapping ``Result[GEEParams]``.

    Exposes every fit output as a read-only property plus the uniform
    ``.backend_name`` / ``.timing`` / ``.warnings`` / ``.info`` metadata and a
    Jupyter ``_repr_html_`` (via :class:`SolutionReprMixin`).
    """

    def __init__(self, result: Result[GEEParams]) -> None:
        self._result = result

    # --- Metadata (from the Result envelope) ---
    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def timing(self) -> dict[str, float] | None:
        return self._result.timing

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._result.warnings

    @property
    def info(self) -> dict:
        return self._result.info

    # --- Fit outputs (from the payload) ---
    @property
    def coefficients(self) -> NDArray:
        return self._result.params.coefficients

    @property
    def naive_se(self) -> NDArray:
        return self._result.params.naive_se

    @property
    def robust_se(self) -> NDArray:
        return self._result.params.robust_se

    @property
    def naive_vcov(self) -> NDArray:
        return self._result.params.naive_vcov

    @property
    def robust_vcov(self) -> NDArray:
        return self._result.params.robust_vcov

    @property
    def z_values(self) -> NDArray:
        return self._result.params.z_values

    @property
    def p_values(self) -> NDArray:
        return self._result.params.p_values

    @property
    def fitted_values(self) -> NDArray:
        return self._result.params.fitted_values

    @property
    def residuals(self) -> NDArray:
        return self._result.params.residuals

    @property
    def correlation_type(self) -> str:
        return self._result.params.correlation_type

    @property
    def correlation_params(self) -> dict:
        return self._result.params.correlation_params

    @property
    def scale(self) -> float:
        return self._result.params.scale

    @property
    def n_clusters(self) -> int:
        return self._result.params.n_clusters

    @property
    def n_obs(self) -> int:
        return self._result.params.n_obs

    @property
    def family_name(self) -> str:
        return self._result.params.family_name

    @property
    def link_name(self) -> str:
        return self._result.params.link_name

    @property
    def converged(self) -> bool:
        return self._result.params.converged

    @property
    def n_iter(self) -> int:
        return self._result.params.n_iter

    @property
    def names(self) -> tuple[str, ...] | None:
        return self._result.params.names

    @property
    def coef(self) -> dict[str, float]:
        """Coefficient dict (name -> value).

        Uses provided names, or 'x0', 'x1', ... if names were not given.
        """
        p = self._result.params
        if p.names is not None:
            labels = p.names
        else:
            labels = tuple(f"x{i}" for i in range(len(p.coefficients)))
        return {name: float(val) for name, val in zip(labels, p.coefficients)}

    def summary(self) -> str:
        """R-style summary matching geepack::geeglm() output."""
        p = self._result.params
        lines = [
            "GEE Model Summary",
            "=" * 65,
            "",
            f"Family : {p.family_name}",
            f"Link   : {p.link_name}",
            f"Corr   : {p.correlation_type}",
            f"Scale  : {p.scale:.4f}",
            "",
            f"Number of clusters   : {p.n_clusters}",
            f"Number of observations: {p.n_obs}",
            f"Converged            : {p.converged} ({p.n_iter} iterations)",
            "",
        ]

        if p.names is not None:
            labels = list(p.names)
        else:
            labels = [f"x{i}" for i in range(len(p.coefficients))]

        header = f"{'':>12s} {'Estimate':>10s} {'Naive SE':>10s} {'Robust SE':>10s} {'z':>8s} {'Pr(>|z|)':>10s}"
        lines.append("Coefficients:")
        lines.append(header)
        lines.append("-" * 65)
        for i, label in enumerate(labels):
            lines.append(
                f"{label:>12s} {p.coefficients[i]:10.4f} "
                f"{p.naive_se[i]:10.4f} {p.robust_se[i]:10.4f} "
                f"{p.z_values[i]:8.4f} "
                + (
                    f"{'< .0001':>10s}"
                    if p.p_values[i] < 0.0001
                    else f"{p.p_values[i]:10.4f}"
                )
            )

        lines.append("")
        lines.append("Estimated correlation parameters:")
        if p.correlation_params:
            for key, val in p.correlation_params.items():
                lines.append(f"  {key}: {val:.4f}")
        else:
            lines.append("  (none)")

        return "\n".join(lines)

    def __repr__(self) -> str:
        p = self._result.params
        return (
            f"GEESolution(family={p.family_name!r}, "
            f"correlation={p.correlation_type!r}, n_obs={p.n_obs}, "
            f"converged={p.converged})"
        )
