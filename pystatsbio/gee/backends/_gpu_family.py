"""Torch-native GLM family operations for GEE GPU fitting.

The pystatistics ``Family`` / ``Link`` classes compute on numpy. Pulling
per-iteration eta, mu, var, mu_eta vectors through numpy on a tensor-
resident fit would round-trip the entire response vector through PCIe
every iteration — defeating the device-resident pipeline.

This module provides the minimal set of elementwise torch ops needed by
the GEE iteration (and sandwich) for the four supported family/link
pairs. It intentionally mirrors only the canonical link for each family
(the pair ``resolve_family`` returns), matching the CPU GEE path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


# A bundle of elementwise torch callables for a (family, canonical link)
# pair. ``mu_from_y`` gives a safe initial mean vector that stays in the
# valid support of the family, used by GPU IRLS initialization.
@dataclass(frozen=True)
class GPUFamilyOps:
    """Torch elementwise operations for a GLM family/link pair."""

    name: str
    link_name: str
    linkinv: Callable[[Any], Any]
    mu_eta: Callable[[Any], Any]
    variance: Callable[[Any], Any]
    link_fn: Callable[[Any], Any]
    mu_from_y: Callable[[Any], Any]


def resolve_gpu_family(name: str) -> GPUFamilyOps:
    """Return GPU family ops for a supported family name.

    Parameters
    ----------
    name : str
        One of 'gaussian', 'binomial', 'poisson', 'gamma'.

    Returns
    -------
    GPUFamilyOps
        Elementwise torch callables for that family's canonical link.

    Raises
    ------
    ValueError
        If the family is not supported on the GPU path. The CPU path
        remains available for any family the CPU GEE accepts.
    """
    import torch

    name_lc = name.lower()

    if name_lc == "gaussian":
        return GPUFamilyOps(
            name="gaussian",
            link_name="identity",
            linkinv=lambda eta: eta,
            mu_eta=lambda eta: torch.ones_like(eta),
            variance=lambda mu: torch.ones_like(mu),
            link_fn=lambda mu: mu,
            mu_from_y=lambda y: y.clone(),
        )

    if name_lc == "binomial":
        # Logistic link: (y + 0.5)/2 keeps init mu away from 0/1 boundary
        # so log-odds are finite, matching R's binomial$initialize.
        def _mu_from_y(y):
            return (y + 0.5) / 2.0

        def _linkinv(eta):
            return torch.sigmoid(eta)

        def _mu_eta(eta):
            s = torch.sigmoid(eta)
            return s * (1.0 - s)

        def _variance(mu):
            return mu * (1.0 - mu)

        def _link_fn(mu):
            # logit with clamp to avoid inf on init
            mu_c = torch.clamp(mu, min=1e-10, max=1.0 - 1e-10)
            return torch.log(mu_c / (1.0 - mu_c))

        return GPUFamilyOps(
            name="binomial", link_name="logit",
            linkinv=_linkinv, mu_eta=_mu_eta, variance=_variance,
            link_fn=_link_fn, mu_from_y=_mu_from_y,
        )

    if name_lc == "poisson":
        return GPUFamilyOps(
            name="poisson", link_name="log",
            linkinv=lambda eta: torch.exp(eta),
            mu_eta=lambda eta: torch.exp(eta),
            variance=lambda mu: mu,
            link_fn=lambda mu: torch.log(torch.clamp(mu, min=1e-10)),
            mu_from_y=lambda y: y + 0.1,
        )

    if name_lc == "gamma":
        # Canonical inverse link. ``mu_from_y`` clamps to positive domain.
        return GPUFamilyOps(
            name="gamma", link_name="inverse",
            linkinv=lambda eta: 1.0 / eta,
            mu_eta=lambda eta: -1.0 / (eta * eta),
            variance=lambda mu: mu * mu,
            link_fn=lambda mu: 1.0 / torch.clamp(mu, min=1e-10),
            mu_from_y=lambda y: torch.clamp(y, min=1e-2),
        )

    raise ValueError(
        f"GPU GEE: unsupported family {name!r}. Supported: "
        "gaussian, binomial, poisson, gamma."
    )
