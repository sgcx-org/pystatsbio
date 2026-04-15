"""Dose-response model functions.

Each function computes the mean response at given dose levels for a
specific parametric model. These are the building blocks for curve fitting.

All models use a parameterization where ``hill > 0`` means the response
**increases** with dose (agonist) and ``hill < 0`` means it **decreases**
(antagonist/inhibitor).  This is the opposite of R ``drc``'s ``b`` parameter
convention (where ``b < 0`` denotes an increasing curve).

Dose = 0 is handled via IEEE 754 arithmetic: ``log(0) = -inf``, and the
exponential terms evaluate correctly to give the ``bottom`` (or ``top``)
asymptote.

Validates against: R drc::LL.4(), LL.5(), W1.4(), W2.4(), BC.5()
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Safe log-dose utility
# ---------------------------------------------------------------------------

def _safe_log_dose(dose: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute log(dose) with dose=0 mapped to -inf (IEEE 754 compliant).

    This avoids runtime warnings while producing correct asymptotic values
    in all model formulas: when dose→0, the models evaluate to the
    ``bottom`` or ``top`` asymptote depending on the sign of ``hill``.
    """
    with np.errstate(divide="ignore"):
        return np.where(dose > 0, np.log(dose), -np.inf)


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def ll4(
    dose: NDArray[np.floating],
    bottom: float,
    top: float,
    ec50: float,
    hill: float,
) -> NDArray[np.floating]:
    """4-parameter log-logistic (LL.4) model.

    .. math::
        f(x) = c + \\frac{d - c}{1 + \\exp\\bigl(-b \\cdot (\\ln x - \\ln e)\\bigr)}

    where ``c = bottom``, ``d = top``, ``e = ec50``, ``b = hill``.

    Parameters
    ----------
    dose : array
        Dose (concentration) values.  May contain zeros.
    bottom : float
        Lower asymptote (response at dose → 0 for hill > 0).
    top : float
        Upper asymptote (response at dose → ∞ for hill > 0).
    ec50 : float
        Dose producing 50 % of the maximal effect.
    hill : float
        Hill slope.  Positive for increasing, negative for decreasing.

    Returns
    -------
    NDArray
        Predicted response values.

    Validates against: R drc::LL.4()
    """
    dose = np.asarray(dose, dtype=np.float64)
    log_dose = _safe_log_dose(dose)
    log_ec50 = np.log(ec50)
    exponent = -hill * (log_dose - log_ec50)
    return bottom + (top - bottom) / (1.0 + np.exp(exponent))


def ll5(
    dose: NDArray[np.floating],
    bottom: float,
    top: float,
    ec50: float,
    hill: float,
    asymmetry: float,
) -> NDArray[np.floating]:
    """5-parameter log-logistic (LL.5) model.

    Asymmetric extension of LL.4 with an extra shape parameter ``f``.
    When ``asymmetry = 1``, this reduces to LL.4.

    .. math::
        f(x) = c + \\frac{d - c}{\\bigl(1 + \\exp(-b \\cdot (\\ln x - \\ln e))\\bigr)^f}

    Parameters
    ----------
    dose : array
        Dose (concentration) values.
    bottom, top, ec50, hill : float
        Same as :func:`ll4`.
    asymmetry : float
        Asymmetry parameter (``f``).  ``1.0`` = symmetric (LL.4).

    Returns
    -------
    NDArray

    Validates against: R drc::LL.5()
    """
    dose = np.asarray(dose, dtype=np.float64)
    log_dose = _safe_log_dose(dose)
    log_ec50 = np.log(ec50)
    exponent = -hill * (log_dose - log_ec50)
    return bottom + (top - bottom) / (1.0 + np.exp(exponent)) ** asymmetry


def weibull1(
    dose: NDArray[np.floating],
    bottom: float,
    top: float,
    ec50: float,
    hill: float,
) -> NDArray[np.floating]:
    """Weibull type 1 (W1.4) model.

    Asymmetric dose-response, left-skewed.

    .. math::
        f(x) = c + (d - c) \\exp\\bigl(-\\exp(-b \\cdot (\\ln x - \\ln e))\\bigr)

    Parameters
    ----------
    dose : array
        Dose (concentration) values.
    bottom, top, ec50, hill : float
        Same as :func:`ll4`.

    Returns
    -------
    NDArray

    Validates against: R drc::W1.4()
    """
    dose = np.asarray(dose, dtype=np.float64)
    log_dose = _safe_log_dose(dose)
    log_ec50 = np.log(ec50)
    inner = -hill * (log_dose - log_ec50)
    return bottom + (top - bottom) * np.exp(-np.exp(inner))


def weibull2(
    dose: NDArray[np.floating],
    bottom: float,
    top: float,
    ec50: float,
    hill: float,
) -> NDArray[np.floating]:
    """Weibull type 2 (W2.4) model.

    Asymmetric dose-response, right-skewed.

    .. math::
        f(x) = c + (d - c) \\bigl(1 - \\exp(-\\exp(-b \\cdot (\\ln x - \\ln e)))\\bigr)

    Parameters
    ----------
    dose : array
        Dose (concentration) values.
    bottom, top, ec50, hill : float
        Same as :func:`ll4`.

    Returns
    -------
    NDArray

    Validates against: R drc::W2.4()
    """
    dose = np.asarray(dose, dtype=np.float64)
    log_dose = _safe_log_dose(dose)
    log_ec50 = np.log(ec50)
    inner = -hill * (log_dose - log_ec50)
    return bottom + (top - bottom) * (1.0 - np.exp(-np.exp(inner)))


def brain_cousens(
    dose: NDArray[np.floating],
    bottom: float,
    top: float,
    ec50: float,
    hill: float,
    hormesis: float,
) -> NDArray[np.floating]:
    """Brain-Cousens hormesis model (BC.5).

    Biphasic dose-response with low-dose stimulation.  The ``hormesis``
    parameter adds a linear term to the numerator that can push the
    response above the upper asymptote at low doses.

    .. math::
        f(x) = c + \\frac{d - c + f \\cdot x}{1 + \\exp(-b \\cdot (\\ln x - \\ln e))}

    Parameters
    ----------
    dose : array
        Dose (concentration) values.
    bottom, top, ec50, hill : float
        Same as :func:`ll4`.
    hormesis : float
        Hormesis coefficient (``f``).  ``0.0`` reduces to LL.4.

    Returns
    -------
    NDArray

    Validates against: R drc::BC.5()
    """
    dose = np.asarray(dose, dtype=np.float64)
    log_dose = _safe_log_dose(dose)
    log_ec50 = np.log(ec50)
    exponent = -hill * (log_dose - log_ec50)
    return bottom + (top - bottom + hormesis * dose) / (1.0 + np.exp(exponent))


# ---------------------------------------------------------------------------
# Model registry — maps model name to (function, parameter_names)
# ---------------------------------------------------------------------------

_MODEL_MAP: dict[str, tuple[Callable, list[str]]] = {
    "LL.4": (ll4, ["bottom", "top", "ec50", "hill"]),
    "LL.5": (ll5, ["bottom", "top", "ec50", "hill", "asymmetry"]),
    "W1.4": (weibull1, ["bottom", "top", "ec50", "hill"]),
    "W2.4": (weibull2, ["bottom", "top", "ec50", "hill"]),
    "BC.5": (brain_cousens, ["bottom", "top", "ec50", "hill", "hormesis"]),
}

VALID_MODELS = tuple(_MODEL_MAP.keys())
