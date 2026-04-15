"""Shared result types and helpers for power/sample size calculations."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from scipy.optimize import brentq


@dataclass(frozen=True)
class PowerResult:
    """Result of a power/sample size calculation.

    Exactly one of n, power, or effect_size will have been solved for
    (the parameter passed as None). The others are the user-supplied inputs.
    """

    n: int | None
    power: float | None
    effect_size: float | None
    alpha: float
    alternative: str
    method: str
    note: str = ""

    def summary(self) -> str:
        """Human-readable summary, similar to R's print.power.htest."""
        lines = [self.method, ""]
        if self.n is not None:
            lines.append(f"              n = {self.n}")
        if self.effect_size is not None:
            lines.append(f"    effect size = {self.effect_size:.6f}")
        lines.append(f"          alpha = {self.alpha}")
        if self.power is not None:
            lines.append(f"          power = {self.power:.6f}")
        lines.append(f"    alternative = {self.alternative}")
        if self.note:
            lines.append("")
            lines.append(f"NOTE: {self.note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------

def _check_power_args(
    *,
    n: int | float | None,
    effect: float | None,
    power: float | None,
    alpha: float,
    effect_name: str = "effect_size",
) -> str:
    """Validate power-analysis inputs. Return the name of the parameter to solve for.

    Rules
    -----
    - Exactly one of *n*, *effect*, *power* must be ``None``.
    - *alpha* must be in (0, 1).
    - If provided, *n* must be >= 2.
    - If provided, *power* must be in (0, 1).
    - If provided, *effect* must be finite.

    Returns
    -------
    str
        ``'n'``, ``'effect'``, or ``'power'`` — the parameter to solve for.

    Raises
    ------
    ValueError
        On any validation failure.
    """
    none_count = sum(x is None for x in (n, effect, power))
    if none_count != 1:
        raise ValueError(
            f"Exactly one of n, {effect_name}, power must be None "
            f"(got {none_count} None values)"
        )

    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if n is not None and n < 2:
        raise ValueError(f"n must be >= 2, got {n}")

    if power is not None and not (0.0 < power < 1.0):
        raise ValueError(f"power must be in (0, 1), got {power}")

    if effect is not None and not math.isfinite(effect):
        raise ValueError(f"{effect_name} must be finite, got {effect}")

    if n is None:
        return "n"
    if effect is None:
        return "effect"
    return "power"


# ---------------------------------------------------------------------------
# Shared root-finding
# ---------------------------------------------------------------------------

def _solve_parameter(
    func: Callable[[float], float],
    target: float,
    bracket: tuple[float, float],
    *,
    xtol: float = 1e-10,
    maxiter: int = 1000,
) -> float:
    """Solve ``func(x) == target`` via Brent's method.

    Parameters
    ----------
    func : callable
        Monotonic function of one variable (e.g. computes power as f(n)).
    target : float
        Target value (e.g. desired power).
    bracket : tuple
        ``(lower, upper)`` bracket. ``func(lower) - target`` and
        ``func(upper) - target`` must have opposite signs.

    Returns
    -------
    float
        The solution *x* such that ``func(x) ≈ target``.

    Raises
    ------
    ValueError
        If the bracket does not straddle the target (no sign change).
    """
    lo, hi = bracket
    f_lo = func(lo) - target
    f_hi = func(hi) - target

    # Check bracket validity
    if f_lo * f_hi > 0:
        raise ValueError(
            f"Cannot solve: target {target:.6f} is outside achievable range "
            f"[{func(lo):.6f}, {func(hi):.6f}] for the given parameters. "
            f"Try different input values."
        )

    return brentq(lambda x: func(x) - target, lo, hi, xtol=xtol, maxiter=maxiter)
