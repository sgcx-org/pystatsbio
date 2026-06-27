"""Shared result types and helpers for power/sample size calculations."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.result import Result, SolutionReprMixin
from scipy.optimize import brentq


@dataclass(frozen=True)
class PowerParams:
    """Computed payload of a power/sample size calculation.

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


class PowerSolution(SolutionReprMixin):
    """Public result of a power/sample size calculation.

    A Solution wrapping ``Result[PowerParams]`` — exposes every computed value
    as a read-only property plus the uniform
    ``.backend_name`` / ``.timing`` / ``.warnings`` / ``.info`` metadata and a
    Jupyter ``_repr_html_`` (via :class:`SolutionReprMixin`).
    """

    def __init__(self, result: Result[PowerParams]) -> None:
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

    # --- Computed values (from the payload) ---
    @property
    def n(self) -> int | None:
        return self._result.params.n

    @property
    def power(self) -> float | None:
        return self._result.params.power

    @property
    def effect_size(self) -> float | None:
        return self._result.params.effect_size

    @property
    def alpha(self) -> float:
        return self._result.params.alpha

    @property
    def alternative(self) -> str:
        return self._result.params.alternative

    @property
    def method(self) -> str:
        return self._result.params.method

    @property
    def note(self) -> str:
        return self._result.params.note

    def summary(self) -> str:
        """Human-readable summary, similar to R's print.power.htest."""
        p = self._result.params
        lines = [p.method, ""]
        if p.n is not None:
            lines.append(f"              n = {p.n}")
        if p.effect_size is not None:
            lines.append(f"    effect size = {p.effect_size:.6f}")
        lines.append(f"          alpha = {p.alpha}")
        if p.power is not None:
            lines.append(f"          power = {p.power:.6f}")
        lines.append(f"    alternative = {p.alternative}")
        if p.note:
            lines.append("")
            lines.append(f"NOTE: {p.note}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        p = self._result.params
        return (
            f"PowerSolution(method={p.method!r}, n={p.n}, "
            f"power={p.power}, effect_size={p.effect_size})"
        )


def _solution(
    *,
    n: int | None,
    power: float | None,
    effect_size: float | None,
    alpha: float,
    alternative: str,
    method: str,
    note: str = "",
) -> PowerSolution:
    """Build a :class:`PowerSolution` from the computed power-analysis values.

    Wraps the payload in a ``Result`` envelope (CPU backend, ``method`` recorded
    in ``info``) so every power function returns the uniform Solution shape.
    """
    params = PowerParams(
        n=n,
        power=power,
        effect_size=effect_size,
        alpha=alpha,
        alternative=alternative,
        method=method,
        note=note,
    )
    result = Result(
        params=params,
        info={"method": method},
        timing=None,
        backend_name="cpu",
    )
    return PowerSolution(result)


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
    ValidationError
        On any validation failure.
    """
    none_count = sum(x is None for x in (n, effect, power))
    if none_count != 1:
        raise ValidationError(
            f"Exactly one of n, {effect_name}, power must be None "
            f"(got {none_count} None values)"
        )

    if not (0.0 < alpha < 1.0):
        raise ValidationError(f"alpha must be in (0, 1), got {alpha}")

    if n is not None and n < 2:
        raise ValidationError(f"n must be >= 2, got {n}")

    if power is not None and not (0.0 < power < 1.0):
        raise ValidationError(f"power must be in (0, 1), got {power}")

    if effect is not None and not math.isfinite(effect):
        raise ValidationError(f"{effect_name} must be finite, got {effect}")

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
    ValidationError
        If the bracket does not straddle the target (no sign change).
    """
    lo, hi = bracket
    f_lo = func(lo) - target
    f_hi = func(hi) - target

    # Check bracket validity
    if f_lo * f_hi > 0:
        raise ValidationError(
            f"Cannot solve: target {target:.6f} is outside achievable range "
            f"[{func(lo):.6f}, {func(hi):.6f}] for the given parameters. "
            f"Try different input values."
        )

    return brentq(lambda x: func(x) - target, lo, hi, xtol=xtol, maxiter=maxiter)
