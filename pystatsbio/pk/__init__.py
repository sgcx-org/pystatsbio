"""
Non-compartmental pharmacokinetic analysis (NCA).

NCA is required for every PK study: AUC, Cmax, half-life, clearance, and
volume of distribution. Self-contained, well-defined, formulaic calculations.

Phase 1: NCA only. Compartmental/PopPK is Phase 4+.

Validates against: R packages PKNCA, NonCompart.
"""

from pystatsbio.pk._common import NCAResult
from pystatsbio.pk._nca import LambdaZEstimationError, nca

__all__ = [
    "NCAResult",
    "LambdaZEstimationError",
    "nca",
]
