"""
Epidemiological analysis: measures of association, rate standardization,
and stratified analysis.

Computes risk ratios, odds ratios, risk differences, attributable fractions,
age-standardized rates, and Mantel-Haenszel pooled estimates from 2x2 tables.

Validates against: R epiR, epitools, stats::mantelhaen.test()
"""

from pystatsbio.epi._common import (
    Epi2x2Result,
    EpiMeasure,
    MantelHaenszelResult,
    StandardizedRate,
)
from pystatsbio.epi._mantel_haenszel import mantel_haenszel
from pystatsbio.epi._measures import epi_2by2
from pystatsbio.epi._standardize import rate_standardize

__all__ = [
    "EpiMeasure",
    "Epi2x2Result",
    "StandardizedRate",
    "MantelHaenszelResult",
    "epi_2by2",
    "rate_standardize",
    "mantel_haenszel",
]
