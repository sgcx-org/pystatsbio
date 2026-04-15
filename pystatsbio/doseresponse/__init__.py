"""
Dose-response modeling for preclinical pharmacology.

The workhorse of in vitro assay analysis and toxicology studies.
Provides 4PL/5PL curve fitting, EC50/IC50 estimation, relative potency,
benchmark dose analysis, and GPU-accelerated batch fitting for HTS campaigns.

Validates against: R packages drc, nplr, BMDS.
"""

from pystatsbio.doseresponse._batch import fit_drm_batch
from pystatsbio.doseresponse._bmd import BMDResult, bmd
from pystatsbio.doseresponse._common import (
    BatchDoseResponseResult,
    CurveParams,
    DoseResponseResult,
)
from pystatsbio.doseresponse._fit import fit_drm
from pystatsbio.doseresponse._models import (
    brain_cousens,
    ll4,
    ll5,
    weibull1,
    weibull2,
)
from pystatsbio.doseresponse._potency import (
    EC50Result,
    RelativePotencyResult,
    ec50,
    relative_potency,
)

__all__ = [
    "CurveParams",
    "DoseResponseResult",
    "BatchDoseResponseResult",
    "EC50Result",
    "RelativePotencyResult",
    "BMDResult",
    "ll4",
    "ll5",
    "weibull1",
    "weibull2",
    "brain_cousens",
    "fit_drm",
    "fit_drm_batch",
    "ec50",
    "relative_potency",
    "bmd",
]
