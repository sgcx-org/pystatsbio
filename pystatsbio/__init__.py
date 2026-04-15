"""
PyStatsBio: Biotech and pharmaceutical statistical computing for Python.

Built on top of pystatistics for the general statistical computing layer.
PyStatsBio provides domain-specific methods for the drug development pipeline:
dose-response modeling, sample size/power, diagnostic accuracy, and pharmacokinetics.

Usage:
    from pystatsbio import power, doseresponse, diagnostic, pk
"""

__version__ = "1.0.0"
__author__ = "Hai-Shuo"
__email__ = "contact@sgcx.org"

from pystatsbio import diagnostic, doseresponse, pk, power

__all__ = [
    "__version__",
    "power",
    "doseresponse",
    "diagnostic",
    "pk",
]
