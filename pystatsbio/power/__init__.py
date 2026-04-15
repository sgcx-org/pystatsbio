"""
Sample size and power calculations for clinical trial planning.

Every clinical trial starts with "how many subjects do we need?" This module
provides solve-for-any-one-parameter power functions for common trial designs.

Validates against: R packages pwr, TrialSize, gsDesign, PowerTOST, samplesize.
"""

from pystatsbio.power._anova import power_anova_factorial, power_anova_oneway
from pystatsbio.power._cluster import power_cluster
from pystatsbio.power._common import PowerResult
from pystatsbio.power._crossover import power_crossover_be
from pystatsbio.power._means import power_paired_t_test, power_t_test
from pystatsbio.power._noninferiority import (
    power_equiv_mean,
    power_noninf_mean,
    power_noninf_prop,
    power_superiority_mean,
)
from pystatsbio.power._proportions import power_fisher_test, power_prop_test
from pystatsbio.power._survival import power_logrank

__all__ = [
    "PowerResult",
    "power_t_test",
    "power_paired_t_test",
    "power_prop_test",
    "power_fisher_test",
    "power_logrank",
    "power_anova_oneway",
    "power_anova_factorial",
    "power_noninf_mean",
    "power_noninf_prop",
    "power_equiv_mean",
    "power_superiority_mean",
    "power_crossover_be",
    "power_cluster",
]
