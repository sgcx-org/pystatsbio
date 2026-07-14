"""Regression tests against R metafor on the BCG log-odds-ratio dataset.

Locks in two fixes from the 4.0.0 validation bundle:

* I2/H2 are estimator-specific (derived from each method's own tau2), matching
  ``metafor::rma``. Previously the Q-based (DerSimonian-Laird) I2/H2 were reported
  for every estimator, so REML/PM reported the wrong heterogeneity statistics.
* The REML tau2 standard error uses the expected (Fisher) information, matching
  ``metafor::rma(method="REML")$se.tau2``.

Reference values are from metafor on ``escalc(measure="OR", data=dat.bcg)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pystatsbio import meta

# escalc(measure="OR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
_YI = np.array([-0.938694, -1.666191, -1.386294, -1.456444, -0.219141, -0.958122,
                -1.633776, 0.012021, -0.471746, -1.40121, -0.34085, 0.446635,
                -0.017342])
_VI = np.array([0.357125, 0.208132, 0.433413, 0.020314, 0.051952, 0.009905,
                0.22701, 0.004007, 0.056977, 0.075422, 0.012525, 0.534162, 0.071635])

# metafor::rma reference (method -> I2, H2, tau2)
_METAFOR = {
    "DL":   {"I2": 92.645478, "H2": 13.597076},
    "REML": {"I2": 92.072691, "H2": 12.614621, "se_tau2": 0.178401},
    "PM":   {"I2": 92.146188, "H2": 12.732671},
}


@pytest.mark.parametrize("method", list(_METAFOR))
def test_i2_h2_match_metafor(method):
    """I2/H2 match metafor for each estimator (not the DL value for all)."""
    r = meta.rma(_YI, _VI, method=method)
    assert r.I2 == pytest.approx(_METAFOR[method]["I2"], rel=1e-4)
    assert r.H2 == pytest.approx(_METAFOR[method]["H2"], rel=1e-4)


def test_reml_i2_differs_from_dl():
    """Regression: REML I2 must be its own value, not DL's (the 3.0.0 bug)."""
    reml = meta.rma(_YI, _VI, method="REML")
    dl = meta.rma(_YI, _VI, method="DL")
    assert reml.I2 != pytest.approx(dl.I2, rel=1e-6)
    assert reml.I2 == pytest.approx(92.072691, rel=1e-4)


def test_reml_tau2_se_matches_metafor():
    """REML tau2 SE from expected information matches metafor."""
    r = meta.rma(_YI, _VI, method="REML")
    assert r.tau2_se == pytest.approx(_METAFOR["REML"]["se_tau2"], rel=1e-4)


def test_moment_estimators_tau2_se_is_none():
    """DL/PM tau2 SE is deliberately not provided (documented scope decision)."""
    assert meta.rma(_YI, _VI, method="DL").tau2_se is None
    assert meta.rma(_YI, _VI, method="PM").tau2_se is None
