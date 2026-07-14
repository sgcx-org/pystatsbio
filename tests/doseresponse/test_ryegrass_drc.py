"""Regression tests against R drc on the ryegrass dataset.

Locks in two fixes from the 4.0.0 validation bundle:

* `ec50()` returns the ED50 (dose at the half-maximal response), obtained by
  solving the fitted curve — which equals the model's ``e`` parameter only for
  the symmetric LL.4 and differs for the asymmetric models. Reference values are
  ``drc::ED(fit, 50, type="relative")`` on ``drc::ryegrass``.
* `fit_drm(model="W2.4")` reaches the global optimum on decreasing data (it
  previously self-started into a ~14%-worse local minimum with swapped
  asymptotes). Reference RSS is ``drc::drm(rootl~conc, fct=W2.4())``.

Data and reference numbers are frozen from R drc 3.0-1 at 6 significant figures.
"""

from __future__ import annotations

import numpy as np
import pytest

from pystatsbio import doseresponse as dr

# drc::ryegrass (conc = herbicide concentration, rootl = root length)
_CONC = np.array([0, 0, 0, 0, 0, 0, 0.94, 0.94, 0.94, 1.88, 1.88, 1.88,
                  3.75, 3.75, 3.75, 7.5, 7.5, 7.5, 15, 15, 15, 30, 30, 30], dtype=float)
_ROOTL = np.array([7.58, 8, 8.3286, 7.25, 7.375, 7.9625, 8.3556, 6.9143, 7.75,
                   6.8714, 6.45, 5.9222, 1.925, 2.8857, 4.2333, 1.1875, 0.8571,
                   1.0571, 0.6875, 0.525, 0.825, 0.25, 0.22, 0.44], dtype=float)

# drc reference: ED50 = ED(fit, 50, type="relative"); RSS = sum(residuals(fit)^2)
_DRC = {
    "LL.4": {"ed50": 3.057955, "rss": 5.400215},
    "LL.5": {"ed50": 3.023549, "rss": 5.277047},
    "W1.4": {"ed50": 3.088964, "rss": 6.024151},
    "W2.4": {"ed50": 2.996913, "rss": 5.292565},
    "BC.5": {"ed50": 3.051705, "rss": 5.310729},
}


@pytest.mark.parametrize("model", list(_DRC))
def test_ec50_matches_drc_ed50(model):
    """ec50() equals drc::ED(50) — including the asymmetric models where e != ED50."""
    fit = dr.fit_drm(_CONC, _ROOTL, model=model)
    ec = dr.ec50(fit)
    assert ec.estimate == pytest.approx(_DRC[model]["ed50"], rel=1e-3)


def test_ec50_asymmetric_differs_from_raw_e():
    """For an asymmetric model ec50() must NOT be the raw `e` parameter (the 3.0.0 bug)."""
    fit = dr.fit_drm(_CONC, _ROOTL, model="LL.5")
    ec = dr.ec50(fit)
    # e (params.ec50) is ~2.21; the true ED50 is ~3.02 — they must differ by >5%.
    assert abs(ec.estimate - fit.params.ec50) / ec.estimate > 0.05
    assert ec.estimate == pytest.approx(_DRC["LL.5"]["ed50"], rel=1e-3)


def test_ec50_ll4_equals_raw_e():
    """For the symmetric LL.4, ED50 == e (the solve must reproduce the parameter)."""
    fit = dr.fit_drm(_CONC, _ROOTL, model="LL.4")
    ec = dr.ec50(fit)
    assert ec.estimate == pytest.approx(fit.params.ec50, rel=1e-6)


@pytest.mark.parametrize("model", list(_DRC))
def test_fit_reaches_drc_rss(model):
    """Every model reaches drc's optimum RSS (W2.4 no longer stuck in the mirror basin)."""
    fit = dr.fit_drm(_CONC, _ROOTL, model=model)
    rss = float(np.sum((_ROOTL - fit.params.predict(_CONC)) ** 2))
    assert rss == pytest.approx(_DRC[model]["rss"], rel=1e-4)


def test_w24_not_stuck_in_mirror_basin():
    """W2.4 regression: must reach RSS ~5.29, not the ~6.02 mirror local minimum."""
    fit = dr.fit_drm(_CONC, _ROOTL, model="W2.4")
    rss = float(np.sum((_ROOTL - fit.params.predict(_CONC)) ** 2))
    assert rss < 5.6  # the stuck basin was 6.024
    assert fit.params.bottom < fit.params.top  # natural asymptote labelling
