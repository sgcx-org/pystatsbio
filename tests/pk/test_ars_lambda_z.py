"""Regression test for the lambda_z terminal-window selection (4.0.0 bundle).

Auto-selection uses the "Adjusted R-squared Best Fit" (ARS) rule from Phoenix
WinNonlin / R NonCompart::sNCA: among terminal windows, choose the one with the
most points whose adjusted R-squared is within 0.0001 of the maximum — not the
strict argmax. On Theophylline subject 6 the strict argmax picked 3 points while
NonCompart's ARS picks 7; pystatsbio now matches (n_terminal == 7).
"""

from __future__ import annotations

import numpy as np
import pytest

from pystatsbio import pk

# Theophylline subject 6 (from R datasets::Theoph, dose = Dose*Wt = 320 mg)
_TIME = np.array([0, 0.27, 0.58, 1.15, 2.03, 3.57, 5.00, 7.00, 9.22, 12.10, 23.85])
_CONC = np.array([0.00, 1.29, 3.08, 6.44, 6.32, 5.53, 4.94, 4.02, 3.46, 2.78, 0.92])


def test_ars_prefers_more_points_on_near_tie():
    """ARS selects the 7-point terminal window (matching NonCompart), not 3."""
    result = pk.nca(_TIME, _CONC, dose=320.0, route="ev")
    assert result.n_terminal == 7  # strict argmax gave 3 in 3.0.0


def test_lambda_z_matches_noncompart():
    """lambda_z / half-life for the 7-point ARS fit match NonCompart::sNCA."""
    result = pk.nca(_TIME, _CONC, dose=320.0, route="ev")
    # NonCompart::sNCA on this profile: LAMZ = 0.0877957, LAMZHL = 7.89496
    assert result.lambda_z == np.float64(result.lambda_z)  # finite
    assert abs(result.lambda_z - 0.0877957) / 0.0877957 < 1e-4
    assert abs(result.half_life - 7.89496) / 7.89496 < 1e-4


def test_unfittable_terminal_phase_warns_instead_of_failing_silently():
    """Regression: a profile with no elimination phase must announce itself.

    Previously LambdaZEstimationError was suppressed and lambda_z (plus every
    derived parameter) came back None with an EMPTY .warnings tuple — a silent
    failure. It must now surface the reason in .warnings.
    """
    t = np.array([0, 1, 2, 3, 4, 5.0])
    c = np.array([1, 2, 3, 4, 5, 6.0])  # monotonically rising: no terminal phase
    r = pk.nca(t, c, dose=100.0, route="ev")

    assert r.lambda_z is None
    assert r.half_life is None and r.auc_inf is None
    assert r.clearance is None and r.vz is None
    assert len(r.warnings) == 1
    assert "lambda_z could not be estimated" in r.warnings[0]


def test_all_zero_concentrations_warns():
    """The degenerate all-zero profile must also warn rather than pass silently."""
    r = pk.nca(np.array([0, 1, 2, 3.0]), np.zeros(4), dose=100.0, route="ev")
    assert r.lambda_z is None
    assert len(r.warnings) == 1
    assert "all concentrations are zero" in r.warnings[0]


def test_aumc_and_mrt_match_noncompart():
    """AUMC and MRT are computed, exposed, and match NonCompart::sNCA."""
    t = np.array([0, 0.25, 0.57, 1.12, 2.02, 3.82, 5.10, 7.03, 9.05, 12.12, 24.37])
    c = np.array([0.74, 2.84, 6.57, 10.50, 9.66, 8.58, 8.36, 7.47, 6.89, 5.94, 3.28])
    r = pk.nca(t, c, dose=319.992, route="ev", auc_method="linear-up/log-down")
    # NonCompart::sNCA (down="Log"): AUMCLST, AUMCIFO, MRTEVIFO
    assert r.aumc_last == pytest.approx(1499.12908516, rel=1e-8)
    assert r.aumc_inf == pytest.approx(4545.59280107, rel=1e-8)
    assert r.mrt == pytest.approx(21.14980455, rel=1e-8)


def test_aumc_mrt_none_without_lambda_z():
    """AUMC to last is always available; AUMC_inf / MRT are None if lambda_z isn't."""
    t = np.array([0, 1, 2, 3, 4, 5.0])
    c = np.array([1, 2, 3, 4, 5, 6.0])  # rising -> no terminal phase
    r = pk.nca(t, c, dose=100.0, route="ev")
    assert r.aumc_last > 0        # AUMClast does not need lambda_z
    assert r.aumc_inf is None and r.mrt is None
