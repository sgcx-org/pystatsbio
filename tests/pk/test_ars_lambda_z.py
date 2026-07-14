"""Regression test for the lambda_z terminal-window selection (4.0.0 bundle).

Auto-selection uses the "Adjusted R-squared Best Fit" (ARS) rule from Phoenix
WinNonlin / R NonCompart::sNCA: among terminal windows, choose the one with the
most points whose adjusted R-squared is within 0.0001 of the maximum — not the
strict argmax. On Theophylline subject 6 the strict argmax picked 3 points while
NonCompart's ARS picks 7; pystatsbio now matches (n_terminal == 7).
"""

from __future__ import annotations

import numpy as np

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
