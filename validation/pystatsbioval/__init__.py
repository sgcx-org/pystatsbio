"""pystatsbioval — the pystatsbio validation harness core.

Reusable, subsystem-agnostic machinery for producing FROZEN validation evidence:
repeated timing, a uniform benchmark-record envelope, reusable estimate summaries,
the environment/reproducibility manifest (with the PyPI-vs-editable guard), run
serialization to the artifact schema, and a generic R-subprocess bridge.

This package lives in the pystatsbio repo but is NOT part of the shipped
``pystatsbio`` wheel: it is developed *with* the library yet evolves on its own
cadence (regenerating evidence must never require cutting a library release). It is
run against a PyPI-installed ``pystatsbio`` of the exact version being validated.

It is a vendored, library-neutral copy of the pystatistics ``pystatsval`` harness
(the env manifest emits a generic ``library``/``library_version`` pair rather than a
pystatistics-specific key). Subsystem-specific drivers (which estimator to fit, which
R script to call) live in the pystatsbio-validation repo's ``drivers/`` and import
this package.
"""

from __future__ import annotations

from . import device, estimates, measure, record, rrunner, serialize, timing

__all__ = ["timing", "measure", "record", "estimates", "device", "serialize", "rrunner"]
