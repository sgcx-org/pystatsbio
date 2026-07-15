# pystatsbioval — the pystatsbio validation harness core

Reusable, subsystem-agnostic machinery for producing **frozen validation evidence**
for `pystatsbio`: repeated timing, a uniform benchmark-record envelope, reusable
estimate summaries, the environment/reproducibility manifest (with the
PyPI-vs-editable guard), run serialization to the artifact schema, and a generic
R-subprocess bridge.

This package lives in the `pystatsbio` repo but is **not** part of the shipped
`pystatsbio` wheel (hatchling packages `pystatsbio` only). It is developed *with* the
library yet evolves on its own cadence — regenerating evidence must never require
cutting a library release. It is run against a **PyPI-installed** `pystatsbio` of the
exact version being validated (`device.require_pypi` fails loud on an editable/local
install).

It is a library-neutral vendored copy of the pystatistics `pystatsval` harness: the
env manifest records a generic `library` / `library_version` pair (default
`pystatsbio`) rather than a pystatistics-specific key, so the same machinery can
validate any `pystats*` package. If more than one vertical needs it, promoting this
to a single shared package is a separate, deliberately-authorized task.

## Modules

| Module | Job |
|--------|-----|
| `timing` | `time_call(fn, warmup, reps)` → wall-clock summary (median/min/max) |
| `measure` | device-aware timing + peak memory (GPU sync, RSS high-water) |
| `record` | `make_record(...)` → one flat, cross-engine-comparable benchmark record |
| `estimates` | reusable numeric summaries (e.g. `summarize_covariance`) |
| `device` | env/reproducibility manifest + `detect_install_source` / `require_pypi` |
| `serialize` | freeze a run to `validation-run/v1` JSON |
| `rrunner` | generic R-subprocess bridge (timing done inside R) |

## Usage

Subsystem drivers live in the **`pystatsbio-validation`** repo and import this
package (installed editable/path):

```bash
pip install -e /path/to/pystatsbio/validation
```

Run the harness tests:

```bash
cd pystatsbio/validation && pytest
```
