# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py <version>`
> That script uses this file to build the CHANGELOG entry, bumps versions
> everywhere, and resets this file for the next cycle.

## Changes

- **`gee()`: GPU precision now lives in the `backend=` string; `use_fp64` is
  removed.** Mirroring the pystatistics 4.0 convention, `gee(..., backend=)`
  now accepts `'cpu'` (float64), `'gpu'` (float32), `'gpu_fp64'` (CUDA float64,
  raises on Apple Silicon/MPS), or `'auto'`. The separate `use_fp64=` keyword
  is gone — pass `backend='gpu_fp64'` for double precision. Breaking change for
  callers that passed `use_fp64=`. The GPU-unavailable / fp64-on-MPS errors now
  reuse pystatistics' canonical messages (`core.compute.backend`).
- **Requires `pystatistics>=4.0`** (was `>=0.1.0`): uses the 4.0 backend-string
  precision convention and `core.compute.backend`.

### 2.0 — the consistency release (in progress)

A library-wide interface-consistency pass adopting the pystatistics constitution
(`pystatsbio/CONVENTIONS.md`). Breaking interface changes; statistical results
unchanged.

- **New: `pystatsbio/CONVENTIONS.md`** — adopts pystatistics' constitution as
  binding and adds amendments B1–B5 (power symbols descriptivized, `*Solution`
  result envelope, reuse of `pystatistics.core.exceptions`, batch GPU paths on
  the backend convention, descriptive epi measure values) plus the migration
  table.
- **Exceptions (B3): no module raises a bare `ValueError`/`RuntimeError` for
  validation.** All 161 validation `raise ValueError` sites now raise
  `ValidationError` (which subclasses `ValueError`, so `except ValueError` still
  works). Optimizer failures (meta REML / Paule-Mandel) raise `ConvergenceError`;
  BMD numerical failures raise `NumericalError`; `pk.LambdaZEstimationError` now
  subclasses `ConvergenceError` (was `ValueError`). GPU-environment failures stay
  `RuntimeError`.
- **Backend (B4): `batch_auc` and `fit_drm_batch` route `backend=` through the
  pystatistics resolver.** Precision lives in the string — `'gpu'` is float32,
  `'gpu_fp64'` is CUDA float64; unknown backends are rejected with the canonical
  message; an unavailable GPU fails loud instead of silently using CPU-torch.
  `batch_auc` rejects MPS (slow scatter); `fit_drm_batch` runs float32 on MPS.
- **Naming (B1/B5, breaking):** power-analysis parameters are descriptivized —
  `d`/`f`/`h` → `effect_size`, `k` → `n_groups`, `hr` → `hazard_ratio`,
  `cv` → `coef_variation`, `sd` → `std`, `p1`/`p2` → `prop1`/`prop2`, and
  `power_t_test(type=)` → `test_type` (kept `n`, `icc`). Option values lose their
  dots: `alternative="two-sided"`/`"one-sided"`, `test_type="two-sample"`/
  `"one-sample"`, anova `effect="main-a"`/`"main-b"`. `epi.mantel_haenszel`'s
  `measure` takes `"odds-ratio"`/`"risk-ratio"` (was `"OR"`/`"RR"`).
