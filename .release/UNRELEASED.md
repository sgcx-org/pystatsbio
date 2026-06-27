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
- **Result envelope (B2, breaking): every public return is now a `…Solution`
  wrapping `Result[…Params]`.** Across all seven modules the 17 former
  `…Result`/bare result dataclasses are renamed to `…Params` (the immutable
  computed payload) and are no longer returned directly; each public function
  returns a `…Solution` that exposes every former attribute and `summary()` as
  read-only properties plus the uniform `.backend_name` / `.timing` /
  `.warnings` / `.info` accessors and a Jupyter `_repr_html_`. This includes
  `gee()` → `GEESolution` (`backend_name` tags the device, e.g.
  `'gee_gpu (cuda, fp32)'`) and `meta` (`rma()` → `MetaSolution`). Nested value
  objects (`EpiMeasure`, dose-response `CurveParams`) are unchanged. The
  per-module details follow.
- **Result envelope (B2, breaking): `pk.nca()` now returns `NCASolution`
  wrapping `Result[NCAParams]`.** The frozen `NCAResult` dataclass is renamed
  `NCAParams` (the computed payload, all fields unchanged) and is no longer the
  public return. `nca()` returns `NCASolution`, which exposes every PK parameter
  (`cmax`, `tmax`, `auc_last`, `auc_inf`, `half_life`, `lambda_z`, `clearance`,
  `vz`, ...) and `summary()` as read-only properties, plus the uniform
  `.backend_name` (`'cpu'`), `.timing`, `.warnings`, and `.info` accessors and a
  Jupyter `_repr_html_`. `.info` carries the route, AUC method, point count, and
  lambda_z diagnostics (estimated?, n terminal points, adjusted r-squared).
  `pystatsbio.pk` now exports `NCASolution`/`NCAParams` (was `NCAResult`);
  `LambdaZEstimationError` is unchanged. Numeric results are identical.
- **Result envelope (B2, breaking): every `power.power_*()` function now returns
  `PowerSolution` wrapping `Result[PowerParams]`.** The frozen `PowerResult`
  dataclass is renamed `PowerParams` (the computed payload — `n`, `power`,
  `effect_size`, `alpha`, `alternative`, `method`, `note`, all unchanged) and is
  no longer the public return. `PowerSolution` exposes every computed value and
  `summary()` as read-only properties, plus the uniform `.backend_name`
  (`'cpu'`), `.timing`, `.warnings`, and `.info` (carries `method`) accessors and
  a Jupyter `_repr_html_`. `pystatsbio.power` now exports
  `PowerSolution`/`PowerParams` (was `PowerResult`). Affects `power_t_test`,
  `power_paired_t_test`, `power_prop_test`, `power_fisher_test`, `power_logrank`,
  `power_anova_oneway`, `power_anova_factorial`, `power_noninf_mean`,
  `power_noninf_prop`, `power_equiv_mean`, `power_superiority_mean`,
  `power_crossover_be`, and `power_cluster`. Numeric results are identical.
- **Result envelope (B2, breaking): the three top-level `epi` results now
  return `*Solution` objects wrapping `Result[*Params]`.** `epi_2by2()` returns
  `Epi2x2Solution` (was `Epi2x2Result`), `mantel_haenszel()` returns
  `MantelHaenszelSolution` (was `MantelHaenszelResult`), and `rate_standardize()`
  returns `StandardizedRateSolution` (was `StandardizedRate`). Each frozen
  dataclass is renamed to `*Params` (the computed payload, all fields unchanged)
  and is no longer the public return. Each `*Solution` exposes every former
  attribute and `summary()` as read-only properties, plus the uniform
  `.backend_name` (`'cpu'`), `.timing`, `.warnings`, and `.info` accessors and a
  Jupyter `_repr_html_`. The nested `EpiMeasure` value object (the estimate + CI
  carried in fields like `risk_ratio` / `pooled_estimate`) is unchanged.
  `pystatsbio.epi` now exports the `*Solution`/`*Params` names and still exports
  `EpiMeasure`. Numeric results are identical.
- **Result envelope (B2, breaking): the five `diagnostic` results now return
  `*Solution` objects wrapping `Result[*Params]`.** `roc()` returns
  `ROCSolution` (was `ROCResult`), `roc_test()` returns `ROCTestSolution` (was
  `ROCTestResult`), `diagnostic_accuracy()` returns `DiagnosticSolution` (was
  `DiagnosticResult`), `optimal_cutoff()` returns `CutoffSolution` (was
  `CutoffResult`), and `batch_auc()` returns `BatchAUCSolution` (was
  `BatchAUCResult`). Each frozen dataclass is renamed to `*Params` (the computed
  payload, all fields unchanged) and is no longer the public return. Each
  `*Solution` exposes every former attribute and `summary()` as read-only
  properties, plus the uniform `.backend_name`, `.timing`, `.warnings`, and
  `.info` accessors and a Jupyter `_repr_html_`. `backend_name` is `'cpu'` for
  the CPU paths (`roc`, `roc_test`, `diagnostic_accuracy`, `optimal_cutoff`, and
  CPU `batch_auc`) and `'batch_auc_gpu (<device>)'` for the GPU `batch_auc`
  path. `pystatsbio.diagnostic` now exports the `*Solution`/`*Params` names (the
  old `*Result` names are removed). Numeric results are identical.
- **Result envelope (B2, breaking): the five `doseresponse` results now return
  `*Solution` objects wrapping `Result[*Params]`.** `fit_drm()` returns
  `DoseResponseSolution` (was `DoseResponseResult`), `fit_drm_batch()` returns
  `BatchDoseResponseSolution` (was `BatchDoseResponseResult`), `ec50()` returns
  `EC50Solution` (was `EC50Result`), `relative_potency()` returns
  `RelativePotencySolution` (was `RelativePotencyResult`), and `bmd()` returns
  `BMDSolution` (was `BMDResult`). Each frozen dataclass is renamed to `*Params`
  (the computed payload, all fields unchanged) and is no longer the public
  return. Each `*Solution` exposes every former attribute and method
  (`fit_drm`'s `predict()`/`summary()`) as read-only properties, plus the
  uniform `.backend_name`, `.timing`, `.warnings`, and `.info` accessors and a
  Jupyter `_repr_html_`. `DoseResponseSolution.params` still returns the fitted
  `CurveParams`, so existing `.params.ec50` / `.params.to_array()` access is
  unchanged; the payload stores the curve under a `curve` field internally.
  `CurveParams` (the fitted-curve value object) is unchanged. `backend_name` is
  `'cpu'` for `fit_drm`, `ec50`, `relative_potency`, `bmd`, and CPU
  `fit_drm_batch`, and `'fit_drm_batch_gpu (<device>)'` for the GPU
  `fit_drm_batch` path. `pystatsbio.doseresponse` now exports the
  `*Solution`/`*Params` names (the old `*Result` names are removed). Numeric
  results are identical.
- **Naming (B1/B5, breaking):** power-analysis parameters are descriptivized —
  `d`/`f`/`h` → `effect_size`, `k` → `n_groups`, `hr` → `hazard_ratio`,
  `cv` → `coef_variation`, `sd` → `std`, `p1`/`p2` → `prop1`/`prop2`, and
  `power_t_test(type=)` → `test_type` (kept `n`, `icc`). Option values lose their
  dots: `alternative="two-sided"`/`"one-sided"`, `test_type="two-sample"`/
  `"one-sample"`, anova `effect="main-a"`/`"main-b"`. `epi.mantel_haenszel`'s
  `measure` takes `"odds-ratio"`/`"risk-ratio"` (was `"OR"`/`"RR"`).
