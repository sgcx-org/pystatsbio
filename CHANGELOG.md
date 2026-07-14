# Changelog

## 4.0.0

### Summary

4.0.0 is a correctness bundle from the first full validation sweep of every
subsystem against its R reference (drc, PKNCA/NonCompart, pROC/epiR, pwr/PowerTOST,
epiR/epitools, metafor, geepack). It fixes several silently-wrong results in the
dose-response and epidemiology modules and a heterogeneity-statistic bug in
meta-analysis, plus documentation corrections. Results now agree with the R
references to their stated tolerances. It is a **major** release because one fix
changes a public signature (see Breaking changes).

### Breaking changes

- **`epi.rate_standardize(method="indirect")` now requires a `standard_weights`
  argument** (the standard population's stratum sizes) to compute the standardized
  rate, and raises `ValidationError` without it. Previously the standardized rate
  was silently wrong (it collapsed to the crude study rate); computing it correctly
  needs the standard population's age distribution, which the old signature did not
  carry. The SIR (the primary indirect output) is unaffected. Direct
  standardization is unchanged.

### Fixed

- **`doseresponse.ec50()` now returns the true EC50/ED50 for asymmetric models.**
  It previously returned the model's raw `e` location parameter, which is the
  half-maximal dose only for the symmetric LL.4 model; for LL.5/W1.4/W2.4/BC.5 it
  was off by 9–27% (e.g. LL.5 on `drc::ryegrass`: 2.21 vs the correct 3.02).
  `ec50()` now solves the fitted curve for the dose at the half-maximal response,
  matching `drc::ED(type="relative")` to <2e-5, and its confidence-interval SE
  comes from the delta method applied to the solved ED50 (matching
  `drc::ED(interval="delta")`). LL.4 is unchanged (ED50 == e).
  (`pystatsbio/doseresponse/_potency.py`)
- **`doseresponse.fit_drm(model="W2.4")` no longer converges to an inferior local
  optimum on decreasing data.** The data-driven self-start seeded the wrong basin,
  silently returning a ~14%-worse RSS with swapped asymptotes (RSS 6.02 vs drc's
  5.29 on `ryegrass`). An auto-start W2.4 fit now also tries the mirror start and
  keeps the lower-RSS result, recovering the natural-label global optimum.
  Multistart never worsens a fit. (`pystatsbio/doseresponse/_fit.py`)
- **`epi.epi_2by2` population attributable fraction now uses the exposure
  prevalence.** Levin's PAF was computed with the disease prevalence `(a+c)/n`
  instead of the exposure prevalence `(a+b)/n`, giving a value matching no
  standard estimand (e.g. 0.286 instead of the correct 0.500 on a table with
  RR=3 and 50% exposed). (`pystatsbio/epi/_measures.py`)
- **`epi.rate_standardize(method="indirect")` now returns the correct
  standardized rate.** The adjusted rate weighted the standard rates by the
  *study* person-time, which algebraically collapsed it to the crude study rate.
  It now takes a new **`standard_weights`** argument (the standard population's
  stratum sizes, matching `epitools::ageadjust.indirect`'s `stdpop`) and returns
  `SIR × standard-population crude rate`, matching epitools to ~1e-13. The
  `indirect` method now requires `standard_weights` and fails loud without it
  (the SIR itself is unaffected). (`pystatsbio/epi/_standardize.py`)
- **`meta.rma` now reports estimator-specific I² and H².** They were computed
  from Cochran's Q (the DerSimonian-Laird value) for every estimator, so
  `method="REML"` (the default) and `method="PM"` reported DL's heterogeneity
  statistics instead of their own (e.g. REML I² 92.65 instead of 92.07 on the
  BCG dataset). I²/H² now come from each method's tau² via the "typical"
  within-study variance, matching `metafor::rma`; DL is unchanged. Also,
  `method="REML"`'s tau² standard error now uses the expected (Fisher)
  information, matching `metafor` to ~1e-8. (`pystatsbio/meta/_random.py`)
- **`pk.nca` terminal-slope (lambda_z) auto-selection now uses the WinNonlin/
  NonCompart "Adjusted R-squared Best Fit" (ARS) rule.** It previously took the
  strict argmax of the adjusted R², which on some profiles selects fewer terminal
  points than the pharmacometrics standard (e.g. 3 vs 7 points on one
  Theophylline subject, a ~4% half-life difference). It now chooses the window
  with the most points whose adjusted R² is within 0.0001 of the maximum,
  matching `NonCompart::sNCA` to machine precision across the Theophylline
  dataset. (`pystatsbio/pk/_nca.py`)

### Documentation

These correct descriptions and references where the code was already correct and
R-matching; no numeric behaviour changed.

- `doseresponse.ec50()`: the confidence interval is a raw-scale symmetric Wald
  interval (delta method), not a log-scale interval.
- `diagnostic.roc()` AUC CI: described as a symmetric Wald interval on the AUC
  scale (matching `pROC::ci.auc(method="delong")`), not "logit-transformed".
- `power.power_crossover_be()`: documents that power uses the noncentral-t
  approximation (PowerTOST `method="nct"`), which differs from PowerTOST's default
  `exact` (Owen's Q) in power by up to ~1e-2 at low power/high CV; sample size is
  unaffected.
- README power R-reference table: corrected three entries that named R functions
  which do not exist in the cited packages (`power_anova_factorial`,
  `power_superiority_mean`, `power_cluster`) and re-pointed `power_fisher_test` and
  `power_crossover_be` to the references they actually match.
- `gee`: documents that `.scale`/`.alpha` use a degrees-of-freedom correction
  (matching statsmodels; differs from geepack's uncorrected moments by <1-2%),
  while coefficients and robust SE match geepack.
- `epi.rate_standardize` (direct): clarified that the CI is a normal approximation
  (the point estimate and variance match `epitools::ageadjust.direct`; the
  Fay-Feuer gamma interval endpoints differ).


## 3.0.0

### Summary

3.0 tracks the PyStatistics 5.0 API. PyStatsBio's statistical results are
unchanged; this is a breaking release because the required PyStatistics floor
moves to 5.0 (the 4.x API it was built against is gone) and one relayed value
changes.

### Changed

- **Requires `pystatistics>=5.0`** (was `>=4.0`). PyStatistics 5.0 hard-renamed
  ~40 public names and removed the 4.x surface with no shim, so PyStatsBio
  cannot run on 4.x. PyStatsBio's own source needed no code changes: its only
  coupling to PyStatistics is through internal infrastructure
  (`core.exceptions`, `core.result`, `core.compute.{backend,device,tolerances}`,
  `regression.families`), all of which 5.0 left stable. Verified by running the
  full test suite against `pystatistics==5.0.0` from PyPI. (`pyproject.toml`)
- **GLM/GEE family names are lowercase.** `gee(..., family="gamma").family_name`
  now returns `"gamma"` instead of `"Gamma"`, matching the sibling families
  (`"gaussian"`, `"binomial"`, `"poisson"`). The value is relayed straight from
  PyStatistics' `Family.name`, which 5.0 changed for naming consistency
  (`GammaFamily`→`Gamma`, `.name` `'Gamma'`→`'gamma'`); PyStatsBio's GEE code
  (`gee/__init__.py`) was already correct and needed no change. Updated the one
  stale test assertion in `tests/gee/test_gee.py`
  (`TestBasicFitting::test_gamma_family`).

### Fixed

- **README dose-response model identifier corrected**: the `doseresponse`
  models list advertised `BC.4` for the Brain-Cousens hormesis model, but
  `fit_drm`/`fit_drm_batch` only accept `BC.5` (the shipped model is the
  5-parameter Brain-Cousens). Following the README verbatim raised a
  `ValidationError`. The README now lists `BC.5`; the code is unchanged.
- `docs/conf.py` `version`/`release` were stuck at `0.1.0` (never tracked the
  package version); set to `3.0.0`.
- `pystatsbio/CONVENTIONS.md` §0 now cites the pystatistics **5.0** constitution
  and its amendments **A1–A14** (was "4.0" / "A1–A5").


## 2.0.0 — the consistency release

A library-wide pass that aligns PyStatsBio with the PyStatistics 4.0 API
conventions. **This release contains breaking interface changes** — parameters,
option values, and result classes were renamed and old spellings removed (no
alias). **No statistical or numerical behavior changed.** Requires
`pystatistics>=4.0`.

#### Result objects (breaking)

- **Every public function now returns a `…Solution`** wrapping an immutable
  result payload. Each Solution exposes the former result fields as read-only
  properties (so `result.n`, `result.auc`, `result.coefficients`, etc. keep
  working) plus the uniform `.backend_name`, `.timing`, `.warnings`, and `.info`
  accessors and a Jupyter `_repr_html_`. Renamed: `PowerResult` →
  `PowerSolution`, `GEEResult` → `GEESolution`, `MetaResult` → `MetaSolution`,
  `NCAResult` → `NCASolution`, the diagnostic `ROCResult` / `ROCTestResult` /
  `DiagnosticResult` / `CutoffResult` / `BatchAUCResult`, the dose-response
  `DoseResponseResult` / `BatchDoseResponseResult` / `EC50Result` /
  `RelativePotencyResult` / `BMDResult`, and the epi `Epi2x2Result` /
  `MantelHaenszelResult` / `StandardizedRate` → their `*Solution` equivalents.
- `ec50()` results expose the EC50 as `.estimate` (was `.ec50`); the fitted
  curve coefficients remain available as `fit.params` (a `CurveParams`).

#### Power-analysis parameters (breaking)

Descriptive names replace the single-letter symbols:

| Old | New |
|---|---|
| `d` / `f` / `h` (effect sizes) | `effect_size` |
| `k` | `n_groups` |
| `hr` | `hazard_ratio` |
| `cv` | `coef_variation` |
| `sd` | `std` |
| `p1` / `p2` | `prop1` / `prop2` |
| `power_t_test(type=)` | `test_type` |

`n` (sample size) and `icc` are unchanged.

#### Option values (breaking)

- `alternative`: `"two-sided"` / `"one-sided"` (were `"two.sided"` /
  `"one.sided"`).
- `test_type`: `"two-sample"` / `"one-sample"` (were `"two.sample"` /
  `"one.sample"`).
- `power_anova_factorial(effect=)`: `"main-a"` / `"main-b"` (were `"main_A"` /
  `"main_B"`).
- `epi.mantel_haenszel(measure=)`: `"odds-ratio"` / `"risk-ratio"` (were
  `"OR"` / `"RR"`).

#### Backend & precision (breaking)

- GPU precision now lives in the `backend=` string: `"cpu"` (float64), `"gpu"`
  (float32), `"gpu_fp64"` (CUDA float64), or `"auto"`. The separate `use_fp64`
  flag is removed — pass `backend="gpu_fp64"` for double precision. Applies to
  `gee`, `diagnostic.batch_auc`, and `doseresponse.fit_drm_batch`.
- The batch GPU paths now reject unknown backend strings with a clear message
  and fail loudly when a GPU is requested but unavailable (instead of silently
  falling back). `batch_auc` is CUDA-only; `fit_drm_batch` runs float32 on
  Apple Silicon.

#### Errors

- Input validation raises `ValidationError` (which subclasses the builtin
  `ValueError`, so existing `except ValueError` code keeps working);
  non-convergence raises `ConvergenceError`; numerical failures raise
  `NumericalError`; an unavailable GPU raises `RuntimeError`.


## 1.6.1

### Changed
- Promoted the PyPI Development Status classifier from "3 - Alpha" to
  "5 - Production/Stable", reflecting the maturity of the seven shipped,
  R-validated modules.


## 1.6.0

- **GEE GPU backend** — `pystatsbio/gee/backends/gpu_fit.py`,
  `backends/_gpu_correlation.py`, `backends/_gpu_family.py`. The
  CPU GEE iterates over clusters in a Python loop, solving per-
  cluster working-covariance systems one at a time. The GPU backend
  groups clusters by size and batches the `(K_s, s, s)` working-
  covariance solves into a single `torch.linalg.solve`, then reduces
  the bread/score/sandwich-meat contributions across all clusters in
  vectorized tensor ops. The wins grow with cluster count:

    | shape (K, m, p)           | CPU      | GPU (numpy per-call) | GPU (DataSource) | speedup |
    |---------------------------|---------:|---------------------:|-----------------:|--------:|
    | geepack::dietox-like 72×12×4 | 7.1 ms  | 2.9 ms              | 2.9 ms          | 2.4×   |
    | K=100 m=5 p=4             | 11.8 ms  | 4.0 ms              | 4.0 ms          | 2.9×   |
    | K=500 m=5 p=4             | 57.9 ms  | 3.7 ms              | 3.8 ms          | 15.4×  |
    | K=1000 m=5 p=4            | 115.5 ms | 3.8 ms              | 3.9 ms          | 30.0×  |
    | K=2000 m=8 p=6            | 173.3 ms | 3.9 ms              | 4.0 ms          | 43.5×  |
    | K=5000 m=10 p=8           | 454.2 ms | 6.0 ms              | 6.7 ms          | 67.5×  |

  Supports all four CPU correlation structures (independence,
  exchangeable, AR(1), unstructured) and the four supported GLM
  families (gaussian/identity, binomial/logit, poisson/log,
  gamma/inverse). Unequal cluster sizes handled via size-grouped
  batching (one batched solve per distinct cluster size). Default
  precision is FP32 (`use_fp64=False`); FP64 available on CUDA for
  machine-precision parity with CPU.

- **GEE accepts torch.Tensor input** — `gee()` now accepts either
  numpy arrays or `torch.Tensor` for `y`, `X`, and `cluster_id`.
  Device-resident tensors (from `DataSource.from_arrays(...).to('cuda')`)
  skip per-fit H2D transfer of the design matrix, following the same
  convention as `pca()` and `multinom()`. Tensor input infers
  `backend='gpu'`; explicit `backend='cpu'` with a GPU tensor raises
  (Rule 1: no silent device migration).

- **Two-tier validation for GEE** — CPU path remains validated
  against R `geepack::geeglm()`. GPU path is validated against CPU at
  the `GPU_FP32` tolerance tier (rtol = 1e-4, atol = 1e-5) on
  coefficients and robust SE; CUDA FP64 matches CPU to machine
  precision. `TestGeeGPU` in `tests/gee/test_gee.py` adds the standard
  7 GPU-backend tests mirroring the existing `TestMultinomGPU` and
  `TestPCAGPU` suites.


## 1.5.0

### Summary

Three new modules covering epidemiology, meta-analysis, and longitudinal data
analysis. The `gee` module establishes the first real cross-package dependency
on PyStatistics. Adds 190 new tests.

### Added

#### Epidemiology Module

- **`epi_2by2(table)`** — RR, OR, RD, attributable fraction, PAF, NNT from 2×2
  tables. Validates against R `epiR::epi.2by2()`.
- **`rate_standardize(counts, person_time, standard_pop)`** — Direct and indirect
  age-standardization. Validates against R `epitools`.
- **`mantel_haenszel(tables, measure='OR')`** — MH pooled OR/RR, CMH chi-squared,
  Breslow-Day homogeneity test. Validates against R `stats::mantelhaen.test()`.

#### Meta-Analysis Module

- **`rma(yi, vi, method='REML')`** — Inverse-variance weighted meta-analysis
  matching R `metafor::rma()`. Methods: FE, DerSimonian-Laird, REML,
  Paule-Mandel. Heterogeneity: Cochran's Q, I², H², tau² with SE.

#### GEE Module

- **`gee(y, X, cluster_id, family='gaussian', corr_structure='exchangeable')`**
  — Generalized Estimating Equations matching R `geepack::geeglm()`. First
  PyStatsBio module to import Family/Link from `pystatistics.regression`.
  Working correlations: independence, exchangeable, AR(1), unstructured.
  Sandwich (robust) variance estimator.

### Tests

190 new tests. Total: 633.

## 1.1.0

### Summary

Fully vectorized GPU `batch_auc` kernel (49-63x speedup on CUDA), 19x faster CPU dose-response fitting via MINPACK LM optimizer with analytical Jacobians (3.3x faster than R drc), and bug fixes from 1.0.1.

### Changed

- **`batch_auc(backend='gpu')`**: Replaced sequential per-marker Python loops with fully vectorized tie detection using `diff` + `cumsum` for group IDs and `scatter_add_` for midrank computation. Zero Python loops touch GPU tensors.
- **`batch_auc(backend='gpu')` on MPS**: Now raises `RuntimeError` instead of silently running ~1000x slower than CPU. Metal's `scatter_add_` does not handle the sparse scatter pattern used by the vectorized midrank kernel efficiently (tested at 1350x slower on M2 Ultra with 5K markers).
- **`batch_auc(backend='auto')` on Apple Silicon**: Now correctly routes to CPU.
- **`fit_drm` optimizer**: Uses `method='lm'` (MINPACK `lmder` Fortran routine) instead of `method='trf'`. The entire Levenberg-Marquardt iteration loop runs in compiled Fortran, eliminating Python-level overhead (~150 iterations x function call overhead per fit). Falls back to TRF only when custom bounds or weights are explicitly requested.
- **LL.4 analytical Jacobian**: Closed-form derivatives replace 2-point numerical finite differences. Eliminates 4x redundant function evaluations per Jacobian computation. Uses `scipy.special.expit` for numerically stable sigmoid.
- **log(ec50) reparameterization**: Fits `log(ec50)` instead of `ec50`, removing the positivity bound (`ec50 > 0` is automatic via `exp`). Enables `method='lm'` which does not support bounds. Jacobian column transformed back to natural scale for correct SE computation.
- **Fail-fast on hopeless data**: `max_nfev=200` for the LM path. Converged fits use ~20-50 evaluations; if 200 is not enough, the data has no dose-response signal. Prevents burning 2000 evaluations on flat/inactive compounds.

### Fixed

- **`power_crossover_be`**: TOST alpha convention now matches R PowerTOST.
- **`roc` DeLong CI**: Uses normal (Wald) interval on original scale, matching R pROC `ci.auc(method="delong")`.
- **`ec50` CI**: Uses t-distribution with residual df on raw scale, matching R drc `ED(interval="delta")`.

### Performance

- **`batch_auc` GPU** (RTX 5070 Ti, 1,155 TCGA BRCA samples):
  - 1,000 markers: 63x speedup (CPU 0.18s, GPU 0.003s)
  - 5,000 markers: 63x speedup (CPU 0.92s, GPU 0.015s)
  - 20,000 markers: 49x speedup (CPU 3.6s, GPU 0.074s)
  - Previous: GPU was 2-5x slower than CPU at all scales
- **`fit_drm` CPU** (Tox21 AID 743083, 8,358 compounds x 8 doses):
  - Before: 433s (19 cmpd/s) — 5.8x slower than R
  - After: 22.6s (369 cmpd/s) — 3.3x faster than R
  - R drc: 74.6s (112 cmpd/s)
  - EC50 correlation vs R on active compounds: 0.978

## 1.0.1

### Summary

Bug fixes to match R reference implementations for power analysis, ROC confidence intervals, and EC50 confidence intervals.

### Fixed

- **`power_crossover_be`**: TOST alpha convention now matches R PowerTOST — `alpha` is the per-test significance level (each one-sided test at `alpha`, producing a `1 - 2a` confidence interval). Previously `alpha` was incorrectly split as `alpha/2` per test, resulting in overly conservative sample sizes.
- **`roc` DeLong CI**: AUC confidence interval now uses the normal (Wald) interval on the original scale, matching R pROC `ci.auc(method="delong")`. Previously used a logit-transformed interval.
- **`ec50` CI**: Confidence interval now uses the t-distribution with residual degrees of freedom on the raw scale, matching R drc `ED(interval="delta")`. Previously used the normal distribution on the log scale.

## 1.0.0

### Summary

Initial release of pystatsbio with modules for power analysis, dose-response modeling, diagnostic accuracy, and pharmacokinetic analysis.

### Added

- **`power/`** — Sample size and power calculations for clinical trial designs: two-sample and paired t-tests, proportions (chi-squared, Fisher exact), log-rank (survival), one-way and factorial ANOVA, non-inferiority/equivalence/superiority for means and proportions, crossover bioequivalence (PowerTOST method), and cluster-randomized trials. Validated against R packages `pwr`, `TrialSize`, `gsDesign`, `PowerTOST`, and `samplesize`.
- **`doseresponse/`** — Dose-response modeling for preclinical pharmacology: 4-parameter log-logistic (4PL/LL.4), 5-parameter log-logistic (5PL/LL.5), Weibull-1, Weibull-2, Brain-Cousens hormesis models. EC50/IC50 estimation, relative potency (parallelism-tested), benchmark dose (BMD/BMDL) analysis, and GPU-accelerated batch fitting for high-throughput screening. Validated against R packages `drc` and `BMDS`.
- **`diagnostic/`** — Diagnostic accuracy analysis for biomarker evaluation: ROC curves with DeLong AUC and confidence intervals, DeLong AUC comparison test, sensitivity/specificity/PPV/NPV/likelihood ratios at any threshold, optimal cutoff selection (Youden, min-distance, cost-weighted), and batch AUC computation for biomarker panel screening. Validated against R packages `pROC`, `OptimalCutpoints`, and `epiR`.
- **`pk/`** — Non-compartmental pharmacokinetic analysis (NCA): AUC (linear, log-linear, linear-up/log-down trapezoidal), Cmax, Tmax, terminal elimination rate constant (lambda_z), half-life, clearance (IV and extravascular), volume of distribution, AUMC, and MRT. Validated against R packages `PKNCA` and `NonCompart`.
