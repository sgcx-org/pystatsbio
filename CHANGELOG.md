# Changelog

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
