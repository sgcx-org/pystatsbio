# Changelog

## 1.1.0 (unreleased)

### batch_auc GPU — Fully Vectorized (49-63x speedup on CUDA)

Linux/NVIDIA validation on RTX 5070 Ti revealed that the GPU `batch_auc`
backend was **slower than CPU** due to sequential Python `for`/`while` loops
for tie detection. This release replaces the entire ranking kernel.

#### Changed

- **`batch_auc(backend='gpu')`**: Replaced sequential per-marker Python loops
  with fully vectorized tie detection using `diff` + `cumsum` for group IDs
  and `scatter_add_` for midrank computation. Zero Python loops touch GPU
  tensors.
- **Benchmarks** (RTX 5070 Ti, 1,155 TCGA BRCA samples):
  - 1,000 markers: 63x speedup (CPU 0.18s, GPU 0.003s)
  - 5,000 markers: 63x speedup (CPU 0.92s, GPU 0.015s)
  - 20,000 markers: 49x speedup (CPU 3.6s, GPU 0.074s)
  - Previous: GPU was 2-5x *slower* than CPU at all scales

#### MPS (Apple Silicon) — Fail Fast

- **`batch_auc(backend='gpu')` on MPS now raises `RuntimeError`** instead of
  silently running ~1000x slower than CPU. Metal's `scatter_add_` does not
  handle the sparse scatter pattern used by the vectorized midrank kernel
  efficiently (tested at 1350x slower on M2 Ultra with 5K markers).
- `backend='auto'` on Apple Silicon now correctly routes to CPU.
- `backend='gpu'` on Apple Silicon fails loud with an actionable error message.

#### Dose-response improvements (Mac session)

- *(Changes from Mac session — to be detailed by that session)*

## 1.0.1

Bug fixes to match R reference implementations.

### Fixed

- **`power_crossover_be`**: TOST alpha convention now matches R PowerTOST — `alpha`
  is the per-test significance level (each one-sided test at `alpha`, producing a
  `1 − 2α` confidence interval). Previously `alpha` was incorrectly split as
  `alpha/2` per test, resulting in overly conservative sample sizes.

- **`roc` DeLong CI**: AUC confidence interval now uses the normal (Wald) interval
  on the original scale, matching R pROC `ci.auc(method="delong")`. Previously used
  a logit-transformed interval.

- **`ec50` CI**: Confidence interval now uses the t-distribution with residual
  degrees of freedom on the raw scale, matching R drc `ED(interval="delta")`.
  Previously used the normal distribution on the log scale.

## 1.0.0

Initial release.

### Modules

- **`power/`** — Sample size and power calculations for clinical trial designs:
  two-sample and paired t-tests, proportions (chi-squared, Fisher exact), log-rank
  (survival), one-way and factorial ANOVA, non-inferiority/equivalence/superiority
  for means and proportions, crossover bioequivalence (PowerTOST method), and
  cluster-randomized trials. Validated against R packages `pwr`, `TrialSize`,
  `gsDesign`, `PowerTOST`, and `samplesize`.

- **`doseresponse/`** — Dose-response modeling for preclinical pharmacology:
  4-parameter log-logistic (4PL/LL.4), 5-parameter log-logistic (5PL/LL.5),
  Weibull-1, Weibull-2, Brain-Cousens hormesis models. EC50/IC50 estimation,
  relative potency (parallelism-tested), benchmark dose (BMD/BMDL) analysis,
  and GPU-accelerated batch fitting for high-throughput screening. Validated
  against R packages `drc` and `BMDS`.

- **`diagnostic/`** — Diagnostic accuracy analysis for biomarker evaluation:
  ROC curves with DeLong AUC and confidence intervals, Delong AUC comparison test,
  sensitivity/specificity/PPV/NPV/likelihood ratios at any threshold, optimal
  cutoff selection (Youden, min-distance, cost-weighted), and batch AUC computation
  for biomarker panel screening. Validated against R packages `pROC`,
  `OptimalCutpoints`, and `epiR`.

- **`pk/`** — Non-compartmental pharmacokinetic analysis (NCA): AUC (linear,
  log-linear, linear-up/log-down trapezoidal), Cmax, Tmax, terminal elimination
  rate constant (lambda_z), half-life, clearance (IV and extravascular), volume of
  distribution, AUMC, and MRT. Validated against R packages `PKNCA` and
  `NonCompart`.
