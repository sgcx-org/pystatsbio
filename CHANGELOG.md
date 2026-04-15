# Changelog

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
