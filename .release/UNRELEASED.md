# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py <version>`
> That script uses this file to build the CHANGELOG entry, bumps versions
> everywhere, and resets this file for the next cycle.

## Summary

Three new modules covering epidemiology, meta-analysis, and longitudinal data
analysis. The `gee` module establishes the first real cross-package dependency
on PyStatistics (importing `Family`/`Link` from `pystatistics.regression`).
Adds 190 new tests. Combined with existing modules (power, doseresponse,
diagnostic, pk), PyStatsBio now covers the core biostatistical toolkit from
clinical trial planning through evidence synthesis.

## Added

### Epidemiology Module

- **`epi_2by2(table)`** — Computes risk ratio, odds ratio, risk difference,
  attributable fraction in exposed, population attributable fraction, and
  number needed to treat from a 2×2 contingency table. Log-transformed CIs
  for RR/OR (Woolf method), Wald CIs for RD, delta method for PAF. Applies
  0.5 continuity correction for zero cells. Validates against R
  `epiR::epi.2by2()`.

- **`rate_standardize(counts, person_time, standard_pop)`** — Direct and
  indirect age-standardization of rates. Direct method uses normal
  approximation CI; indirect method computes SIR with exact Poisson CI.
  Validates against R `epitools::ageadjust.direct()` / `ageadjust.indirect()`.

- **`mantel_haenszel(tables, measure='OR')`** — Mantel-Haenszel pooled OR or
  RR across K strata. Robins-Breslow-Greenland CI for OR, Greenland-Robins CI
  for RR. Continuity-corrected CMH chi-squared test. Breslow-Day homogeneity
  test for OR. Validates against R `stats::mantelhaen.test()`.

  Files: `epi/__init__.py`, `epi/_common.py`, `epi/_measures.py`,
  `epi/_standardize.py`, `epi/_mantel_haenszel.py`.

### Meta-Analysis Module

- **`rma(yi, vi, method='REML')`** — Inverse-variance weighted meta-analysis
  of pre-computed effect sizes and variances, matching R `metafor::rma()`.
  Four estimation methods: `'FE'` (fixed-effects), `'DL'` (DerSimonian-Laird),
  `'REML'` (restricted maximum likelihood, default), `'PM'` (Paule-Mandel).

- **Heterogeneity statistics** — Cochran's Q with chi-squared p-value, I² (%)
  quantifying between-study variability, H² ratio. REML provides tau² SE via
  observed Fisher information.

- Returns `MetaResult` frozen dataclass with pooled estimate, SE, CI, z-value,
  p-value, tau², I², H², Q, study weights, and R-style `summary()`.

  Files: `meta/__init__.py`, `meta/_common.py`, `meta/_fixed.py`,
  `meta/_random.py`, `meta/_heterogeneity.py`.

### GEE Module

- **`gee(y, X, cluster_id, family='gaussian', corr_structure='exchangeable')`**
  — Generalized Estimating Equations for clustered and longitudinal data,
  matching R `geepack::geeglm()`. Implements the Liang & Zeger (1986) iterative
  algorithm.

- **Cross-package dependency** — First PyStatsBio module to import from
  PyStatistics. Uses `Family`/`Link` from `pystatistics.regression.families`
  for mean model specification, supporting Gaussian, binomial, Poisson, and
  Gamma families.

- **Working correlation structures** — Independence, exchangeable, AR(1), and
  unstructured. Correlation parameters estimated iteratively from Pearson
  residuals.

- **Sandwich variance estimator** — Huber-White robust standard errors provide
  valid inference even when the working correlation is misspecified. Both naive
  and robust SEs reported.

  Files: `gee/__init__.py`, `gee/_common.py`, `gee/_correlation.py`,
  `gee/_estimating_equations.py`, `gee/_sandwich.py`.

## Changed

None.

## Fixed

None.

## Breaking

None.

## Performance

No performance-targeted changes in this release.

## Tests

- **190 new tests** across all added modules.
- Total test count: 633 (510 existing + 123 new, approximately).
- Test breakdown by module:
  - Epidemiology: 67 tests
  - Meta-analysis: 67 tests
  - GEE: 56 tests
