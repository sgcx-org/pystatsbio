# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py <version>`
> That script uses this file to build the CHANGELOG entry, bumps versions
> everywhere, and resets this file for the next cycle.

## Changes

- **New module: `epi`** — Epidemiological measures for 2×2 tables, rate
  standardization, and stratified analysis.

  - `epi_2by2(table)` — Computes risk ratio, odds ratio, risk difference,
    attributable fraction in exposed, population attributable fraction, and
    number needed to treat from a 2×2 contingency table. Log-transformed CIs
    for RR/OR (Woolf method), Wald CIs for RD, delta method for PAF. Applies
    0.5 continuity correction for zero cells. Validates against R `epiR::epi.2by2()`.

  - `rate_standardize(counts, person_time, standard_pop)` — Direct and indirect
    age-standardization of rates. Direct method uses normal approximation CI;
    indirect method computes SIR with exact Poisson CI.
    Validates against R `epitools::ageadjust.direct()` / `ageadjust.indirect()`.

  - `mantel_haenszel(tables, measure='OR')` — Mantel-Haenszel pooled OR or RR
    across K strata. Robins-Breslow-Greenland CI for OR, Greenland-Robins CI
    for RR. Continuity-corrected CMH chi-squared test. Breslow-Day homogeneity
    test for OR. Validates against R `stats::mantelhaen.test()`.

  Files: `epi/__init__.py`, `epi/_common.py`, `epi/_measures.py`,
  `epi/_standardize.py`, `epi/_mantel_haenszel.py`.
  67 tests in `tests/epi/test_epi.py`.

- **New module: `meta`** — Fixed-effects and random-effects meta-analysis
  matching R's `metafor::rma()`.

  - `rma(yi, vi, method='REML')` — Main entry point for inverse-variance
    weighted meta-analysis of pre-computed effect sizes and variances. Supports
    four methods: `'FE'` (fixed-effects), `'DL'` (DerSimonian-Laird),
    `'REML'` (restricted maximum likelihood, default), `'PM'` (Paule-Mandel).

  - Heterogeneity statistics: Cochran's Q with chi-squared p-value, I² (%)
    quantifying between-study variability, H² ratio. REML provides tau² SE
    via observed Fisher information.

  - Returns `MetaResult` frozen dataclass with pooled estimate, SE, CI,
    z-value, p-value, tau², I², H², Q, study weights, and R-style `summary()`.

  Files: `meta/__init__.py`, `meta/_common.py`, `meta/_fixed.py`,
  `meta/_random.py`, `meta/_heterogeneity.py`.
  67 tests in `tests/meta/test_meta.py`.

- **New module: `gee`** — Generalized Estimating Equations for clustered and
  longitudinal data, matching R's `geepack::geeglm()`.

  - `gee(y, X, cluster_id, family='gaussian', corr_structure='exchangeable')` —
    Fits marginal models for correlated data using the Liang & Zeger (1986)
    algorithm. First PyStatsBio module to use `pystatistics.regression.families`
    as a real cross-package dependency (imports Family/Link for mean model).

  - Four working correlation structures: independence, exchangeable, AR(1),
    unstructured. Correlation parameters estimated iteratively from Pearson
    residuals.

  - Sandwich (Huber-White) robust variance estimator provides valid inference
    even when the working correlation is misspecified. Both naive and robust
    standard errors reported.

  - Supports Gaussian, binomial, Poisson, and Gamma families.

  Files: `gee/__init__.py`, `gee/_common.py`, `gee/_correlation.py`,
  `gee/_estimating_equations.py`, `gee/_sandwich.py`.
  56 tests in `tests/gee/test_gee.py`.
