# Context Document for PyStatsBio (and SGC-Bio)

**Purpose:** This document is the original build-planning primer for `pystatsbio` — a Python package for biotech/pharma statistical computing. It describes what PyStatsBio is, how it relates to SGC-Bio, and what it can build on from `pystatistics`.

**Status (updated for pystatsbio 3.0 / pystatistics 5.0):** PyStatsBio is now built and shipped. The **vision** and **rationale** sections below are preserved as originally written; the **Release Phases**, **file tree**, **R-package**, **import**, and **dependency** sections have been reconciled with what actually shipped and re-verified against `pystatistics==5.0.0`. Where the plan and the shipped reality diverge, the divergence is called out rather than hidden. For the authoritative *current* public API, the `README.md` and the rendered Sphinx docs are canonical; this file is context, not reference.

---

## What Is PyStatsBio?

PyStatsBio is a Python package for **biotech and pharmaceutical statistical computing** — the methods used across the drug development pipeline, from preclinical assay analysis through clinical trial design.

It is a separate package (separate repo, separate `pip install`) that depends on `pystatistics` for its general statistical computing layer.

Think of it as the domain-specific layer:
- `pystatistics` = general statistical computing (regression, survival, mixed models, tests)
- `pystatsbio` = biotech/pharma-specific methods that *use* those tools

The relationship is analogous to R's ecosystem: R ships `survival`, `lme4`, `stats` — and then domain packages like `drc`, `pROC`, `gsDesign`, `rpact` build pharma-specific functionality on top.

### What PyStatsBio Is NOT

- Not a GUI or dashboard (that's SGC-Bio's job)
- Not a clinical data management system (no EDC, no CDISC-SDTM conversion)
- Not a reporting tool (no RTF/PDF table generation — that's SGC-Bio's job)
- Not a regulatory submission builder
- Not a bioinformatics/genomics pipeline (no sequence alignment, no variant calling)
- Not a second engine — **PyStatsBio should stay smaller than PyStatistics**. If it grows larger, you're building a second engine and something went wrong.

### Target Users

Scientists and biostatisticians at 5-20 person Kendall Square biotechs who daily:
- Fit dose-response curves
- Run PK summaries
- Calculate sample sizes / power
- Evaluate biomarkers
- Run survival models

They do NOT typically:
- Design Lan-DeMets adaptive boundaries from scratch
- Implement Bretz graphical multiplicity procedures by hand
- Run population PK with NONMEM-grade NLME

Those are outsourced to CRO statisticians. The early modules must align with real daily workflows.

---

## What Is SGC-Bio?

SGC-Bio is a **web application** (built on the SGC platform) that provides a user-facing interface for biotech/pharma statistical computing. It sits on top of PyStatsBio:

```
┌─────────────────────────────┐
│  SGC-Bio (Web App)          │   ← user-facing UI, tables, reports, GPU infra
├─────────────────────────────┤
│  PyStatsBio (Package)       │   ← biotech/pharma statistical methods
├─────────────────────────────┤
│  PyStatistics (Package)     │   ← general statistical computing
└─────────────────────────────┘
```

This is the same pattern as:
- NumPy → SciPy → scikit-learn
- torch → transformers → HuggingFace Hub
- TensorFlow → Keras → enterprise ML stack

SGC-Bio matters for PyStatsBio's design because:
1. PyStatsBio's API must be clean enough for SGC-Bio to call programmatically
2. PyStatsBio should return structured results (not just print output) so SGC-Bio can render them in tables/reports
3. PyStatsBio should NOT assume interactive use — no plots, no print-to-console
4. **SGC-Bio provides GPU infrastructure** — cloud GPUs for compute-intensive modules (high-throughput screening, population PK). PyStatsBio should expose `backend=` parameters where GPU acceleration is meaningful.

The SGC-Bio layer will handle:
- Table formatting (regulatory-grade output, CDISC-style TLFs)
- PDF/RTF report generation
- Interactive parameter exploration (UI sliders for sample size, power curves)
- Study protocol templates
- Cloud GPU provisioning for heavy compute
- Integration with SGC's broader platform

---

## Release Phases

### CRITICAL: Build Like a Founder, Not a PhD

PyStatsBio must ship in phases. Not all modules are created equal. Some are bounded and high-leverage; others are multi-month research projects that could sink the project.

### Shipped status (as of 3.0)

| Module | Original phase | Status | R references |
|--------|----------------|--------|--------------|
| `power/` | 1 | ✅ shipped | `pwr`, `TrialSize`, `gsDesign`, `samplesize`, `PowerTOST` |
| `doseresponse/` | 1 | ✅ shipped | `drc`, `nplr`, `BMDS` |
| `diagnostic/` | 1 | ✅ shipped | `pROC`, `OptimalCutpoints`, `epiR` |
| `pk/` (NCA) | 1 | ✅ shipped | `PKNCA`, `NonCompart` |
| `epi/` | 2 | ✅ shipped | `epiR`, `epitools`, `stats` |
| `meta/` | — (added beyond the original plan) | ✅ shipped | `meta`, `metafor` |
| `gee/` | — (added beyond the original plan) | ✅ shipped | `geepack` |
| `agreement/` | 2 | ❌ not built | `irr`, `psych`, `DescTools` |
| `equivalence/` | 2 | ❌ not built as a module (some BE/NI power lives in `power/`: `power_crossover_be`, `power_equiv_mean`, `power_noninf_*`) | `PowerTOST`, `TOSTER` |
| `assay/`, `pd/`, `multiplicity/` | 3 | ⏳ future | see phase 3 below |
| PopPK, adaptive, indirect PD, graphical multiplicity | 4+ | ⏳ deferred | see phase 4 below |

The phase descriptions below are preserved as the original plan and rationale.

---

### Phase 1: Wedge Release  — ✅ SHIPPED

These four modules cover trial planning, preclinical HTS, biomarker validation, and PK summary. That alone serves the core daily workflow of a Kendall Square biotech.

#### `power/` — Sample Size and Power Calculations  — ✅ shipped

The bread and butter of clinical trial planning. Every trial starts with "how many subjects do we need?"

**Scope:**
- **Two-sample tests**: t-test (equal/unequal variance), proportion test (chi-squared, Fisher), rate comparison (Poisson)
- **Paired tests**: paired t-test, McNemar's test
- **ANOVA**: one-way, factorial
- **Survival**: log-rank test (Schoenfeld formula, Freedman formula, Lachin-Foulkes)
- **Non-inferiority / equivalence / superiority**: all three framings for means and proportions
- **Crossover designs**: 2x2 crossover (bioequivalence)
- **Cluster randomized trials**: design effect, ICC-adjusted sample size
- **Multi-arm trials**: Dunnett-style many-to-one comparisons

**R packages to match:** `pwr`, `TrialSize`, `gsDesign` (power functions), `samplesize`, `PowerTOST`

**CPU-only.** Solving one equation — microseconds on CPU.

**Key design principle:** Each function supports "solve for any one parameter given the others":
```python
from pystatsbio import power

# Solve for n (given effect size, alpha, power)
result = power.power_t_test(effect_size=0.5, alpha=0.05, power=0.80)
print(result.n)  # per-group sample size

# Solve for power (given n, effect size, alpha)
result = power.power_t_test(n=50, effect_size=0.5, alpha=0.05)
print(result.power)

# Solve for detectable effect size (given n, alpha, power)
result = power.power_t_test(n=50, alpha=0.05, power=0.80)
print(result.effect_size)
```

#### `doseresponse/` — Dose-Response Modeling  — ✅ shipped

The workhorse of preclinical pharmacology. Every in vitro assay, every toxicology study. **There is no good modern Python equivalent to R's `drc` — this is the killer wedge.**

**Scope:**
- **4-parameter logistic (4PL)**: the standard sigmoidal dose-response curve. `Bottom + (Top - Bottom) / (1 + (EC50/x)^Hill)`
- **5-parameter logistic (5PL)**: asymmetric 4PL with extra shape parameter
- **Log-logistic**: `drc`-style LL.4, LL.5 models
- **Weibull models**: W1.4, W2.4 (for asymmetric dose-response)
- **Brain-Cousens hormesis models**: biphasic dose-response with low-dose stimulation
- **EC50/IC50 estimation**: with delta method confidence intervals
- **Relative potency**: ratio of EC50s with Fieller's CI
- **BMD (benchmark dose)**: BMDL/BMDU computation for toxicology
- **Model comparison**: AIC/BIC/lack-of-fit F-test for model selection
- **High-throughput screening (HTS)**: fit thousands of dose-response curves simultaneously (**GPU**: batch 4PL fitting across compounds)

**R packages to match:** `drc`, `nplr`, `BMDS` (EPA benchmark dose software)

**GPU: Yes.** This is the primary GPU showcase. HTS campaigns generate thousands of compounds x multiple doses. Fitting a 4PL to each is independent — perfect for GPU batching. The inner solver is Levenberg-Marquardt nonlinear least squares, which itself can be GPU-accelerated (batched Jacobian computation, batched normal equations).

```python
from pystatsbio import doseresponse

# Single curve (CPU). 4PL == the log-logistic 4-parameter model 'LL.4' (the default).
result = doseresponse.fit_drm(dose, response, model='LL.4')
print(result.params)          # CurveParams(bottom, top, ec50, hill)
ec50_result = doseresponse.ec50(result)   # EC50 + CI is a separate call on the fit
print(ec50_result.estimate, ec50_result.ci_lower, ec50_result.ci_upper)

# High-throughput: fit thousands of curves at once (GPU)
results = doseresponse.fit_drm_batch(dose_matrix, response_matrix, model='LL.4', backend='auto')
print(results.ec50)           # one EC50 per curve
```

Model strings accepted by `fit_drm` / `fit_drm_batch` (verified against the shipped package):
`'LL.4'` (4PL, default), `'LL.5'` (5PL), `'W1.4'` (Weibull-1), `'W2.4'` (Weibull-2),
`'BC.5'` (Brain-Cousens hormesis). *(Note: the shipped code accepts `'BC.5'`, not `'BC.4'`.)*

**The real GPU killer feature of SGC-Bio is batched nonlinear curve fitting.** R is not GPU-native. Torch is. If you make HTS 4PL fitting absurdly fast, that's flashy *and* practical.

#### `diagnostic/` — Diagnostic Accuracy  — ✅ shipped

Evaluating biomarkers, screening tests, and diagnostic tools. Python's ROC ecosystem is weak. DeLong test especially.

**Scope:**
- **ROC analysis**: empirical ROC curve, AUC with DeLong confidence intervals, optimal cutoff (Youden index, closest-to-corner)
- **ROC comparison**: DeLong test for comparing two correlated ROC curves
- **Sensitivity / specificity**: point estimates with exact (Clopper-Pearson) confidence intervals
- **Predictive values**: PPV, NPV with prevalence adjustment
- **Likelihood ratios**: LR+, LR- with confidence intervals
- **Diagnostic odds ratio**: with confidence interval
- **Multi-class**: extension to >2 categories (multi-class AUC)
- **High-throughput panel**: evaluate hundreds/thousands of biomarker candidates simultaneously (**GPU**: batch AUC computation across markers)

**R packages to match:** `pROC`, `OptimalCutpoints`, `epiR`

**GPU: Optional.** Single ROC curves are CPU-fine. Batch AUC over thousands of biomarkers in HTS benefits from GPU.

#### `pk/` (NCA only) — Non-Compartmental Pharmacokinetic Analysis  — ✅ shipped

NCA is required for every PK study. Self-contained, well-defined, formulaic calculations. **Phase 1 is NCA only — no compartmental/PopPK.**

**Scope (NCA only):**
- **AUC**: linear trapezoidal, log-linear trapezoidal, linear-up/log-down
- **Cmax, Tmax**: peak concentration and time to peak
- **Half-life**: terminal elimination rate constant via log-linear regression
- **Clearance**: CL = Dose / AUC
- **Volume of distribution**: Vz = Dose / (lambda_z * AUC)
- **PK summary statistics**: geometric means and CVs (standard for PK data), confidence intervals on log-scale parameters
- **Bioequivalence PK**: Cmax and AUC ratio analysis

**R packages to match:** `PKNCA`, `NonCompart`

**CPU-only for NCA.** Always small data.

```python
from pystatsbio import pk

# NCA (CPU - always small data)
result = pk.nca(time, concentration, dose=100, route='ev')
print(result.auc_inf, result.cmax, result.half_life, result.clearance)
```

---

### Phase 2: Additive Modules (Clean, Bounded)

These are safe, bounded, and add real value. Build after Phase 1 ships.

#### `agreement/` — Inter-Rater Agreement and Method Comparison  — ❌ not built

Critical for assay validation and analytical method bridging.

**Scope:**
- **Cohen's kappa**: unweighted, linear-weighted, quadratic-weighted; with SE and CI
- **Fleiss' kappa**: multi-rater extension
- **ICC**: intraclass correlation wrapper with all 6 Shrout-Fleiss forms: ICC(1,1), ICC(2,1), ICC(3,1), ICC(1,k), ICC(2,k), ICC(3,k). Delegates to `pystatistics.mixed.lmm()` internally.
- **Bland-Altman**: bias, limits of agreement (+/-1.96 SD), with confidence intervals; proportional bias detection; repeated measures extension
- **Concordance correlation coefficient (CCC)**: Lin's CCC with CI
- **Total deviation index (TDI)**: and coverage probability (CP)

**R packages to match:** `irr`, `BlandAltmanLeh`, `psych::ICC`, `DescTools::CCC`

**CPU-only.** Small-n rater studies.

#### `equivalence/` — Bioequivalence and Non-Inferiority  — ❌ not built as a standalone module

Specialized inference for showing treatments are "close enough" rather than "different." Critical for generic drug approval. *(Some of this landed in `power/` instead — `power_crossover_be`, `power_equiv_mean`, `power_noninf_mean`, `power_noninf_prop` — but the analysis-side `equivalence/` module was not built.)*

**Scope:**
- **TOST (Two One-Sided Tests)**: for means (bioequivalence)
- **Schuirmann's test**: standard bioequivalence test
- **2x2 crossover analysis**: period effects, carryover effects, sequence effects
- **Non-inferiority margins**: for means, proportions, survival (hazard ratios)
- **Equivalence of proportions**: Farrington-Manning, Miettinen-Nurminen
- **Ratio tests**: for geometric means (log-scale analysis, common in PK studies)

**R packages to match:** `PowerTOST`, `equivalence`, `TOSTER`

**CPU-only.** Small-n crossover studies.

#### `epi/` — Epidemiological Measures  — ✅ shipped

Common in safety analyses and observational components of clinical programs.

**Scope:**
- **Relative risk (RR)**: with Wald and score CIs
- **Odds ratio (OR)**: Woolf, Gart, conditional MLE
- **Risk difference (RD)**: with Newcombe CI, Miettinen-Nurminen CI
- **Number needed to treat (NNT)**: with CI (from RD)
- **Incidence rate ratio**: with exact CI
- **Rate standardization**: direct and indirect age-standardization
- **Stratified analysis**: Mantel-Haenszel OR/RR, Breslow-Day test for homogeneity

**R packages to match:** `epiR`, `epitools`, `stats` (`mantelhaen.test`)

**CPU-only.** Contingency tables, small-n.

**NOTE: Propensity score methods (matching, IPTW) are explicitly OUT of scope.** Once you add matching + IPTW + stabilized weights + balance diagnostics, you're in causal inference land — that's a whole ecosystem. Keep it minimal.

#### `meta/` — Meta-Analysis  — ✅ shipped (added beyond the original phase plan)

Fixed- and random-effects meta-analysis of study-level effect estimates.

**Scope:**
- **Random-effects `rma`**: DerSimonian-Laird (`DL`), REML, Paule-Mandel (`PM`) between-study variance estimators
- **Heterogeneity**: Cochran's Q (`cochran_q`), I² (`i_squared`), H² (`h_squared`)

**R packages to match:** `meta` (`metagen`), `metafor` (`rma`)

**CPU-only.** Study-level data.

#### `gee/` — Generalized Estimating Equations  — ✅ shipped (added beyond the original phase plan)

Population-averaged models for clustered / longitudinal data, with a robust sandwich covariance.

**Scope:**
- **Families**: gaussian, binomial, poisson, gamma (via `pystatistics.regression.families`)
- **Working correlation structures**: independence, exchangeable, AR(1), unstructured
- **Robust (sandwich) standard errors**

**R packages to match:** `geepack` (`geeglm`)

**GPU: Optional.** GEE has a GPU-capable backend (`gee/backends/`) for large clustered fits.

---

### Phase 3: Extensions (Build After Revenue Exists)  — ⏳ not built

These modules are valuable but either depend on Phase 1/2 predecessors or require significant effort. None are built yet.

#### `assay/` — Assay Validation and Analytical Methods

Bioanalytical method validation for regulated environments. Pulls from `doseresponse/` and `pystatistics.mixed.lmm()`.

**Scope:**
- **Linearity assessment**: weighted regression, residual analysis, lack-of-fit test
- **Precision**: repeatability (within-run), intermediate precision (between-run), reproducibility. Uses nested ANOVA or mixed models from `pystatistics.mixed.lmm()`
- **Accuracy**: bias, recovery, percent relative error
- **Limit of detection (LOD)** and **limit of quantitation (LOQ)**: signal-to-noise, calibration curve, blank-based methods
- **Parallelism testing**: for immunoassays (PLA — parallel line analysis)
- **Stability**: real-time and accelerated (Arrhenius), shelf-life estimation via regression
- **Standard curve fitting**: 4PL/5PL (shared with `doseresponse/`), back-calculation of concentrations from standard curves

**R packages to match:** No single R package dominates here — scattered across `drc`, custom code, and commercial tools (SoftMax Pro, Watson LIMS). This is a real gap.

**GPU: Optional** for high-throughput plate processing (batch curve fits).

#### `pd/` — Pharmacodynamic Modeling (Direct Effect Only in v1)

PD is "what the drug does to the body." **v1: Emax and sigmoid Emax only.** Indirect response models and PK/PD link models are Phase 4.

**Scope (v1):**
- **Emax model**: direct effect. `E0 + Emax * C / (EC50 + C)`
- **Sigmoid Emax**: with Hill coefficient
- **Exposure-response (E-R)**: logistic regression with exposure metrics as predictors (uses `pystatistics.regression.fit(family='binomial')`)

**R packages to match:** `mrgsolve` (simulation), `RxODE`

**GPU: Deferred.** Direct Emax is just nonlinear regression — CPU is fine. GPU matters for simulation-based analyses which are Phase 4.

#### `multiplicity/` — Multiple Testing in Clinical Trials (Simple Procedures Only in v1)

Standard `p_adjust` isn't enough for regulatory work. But the full graphical approach is complex.

**Scope (v1 — simple only):**
- **Hierarchical (fixed-sequence)**: test in pre-specified order
- **Bonferroni-Holm with weights**: weighted versions for unequal importance
- **Fallback procedures**: for primary/secondary endpoint hierarchies

**Explicitly deferred to v2:** Bretz-Maurer-Brannath graphical approach, gatekeeping, closed testing. These are algorithmically complex (graph-based weight propagation, dynamic alpha redistribution, edge-case heavy) and need extreme validation care.

**R packages to match:** `gMCP` (v2), `multcomp`

**CPU-only.**

---

### Phase 4: Danger Zones (DO NOT BUILD IN v1)  — ⏳ deferred

These are explicitly out of scope for initial releases. Each is a multi-month research-grade project.

#### `pk/` (Population PK — NLME / SAEM) — DO NOT BUILD EARLY

**Why this is dangerous:** This is NONMEM territory. You are now in:
- Nonlinear mixed effects models
- ODE systems + random effects on parameters
- Stochastic EM (SAEM) algorithm
- Likelihood approximations (Laplace, adaptive Gaussian quadrature)
- High-dimensional integration
- Boundary constraints on variance components
- Convergence pathologies

This is a research-grade problem. R's `saemix` is ~8,000 lines. NONMEM has been developed for 40+ years. Monolix is a commercial product by a funded team.

**Do not put this in v1. This is v2 or v3 after revenue exists.**

When you do build it:
- **GPU is the killer feature**: each subject's PK profile = independent ODE solve. 500-5,000 subjects = embarrassingly parallel. SAEM simulation step also parallel.
- **CPU**: `scipy.integrate.solve_ivp` with RK45 or LSODA
- **GPU**: `torchdiffeq` for batched forward solves
- **ODE solver is an implementation detail** — user specifies the PK model, not the integration method. Do NOT expose solver knobs (RK45 vs BDF vs adjoint sensitivity). The user says `pk_model("2cmt_oral")`, not `solve_ivp(method='RK45')`.

#### `adaptive/` — Group Sequential and Adaptive Designs — DO NOT BUILD EARLY

**Why this is dangerous:** On paper it's "just alpha spending." In practice:
- Recursive boundary computation with numerical integration
- Simulation validation for type I error guarantees
- Subtle regulatory requirements
- Heavy statistical literature (Lan-DeMets, Hwang-Shih-DeCani)

This requires extreme validation care and deep domain expertise.

**Scope (when built):**
- Group sequential boundaries: O'Brien-Fleming, Pocock, alpha spending functions (Lan-DeMets)
- Alpha spending: Hwang-Shih-DeCani family, custom spending functions
- Futility boundaries: binding and non-binding
- Information fractions: equal and unequal spacing
- Sample size re-estimation: Chen-DeMets-Lan (promising zone), conditional power

**R packages to match:** `gsDesign`, `rpact`, `GroupSeq`

#### `pd/` (Indirect Response, PK/PD Link) — DO NOT BUILD EARLY

Jusko's 4 indirect response models, turnover models, effect compartment, hysteresis correction. These require ODE infrastructure from PopPK. Build after PopPK exists.

#### Full Graphical Multiplicity (Bretz-Maurer-Brannath) — DO NOT BUILD EARLY

The graph-based weight propagation approach is the regulatory standard for multi-endpoint trials, but it's algorithmically complex with many edge cases. Defer to v2.

---

## GPU Strategy Summary

| Module | GPU? | Phase | Status | Rationale |
|--------|------|-------|--------|-----------|
| `power/` | No | 1 | ✅ | Solving one equation — microseconds on CPU |
| `doseresponse/` | **Yes** | 1 | ✅ | Batch curve fitting: thousands of compounds x multiple doses. **The GPU showcase.** |
| `diagnostic/` | Optional | 1 | ✅ | Batch AUC over thousands of biomarkers in HTS (`batch_auc`) |
| `pk/` (NCA) | No | 1 | ✅ | Formulaic, small data |
| `epi/` | No | 2 | ✅ | Contingency tables, small-n |
| `meta/` | No | — | ✅ | Study-level data, small-n |
| `gee/` | Optional | — | ✅ | GPU-capable backend for large clustered fits |
| `agreement/` | No | 2 | ❌ | Small-n rater studies |
| `equivalence/` | No | 2 | ❌ | Small-n crossover studies |
| `assay/` | Optional | 3 | ⏳ | Batch curve fitting for high-throughput plate processing |
| `pd/` (Emax) | No | 3 | ⏳ | Direct Emax is just nonlinear regression |
| `multiplicity/` | No | 3 | ⏳ | Graph algorithms, small-n |
| `pk/` (PopPK) | **Yes** | 4+ | ⏳ | Parallel ODE solves across subjects. Killer use case. |
| `pd/` (indirect) | **Yes** | 4+ | ⏳ | PK/PD simulation: parallel ODE + likelihood |
| `adaptive/` | No | 4+ | ⏳ | Boundaries are recursive but small |

Shipped modules that expose a `backend=` parameter: `doseresponse/` (`fit_drm_batch`), `diagnostic/` (`batch_auc`), and `gee/` (`gee`). All other shipped modules are CPU-only.

---

## What PyStatsBio Can Build On From PyStatistics

This is the critical section: the `pystatistics` public surface available to build on.

**How the shipped pystatsbio actually couples to pystatistics.** In practice pystatsbio imports only a *narrow* slice of pystatistics — its infrastructure, not its statistical functions:

- `pystatistics.core.exceptions` — `ValidationError`, `ConvergenceError`, `NumericalError`
- `pystatistics.core.result` — `Result`, `SolutionReprMixin`
- `pystatistics.core.compute.backend` / `.device` / `.tolerances` — backend resolution, device selection, GPU tolerances (used by the GPU-capable modules)
- `pystatistics.regression.families` — `Family`, `resolve_family` (used by `gee`)

The catalogue below documents the broader pystatistics **public surface that is available** to build on. Every signature, accessor, and value-enum below was **regenerated by introspecting `pystatistics==5.0.0`** (the shipped, PyPI-installed version), not copied from memory.

### `pystatistics.regression`

```python
from pystatistics.regression import fit, Design, Family, Gaussian, Binomial, Poisson
# (also exported: Gamma, NegativeBinomial, InverseGaussian, QuasiBinomial, QuasiPoisson,
#  GLMSolution, LinearSolution, DevianceTable, C, DataSource, deviance_table, drop1, ridge)

# fit(X_or_design, y=None, *, family=None, backend=None, solver=None,
#     force=False, tol=1e-08, max_iter=25, names=None, l2=0.0,
#     weights=None, offset=None, conf_level=0.95)
#   X_or_design: 2-D array (design matrix) OR a Design object
#   family=None -> OLS (LinearSolution); else GLM (GLMSolution)
#   backend: None | 'auto' | 'cpu' | 'gpu' | 'gpu_fp64'
#   solver:  None | 'qr' | 'svd'
#   Returns: LinearSolution | GLMSolution
#
#   family= accepted strings (case-insensitive; aliases in parens):
#     'gaussian' ('normal'), 'binomial', 'poisson', 'gamma',
#     'negative-binomial' ('nb'), 'quasipoisson', 'quasibinomial',
#     'inverse-gaussian'
#   link= accepted strings (on a Family instance, e.g. Binomial(link=...)):
#     'identity', 'logit', 'log', 'inverse', 'probit', 'cloglog',
#     'cauchit', 'sqrt', 'inverse-squared'
#
# LinearSolution accessors:
#   .coefficients (alias .coef), .standard_errors, .t_values, .p_values,
#   .conf_int, .conf_level, .residuals, .residuals_standardized,
#   .fitted_values, .rss, .tss, .r_squared, .adjusted_r_squared,
#   .residual_std_error, .rank, .df_residual, .hat_values,
#   .cooks_distance, .backend_name, .timing, .info, .warnings
#   .summary() -> str
#
# GLMSolution accessors:
#   .coefficients (alias .coef), .standard_errors, .z_values, .p_values,
#   .conf_int, .profile_conf_int, .conf_level, .fitted_values,
#   .linear_predictor, .residuals_deviance, .residuals_pearson,
#   .residuals_working, .residuals_response, .residuals_standardized,
#   .deviance, .null_deviance, .aic, .bic, .dispersion,
#   .rank, .df_residual, .df_null, .converged, .n_iter,
#   .family_name, .link_name, .hat_values, .cooks_distance,
#   .backend_name, .timing, .info, .warnings
#   .summary() -> str
#
#   .family_name canonical values: 'gaussian', 'binomial', 'poisson',
#     'gamma', 'negative-binomial', 'quasipoisson', 'quasibinomial',
#     'inverse-gaussian'
#   .link_name canonical values: 'identity', 'logit', 'log', 'inverse',
#     'probit', 'cloglog', 'cauchit', 'sqrt', 'inverse-squared'
#     (default links: gaussian->identity, binomial/quasibinomial->logit,
#      poisson/quasipoisson/negative-binomial->log, gamma->inverse,
#      inverse-gaussian->inverse-squared)
```

### `pystatistics.descriptive`

```python
from pystatistics.descriptive import describe, cor, cov, var, quantile, summary

# All six functions return a DescriptiveSolution.
# Real 5.0 signatures (backend defaults to None = auto-select).
#
# describe(data, *, na_action='everything', quantile_type=7, backend=None) -> DescriptiveSolution
#   computes everything: mean, var, cov, cor_pearson, quantiles, summary, skewness, kurtosis
# cor(x, y=None, *, method='pearson', na_action='everything', backend=None) -> DescriptiveSolution
#   method: 'pearson' | 'spearman' | 'kendall'
# cov(x, y=None, *, na_action='everything', backend=None) -> DescriptiveSolution
# var(x, *, na_action='everything', backend=None) -> DescriptiveSolution
# quantile(x, probs=None, *, quantile_type=7, na_action='everything', backend=None) -> DescriptiveSolution
#   quantile_type: int 1-9 (R quantile types); probs=None -> [0, .25, .5, .75, 1]
# summary(x, *, na_action='everything', backend=None) -> DescriptiveSolution
#
# na_action: 'everything' | 'complete' | 'pairwise'   (values de-dotted in 5.0)
# backend:   'auto' | 'cpu' | 'gpu' | None  (None = auto-select)
#
# DescriptiveSolution accessors (each function fills only its fields; unset -> None):
#   mean, variance, sd, skewness, kurtosis          # per-column arrays, shape (p,)
#   covariance_matrix                                # shape (p, p)
#   correlation_matrix                               # whichever cor was computed
#   correlation_pearson, correlation_spearman, correlation_kendall
#   quantiles                                        # shape (n_probs, p)
#   quantile_probs, quantile_type
#   summary_table                                    # (6, p): Min, Q1, Median, Mean, Q3, Max
#   n_complete, pairwise_n, columns
#   info, timing, backend_name, warnings
#   summary()                                        # METHOD -> R-style summary string
```

### `pystatistics.hypothesis`

```python
from pystatistics.hypothesis import (
    t_test, chisq_test, fisher_test, wilcox_test,
    ks_test, prop_test, var_test, p_adjust,
)

# alternative accepts: 'two-sided' (default) | 'less' | 'greater'
# Every test function takes a trailing `backend: str | None = None` kwarg.
#
# t_test(x, y=None, *, alternative='two-sided', pop_mean=0.0,
#        paired=False, equal_var=False, conf_level=0.95, backend=None)
#   -> HTestSolution.  equal_var defaults False (Welch t-test).
#
# chisq_test(x, y=None, *, correct=True, expected_probs=None,
#            rescale_probs=False, simulate_p_value=False,
#            n_resamples=2000, seed=None, backend=None)
#   -> HTestSolution.  (No `alternative` param.)
#
# fisher_test(x, y=None, *, alternative='two-sided', conf_int=True,
#             conf_level=0.95, simulate_p_value=False,
#             n_resamples=2000, seed=None, backend=None) -> HTestSolution
#
# wilcox_test(x, y=None, *, alternative='two-sided', null_value=0.0,
#             paired=False, exact=None, correct=True,
#             conf_int=True, conf_level=0.95, backend=None) -> HTestSolution
#
# ks_test(x, y=None, *, alternative='two-sided', distribution=None,
#         backend=None, **dist_params) -> HTestSolution
#   distribution: 'norm' | 'unif' | 'exp' | None
#
# prop_test(x, n_trials=None, *, null_value=None, alternative='two-sided',
#           conf_level=0.95, correct=True, backend=None) -> HTestSolution
#
# var_test(x, y=None, *, null_value=1.0, alternative='two-sided',
#          conf_level=0.95, backend=None) -> HTestSolution
#
# p_adjust(p_values, method='holm', n_comparisons=None) -> NDArray[np.floating]
#   method: 'holm' | 'hochberg' | 'hommel' | 'bonferroni'
#         | 'BH' | 'BY' | 'fdr' (alias for BH) | 'none'
#
# HTestSolution accessors (read-only properties; .summary() is a method):
#   .statistic  .statistic_name  .parameter  .p_value  .conf_int  .conf_level
#   .estimate  .null_value  .alternative  .method  .data_name
#   .observed  .expected  .residuals  .stdres          # populated for chisq_test
#   .extras  .info  .timing  .backend_name  .warnings
#   .summary() -> str   (R-style print.htest output)
```

### `pystatistics.montecarlo`

```python
from pystatistics.montecarlo import boot, boot_ci, permutation_test

# boot(data, statistic, n_resamples=999, *, method='ordinary',
#      statistic_type='index', strata=None, ran_gen=None, mle=None,
#      seed=None, backend=None, gpu_statistic=None) -> BootstrapSolution
#   method: 'ordinary' | 'parametric' | 'balanced'
#   statistic_type: 'index' | 'frequency' | 'weight'
#   BootstrapSolution accessors:
#     .t0, .t, .standard_errors, .bias, .conf_int (None until boot_ci),
#     .method, .n_resamples, .conf_level, .data, .seed,
#     .backend_name, .info, .timing, .warnings, .summary()
#
# boot_ci(boot_out, *, conf_level=0.95, ci_type='all', index=0,
#         var_t0=None, var_t=None) -> BootstrapSolution
#   ci_type: 'normal' | 'basic' | 'percentile' | 'bca' | 'studentized' | 'all'
#   CIs live in .conf_int (dict keyed by ci_type), each value shape (k, 2)
#
# permutation_test(x, y, statistic, n_resamples=9999, *,
#                  alternative='two-sided', seed=None, backend=None,
#                  gpu_statistic=None) -> PermutationSolution
#   PermutationSolution accessors:
#     .observed_stat, .p_value, .perm_stats, .alternative, .n_resamples,
#     .backend_name, .info, .timing, .warnings, .summary()
```

### `pystatistics.survival`

```python
from pystatistics.survival import kaplan_meier, survdiff, coxph, discrete_time

# kaplan_meier(time, event, *, strata=None, entry=None,
#              conf_level=0.95, conf_type='log') -> KMSolution
#   conf_type: 'log' | 'plain' | 'log-log';  entry= supports left-truncation
#   KMSolution accessors:
#     .time, .survival, .se, .standard_errors (alias of .se, Greenwood),
#     .ci_lower, .ci_upper, .conf_int,
#     .n_risk, .n_events, .n_censored, .n_events_total,
#     .median_survival, .n_observations,
#     .conf_level, .conf_type, .backend_name, .timing, .warnings, .summary()
#
# survdiff(time, event, group, *, rho=0.0) -> LogRankSolution
#   rho=0 -> log-rank, rho=1 -> Peto-Peto
#   LogRankSolution accessors:
#     .statistic, .p_value, .df, .rho, .observed, .expected,
#     .group_labels, .n_groups, .n_per_group,
#     .backend_name, .timing, .warnings, .summary()
#
# coxph(time, event, X, *, terms=None, names=None, strata=None, start=None,
#       robust=False, cluster=None, ties='efron', tol=1e-09, max_iter=20,
#       conf_level=0.95) -> CoxSolution
#   ties: 'efron' | 'breslow'
#   CoxSolution accessors:
#     .coefficients (alias .coef), .hazard_ratios (alias .hr),
#     .standard_errors, .naive_standard_errors, .z_values, .p_values,
#     .conf_int (HR scale), .concordance, .loglik (null, full),
#     .n_events, .n_observations, .n_strata, .converged, .n_iter,
#     .robust, .ties, .conf_level, .backend_name, .timing, .warnings, .summary()
#
# discrete_time(time, event, X, *, names=None, intervals=None,
#               backend=None, conf_level=0.95) -> DiscreteTimeSolution
#   DiscreteTimeSolution accessors:
#     .coefficients (alias .coef), .hazard_ratios (alias .hr),
#     .standard_errors, .z_values, .p_values, .conf_int,
#     .baseline_hazard, .interval_labels, .n_intervals, .person_period_n,
#     .glm_aic, .glm_deviance, .n_events, .n_observations,
#     .converged, .n_iter, .conf_level, .backend_name, .timing, .warnings, .summary()
```

### `pystatistics.anova`

```python
from pystatistics.anova import (
    anova_oneway, anova, anova_rm, anova_posthoc, levene_test,
)

# anova_oneway(y, group, *, ss_type=1) -> AnovaSolution
# anova(y, factors, *, covariates=None, ss_type=2, interactions=True) -> AnovaSolution
#     factors:    dict[str, ArrayLike]  (e.g. {'A': fac_a, 'B': fac_b})
#     covariates: dict[str, ArrayLike] | None  (ANCOVA); ss_type: 1 | 2 | 3
#   AnovaSolution accessors:
#     .table (list of ANOVA rows; row.f_value etc.), .eta_squared,
#     .partial_eta_squared, .omega_squared, .partial_omega_squared,
#     .group_means, .grand_mean, .residual_ss, .residual_df, .residual_ms,
#     .ss_type, .n_obs, .backend_name, .info, .warnings, .timing, .summary()
#
# anova_rm(y, subject, within, *, between=None, correction='auto') -> AnovaRMSolution
#     within/between: dict[str, ArrayLike]
#     correction: 'none' | 'greenhouse-geisser' | 'huynh-feldt' | 'auto'
#   AnovaRMSolution accessors:
#     .table, .sphericity (list; row.gg_epsilon etc.), .eta_squared,
#     .partial_eta_squared, .correction, .n_obs, .n_subjects,
#     .info, .warnings, .timing, .summary()
#
# anova_posthoc(anova_result, *, method='tukey', factor=None,
#               control=None, conf_level=0.95) -> PostHocSolution
#     method: 'tukey' | 'bonferroni' | 'dunnett' | 'games-howell'
#             (dunnett requires control=)
#   PostHocSolution accessors:
#     .comparisons, .method, .factor, .conf_level, .summary()
#
# levene_test(y, group, *, location='median') -> LeveneSolution
#     location: 'median' (Brown-Forsythe, default) | 'mean' (original Levene)
#   LeveneSolution accessors:
#     .f_value, .p_value, .df_between, .df_within, .group_vars, .location, .summary()
```

### `pystatistics.mixed`

```python
from pystatistics.mixed import lmm, glmm, LMMSolution, GLMMSolution

# lmm(y, X, groups, *, random_effects=None, random_data=None,
#     reml=True, tol=1e-8, max_iter=200, compute_satterthwaite=True,
#     conf_level=0.95) -> LMMSolution
#   groups: dict[str, ArrayLike]; random_effects: dict[str, list[str]] | None
#   reml=True -> REML; reml=False -> ML (needed for .compare LRT)
#   LMMSolution accessors:
#     .coefficients, .standard_errors, .df_satterthwaite, .t_values,
#     .p_values, .conf_int, .coef, .fixef (dict), .ranef (dict, BLUPs),
#     .icc (dict), .var_components (tuple), .log_likelihood, .aic, .bic,
#     .conf_level, .fitted_values, .residuals, .converged, .is_singular,
#     .n_iter, .backend_name, .timing, .warnings, .params (LMMParams),
#     .compare(other) -> str (LRT; use ML), .summary()
#   NOTE 5.0: .se -> .standard_errors; .residual_variance / .residual_std removed.
#
# glmm(y, X, groups, *, family='binomial', random_effects=None,
#      random_data=None, offset=None, weights=None, correlated=None,
#      tol=1e-8, max_iter=200, conf_level=0.95) -> GLMMSolution
#   family: only FIXED-dispersion families ('binomial', 'poisson',
#     'negative-binomial'/'nb'); free-dispersion families raise ValidationError
#     (use lmm() for Gaussian).
#   GLMMSolution accessors:
#     .coefficients, .standard_errors, .z_values, .p_values, .conf_int,
#     .coef, .fixef, .ranef, .icc, .var_components, .deviance,
#     .log_likelihood, .aic, .bic, .conf_level, .fitted_values, .residuals,
#     .linear_predictor, .converged, .is_singular, .n_iter, .backend_name,
#     .timing, .warnings, .params (GLMMParams), .summary()
#   NOTE 5.0: .se -> .standard_errors; .family_name / .link_name moved to
#             .params.family_name / .params.link_name.
```

### `pystatistics.mvnmle`

```python
from pystatistics.mvnmle import mlest

# mlest(data_or_design, *, method='direct', backend=None, solver=None,
#       tol=None, max_iter=None, regularize=True, force=False,
#       collinearity_tol=None, verbose=False) -> MVNSolution
#   data_or_design : array/DataFrame (NaN = missing) OR an MVNDesign object
#   method  : 'direct' (gradient) | 'em' | 'monotone' (closed-form, monotone
#             missingness only — else ValidationError; check is_monotone() first)
#   backend : None | 'auto' | 'cpu' | 'gpu' | 'gpu_fp64'  (None -> 'cpu')
#             NOTE: old 'cpu-reference' backend REMOVED — use solver='reference'.
#   solver  : 'reference' (R-exact numpy inverse-Cholesky; method='direct' only)
#             | <scipy optimizer name, e.g. 'BFGS'> | None (auto)
#   tol/max_iter : None -> direct (1e-5, 100), em (1e-4, 1000)
#   MVNSolution accessors:
#     .muhat (was .mean), .sigmahat (was .sigma), .correlation_matrix,
#     .standard_deviations, .loglik, .aic, .bic, .converged, .n_iter,
#     .gradient_norm, .backend_name, .warnings, .timing, .info,
#     .summary(), .to_dict()
# Also exported: is_monotone, mcar_test, little_mcar_test, analyze_patterns,
#   MVNDesign, MVNSolution, MCARTestSolution, ...
```

### Shared Infrastructure

```python
from pystatistics import DataSource
from pystatistics.regression import Family, Gaussian, Binomial, Poisson, Gamma

# DataSource: unified data container for the fit/regression API (top-level import).
# Keyword-only classmethod factories:
#   DataSource.from_arrays(*, X=None, y=None, data=None, columns=None, **named_arrays)
#   DataSource.from_tensors(*, X=None, y=None, **named_tensors)
#   DataSource.from_dataframe(df, *, source_path=None)
#   DataSource.from_file(path, *, columns=None)
# Instance API: .to(device), .supports(capability), .keys(),
#   .metadata, .n_observations, .device

# Family base class (GLM/GLMM error families). Constructor: Family(link=None)
#   Family.variance(mu) -> NDArray
#   Family.deviance(y, mu, wt) -> float
#   Family.initialize(y, weights=None) -> NDArray     # gained optional weights in 5.0
#   Family.log_likelihood(y, mu, wt, dispersion) -> float
#   Family.aic(y, mu, wt, rank, dispersion) -> float  # new in 5.0
#   properties: .name, .link, .dispersion_estimator, .dispersion_is_fixed
#
# Concrete families (.name / default link):
#   Gaussian() 'gaussian'/identity   Binomial() 'binomial'/logit
#   Poisson()  'poisson'/log         Gamma()    'gamma'/inverse
#   Also: NegativeBinomial ('negative-binomial'), InverseGaussian
#   ('inverse-gaussian'), QuasiBinomial ('quasibinomial'), QuasiPoisson ('quasipoisson')
#
# resolve_family(name) accepts the family strings above; pystatsbio.gee uses it.
```

---

## Design Principles for PyStatsBio

### 1. Return Structured Results, Not Strings

Every function returns a frozen dataclass or Solution object. SGC-Bio needs to extract numbers, not parse text.

```python
# Good:
@dataclass(frozen=True)
class PowerParams:
    n: int | None
    power: float | None
    effect_size: float | None
    alpha: float
    method: str

# Bad:
def power_t_test(...) -> str:
    return "n = 64 per group"
```

### 2. Follow PyStatistics Patterns

- Use the `Result[Params]` wrapper pattern wrapped in a `…Solution` for consistency
- Provide `.summary()` for human-readable output
- Use numpy arrays, not lists
- Validate inputs early with clear error messages
- Reuse `pystatistics.core.exceptions`, `pystatistics.core.result`, and
  `pystatistics.core.compute.backend` rather than defining parallels

### 3. Match R Reference Implementations

Same validation strategy as pystatistics:
1. Generate fixture data in Python
2. Compute reference results in R (with 17-digit precision)
3. Parametrized pytest against the JSON fixtures
4. Document which R package + function each result is validated against

### 4. "Solve For Any One" Pattern (power/ module)

Power/sample size functions accept all-but-one parameter and solve for the missing one. Use `None` as the signal for "solve for this."

```python
def power_t_test(
    n: int | None = None,
    effect_size: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    alternative: str = 'two-sided',
    test_type: str = 'two-sample',
) -> PowerSolution:
    """Exactly one of n, effect_size, power must be None."""
```

### 5. GPU Only Where It Matters

- Shipped: `doseresponse/` (`fit_drm_batch`), `diagnostic/` (`batch_auc`), `gee/` get `backend=`
- Phase 4+: `pk/` (PopPK), `pd/` (indirect) will get `backend=`
- All clinical trial planning modules: CPU-only, no `backend=` parameter

GPU backends follow the same two-tier validation as pystatistics: CPU matches R, GPU matches CPU. Precision is encoded in the backend string (`'gpu'` = fp32, `'gpu_fp64'` = CUDA fp64); there is no separate `use_fp64` flag.

### 6. Separate Computation From Presentation

PyStatsBio computes. SGC-Bio formats. Don't put table rendering, LaTeX output, or plot generation in PyStatsBio.

### 7. Hide Implementation Details

For PK/PD and dose-response: do NOT expose ODE solver knobs. The user says `pk_model("2cmt_oral")`, not `solve_ivp(method='RK45')`. Solver selection is an implementation detail.

### 8. Stay Small

Every module should be a thin domain layer over pystatistics primitives, not a reimplementation. If pystatsbio exceeds pystatistics, you're building a second engine.

---

## Shipped File Tree

```
pystatsbio/
    __init__.py
    power/
        __init__.py
        _means.py            # t-test power, paired t-test power
        _proportions.py      # proportion test power, Fisher power
        _survival.py         # log-rank power (Schoenfeld, Freedman, Lachin-Foulkes)
        _anova.py            # one-way and factorial ANOVA power
        _noninferiority.py   # NI/equivalence/superiority power
        _crossover.py        # 2x2 crossover, bioequivalence
        _cluster.py          # cluster randomized trial power
        _common.py           # PowerSolution/PowerParams, shared utilities
    doseresponse/
        __init__.py
        _models.py           # LL.4, LL.5, W1.4, W2.4, BC.5
        _fit.py              # single curve fitting (Levenberg-Marquardt)
        _batch.py            # batch fitting for HTS (GPU-capable)
        _potency.py          # EC50, IC50, relative potency, Fieller CI
        _bmd.py              # benchmark dose (BMDL/BMDU)
        _common.py           # DoseResponseSolution, CurveParams
    diagnostic/
        __init__.py
        _roc.py              # ROC curve, AUC, DeLong CI, DeLong test
        _accuracy.py         # sensitivity, specificity, PPV, NPV, LR
        _cutoff.py           # optimal cutoff methods
        _batch.py            # batch AUC for biomarker panels (GPU-capable)
        _common.py           # ROCSolution, DiagnosticSolution
    pk/
        __init__.py
        _nca.py              # non-compartmental analysis
        _common.py           # NCASolution
    epi/
        __init__.py
        _measures.py         # 2x2 measures (RR, OR, RD, NNT)
        _standardize.py      # direct/indirect rate standardization
        _mantel_haenszel.py  # stratified analysis
        _common.py           # EpiMeasure, Solutions
    meta/
        __init__.py
        _fixed.py            # fixed-effect meta-analysis
        _random.py           # random-effects rma (DL/REML/PM)
        _heterogeneity.py    # Cochran's Q, I^2, H^2
        _common.py           # MetaSolution/MetaParams
    gee/
        __init__.py          # gee() entry point
        _estimating_equations.py
        _correlation.py      # independence/exchangeable/AR1/unstructured
        _sandwich.py         # robust covariance
        _common.py           # GEESolution/GEEParams
        backends/
            __init__.py
            gpu_fit.py       # GPU-capable GEE fit
            _gpu_family.py
            _gpu_correlation.py
tests/
    power/  doseresponse/  diagnostic/  pk/  epi/  meta/  gee/
```

Note the divergences from the original sketch: `doseresponse/` has no `backends/` subdir (batching lives in `_batch.py`); `gee/` is the module that carries a `backends/` subdir; and `epi/`, `meta/`, `gee/` were added.

---

## R Packages to Validate Against

| Module | Status | R Packages | Key Functions |
|--------|--------|-----------|---------------|
| `power/` | ✅ | `pwr`, `TrialSize`, `gsDesign`, `samplesize`, `PowerTOST` | `pwr.t.test()`, `pwr.2p.test()`, `nSurv()`, `sampleSize()` |
| `doseresponse/` | ✅ | `drc`, `nplr`, `BMDS` | `drm()`, `ED()`, `EDcomp()`, `bmd()` |
| `diagnostic/` | ✅ | `pROC`, `OptimalCutpoints`, `epiR` | `roc()`, `roc.test()`, `epi.tests()`, `optimal.cutpoints()` |
| `pk/` (NCA) | ✅ | `PKNCA`, `NonCompart` | `pk.nca()`, `sNCA()` |
| `epi/` | ✅ | `epiR`, `epitools`, `stats` | `epi.2by2()`, `ageadjust.direct/indirect()`, `mantelhaen.test()` |
| `meta/` | ✅ | `meta`, `metafor` | `metagen()`, `rma()` |
| `gee/` | ✅ | `geepack` | `geeglm()` |
| `agreement/` | ❌ | `irr`, `psych`, `BlandAltmanLeh`, `DescTools` | `kappa2()`, `ICC()`, `bland.altman.stats()`, `CCC()` |
| `equivalence/` | ❌ | `PowerTOST`, `equivalence`, `TOSTER` | `power.TOST()`, `TOSTtwo()` |
| `assay/` | ⏳ | `drc`, custom | `drm()` for calibration curves |
| `pd/` (Emax) | ⏳ | `mrgsolve`, `RxODE` | `mrgsolve::mrgsim()` |
| `multiplicity/` | ⏳ | `multcomp` | `simConfint()` |
| `pk/` (PopPK) | ⏳ | `nlme`, `saemix` | `nlme()`, `saemix()` |
| `adaptive/` | ⏳ | `gsDesign`, `rpact`, `GroupSeq` | `gsDesign()`, `getDesignGroupSequential()` |
| `multiplicity/` (graphical) | ⏳ | `gMCP` | `graphTest()` |

---

## Dependencies

```toml
# pyproject.toml (pystatsbio 3.0)
[project]
name = "pystatsbio"
requires-python = ">=3.11"
dependencies = [
    "pystatistics>=5.0",
    "numpy>=1.24",
    "scipy>=1.10",
]

[project.optional-dependencies]
gpu = ["torch>=2.0"]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "ruff>=0.1", "mypy>=1.0"]
docs = ["sphinx>=6.0", "furo"]
```

`pystatsbio` requires `pystatistics>=5.0` (the 3.0 release tracks the settled post-5.0 surface). It brings numpy and scipy as transitive dependencies. The `gpu` extra adds PyTorch for the GPU-accelerated dose-response batch fitting, batch AUC, and GEE backends.

**Phase 4+ will add `torchdiffeq>=0.2` to the gpu extra** for ODE-based PK/PD. Don't add it until PopPK is actually being built.
