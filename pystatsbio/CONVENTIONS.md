# PyStatsBio Conventions (adopt-and-extend)

This document governs the **pystatsbio public API**. It does not restate the law
from scratch — it **adopts the pystatistics constitution as binding** and adds a
small set of pystatsbio-specific amendments for this library's domain
(GEE, power analysis, epidemiology, meta-analysis, PK/NCA, diagnostics,
dose-response).

## 0. Adoption

The binding base is **`pystatistics/CONVENTIONS.md`** (the pystatistics 5.0
constitution). Everything in it applies to pystatsbio verbatim:

- the **naming law** S0–S6 (one name one meaning; descriptive `snake_case`; no
  single-letter public params except `x`/`y`/`X`; no dotted string values; never
  shadow a builtin; one constitutional name per concept; `method`/`solver`/`link`
  distinct; prefer the Python ecosystem on a tie);
- the **selector taxonomy** (`backend` = device+precision, `family`, `link`,
  `method` = statistical choice, `solver` = numerical routine, `na_action`);
- the **backend & precision convention** (`backend=` jointly encodes device and
  precision: `'cpu'` fp64, `'gpu'` fp32, `'gpu_fp64'` CUDA-only fp64, `'auto'`;
  no `use_fp64` flag; honest subsets; canonical error messages from
  `pystatistics.core.compute.backend`);
- the **result-object conventions** (`…Solution` wrapping `Result[…Params]`;
  uniform accessors; `summary()` + `__repr__` + `_repr_html_` via
  `core.result.SolutionReprMixin`);
- the **exception conventions** and amendments **A1–A14**.

When this document and the base disagree, **this document wins for pystatsbio**;
where this document is silent, the base governs. Like the base, this document is
**self-amending**: a new ambiguity is resolved once, here, as a numbered
amendment (B-series), not re-litigated per occurrence.

Reuse, don't fork: pystatsbio imports `pystatistics.core.exceptions`,
`pystatistics.core.result`, and `pystatistics.core.compute.backend` rather than
defining its own parallels. One hierarchy across the ecosystem.

---

## pystatsbio amendments (B-series)

### B1 — Power-analysis symbols are descriptivized

The power module's R/`pwr`-style single-letter symbols violate S1 and are
renamed to descriptive names (following statsmodels and S6). One effect-size
parameter per function is named `effect_size` regardless of which Cohen measure
it carries (d, f, or h) — each function takes exactly one, so S0 holds (it means
"the effect size for this test"); the docstring names the specific measure.

| Old | New |
|---|---|
| `d` (Cohen's d), `f` (Cohen's f), `h` (Cohen's h) | `effect_size` |
| `k` (number of groups) | `n_groups` |
| `hr` (hazard ratio) | `hazard_ratio` |
| `cv` (coefficient of variation) | `coef_variation` |
| `sd` (standard deviation) | `std` |
| `p1`, `p2` (proportions) | `prop1`, `prop2` |

`n` (sample size) and `icc` (intraclass correlation) are kept: `n` is the
universal sample-size symbol (treated like the `x`/`y`/`X` carve-out — it is the
quantity, not an abbreviation of one), and `icc` is an established multi-letter
term, not a single letter.

### B2 — Every public return is a `…Solution`

All 14 result classes adopt the full envelope (the "all-in" decision): renamed
to the `…Solution` suffix, wrapping a frozen `…Params` payload in
`core.result.Result`, exposing the uniform metadata accessors (`.backend_name`,
`.timing`, `.warnings`, `.info`), `summary()`, and `_repr_html_` (via
`SolutionReprMixin`). This includes derived/extracted results (`ec50`,
`relative_potency`, the epi measures): consistency over ceremony. CPU-only
results report `backend_name='cpu'`.

### B3 — Exceptions come from `pystatistics.core.exceptions`

No bare `raise ValueError`/`RuntimeError` for validation anywhere in pystatsbio.
Input validation → `ValidationError` (or `DimensionError`); iterative
non-convergence → `ConvergenceError` (with `iterations`/`final_change`/`reason`);
GPU unavailable / `gpu_fp64`-on-MPS → `RuntimeError` with the canonical message.
Domain-specific exceptions subclass the right base: `LambdaZEstimationError`
becomes a subclass of `ConvergenceError` (terminal-slope estimation is a
fit-quality failure), not of `ValueError`.

### B4 — Batch GPU paths follow the backend convention

`diagnostic.batch_auc` and `doseresponse.fit_drm_batch` route `backend=` through
`pystatistics.core.compute.backend.resolve_backend`, exposing only the backend
values they can honor (honest subset) with precision in the string — no ad-hoc
`'cpu'/'gpu'/'auto'` sets and no internal `use_fp64`/MPS float32 floor reached
outside the resolver. `gee` already complies.

### B5 — Epidemiology measure values are descriptive

`epi.mantel_haenszel`'s `measure` takes descriptive hyphenated values
`'odds-ratio'` / `'risk-ratio'`, never the caps abbreviations `'OR'` / `'RR'`
(A1). The same applies to any other cryptic option value surfaced during the
sweep (e.g. `power_anova_factorial`'s `effect` values `'main_A'`/`'main_B'` →
`'main-a'`/`'main-b'`).

---

## 2.0 migration table (old → new → reason)

Breaking interface changes, shipped together as **pystatsbio 2.0**. No
deprecation shim. Statistical results are unchanged.

### Parameters & option values

| Old | Module / function | → 2.0 | Reason |
|---|---|---|---|
| `type` | `power_t_test` | `test_type` | S3: shadows a builtin. |
| `d` / `f` / `h` | power (effect sizes) | `effect_size` | S1 / B1. |
| `k` | `power_anova_oneway` | `n_groups` | S1 / B1. |
| `hr` | `power_logrank` | `hazard_ratio` | S1 / B1. |
| `cv` | `power_crossover_be` | `coef_variation` | S1 / B1. |
| `sd` | power (mean tests) | `std` | S1 / B1. |
| `p1` / `p2` | power (proportion tests) | `prop1` / `prop2` | S1 / B1. |
| `"two.sided"` | power (`alternative`) | `"two-sided"` | S2 / A1. |
| `"one.sided"` | power (`alternative`) | `"one-sided"` | S2 / A1. |
| `"two.sample"` / `"one.sample"` | power (`test_type`) | `"two-sample"` / `"one-sample"` | S2 / A1. |
| `"main_A"` / `"main_B"` | `power_anova_factorial` (`effect`) | `"main-a"` / `"main-b"` | A1 / B5. |
| `measure="OR"` / `"RR"` | `epi.mantel_haenszel` | `"odds-ratio"` / `"risk-ratio"` | A1 / B5. |
| `use_fp64=` | `gee` | `backend='gpu_fp64'` | precision in the backend string (already landed). |

### Result objects (`…Result` → `…Solution`, all wrapping `Result[…Params]`)

`GEEResult`→`GEESolution`, `PowerResult`→`PowerSolution`,
`Epi2x2Result`→`Epi2x2Solution`, `MantelHaenszelResult`→`MantelHaenszelSolution`,
`MetaResult`→`MetaSolution`, `NCAResult`→`NCASolution`, `ROCResult`→`ROCSolution`,
`ROCTestResult`→`ROCTestSolution`, `DiagnosticResult`→`DiagnosticSolution`,
`CutoffResult`→`CutoffSolution`, `BatchAUCResult`→`BatchAUCSolution`,
`DoseResponseResult`→`DoseResponseSolution`,
`BatchDoseResponseResult`→`BatchDoseResponseSolution`,
`EC50Result`→`EC50Solution`, `RelativePotencyResult`→`RelativePotencySolution`,
`BMDResult`→`BMDSolution`, and the suffix-less `StandardizedRate`→
`StandardizedRateSolution` (every top-level return reads as a Solution, even
when the old name had no `…Result` suffix).

`EpiMeasure` is **not** a Solution: it is a nested `{estimate, ci, se}` value
object that appears as a *field* of `Epi2x2Solution` / `MantelHaenszelSolution`
(and is returned by internal helpers). It stays a lightweight frozen dataclass
— the `…Solution` envelope is for top-level public returns, not the value
objects nested inside them.

### Exceptions

Every bare `raise ValueError`/`RuntimeError` for validation across all modules →
`ValidationError`/`ConvergenceError` from `pystatistics.core.exceptions`;
`LambdaZEstimationError` re-rooted under `ConvergenceError` (B3).
