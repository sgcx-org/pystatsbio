# SGC-Bio — Biotech Plugin Pack Directive

**Version:** 1.0
**Status:** Active
**Repo:** `/dev/sgcbio`
**Depends on:** SGC-Core (`/dev/sgccore`), PyStatsBio, PyStatistics
**Scope:** SGC-Bio is the first vertical plugin pack for SGC-Core, targeting
biotech/pharma. This document covers market strategy, monetization, the plugin
catalog, and bio-specific decisions.

For platform architecture (auth, plugins, jobs, audit, UI), see
`SGC_CORE_DIRECTIVE.md`.

---

## 1. What SGC-Bio Is

SGC-Bio is a **plugin pack** that registers pystatsbio analysis functions into
SGC-Core's plugin registry. It is the first vertical on the platform.

```
SGC-Core              (domain-agnostic platform — /dev/sgccore)
    ↑ registers plugins into
SGC-Bio               (biotech plugin pack — /dev/sgcbio)
    ↑ calls
PyStatsBio            (open-core biotech statistics — pip)
    ↑ calls
PyStatistics          (open-core general statistics — pip)
```

**SGC-Bio does not contain platform infrastructure.** No auth, no job queue, no
audit log, no UI components. Those live in SGC-Core.

**SGC-Bio contains:**

- Plugin definitions that wire pystatsbio functions to SGC-Core's registry
- Parameter schemas (JSON Schema) for each analysis type
- Result schemas for each analysis type
- Bio-specific metadata: categories (PK, Dose-Response, Diagnostic, Power),
  display names, descriptions
- Execution metadata (backend requirements: cpu, gpu, either)
- Thin adapter functions that bridge pystatsbio's API to the plugin callable
  contract (DatasetHandle → numpy arrays, parameter mapping, result
  serialization)

**SGC-Bio does not sell math.** The math is in pystatsbio (open-core, pip).
SGC-Bio sells the platform experience: governance, auditability, structured
execution, GPU routing — via SGC-Core.

---

## 2. Market Strategy

### Target Customer

Kendall Square biotech startups, Cambridge/Boston MA.

- 5-50 employees (initial sweet spot: 5-10)
- Series A or B funded ($20-50M+ in the bank)
- 1-3 biostatisticians on staff
- Currently running: hand-rolled R scripts, internal Excel pipelines, script
  chaos with no governance

### Disruption Strategy

Classic bottom-up disruption:

- **Incumbent:** SAS. Expensive ($100k+/year), won't take calls from 10-person
  biotechs, overkill for their needs.
- **Current alternative:** R scripts on someone's laptop. No audit trail, no
  version control on analyses, no reproducibility guarantees. The
  biostatistician is the most expensive bottleneck.
- **Our position:** Target the customers SAS ignores. Provide governance,
  reproducibility, and structured execution at a price point that's a rounding
  error on their burn rate. Eat from the bottom up.

### Competitive Positioning

SGC-Bio competes with:

- Hand-rolled R scripts (no governance)
- Internal Excel pipelines (no reproducibility)
- Script chaos (no auditability)
- SAS licensing cost (prohibitive for small biotech)

SGC-Bio does **not** compete primarily on statistical novelty. It competes on:

- **Stability** — it works the same way every time
- **Reproducibility** — every analysis run is versioned and replayable
- **Auditability** — immutable log of who ran what, when, on which data
- **Infrastructure** — GPU routing, async execution, zero-install web access
- **Removing friction** — structured parameter entry instead of writing code

### Critical Framing

SGC-Bio is not "no-code statistics for beginners." The target user is a PhD
biostatistician. They pick the model, set the parameters, interpret the
results. The platform removes the infrastructure burden. The value proposition
is: **you focus on the science, we handle the infrastructure.**

---

## 3. Monetization Model

### Design Partners (Phase 1)

**$10,000 one-time perpetual license.**

- Target: 2-3 Kendall Square biotechs willing to co-design the product
- They get: perpetual license to the version they help shape, direct input on
  feature priority, early access
- We get: weekly feedback on real workflows, real pain points, real data shapes,
  and validation that the product solves actual problems
- This feedback is worth far more than the license revenue — it shapes what we
  build vs what we defer
- No annual renewal. Genuine thank-you for design partnership.
- Handshake deal, simple invoice. Do not over-lawyer a $10k deal.

### General Availability

**Annual enterprise license per organization.**

- No monthly plans (biotech budgets are annual)
- No one-time perpetual licenses (design partners excepted)
- Includes updates, maintenance, support

Pricing tiers (target):

| Tier | Price | Customer |
|------|-------|----------|
| Early biotech | $10-15k/year | 5-15 employees |
| Growth biotech | $25-50k/year | 15-50 employees |
| Enterprise pharma | Custom | 50+ employees |

**GPU compute is included** in the license at reasonable usage levels. No
metering, no usage tracking, no billing complexity.

---

## 4. Plugin Catalog

SGC-Bio registers the following pystatsbio analyses as SGC-Core plugins.
Each plugin has a parameter schema (for the UI form) and a result schema (for
the result viewer).

### Category: Power & Sample Size

| Plugin ID | Display Name | PyStatsBio Function | Backend |
|-----------|-------------|--------------------|----|
| `bio.power.t_test` | Two-Sample t-Test Power | `power.power_t_test` | cpu |
| `bio.power.paired_t_test` | Paired t-Test Power | `power.power_paired_t_test` | cpu |
| `bio.power.prop_test` | Proportion Test Power | `power.power_prop_test` | cpu |
| `bio.power.fisher_test` | Fisher Exact Test Power | `power.power_fisher_test` | cpu |
| `bio.power.logrank` | Log-Rank Survival Power | `power.power_logrank` | cpu |
| `bio.power.anova_oneway` | One-Way ANOVA Power | `power.power_anova_oneway` | cpu |
| `bio.power.anova_factorial` | Factorial ANOVA Power | `power.power_anova_factorial` | cpu |
| `bio.power.noninf_mean` | Non-Inferiority (Mean) Power | `power.power_noninf_mean` | cpu |
| `bio.power.noninf_prop` | Non-Inferiority (Prop) Power | `power.power_noninf_prop` | cpu |
| `bio.power.equiv_mean` | Equivalence (Mean) Power | `power.power_equiv_mean` | cpu |
| `bio.power.superiority_mean` | Superiority (Mean) Power | `power.power_superiority_mean` | cpu |
| `bio.power.crossover_be` | Crossover Bioequivalence Power | `power.power_crossover_be` | cpu |
| `bio.power.cluster` | Cluster-Randomized Power | `power.power_cluster` | cpu |

### Category: Dose-Response

| Plugin ID | Display Name | PyStatsBio Function | Backend |
|-----------|-------------|--------------------|----|
| `bio.dr.fit` | Dose-Response Curve Fit | `doseresponse.fit_drm` | cpu |
| `bio.dr.fit_batch` | Batch Dose-Response Fit | `doseresponse.fit_drm_batch` | either |
| `bio.dr.ec50` | EC50 Estimation | `doseresponse.ec50` | cpu |
| `bio.dr.relative_potency` | Relative Potency | `doseresponse.relative_potency` | cpu |
| `bio.dr.bmd` | Benchmark Dose | `doseresponse.bmd` | cpu |

### Category: Diagnostic Accuracy

| Plugin ID | Display Name | PyStatsBio Function | Backend |
|-----------|-------------|--------------------|----|
| `bio.diag.roc` | ROC Analysis | `diagnostic.roc` | cpu |
| `bio.diag.roc_test` | ROC Curve Comparison | `diagnostic.roc_test` | cpu |
| `bio.diag.accuracy` | Diagnostic Accuracy | `diagnostic.diagnostic_accuracy` | cpu |
| `bio.diag.cutoff` | Optimal Cutoff Selection | `diagnostic.optimal_cutoff` | cpu |
| `bio.diag.batch_auc` | Batch AUC (Biomarker Panel) | `diagnostic.batch_auc` | either |

### Category: Pharmacokinetics

| Plugin ID | Display Name | PyStatsBio Function | Backend |
|-----------|-------------|--------------------|----|
| `bio.pk.nca` | Non-Compartmental Analysis | `pk.nca` | cpu |

### Future Plugins (PyStatsBio Phase 2+)

- `bio.pk.nca_summary` — PK summary statistics (geometric mean, CV)
- `bio.equiv.*` — Bioequivalence analysis
- `bio.survival.*` — Survival analysis (Kaplan-Meier, Cox)
- `bio.mixed.*` — Mixed models (MMRM)

---

## 5. Plugin Implementation Pattern

Each plugin is a thin adapter between pystatsbio and SGC-Core's plugin
contract. Plugins receive a `DatasetHandle` (not a raw DataFrame) and declare
both parameter and result schemas. Results are validated against the result
schema by SGC-Core before storage — non-conforming results are rejected.

For the full plugin contract specification (DatasetHandle interface, execution
metadata, result validation, registry lifecycle), see `SGC_CORE_DIRECTIVE.md`
Section 2.

```python
# Example: bio/pk/nca_plugin.py

from sgccore.plugins import registry, DatasetHandle
from pystatsbio.pk import nca


NCA_PARAMETER_SCHEMA = {
    "type": "object",
    "properties": {
        "time_column": {
            "type": "string",
            "description": "Column name for time points",
        },
        "concentration_column": {
            "type": "string",
            "description": "Column name for concentration values",
        },
        "dose": {
            "type": "number",
            "minimum": 0,
            "description": "Administered dose (optional)",
            "default": None,
        },
        "route": {
            "type": "string",
            "enum": ["iv", "ev"],
            "default": "ev",
            "description": "iv (intravenous) or ev (extravascular/oral)",
        },
        "auc_method": {
            "type": "string",
            "enum": ["linear", "log-linear", "linear-up/log-down"],
            "default": "linear-up/log-down",
            "description": "AUC trapezoidal method",
        },
    },
    "required": ["time_column", "concentration_column"],
}


NCA_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "auc_last": {"type": "number"},
        "auc_inf": {"type": ["number", "null"]},
        "auc_pct_extrap": {"type": ["number", "null"]},
        "cmax": {"type": "number"},
        "tmax": {"type": "number"},
        "half_life": {"type": ["number", "null"]},
        "lambda_z": {"type": ["number", "null"]},
        "lambda_z_r_squared": {"type": ["number", "null"]},
        "clearance": {"type": ["number", "null"]},
        "vz": {"type": ["number", "null"]},
        "summary": {"type": "string"},
    },
    "required": ["auc_last", "cmax", "tmax", "summary"],
}


@registry.register(
    id="bio.pk.nca",
    name="Non-Compartmental Analysis",
    category="PK",
    description="AUC, Cmax, Tmax, half-life, clearance, Vz via NCA",
    version="1.0.0",
    execution={"backend": "cpu"},
    parameter_schema=NCA_PARAMETER_SCHEMA,
    result_schema=NCA_RESULT_SCHEMA,
)
def run_nca(params: dict, dataset: DatasetHandle) -> dict:
    time = dataset.column_as_array(params["time_column"])
    conc = dataset.column_as_array(params["concentration_column"])

    result = nca(
        time, conc,
        dose=params.get("dose"),
        route=params.get("route", "ev"),
        auc_method=params.get("auc_method", "linear-up/log-down"),
    )

    # Serialize NCAResult to dict for JSON storage.
    # Must conform to NCA_RESULT_SCHEMA — SGC-Core validates before storing.
    return {
        "auc_last": result.auc_last,
        "auc_inf": result.auc_inf,
        "auc_pct_extrap": result.auc_pct_extrap,
        "cmax": result.cmax,
        "tmax": result.tmax,
        "half_life": result.half_life,
        "lambda_z": result.lambda_z,
        "lambda_z_r_squared": result.lambda_z_r_squared,
        "clearance": result.clearance,
        "vz": result.vz,
        "summary": result.summary(),
    }
```

**Key points:**

- The plugin adapter is thin — it maps dataset columns to arrays, calls
  pystatsbio, serializes the result
- No statistical logic lives in the plugin. If pystatsbio doesn't have a
  function, add it to pystatsbio.
- Plugins receive `DatasetHandle`, not raw DataFrames — use
  `dataset.column_as_array()` to extract columns as numpy arrays
- Plugins must not import Django or access storage directly — only
  `sgccore.plugins` is allowed
- Parameter schemas reference dataset columns by name (the UI shows a column
  picker populated from the uploaded dataset)
- Result schemas define the expected output structure — SGC-Core validates
  the returned dict against this schema before storage
- The `execution` dict declares backend requirements (`"cpu"`, `"gpu"`, or
  `"either"`) — SGC-Core routes accordingly

---

## 6. GPU-Capable Plugins

Plugins with `execution={"backend": "either"}`:

- `bio.dr.fit_batch` — batch dose-response curve fitting via GPU
  Levenberg-Marquardt
- `bio.diag.batch_auc` — batch AUC computation for biomarker panels

These plugins call pystatsbio functions with `backend='gpu'` when routed to a
GPU worker. SGC-Core reads `execution.backend` from plugin metadata and routes
accordingly. The `"either"` value means the plugin benefits from GPU but works
fine on CPU — SGC-Core routes to GPU if available, falls back to CPU
transparently.

Inside the plugin adapter, the backend selection is passed to pystatsbio:

```python
# In the batch dose-response plugin adapter:
results = fit_drm_batch(
    dose_matrix, response_matrix,
    backend="gpu" if gpu_available else "cpu",
)
```

The worker communicates GPU availability to the plugin via context (TBD in
SGC-Core implementation). Pystatsbio handles the actual CPU/GPU dispatch
internally.

GPU compute is included in the license. No metering.

---

## 7. Bio-Specific Compliance Notes

- Biotech PK/dose-response data is **not** patient-identifiable in Phase 1.
  No HIPAA concerns.
- 21 CFR Part 11 (electronic signatures for regulatory submissions) is a
  future concern — add when pharma customers appear.
- The audit trail (provided by SGC-Core) satisfies biotech governance needs
  for Phase 1.

---

## 8. Repository Structure

```
sgcbio/
  plugins/
    bio/
      __init__.py       ← plugin pack registration entry point
      power/            ← power/sample size plugin definitions
      doseresponse/     ← dose-response plugin definitions
      diagnostic/       ← diagnostic accuracy plugin definitions
      pk/               ← PK plugin definitions
      schemas/          ← shared JSON Schema fragments (if needed)
  tests/
    plugins/            ← plugin registration + execution tests
  docs/
```

SGC-Bio is installed into the SGC-Core worker environment. At startup, the
worker imports `sgcbio.plugins.bio`, which registers all bio plugins into
SGC-Core's registry.

---

## 9. Development Principles (Bio-Specific)

1. **Never duplicate statistical logic.** If pystatsbio doesn't have a
   function, add it to pystatsbio, not here.
2. **Plugins are thin adapters.** They map dataset columns to arrays, call
   pystatsbio, serialize the result. Nothing more.
3. **Parameter schemas are the UI.** The schema IS the form definition. No
   frontend code in SGC-Bio.
4. **Test plugin registration.** Every plugin must have a test that registers
   it, feeds it sample data, and verifies the result structure.

---

## 10. Relationship to Other Repos

| Repo | Role | Dependency Direction |
|------|------|---------------------|
| `/dev/pystatistics` | General statistics | PyStatsBio depends on this |
| `/dev/pystatsbio` | Biotech statistics | SGC-Bio depends on this |
| `/dev/sgccore` | Platform | SGC-Bio depends on this (registry API) |
| `/dev/sgcbio` | Bio plugin pack | Top of the stack |

**SGC-Core never depends on SGC-Bio.** SGC-Bio depends on SGC-Core's plugin
registry API and on pystatsbio.

---

## 11. Development Machines

- **Powerhouse** (Mac Studio M2 Max, 96GB) — primary development, MPS GPU
- **Forge** (AMD Ryzen 5 7600X, RTX 5070 Ti, Ubuntu 24.04) — CUDA GPU,
  staging/CI target, production-like environment

---

## 12. Long-Term Vision

**Goal:** SGC-Bio becomes the dominant biostatistics platform for Kendall
Square biotech, then expands to mid-size biotech and pharma.

- Open-core pystatsbio builds trust and community
- SGC-Bio plugin pack + SGC-Core platform builds revenue
- Design partner relationships build product-market fit
- Bottom-up disruption of SAS starting from the customers they ignore
- The platform (SGC-Core) enables future verticals without rewrites

---

*This document governs SGC-Bio development. For platform architecture
(auth, plugins, jobs, audit, UI), see `SGC_CORE_DIRECTIVE.md`. Update this
document when strategy evolves.*
