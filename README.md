# PyStatsBio

Biotech and pharmaceutical statistical computing for Python.

Built on [PyStatistics](https://github.com/sgcx-org/pystatistics) for the general statistical
computing layer. PyStatsBio provides domain-specific methods for the drug development pipeline:
dose-response modeling, sample size/power, diagnostic accuracy, and non-compartmental
pharmacokinetics.

---

## Design Philosophy

PyStatsBio follows the same principles as PyStatistics:

1. **Fail fast, fail loud** — no silent fallbacks or "helpful" defaults
2. **Explicit over implicit** — require parameters, don't assume intent
3. **R-level validation** — every function is validated against a named R reference

Each function states exactly which R function it replicates and to what tolerance.

---

## Quick Start

```python
# --- Clinical trial power / sample size ---
from pystatsbio import power

# Solve for sample size (two-sample t-test)
result = power.power_t_test(d=0.5, power=0.80, alpha=0.05, type="two.sample")
print(result.n)          # per-group sample size
print(result.summary())

# Solve for power given n
result = power.power_t_test(n=64, d=0.5, alpha=0.05, type="two.sample")
print(result.power)

# Paired t-test
result = power.power_paired_t_test(d=0.3, power=0.80, alpha=0.05)
print(result.n)

# Proportions
result = power.power_prop_test(p1=0.30, p2=0.50, power=0.80, alpha=0.05)
print(result.n)

# Survival (log-rank)
result = power.power_logrank(
    p1=0.60, p2=0.40, accrual=12, follow=24, power=0.80, alpha=0.05
)
print(result.n_events, result.n_total)

# Non-inferiority for means
result = power.power_noninf_mean(
    delta=0.0, sigma=1.0, margin=0.5, power=0.80, alpha=0.05
)
print(result.n)

# Crossover bioequivalence (PowerTOST method)
result = power.power_crossover_be(cv=0.20, power=0.80, alpha=0.05)
print(result.n)          # subjects per sequence

# Cluster-randomized trial
result = power.power_cluster(
    d=0.5, icc=0.05, m=20, power=0.80, alpha=0.05
)
print(result.n_clusters)


# --- Dose-response modeling ---
import numpy as np
from pystatsbio import doseresponse

dose = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
response = np.array([2.1, 3.5, 12.0, 48.0, 87.5, 97.8])

# Fit 4PL (LL.4) model
result = doseresponse.fit_drm(dose, response, model="LL.4")
print(result.params)     # CurveParams(b, c, d, e) — Hill slope, lower, upper, EC50
print(result.ec50)
print(result.summary())

# Fit 5PL (asymmetric)
result = doseresponse.fit_drm(dose, response, model="LL.5")
print(result.params)

# Extract EC50 with confidence interval
ec50_result = doseresponse.ec50(result)
print(ec50_result.ec50, ec50_result.ci_lower, ec50_result.ci_upper)

# Relative potency (reference vs. test compound)
ref_result = doseresponse.fit_drm(dose, response, model="LL.4")
test_result = doseresponse.fit_drm(dose * 3, response, model="LL.4")
rp = doseresponse.relative_potency(ref_result, test_result)
print(rp.ratio, rp.ci_lower, rp.ci_upper, rp.parallel)

# Benchmark dose (BMD/BMDL)
bmd_result = doseresponse.bmd(result, bmr=0.10)
print(bmd_result.bmd, bmd_result.bmdl)

# Batch fitting (GPU-accelerated for HTS)
# responses: (n_curves, n_doses) array
responses = np.random.rand(500, 6) * 100
batch = doseresponse.fit_drm_batch(dose, responses, model="LL.4", backend="auto")
print(batch.ec50)        # shape (500,)
print(batch.params)      # CurveParams with shape-(500,) arrays


# --- Diagnostic accuracy ---
from pystatsbio import diagnostic

scores = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.15, 0.6, 0.75, 0.55, 0.95])
labels = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 1])

# ROC curve + AUC
roc_result = diagnostic.roc(scores, labels)
print(roc_result.auc, roc_result.ci_lower, roc_result.ci_upper)
print(roc_result.sensitivity, roc_result.specificity)

# Compare two biomarkers (DeLong test)
scores2 = np.random.rand(10)
test_result = diagnostic.roc_test(scores, scores2, labels)
print(test_result.p_value, test_result.summary())

# Full diagnostic accuracy at a threshold
da = diagnostic.diagnostic_accuracy(scores, labels, threshold=0.5)
print(da.sensitivity, da.specificity, da.ppv, da.npv, da.lr_pos, da.lr_neg)

# Optimal cutoff selection
cutoff = diagnostic.optimal_cutoff(roc_result, method="youden")
print(cutoff.threshold, cutoff.sensitivity, cutoff.specificity, cutoff.youden_index)

# Batch AUC for biomarker panel screening
# panel: (n_biomarkers, n_subjects) array
panel = np.random.rand(200, 100)
batch_auc = diagnostic.batch_auc(panel, labels[:100] if len(labels) >= 100 else np.random.randint(0, 2, 100))
print(batch_auc.auc)     # shape (200,) — one AUC per biomarker


# --- Pharmacokinetics (NCA) ---
from pystatsbio import pk

time = np.array([0, 0.5, 1, 2, 4, 6, 8, 12, 24])
conc = np.array([0, 8.5, 12.1, 10.4, 7.2, 4.9, 3.1, 1.4, 0.3])

result = pk.nca(time, conc, route="ev", dose=100.0)
print(result.cmax, result.tmax)
print(result.auc_last, result.auc_inf)
print(result.half_life)
print(result.cl, result.vd)
print(result.aumc_last, result.mrt)
print(result.summary())
```

---

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| `power/` | Complete | Sample size and power for clinical trial designs |
| `doseresponse/` | Complete | 4PL/5PL curve fitting, EC50, relative potency, BMD, batch HTS |
| `diagnostic/` | Complete | ROC, AUC, sensitivity/specificity, optimal cutoff, batch biomarker |
| `pk/` | Complete | Non-compartmental PK analysis (NCA): AUC, Cmax, CL, Vd, MRT |

### `power` — Sample Size and Power

| Function | R equivalent |
|----------|--------------|
| `power_t_test()` | `pwr::pwr.t.test()` |
| `power_paired_t_test()` | `pwr::pwr.t.test(type="paired")` |
| `power_prop_test()` | `pwr::pwr.2p.test()` |
| `power_fisher_test()` | `TrialSize::TwoSampleProportion.Equality()` |
| `power_logrank()` | `gsDesign::nSurv()` |
| `power_anova_oneway()` | `pwr::pwr.anova.test()` |
| `power_anova_factorial()` | `TrialSize::FactorialDesign()` |
| `power_noninf_mean()` | `TrialSize::TwoSampleMean.NIS()` |
| `power_noninf_prop()` | `TrialSize::TwoSampleProportion.NIS()` |
| `power_equiv_mean()` | `TrialSize::TwoSampleMean.Equivalence()` |
| `power_superiority_mean()` | `TrialSize::TwoSampleMean.Superiority()` |
| `power_crossover_be()` | `PowerTOST::sampleSize()` |
| `power_cluster()` | `samplesize::n.twogroup()` with ICC |

### `doseresponse` — Dose-Response Modeling

| Function | R equivalent |
|----------|--------------|
| `fit_drm()` | `drc::drm()` |
| `fit_drm_batch()` | vectorized `drc::drm()` |
| `ec50()` | `drc::ED()` |
| `relative_potency()` | `drc::EDcomp()` with parallelism test |
| `bmd()` | `drc::bmd()` / `BMDS` |

Models: `LL.4` (4PL), `LL.5` (5PL), `W1.4` (Weibull-1), `W2.4` (Weibull-2),
`BC.4` (Brain-Cousens hormesis).

### `diagnostic` — Diagnostic Accuracy

| Function | R equivalent |
|----------|-------------|
| `roc()` | `pROC::roc()` with DeLong CI |
| `roc_test()` | `pROC::roc.test()` (DeLong) |
| `diagnostic_accuracy()` | `epiR::epi.tests()` |
| `optimal_cutoff()` | `OptimalCutpoints::optimal.cutpoints()` |
| `batch_auc()` | vectorized `pROC::auc()` |

### `pk` — Non-Compartmental Analysis

| Function | R equivalent |
|----------|-------------|
| `nca()` | `PKNCA::pk.nca()` / `NonCompart::sNCA()` |

AUC methods: `linear`, `log`, `linear-up/log-down` (FDA default).
Routes: `iv` (intravenous), `ev` (extravascular).

---

## Installation

```bash
pip install pystatsbio

# With GPU support (requires PyTorch)
pip install pystatsbio[gpu]

# Development
pip install pystatsbio[dev]
```

Requires Python 3.11+. Core dependencies: `pystatistics`, `numpy`, `scipy`.

---

## License

MIT

## Author

Hai-Shuo (contact@sgcx.org)
