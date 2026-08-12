# GPU Backend Notes

Hard-won knowledge about GPU behavior across CUDA and MPS backends.
Written because people won't know this unless they hit the specific
issue in the specific way.

---

## CUDA vs MPS: When Metal Falls Off a Cliff

### The Short Version

NVIDIA CUDA and Apple MPS are not interchangeable GPU backends. Certain
memory access patterns that are fast on CUDA are **catastrophically slow**
on MPS — not 2x slower, but 1000x slower. If your algorithm uses
`scatter_add_` with sparse, irregular bucket IDs, MPS is the wrong backend.

The pystatistics and pystatsbio libraries detect this and fail fast rather
than silently delivering a 15-minute wait.

---

### The Pattern That Breaks MPS: `scatter_add_` with Sparse Targets

**What the code needs to do** (example: midrank computation for batch AUC
with 5,000 biomarkers × 500 patients):

1. Sort each column — MPS is fine at this.
2. Find groups of tied values — MPS fine.
3. **For each tie group, sum up the ranks of its members and count them**
   — this is `scatter_add_`.
4. Divide to get the average rank per group — MPS fine.

Step 3 is the problem. `scatter_add_` says: "I have 2.5 million items.
Each one has a bucket ID. Add each item's value into its bucket." The
bucket IDs are sparse and irregular — each column has different tie
patterns.

### Why CUDA Handles This Fine

NVIDIA GPUs have **atomic operations in shared memory**. When 1,000
threads simultaneously try to add values to the same bucket, CUDA
serializes them at the hardware level using atomic compare-and-swap — a
few clock cycles per conflict. The GPU's memory controller is designed
for this pattern.

**Measured**: ~0.05 ms per `scatter_add_` call on RTX 5070 Ti.

### Why MPS Is Catastrophically Slow

Apple's Metal GPU architecture was designed for **graphics** — rendering
pixels, texture mapping, vertex shading. These workloads are *regular*:
every pixel does the same work, reads from the same textures, writes to
a predictable framebuffer location.

`scatter_add_` is the opposite: thousands of threads writing to *random,
unpredictable* memory locations with *read-modify-write* semantics.
Metal has to:

1. **Serialize conflicting writes** — Metal's atomic support is weaker
   than CUDA's. When two threads hit the same bucket, one stalls.
2. **The memory access pattern defeats the cache** — buckets are spread
   across a 2.5M-element array. Each access is essentially a cache miss.
3. **Metal's command encoding adds overhead** — Metal batches GPU
   commands through a command buffer; every scatter operation requires a
   full encode-dispatch-wait cycle, unlike CUDA's inline kernel execution.

**Measured**: ~150 ms per `scatter_add_` call on M2 Max — **3,000x
slower** than the same operation on CUDA.

### The Cascading Effect

The midrank algorithm calls `scatter_add_` 3 times (sum ranks, count
members, map back) inside `_midranks_vectorized`, which is itself called
3 times (pooled ranks, case within-ranks, control within-ranks). The
flattening trick that makes CUDA fast (process all 5,000 columns in one
scatter) makes MPS *worse* because it creates a single huge sparse array.

**Result**: Batch AUC for 5,000 markers takes ~0.015s on CUDA, ~20s on
MPS, vs ~0.9s on CPU. MPS is 22x slower than CPU.

> **Update (2026-08-12, torch 2.13.0):** the numbers above are from the
> April 2026 stack and no longer hold. PyTorch 2.13 (July 2026) replaced
> the MPSGraph-routed MPS `scatter`/`gather` with hand-written native
> Metal kernels, and the end-to-end re-benchmark on Powerhouse (M2 Max,
> macOS 26.5, torch 2.13.0) puts batch AUC for 5,000 markers × 500
> samples at **0.012 s on MPS vs 0.67 s on CPU — MPS is now ~55x FASTER
> than the CPU**, in the same league as the historical CUDA number
> (0.015 s). Full grid below. Two forensic notes:
>
> 1. On this machine the pathology is gone at *every* testable torch
>    (2.9, 2.10, 2.11, 2.12, 2.13 all run the 5,000×500 workload in
>    35–38 ms warm; `scatter_add_` on 2.5M items is ≤ 1.2 ms with random
>    buckets). The April 2026 measurement was made on this same physical
>    machine (then named "Mainframe", since renamed Powerhouse), so
>    hardware is a controlled variable: MPSGraph ships with the OS, and
>    the macOS/Metal stack is the only thing that changed. The original
>    ~150 ms/call was a property of that stack, not of the torch version.
> 2. torch >= 2.13 is still the right gate: it is the first version whose
>    scatter/gather speed is guaranteed by torch itself (kernels bundled
>    with the wheel) rather than by whatever MPSGraph the host OS ships.
>    Older torch on an older OS may still hit the ~20 s cliff, and Python
>    cannot see the Metal framework version.

---

## What We Did About It

### pystatsbio `batch_auc` (updated 2026-08-12)

- `backend='gpu'` on CUDA: uses the vectorized `scatter_add_` kernel.
  49-63x faster than CPU.
- `backend='gpu'` on MPS with torch >= 2.13: **supported** — same
  vectorized kernel, 2.6–67x faster than CPU across the benchmark grid,
  validated against the CPU fp64 reference at the `GPU_FP32` tier
  (rtol=1e-4, atol=1e-5). Gated by `_mps_native_kernels()` in
  `pystatsbio/diagnostic/_batch.py` (the
  `pystatistics.core.compute.device.mps_native_kernels` predicate
  pattern).
- `backend='gpu'` on MPS with torch < 2.13: **raises `RuntimeError`**
  with an actionable message (upgrade torch, or use `backend='cpu'`).
  Fail fast, fail loud (Coding Bible Rule 1).
- `backend='auto'` on MPS: picks MPS (float32) on torch >= 2.13, CPU
  below — auto means "best available", and the version gate is what
  decides which one that is. This deliberately overrides the
  pystatistics core policy that `'auto'` never resolves to MPS; the
  override lives in `batch_auc` itself, next to the measurement that
  justifies it.

### pystatsbio `gee` (updated 2026-08-12)

The GEE GPU backend was written with MPS in mind (it rejects only
MPS + fp64) but had never actually run on MPS. Two latent defects, both
fixed 2026-08-12:

- **`torch.linalg.lstsq` has never been implemented on MPS** (still
  `NotImplementedError` on torch 2.13). The independence-IRLS
  initialization used it for weighted least squares, so
  `gee(backend='gpu')` crashed on every MPS machine. On non-CUDA
  devices the initialization now solves the normal equations
  `X'WX beta = X'Wz` by Cholesky (SPD for full-rank X; the squared
  condition number is fine for a starting value the GEE iteration
  refines). CUDA keeps the QR-based `lstsq` unchanged.
- **MPS tensors cannot hold float64**, so the return path's on-device
  `.to(torch.float64)` casts also raised. Results now transfer D2H
  first and widen on the host — numerically identical on CUDA
  (fp32→fp64 widening is exact).

Status after the fix, measured on an M2 Max (torch 2.13.0):

- `backend='gpu'` on MPS: **works**, and matches the CPU fp64 reference
  well inside the `GPU_FP32` tier (max coefficient deviation ~1e-8 at
  n up to 20,000). But it is **~12x slower than the CPU** end-to-end at
  every shape tested (n=250: 91 ms vs 6 ms; n=20,000, p=11: 3.2 s vs
  0.25 s) — the fit loop is bound by batched `torch.linalg.solve` over
  the per-cluster working covariances, which is one of the ops still
  MPSGraph-slow on torch 2.13. Explicit `backend='gpu'` honors the
  caller's device choice (useful when the data already lives on MPS via
  a DataSource); the numbers above are the disclosure.
- `backend='auto'` on MPS-only machines: **CPU**, per the core
  pystatistics policy that `'auto'` means CUDA-else-CPU and never
  resolves to MPS. Unlike `batch_auc` there is no measured MPS win here
  to justify a local override — the same measurement that lifted the
  `batch_auc` ban keeps gee's `'auto'` on the CPU. (Before the fix,
  `'auto'` on an MPS-only machine crashed into the `lstsq` hole; the
  policy check did not exist.)

### Could It Be Fixed for MPS?

**It fixed itself** — torch 2.13's native Metal scatter/gather kernels
made the existing CUDA-shaped algorithm fast on MPS with no code change
(see the update note above). The alternatives below were the options
considered while the old kernels were the constraint; they are kept for
the record and for any future op that hits the same wall:

- **Don't scatter.** Use `torch.unique_consecutive` on the sorted data
  to group ties *in-place* without random memory access. Then use
  `cumsum` to compute group sizes and rank sums. All operations are
  sequential/streaming, which Metal handles well.
- **Or process columns one at a time** — 5,000 iterations of a
  500-element sort + midrank is fast even in Python because each
  operation is small and cache-friendly.
- **Or use the CPU.** `scipy.stats.rankdata` is Cython running in L1
  cache on a single core. For 500 elements it takes ~0.13ms. Even
  looping 5,000 times: 650ms. The CPU wins not because it's "faster"
  than the GPU, but because the problem isn't parallel enough to
  justify GPU dispatch overhead for this workload shape.

The fundamental issue isn't Apple Silicon's compute power — it's that
Metal's programming model doesn't expose the low-level atomic memory
operations that make scatter patterns efficient on CUDA. This is a
deliberate Apple design choice (simpler programming model, optimized for
graphics/ML inference, not scientific computing). It may improve in
future Metal versions, but today, if your algorithm requires
`scatter_add_` into sparse targets, MPS is the wrong backend.

---

## General Rules for GPU Backend Selection

Based on validation across Mac Studio M2 Max and Linux RTX 5070 Ti:

### Operations That Are Fast on Both CUDA and MPS

- Matrix multiply (`X.T @ X`, `X @ beta`)
- Cholesky decomposition (`torch.linalg.cholesky`, including batched)
- Element-wise operations (add, multiply, exp, log)
- Reductions (sum, mean, max along a dimension)
- `argsort` (used for ranking)
- `torch.rand`, `torch.randint` (random number generation)

### Operations That Are Fast on CUDA but Slow on MPS

- **`scatter_add_` with sparse/irregular indices** — the specific killer
  *(fixed on torch >= 2.13 / current macOS — see the update note above;
  ≤ 1.2 ms for 2.5M items where it used to be ~150 ms)*
- `scatter_` in general with non-contiguous write patterns
- Any operation that requires atomic read-modify-write to random locations
- Operations that create very large intermediate tensors with irregular
  access patterns
- **The dense-solve family**: `torch.linalg.solve`, `solve_triangular`,
  `cholesky_solve`, `inv` — still MPSGraph-routed and 1-3 orders of
  magnitude off CUDA on torch 2.13 (unlike the factorization itself,
  which is fast). This is what keeps gee's MPS path ~12x behind the
  CPU. `torch.linalg.lstsq` and `eigh` are not implemented on MPS at
  all — they raise `NotImplementedError`.

### When GPU Wins Over CPU

- **Large matrix operations**: n × p with n > 10,000 and p > 50.
  The GPU amortizes transfer overhead. Below this, CPU is often faster.
- **Embarrassingly parallel tasks**: R=50,000 permutation tests where
  each permutation is independent. GPU computes all R at once.
- **Batch operations**: fitting 5,000 dose-response curves simultaneously,
  computing AUC for 20,000 genes at once.

### When CPU Wins Over GPU

- **Small problems**: n < 1,000. GPU launch overhead dominates.
- **Sequential algorithms**: iterative methods where each step depends
  on the previous (e.g., Cox PH Newton-Raphson with few iterations).
- **Sparse scatter patterns on MPS**: see above.
- **User-supplied Python callbacks**: bootstrap/permutation with arbitrary
  statistic functions — the function runs on CPU regardless.

---

## Benchmark Reference

All benchmarks measured on Forge (RTX 5070 Ti, CUDA 12.0) and Mainframe
(Mac Studio M2 Max — the same machine since renamed Powerhouse) during
the April 2026 Linux/NVIDIA validation. Historical note: these numbers
were long mis-attributed to an "M2 Ultra"; no such machine ever existed
in the fleet.

### Regression (pystatistics)

| Problem | CPU | GPU (CUDA) | Speedup |
|---------|-----|------------|---------|
| OLS 500K × 200 | 5.4s | 0.13s | **42x** |
| OLS 1M × 100 | 5.4s | 0.12s | **44x** |
| GLM binomial 50K × 100 | 0.4s | 0.08s | **5x** |

### Batch AUC (pystatsbio)

| Markers × Samples | CPU | GPU (CUDA) | GPU (MPS) | CUDA Speedup |
|--------------------|-----|------------|-----------|--------------|
| 100 × 1,155 | 0.018s | 0.9s | N/A | 0.02x (CPU wins) |
| 1,000 × 1,155 | 0.18s | 0.003s | N/A | **63x** |
| 20,000 × 1,155 | 3.6s | 0.074s | N/A | **49x** |

### Batch AUC on MPS, torch 2.13 re-benchmark (2026-08-12)

Measured on Powerhouse (M2 Max, macOS 26.5, torch 2.13.0), warm medians
through the public `batch_auc` API; every cell passes the `GPU_FP32`
check against the CPU fp64 reference (worst max-rel-diff 1.8e-5 vs
rtol 1e-4). First-ever MPS call pays ~0.4–0.7 s of one-time init.

| Markers × Samples | CPU | GPU (MPS) | MPS Speedup |
|--------------------|--------|--------|-------------|
| 100 × 500 | 0.014s | 0.005s | 2.6x |
| 100 × 1,155 | 0.021s | 0.006s | 3.9x |
| 1,000 × 500 | 0.135s | 0.006s | **23x** |
| 1,000 × 1,155 | 0.212s | 0.009s | **24x** |
| 5,000 × 500 | 0.67s | 0.012s | **55x** |
| 5,000 × 1,155 | 1.09s | 0.027s | **40x** |
| 20,000 × 500 | 2.67s | 0.040s | **67x** |
| 20,000 × 1,155 | 4.30s | 0.102s | **42x** |

### Permutation Test (pystatistics)

| n samples | R perms | CPU | GPU (CUDA) | Speedup |
|-----------|---------|-----|------------|---------|
| 1,000 | 50,000 | 1.4s | 0.28s | **5x** |
| 10,000 | 50,000 | 6.7s | 0.29s | **23x** |
| 50,000 | 50,000 | 33s | 1.4s | **23x** |
