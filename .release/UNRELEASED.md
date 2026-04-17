# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py <version>`
> That script uses this file to build the CHANGELOG entry, bumps versions
> everywhere, and resets this file for the next cycle.

## Changes

- **GEE GPU backend** — `pystatsbio/gee/backends/gpu_fit.py`,
  `backends/_gpu_correlation.py`, `backends/_gpu_family.py`. The
  CPU GEE iterates over clusters in a Python loop, solving per-
  cluster working-covariance systems one at a time. The GPU backend
  groups clusters by size and batches the `(K_s, s, s)` working-
  covariance solves into a single `torch.linalg.solve`, then reduces
  the bread/score/sandwich-meat contributions across all clusters in
  vectorized tensor ops. The wins grow with cluster count:

    | shape (K, m, p)           | CPU      | GPU (numpy per-call) | GPU (DataSource) | speedup |
    |---------------------------|---------:|---------------------:|-----------------:|--------:|
    | geepack::dietox-like 72×12×4 | 7.1 ms  | 2.9 ms              | 2.9 ms          | 2.4×   |
    | K=100 m=5 p=4             | 11.8 ms  | 4.0 ms              | 4.0 ms          | 2.9×   |
    | K=500 m=5 p=4             | 57.9 ms  | 3.7 ms              | 3.8 ms          | 15.4×  |
    | K=1000 m=5 p=4            | 115.5 ms | 3.8 ms              | 3.9 ms          | 30.0×  |
    | K=2000 m=8 p=6            | 173.3 ms | 3.9 ms              | 4.0 ms          | 43.5×  |
    | K=5000 m=10 p=8           | 454.2 ms | 6.0 ms              | 6.7 ms          | 67.5×  |

  Supports all four CPU correlation structures (independence,
  exchangeable, AR(1), unstructured) and the four supported GLM
  families (gaussian/identity, binomial/logit, poisson/log,
  gamma/inverse). Unequal cluster sizes handled via size-grouped
  batching (one batched solve per distinct cluster size). Default
  precision is FP32 (`use_fp64=False`); FP64 available on CUDA for
  machine-precision parity with CPU.

- **GEE accepts torch.Tensor input** — `gee()` now accepts either
  numpy arrays or `torch.Tensor` for `y`, `X`, and `cluster_id`.
  Device-resident tensors (from `DataSource.from_arrays(...).to('cuda')`)
  skip per-fit H2D transfer of the design matrix, following the same
  convention as `pca()` and `multinom()`. Tensor input infers
  `backend='gpu'`; explicit `backend='cpu'` with a GPU tensor raises
  (Rule 1: no silent device migration).

- **Two-tier validation for GEE** — CPU path remains validated
  against R `geepack::geeglm()`. GPU path is validated against CPU at
  the `GPU_FP32` tolerance tier (rtol = 1e-4, atol = 1e-5) on
  coefficients and robust SE; CUDA FP64 matches CPU to machine
  precision. `TestGeeGPU` in `tests/gee/test_gee.py` adds the standard
  7 GPU-backend tests mirroring the existing `TestMultinomGPU` and
  `TestPCAGPU` suites.
