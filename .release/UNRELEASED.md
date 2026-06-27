# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py <version>`
> That script uses this file to build the CHANGELOG entry, bumps versions
> everywhere, and resets this file for the next cycle.

## Changes

- **`gee()`: GPU precision now lives in the `backend=` string; `use_fp64` is
  removed.** Mirroring the pystatistics 4.0 convention, `gee(..., backend=)`
  now accepts `'cpu'` (float64), `'gpu'` (float32), `'gpu_fp64'` (CUDA float64,
  raises on Apple Silicon/MPS), or `'auto'`. The separate `use_fp64=` keyword
  is gone — pass `backend='gpu_fp64'` for double precision. Breaking change for
  callers that passed `use_fp64=`. The GPU-unavailable / fp64-on-MPS errors now
  reuse pystatistics' canonical messages (`core.compute.backend`).
- **Requires `pystatistics>=4.0`** (was `>=0.1.0`): uses the 4.0 backend-string
  precision convention and `core.compute.backend`.
