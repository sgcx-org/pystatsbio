# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py <version>`
> That script uses this file to build the CHANGELOG entry, bumps versions
> everywhere, and resets this file for the next cycle.

## Changes

- `batch_auc` now runs on Apple Silicon GPUs (MPS) with torch >= 2.13:
  `backend='gpu'` uses MPS instead of raising `RuntimeError`, and
  `backend='auto'` picks MPS (float32) over the CPU on MPS-only machines.
  PyTorch 2.13 replaced the MPSGraph-routed `scatter_add_` — the op the
  old prohibition was built on (~150 ms/call, ~20 s end-to-end for 5,000
  markers) — with native Metal kernels; re-benchmarked end-to-end
  2026-08-12 on an M2 Max (torch 2.13.0), MPS is 2.6–67x FASTER than the
  CPU across 100–20,000 markers × 500–1,155 samples (e.g. 5,000×500:
  0.012 s vs 0.67 s CPU), and matches the CPU float64 reference at the
  `GPU_FP32` tier (rtol=1e-4, atol=1e-5) on AUC and DeLong SE in every
  grid cell, including heavily tied data. On torch < 2.13, `'gpu'` still
  raises (now naming the installed torch and the >= 2.13 requirement)
  and `'auto'` still routes to CPU — gated by a local
  `_mps_native_kernels()` predicate in `pystatsbio/diagnostic/_batch.py`.
  `_batch_auc_gpu` takes the device type from the resolved backend
  instead of hardcoding CUDA, and casts to float32 host-side before
  transfer (MPS has no float64). Measurement details and gate rationale:
  `docs/GPU_BACKEND_NOTES.md`.

- `gee(backend='gpu')` now works on Apple Silicon (MPS). Two defects
  made the GEE GPU backend crash on every MPS machine even though it
  was written to support MPS float32 (it explicitly rejects only
  MPS + fp64): (1) the independence-IRLS initialization in
  `pystatsbio/gee/backends/gpu_fit.py` used `torch.linalg.lstsq`,
  which has never been implemented on MPS (`NotImplementedError`
  through torch 2.13) — on non-CUDA devices the weighted least squares
  now solves the normal equations `X'WX beta = X'Wz` by Cholesky (SPD
  for full-rank X; CUDA keeps QR-based `lstsq` unchanged); (2) the
  return path cast results to float64 on-device, which MPS cannot
  represent — results now transfer to host first and widen there
  (numerically identical on CUDA; fp32→fp64 widening is exact).
  Validated on an M2 Max (torch 2.13.0): MPS matches the CPU float64
  reference within the `GPU_FP32` tier (rtol=1e-4, atol=1e-5; observed
  coefficient/robust-SE deviations ~1e-8) at n up to 20,000.
  Disclosure: MPS is ~12x SLOWER than the CPU end-to-end at every
  shape tested (e.g. n=20,000, p=11: 3.2 s vs 0.25 s) because the fit
  loop is bound by batched `torch.linalg.solve`, still slow on Metal —
  explicit `backend='gpu'` honors the device request (useful for
  MPS-resident DataSource pipelines); see `docs/GPU_BACKEND_NOTES.md`.

- `gee(backend='auto')` no longer crashes on MPS-only machines. The
  dispatcher routed `'auto'` to any detected GPU, so on Apple Silicon
  it fell into the MPS path (and the `lstsq` crash above) instead of
  the documented "CUDA if present, else CPU" behavior. `'auto'` now
  never resolves to MPS for gee — matching the gee docstring and the
  core pystatistics fitting-function policy, and the right call given
  the measured MPS slowdown (unlike `batch_auc`, where a measured MPS
  win justifies its local auto-picks-MPS override).

- Fixed `TestGeeGPU` hardcoding `DataSource.to("cuda")` in
  `test_gpu_datasource_input_matches_gpu_numpy` and
  `test_gpu_tensor_with_cpu_backend_raises`, which failed on MPS-only
  machines: both now target the available GPU device (CUDA preferred,
  else MPS). Added `test_auto_backend_routes_mps_to_cpu` and
  `test_gpu_fp64_on_mps_raises` (both monkeypatch-driven, so they run
  on any hardware) covering the two policy decisions above. The gee
  suite is green on Apple Silicon: 64 passed, 1 skipped (the fp64
  test, which requires CUDA).

- Docs: corrected hardware attribution throughout — the Apple Silicon
  benchmark machine was always a Mac Studio **M2 Max** ("Mainframe",
  since renamed "Powerhouse"; the fleet never included an M2 Ultra).
  Fixed in `docs/GPU_BACKEND_NOTES.md`, `docs/Forge.md`, and the 2.0.0
  `CHANGELOG.md` entry ("1350x slower on M2 Ultra" → "M2 Max"). Numbers
  are unchanged; only the chip name was wrong.

- Docs: fixed docstring formatting so the Sphinx API reference renders
  cleanly. Set ``napoleon_use_ivar`` so dataclass "Attributes" sections
  render as inline field lists instead of duplicate object descriptions
  (removed ~73 "duplicate object description" warnings), and escaped the
  ``1 / |RD|`` expression in ``Epi2x2Params`` so it is no longer misread
  as an undefined RST substitution. No behavior, signature, or
  documented-value changes — docstring/config formatting only. The docs
  now build warning-free under ``sphinx-build -W``.
