# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py <version>`
> That script uses this file to build the CHANGELOG entry, bumps versions
> everywhere, and resets this file for the next cycle.

## Changes

- Docs: fixed docstring formatting so the Sphinx API reference renders
  cleanly. Set ``napoleon_use_ivar`` so dataclass "Attributes" sections
  render as inline field lists instead of duplicate object descriptions
  (removed ~73 "duplicate object description" warnings), and escaped the
  ``1 / |RD|`` expression in ``Epi2x2Params`` so it is no longer misread
  as an undefined RST substitution. No behavior, signature, or
  documented-value changes — docstring/config formatting only. The docs
  now build warning-free under ``sphinx-build -W``.
