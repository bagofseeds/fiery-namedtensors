# CLAUDE.md — fiery-namedtensors

Repo-specific guidance for coding agents. bagofseeds publishes two families of
PyTorch packages — standalone **`bagof-*`** packages and the **`fiery-*`**
namespace matches (this repo is one of the latter); they share the same
packaging, CI, docs, and workflow conventions. For those shared conventions see
the org guide (`bagofseeds/.github`, `CONTRIBUTING.md` + `CLAUDE.md`). This file
only records what is specific to `fiery.namedtensors`.

## What this package is

A [`fiery`](https://bagofseeds.github.io/fiery/) match that makes **names a
first-class citizen** of `torch.Tensor` via thin subclasses:

- `NamedTensor` — named **axes** (extends PyTorch's builtin named-tensor
  feature for ops it does not propagate).
- `TensorWithNamedIndices` — named **indices** (address positions along an axis
  by name; metadata self-managed in `_index_names` / `_index_dims`).
- `NamedVector` / `NamedMatrix` — 1-D / 2-D specializations.

Ported from a work-in-progress in `balbasty/magnetix`
(`magnetix/core/{namedtensors,arrayutils}.py`).

## Layout

```
src/fiery/namedtensors/
  __init__.py       # public API re-exports
  _tensors.py       # the tensor subclasses + torch-function overrides
  _arrayutils.py    # slicer parsing / axis-mapping helpers (no torch subclass)
tests/
  test_namedtensors.py
  test_arrayutils.py
```

## How the subclassing works

- `ExtendedTensor` holds a per-subclass registry (`_OVERRIDES`) and a
  `__torch_function__` that (a) dispatches to a registered override and
  (b) propagates subclass attributes listed in `_ATTRS` (e.g. the named-index
  metadata) from the first tensor argument onto the output — **but only when
  the output is a real `Tensor`** (many ops / property setters return `None`).
- Register an override with `@Cls.overrides(func)`. The decorator also shadows
  the tensor method of the same name, so both `torch.f(x)` and `x.f()` hit it.

## Conventions specific to this repo (do not regress)

1. **Wide Python (3.7+).** Runtime code must stay old-compatible: no PEP 695
   `type` statements, no walrus, no `zip(strict=)`, no runtime PEP 604 `|` /
   PEP 585 `list[...]` in values. Put modern typing in **annotations only**
   (they are lazy strings thanks to `from __future__ import annotations`) and
   build runtime type aliases from `typing` generics. Import `Self` / `Literal`
   / `Final` from `typing_extensions`.
2. **Wide PyTorch.** Never register an override for a function that may be
   absent: resolve it through `_torch_func("name")` (returns `None` if the op
   does not exist in the running torch) and pass that to `overrides()`, which
   skips `None`. Bespoke methods that are *not* torch ops (e.g.
   `TensorWithNamedIndices.index`) are defined as plain methods, never via the
   version-guarded override path.
3. **Index metadata is canonical when set directly.** Internal code that
   already holds `(names, dims)` assigns `out._index_names` / `out._index_dims`
   directly; going through the public setters re-runs `_prepare_index_names`
   and cross-couples the two values.
4. **A tensor with no named indices reports `None`** for `index_names` /
   `index_dims` and must still slice without error.

## Known follow-ups (see the tracking issues)

- **Stop trusting the builtin named-tensor feature.** `NamedTensor` still stores
  axis names via PyTorch's experimental `.names` / `.rename`, which has been
  dropped in some future torch builds. The plan is to self-manage axis names the
  way `TensorWithNamedIndices` already self-manages index metadata.
- **Per-method survey.** Every name-related torch op should have a name-aware
  override + a test; coverage is tracked with one sub-issue per function.
- **Functional-form metadata parity.** An override sets metadata on its result,
  but a *functional* call (`torch.op(x, ...)`) routes through the outer
  `__torch_function__`, which re-wraps the result and then re-propagates the
  source's original metadata — so `torch.index_select(x, ...)` currently drops
  the re-sliced index names while the method form `x.index_select(...)` keeps
  them. The method forms are the documented API; unifying the two is part of the
  propagation redesign.

## Gate before a PR

```sh
pip install .[test]
cd /tmp && python -m pytest <repo>/tests -q     # run from a neutral cwd
ruff check src tests && ruff format --check src tests
codespell src tests
```
