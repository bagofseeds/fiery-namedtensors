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
  _compat.py        # version shims: EllipsisType, broadcast_shape, torch_func
tests/
  test_namedtensors.py
  test_arrayutils.py
  test_compat.py
```

## How the subclassing works

- `ExtendedTensor` holds a per-subclass registry (`_OVERRIDES`) and a
  `__torch_function__` that, for a **registered override**, runs it under
  `_compat.no_dispatch()` (so the plain torch ops it calls don't recurse) and
  returns its result directly; for **any other op** it propagates the subclass
  attributes listed in `_ATTRS` (axis names `_axis_names`, named-index metadata
  `_index_names`/`_index_dims`) from the first tensor argument onto the output
  (only when the output is a real `Tensor`).
- Register an override with `@Cls.overrides(func)`. The decorator also shadows
  the tensor method of the same name, so both `torch.f(x)` and `x.f()` hit it.
- **Overrides return `_carry(input, <op>(input, ...), **new_meta)`.** `_carry`
  re-tags the result as `input`'s subclass, copies `input`'s metadata, then
  applies the overrides — so the *same* metadata lands whether the op was
  reached as `x.op(...)` or `torch.op(x, ...)` (functional/method parity). Do
  **not** set metadata attributes on the raw op result and return it directly.
- **Axis names are self-managed** (`_axis_names`), *not* PyTorch's builtin
  named-tensor feature. The underlying tensor is never given builtin names; the
  `names` property, `rename`/`rename_` and the axis-name overrides all read/write
  `_axis_names`. So the package works even where the builtin `.names`/`.rename`
  API has been removed. **Do not call builtin `.rename` / `.refine_names` or set
  builtin names anywhere.**
- **Name-as-`dim` is a *method-form* feature.** Overrides that take a `dim`
  resolve a `str` (or a sequence mixing names/ints) via `_resolve_axis` /
  `_resolve_dims`. This only works on the method form (`x.transpose("a","c")`,
  `x.sum(dim="a")`): newer PyTorch rejects a non-int `dim` at the C dispatcher
  *before* `__torch_function__` runs, so `torch.transpose(x, "a", ...)` cannot
  be relied on. Don't promise or test functional-form name-as-dim. Ops with no
  method form at all (`cat`, `stack`) don't offer name-as-dim.
- **Operators dispatch with the bound `Tensor` method, not the `torch`
  function.** `a @ b` reaches `__torch_function__` with `Tensor.matmul` — a
  *different* callable than `torch.matmul`. For such ops, register the override
  under **both** (see `_make_matmul`) or the operator silently misses it.
- **Ops whose first argument is a *sequence*** (`cat`, `stack`) still trigger
  dispatch (torch inspects nested lists for subclasses). The override takes the
  first operand as the `_carry` metadata source.

## Where the overrides live (`_tensors.py` sections)

Named-aware overrides are grouped into labelled banners; add new ops to the
matching section (or a new one):

- **NAMED AXES** — `NamedTensor`, the axis-name overrides (`permute`,
  `unsqueeze`, `squeeze`, `view`, `reshape`, transpose/movedim family, …).
- **NAMED INDICES** — `TensorWithNamedIndices` / `NamedVector` / `NamedMatrix`
  and their `__getitem__` / `index` / `permute` / `index_select`.
- **REDUCTIONS** — `_make_reduction` factory (`sum`/`mean`/`amax`/…): drop the
  reduced axis' name (keep it under `keepdim`), reduce-all → unnamed scalar.
- **SLICE / SPLIT** — `select`/`narrow`/`unbind`/`split`/`chunk` (expressed as
  single-axis `__getitem__`, so index metadata tracks for free) and
  `flip`/`roll`.
- **RESHAPE (RANK)** — `flatten`/`unflatten`/`expand`/`broadcast_to`/`diagonal`
  (merged/split axes become unnamed, following the `view`/`reshape` rule).
- **COMBINE** — `cat`/`stack` (name reconciliation across operands) and the
  `matmul`/`mm`/`bmm` family.

Shared helpers: `_carry`, `_axes_removed_meta` / `_reduce_index_meta` (drop &
shift metadata), `_reconcile_axis_names` (multi-operand), `_matmul_names`.

## Conventions specific to this repo (do not regress)

1. **Wide Python (3.7+).** Runtime code must stay old-compatible: no PEP 695
   `type` statements, no walrus, no `zip(strict=)`, no runtime PEP 604 `|` /
   PEP 585 `list[...]` in values, and **never subscript an abc/builtin generic
   at runtime** (`collections.abc.Sequence[...]` is not subscriptable before
   3.9). Modern typing lives in **annotations only** (lazy strings thanks to
   `from __future__ import annotations`). All typing — annotations *and* the
   runtime type aliases — goes through **`import typing_extensions as tx`**
   (`tx.Union`, `tx.Sequence`, `tx.Self`, …); do not import from `typing` or
   `collections.abc` (`tx.Sequence` also works for `isinstance`). This matches
   the bagof-hints house style.
2. **Wide PyTorch.** Never register an override for a function that may be
   absent: resolve it through `_torch_func("name")` (from `_compat`, returns
   `None` if the op does not exist in the running torch) and pass that to
   `overrides()`, which skips `None`. Bespoke methods that are *not* torch ops
   (e.g. `TensorWithNamedIndices.index`) are defined as plain methods, never via
   the version-guarded override path. Version shims (an `EllipsisType` fallback,
   a pure-shape `broadcast_shape` that allocates nothing) live in `_compat`.
3. **Index metadata is canonical when set directly.** Internal code that
   already holds `(names, dims)` assigns `out._index_names` / `out._index_dims`
   directly; going through the public setters re-runs `_prepare_index_names`
   and cross-couples the two values.
4. **A tensor with no named indices reports `None`** for `index_names` /
   `index_dims` and must still slice without error.

## Known follow-ups (see the tracking issues)

- **Per-method survey.** Every name-related torch op should have a name-aware
  override + a test; coverage is tracked with one sub-issue per function under
  the umbrella (#3). Landed so far: axis-name core ops, name-as-`dim`,
  reductions, the slice/split and reshape families (survey section A), and
  `cat`/`stack`/`matmul` (section B).
- **Open batches:** irregular / namedtuple reducers (`std`/`var`/`norm`,
  `max`/`min`/`sort`/`topk`, …); `einsum`/`tensordot`; gather/scatter (section
  D); channel-name contraction for `NamedMatrix`/`NamedVector`; broadcasting-
  by-name for pointwise ops (R5); `align_as`/`refine_names` (section E).

## Gate before a PR

```sh
pip install .[test]
cd /tmp && python -m pytest <repo>/tests -q     # run from a neutral cwd
ruff check src tests && ruff format --check src tests
codespell src tests
```
