# CLAUDE.md — fiery-xtensor

Repo-specific guidance for coding agents. bagofseeds publishes two families of
PyTorch packages — standalone **`bagof-*`** packages and the **`fiery-*`**
namespace matches (this repo is one of the latter); they share the same
packaging, CI, docs, and workflow conventions. For those shared conventions see
the org guide (`bagofseeds/.github`, `CONTRIBUTING.md` + `CLAUDE.md`). This file
only records what is specific to `fiery.xtensor`.

## What this package is

A [`fiery`](https://bagofseeds.github.io/fiery/) match that makes **names a
first-class citizen** of `torch.Tensor`. `XTensor` is an
[xarray](https://docs.xarray.dev)-like `DataArray` over a live torch tensor:

- **dimensions** are named (`names`, self-managed in `_axis_names`);
- **coordinates** label positions along a named dimension (`coords`, a
  `{dim name: labels}` dict, self-managed in `_coords`) — **keyed by dimension
  name**, so they follow their dim through permute/reduce with no positional
  bookkeeping. A labelled dim must be named.
- `XVector` / `XMatrix` — conveniences that pre-name+label their channel
  axes (`"channel"`; `"row"`/`"col"`).

Select by label with `.sel`, by position with `.isel`, or reach a single label
by attribute (`x.red`). Ported (and since substantially reshaped) from a
work-in-progress in `balbasty/magnetix`
(`magnetix/core/{namedtensors,arrayutils}.py`); the old positional
`TensorWithNamedIndices` / `index_names` / `index_dims` API was replaced by the
name-keyed `coords` model (see #37).

## Layout

```
src/fiery/xtensor/
  __init__.py       # public API re-exports
  _tensors.py       # the tensor subclasses + torch-function overrides
  _arrayutils.py    # slicer parsing / axis-mapping helpers (no torch subclass)
  _compat.py        # version shims: EllipsisType, broadcast_shape, torch_func
tests/
  test_xtensor.py
  test_arrayutils.py
  test_compat.py
```

## How the subclassing works

- `ExtendedTensor` holds a per-subclass registry (`_OVERRIDES`) and a
  `__torch_function__` that, for a **registered override**, runs it under
  `_compat.no_dispatch()` (so the plain torch ops it calls don't recurse) and
  returns its result directly; for **any other op** it propagates the subclass
  attributes listed in `_ATTRS` (`_axis_names`, `_coords`) from the first tensor
  argument onto the output (only when the output is a real `Tensor`).
- Register an override with `@Cls.overrides(func)`. The decorator also shadows
  the tensor method of the same name, so both `torch.f(x)` and `x.f()` hit it.
- **Overrides return `_carry(input, <op>(input, ...), **new_meta)`.** `_carry`
  re-tags the result as `input`'s subclass, copies `input`'s metadata, then
  applies the overrides — so the *same* metadata lands whether the op was
  reached as `x.op(...)` or `torch.op(x, ...)` (functional/method parity). Do
  **not** set metadata attributes on the raw op result and return it directly.
- **Names and coordinates are self-managed** (`_axis_names` / `_coords`), *not*
  PyTorch's builtin named-tensor feature. The underlying tensor is never given
  builtin names; `names`, `coords`, `rename`, `sel`/`isel` and the overrides all
  read/write our own attributes. So the package works even where the builtin
  `.names`/`.rename` API has been removed. **Do not call builtin `.rename` /
  `.refine_names` or set builtin names anywhere.**
- **Coordinates are keyed by dim name.** Most overrides therefore need *no*
  coordinate code — `permute`/`transpose`/`movedim` and any op that keeps dim
  names+sizes just let `_carry` copy `_coords` through. Only ops that **remove**
  a dim (reductions, `select`, `diagonal`), **merge/split** one (`flatten`,
  `unflatten`, reshape), **resize** one (`narrow`, `index_select`, `gather`,
  `cat`), or **reorder its labels** (`flip`, `roll`) touch `_coords`. The
  `coords` **getter filters stale entries** (dim gone / length mismatch), so
  auto-propagated coordinates never lie. Helpers: `_coords_for(input, names)`
  keeps the coords whose dim survives; `_slice_labels` applies a 1-D slicer.
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

- **NAMED TENSOR** — the `XTensor` class: `names`/`coords` properties,
  `sel`/`isel`, `__getitem__` (slices labels of kept axes), `__getattr__`
  (label access), `rename`, `refine_names`/`align_to`/`align_as`.
- **RESHAPE / REORDER** — `permute` + special cases (transpose/movedim family,
  `view`/`reshape`), and rank-changers `flatten`/`unflatten`/`expand`/
  `broadcast_to`/`diagonal`.
- **REDUCTIONS** — `_make_reduction` factory (`sum`/`mean`/`amax`/…): drop the
  reduced axis' name+coords (keep under `keepdim`), reduce-all → unnamed scalar.
- **SLICE / SPLIT** — `select`/`narrow`/`unbind`/`split`/`chunk` (single-axis
  `__getitem__`, so coords track for free) and `flip`/`roll` (reorder labels).
- **COMBINE** — `cat`/`stack` (name reconciliation across operands; `cat`
  concatenates the join-axis labels) and `matmul`/`mm`/`bmm`.
- **GATHER / SCATTER** — `index_select`/`gather`/`scatter`/`where`/
  `masked_select`.
- **POINTWISE (BY NAME)** — `_make_pointwise` factory over `add`/`mul`/`eq`/…:
  when **both** operands are fully-named, axes align **by name** (`_align_by_name`
  → transpose + size-1 expand to the union of dims), else positional fallback.
  Registers both `torch.<op>` and `Tensor.<op>` (operators dispatch the latter).
- **CONVENIENCE** — `XVector`/`XMatrix`.

Shared helpers: `_carry`, `_coords_for` (keep surviving coords), `_slice_labels`
(1-D label slicer), `_reconcile_axis_names` (multi-operand), `_matmul_names`.

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
   (`sel`/`isel`, `rename`, `refine_names`/`align_*`) are defined as plain
   methods, never via the version-guarded override path. Version shims (an
   `EllipsisType` fallback,
   a pure-shape `broadcast_shape` that allocates nothing) live in `_compat`.
3. **Coordinates are keyed by dim name and must stay length-consistent.** Set
   `out._coords` to a `{name: labels}` dict where every key is a current dim
   name and every label tuple matches that dim's size. Prefer building it from
   `_coords_for` / `input.coords` (the guarded getter) rather than the raw
   `input.__dict__["_coords"]`. Anything else the `coords` getter silently
   drops.
4. **A tensor with no coordinates reports `{}`** and must still slice/select
   without error; a labelled dim must be named.

## Known follow-ups (see the tracking issues)

- **Per-method survey.** Every name-related torch op should carry names+coords
  and have a test; coverage is tracked with one sub-issue per function under the
  umbrella (#3). Landed: the reshape/reorder, slice/split, reduction, combine
  (`cat`/`stack`/`matmul`), gather/scatter families, name-as-`dim`, the
  `refine_names`/`align_*` API, and factories — now all on the name-keyed
  `coords` model (#37).
- **Open batches:** broadcasting-by-name for pointwise ops (R5 / #8, **decided**
  — xarray-style align-by-name); irregular / namedtuple reducers
  (`std`/`var`/`norm`, `max`/`min`/`sort`/`topk`, #20); `einsum`/`tensordot`;
  coordinate **alignment** (xarray inner-join on labels) if pursued.

## Gate before a PR

```sh
pip install .[test]
cd /tmp && python -m pytest <repo>/tests -q     # run from a neutral cwd
ruff check src tests && ruff format --check src tests
codespell src tests
```
