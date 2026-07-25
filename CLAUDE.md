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
  bookkeeping. A labelled dim must be named. A label may be a bare `str`/`None`
  or a **structured** dict (Proposal 0002): its `"name"` is its selection
  identity (`_label_name`), and a **query** dict in a `[]`/`sel` slot selects
  the matching *positions* (`_match_positions` → `slice`/list, keeps the axis)
  — the position-level analogue of the axis-descriptor query.
- `xvector` / `xmatrix` (in `_factories.py`) — one-line **factory functions**
  that name+label a `"channel"` axis (or `"row"`/`"col"`) and return a **plain
  `XTensor`**. Deliberately *not* subclasses: an op that drops the labelled
  axis must yield an ordinary `XTensor`, so the type never outlives its
  meaning (the removed `XVector`/`XMatrix` subclasses did not maintain that).
- **axis descriptors** (OME-NGFF-style, #39): a name may be given as a dict
  `{"name": "x", "type": "space", "orientation": "left-to-right"}` instead of a
  bare string. The extra fields (`type`/`unit`/`orientation`) live in
  `_axis_meta`, **keyed by dim name** (so they follow the dim exactly like
  coords). `.names` stays the bare view; `.axes` returns the full descriptors.
  `flip` reverses a flipped axis' `orientation`; `rename` remaps meta keys.
  A descriptor can also be used to **address** axes: pass a query dict in
  place of a `dim` and it expands to *every* matching axis —
  `movedim({"type": "space"}, -1)` blocks all space axes to the back
  (preserving order), `sum(dim={"type": "channel"})` reduces every channel
  axis (see `_query_positions` / `_movedim_block_order` / `_resolve_reduce_dim`).

- **data unit** (`unit`, Proposal 0003 phase 1): the physical unit of the
  tensor **values**, self-managed in `_data_unit` (in `_ATTRS`, so it
  propagates like names/coords). `.unit` gets/sets it; `unit=` is a constructor
  kwarg; `to_unit` converts (rescaling data). Opaque unless the `unit_backend`
  option selects one (`"pint"` → validate/normalise/convert via `_units`).
  Under a backend the ops do dimensional **algebra** (`*`/`/`/`pow`/matmul
  multiply/divide units, `add`/compare need compatible units, transcendentals
  require dimensionless) — an invalid step drops the unit, or raises under
  `unit_policy="strict"`. See the POINTWISE `_UNIT_RULE`/`_binary_unit` and the
  transcendental factory.

- **heterogeneous data units** (Proposal 0003 phase 3): units that vary along
  an axis live on the `unit` field of a structured coordinate (0002) —
  `_label_unit` reads them; effective unit = base · Π(coord units). Selecting a
  single position on such an axis (`__getitem__`/`isel`/`sel`) folds that unit
  into `_data_unit`; reducing over it folds a **uniform** axis unit (via
  `_uniform_unit`/`_reduce_unit`) into the base, or drops/raises (policy) on
  **incompatible** per-position units. Backend-gated (inert with
  `unit_backend=None`). Heterogeneous matmul/einsum contraction is not yet
  wired (rides on base units only).

- **numeric coordinates** (Proposal 0001 phase 1): a `coords[dim]` may be a
  compact **`Coordinate`** (`{spacing[, origin]}`) instead of a label tuple,
  stored in a separate `_axis_coord` attr (in `_ATTRS`, so it rides through ops
  like the others; the `coords` getter filters stale entries by name and binds
  the axis size). `coords[dim]["values"]` is a **derived** key materialising
  `origin + i*spacing` fresh each access (no cache) as a 1-D unitful `XTensor`
  — differentiable when `spacing` is a 0-rank tensor. `spacing`/`origin` are
  `Unitful` **magic dicts** (`{value, unit}`, in `_units`): dict-inheriting,
  tensor-preserving, item-access canonical + whitelisted attribute sugar
  (`.value`/`.unit`/…, never the colliding `.values`/`.keys`/`.items`), with a
  `.to(unit)` conversion. `_units.as_unitful` ingests tuple/dict/pint/str/
  number; the family is internal (no export → no `pint.Quantity` clash). The
  **position** unit (a coordinate's values) is distinct from the **data** unit
  (0003, the tensor values).

- **attaching a unit by `*`** (Proposal 0003 phase 4): `x * u.mm` / `x / u.s`
  attach/derive a data unit from a backend `Unit`/`Quantity`. This is caught in
  `XTensor`'s operator **dunders** (`__mul__`/`__rmul__`/`__truediv__`), *not*
  the `__torch_function__` overrides — otherwise pint's reflected `__rmul__`
  grabs `x * <unit>` first and returns a wrapped object. `_attach_unit` splits
  the operand into `(magnitude, unit)`, scales the data (through a **fresh
  view**, so the original is never annotated in place), and combines the unit.
  Non-unit operands fall straight back to `Tensor.__mul__` &c. `unit * x`
  (unit on the left) is not interceptable — use `x * unit`.

- **more phase 4** — `.magnitude` (property) drops the data unit, returning a
  unit-free **view** that keeps names/coords (original untouched). `add`/`cmp`
  of **compatible-but-different** units implicitly convert the *right* operand
  to the left's unit (`_reconcile_units` rescales via `_units.factor` before
  the op; only incompatible dims drop/raise). **Contraction** unit algebra
  (`matmul`/`einsum`/`tensordot`) folds each side's *uniform* contracted-axis
  unit into its base and multiplies (`_contraction_unit` / `_axis_uniform_unit`
  / `_matmul_contracted_axes` / `_einsum_contracted_axes`); a non-uniform
  contracted axis drops/raises. An `einsum` equation the parser can't read
  (ellipsis) falls back to the base-unit product.

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
  _options.py       # global options + `set_options` context manager
  _units.py         # optional unit backend (pint) for the `.unit` data unit
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
  `sel`/`isel`, `__getitem__` (positional slicing that also slices the labels
  of kept axes; a **positional** coordinate label resolves against the axis it
  indexes — a bare `str` like an int there, a `list` of `str` as an advanced
  index — via `_resolve_label_slicer`/`_label_to_index`, so `x[..., "y", "z"]`
  addresses the last two axes by label), `__getattr__` (label access),
  `rename`, `refine_names`/`align_to`/`align_as`. A single `...` in any
  name-tuple expands via `_expand_name_ellipsis` to the run of axes it stands
  for: unnamed (`None`) on assignment (`names=`/setter), unchanged on
  modification (`rename`/`refine_names`), the remaining axes in current order
  on reorder (`permute`/`align_to`).
- **RESHAPE / REORDER** — `permute` + special cases (transpose/movedim family,
  `view`/`reshape`), and rank-changers `flatten`/`unflatten`/`expand`/
  `broadcast_to`/`diagonal`.
- **REDUCTIONS** — `_make_reduction` factory (`sum`/`mean`/`amax`/…): drop the
  reduced axis' name+coords (keep under `keepdim`), reduce-all → unnamed scalar.
- **SLICE / SPLIT** — `select`/`narrow`/`unbind`/`split`/`chunk` (single-axis
  `__getitem__`, so coords track for free) and `flip`/`roll` (reorder labels).
- **COMBINE** — `cat`/`stack` (name reconciliation across operands; `cat`
  concatenates the join-axis labels), `hstack`/`vstack`/`dstack` (same
  reconciliation, but only when every operand already has the result's rank
  — these promote lower-rank operands first, which can shift axes; always
  drop coordinates), `matmul`/`mm`/`bmm`, and the contraction ops
  `einsum`/`tensordot` (equation- or axis-list-driven; free-function only, no
  method form — `_einsum_output_names` parses explicit/implicit equations and
  falls back to unnamed on an ellipsis; both drop coords). **All** of these
  multi-operand ops merge axis **descriptors** across their operands via
  `_merge_axis_meta` (union of surviving dims; per shared dim keep agreeing
  fields, drop conflicting — under the `combine_axes` option), the same helper
  the pointwise factory uses, so a surviving axis keeps its descriptor no
  matter which operand it came from.
- **GATHER / SCATTER** — `index_select`/`gather`/`scatter`/`scatter_add`/
  `index_add`/`index_copy`/`index_fill`/`where`/`masked_select`.
- **POINTWISE (BY NAME)** — `_make_pointwise` factory over `add`/`mul`/`eq`/…:
  when **both** operands are fully-named, axes align **by name** (`_align_by_name`
  → transpose + size-1 expand to the union of dims), else positional fallback.
  A shared dim **labelled on both** operands with differing labels is also
  aligned **by label** — `_reindex_axis` inner-joins both operands to the
  intersection (in the left operand's order) before the op (xarray
  `join="inner"`). **Axis descriptors** are merged across operands by
  `_merge_axis_meta` (union of dims; per shared dim keep agreeing fields, drop
  conflicting). The policy is resolved **per descriptor field** by
  `_options.combine_axes_policy` under the `combine_axes` option — a policy str
  (`drop_conflicts`/`strict`(=`raise`)/`override`/`drop`) or a `{field: policy}`
  dict with `"*"` as the default. Registers both `torch.<op>` and
  `Tensor.<op>` (operators dispatch the latter).
- **CONVENIENCE** — `xvector`/`xmatrix` factory functions (in `_factories.py`),
  not subclasses.

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
- **Open batches:** irregular / namedtuple reducers (`max`/`min`/`sort`/`topk`,
  #20); axis `unit` propagation (#48, after einsum); coordinate **alignment**
  (xarray inner-join on labels) if pursued. Landed since: broadcasting-by-name
  for pointwise ops (#8, xarray-style align-by-name), `std`/`var`/`norm`,
  `einsum`/`tensordot`, and rich axis descriptors (#39).

## Gate before a PR

```sh
pip install .[test]
cd /tmp && python -m pytest <repo>/tests -q     # run from a neutral cwd
ruff check src tests && ruff format --check src tests
codespell src tests
```
