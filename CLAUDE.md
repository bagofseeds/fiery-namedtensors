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
- **axis descriptors** (OME-NGFF-style, #39): an axis may be given as a dict
  `{"name": "x", "type": "space", "orientation": "left-to-right"}` through
  **`axes=`** (the general per-axis container; `names=` takes bare strings only
  and rejects dicts — `_parse_axes` splits an `axes=` spec into names +
  `_axis_meta` + coord specs, and a descriptor's `coord`/`labels` key feeds the
  `coords` setter). The extra fields (`type`/`unit`/`orientation`) live in
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
  compact **`Coordinate`** (`{spacing[, origin]}`) instead of a label tuple.
  Storage is **unified** (Proposal 0005 step 1): the single `_coords` attr (in
  `_ATTRS`) holds `{name: (dims, coord)}`. A coordinate is a **dimension
  coordinate** (the `.sel`-able index) iff `dims == (name,)`; a
  **non-dimension coordinate** (Proposal 0005 step 2, landed) has a `name`
  that is not itself an axis, and rides along whatever dim(s) `dims` names
  without being an index — `coords={"season": ("t", [...])}` stores
  `{"season": (("t",), coord)}`. The `coords` **property** is the flat,
  validated `{name: coord}` view everything else reads (filters stale entries
  — dim gone, resized, or a non-dim coordinate's dim renamed away — and binds
  the axis size); `_pack_coord`/`_pack_coords` wrap a flat value/dict back
  into the unified storage shape when writing `_coords`. `coords[dim]["values"]`
  is a **derived** key materialising `origin + i*spacing` fresh each access
  (no cache) as a 1-D unitful `XTensor` — differentiable when `spacing` is a
  0-rank tensor. `spacing`/`origin` are `Unitful` **magic dicts** (`{value,
  unit}`, in `_units`): dict-inheriting, tensor-preserving, item-access
  canonical + whitelisted attribute sugar (`.value`/`.unit`/…, never the
  colliding `.values`/`.keys`/`.items`), with a `.to(unit)` conversion.
  `_units.as_unitful` ingests tuple/dict/pint/str/number; the family is
  internal (no export → no `pint.Quantity` clash). The **position** unit (a
  coordinate's values) is distinct from the **data** unit (0003, the tensor
  values). A `Coordinate` may also be **explicit** (`{"values": <unitful 1-D
  tensor>}`, from `coords={dim: tensor}`); `__getitem__` slices a *dimension*
  coordinate **affinely** (`_slice_coordinate`: compact updates
  `spacing*=step`/`origin+=start*spacing`, explicit slices the array, advanced
  index materialises a compact coord to explicit) — a non-dimension
  coordinate isn't re-sliced (no slice-tracking yet, Proposal 0005 step 6), it
  just rides through unchanged or drops; only labels/explicit values are
  accepted for one (a **compact** non-dim spec raises `NotImplementedError` —
  see the pitfall below). `Coordinate.to(unit)` converts the position unit.
  `rename`/`rename_` remap both the storage key **and** the embedded `dims`
  (`_remap_coords`), raising on a name collision (renaming an axis onto an
  existing coordinate's name) rather than silently dropping one. A raw
  `input.__dict__.get("_coords")` read must unpack `(dims, coord)` per entry —
  code that only wants the flat view should go through `input.coords` instead
  (see `_reduce_unit`/`_axis_uniform_unit`, and the pitfall below).

### Pitfalls that have caused real bugs here — read before touching `_coords`

These are lessons from bugs actually shipped (some caught in review, one
[#85](https://github.com/bagofseeds/fiery-xtensor/issues/85) that reached
`main`) while building Proposal 0005. Each cost real debugging time; the
pattern is worth internalising rather than re-discovering.

1. **A `Coordinate` is a `dict` subclass — generic sequence ops act on its
   *keys*, not its materialised values.** `reversed(coord)` yields something
   like `("origin", "spacing")`, not the reversed numeric values; `coord[1]`
   falls through to `dict.__getitem__` and raises `KeyError` rather than
   indexing position 1. This bit `flip`/`roll` (#85): they read a coordinate
   via `.coords.get(name)` and called `reversed()`/integer-indexed it
   directly, assuming "numeric coordinate" meant "sequence of values". It
   silently dropped the coordinate on flip (fails the `coords` getter's
   length check) and crashed on roll. **Rule**: never call a sequence op on a
   `Coordinate` directly — always go through `_slice_coordinate(coord,
   slicer, size)`, which already knows how to reslice both compact (affine
   update) and explicit (tensor index) forms correctly. For a reversal, use
   `slice(None, None, -1)` for a compact coordinate (stays exact/compact/
   differentiable) but an explicit reversed-position `list` for an explicit
   one — **PyTorch itself rejects a negative-step slice on a real tensor**
   (`t[::-1]` raises `"step must be greater than zero"`), so the slice-object
   shortcut only works when nothing is actually indexed with it.
2. **Raw `__dict__.get("_coords")` access bypasses the `coords` property's
   unification and validation.** After the storage moved from `{dim: coord}`
   to `{name: (dims, coord)}`, every internal reader that still did
   `x.__dict__.get("_coords")` and used the result as if it were a flat
   `{name: coord}` map got the `(dims, coord)` **tuple** instead of `coord` —
   silently wrong (e.g. `_reduce_unit`/`_axis_uniform_unit` folded `None`
   units everywhere instead of raising or working, because iterating the
   tuple where a `Coordinate` was expected just didn't match and fell
   through). **Rule**: always read coordinates through the `.coords`
   property unless you are specifically implementing storage-level logic
   (propagation/renaming) that must see the `dims` half too.
3. **A helper that filters by "does this survive" must check every dim a
   coordinate touches, not just its own key.** `_coords_for` initially kept
   an entry only if `entry_key in result_names` — correct for dimension
   coordinates (`key == the one dim`), wrong for non-dimension ones (`key`
   isn't a dim at all; what must survive is every name in `dims`). This
   silently dropped non-dimension coordinates on squeeze/reshape/flatten/
   unflatten/diagonal/reduce even when the dim they rode on was untouched.
   **Rule**: any coordinate-survival check must be `all(dim in kept for dim
   in dims)`, never a check against the entry's own key.
4. **A per-axis rewrite of `_coords` must start from a full copy, not just
   the entries the loop actually visits.** The first post-unification
   `__getitem__` rewrite built `new_stored` only from axes the slicing loop
   touched, which never looks up non-axis-named keys — so it dropped every
   non-dimension coordinate on *any* slice, touched axis or not. **Rule**:
   start from `new_stored = dict(stored)` (copy everything through) and only
   overwrite/remove the specific entries the op actually invalidates; let
   everything else ride through and self-heal via the `coords` property's own
   validation.
5. **A "quick fix" that removes a validation error can trade a loud failure
   for a silent one — check what happens on the *next* op, not just this
   one.** A compact (`spacing`/`origin`) coordinate has no length-based
   staleness check (it fits *any* size by construction). Dimension
   coordinates get away with this because they're actively re-sliced
   (`_slice_coordinate` recomputes `origin`/`spacing` to match). Non-dimension
   coordinates are **not** re-sliced (step 6 isn't built yet) — so accepting
   a compact one and letting it "just ride along" produces a coordinate that
   silently reports the *wrong* values after its dim is sliced (verified:
   `x[1:3].coords["wl"]["values"]` gave the pre-slice window, not the correct
   one — no error, just wrong numbers). The fix was **not** to accept the
   input more permissively; it was to make `_parse_nondim_coord` explicitly
   raise `NotImplementedError` for a compact non-dim spec until slice-tracking
   exists. **Rule**: when a fix widens what's accepted, trace it through the
   *next* mutating op before shipping — "parses now" is not the same as
   "stays correct after being carried through a slice/rename/reduce".
6. **When a structural refactor lands underneath a long-lived feature branch,
   reset and reimplement rather than resolving a deep rebase conflict.**
   Rebasing the non-dimension-coordinate branch onto the just-merged
   storage-unification commit produced conflicts throughout `_tensors.py`
   because the code both branches touched had been restructured, not just
   edited. Resolving them line-by-line risked silently reintroducing the
   pre-unification shape. Reimplementing the feature directly against the new
   `{name: (dims, coord)}` model (via `git reset --hard origin/main` on the
   branch) was faster and produced a cleaner diff than untangling the
   conflict markers would have.

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
