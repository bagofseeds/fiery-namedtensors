# Proposal 0005 — Multiple coordinates per axis (incl. affine coordinates)

| | |
| --- | --- |
| **Status** | Draft — steps 1–4 implemented; step 5 (`.sel`/`.interp`) fully landed (#82 phases 1 and 2) |
| **Author** | (proposed) |
| **Created** | 2026-07-26 |
| **Updated** | 2026-07-29 — affine joint `.interp` (step 5, #82 phase 2) implemented |
| **Tracking** | [#65](https://github.com/bagofseeds/fiery-xtensor/issues/65); generalises 0001/0002; interacts with 0004 (numeric `.sel`), #82 (curvilinear interp) |

## Abstract

Today an axis carries **exactly one** coordinate (`coords = {dim: coordinate}`).
xarray lets a dimension carry **many** — one **dimension coordinate** (the index
used by `.sel`/alignment) plus any number of **non-dimension coordinates** — and
coordinates that span **several** dimensions (curvilinear grids). This proposal
adopts that model, and extends the compact `spacing`/`origin` form of 0001 into
a multi-dimensional **affine** coordinate — i.e. a NIfTI-like voxel↔world map.

## Motivation

Real datasets label an axis more than one way at once:

- a `time` axis that is a numeric coordinate (seconds) **and** carries a
  `season` label per step;
- a `step` axis indexed by integer, with a `wavelength` coordinate alongside;
- an image whose `(y, x)` grid has 2-D `lat`/`lon` coordinates — a coordinate
  over **two** dims at once. In imaging this is the **affine**: `world = A·index
  + origin`, exactly a NIfTI `sform`/`qform`.

One-coordinate-per-axis expresses none of these. This is the last major gap
between our coordinate model and xarray's — and the affine case makes it a
first-class feature for medical / neuro imaging, not just parity.

## xarray's model (the target)

- `coords` is keyed by **coordinate name**, each an array over one or more dims:
  `{name: (dims, values)}`.
- A **dimension coordinate** has `name == dim`, is 1-D over that dim, and is the
  **index** for `.sel`/alignment. At most one per dim.
- **Non-dimension coordinates** ride along (carried, sliced; selectable only
  after promotion to the index).
- **`swap_dims` / `set_index`** change which coordinate is a dim's index.
- Coordinates may be **multi-dimensional** (`lat(y, x)`) — curvilinear grids.

## Proposed model for `fiery.xtensor`

### Storage — unified into `{name: (dims, coord)}` *(resolves open Q2 — done, #84)*

All coordinates live in one map keyed by coordinate name, each entry a
`(dims, coord)` pair —

```
_coords = { name: (dims_tuple, coord) }
```

where `coord` is any of the existing kinds (categorical labels; a numeric
`Coordinate` — compact `spacing`/`origin` or explicit values) and `dims` is the
tuple of dim names it spans (length 1 for today's coordinates, ≥ 2 for
curvilinear/affine, landing in step 3 below). A coordinate is a **dimension
coordinate** (the index) iff `dims == (name,)`; a **non-dimension** coordinate
has a `name` that is not itself a dim, and rides along the dim(s) in `dims`
without being an index. Reconciliation, slicing, and propagation are one code
path keyed by coord-name over a tuple of dims, rather than the previous
incremental `_coords`/`_axis_coord`/`_extra_coords` split.

### Input

A `coords` value keyed by a dim name is that dim's index coordinate, as today;
a value keyed by any other name is `(dim, values)` — a non-dimension coordinate
riding along `dim`:

```python
xtensor(data, names=("t",), coords={
    "t":      {"spacing": (0.5, "s")},        # dimension coord (index)
    "season": ("t", ["w", "w", "sp", "sp"]),  # non-dimension coord along t
})
```

### `.sel`

Resolves against the **index** (the dimension coordinate). Selecting on a
non-dimension coordinate raises, directing to `swap_dims`/`set_index` (xarray).

### Propagation

A dimension coordinate is carried and re-sliced exactly like today (affine
update on a basic slice, materialise-then-index on an advanced one). A
non-dimension coordinate rides through **unchanged** as long as every dim in
its `dims` survives at its original size; it drops the moment one of them is
renamed away, removed, or resized (conservative — no slice-tracking yet, see
the delivery plan's step 6).

## Compact **affine** coordinates (step 3, implemented)

Generalise 0001's 1-D compact form `value[i] = origin + i·spacing` by letting
**spacing be a per-source-dim vector** and the coordinate **span several dims**:

```python
xtensor(field, names=["y", "x"], coords={
    "lat": (["y", "x"], {"spacing": ([sy_lat, sx_lat], "deg"), "origin": (lat0, "deg")}),
    "lon": (["y", "x"], {"spacing": ([sy_lon, sx_lon], "deg"), "origin": (lon0, "deg")}),
})
# lat[i,j] = lat0 + sy_lat·i + sx_lat·j ;  lon[i,j] = lon0 + sy_lon·i + sx_lon·j
```

Stacked, `(lat, lon)` is a 2×2 linear map + translation from index to world — a
NIfTI `sform`. Properties:

- **Materialisation**: `origin + Σ_d spacing[d]·index_d` via broadcast `arange`s
  → an N-D grid, **differentiable** (spacing/origin may be **learnable** — a fit
  for registration).
- **Exact slicing**: slicing dim `d_k` by `(start, step)` updates
  `origin += start·spacing[k]; spacing[k] *= step` — 0001's trick, per component.
- **Closed-form inverse**: `index = A⁻¹·(world − origin)`. So affine
  curvilinear `.sel`/`.interp` is tractable — invert the (small) affine, then
  nearest / `grid_pull` — sidestepping the nonlinear inversion a general
  `lat(y,x)` array needs ([#82](https://github.com/bagofseeds/fiery-xtensor/issues/82)).
  Requires querying the coupled coords **jointly** (`.sel(lat=…, lon=…)`), and a
  full-rank (invertible) map.

## Coordinate reconciliation on broadcast *(resolves open Q3)*

When two operands share a coordinate **name** (a dim coord or a non-dim coord),
reconcile per coordinate — the same "keep on agreement, drop on conflict" rule
descriptors and labels already follow:

- **same kind, equal** (labels / affine / explicit values) → keep;
- **both categorical, differing** → inner-join by label;
- **both numeric, differing** → require equality, else **drop** (fuzzy float
  inner-join is a trap; value-based reindex is `.interp`-alignment, separate);
- **different kinds** (numeric vs categorical, 1-D vs curvilinear) → **drop**
  (the axis still broadcasts by name; only the contested coord is dropped).

The dimension-coordinate half of this shipped with the partial-name broadcasting
work ([#83](https://github.com/bagofseeds/fiery-xtensor/pull/83),
`_reconcile_coords`); binary ops don't yet carry non-dimension coordinates
through at all (they're dropped, conservatively) — extending `_reconcile_coords`
to them is future work, not blocking this proposal.

## Delivery plan — split across PRs

Per review, this lands as a **sequence of small PRs**, not one massive change:

1. ✅ **Storage unification** ([#84](https://github.com/bagofseeds/fiery-xtensor/pull/84)) —
   migrated the internals to `{name: (dims, coord)}` with a dimension
   coordinate = `dims == (name,)`; behaviour-preserving refactor, no new user
   surface.
2. ✅ **Non-dimension (1-D) coordinates** ([#72](https://github.com/bagofseeds/fiery-xtensor/pull/72)) —
   `(dim, values)` input, `.coords` exposure, `.sel` "not an index" guard,
   conservative propagation (implemented directly against the unified storage
   — no separate `_extra_coords` attr).
3. ✅ **Compact affine / multi-dim coordinates** — vector `spacing` over a
   `dims` tuple: materialisation, exact per-component affine slicing,
   learnable. *(No `.sel` yet — that's step 5.)*
4. ✅ **`swap_dims`** — promote a non-dimension coordinate to the index.
5. ✅ **Affine `.sel` / `.interp`** — closed-form `A⁻¹` inverse + (for interp)
   N-D `grid_pull`, joint query over the coupled dims (ties off
   [#82](https://github.com/bagofseeds/fiery-xtensor/issues/82)).
   - ✅ `.sel` half (phase 1) — a joint query (`x.sel(lat=…, lon=…)`) over a
     square, invertible affine solves `index = A⁻¹(world − origin)` and
     snaps to the nearest position; under/over-determined and non-invertible
     queries raise rather than approximating.
   - ✅ `.interp` half (phase 2) — the same closed-form inverse, kept
     **fractional**, feeds a genuine N-D `grid_pull` (or a built-in
     separable nearest gather); a multi-valued joint query collapses the
     spanned dims into one new axis, point-wise.
6. **Non-dim coordinate slice-tracking** — carry non-dim coords through slicing
   instead of dropping on resize.

Each PR is independently reviewable/mergeable; (3) and (5) are the headline
affine feature.

## What this slice implements (step 2)

- A `coords` value keyed by a non-axis name is parsed as `(dim, values)` and
  stored as `{key: ((dim,), coord)}` in the same unified `_coords` map a
  dimension coordinate uses — `dims != (key,)` is exactly what marks it
  non-dimension.
- `.coords` exposes it by name, validated the same way a dimension coordinate
  is (its dim must still be named, at the same size).
- `.sel(name=...)` raises "not an index coordinate" for one, pointing at
  `swap_dims` (step 4, below).
- `rename` remaps the dim it rides on (not its own key); ops that reslice a
  named axis (`sort`/`flip`/`roll`/`gather`/`index_select`/`take_along_dim`)
  drop any non-dimension coordinate riding on the touched axis, alongside that
  axis' own dimension coordinate — both are conservatively invalidated the
  same way. `rename` also raises on a coordinate-name collision (renaming an
  axis onto an existing coordinate's name) rather than silently dropping one.
- Binary ops (broadcasting) don't carry non-dimension coordinates through yet
  (dropped) — see the reconciliation section above.
- Only **labels** or an **explicit** numeric tensor are accepted as `values`
  for a **single-dim** non-dimension coordinate — a compact
  (`spacing`/`origin`) spec over one dim raises `NotImplementedError`. Unlike
  a dimension coordinate, a single-dim non-dimension one isn't re-sliced when
  its dim is (no slice-tracking yet, step 6); a label or explicit coordinate
  is at least caught by the length check on resize, but a compact one binds
  to *any* size, so it would silently rebind to the wrong affine after a
  non-trivial slice instead of raising or dropping. Lifting this restriction
  is part of step 6.

## What step 3 implements

- A `coords` value keyed by a non-axis name, given as `(dims, {spacing,
  [origin]})` where `dims` is a **sequence of two or more** dim names, is a
  compact **affine** coordinate: `spacing` is a vector with one component per
  dim (`_as_unitful_vector`, preserving a tensor component's autograd graph
  via `torch.stack` rather than flattening it through `torch.as_tensor`);
  `origin` stays a single scalar shared across all of them.
- `["values"]` materialises the N-D grid lazily (`Coordinate._materialise_axes`,
  bound to per-dim sizes by `_bound_axes`): `origin + Σ_d spacing[d]·index_d`
  via a broadcast `arange` per dim — no dense grid cached, so it stays
  differentiable exactly like the 1-D case. The grid is laid out in the
  **tensor's** axis order rather than in `dims` order (the two differ when
  `dims` is given out of order, or after a `permute`/`transpose`/`movedim`
  carries the coordinate through), since `["values"]` is a bare array
  carrying no dims of its own to disambiguate the layout.
- `__getitem__` re-slices it **exactly**, per spanned dim
  (`_slice_affine_coordinate`): a basic slice updates that dim's component
  (`origin += start·component; component *= step`); an **integer** index
  folds the dim out of `dims`/`spacing` entirely (`origin += index·component`)
  — the coordinate survives with one fewer dim, and collapsing all the way to
  one remaining dim yields an ordinary 1-D compact non-dimension coordinate
  (a plain scalar `spacing`, not a length-1 vector — reslicing code has to
  branch on this, it isn't just a vector of length 1); anything else
  (boolean/advanced indexing) can't stay affine, so the *whole* coordinate
  drops. `rename` remaps every dim in `dims` the same way a 1-D non-dimension
  coordinate's single dim already does (already generic over `dims` length),
  but now also refuses to rename an axis *onto* a multi-dim coordinate's key
  — that would leave a key which is a dim yet isn't that dim's index, which
  `.sel` and the dimension-coordinate slicing pass would then misread.
- A general multi-dim **explicit** coordinate (arbitrary curvilinear
  `lat(y,x)` values, not a compact affine map) is **not** implemented —
  raises `NotImplementedError`, pointing at the compact form. That's separate,
  harder work (nonlinear inverse for `.sel`/`.interp`, [#82](https://github.com/bagofseeds/fiery-xtensor/issues/82)),
  not a natural extension of this slice.
- `.to(unit)` (position-unit conversion) needed no changes — it already
  rescales `spacing["value"]` elementwise, which works the same whether that
  value is a scalar or a vector.

## What step 4 implements

- `swap_dims({old_dim: new_name})` (positional dict, or `swap_dims(old_dim=
  new_name)` as keywords): `new_name` must already be a **non-dimension**
  coordinate riding `old_dim` **alone** (`dims == (old_dim,)`) — the same
  restriction xarray's `swap_dims` has (it needs a coordinate that is a
  function of exactly that one dim). Renames the axis `old_dim -> new_name`
  (so `.names` and any axis descriptor follow, exactly like `rename`), and
  demotes `old_dim`'s previous dimension coordinate to a non-dimension
  coordinate riding the renamed axis, **under its old key** — matching
  xarray's own behaviour (`da.swap_dims({"time": "label"})` keeps a `"time"`
  coordinate around, now riding the `"label"` axis).
- Not a thin wrapper over `rename`: renaming `old_dim` onto `new_name` would
  re-key `old_dim`'s own dimension coordinate onto `new_name` too, colliding
  with the very coordinate being promoted — the same collision `rename`
  itself already raises on (step 3). `swap_dims` never re-keys a coordinate;
  it only remaps `dims` tuples through the axis substitution (exactly what
  `rename` already does to a coordinate's `dims`), so which entry counts as
  *the* dimension coordinate falls out structurally afterwards (`dims ==
  (key,)`), rather than needing to be assigned explicitly.
- A demoted **compact** dimension coordinate (e.g. `spacing`/`origin`)
  becomes, structurally, a single-dim compact *non-dimension* coordinate —
  a state the `coords=` constructor input itself refuses to create directly
  (see step 2's restriction above), but one the existing per-component affine
  slicing (`_slice_affine_coordinate`, already generic over `dims` length)
  handles correctly regardless of how it arose, so it keeps re-slicing
  exactly after `swap_dims`.
- `swap_dims_` is the in-place variant, matching `rename`/`rename_`.
- `set_index` (xarray's more general sibling, supporting a `MultiIndex`) is
  **not** implemented — there's no `MultiIndex` analogue planned here, and
  `swap_dims` already covers every case this model can express (promoting a
  single existing non-dimension coordinate to be the index).

## What step 5's `.sel` half implements (#82 phase 1)

- `.sel`'s indexer loop groups every queried coordinate **name** that spans
  more than one dim (a compact affine `Coordinate`, step 3) by its `dims`
  tuple. A group must supply exactly `len(dims)` values (one per spanned
  dim) — a square system — or it raises rather than falling back to a
  least-squares fit; a lone affine coordinate queried by itself is
  under-determined this way, not "not an index" (that error is now reserved
  for an ordinary, non-affine non-dimension coordinate).
- Each queried coordinate name in a group contributes one row of `A`
  (its `spacing` vector, already ordered along `dims`) and one entry of `b`
  (`target − origin`); `index = A⁻¹ b` (`torch.inverse`, chosen over
  `torch.linalg.solve` for wider old-torch support — the floor is 1.7),
  rounded to the nearest integer per dim. A singular (non-invertible) `A`
  raises, naming the offending coordinates.
- Never materialises the affine grid — the raw `spacing`/`origin` alone are
  enough, mirroring the 1-D compact `.sel` fast path (#110). Solved in
  float64 regardless of the spacing's own or default dtype, matching the
  1-D closed-form path's convention — a spacing difference near float32
  epsilon must not be silently rounded away into a near-singular system.
- Only `mode="round"` (the default) is supported; `floor`/`ceil`/`prev`/
  `next` have no well-defined meaning jointly across several coupled dims
  and raise `NotImplementedError`. `tolerance` applies per queried
  coordinate name against the rounded position's own value, same as a 1-D
  numeric `.sel` — a bare joint query is exact by default too.
- A joint affine query composes with ordinary 1-D `.sel` indexers in the
  same call (`x.sel(t=1.0, lat=52.1, lon=4.3)`) — but a dim resolved by the
  joint solve can't *also* be queried directly in the same call (raises,
  rather than silently letting one overwrite the other).
- `.interp`'s N-D `grid_pull` half is separate, follow-up work (not
  implemented here) — the pull itself needs genuine multi-axis simultaneous
  sampling, not the axis-by-axis separable interpolation `.interp` already
  does for 1-D coordinates.

## What step 5's `.interp` half implements (#82 phase 2)

- Groups `.interp`'s indexers the same way `.sel` does; the square-system
  requirement is identical, and only **one** joint group per call is
  supported for now (a second group raises `NotImplementedError` — call
  `interp` again for it).
- Inverts `A` **once** per call (shared across every query point, unlike
  `.sel`'s single-point solve) to a **fractional** index — never rounded —
  then genuinely pulls the tensor's values at those N-D positions via
  `fiery.interpol.grid_pull` (order ≥ 1, or order 0 with the backend
  installed) or a built-in separable nearest gather (order 0, no backend —
  nearest-neighbour rounding needs no cross-axis interpolation weight, so
  it stays exact without the optional dependency). Solved in float64
  regardless of the spacing's own/default dtype, mirroring both `.sel`'s
  and the 1-D `.interp` path's conventions.
- A query with **every** name given as a scalar is a single point: the
  spanned dims drop entirely, the same scalar-drops-the-axis convention
  `.interp`/`.sel` already have for a 1-D coordinate.
- Any name given as a **list**/tensor makes the whole group "many": every
  name's query broadcasts to a common length `N` (a length-1 query
  broadcasts against a longer one; two different non-1 lengths raise), and
  the spanned dims collapse into **one new axis** of `N` sampled points —
  *not* an outer-product grid. The dims are coupled, so you can't vary one
  queried name's value without moving through every spanned dim at once;
  this mirrors xarray's own vectorized/pointwise-indexing convention for a
  value-based query on a multi-dim coordinate (confirmed against
  `xarray.indexes.CoordinateTransformIndex`, its own affine-inversion
  index type — a bare scalar/list query isn't accepted there at all, only
  point-wise `DataArray` indexers sharing a dimension, collapsing to that
  one dimension).
- The new axis is named via `interp`'s new `name=` keyword if given, else
  the shared name of any query that is itself a named 1-D `XTensor` —
  `x.interp(lat=XTensor([...], names=("pts",)), lon=[...])` needs no
  `name=` at all, mirroring how xarray derives its vectorized-indexing
  result's new dimension from the *indexer* arrays' own shared dim name,
  not a separate parameter; disagreeing indexer names with no `name=`
  override to resolve them raise. With neither, the axis stays unnamed —
  `xstack`'s convention for a brand-new axis with nothing to infer from,
  since `torch.stack` itself gives no way to name one either. Either way
  it carries every queried name's own sampled values as a riding
  (non-dimension) coordinate — you get the world coordinates of your
  sampled points back, the same way xarray's vectorized indexing collapses
  its own auxiliary coordinates onto the new dimension for free.
- The new axis is inserted at the left-most spanned dim's position (a
  well-defined choice regardless of whether the spanned dims are adjacent
  in the tensor); every other axis keeps its name and relative order.
- Composes with ordinary 1-D `.interp` indexers in the same call
  (`x.interp(t=1.0, lat=52.1, lon=4.3)`) — attempting to *also* query a
  spanned dim directly afterward fails naturally (the dim no longer
  exists once the joint pull replaces it), with no special-case guard
  needed (unlike `.sel`, where the dim's integer position still exists
  after the solve, making a silent overwrite possible there).

## Open questions — resolved

1. **`.sel` on a non-index coordinate** — *resolved: no sugar.* Same as
   xarray: `.sel` only ever resolves against the index; selecting by a
   non-index coordinate always needs an explicit `swap_dims` first, with no
   implicit one-shot shortcut. Matches xarray exactly, and avoids the
   implicit-magic risk of silently redefining "the index" for one call.
2. **Affine query ergonomics** — *resolved.* No dedicated syntax for the
   joint query: `x.sel(lat=…, lon=…)` is just ordinary kwargs, and the
   implementation (step 5, #82 phase 1) recognises that `lat`/`lon` share
   `dims` and solves `A⁻¹` once for the pair. **Posedness**: `.sel` only
   supports a **square, invertible** affine (one world coordinate per
   spanned dim) — an under/over-determined map raises rather than falling
   back to a least-squares pseudo-inverse. Keeps the feature to the case the
   closed-form inverse actually covers; a non-square affine is out of scope
   here, not silently approximated.

*(Resolved since the first draft: storage unification — done; alignment /
reconciliation policy — above; `swap_dims` does rename the dim to the
promoted coordinate's name, matching xarray — above; both items in this
section — above.)*

## References

- xarray coordinates & indexes — <https://docs.xarray.dev/en/stable/user-guide/data-structures.html#coordinates>
- Proposals 0001 / 0002 — the single (dimension) coordinate this generalises
- Proposal 0004 — numeric `.sel`, which resolves against the index coordinate
- [#82](https://github.com/bagofseeds/fiery-xtensor/issues/82) — curvilinear / N-D `.interp` (affine = closed-form inverse)
- [#83](https://github.com/bagofseeds/fiery-xtensor/pull/83) — `_reconcile_coords` (dimension-coordinate reconciliation)
- [#84](https://github.com/bagofseeds/fiery-xtensor/pull/84) — storage unification
- [#72](https://github.com/bagofseeds/fiery-xtensor/pull/72) — non-dimension coordinates (step 2)
