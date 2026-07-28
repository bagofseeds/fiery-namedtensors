# Proposal 0005 — Multiple coordinates per axis (incl. affine coordinates)

| | |
| --- | --- |
| **Status** | Draft — **for discussion** (steps 1–4 below are implemented) |
| **Author** | (proposed) |
| **Created** | 2026-07-26 |
| **Updated** | 2026-07-28 — `swap_dims` (step 4) implemented |
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
5. **Affine `.sel` / `.interp`** — closed-form `A⁻¹` inverse + (for interp)
   N-D `grid_pull`, joint query over the coupled dims (ties off
   [#82](https://github.com/bagofseeds/fiery-xtensor/issues/82) phase 1).
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

## Open questions (remaining)

1. **`.sel` on a non-index coordinate** — once promotable via `swap_dims`, is
   one-shot `.sel` by a non-index coordinate (implicit swap) worth sugar?
2. **Affine query ergonomics** — spelling for the joint query
   (`.sel(lat=…, lon=…)`), and behaviour for an under/over-determined
   (non-square) affine (least-squares?).

*(Resolved since the first draft: storage unification — done; alignment /
reconciliation policy — above; `swap_dims` does rename the dim to the
promoted coordinate's name, matching xarray — above.)*

## References

- xarray coordinates & indexes — <https://docs.xarray.dev/en/stable/user-guide/data-structures.html#coordinates>
- Proposals 0001 / 0002 — the single (dimension) coordinate this generalises
- Proposal 0004 — numeric `.sel`, which resolves against the index coordinate
- [#82](https://github.com/bagofseeds/fiery-xtensor/issues/82) — curvilinear / N-D `.interp` (affine = closed-form inverse)
- [#83](https://github.com/bagofseeds/fiery-xtensor/pull/83) — `_reconcile_coords` (dimension-coordinate reconciliation)
- [#84](https://github.com/bagofseeds/fiery-xtensor/pull/84) — storage unification
- [#72](https://github.com/bagofseeds/fiery-xtensor/pull/72) — non-dimension coordinates (step 2)
