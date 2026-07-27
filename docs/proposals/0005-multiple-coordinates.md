# Proposal 0005 — Multiple coordinates per axis (incl. affine coordinates)

| | |
| --- | --- |
| **Status** | Draft — **for discussion** (steps 1 and 2 below are implemented; not merged) |
| **Author** | (proposed) |
| **Created** | 2026-07-26 |
| **Updated** | 2026-07-27 — storage unification landed ([#84](https://github.com/bagofseeds/fiery-xtensor/pull/84)); non-dimension coordinates rebased on it |
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

## Compact **affine** coordinates (not yet implemented)

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
2. **Non-dimension (1-D) coordinates** — this PR, rebased on (1): `(dim,
   values)` input, `.coords` exposure, `.sel` "not an index" guard,
   conservative propagation (implemented directly against the unified storage
   — no separate `_extra_coords` attr).
3. **Compact affine / multi-dim coordinates** — vector `spacing` over a `dims`
   tuple: materialisation, exact affine slicing, learnable. *(No `.sel` yet.)*
4. **`swap_dims` / `set_index`** — promote a non-dimension coordinate to the
   index.
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
  `swap_dims` (not yet implemented — step 4).
- `rename` remaps the dim it rides on (not its own key); ops that reslice a
  named axis (`sort`/`flip`/`roll`/`gather`/`index_select`/`take_along_dim`)
  drop any non-dimension coordinate riding on the touched axis, alongside that
  axis' own dimension coordinate — both are conservatively invalidated the
  same way. `rename` also raises on a coordinate-name collision (renaming an
  axis onto an existing coordinate's name) rather than silently dropping one.
- Binary ops (broadcasting) don't carry non-dimension coordinates through yet
  (dropped) — see the reconciliation section above.
- Only **labels** or an **explicit** numeric tensor are accepted as `values`
  — a **compact** (`spacing`/`origin`) non-dimension coordinate raises
  `NotImplementedError`. Unlike a dimension coordinate, a non-dimension one
  isn't re-sliced when its dim is (no slice-tracking yet, step 6); a label or
  explicit coordinate is at least caught by the length check on resize, but a
  compact one binds to *any* size, so it would silently rebind to the wrong
  affine after a non-trivial slice instead of raising or dropping. Lifting
  this restriction is part of step 6.

## Open questions (remaining)

1. **`swap_dims` vs `set_index` surface** — which verbs; do we rename the dim to
   the promoted coordinate's name (xarray does)?
2. **`.sel` on a non-index coordinate** — once promotable, is one-shot `.sel` by
   a non-index coordinate (implicit swap) worth sugar?
3. **Affine query ergonomics** — spelling for the joint query
   (`.sel(lat=…, lon=…)`), and behaviour for an under/over-determined
   (non-square) affine (least-squares?).

*(Resolved since the first draft: storage unification — done (Q2); alignment /
reconciliation policy — above (Q3).)*

## References

- xarray coordinates & indexes — <https://docs.xarray.dev/en/stable/user-guide/data-structures.html#coordinates>
- Proposals 0001 / 0002 — the single (dimension) coordinate this generalises
- Proposal 0004 — numeric `.sel`, which resolves against the index coordinate
- [#82](https://github.com/bagofseeds/fiery-xtensor/issues/82) — curvilinear / N-D `.interp` (affine = closed-form inverse)
- [#83](https://github.com/bagofseeds/fiery-xtensor/pull/83) — `_reconcile_coords` (dimension-coordinate reconciliation)
- [#84](https://github.com/bagofseeds/fiery-xtensor/pull/84) — storage unification
