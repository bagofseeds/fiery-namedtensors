# Proposal 0001 — Coordinates (values, spacing, units)

| | |
| --- | --- |
| **Status** | Accepted — implemented (phases 1-3). Compact + explicit numeric coordinates (`Unitful`/magic-dict family, `coords={dim: {spacing, origin}}` or a unitful tensor, `coords[dim]["values"]` materialisation, affine-correct slicing, `Coordinate.to(unit)`, autograd-preserving) **and** the constructor cleanup: `names=` takes bare strings only, `axes=` is the general per-axis container (descriptors with `type`/`unit`/`orientation`/`coord`/`labels`), `names`/`coords` are shortcuts into it. Deferred: numeric `.sel` (#66), multiple coords per axis (#65) |
| **Author** | (proposed) |
| **Created** | 2026-07-25 |
| **Tracking** | part of [#3](https://github.com/bagofseeds/fiery-xtensor/issues/3); builds on Proposal 0002 (structured coordinates); multi-coordinate follow-up in [#65](https://github.com/bagofseeds/fiery-xtensor/issues/65); supersedes the "axis unit" sketch in [#39](https://github.com/bagofseeds/fiery-xtensor/issues/39) / [#48](https://github.com/bagofseeds/fiery-xtensor/issues/48) |

## Abstract

A **coordinate** describes the positions along an axis. It is **one concept**
with several representations — categorical **labels**, explicit **numeric**
values, or a **compact regular grid** (`spacing` + `origin`). A regular grid is
just a coordinate you have not materialised; spacing is not a separate "metric".

This corrects an earlier draft that modelled spacing as a distinct axis
*metric*, stored and handled apart from coordinates. It is not: a regular grid
is the compact form of a numeric coordinate, exactly as pandas `RangeIndex` is a
compact integer index. Numeric coordinates carry a **position unit** (a `t` axis
in seconds), which is orthogonal to the **data unit** of the tensor *values*
(Proposal 0003).

## The model

### A coordinate is one concept

`coords` maps **one coordinate per axis** (`{axis_name: coordinate}`). A
coordinate is either:

- an **array / (x)tensor** of *unitful values* (numeric positions) or *labels*
  (categorical), or
- a **dict** with `spacing` and/or `origin` — the compact form of a regular
  numeric coordinate, which materialises to `origin + i · spacing` on demand.

Labels are the categorical case of the same thing — a coordinate whose values
are unordered and unitless — so there is no separate "labels vs coordinates"
split. (Multiple coordinates per axis — xarray's dimension + non-dimension
coordinates, curvilinear grids — are a later additive step, tracked in
[#65](https://github.com/bagofseeds/fiery-xtensor/issues/65).)

### Unitful values

`spacing`, `origin`, and explicit numeric values are **unitful**. Any of these
forms is accepted on the way in:

- an `XTensor` with a unit (`xtensor(0.5, unit="mm")`),
- a backend quantity (`0.5 * ureg.mm`),
- a `(value, unit)` tuple (`(0.5, "mm")`),
- a `{"value": …, "unit": …}` dict,
- a bare value (dimensionless) or a bare unit string (value 1).

Once ingested, a unitful value is stored as a small **`dict` subclass** carrying
`value` and `unit`. It *is* a dict — `q["value"]`, `q["unit"]`, `dict(q)` all
work — so it needs no bespoke API, but it also:

- **preserves a tensor** `value` (a 0-rank `XTensor`) untouched, so a *learnable*
  spacing keeps its autograd graph;
- **converts** with `.to("um")` (rescales value + unit); its parts are read by
  item/attribute access (`sp["value"]`, `sp.unit`), so no `(value, unit)` helper
  is needed;
- stores `unit` as a **canonical string** (same rule as 0003), so it is
  picklable and backend-independent.

This replaces the earlier `Quantity` named-tuple. A named-tuple is a
2-*sequence* (`len(q) == 2`, `m, u = q`) while a 0-rank `XTensor` is a scalar, so
the two "forms of a quantity" diverged exactly on the sequence affordances — a
bug generator. A `dict` subclass has no such trap, and the name no longer
collides with `pint.Quantity`.

### Labels

A label is a `str` or a `{"name": …, "unit": …}` dict; `"red"` ≡
`{"name": "red"}` (Proposal 0002). The optional `unit` on a structured label is
the **data unit** of the tensor values at that position (Proposal 0003,
heterogeneous units) — *not* a coordinate position unit. The two never merge.

### One "magic dict" family

The unitful value, the coordinate, the structured label, and the axis
descriptor are all **the same kind of thing**: a `dict` subclass that stays a
plain dict (so it interops, pickles, and prints as one) while adding a little
magic:

- **derived / lazy keys** — a compact coordinate synthesises `["values"]` from
  `spacing`/`origin` on demand (**recomputed each access, not cached** — so a
  learnable spacing never goes stale);
- **tensor-preserving** — a tensor stored under a key (`value`, `spacing`,
  `values`) is kept live, never coerced, so autograd survives;
- **conversion** — a single `.to(unit)` helper (rescales value + unit); anything
  else is item access.

**Item access (`d["values"]`) is canonical** and works for every key.
**Attribute access is sugar**, whitelisted to keys that don't shadow the `dict`
API — `.value`, `.unit`, `.name`, `.spacing`, `.origin` — so `sp.value` reads
`sp["value"]`. The colliding names stay their `dict` methods: `d.values`,
`d.keys`, `d.items`, `d.get`, `d.update` are the mapping API, and the *key*
`values` is reached only as `coord["values"]`.

The class is **internal** — users pass plain inputs (`"0.5mm"`, `(0.5, "mm")`,
dicts, tensors) and receive magic dicts back from `.coords`/`.axes`; there is no
public constructor or exported name (so nothing collides with `pint.Quantity`).

### Two units, two roles

| unit | of what | where |
| --- | --- | --- |
| **position unit** | the coordinate's own values (a `t` axis in `s`) | `coords[axis]`'s `spacing`/`values` unit (0001) |
| **data unit** | the tensor *values* (`.unit`), incl. per-position on a label | `.unit` / structured-label `unit` (0003) |

## Construction

`names=` takes **strings only**; anything richer goes through `axes=`.

```python
# names: bare strings
xtensor(data, names=("c", "y", "x"))

# coords: {axis: coordinate}
xtensor(data, names=("c", "y", "x"), coords={
    "c": ["red", "green", "blue"],                 # categorical labels
    "y": {"spacing": "0.5mm"},                     # regular numeric (compact)
    "x": {"spacing": "0.5mm", "origin": "-16mm"},
})

# explicit (irregular) numeric coordinate = a unitful 1-D tensor
xtensor(sig, names=("t",), coords={
    "t": xtensor([0.0, 0.5, 2.0, 4.0], unit="s"),
})

# structured labels carry a per-position DATA unit (0003)
xtensor(v, names=("q", "t"), coords={
    "q": [{"name": "voltage", "unit": "V"},
          {"name": "current", "unit": "A"}],
})

# axes=: full descriptors; the coordinate lives under a `coord` key
xtensor(data, unit="mV", axes=[
    {"name": "c", "coord": ["red", "green", "blue"]},
    {"name": "x", "type": "space", "coord": {"spacing": "0.5mm", "origin": "-16mm"}},
])
```

`names=` and `coords=` are shortcuts that write into the general `axes=`
container; use whichever fits. Every unitful form folds to the same stored dict:

```python
{"spacing": "0.5mm"} == {"spacing": (0.5, "mm")} == {"spacing": {"value": 0.5, "unit": "mm"}}
                     == {"spacing": 0.5 * ureg.mm} == {"spacing": xtensor(0.5, unit="mm")}
```

## Access

```python
img.names             # ("c", "y", "x")                         — bare strings
img.axes              # descriptor dicts, coordinate under "coord"
img.coords            # {axis: coordinate}

img.coords["c"]       # ("red", "green", "blue")                — labels
img.coords["x"]       # {"spacing": {"value": 0.5, "unit": "mm"}, "origin": {...}}
img.coords["x"]["spacing"]["value"]   # 0.5    (a 0-rank tensor when learnable)
img.coords["x"]["spacing"].unit       # "mm"   — POSITION unit  (attribute sugar)

# a coordinate is a "magic" dict: `["values"]` is a DERIVED key, materialised
# from spacing/origin on demand — recomputed each access (no cache), so a
# learnable spacing never goes stale.  (`values` is item-access only: it
# collides with dict.values.)
img.coords["x"]["values"]   # 1-D unitful tensor [-16, -15.5, …]  (lazy, differentiable)

sp = img.coords["x"]["spacing"]
sp["value"], sp.unit         # 0.5, "mm"     — item access, or attribute sugar
sp.to("um")                  # {"value": 500.0, "unit": "um"}   — the one conversion helper

img.unit                     # "mV"  — whole-tensor DATA unit (0003)
img.coords["q"][0]["unit"]   # "V"   — per-position DATA unit (0003), distinct from any position unit
```

With `unit_backend=None` a unitful value is inert (carried, `unit` an opaque
string, no algebra/conversion) — today's behaviour, now with a value. Under a
backend it normalises, compares, and converts; converting a coordinate rescales
`spacing`/`origin`/`values` and their unit — **the tensor data is never
touched**.

## Selection

`.sel` selects by **categorical label** only (as today); numeric positions use
`.isel`. Selecting by a numeric coordinate value (xarray's `da.sel(x=0.5)`,
`method="nearest"` + tolerance) is a deliberate **later** addition, tracked in
[#66](https://github.com/bagofseeds/fiery-xtensor/issues/66) — it brings the
exact/nearest/tolerance machinery, and named-label selection covers the current
need. So this proposal adds coordinate *storage* and *arithmetic* (distance,
conversion, materialisation), not numeric selection.

## Autograd

A learnable spacing keeps its gradient because nothing is coerced to a Python
float:

```python
step = xtensor(torch.tensor(0.5), unit="mm", requires_grad=True)   # 0-rank unitful tensor
img  = xtensor(data, names=("y", "x"), coords={"x": {"spacing": step}})
img.coords["x"]["values"].sum().backward()   # gradient flows back to `step`
```

`{"spacing": step}` stores the live tensor; `coord_values` = `origin + arange(n)
· spacing` is a differentiable op.

## Relationship to xarray

We stay xarray-shaped and add one thing xarray lacks:

- our `coords[dim]` (one coordinate, name == dim) **is** xarray's *dimension
  coordinate* — the per-axis index;
- a coordinate is "values along the axis", any dtype (numeric **or**
  categorical), exactly like an xarray coordinate variable;
- **our extension:** coordinates may be stored **compactly**
  (`spacing`+`origin`) and be **unitful** — folding in pandas's
  `RangeIndex`/`DatetimeIndex(freq=…)` compactness and `pint-xarray`'s units as
  first-class, additive and xarray-compatible.

xarray's *multiple* coordinates per axis (non-dimension coordinates, a swappable
index, curvilinear multi-dim coordinates) are deferred to
[#65](https://github.com/bagofseeds/fiery-xtensor/issues/65).

## Prior art — how array packages represent "array coordinates"

| package | representation |
| --- | --- |
| **xarray** | coordinate **variables** (labeled arrays), any dtype, **irregular** OK; `units` via `pint-xarray` |
| **pandas** | the `Index` family — compact **`RangeIndex`** / **`DatetimeIndex(freq=…)`** (regular) through explicit `Index` (irregular) |
| **ITK / SimpleITK** | image **`Spacing` + `Origin` + `Direction`**: `world = origin + direction · (spacing ∘ index)` |
| **NIfTI / nibabel** | a 4×4 **`affine`** (voxel→world) + `pixdim` + `xyzt_units` |
| **OME-NGFF** | axes (`name`/`type`/`unit`) + **`coordinateTransformations`** (`scale` + `translation`) |
| **napari** | per-layer **`scale` + `translate` + `units`** + `axis_labels` |

The recurring split — **compact affine** (`scale`/`spacing` [+ `origin`]) for
regular grids vs **explicit arrays** for irregular ones — is unified here: both
are one coordinate, `O(1)` compact or `O(n)` explicit, over the same
`i ↦ coordinate(i)` function. **Rotated / sheared** grids (a full affine /
`Direction`; curvilinear coordinates) couple axes and land with the multi-dim
coordinates of [#65](https://github.com/bagofseeds/fiery-xtensor/issues/65).

## Backwards compatibility

- **`names=` becomes strings-only** — a small breaking change (it currently also
  accepts descriptor dicts; those move to `axes=`). Bare-name usage is
  unaffected.
- `coords` values may now be a compact `{spacing/origin}` dict or a unitful
  tensor in addition to label sequences — additive.
- With `unit_backend=None` everything is inert and carried, as today.

## Settled

- **Materialisation** — `coords["x"]["values"]`, a derived key on the magic
  coordinate dict; `coords[axis]` returns the stored form (compact
  `{spacing/origin}` or explicit) and `["values"]` materialises **fresh each
  access, no cache**.
- **Attribute access** — item access is canonical for every key; whitelisted
  attribute sugar (`.value`/`.unit`/`.name`/`.spacing`/`.origin`) for keys that
  don't shadow the `dict` API.
- **Helper surface & visibility** — one `.to(unit)` conversion helper; the magic
  dict family is **internal** (no public constructor / exported name, so no
  `pint.Quantity` clash).

## Deferred (tracked)

- **Numeric `.sel`** — value-based selection with nearest/tolerance semantics
  (xarray parity) → [#66](https://github.com/bagofseeds/fiery-xtensor/issues/66).
- **Multiple coordinates per axis** — xarray non-dimension / curvilinear
  coordinates, a swappable index; the singular `coord` key revisits then →
  [#65](https://github.com/bagofseeds/fiery-xtensor/issues/65).

## References

- OME-NGFF axes + `coordinateTransformations` — <https://ngff.openmicroscopy.org/latest/>
- ITK image geometry (spacing/origin/direction) — <https://itk.org>
- xarray coordinates — <https://docs.xarray.dev>
- pandas `RangeIndex` — the compact-index prior art
- Proposal 0002 (structured coordinates) — labels / per-position storage
- Proposal 0003 (data units) — the *other* unit (of the values)
