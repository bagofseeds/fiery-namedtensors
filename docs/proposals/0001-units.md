# Proposal 0001 — Coordinate metric (spacing)

| | |
| --- | --- |
| **Status** | Draft — representation under discussion |
| **Author** | (proposed) |
| **Created** | 2026-07-25 |
| **Tracking** | part of [#3](https://github.com/bagofseeds/fiery-xtensor/issues/3); builds on Proposal 0002 (structured coordinates); supersedes the "axis unit" sketch in [#39](https://github.com/bagofseeds/fiery-xtensor/issues/39) / [#48](https://github.com/bagofseeds/fiery-xtensor/issues/48) |

## Abstract

An axis' **metric** is how its integer positions map to physical coordinates —
the distance between consecutive elements, in physical units. That is a
**spacing**: a *value **and** a unit* (a `t` axis with spacing `0.5 s`), **not**
a bare unit. It is a property of the dimension, composes trivially across axes,
and a conversion rescales the **metric**, never the tensor data. This is one of
the two meanings of "unit"; the other is the **data** unit (Proposal 0003).

## Why "spacing", not "unit" (fixing an earlier slip)

A unit alone — "this axis is in seconds" — describes nothing measurable: you
cannot place element *i* without a value. What an axis needs is the **spacing**
`Δ` **with** its unit, so `distance(i, j) = |i − j| · Δ`; optionally an
**origin** `o`, giving `coordinate(i) = o + i · Δ`. An earlier draft carried a
bare `unit` field — that was the mistake this proposal corrects: the metric is a
*spacing*, not a *unit*.

## Prior art — how array packages represent "array coordinates"

Two families recur across scientific array libraries:

| package | representation |
| --- | --- |
| **xarray** | coordinate **variables** (labeled arrays), any dtype, **irregular** OK; `units` as a free `.attrs` entry (real units via `pint-xarray`) |
| **pandas** | the `Index` family — incl. compact **`RangeIndex`** and **`DatetimeIndex(freq=…)`** (regular grids), `IntervalIndex`, `MultiIndex` |
| **ITK / SimpleITK** | image **`Spacing` + `Origin` + `Direction`** (cosine matrix): `world = origin + direction · (spacing ∘ index)` |
| **NIfTI / nibabel** | a 4×4 **`affine`** (voxel→world, mm) + `pixdim` spacing + `xyzt_units` |
| **OME-NGFF** | axes (`name`/`type`/`unit`) + **`coordinateTransformations`** (`scale` + `translation`) |
| **napari** | per-layer **`scale` + `translate` + `units`** + `axis_labels` |

The split is clear:

- **Compact affine** — `scale`/`spacing` (+ `origin`/`translation`, sometimes
  `direction`) — for **regular** grids. Cheap, no per-element storage. This is
  the imaging-world default (ITK, NIfTI, OME-NGFF, napari), and *your* spacing is
  its diagonal `scale` term.
- **Explicit coordinate arrays** — a value per position — for **irregular**
  axes. Fully general (xarray, pandas, CF conventions), at the cost of storing a
  tick per element.

They are **not exclusive**: the general model is a function `i ↦ coordinate(i)`,
which a compact affine expresses in `O(1)` and an explicit array in `O(n)`.

## How generic should the metric be? (the open question)

In increasing power:

1. **Spacing** `Δ` + unit — regular, no offset. *(OME `scale`, napari `scale`,
   ITK `Spacing`.)* — your bullet.
2. **Spacing + origin** — `coordinate(i) = o + i·Δ`. *(OME `scale`+`translation`,
   ITK `Spacing`+`Origin`, pandas `DatetimeIndex`+`freq`.)*
3. **+ direction / full affine** — index→world affine, incl. rotated/sheared
   grids. *(ITK `Direction`, NIfTI `affine`.)* Couples axes.
4. **Explicit coordinate values** — arbitrary/irregular ticks, a value per
   position. *(xarray, pandas, CF.)*

**Leaning: 2 (spacing + origin), with 4 available separately** — it matches
OME-NGFF/ITK exactly, keeps regular grids `O(1)`, and irregular axes fall back to
explicit numeric coordinates (which structured coordinates, 0002, already store
per position). Full affine (3) couples axes and is a bigger, separable step.

## Representation — one form

The metric lives on the axis descriptor as a **`spacing`** (and, under option 2,
an `origin`):

- **Without a backend** (`unit_backend=None`, default): a **`(value, unit)`
  tuple** — `{"name": "x", "spacing": (0.5, "um")}` — stored and carried, not
  interpreted (today's opaque behaviour, now *with a value*).
- **With a backend**: a `Quantity` (`0.5 * ureg.um`), enabling normalised
  equality and conversion.

```python
names=[{"name": "x", "type": "space", "spacing": (0.5, "um")}]
```

Under a backend, `to_unit(x="nm")` rescales the spacing value + unit
(`0.5 µm → 500 nm`); the tensor data is untouched. Irregular axes use explicit
numeric coordinates (0002) instead of a spacing.

## Backwards compatibility

Unchanged by default. `spacing` is new optional axis-descriptor metadata; with
`unit_backend=None` it is a carried `(value, unit)` tuple with no behaviour.

## Open questions

1. **Generality 1–4 above** — spacing only, spacing+origin (leaning), affine, or
   explicit — and whether to include `origin` from the start.
2. Conversion API surface (`to_unit`, unit-aware `sel`, a `Quantity` input).
3. Backend interface shared with 0003 (`normalise`, `equal`, `convert`).
4. `spacing`/`origin` naming vs the coordinate-label `unit` used for *data*
   units in 0003 — different fields, different levels, so probably no clash now
   that the axis metric is `spacing` (not `unit`).

## References

- OME-NGFF axes + `coordinateTransformations` — <https://ngff.openmicroscopy.org/latest/>
- ITK image geometry (spacing/origin/direction) — <https://itk.org>
- xarray coordinates — <https://docs.xarray.dev>
- Proposal 0002 (structured coordinates) — explicit numeric coordinates
- Proposal 0003 (data units) — the *other* meaning of "unit"
