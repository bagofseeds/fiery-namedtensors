# Proposal 0001 — Coordinate metric (spacing)

| | |
| --- | --- |
| **Status** | Draft — converging (spacing + origin; `Quantity` representation) |
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

## Generality — decided: spacing + origin

The metric is **spacing + origin** (level 2 of the ladder): `coordinate(i) =
origin + i · spacing`, each of `spacing`/`origin` a value **and** a unit. It
matches OME-NGFF (`scale` + `translation`) and ITK (`Spacing` + `Origin`)
exactly, keeps regular grids `O(1)`, and leaves the harder cases as clean
extensions:

- **irregular** axes → explicit numeric coordinates (structured coordinates,
  0002, already store a value per position); a `spacing`/`origin` is just the
  compact regular form of the same `i ↦ coordinate(i)` function.
- **rotated / sheared** grids → a full affine (level 3, ITK `Direction` / NIfTI
  affine) — couples axes, a separable step for later.

## Representation — a lightweight `Quantity`

`spacing` and `origin` are each a **`Quantity`** — a small `(value, unit)`
named-tuple, deliberately parallel to how 0003 stores a unit as a string rather
than wrapping data in `pint`:

- **`value`** — a Python scalar, **or** a **0-rank `XTensor`** when the value
  needs autograd (a *learnable* spacing). This is the neat bit: a
  differentiable scalar-with-a-unit already *is* a 0-d data-united tensor
  (0003), so `Quantity` and "0-d `XTensor` with `.unit`" are two forms of the
  same thing — swap in the tensor form exactly when you need gradients.
- **`unit`** — a **canonical string** (backend-normalised), same storage rule as
  0003, so the metric is picklable and backend-independent.

A `Quantity` is really `value · unit`, so a bare half **folds** into the
canonical pair — the missing half taking its identity:

| input | folds to | meaning |
| --- | --- | --- |
| a bare **value** `v` (number / 0-d tensor) | `Quantity(v, "")` | dimensionless |
| a bare **unit** `u` (str / pint `Unit`) | `Quantity(1, u)` | one `u` |
| a `(value, unit)` pair, or a `Quantity` | itself | — |

So `spacing=0.5` ≡ `Quantity(0.5, "")` and `spacing="um"` ≡ `Quantity(1, "um")`;
both normalise to the same `(value, unit)` shape. (`""` is the dimensionless
unit — the backend normalises it; distinct from `None`/*unset*.)

`Quantity` carries **its own light arithmetic**, deferring the unit part to the
active `unit_backend` (the shared `_units` algebra) and never wrapping the data:

```python
Quantity(0.5, "um") * 2            # Quantity(1.0, "um")           — scale
Quantity(0.5, "um") * u.mm         # combine with a real pint unit → "um·mm"…
Quantity(0.5, "um").to("nm")       # Quantity(500.0, "nm")         — convert
Quantity(grad_scalar_xt, "um")     # value is a 0-d XTensor → learnable + autograd
```

Stored on the axis descriptor:

```python
names=[{"name": "x", "type": "space",
        "spacing": Quantity(0.5, "um"), "origin": Quantity(0.0, "um")}]
# a bare (value, unit) tuple is accepted and coerced to a Quantity
```

With `unit_backend=None` a `Quantity` is inert (carried, its `unit` an opaque
string, no algebra) — today's behaviour, now *with a value*. Under a backend it
normalises, compares, and converts (`to_unit(x="nm")` rescales `spacing`/`origin`
and their unit; the **tensor data is never touched**).

## Backwards compatibility

Unchanged by default. `spacing`/`origin` are new optional axis-descriptor
metadata; with `unit_backend=None` they are inert carried `Quantity`s.

## One `Quantity` *interface*, two implementations (duck-typed)

The named-tuple form and the 0-d-`XTensor` form should share an **API
(protocol), not a base class**:

- **Not a shared base class.** `XTensor` is already a `torch.Tensor` subclass,
  and Tensor subclassing is delicate — bolting on a second base (a
  `QuantityBase`) risks metaclass/`__new__` conflicts for no real gain, since
  the two forms store their value completely differently (a Python scalar in a
  tuple vs. a live tensor). Sharing *implementation* buys little.
- **A shared duck-typed API.** Define a small, `runtime_checkable`
  `Quantity` **protocol** — `.unit` (str), `.magnitude` (the value),
  `to_unit(u)` (convert), and the arithmetic (`*`/`/` combine units, `+`/`-`
  require compatible, `* <unit>` attaches). Anything implementing it *is* a
  quantity; `is_quantity(x)` is a structural check, not `isinstance` on a
  concrete class.
- **The tensor form nearly implements it already.** A 0-d `XTensor` with a data
  unit (0003) already has `.unit` and `to_unit`; its `.magnitude` is the tensor
  itself, and its arithmetic is plain torch. So the *same* work that builds
  data-unit algebra (0003 phase 2) makes the tensor form satisfy the protocol —
  we then add a lightweight named-tuple `Quantity` (backed by `_units`) that
  implements the same surface for the cheap, non-grad, no-allocation case.

Each form's arithmetic returns *its own* kind (tuple·scalar → tuple;
tensor·scalar → tensor); consumers (spacing math, conversion, `sel`) only touch
the protocol, so they never care which they hold. `Quantity` (the protocol +
the named-tuple impl) lives in `_units`, shared by 0001 and 0003.

## Open questions

1. Conversion API surface (`to_unit`, unit-aware `sel` taking a `Quantity`).
2. The 0-d-`XTensor`-as-`Quantity` unification — how automatic: does a bare 0-d
   united tensor auto-satisfy `is_quantity`, and does `Quantity(tensor)` read
   the unit off the tensor rather than double-storing it?
3. Exact protocol surface — `.magnitude` vs `.value`; which operators are
   required vs optional.
4. Naming: `spacing`/`origin` on the axis descriptor vs the coordinate-label
   `unit` (data units, 0003) — different fields/levels, so no clash.

## References

- OME-NGFF axes + `coordinateTransformations` — <https://ngff.openmicroscopy.org/latest/>
- ITK image geometry (spacing/origin/direction) — <https://itk.org>
- xarray coordinates — <https://docs.xarray.dev>
- Proposal 0002 (structured coordinates) — explicit numeric coordinates
- Proposal 0003 (data units) — the *other* meaning of "unit"
