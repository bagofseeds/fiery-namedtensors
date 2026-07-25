# Proposal 0001 — Coordinate units

| | |
| --- | --- |
| **Status** | Draft — converging |
| **Author** | (proposed) |
| **Created** | 2026-07-25 |
| **Tracking** | part of [#3](https://github.com/bagofseeds/fiery-xtensor/issues/3); builds on Proposal 0002 (structured coordinates); supersedes the "axis unit" sketch in [#39](https://github.com/bagofseeds/fiery-xtensor/issues/39) / [#48](https://github.com/bagofseeds/fiery-xtensor/issues/48) |

## Abstract

A **coordinate unit** qualifies the *values along an axis* — a `t` axis whose
ticks are `0.0, 0.5, 1.0` **seconds**. It is one of the two genuinely different
things "unit" can mean (the other is the **data** unit — the unit of the tensor
*values* — which is Proposal 0003). This is the *coordinate metric*: a property
of the **dimension**, it composes trivially, and a conversion rescales the
**ticks**, never the tensor data.

## Why this replaces "axis units"

An earlier draft proposed a unit on an axis *name* with nothing else. That was
incomplete: a unit with no value describes nothing measurable. What an axis
actually needs is the **coordinate metric** — the value at each position and its
unit (equivalently, for a regular grid, a *spacing*). Spacing is just the
regular special case of coordinate values (`Δ = 0.5 µm` ≡ ticks
`0, 0.5, 1.0, … µm`). So the "axis unit" folds into **coordinate values + a
unit**, and there is no separate axis-level unit concept.

## Representation — one form

A coordinate stays **a single tuple of per-position values** (today's labels;
they may already be numbers). There is **no** second "compact spacing" type — a
regular grid is written as its explicit values.

- **Without a unit backend** (`unit_backend=None`, the default) a coordinate is
  a **plain tuple**, exactly as today — no unit, no conversion, string/label
  semantics unchanged.
- **With a backend selected** (`set_options(unit_backend="pint")`) a numeric
  coordinate gains a **unit**, enabling normalised equality and conversion.

The unit lives on the axis descriptor's existing `unit` field (one unit per
coordinate, since all ticks of one axis share it) — so
`names=[{"name": "t", "unit": "s"}]` + `coords={"t": (0.0, 0.5, 1.0)}` is a time
coordinate in seconds. (When the axis has no explicit coordinate, the unit is
the metric of the implicit integer index — still just metadata.)

## Behaviour under a backend

- **Normalise + normalised equality** (from the axis-unit discussion): the
  `unit` is normalised on set, and `_merge_axis_meta` compares normalised units
  so `"um"` and `"micrometer"` agree.
- **Conversion**: `x.to_unit(t="ms")` rescales the numeric coordinate values
  (`0, 0.5, 1.0 → 0, 500, 1000`) and updates the unit — **the tensor data is
  untouched**.
- **Unit-aware `sel`** *(optional)*: `x.sel(t=Quantity(500, "ms"))` converts to
  the coordinate's unit, then matches a position.

Everything is gated on `unit_backend`; with `None` none of it activates and the
package behaves exactly as it does now.

## Open questions

1. The conversion API surface (`to_unit`, unit-aware `sel`, a `Quantity` input
   type) — how much to expose first.
2. Whether a unit on an axis with **no** coordinate values is allowed (a pure
   metric label) or requires coordinates to be meaningful.
3. Backend interface shared with Proposal 0003 (`normalise`, `equal`, and — new
   here — `convert(value, from, to)`).
4. **The one thing worth confirming: `unit` at two levels.** The *tick* unit
   here lives on the **axis descriptor** (`names=[{"name": "t", "unit": "s"}]`),
   while a *data* unit that varies per position lives on a **coordinate label**
   (`coords={"q": [{"name": "v", "unit": "V"}]}`, Proposal 0003). Same key,
   different meaning at different levels (whole-axis metric vs per-position data
   unit). They're structurally distinct (axis descriptor vs coordinate label),
   so this is workable — but if the shared word is a concern we could rename one
   (e.g. axis `spacing_unit` / coordinate `unit`). Otherwise 0001 is settled.

## References

- OME-NGFF axes + `coordinateTransformations` — <https://ngff.openmicroscopy.org/latest/>
- Proposal 0002 (structured coordinates) — the substrate for numeric/rich coordinates
- Proposal 0003 (data units) — the *other* meaning of "unit"
