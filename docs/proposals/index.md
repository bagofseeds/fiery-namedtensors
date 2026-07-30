---
icon: material/lightbulb-outline
---

# Proposals

Design proposals for `fiery.xtensor` — the larger, cross-cutting decisions,
written up for discussion before (and as) they land. Each one states the
motivation, the API, how it relates to [xarray](../guide/vs-xarray.md), and the
open questions.

- [0001 — Coordinates (values, spacing, units)](0001-units.md) — numeric
  coordinates: a physical position per element, compact `spacing`/`origin` or
  an explicit tensor, with position units.
- [0002 — Structured coordinates](0002-structured-coordinates.md) — richer
  coordinate labels (a label as a dict of fields) and querying by field.
- [0003 — Data units](0003-data-units.md) — physical units on a tensor's
  *values*, with dimensional algebra and per-axis units.
- [0004 — Numeric coordinate selection](0004-numeric-selection.md) — selecting
  by coordinate *value* with `.sel`, and computing off-grid values with
  `.interp`.
- [0005 — Multiple coordinates per axis](0005-multiple-coordinates.md) —
  non-dimension coordinates riding along a dim, and compact **affine**
  coordinates spanning several dims at once.
- [0006 — A `pint.Quantity`-shaped API](0006-quantity-api.md) — the small
  query surface (`dimensionality`, compatibility checks, a convert-and-drop
  shortcut) that rounds out data units into feeling like a `pint.Quantity`.
