# Proposal 0004 — Numeric coordinate selection

| | |
| --- | --- |
| **Status** | Draft — **for discussion** (a first slice is implemented; not merged) |
| **Author** | (proposed) |
| **Created** | 2026-07-26 |
| **Tracking** | [#66](https://github.com/bagofseeds/fiery-xtensor/issues/66); builds on Proposal 0001 (numeric coordinates) |

## Abstract

Proposal 0001 landed **numeric coordinates** (a physical position per element,
compact `spacing`/`origin` or an explicit tensor) but kept `.sel` to
**categorical labels** only — numeric positions were reached with `.isel`. This
proposal adds **value-based selection**, the xarray way: `x.sel(t="2s")` picks
the position at (or nearest to) a coordinate value, unit-aware, with xarray's
`method` / `tolerance` controls.

## Motivation

A numeric coordinate exists to say *where* each element sits. Selecting by that
position — "the frame at 2 s", "the slice nearest 12 mm" — is the natural next
verb, and matches how xarray/pandas index numeric coordinates. Without it, users
must convert a value to an index by hand (defeating the coordinate).

## xarray parity

xarray's `.sel(x=value, method=None, tolerance=None)`:

- **`method=None`** (default) — exact match; raise if the value is not present.
- **`method="nearest"`** — snap to the closest coordinate value.
- **`tolerance`** — cap the allowed gap for `"nearest"`; raise if exceeded.
- (`"ffill"`/`"bfill"` — pad forward/back; deferred, see open questions.)

We mirror this surface. We *add* one thing xarray core lacks: the selector is
**unit-aware** — `x.sel(t="2000ms")` converts into the coordinate's position
unit before matching (Proposal 0001's `Unitful`/backend).

## The API

```python
x.sel(t=2.0)                                  # exact tick -> drops the axis
x.sel(t="2s")                                 # unitful value (backend converts)
x.sel(t=(2, "s"))                             # a (value, unit) tuple
x.sel(t=[0.5, 2.0])                           # a list keeps the axis (advanced)
x.sel(t=1.7, method="nearest")                # snap to the closest tick
x.sel(t=1.7, method="nearest", tolerance=0.1) # ... but fail if the gap > 0.1
```

- A **scalar** selector drops the axis (like integer indexing); a **list**
  keeps it (advanced index over the chosen positions).
- A **tuple** is a unitful `(value, unit)`, *not* a list of selectors — use a
  list for several values.
- **Exact** match uses a small relative tolerance (materialised positions are
  floats); with no match and no `method`, it raises and suggests
  `method="nearest"`.
- Works on both coordinate forms: a **compact** coordinate materialises its
  positions on demand; an **explicit** one searches its array. (A future
  optimisation can make the compact case `O(1)` via
  `round((value − origin) / spacing)`; the first slice materialises and
  `argmin`s, which is simple and correct.)

`.sel` gains two reserved keyword arguments, `method` and `tolerance`; every
other keyword is a `dim=selector`. (A dim literally named `method`/`tolerance`
is unreachable through kwargs — the same limitation xarray has; a dict form can
be added if needed.)

## What the first slice implements

- `sel(method=None, tolerance=None, **indexers)`; a numeric `Coordinate`
  selector routes to `_numeric_select` (scalar → one index, list → several),
  unit-aware via `_selector_value`.
- Exact (default) vs `method="nearest"`, with `tolerance` in the position unit.
- Compact **and** explicit coordinates; tests for each, incl. unit conversion.

## Open questions (for discussion)

1. **Range selection** — `x.sel(t=slice("1s", "5s"))` (a value range → a
   contiguous slice). Natural and useful; not in the first slice.
2. **`method="ffill"/"bfill"`** — pad-forward/back like xarray; worth it?
3. **`O(1)` compact fast path** vs the simple materialise-and-`argmin`.
4. **Attribute access** — should `x.<something>` ever reach numeric sel? (No —
   attribute access stays categorical-labels-only.)
5. **Multi-coordinate interaction** — once an axis can carry several
   coordinates ([#65](https://github.com/bagofseeds/fiery-xtensor/issues/65) /
   Proposal 0005), `.sel(dim=…)` targets that dim's **index** coordinate; this
   proposal assumes the current one-coordinate-per-axis model.

## References

- xarray `.sel` (`method`, `tolerance`) — <https://docs.xarray.dev>
- Proposal 0001 (numeric coordinates) — the positions this selects over
- Proposal 0005 (multiple coordinates per axis) — the index a future `.sel`
  resolves against
