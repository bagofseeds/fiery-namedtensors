# Proposal 0004 — Numeric coordinate selection & interpolation

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
proposal adds value-based access along a numeric coordinate, and — following
xarray — splits it across **two verbs**:

- **`.sel`** — *pick* an existing position at (or nearest to) a value. The
  result is a **subset** of the original data; it needs no extra dependency.
- **`.interp`** — *compute* values at arbitrary positions. The result is
  **new** data sampled off the grid; higher orders use the optional
  `fiery.interpol` backend.

## Motivation

A numeric coordinate exists to say *where* each element sits. Two different
questions follow, and xarray gives each its own verb:

- "the frame **at** 2 s / **nearest** 12 mm" — a lookup that returns an element
  that already exists → **`.sel`**.
- "the value **interpolated at** 2.5 s" — a resample that returns something
  between the samples → **`.interp`**.

Keeping them separate (rather than folding interpolation into `.sel`'s
`method=`) matches xarray, keeps `.sel` dependency-free and lossless, and lets
`.interp` grow its own controls (order, boundary, extrapolation) without
overloading `.sel`.

## `.sel` — selection (built in)

xarray's `.sel(x=value, method=None, tolerance=None)`:

- **`method=None`** (default) — exact match; raise if the value is not present.
- **`method="nearest"`** — snap to the closest coordinate value.
- **`tolerance`** — cap the allowed gap for `"nearest"`; raise if exceeded.

We mirror this surface. We *add* one thing xarray core lacks: the selector is
**unit-aware** — `x.sel(t="2000ms")` converts into the coordinate's position
unit before matching (Proposal 0001's `Unitful`/backend).

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
  positions on demand; an **explicit** one searches its array.

`.sel` gains two reserved keyword arguments, `method` and `tolerance`; every
other keyword is a `dim=selector`. (A dim literally named `method`/`tolerance`
is unreachable through kwargs — the same limitation xarray has; a dict form can
be added if needed.)

## `.interp` — interpolation

Where `.sel` picks, `.interp` **computes** a value at any position — the natural
next verb once a coordinate carries real distances. `fiery` already ships a
resampler, [`fiery.interpol`](https://github.com/bagofseeds/fiery-interpol), so
`.interp` maps the coordinate values to fractional indices and hands the work to
`grid_pull`; the coordinate provides the affine (`index = (value − origin) /
spacing`).

```python
x.interp(t=2.5)                     # one point -> drops the axis
x.interp(t=[0.0, 0.5, 1.0])        # several  -> keeps the axis
x.interp(t="2.5s")                 # unitful (backend converts)
x.interp(t=q, method="cubic")      # a query tensor; gradients flow to q
```

### Order (`method`)

`method` is the interpolation **order**, sharing `fiery.interpol`'s vocabulary:

| `method` | order | backend |
| --- | --- | --- |
| `"nearest"` / `0` | 0 | **built in** (a gather) |
| `"linear"` *(default)* / `1` | 1 | `fiery.interpol` |
| `"quadratic"` / `2` | 2 | `fiery.interpol` |
| `"cubic"` / `3` | 3 | `fiery.interpol` |
| any int order | n | `fiery.interpol` |

**Nearest is built in** — a numeric selection, essentially — so `.interp` works
with no extra dependency for the common case. Every higher order needs the
optional backend:

```
pip install fiery-xtensor[interp]
```

Calling `x.interp(..., method="linear")` without it raises with that hint.

### Out-of-range: `bound` and `extrapolate`

A query can land past the ends. Two controls, both mirroring `fiery.interpol`
and both exposed as **`fiery` options** with per-call overrides:

- **`bound`** — the boundary condition: `"replicate"` *(default)* clamps to the
  edge value, `"wrap"` wraps, `"reflect"`/`"mirror"` fold, `"zero"` pads with
  zeros, … .
- **`extrapolate`** — whether to sample past the ends at all (`True` by
  default; with `"replicate"` this is exactly the clamp).

```python
set_options(interp_bound="reflect")            # global default
x.interp(t=[...])                              # uses reflect
x.interp(t=[...], bound="wrap")                # per-call override
```

The default `interp_bound="replicate"` was chosen as the least-surprising
"don't invent data past the edge" behaviour (clamp/hold), matching how images
are usually resampled.

### Result

- A **scalar** query drops the axis (like `.sel`); a **list**/tensor keeps it.
- The interpolated axis' coordinate becomes the **queried positions** (an
  explicit coordinate); categorical labels on that axis are dropped (their
  positions no longer exist). Other axes, names, and the data unit ride
  through.
- Because the query flows through `grid_pull`, **gradients propagate** to both
  the queried positions and (for a learnable compact coordinate) the spacing.
- Several dims interpolate **independently** (axis by axis), matching xarray's
  orthogonal `.interp(x=…, y=…)`.

### Regular coordinates first

The first slice supports **regular** (compact `spacing`/`origin`) coordinates
only — the affine value→index map is exact. Interpolation over an **irregular**
(explicit, non-uniform) coordinate needs a monotonic value→index inversion
(a searchsorted-style step) and is deferred to its own issue; calling `.interp`
on one raises `NotImplementedError` for now.

## What the first slice implements

- `.sel(method=None, tolerance=None, **indexers)` — value-based selection,
  exact/nearest, unit-aware, on compact **and** explicit coordinates
  (unchanged from the prior slice).
- `.interp(method="linear", bound=None, extrapolate=None, **indexers)` —
  nearest built in; orders ≥ 1 via `fiery.interpol.grid_pull`; `interp_bound`
  (default `"replicate"`) and `interp_extrapolate` (default `True`) options
  with per-call overrides; **regular coordinates only**.
- The optional `fiery-xtensor[interp]` extra (the `fiery.interpol` backend).
- Tests: nearest without the backend, linear/cubic with it, scalar-drops-axis,
  multi-axis preservation, unit conversion, gradient flow, `bound` option +
  override, and the irregular-coordinate guard.

## Open questions (for discussion)

1. **Irregular-coordinate interpolation** — the deferred half above. Worth an
   O(log n) searchsorted inversion, or is regular-only enough in practice?
2. **Range selection** on `.sel` — `x.sel(t=slice("1s", "5s"))` (a value range
   → a contiguous slice). Natural and useful; not in this slice.
3. **`.sel(method="ffill"/"bfill")`** — pad-forward/back like xarray; worth it,
   or does `.interp(method="nearest")` cover the intent?
4. **`O(1)` compact fast path for `.sel`** (`round((v − origin) / spacing)`) vs
   the simple materialise-and-`argmin`.
5. **True N-D interpolation** — separable axis-by-axis (this slice, == xarray)
   vs a single N-D `grid_pull` for a genuine multivariate spline.
6. **Multi-coordinate interaction** — once an axis can carry several
   coordinates ([#65](https://github.com/bagofseeds/fiery-xtensor/issues/65) /
   Proposal 0005), `.sel`/`.interp(dim=…)` target that dim's **index**
   coordinate; this proposal assumes one coordinate per axis.

## References

- xarray `.sel` (`method`, `tolerance`) and `.interp` — <https://docs.xarray.dev>
- Proposal 0001 (numeric coordinates) — the positions this selects/interpolates
- [`fiery.interpol`](https://github.com/bagofseeds/fiery-interpol) — `grid_pull`,
  interpolation orders, and boundary conditions
- Proposal 0005 (multiple coordinates per axis) — the index a future
  `.sel`/`.interp` resolves against
