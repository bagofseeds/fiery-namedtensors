# Proposal 0004 — Numeric coordinate selection & interpolation

| | |
| --- | --- |
| **Status** | Draft — **for discussion** (the first slice — regular coordinates — landed; irregular `.interp` (#73) landed too) |
| **Author** | (proposed) |
| **Created** | 2026-07-26 |
| **Updated** | 2026-07-28 — irregular (non-uniform) 1-D `.interp` landed ([#73](https://github.com/bagofseeds/fiery-xtensor/issues/73), nearest/linear via `searchsorted`) |
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

Selection snaps a value onto an existing tick. Two orthogonal controls, chosen
so that a *descending* coordinate stays unambiguous (where xarray's single
`method` gets confusing):

- **`mode`** — which tick to snap to:
  - `"round"` *(default)* — the **nearest** tick by value;
  - `"floor"` / `"ceil"` — the largest tick **≤** / smallest tick **≥** the
    value (**value** space — orientation-robust);
  - `"prev"` / `"next"` — the neighbouring tick at the **lower** / **higher**
    index (**tick order**; needs a monotonic coordinate).
- **`tolerance`** — a cap on the gap (in the position unit). A **bare**
  `.sel(t=v)` is **exact** (`tolerance=0`); passing a `mode` implies an
  **unbounded** snap unless a `tolerance` is given (matching xarray, where
  supplying a `method` defaults `tolerance` to `None`).

`floor`/`ceil` and `prev`/`next` coincide on an **ascending** coordinate and
**swap** on a **descending** one; `round` is direction-agnostic. We *add* one
thing xarray core lacks: the selector is **unit-aware** — `x.sel(t="2000ms")`
converts into the coordinate's position unit before matching.

```python
x.sel(t=2.0)                          # exact tick -> drops the axis
x.sel(t="2s")                         # unitful value (backend converts)
x.sel(t=(2, "s"))                     # a (value, unit) tuple
x.sel(t=[0.5, 2.0])                   # a list keeps the axis (advanced)
x.sel(t=1.7, mode="round")            # nearest tick (unbounded)
x.sel(t=1.7, mode="floor")            # largest tick <= 1.7 (value space)
x.sel(t=1.7, mode="round", tolerance="0.1s")  # ... but fail if the gap > 0.1 s
```

- A **scalar** selector drops the axis (like integer indexing); a **list**
  keeps it (advanced index over the chosen positions).
- A **tuple** is a unitful `(value, unit)`, *not* a list of selectors — use a
  list for several values.
- **Exact** (`tolerance=0`) uses a small relative tolerance (materialised
  positions are floats); with no tick within tolerance it raises.
- Works on both coordinate forms: a **compact** coordinate materialises its
  positions on demand; an **explicit** one searches its array.

### xarray compatibility

`method=` is accepted as an **alias** for `mode` (`nearest`→`round`,
`pad`/`ffill`→`prev`, `backfill`/`bfill`→`next`) — pass one of `mode`/`method`,
not both. xarray's fill methods are *positional*, so they map to the tick-order
family; on the usual ascending coordinate they coincide with `floor`/`ceil`.

`.sel` reserves the keyword arguments `mode`, `tolerance`, and `method`; every
other keyword is a `dim=selector`. (A dim literally named one of those
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

### Regular coordinates, then irregular (#73, landed)

The first slice supported **regular** (compact `spacing`/`origin`) coordinates
only — the affine value→index map is exact. Interpolation over an
**irregular** (explicit, non-uniform) coordinate needed a monotonic
value→index inversion instead of a closed-form affine one; that landed in
[#73](https://github.com/bagofseeds/fiery-xtensor/issues/73) for `nearest`/
`linear` (see "What #73 adds" below) — **exact**, via `torch.searchsorted` +
a local linear inverse between the two bracketing ticks, the same
piecewise-linear map those two methods already sample between. Higher orders
(`"quadratic"`/`"cubic"`/…) on an irregular coordinate raise
`NotImplementedError` **permanently**, not as a pending TODO — resolved as
**not planned** in
[#81](https://github.com/bagofseeds/fiery-xtensor/issues/81): the exactness
of nearest/linear relies on both the inversion and the interpolation being
locally affine between the same two bracketing ticks. A uniform-index-space
spline basis (what `fiery.interpol.grid_pull` provides) is a different,
incompatible thing from a true non-uniform spline in value space — there's no
xtensor-side inversion that bridges the two, so only 0th/1st order are
supported on an irregular coordinate.

## What the first slice implements

- `.sel(mode="round", tolerance=None, method=None, **indexers)` — value-based
  selection with the five modes (`round`/`floor`/`ceil`/`prev`/`next` + xarray
  aliases), unit-aware, on compact **and** explicit coordinates; bare = exact,
  a mode implies an unbounded snap.
- `.interp(method="linear", bound=None, extrapolate=None, **indexers)` —
  nearest built in; orders ≥ 1 via `fiery.interpol.grid_pull`; `interp_bound`
  (default `"replicate"`) and `interp_extrapolate` (default `True`) options
  with per-call overrides; **regular coordinates only** at this point.
- The optional `fiery-xtensor[interp]` extra (the `fiery.interpol` backend).
- Tests: nearest without the backend, linear/cubic with it, scalar-drops-axis,
  multi-axis preservation, unit conversion, gradient flow, `bound` option +
  override, and the irregular-coordinate guard.

## What #73 adds

- `.interp(method="nearest"|"linear", **indexers)` on an **irregular**
  (explicit-values) 1-D coordinate: `torch.searchsorted` brackets each query
  between two adjacent ticks (ascending, or a descending coordinate searched
  via its reverse), then `k + (query − ticks[k]) / (ticks[k+1] − ticks[k])`
  is the exact fractional index fed to the same `_interp_pull` the regular
  path already uses.
- The coordinate must be **strictly monotonic** (ascending or descending);
  a non-monotonic one raises `ValueError` (no unique inverse otherwise).
- **Differentiable** w.r.t. both the query and the coordinate's own values
  (only the bracket *search* runs on detached copies — an index has no
  useful gradient; the returned fraction is computed from the original,
  gradient-carrying tensors).
- Higher orders (`method` resolving to order ≥ 2) raise `NotImplementedError`
  rather than silently doing the uniform-index-space approximation — resolved
  as **not planned**, see #81.

## Open questions (for discussion)

1. ~~**Irregular-coordinate interpolation**~~ — landed, see #73 above.
   Higher orders (≥ 2) on an irregular coordinate — *resolved*: **not
   planned**, see #81; the exactness of nearest/linear is structural (both
   the inversion and the pull are locally affine between the same two
   ticks), not a stopgap awaiting a future spline.
2. **Range selection** on `.sel` — `x.sel(t=slice("1s", "5s"))` (a value range
   → a contiguous slice). Natural and useful; not in this slice.
3. **`.sel` directional modes** — *resolved*: `floor`/`ceil` (value space) and
   `prev`/`next` (tick order), with xarray's `ffill`/`bfill` aliased onto
   `prev`/`next`.
4. ~~**`O(1)` compact fast path for `.sel`**~~ — landed, see
   [#110](https://github.com/bagofseeds/fiery-xtensor/issues/110): resolves
   `round`/`floor`/`ceil`/`prev`/`next` directly from `origin`/`spacing`,
   never materialising `["values"]` or searching it *for any realistic
   input*. The endpoints (`k=0`, `k=size-1`) are checked directly against
   the target so an out-of-range value clamps/raises exactly regardless of
   scale; otherwise a division-based seed is corrected by walking to the
   true boundary using the real tick value (`origin + k·spacing`, not
   another division) — exact, not epsilon-tolerant, since the naive
   `round((v − origin) / spacing)` loses precision as a *cancellation*
   error that scales with `|origin/spacing|` (an epoch-seconds axis with
   millisecond spacing, say), which a fixed epsilon guard can't absorb at
   every scale. An astronomically large ratio (beyond the walk's bounded
   step budget) falls back to materialising and searching for that one
   target — still correct, just not O(1) — rather than risk a wrong
   answer; ordinary use never reaches it.
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
