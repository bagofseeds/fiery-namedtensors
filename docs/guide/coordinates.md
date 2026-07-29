# Coordinates

Coordinate labels let you address a position along a dimension by a meaningful
name instead of an integer, and — like names — they follow their dimension
through permutes, transposes, and reductions. Coordinates can also be
**structured** (a descriptor per position) or **numeric** (a physical position
per element).

## Coordinate labels

Labels are keyed by dimension **name**, so — like xarray — they simply follow
their dimension through a `permute`/`transpose`/reduction with no bookkeeping:

```python
import torch
from fiery.xtensor import XTensor

# Coordinate labels: address positions along a dimension by label
m = XTensor(
    torch.arange(6).reshape(2, 3),
    names=("row", "channel"),
    coords={"channel": ("x", "y", "z")},
)
m.coords                        # {'channel': ('x', 'y', 'z')}
m.T.coords                      # {'channel': ('x', 'y', 'z')} — follows the dim
```

Four equivalent ways to select the position labelled `"y"` along `channel`:

=== "By label"

    ```python
    m.sel(channel="y")
    ```

=== "By attribute"

    ```python
    m.y
    ```

=== "Positional label"

    ```python
    m[:, "y"]   # label on the 2nd axis
    ```

=== "By integer position"

    ```python
    m.isel(channel=1)
    ```

A **positional** coordinate label works anywhere an integer index does, resolved
against the axis it sits on — so it composes with ints, slices, `...` and
newaxis:

```python
x[..., "r1", "y"]   # label the last two axes; a bare label drops its axis
x[:, ["w", "y"]]    # a list of labels is an advanced index (keeps the axis)
```

## Structured coordinates

A coordinate **label** can itself be a descriptor dict, so each *position* along
an axis is described — the position-level analogue of an
[axis descriptor](descriptors.md). A label's `"name"` is still its identity for
selection; the other fields are queryable:

```python
img = xtensor(data, names=("c", "y", "x"), coords={"c": [
    {"name": "DAPI", "type": "nucleus"},
    {"name": "GFP",  "type": "signal"},
    {"name": "RFP",  "type": "signal"},
]})

img.sel(c="GFP")            # by name — drops the axis (as before)
```

A **query** (a dict where a coordinate label is expected) selects every
*matching position* and keeps the axis — two equivalent spellings:

=== "By `[]`"

    ```python
    img[{"type": "signal"}]
    ```

=== "By `.sel`"

    ```python
    img.sel(c={"type": "signal"})
    ```

The matches become a `slice` when they are contiguous and an index list when
they are not; either way the axis is kept. This mirrors the descriptor query
for *axes*: a `{"type": ...}` dict picks **axes** in a `dim=`/`movedim` slot and
**positions** in a `[]`/`sel` slot, so the two never collide.

## Numeric coordinates

A coordinate can also be **numeric** — a physical position per element — given
compactly as `spacing` and/or `origin`, each a value with a unit
([Proposal 0001](../proposals/0001-units.md)):

```python
img = xtensor(data, names=("y", "x"),
              coords={"x": {"spacing": (0.5, "mm"), "origin": (-16, "mm")}})

img.coords["x"]["spacing"].unit   # "mm"  — the position unit
img.coords["x"]["values"]         # tensor([-16, -15.5, …]) with .unit == "mm"
```

Only one of `spacing`/`origin` needs to be given — the other defaults, and the
default always takes the unit of the one you did give:

```python
xtensor(data, names=("x",), coords={"x": {"spacing": (0.5, "mm")}})
# origin defaults to 0 mm  -> positions 0, 0.5, 1, …

xtensor(data, names=("x",), coords={"x": {"origin": (-16, "mm")}})
# spacing defaults to 1 mm -> positions -16, -15, -14, …
```

`["values"]` materializes `origin + i·spacing` on demand (differentiable — a
learnable `spacing` tensor keeps its gradient). An **irregular** axis instead
takes an explicit unitful tensor of positions:

```python
sig = xtensor(trace, names=("t",),
              coords={"t": xtensor([0., 0.5, 2., 4.], unit="s")})
```

A bare tuple/list of plain numbers is sugar for the same thing (dimensionless,
unless you use one of the forms above) — it is a numeric coordinate, not a set
of labels that happen to be numbers:

```python
xtensor(trace, names=("t",), coords={"t": (0., 0.5, 2., 4.)})
```

Numeric coordinates slice **affinely** (`img[..., 2:]` shifts the origin,
`img[..., ::2]` scales the spacing) and convert with `img.coords["x"].to("um")`.
The position unit is separate from the *data* unit of the tensor's values (see
[Data units](data-units.md)).

A `slice` selector on `.sel` picks a **value range** instead of a single tick
— unit-aware, and half-open like ordinary Python slicing (`lo <= value <
hi`, **not** xarray's inclusive-both-ends convention):

```python
sig.sel(t=slice("1s", "5s"))   # every tick with 1s <= value < 5s
sig.sel(t=slice(None, "2s"))   # value < 2s
```

Bounds are compared numerically regardless of the order they're given in or
of the coordinate's own direction — `slice(lo, hi)` and `slice(hi, lo)`
select the same range. An out-of-range or empty result is a well-formed
empty axis, not an error.

### Selecting near a value

A bare `.sel(t=v)` (no `mode`) is **exact** — it raises if `v` isn't an
existing tick. Pass `mode` to snap to a nearby one instead:

=== "Nearest (round)"

    ```python
    sig.sel(t=1.2, mode="round")   # the nearest tick by value
    ```

=== "Floor / ceil"

    ```python
    sig.sel(t=1.2, mode="floor")   # largest tick <= 1.2
    sig.sel(t=1.2, mode="ceil")    # smallest tick >= 1.2
    ```

=== "Prev / next"

    ```python
    sig.sel(t=1.2, mode="prev")   # neighbouring tick at the lower index
    sig.sel(t=1.2, mode="next")   # neighbouring tick at the higher index
    ```

A `mode` alone snaps with no limit on the distance; add `tolerance` to cap
how far it's allowed to go:

```python
sig.sel(t=1.2, mode="round", tolerance=0.1)   # raises: nearest tick is farther away than 0.1
```

## Interpolating between ticks

Where `.sel` picks an existing position, `.interp` computes a value at an
arbitrary one:

```python
sig.interp(t=1.25)               # one point -> drops the axis
sig.interp(t=[1.0, 1.25, 1.5])   # several   -> keeps the axis, the queried
                                  # values becoming its new coordinate
```

`method` picks the interpolation order: `"nearest"` works out of the box;
anything higher (`"linear"` *(default)*, `"quadratic"`, `"cubic"`, or a plain
integer order) needs the optional interpolation extra,
`pip install fiery-xtensor[interp]`. A **regular** coordinate (`spacing`/
`origin`) supports every order; an **irregular** one only `"nearest"`/
`"linear"`.

An out-of-range query is governed by `bound` (default `"replicate"`, which
clamps to the edge value) and `extrapolate`, settable per call or as a
standing default with [`set_options`][fiery.xtensor.set_options]:

```python
sig.interp(t=10.0, bound="replicate")   # clamps to the last tick

with set_options(interp_bound="replicate"):
    sig.interp(t=10.0)   # every call in this block clamps too
```

## Multiple coordinates per axis

An axis can carry more than one coordinate at once — its own index, plus any
number of **non-dimension coordinates** riding alongside it. Key a coordinate
by any name that isn't a dim, paired with the dim(s) it rides:

```python
sig = xtensor(
    trace, names=("t",),
    coords={
        "t": {"spacing": 0.5, "origin": 0.0},                  # the index
        "season": ("t", ("w", "w", "sp", "sp", "su", "su")),   # rides along "t"
    },
)
sig.sel(t=1.0)        # selects by the index
sig.sel(season="su")  # ValueError: not an index coordinate
```

`.sel`/`.isel` only ever resolve against a dim's own index — a non-dimension
coordinate isn't selectable directly. Promote one to be the index with
`swap_dims`, which renames the axis to the promoted coordinate's name and
keeps the old index around, now riding alongside under its own key:

```python
sig.swap_dims({"t": "season"}).sel(season="su")
```

`swap_dims_` is the in-place variant.
