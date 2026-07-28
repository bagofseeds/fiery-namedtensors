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
m.sel(channel="y")              # selects position 1 along "channel"
m.y                             # ... same, by attribute
m[:, "y"]                       # ... same, positional label on the 2nd axis
m.isel(channel=1)               # ... same, by integer position
m.coords                        # {'channel': ('x', 'y', 'z')}
m.T.coords                      # {'channel': ('x', 'y', 'z')} — follows the dim
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
img[{"type": "signal"}]     # by query — every matching position, keeps the axis
img.sel(c={"type": "signal"})   # ... the sel-keyword spelling
```

A **query** (a dict where a coordinate label is expected) selects *positions* —
the matches become a `slice` when contiguous, else an index list — and always
keeps the axis. It mirrors the descriptor query for *axes*: a `{"type": ...}`
dict picks **axes** in a `dim=`/`movedim` slot and **positions** in a `[]`/`sel`
slot, so the two never collide.

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

Numeric coordinates slice **affinely** (`img[..., 2:]` shifts the origin,
`img[..., ::2]` scales the spacing) and convert with `img.coords["x"].to("um")`.
The position unit is separate from the *data* unit of the tensor's values (see
[Data units](data-units.md)).
