# fiery-xtensor

Named dimensions and coordinate labels for PyTorch tensors.

`fiery.xtensor` is a [`fiery`](https://bagofseeds.github.io/fiery/) match
that makes **names a first-class citizen** of `torch.Tensor`. Its `XTensor`
(also spelled `xtensor`) is an [xarray](https://docs.xarray.dev)-like
`DataArray` over a *live* torch tensor: it carries named **dimensions** and,
optionally, per-dimension coordinate **labels** through operations — so you can
refer to a dimension by name and a position along it by label, without leaving
torch (autograd, device, and `__torch_function__` all keep working).

## Classes

| Class | What it adds |
| ----- | ------------ |
| `XTensor` | Named **dimensions** (`names`) and coordinate **labels** (`coords`, a `{dim name: labels}` mapping), both self-managed (independent of PyTorch's experimental builtin named tensors) so they work across a wide torch range. Names *and* labels propagate through reshaping/reordering (`permute`, `view`/`reshape`, `squeeze`/`unsqueeze`, transpose & `movedim` families, `flatten`/`unflatten`, `expand`, `diagonal`, `T`/`mT`), slicing/splitting (`__getitem__`, `select`, `narrow`, `unbind`, `split`/`chunk`, `flip`/`roll`), reductions (`sum`, `mean`, `amax`, `argmax`, …), and combine ops (`cat`, `stack`, `matmul`/`@`, `einsum`, `tensordot`). Select by label with `.sel`, by position with `.isel`, or reach a single label by attribute. |
| `XVector` / `XMatrix` | Convenience specializations that pre-name and label their channel axes (`"channel"`; `"row"`/`"col"`). |

Labels are keyed by dimension **name**, so — like xarray — they simply follow
their dimension through a `permute`/`transpose`/reduction with no bookkeeping.

```python
import torch
from fiery.xtensor import XTensor

# Named dimensions
x = XTensor(torch.zeros(2, 3, 4), names=("batch", "height", "width"))
x.T.names                       # ('width', 'height', 'batch')
x.unsqueeze(1).names            # ('batch', None, 'height', 'width')

# `...` stands for a run of axes you don't name here (name just the ends)
XTensor(torch.zeros(2, 3, 4, 5), names=("batch", ..., "width")).names
#                               # ('batch', None, None, 'width')

# Refer to a dimension by name (method form)
x.transpose("height", "width").names   # ('batch', 'width', 'height')
x.sum(dim="batch").names               # ('height', 'width')
x.mean(dim="height", keepdim=True).names  # ('batch', 'height', 'width')

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

## Referring to a dimension by name

Anywhere an operation takes a `dim` (or `dim0`/`dim1`, `source`/`destination`,
…), you can pass an axis **name** instead of an integer — **on the method
form**:

```python
x.transpose("height", "width")   # ok
x.sum(dim="height")              # ok
x.movedim("batch", -1)           # ok (names or ints, mixed)
```

Name-as-dim is **not** available on the *functional* form
(`torch.transpose(x, "height", "width")`, `torch.sum(x, dim="height")`), and
this is by design rather than an oversight:

- The **method** form (`x.op(...)`) resolves to a function this package
  installs on the tensor subclass, so a name is turned into an integer in
  Python *before* PyTorch ever sees the arguments.
- The **functional** form (`torch.op(x, ...)`) goes straight into PyTorch's
  C-level argument parser, which validates that `dim` is an integer *before*
  the `__torch_function__` hook that would let us intercept the call runs. On
  recent PyTorch a string `dim` therefore raises `TypeError` from PyTorch
  itself, before this package is consulted. Older PyTorch happened to dispatch
  first, so the behaviour was version-dependent and is not relied upon.

Intercepting the functional form would require monkey-patching the `torch.*`
functions globally, which this package deliberately does not do. The functional
form still works perfectly with an **integer** `dim` — and still carries names
through the result (`torch.sum(x, 1).names == x.sum(dim=1).names`); only the
name-*as*-dim convenience is method-only. Operations that have no method form
at all (`torch.cat`, `torch.stack`) take an integer `dim` only.

## Broadcast by name

When **both** operands of a pointwise op (`+`, `-`, `*`, `/`, comparisons, …)
are fully-named `XTensor`s, their axes are aligned **by name** — the xarray way
— instead of by position:

```python
a = xtensor(torch.arange(6).reshape(2, 3), names=("x", "y"))
b = xtensor(torch.arange(6).reshape(3, 2), names=("y", "x"))  # transposed
(a + b).names            # ('x', 'y')  — b is transposed to match, then added

c = xtensor(torch.arange(2), names=("x",))
d = xtensor(torch.arange(3), names=("y",))
(c + d).shape            # (2, 3)  — disjoint dims broadcast to the outer grid
```

The result's dimensions are the **union** of the operands' names; a shared name
is broadcast together (its sizes must match, or one must be 1) and coordinates
that agree are carried through. If **any** axis is unnamed — or an operand is a
plain tensor or a scalar — the op falls back to ordinary positional
broadcasting.

### Coordinate alignment

When a shared dimension is **labelled on both operands** but the labels are in a
different order — or only partly overlap — the operands are aligned **by label**
before the op, the xarray `join="inner"` way: both are reindexed to the
intersection of their labels (in the left operand's order), so positions are
matched by *label*, not by position.

```python
a = xtensor(torch.tensor([1., 2., 3.]), names=("x",), coords={"x": ("A", "B", "C")})
b = xtensor(torch.tensor([10., 20., 30.]), names=("x",), coords={"x": ("C", "B", "A")})
(a + b).coords            # {'x': ('A', 'B', 'C')}
(a + b).tolist()          # [31.0, 22.0, 13.0]  — A+A, B+B, C+C

c = xtensor(torch.tensor([1., 2., 3.]), names=("x",), coords={"x": ("A", "B", "C")})
d = xtensor(torch.tensor([10., 20., 30.]), names=("x",), coords={"x": ("B", "C", "D")})
(c + d).coords            # {'x': ('B', 'C')}    — inner join to the overlap
```

A dimension labelled on only one side has nothing to align against, so its
labels simply ride along and the op stays positional.

## Axis descriptors

A name can be enriched into an [OME-NGFF](https://ngff.openmicroscopy.org)-style
**descriptor** — a dict with a required `name` plus optional `type`, `unit`, and
`orientation` — by passing it in place of a bare string:

```python
x = xtensor(
    torch.zeros(2, 3, 4),
    names=[
        {"name": "b", "type": "batch"},
        "h",
        {"name": "w", "type": "space", "orientation": "left-to-right"},
    ],
)
x.names          # ('b', 'h', 'w')          — the bare, ergonomic view
x.axes           # full descriptors, one dict per axis
x.flip("w").axes[2]["orientation"]   # 'right-to-left'  — flip reverses it
```

Descriptor fields are keyed by dimension name, so — like coordinates — they
follow their dimension through `permute`, reductions, `rename`, etc. An
`orientation` must read `"{a}-to-{b}"`; flipping the axis rewrites it to
`"{b}-to-{a}"`.

When two operands meet in a name-aware op (broadcast, alignment), their
descriptors are **merged by dim name** the same way coordinates are: the result
is the union of the axes, and for a shared dim the fields the operands **agree**
on are kept while **conflicting** fields are dropped. That policy is
configurable via `set_options(combine_axes=...)`, usable globally or as a
context manager:

```python
from fiery.xtensor import set_options

a = xtensor(torch.ones(3), names=[{"name": "x", "type": "space"}])
b = xtensor(torch.ones(3), names=[{"name": "x", "type": "time"}])
(a + b).axes                      # ({'name': 'x'},)  — the clash drops 'type'

with set_options(combine_axes="strict"):
    a + b                         # raises ValueError: conflicting 'type' …
```

The policy is one of `"drop_conflicts"` (default), `"strict"` (alias
`"raise"`), `"override"` (keep the left operand's value) or `"drop"` (always
drop the field). Pass a `{field: policy}` dict to set it **per descriptor
field** — `"*"` is the default for fields you don't name:

```python
# drop everything by default, but a clashing unit is an error
with set_options(combine_axes={"*": "drop", "unit": "raise"}):
    ...
```

`combine_axes` accepts `"drop_conflicts"` (default), `"strict"` (raise on any
clash), `"override"` (keep the left operand's fields), or `"drop"` (discard all
descriptors).

A descriptor is also a way to **address** axes. Anywhere you can pass a `dim`
(method form), a query dict selects *every* axis whose descriptor matches — so
one call can act on a whole group of axes at once:

```python
x.movedim({"type": "space"}, -1)   # all space axes to the back, order kept
x.sum(dim={"type": "channel"})     # reduce every channel axis
```

A query that matches a single axis behaves exactly like naming it (so it still
works with single-`dim`-only ops such as `prod`).

## Structured coordinates

A coordinate **label** can itself be a descriptor dict, so each *position* along
an axis is described — the position-level analogue of an axis descriptor. A
label's `"name"` is still its identity for selection; the other fields are
queryable:

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

## Data units

A tensor's **values** can carry a physical unit — the `.unit` property (a
constructor `unit=` kwarg or a settable attribute). It rides through operations
like names and coordinates do:

```python
v = xtensor(data, names=("b", "t"), unit="V")
v.unit                 # "V"
v.T.unit               # "V"  — carried through reshaping/reduction/indexing
v.unit = "mV"          # annotate: never changes the data
```

By default (`unit_backend=None`) a unit is an **opaque string** — stored and
carried, never inspected. Selecting a backend turns on validation, conversion,
and **dimensional algebra**:

```python
from fiery.xtensor import set_options
with set_options(unit_backend="pint"):     # needs fiery-xtensor[units]
    volts = xtensor(v, unit="V")
    amps  = xtensor(a, unit="A")
    volts.to_unit("mV")            # converts: rescales the data ×1000
    (volts * amps).unit           # "ampere * volt"  — units multiply
    (volts / secs).unit           # "volt / second"
    (volts ** 2).unit             # "volt ** 2"
    (volts @ amps).unit           # "ampere * volt"  — matmul multiplies too
```

Whenever a step is dimensionally invalid or ambiguous — adding incompatible
units, or a transcendental like `exp`/`log` of a united value — the result
silently **drops** the unit; `set_options(unit_policy="strict")` makes those
same steps **raise** instead:

```python
(volts + amps).unit               # None    — incompatible, dropped
torch.exp(volts).unit             # None    — exp needs a dimensionless argument
with set_options(unit_policy="strict"):
    volts + amps                  # ValueError: incompatible units 'volt' and 'ampere'
```

The data itself is never wrapped in a `pint.Quantity`, so autograd, GPU, and
`__torch_function__` keep working throughout.

## Design goals

- **Names are first class.** Every operation that can use, manipulate, or
  preserve names should do so. Coverage is tracked in the
  [name-related method survey](../../issues) (one sub-issue per function).
- **Wide Python support** (3.7+): the runtime uses only old-compatible syntax
  plus `typing_extensions`; modern typing lives in lazy annotations via
  `from __future__ import annotations`.
- **Wide PyTorch support.** Function overrides are registered only for ops that
  exist in the running PyTorch version, so the package loads across a broad
  torch range.

## Installation

```sh
pip install fiery-xtensor
```

## Status

Alpha — ported from a work-in-progress in
[`balbasty/magnetix`](https://github.com/balbasty/magnetix). Names and
coordinates are **self-managed**, independent of PyTorch's experimental builtin
named tensors (which have been dropped in some torch builds), so the package
spans a wide torch range. See the tracking issues for the roadmap
(axis `unit` propagation, coordinate alignment, …).
