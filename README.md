# fiery-xtensor

Named dimensions and coordinate labels for PyTorch tensors.

`fiery.xtensor` is a [`fiery`](https://bagofseeds.github.io/fiery/) match
that makes **names a first-class citizen** of `torch.Tensor`. Its `XTensor`
(also spelled `xtensor`) is an [xarray](https://docs.xarray.dev)-like
`DataArray` over a *live* torch tensor: it carries named **dimensions** and,
optionally, per-dimension coordinate **labels** through operations — so you can
refer to a dimension by name and a position along it by label, without leaving
torch (autograd, device, and `__torch_function__` all keep working).

**This is an xarray-like package**, and it deliberately follows xarray's
conventions — named dimensions, `coords` keyed by dimension name, `.sel` /
`.isel`, alignment by name — so xarray users feel at home. The headline
difference is that an `XTensor` *is* a `torch.Tensor` rather than a wrapper
around an array. For what it adds, what it is still missing, and where
behaviour or vocabulary differ, see [Differences from
xarray](https://github.com/bagofseeds/fiery-xtensor/blob/main/docs/guide/vs-xarray.md).

## Quickstart

`XTensor` (lowercase alias `xtensor`) adds named **dimensions** (`names`) and
coordinate **labels** (`coords`, a `{dim name: labels}` mapping), both
self-managed (independent of PyTorch's experimental builtin named tensors) so
they work across a wide torch range. Names *and* labels propagate through
reshaping/reordering (`permute`, `view`/`reshape`, `squeeze`/`unsqueeze`,
transpose & `movedim` families, `flatten`/`unflatten`, `expand`, `diagonal`,
`T`/`mT`), slicing/splitting (`__getitem__`, `select`, `narrow`, `unbind`,
`split`/`chunk`, `flip`/`roll`), reductions (`sum`, `mean`, `amax`, `argmax`,
…), and combine ops (`cat`, `stack`, `matmul`/`@`, `einsum`, `tensordot`).
Select by label with `.sel`, by position with `.isel`, or reach a single label
by attribute.

For the common cases, `xvector` and `xmatrix` are one-line factories that name
and label a `"channel"` axis (or `"row"`/`"col"`) and return a plain `XTensor`:

```python
import torch
from fiery.xtensor import xvector, xmatrix

v = xvector(torch.zeros(2, 3), channels=("x", "y", "z"))   # last axis -> "channel"
m = xmatrix(torch.zeros(2, 3), rows=("r0", "r1"), cols=("c0", "c1", "c2"))
```

They are just `XTensor(..., names=..., coords=...)` spelled shorter — the result
is an ordinary `XTensor`, so a reduction or selection that drops the labelled
axis simply yields a normal `XTensor` (no "vector" that has lost its axis).

Labels are keyed by dimension **name**, so — like xarray — they simply follow
their dimension through a `permute`/`transpose`/reduction with no bookkeeping.

```python
import torch
from fiery.xtensor import XTensor

# Named dimensions
x = XTensor(torch.zeros(2, 3, 4), names=("batch", "height", "width"))
x.T.names                       # ('width', 'height', 'batch')

# Coordinate labels: address positions along a dimension by label
m = XTensor(
    torch.arange(6).reshape(2, 3),
    names=("row", "channel"),
    coords={"channel": ("x", "y", "z")},
)
m.sel(channel="y")              # selects position 1 along "channel"
m.y                             # ... same, by attribute
m.isel(channel=1)               # ... same, by integer position
```

## Documentation

Full docs — with examples for each topic — live at
<https://bagofseeds.github.io/fiery-xtensor/>:

- **Names & dimensions** — named dimensions, `...` in name-tuples, and referring
  to a dimension by name (method vs. functional form).
- **Coordinates** — coordinate labels and positional labels, structured
  coordinates, and numeric coordinates.
- **Broadcasting & alignment** — broadcasting by name and coordinate alignment.
- **Axis descriptors** — OME-NGFF-style descriptors that enrich a name with
  free-form fields (a `type`, an `orientation`, or any custom key).
- **Data units** — physical units on a tensor's values, with dimensional algebra
  and per-axis units.
- **Differences from xarray** — what `xtensor` adds, what it is still missing,
  and where behaviour or vocabulary differ.

## Design goals

- **Names are first class.** Every operation that can use, manipulate, or
  preserve names should do so. Coverage is tracked in the [name-related
  method survey](https://github.com/bagofseeds/fiery-xtensor/issues) (one
  sub-issue per function).
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
spans a wide torch range. See the [tracking
issues](https://github.com/bagofseeds/fiery-xtensor/issues) for the roadmap.
