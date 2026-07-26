---
icon: material/fire
---

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
around an array. For what it adds, what it is still missing, and where behaviour
or vocabulary differ, see [Differences from xarray](guide/vs-xarray.md).

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

## Guide

- [Names & dimensions](guide/names.md) — named dimensions, `...` in name-tuples,
  and referring to a dimension by name (method vs. functional form).
- [Coordinates](guide/coordinates.md) — coordinate labels and positional labels,
  structured coordinates, and numeric coordinates.
- [Broadcasting & alignment](guide/broadcasting.md) — broadcasting by name and
  coordinate alignment.
- [Axis descriptors](guide/descriptors.md) — OME-NGFF-style descriptors that
  enrich a name with a type, unit, and orientation.
- [Data units](guide/data-units.md) — physical units on a tensor's values, with
  dimensional algebra and per-axis units.
- [Differences from xarray](guide/vs-xarray.md) — what `xtensor` adds, what it
  is still missing, and where behaviour or vocabulary differ.

## Proposals

See the [proposals](proposals/index.md) for the larger design decisions:

- [0001 — Coordinates (values, spacing, units)](proposals/0001-units.md)
- [0002 — Structured coordinates](proposals/0002-structured-coordinates.md)
- [0003 — Data units](proposals/0003-data-units.md)

## API

See the [API reference](api/index.md) for the full `fiery.xtensor` surface.
