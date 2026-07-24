# fiery-namedtensors

Named dimensions and named indices for PyTorch tensors.

`fiery.namedtensors` is a [`fiery`](https://bagofseeds.github.io/fiery/) match
that makes **names a first-class citizen** of `torch.Tensor`. It provides thin
`torch.Tensor` subclasses that carry naming metadata through operations, so you
can refer to *dimensions* and *individual positions* by name rather than by
integer index.

## Classes

| Class | What it adds |
| ----- | ------------ |
| `NamedTensor` | Named **axes**, self-managed (independent of PyTorch's experimental builtin named tensors) so it works across a wide torch range. Names propagate through reshaping/reordering (`permute`, `view`/`reshape`, `squeeze`/`unsqueeze`, transpose & `movedim` families, `flatten`/`unflatten`, `expand`, `diagonal`, `T`/`mT`), slicing/splitting (`__getitem__`, `select`, `narrow`, `unbind`, `split`/`chunk`, `flip`/`roll`), reductions (`sum`, `mean`, `amax`, `argmax`, …), and combine ops (`cat`, `stack`, `matmul`/`@`). |
| `TensorWithNamedIndices` | Named **indices**: individual positions along an axis can be addressed by name (e.g. `x.c1`), and the naming metadata is tracked through slicing. |
| `NamedVector` / `NamedMatrix` | Convenience specializations for 1-D and 2-D named-index axes (channels). |

```python
import torch
from fiery.namedtensors import NamedTensor, TensorWithNamedIndices

# Named axes
x = NamedTensor(torch.zeros(2, 3, 4), names=("batch", "height", "width"))
x.T.names                       # ('width', 'height', 'batch')
x.unsqueeze(1).names            # ('batch', None, 'height', 'width')

# Refer to a dimension by name (method form)
x.transpose("height", "width").names   # ('batch', 'width', 'height')
x.sum(dim="batch").names               # ('height', 'width')
x.mean(dim="height", keepdim=True).names  # ('batch', 'height', 'width')

# Named indices: address positions along an axis by name
m = TensorWithNamedIndices(
    torch.arange(6).reshape(2, 3),
    index_names=(("x", "y", "z"),),
    index_dims=(1,),
)
m.y                             # selects position 1 along dim 1
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
pip install fiery-namedtensors
```

## Status

Alpha — ported from a work-in-progress in
[`balbasty/magnetix`](https://github.com/balbasty/magnetix). Axis names are
**self-managed**, independent of PyTorch's experimental builtin named tensors
(which have been dropped in some torch builds), so the package spans a wide
torch range. See the tracking issues for the roadmap (per-op name coverage,
reductions, gather/scatter, …).
