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

> Referring to a dimension by name works on the **method** form
> (`x.sum(dim="batch")`), not the functional form (`torch.sum(x, dim="batch")`):
> recent PyTorch validates a `dim` argument before the naming layer sees it.

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
