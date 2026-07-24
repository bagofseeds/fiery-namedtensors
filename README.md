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
| `NamedTensor` | Named **axes**. Extends PyTorch's builtin named-tensor feature with operations the builtin does not propagate (`permute`, `view`, `squeeze`, `unsqueeze`, `T`, fancy `__getitem__`). |
| `TensorWithNamedIndices` | Named **indices**: individual positions along an axis can be addressed by name (e.g. `x.c1`), and the naming metadata is tracked through slicing. |
| `NamedVector` / `NamedMatrix` | Convenience specializations for 1-D and 2-D named-index axes (channels). |

```python
import torch
from fiery.namedtensors import NamedTensor, TensorWithNamedIndices

# Named axes
x = NamedTensor(torch.zeros(2, 3, 4), names=("batch", "height", "width"))
x.T.names                       # ('width', 'height', 'batch')
x.unsqueeze(1).names            # ('batch', None, 'height', 'width')

# Named indices: address positions along an axis by name
m = TensorWithNamedIndices(
    torch.arange(6).reshape(2, 3),
    index_names=(("x", "y", "z"),),
    index_dims=(1,),
)
m.y                             # selects position 1 along dim 1
```

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
[`balbasty/magnetix`](https://github.com/balbasty/magnetix). See the tracking
issues for the roadmap, including the planned move to **self-managed names**
(independent of PyTorch's experimental builtin named tensors, which have been
dropped in some future torch builds).
