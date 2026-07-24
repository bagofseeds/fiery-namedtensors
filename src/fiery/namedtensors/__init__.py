"""Named dimensions and named indices for PyTorch tensors.

`fiery.namedtensors` provides `torch.Tensor` subclasses that make names a
first-class citizen:

- [`NamedTensor`][fiery.namedtensors.NamedTensor] carries **self-managed** axis
  names (independent of PyTorch's experimental builtin named-tensor feature)
  through a wide range of operations (`permute`, `view`, `squeeze`, reductions,
  `matmul`, ...).
- [`TensorWithNamedIndices`][fiery.namedtensors.TensorWithNamedIndices]
  allows individual positions along an axis to be indexed by name.
- [`NamedVector`][fiery.namedtensors.NamedVector] and
  [`NamedMatrix`][fiery.namedtensors.NamedMatrix] are convenience
  specializations for 1D and 2D named-index axes.

The `named_*` helpers ([`named_zeros`][fiery.namedtensors.named_zeros], ...)
build a `NamedTensor` directly from the matching `torch.*` factory.
"""

from fiery.namedtensors._factories import (
    named_arange,
    named_empty,
    named_eye,
    named_full,
    named_ones,
    named_rand,
    named_randn,
    named_zeros,
)
from fiery.namedtensors._tensors import (
    ExtendedTensor,
    NamedMatrix,
    NamedTensor,
    NamedVector,
    TensorWithNamedIndices,
)

try:
    from fiery.namedtensors._version import __version__
except ImportError:  # pragma: no cover - only during editable/source use
    __version__ = "0+unknown"

__all__ = [
    "ExtendedTensor",
    "NamedTensor",
    "TensorWithNamedIndices",
    "NamedVector",
    "NamedMatrix",
    "named_zeros",
    "named_ones",
    "named_empty",
    "named_full",
    "named_arange",
    "named_rand",
    "named_randn",
    "named_eye",
    "__version__",
]
