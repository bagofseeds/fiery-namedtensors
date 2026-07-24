"""Named dimensions and named indices for PyTorch tensors.

`fiery.namedtensors` provides `torch.Tensor` subclasses that make names a
first-class citizen:

- [`NamedTensor`][fiery.namedtensors.NamedTensor] extends PyTorch's builtin
  named-tensor feature with operations that the builtin implementation does
  not support (`permute`, `view`, `squeeze`, `unsqueeze`, ...).
- [`TensorWithNamedIndices`][fiery.namedtensors.TensorWithNamedIndices]
  allows individual positions along an axis to be indexed by name.
- [`NamedVector`][fiery.namedtensors.NamedVector] and
  [`NamedMatrix`][fiery.namedtensors.NamedMatrix] are convenience
  specializations for 1D and 2D named-index axes.
"""

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
    "__version__",
]
