"""Named dimensions and named indices for PyTorch tensors.

`fiery.namedtensors` provides `torch.Tensor` subclasses that make names a
first-class citizen:

- [`NamedTensor`][fiery.namedtensors.NamedTensor] is an
  [xarray](https://docs.xarray.dev)-like `DataArray` over a live
  `torch.Tensor`: it carries **self-managed** named dimensions and, optionally,
  per-dimension coordinate **labels** (`coords`, keyed by dimension name)
  through a wide range of operations. Select by label with `.sel`, by position
  with `.isel`, or reach a single label by attribute (`x.red`).
- [`NamedVector`][fiery.namedtensors.NamedVector] and
  [`NamedMatrix`][fiery.namedtensors.NamedMatrix] are convenience
  specializations that pre-name and label their channel axes.

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
)

try:
    from fiery.namedtensors._version import __version__
except ImportError:  # pragma: no cover - only during editable/source use
    __version__ = "0+unknown"

__all__ = [
    "ExtendedTensor",
    "NamedTensor",
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
