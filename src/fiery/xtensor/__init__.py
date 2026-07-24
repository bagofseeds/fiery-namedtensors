"""Named dimensions and coordinate labels for PyTorch tensors.

`fiery.xtensor` makes names a first-class citizen of `torch.Tensor`, in the
spirit of [xarray](https://docs.xarray.dev):

- [`XTensor`][fiery.xtensor.XTensor] (also available lowercase as
  [`xtensor`][fiery.xtensor.xtensor]) is an xarray-like `DataArray` over a live
  `torch.Tensor`: it carries **self-managed** named dimensions and, optionally,
  per-dimension coordinate **labels** (`coords`, keyed by dimension name)
  through a wide range of operations. Select by label with `.sel`, by position
  with `.isel`, or reach a single label by attribute (`x.red`).
- [`XVector`][fiery.xtensor.XVector] and
  [`XMatrix`][fiery.xtensor.XMatrix] are convenience
  specializations that pre-name and label their channel axes.

The `named_*` helpers ([`named_zeros`][fiery.xtensor.named_zeros], ...)
build an `XTensor` directly from the matching `torch.*` factory.
"""

from fiery.xtensor._factories import (
    named_arange,
    named_empty,
    named_eye,
    named_full,
    named_ones,
    named_rand,
    named_randn,
    named_zeros,
)
from fiery.xtensor._tensors import (
    ExtendedTensor,
    XMatrix,
    XTensor,
    XVector,
)

#: Lowercase alias of [`XTensor`][fiery.xtensor.XTensor].
xtensor = XTensor

try:
    from fiery.xtensor._version import __version__
except ImportError:  # pragma: no cover - only during editable/source use
    __version__ = "0+unknown"

__all__ = [
    "ExtendedTensor",
    "XTensor",
    "xtensor",
    "XVector",
    "XMatrix",
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
