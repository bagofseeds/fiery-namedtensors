"""Named dimensions and coordinate labels for PyTorch tensors.

`fiery.xtensor` makes names a first-class citizen of `torch.Tensor`, in the
spirit of [xarray](https://docs.xarray.dev):

- [`XTensor`][fiery.xtensor.XTensor] (also available lowercase as
  [`xtensor`][fiery.xtensor.xtensor]) is an xarray-like `DataArray` over a live
  `torch.Tensor`: it carries **self-managed** named dimensions and, optionally,
  per-dimension coordinate **labels** (`coords`, keyed by dimension name)
  through a wide range of operations. Select by label with `.sel`, by position
  with `.isel`, or reach a single label by attribute (`x.red`).
- [`xvector`][fiery.xtensor.xvector] and
  [`xmatrix`][fiery.xtensor.xmatrix] are convenience factories that name and
  label a `"channel"` axis (or `"row"`/`"col"`) and return a plain `XTensor`.

The `x*` factories ([`xzeros`][fiery.xtensor.xzeros],
[`xones`][fiery.xtensor.xones], [`xfill`][fiery.xtensor.xfill], ...) build an
`XTensor` directly from the matching `torch.*` factory, naming (and labelling)
the axes at creation time; the `x*_like` variants inherit an `XTensor` input's
names, coordinates, and unit.
"""

from fiery.xtensor._factories import (
    xarange,
    xempty,
    xempty_like,
    xeye,
    xfill,
    xfull,
    xfull_like,
    xmatrix,
    xones,
    xones_like,
    xrand,
    xrand_like,
    xrandn,
    xrandn_like,
    xvector,
    xzeros,
    xzeros_like,
)
from fiery.xtensor._options import set_options
from fiery.xtensor._tensors import (
    ExtendedTensor,
    XTensor,
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
    "xvector",
    "xmatrix",
    "xzeros",
    "xones",
    "xempty",
    "xfull",
    "xfill",
    "xarange",
    "xrand",
    "xrandn",
    "xeye",
    "xzeros_like",
    "xones_like",
    "xempty_like",
    "xfull_like",
    "xrand_like",
    "xrandn_like",
    "set_options",
    "__version__",
]
