"""Named dimensions and coordinate labels for PyTorch tensors.

`fiery.xtensor` makes names a first-class citizen of `torch.Tensor`, in the
spirit of [xarray](https://docs.xarray.dev):

- [`XTensor`][fiery.xtensor.XTensor] (also available lowercase as `xtensor`) is
  an xarray-like `DataArray` over a live `torch.Tensor`: it carries
  **self-managed** named dimensions and, optionally, per-dimension coordinate
  **labels** (`coords`, keyed by dimension name) through a wide range of
  operations. Select by label with `.sel`, by position with `.isel`, or reach a
  single label by attribute (`x.red`).
- [`xvector`][fiery.xtensor.xvector] and
  [`xmatrix`][fiery.xtensor.xmatrix] are convenience factories that name and
  label a `"channel"` axis (or `"row"`/`"col"`) and return a plain `XTensor`.

The `x*` factories (`xzeros`, `xones`, `xfill`, ...) build an `XTensor`
directly from the matching `torch.*` factory, naming (and labelling) the axes
at creation time; the `x*_like` variants inherit an `XTensor` input's names,
coordinates, and unit.

[`as_xtensor`][fiery.xtensor.as_xtensor] is the `XTensor` analogue of
`torch.as_tensor`: coerce a bare number, a plain `Tensor`, or an `XTensor`
into an `XTensor`, graph-safely, preserving `unit`/`names`/`coords` unless a
keyword explicitly overrides them. Like `torch.as_tensor`, it also accepts
`dtype=`/`device=` to convert the underlying data.

[`is_xtensor`][fiery.xtensor.is_xtensor] is the `XTensor` analogue of
`torch.is_tensor`.
"""

# Each of these registers its overrides onto XTensor as an import side
# effect (see CLAUDE.md, "How the subclassing works"); imported here so
# that importing fiery.xtensor is enough to populate the full registry.
from fiery.xtensor import (
    _combine,  # noqa: F401
    _gather,  # noqa: F401
    _pointwise,  # noqa: F401
    _reduce,  # noqa: F401
    _shape,  # noqa: F401
    _slice,  # noqa: F401
)
from fiery.xtensor._extended import ExtendedTensor
from fiery.xtensor._factories import (
    xarange,
    xempty,
    xempty_like,
    xeye,
    xfill,
    xfull,
    xfull_like,
    xlinspace,
    xlogspace,
    xmatrix,
    xmeshgrid,
    xones,
    xones_like,
    xrand,
    xrand_like,
    xrandn,
    xrandn_like,
    xstack,
    xvector,
    xzeros,
    xzeros_like,
)
from fiery.xtensor._options import set_options
from fiery.xtensor._tensors import XTensor, as_xtensor, is_xtensor

#: Lowercase alias of [`XTensor`][fiery.xtensor.XTensor].
xtensor = XTensor

try:
    from fiery.xtensor._version import __version__
except ImportError:  # pragma: no cover - only during editable/source use
    __version__ = "0+unknown"

__all__ = [
    "ExtendedTensor",
    "XTensor",
    "as_xtensor",
    "is_xtensor",
    "xtensor",
    "xvector",
    "xmatrix",
    "xstack",
    "xmeshgrid",
    "xzeros",
    "xones",
    "xempty",
    "xfull",
    "xfill",
    "xarange",
    "xlinspace",
    "xlogspace",
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
