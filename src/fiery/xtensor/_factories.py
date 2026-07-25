"""Named factory helpers.

Thin wrappers around the `torch.*` construction functions that return a
[`XTensor`][fiery.xtensor.XTensor] directly, so callers can name
the axes at creation time instead of wrapping every call by hand::

    named_zeros(2, 3, names=("row", "col"))

Each wrapper forwards all positional and keyword arguments to the matching
`torch.*` function and accepts an extra ``names=`` keyword.
"""

from __future__ import annotations

import torch
import typing_extensions as tx

from fiery.xtensor._tensors import XTensor


def _make_factory(name: str) -> tx.Optional[tx.Callable]:
    """Build a `named_<name>` wrapper around `torch.<name>` (or `None`)."""
    base = getattr(torch, name, None)
    if base is None:  # pragma: no cover - all wrapped ops are very old
        return None

    def factory(
        *args: tx.Any,
        names: tx.Optional[tx.Sequence[str | None]] = None,
        **kwargs: tx.Any,
    ) -> XTensor:
        return XTensor(base(*args, **kwargs), names=names)

    factory.__name__ = "named_" + name
    factory.__qualname__ = "named_" + name
    factory.__doc__ = (
        f"Like `torch.{name}`, but returns a `XTensor`.\n\n"
        f"All arguments are forwarded to `torch.{name}`; pass `names=(...)`\n"
        f"to name the axes of the result."
    )
    return factory


named_zeros = _make_factory("zeros")
named_ones = _make_factory("ones")
named_empty = _make_factory("empty")
named_full = _make_factory("full")
named_arange = _make_factory("arange")
named_rand = _make_factory("rand")
named_randn = _make_factory("randn")
named_eye = _make_factory("eye")


def xvector(
    data: tx.Any,
    *,
    channels: tx.Any = (...,),
    channel_dim: int = -1,
    **kwargs: tx.Any,
) -> XTensor:
    """
    Wrap `data` as an [`XTensor`][fiery.xtensor.XTensor] with one labelled
    **channel** axis.

    A one-liner over `XTensor(...)`: names axis `channel_dim` (the last by
    default) `"channel"` and labels it with `channels` (a `...` in the labels
    fills the rest with unlabelled positions). Any other `XTensor` keyword
    (`names=`, `coords=`, `unit=`, ...) is forwarded.

    The result is a **plain** `XTensor`, not a distinct type -- so a reduction
    or selection that drops the channel axis just returns a normal `XTensor`,
    and the value is never a "vector" that has lost its vector axis.
    """
    x = XTensor(data, **kwargs)
    names = list(x.names)
    names[channel_dim % x.ndim] = "channel"
    x.names = tuple(names)
    x.coords = dict(x.coords, channel=channels)
    return x


def xmatrix(
    data: tx.Any,
    *,
    rows: tx.Any = (...,),
    cols: tx.Any = (...,),
    dims: tx.Tuple[int, int] = (-2, -1),
    **kwargs: tx.Any,
) -> XTensor:
    """
    Wrap `data` as an [`XTensor`][fiery.xtensor.XTensor] with labelled `"row"`
    and `"col"` axes.

    The matrix analogue of [`xvector`][fiery.xtensor.xvector]: names the two
    axes in `dims` (the last two by default) `"row"` and `"col"` and labels
    them with `rows` / `cols` (a `...` fills the rest unlabelled). Other
    `XTensor` keywords are forwarded, and the result is a plain `XTensor`.
    """
    x = XTensor(data, **kwargs)
    names = list(x.names)
    d0, d1 = (d % x.ndim for d in dims)
    names[d0], names[d1] = "row", "col"
    x.names = tuple(names)
    x.coords = dict(x.coords, row=rows, col=cols)
    return x


__all__ = [
    "named_zeros",
    "named_ones",
    "named_empty",
    "named_full",
    "named_arange",
    "named_rand",
    "named_randn",
    "named_eye",
    "xvector",
    "xmatrix",
]
