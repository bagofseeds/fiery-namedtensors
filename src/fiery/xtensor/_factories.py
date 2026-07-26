"""Named factory helpers.

`x*` wrappers around the `torch.*` construction functions that return an
[`XTensor`][fiery.xtensor.XTensor] directly, so you can name (and describe,
label, unit) the axes at creation time instead of wrapping every call by
hand::

    xzeros(2, 3, names=("row", "col"))
    xfull((2, 2), 7.0, axes=[{"name": "y", "type": "space"}, "x"])
    xones_like(x)                        # inherits x's names / coords / unit

Each wrapper forwards its positional and unrecognised keyword arguments to the
matching `torch.*` function, and understands the `XTensor` metadata keywords
``names`` / ``axes`` / ``coords`` / ``unit`` -- so a factory result carries
names the same way a hand-built `XTensor` does.
"""

from __future__ import annotations

import torch
import typing_extensions as tx

from fiery.xtensor._tensors import XTensor

#: The `XTensor` metadata keywords a factory understands (everything else is
#: forwarded to the underlying `torch.*` op).
_META_KEYS = ("names", "axes", "coords", "unit")


def _split_meta(kwargs: dict) -> dict:
    """Pop the `XTensor` metadata keywords out of `kwargs` (in canonical
    order), leaving the rest to forward to `torch.*`."""
    return {key: kwargs.pop(key) for key in _META_KEYS if key in kwargs}


def _apply_meta(x: XTensor, meta: dict) -> None:
    """Set the metadata keywords on `x` via its property setters."""
    for key in _META_KEYS:
        if key in meta:
            setattr(x, key, meta[key])


def _make_factory(name: str) -> tx.Optional[tx.Callable]:
    """Build an `x<name>` wrapper around `torch.<name>` (or `None`)."""
    base = getattr(torch, name, None)
    if base is None:  # pragma: no cover - all wrapped ops are very old
        return None

    def factory(*args: tx.Any, **kwargs: tx.Any) -> XTensor:
        meta = _split_meta(kwargs)
        return XTensor(base(*args, **kwargs), **meta)

    factory.__name__ = "x" + name
    factory.__qualname__ = "x" + name
    factory.__doc__ = (
        f"Like `torch.{name}`, but returns an `XTensor`.\n\n"
        f"Positional and extra keyword arguments are forwarded to "
        f"`torch.{name}`; pass any of `names=` / `axes=` / `coords=` / "
        f"`unit=` to name, describe, label, and unit the axes of the result."
    )
    return factory


def _make_like_factory(name: str) -> tx.Optional[tx.Callable]:
    """Build an `x<name>` wrapper around `torch.<name>` (the `*_like` ops).

    A `torch.<name>(input, ...)` already carries an `XTensor` input's metadata
    through the generic `__torch_function__` path, so `xones_like(x)` inherits
    `x`'s names / coords / unit; any metadata keyword overrides what is
    inherited.
    """
    base = getattr(torch, name, None)
    if base is None:  # pragma: no cover - all wrapped ops are very old
        return None

    def factory(input: tx.Any, *args: tx.Any, **kwargs: tx.Any) -> XTensor:
        meta = _split_meta(kwargs)
        result = base(input, *args, **kwargs)
        if not isinstance(result, XTensor):
            result = XTensor(result)
        _apply_meta(result, meta)
        return result

    factory.__name__ = "x" + name
    factory.__qualname__ = "x" + name
    factory.__doc__ = (
        f"Like `torch.{name}`, but returns an `XTensor`.\n\n"
        f"When `input` is an `XTensor`, the result **inherits** its names, "
        f"coordinates, descriptors, and unit; pass `names=` / `axes=` / "
        f"`coords=` / `unit=` to override any of them."
    )
    return factory


# -- from-scratch constructors --------------------------------------------
xzeros = _make_factory("zeros")
xones = _make_factory("ones")
xempty = _make_factory("empty")
xfull = _make_factory("full")
#: Alias of [`xfull`][fiery.xtensor.xfull] -- fill a new tensor with a value.
xfill = xfull
xarange = _make_factory("arange")
xlinspace = _make_factory("linspace")
xlogspace = _make_factory("logspace")
xrand = _make_factory("rand")
xrandn = _make_factory("randn")
xeye = _make_factory("eye")

# -- `*_like` constructors (inherit the input's metadata) -----------------
xzeros_like = _make_like_factory("zeros_like")
xones_like = _make_like_factory("ones_like")
xempty_like = _make_like_factory("empty_like")
xfull_like = _make_like_factory("full_like")
xrand_like = _make_like_factory("rand_like")
xrandn_like = _make_like_factory("randn_like")


def xstack(
    tensors: tx.Sequence,
    dim: tx.Any = 0,
    *,
    name: tx.Optional[str] = None,
    coords: tx.Any = None,
    **kwargs: tx.Any,
) -> XTensor:
    """
    Like `torch.stack`, but lets you **name** (and label) the new axis.

    `torch.stack` inserts a brand-new axis that its signature gives no way to
    name, so it always comes out unnamed. `xstack` stacks the same way and then
    names the inserted axis `name` (at position `dim`) and, if given, labels it
    with `coords` -- handy for stacking a list of frames into a named,
    coordinate-carrying axis::

        xstack([r, g, b], name="channel", coords=("r", "g", "b"))

    The existing axes keep whatever names and labels the operands agree on
    (as with a plain `torch.stack`). `coords` needs a `name`.
    """
    result = torch.stack(list(tensors), dim, **kwargs)
    if not isinstance(result, XTensor):
        result = XTensor(result)
    axis = dim % result.ndim
    if name is not None:
        names = list(result.names)
        names[axis] = name
        result.names = tuple(names)
    if coords is not None:
        if name is None:
            raise ValueError("xstack: coords= needs a name= for the new axis")
        result.coords = dict(result.coords, **{name: coords})
    return result


def _single_name(t: tx.Any) -> tx.Optional[str]:
    """The name of a 1-D `XTensor`'s only axis (else `None`)."""
    if isinstance(t, XTensor) and t.ndim == 1:
        return t.names[0]
    return None


def xmeshgrid(
    *tensors: tx.Any,
    indexing: str = "ij",
    names: tx.Optional[tx.Sequence] = None,
) -> tx.Tuple[XTensor, ...]:
    """
    Like `torch.meshgrid`, but each output grid is a named, coordinate-carrying
    [`XTensor`][fiery.xtensor.XTensor].

    Every output spans all the input axes; `xmeshgrid` names those axes after
    the inputs (an `XTensor` input contributes its own axis name, or pass
    `names=` to set them) and attaches **each input as the coordinate** along
    its axis -- exactly the coordinate grid you usually build a meshgrid for::

        y = xarange(3, names=("y",))
        x = xarange(4, names=("x",))
        gy, gx = xmeshgrid(y, x)          # each is ("y", "x") with y/x coords

    `indexing` is `torch.meshgrid`'s (`"ij"` *(default)*, or `"xy"` which swaps
    the first two output axes). Inputs must be 1-D. A `None` axis name gets no
    coordinate.
    """
    if indexing not in ("ij", "xy"):
        raise ValueError(
            f"xmeshgrid: indexing must be 'ij' or 'xy', got {indexing!r}"
        )
    raws = [
        t.as_subclass(torch.Tensor) if isinstance(t, XTensor) else t
        for t in tensors
    ]
    # Always build the "ij" grids (old torch has no `indexing=`, and passing it
    # on new torch silences the "pass indexing" warning), then emulate "xy" by
    # swapping the first two axes -- exactly what "xy" does, on any torch.
    try:
        grids = torch.meshgrid(*raws, indexing="ij")
    except TypeError:  # pragma: no cover - only on torch without `indexing=`
        grids = torch.meshgrid(*raws)
    grids = tuple(grids)
    base = (
        tuple(names)
        if names is not None
        else tuple(_single_name(t) for t in tensors)
    )
    # `"xy"` swaps the first two output axes (and their names).
    order = list(range(len(tensors)))
    if indexing == "xy" and len(order) >= 2:
        order[0], order[1] = order[1], order[0]
        grids = tuple(g.transpose(0, 1) for g in grids)
    grid_names = tuple(base[i] for i in order)
    coords = {}
    for name, source in zip(base, tensors):
        if name is not None:
            coords[name] = source  # an XTensor keeps its position unit
    kwargs = {"coords": coords} if coords else {}
    return tuple(XTensor(grid, names=grid_names, **kwargs) for grid in grids)


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
    "xstack",
    "xmeshgrid",
    "xvector",
    "xmatrix",
]
