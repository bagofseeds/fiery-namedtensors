"""GATHER / SCATTER: `index_select`/`gather`/`scatter`/`index_add`/
`index_copy`/`index_fill`/`where`/`masked_select`/`nonzero`. An index
tensor selects/rewrites positions along a named axis; coordinate labels
follow where they still apply, and drop where the result no longer lines
up with a single dim's own index space.

Importing this module registers its overrides onto `XTensor` (see
CLAUDE.md, "How the subclassing works").
"""

from __future__ import annotations

import torch
import typing_extensions as tx
from torch import Tensor

from fiery.xtensor._common import _carry, _resolve_axis
from fiery.xtensor._compat import no_dispatch as _no_dispatch
from fiery.xtensor._compat import torch_func as _torch_func
from fiery.xtensor._meta import _broadcast_batch_names
from fiery.xtensor._selection import _slice_labels
from fiery.xtensor._tensors import (
    Coordinate,
    XTensor,
    _coords_dropping,
    _names_of,
    _slice_coordinate,
)

# ======================================================================
#
#                     G A T H E R   /   S C A T T E R
#
# ======================================================================


@XTensor.overrides(_torch_func("index_select"))
def _(input: XTensor, dim: int | str, index: Tensor) -> tx.Any:
    dim = _resolve_axis(input.names, dim) % input.ndim
    result = Tensor.index_select(input, dim, index)
    # Rank is unchanged; only the selected axis' labels are re-sliced.
    name = input.names[dim]
    coords = _coords_dropping(input, name)
    labels = input.coords.get(name)
    if isinstance(labels, Coordinate):
        # A numeric `Coordinate` is a dict-like mapping keyed by strings
        # ("value"/"spacing"/"origin"), not a plain sequence -- positionally
        # integer-indexing it (what `_slice_labels` does, correctly, for a
        # tuple of labels) instead hits `Coordinate.__getitem__`, which looks
        # up a *key*, not a *position*, and KeyErrors (#85's pitfall; #162).
        # `index_select`'s index tensor is an advanced index (arbitrary
        # positions, not necessarily a contiguous range), so route it through
        # `_slice_coordinate`'s advanced-index branch instead -- the same
        # machinery `flip`/`roll` already use to re-slice a numeric
        # coordinate by an explicit position list.
        sliced = _slice_coordinate(labels, index, input.shape[dim])
        if sliced is not None:
            coords[name] = (name,), sliced
    elif labels is not None:
        coords[name] = (name,), tuple(_slice_labels(labels, index))
    return _carry(input, result, _coords=coords)


@XTensor.overrides(_torch_func("gather"))
def _(input: XTensor, dim: int | str, index: Tensor, **kwargs) -> tx.Any:
    dim = _resolve_axis(input.names, dim) % input.ndim
    result = torch.gather(input, dim, index, **kwargs)
    # Rank (and each axis' name) is preserved; the gathered positions change
    # per-slice, so the gathered axis' labels are dropped.
    coords = _coords_dropping(input, input.names[dim])
    return _carry(input, result, _coords=coords)


@XTensor.overrides(_torch_func("take_along_dim"))
def _(
    input: XTensor, indices: Tensor, dim: int | str = None, **kwargs
) -> tx.Any:
    result = torch.take_along_dim(
        input, indices, _resolve_axis(input.names, dim), **kwargs
    )
    if dim is not None:
        touched = input.names[_resolve_axis(input.names, dim) % input.ndim]
        coords = _coords_dropping(input, touched)
    else:
        coords = {}
    return _carry(input, result, _coords=coords)


@XTensor.overrides(_torch_func("scatter"))
def _(
    input: XTensor, dim: int | str, index: Tensor, *args, **kwargs
) -> tx.Any:
    dim = _resolve_axis(input.names, dim)
    result = torch.scatter(input, dim, index, *args, **kwargs)
    # Positions and sizes are unchanged, so names and coordinates survive.
    return _carry(input, result)


@XTensor.overrides(_torch_func("scatter_add"))
def _(
    input: XTensor, dim: int | str, index: Tensor, src: Tensor, **kwargs
) -> tx.Any:
    dim = _resolve_axis(input.names, dim)
    result = torch.scatter_add(input, dim, index, src, **kwargs)
    return _carry(input, result)


@XTensor.overrides(_torch_func("index_add"))
def _(
    input: XTensor,
    dim: int | str,
    index: Tensor,
    source: Tensor,
    *,
    alpha: tx.Any = 1,
    **kwargs,
) -> tx.Any:
    dim = _resolve_axis(input.names, dim)
    # `alpha` was added to `index_add` in a later torch; only pass it when it
    # is non-default so the override still works on older versions.
    if alpha != 1:
        kwargs["alpha"] = alpha
    result = torch.index_add(input, dim, index, source, **kwargs)
    # Rank and per-axis positions are unchanged (values at the indexed
    # positions are accumulated into), so names and coordinates survive.
    return _carry(input, result)


@XTensor.overrides(_torch_func("index_copy"))
def _(
    input: XTensor, dim: int | str, index: Tensor, source: Tensor, **kwargs
) -> tx.Any:
    dim = _resolve_axis(input.names, dim)
    result = torch.index_copy(input, dim, index, source, **kwargs)
    # Same shape, same positions -- only the values change.
    return _carry(input, result)


@XTensor.overrides(_torch_func("index_fill"))
def _(
    input: XTensor, dim: int | str, index: Tensor, value: tx.Any, **kwargs
) -> tx.Any:
    dim = _resolve_axis(input.names, dim)
    result = torch.index_fill(input, dim, index, value, **kwargs)
    # Same shape, same positions -- only the values change.
    return _carry(input, result)


@XTensor.overrides(_torch_func("where"))
def _(condition: Tensor, *args) -> tx.Any:
    # The 1-argument form `torch.where(cond)` returns indices (like nonzero);
    # leave it to the generic path.
    if not args:
        with _no_dispatch():
            return torch.where(condition)
    x, y = args
    result = torch.where(condition, x, y)
    names = _broadcast_batch_names(
        _broadcast_batch_names(_names_of(condition), _names_of(x)),
        _names_of(y),
    )
    ref = next(
        (t for t in (condition, x, y) if isinstance(t, XTensor)),
        condition,
    )
    # Reconciling coordinates across broadcast operands is out of scope; drop.
    return _carry(ref, result, _axis_names=names, _coords={})


@XTensor.overrides(_torch_func("masked_select"))
def _(input: XTensor, mask: Tensor, **kwargs) -> tx.Any:
    result = torch.masked_select(input, mask, **kwargs)
    ref = input if isinstance(input, XTensor) else mask
    # The result is 1-D and its length is data-dependent: a single unnamed
    # axis, and no coordinates.
    return _carry(ref, result, _axis_names=(None,), _coords={})


@XTensor.overrides(_torch_func("nonzero"))
def _(input: XTensor, **kwargs) -> tx.Any:
    result = torch.nonzero(input, **kwargs)
    # The output indexes the *nonzero entries* against the input's dimensions
    # -- its axes are not the input's named axes, so names/coords are dropped.
    # `as_tuple=True` gives one 1-D index tensor per input dim; the default
    # gives a single `(nnz, input.ndim)` index tensor.
    if isinstance(result, tuple):
        return tuple(
            _carry(input, part, _axis_names=(None,), _coords={})
            for part in result
        )
    return _carry(input, result, _axis_names=(None,) * result.ndim, _coords={})
