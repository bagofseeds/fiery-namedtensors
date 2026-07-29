"""RESHAPE / REORDER overrides: `permute` and its special cases never change
a dimension's name or size, so coordinates -- keyed by name -- are carried
through untouched by `_carry`; no per-op coordinate bookkeeping is needed
except where an axis is merged, split, or resized.

Importing this module registers its overrides onto `XTensor` (see
CLAUDE.md, "How the subclassing works").
"""

from __future__ import annotations

import torch
import typing_extensions as tx
from torch import Tensor

from fiery.xtensor._common import (
    _carry,
    _query_positions,
    _resolve_axis,
    _resolve_dims,
)
from fiery.xtensor._compat import torch_func as _torch_func
from fiery.xtensor._tensors import (
    Coordinate,
    XTensor,
    _coords_dropping,
    _coords_for,
    _coords_of,
    _slice_affine_coordinate,
)


def _fold_affine_coords(
    input: XTensor, squeezed: tuple, result_names: tuple
) -> dict:
    """
    `input`'s coordinates after squeezing away `squeezed` (dim names, all of
    size 1 by `squeeze`'s own contract). A compact **affine** coordinate
    (Proposal 0005 step 3) that spans one or more of them folds those dims
    out **exactly** -- the same fold `_slice_affine_coordinate` already does
    for an integer index, reused here by treating each squeezed dim as index
    `0` (its only position) and every other dim as a full pass-through slice
    -- rather than being dropped outright the way `_coords_for` would (a
    size-1 axis is exact to fold, not merely "conservative to keep"). Labels
    and explicit coordinates aren't foldable this way, so they fall back to
    the ordinary survives-or-drops rule.
    """
    squeezed_set = set(squeezed)
    valid = _coords_of(input)
    stored = input.__dict__.get("_coords") or {}
    kept = {name for name in result_names if name is not None}
    names = input.names
    out = {}
    for key, (dims, coord) in stored.items():
        if key not in valid:
            continue
        touched = squeezed_set & set(dims)
        if not touched:
            if all(dim in kept for dim in dims):
                out[key] = (dims, coord)
            continue
        if not (isinstance(coord, Coordinate) and coord._compact()):
            continue  # labels / explicit: conservatively dropped, as before
        if any(dim not in kept and dim not in squeezed_set for dim in dims):
            continue  # some other, non-squeezed dim didn't survive either
        pieces = {dim: (0 if dim in touched else slice(None)) for dim in dims}
        sizes = {dim: input.shape[names.index(dim)] for dim in dims}
        result = _slice_affine_coordinate(coord, dims, pieces, sizes)
        if result is not None:
            out[key] = result
    return out


# ======================================================================
#
#                       R E S H A P E   /   R E O R D E R
#
# ======================================================================
#
# Reorder ops (permute and its special cases) never change a dimension's name
# or size, so coordinates -- keyed by name -- are carried through untouched by
# `_carry`; no per-op coordinate bookkeeping is needed.


@XTensor.overrides(_torch_func("permute"))
def _(input: XTensor, *dims: int | str | tuple) -> XTensor:
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = tuple(dims[0])
    names = input.names
    # A single `...` stands for every axis not listed, in their current order
    # (the `align_to` semantics), so `x.permute("w", ...)` moves `w` to front.
    if Ellipsis in dims:
        dims = tuple(input._align_order(dims))
    dims = tuple(_resolve_axis(names, dim) for dim in dims)
    result = Tensor.permute(input, dims)
    return _carry(input, result, _axis_names=tuple(names[dim] for dim in dims))


@XTensor.overrides(_torch_func("unsqueeze"))
def _(input: XTensor, dim: int) -> XTensor:
    names = list(input.names)
    result = Tensor.unsqueeze(input, dim)
    names.insert(dim, None)
    return _carry(input, result, _axis_names=tuple(names))


@XTensor.overrides(_torch_func("squeeze"))
def _(input: XTensor, dim: int | str | tx.Sequence | None = None) -> XTensor:
    ndim = input.ndim
    names = list(input.names)
    if dim is not None:
        dim = _resolve_dims(input.names, dim)
    # `Tensor.squeeze(t, None)` is rejected on some PyTorch versions; when
    # no dim is given, squeeze all singleton dimensions.
    result = (
        Tensor.squeeze(input) if dim is None else Tensor.squeeze(input, dim)
    )
    if dim is None:
        squeezed_positions = [
            i for i, size in enumerate(input.shape) if size == 1
        ]
        names = [name for name, size in zip(names, input.shape) if size != 1]
    else:
        if isinstance(dim, int):
            dim = (dim,)
        dim = [d + ndim if d < 0 else d for d in dim]
        squeezed_positions = list(dim)
        for d in sorted(dim, reverse=True):
            names.pop(d)
    names = tuple(names)
    # a squeezed dim is always size 1, so a compact **affine** coordinate
    # spanning it folds out exactly (Proposal 0005 step 3), rather than being
    # dropped the way `_coords_for` would.
    squeezed_names = tuple(
        input.names[i]
        for i in squeezed_positions
        if input.names[i] is not None
    )
    return _carry(
        input,
        result,
        _axis_names=names,
        _coords=_fold_affine_coords(input, squeezed_names, names),
    )


def _normalize_shape(input: XTensor, shape: tuple) -> list:
    """Flatten a `(shape,)` tuple arg and resolve a single `-1` entry."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
        shape = tuple(shape[0])
    shape = list(shape)
    if -1 in shape:
        known_numel = torch.Size([s for s in shape if s != -1]).numel()
        shape[shape.index(-1)] = input.numel() // known_numel
    return shape


def _reshape_names(
    old_shape: list, old_names: list, new_shape: list
) -> tuple[str | None, ...]:
    """
    Names for a reshape/view. Name-tracking through an arbitrary reshape is
    inherently ambiguous (a dimension may be split or merged), so we take the
    conservative, predictable rule: a name is preserved only for output
    dimensions that align exactly with an input dimension in an unbroken run
    from either the front or the back. Every reshaped axis becomes unnamed.
    """
    n_new, n_old = len(new_shape), len(old_shape)
    new_names: list = [None] * n_new

    # Leading run of exactly-matching dimensions.
    i = 0
    while i < n_new and i < n_old and new_shape[i] == old_shape[i]:
        new_names[i] = old_names[i]
        i += 1

    # Trailing run of exactly-matching dimensions (stopping before the
    # already-matched leading run on either side).
    j = 0
    while (
        j < n_new - i
        and j < n_old - i
        and new_shape[n_new - 1 - j] == old_shape[n_old - 1 - j]
    ):
        new_names[n_new - 1 - j] = old_names[n_old - 1 - j]
        j += 1

    return tuple(new_names)


def _reshape(input: XTensor, result: Tensor, shape: list) -> XTensor:
    names = _reshape_names(list(input.shape), list(input.names), shape)
    return _carry(
        input, result, _axis_names=names, _coords=_coords_for(input, names)
    )


@XTensor.overrides(_torch_func("view"))
def _(input: XTensor, *shape: int | tuple[int, ...]) -> XTensor:
    shape = _normalize_shape(input, shape)
    return _reshape(input, Tensor.view(input, *shape), shape)


@XTensor.overrides(_torch_func("reshape"))
def _(input: XTensor, *shape: int | tuple[int, ...]) -> XTensor:
    shape = _normalize_shape(input, shape)
    return _reshape(input, Tensor.reshape(input, shape), shape)


def _transpose_order(ndim: int, dim0: int, dim1: int) -> list:
    order = list(range(ndim))
    d0, d1 = dim0 % ndim, dim1 % ndim
    order[d0], order[d1] = order[d1], order[d0]
    return order


def _movedim_order(
    ndim: int,
    source: int | tuple[int, ...],
    destination: int | tuple[int, ...],
) -> list:
    src = [source] if isinstance(source, int) else list(source)
    dst = [destination] if isinstance(destination, int) else list(destination)
    src = [s % ndim for s in src]
    dst = [d % ndim for d in dst]
    order = [d for d in range(ndim) if d not in src]
    for dest, s in sorted(zip(dst, src)):
        order.insert(dest, s)
    return order


def _movedim_block_order(ndim: int, block: list, destination: int) -> list:
    """
    Permutation that moves `block` (positions, in their given order) to a
    single contiguous run governed by a scalar `destination` — the block-move
    generalisation of `movedim`. As with a one-axis move, the run *starts* at
    `destination`, or (for a negative `destination`) *ends* there.
    """
    k = len(block)
    remaining = [d for d in range(ndim) if d not in block]
    start = (destination % ndim) - k + 1 if destination < 0 else destination
    start = max(0, min(start, len(remaining)))
    return remaining[:start] + list(block) + remaining[start:]


def _move_permutation(
    input: XTensor, source: tx.Any, destination: tx.Any
) -> list:
    """
    Resolve the `permute` order for a `movedim`/`moveaxis` call. A **descriptor
    query** for `source` (e.g. `{"type": "space"}`) selects *every* matching
    axis and moves them as a contiguous block to the scalar `destination`,
    preserving relative order; otherwise `source`/`destination` pair up as in
    plain `movedim` (names allowed, resolved to ints).
    """
    if isinstance(source, dict):
        return _movedim_block_order(
            input.ndim, _query_positions(input, source), destination
        )
    source = _resolve_dims(input.names, source)
    return _movedim_order(input.ndim, source, destination)


@XTensor.overrides(_torch_func("transpose"))
def _(input: XTensor, dim0: int | str, dim1: int | str) -> XTensor:
    names = input.names
    dim0, dim1 = _resolve_axis(names, dim0), _resolve_axis(names, dim1)
    return input.permute(*_transpose_order(input.ndim, dim0, dim1))


@XTensor.overrides(_torch_func("swapaxes"))
def _(input: XTensor, dim0: int | str, dim1: int | str) -> XTensor:
    names = input.names
    dim0, dim1 = _resolve_axis(names, dim0), _resolve_axis(names, dim1)
    return input.permute(*_transpose_order(input.ndim, dim0, dim1))


@XTensor.overrides(_torch_func("swapdims"))
def _(input: XTensor, dim0: int | str, dim1: int | str) -> XTensor:
    names = input.names
    dim0, dim1 = _resolve_axis(names, dim0), _resolve_axis(names, dim1)
    return input.permute(*_transpose_order(input.ndim, dim0, dim1))


@XTensor.overrides(_torch_func("movedim"))
def _(input: XTensor, source, destination) -> XTensor:
    # `source` names existing axis/axes (resolvable, or a descriptor query);
    # `destination` is a target position, so it stays an integer.
    return input.permute(*_move_permutation(input, source, destination))


@XTensor.overrides(_torch_func("moveaxis"))
def _(input: XTensor, source, destination) -> XTensor:
    return input.permute(*_move_permutation(input, source, destination))


# -- rank-changing reshape --------------------------------------------------


def _broadcast_meta(input: XTensor, result: Tensor) -> dict:
    """
    `_carry` overrides for `expand`/`broadcast_to`: prepends unnamed axes for
    any new leading dims, and drops the coordinate of any *existing* named
    axis whose size actually grew (a size-1 axis broadcast to N). A compact
    coordinate has no length of its own to invalidate the way a label/
    explicit one does, so without this it would silently rebind to the new
    size as if that many positions had always existed (issue #90) -- N
    positions along a broadcast axis are still only ever *one* position's
    worth of underlying data.
    """
    n_new = result.ndim - input.ndim
    in_names = input.names
    changed = {
        in_names[i]
        for i, size in enumerate(input.shape)
        if size != result.shape[i + n_new] and in_names[i] is not None
    }
    overrides = {"_axis_names": (None,) * n_new + in_names}
    if changed:
        overrides["_coords"] = _coords_dropping(input, *changed)
    return overrides


@XTensor.overrides(_torch_func("flatten"))
def _(
    input: XTensor,
    start_dim: int | str = 0,
    end_dim: int | str = -1,
) -> XTensor:
    ndim = input.ndim
    start = _resolve_axis(input.names, start_dim) % ndim
    end = _resolve_axis(input.names, end_dim) % ndim
    result = Tensor.flatten(input, start, end)
    in_names = input.names
    if start == end:
        return _carry(input, result)  # no-op: names/coords unchanged
    names = in_names[:start] + (None,) + in_names[end + 1 :]
    return _carry(
        input, result, _axis_names=names, _coords=_coords_for(input, names)
    )


@XTensor.overrides(_torch_func("unflatten"))
def _(input: XTensor, dim: int | str, sizes: tx.Sequence) -> XTensor:
    ndim = input.ndim
    dim = _resolve_axis(input.names, dim) % ndim
    result = Tensor.unflatten(input, dim, sizes)
    k = len(sizes)
    in_names = input.names
    split = (in_names[dim],) if k == 1 else (None,) * k
    names = in_names[:dim] + split + in_names[dim + 1 :]
    return _carry(
        input, result, _axis_names=names, _coords=_coords_for(input, names)
    )


@XTensor.overrides(_torch_func("expand"))
def _(input: XTensor, *sizes: int | tx.Sequence) -> XTensor:
    if len(sizes) == 1 and isinstance(sizes[0], (tuple, list, torch.Size)):
        sizes = tuple(sizes[0])
    result = Tensor.expand(input, *sizes)
    return _carry(input, result, **_broadcast_meta(input, result))


@XTensor.overrides(_torch_func("broadcast_to"))
def _(input: XTensor, shape: tx.Sequence) -> XTensor:
    result = Tensor.broadcast_to(input, shape)
    return _carry(input, result, **_broadcast_meta(input, result))


@XTensor.overrides(_torch_func("diagonal"))
def _(
    input: XTensor,
    offset: int = 0,
    dim1: int | str = 0,
    dim2: int | str = 1,
) -> XTensor:
    d1 = _resolve_axis(input.names, dim1) % input.ndim
    d2 = _resolve_axis(input.names, dim2) % input.ndim
    result = Tensor.diagonal(input, offset, d1, d2)
    # `dim1`/`dim2` are removed; the new diagonal axis is appended (unnamed).
    names = tuple(
        n for i, n in enumerate(input.names) if i not in (d1, d2)
    ) + (None,)
    return _carry(
        input, result, _axis_names=names, _coords=_coords_for(input, names)
    )
