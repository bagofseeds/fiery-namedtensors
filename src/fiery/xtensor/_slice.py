"""SLICE / SPLIT: `narrow`/`select`/`unbind`/`split`/`chunk` are expressed
as `__getitem__` on a single axis, so both axis names and coordinate labels
are tracked for free; `flip`/`roll` reorder labels (and flip a flipped
axis' `orientation`).

Importing this module registers its overrides onto `XTensor` (see
CLAUDE.md, "How the subclassing works").
"""

from __future__ import annotations

import typing_extensions as tx
from torch import Tensor

from fiery.xtensor._common import (
    _carry,
    _flip_orientation,
    _resolve_axis,
    _resolve_dims,
)
from fiery.xtensor._compat import torch_func as _torch_func
from fiery.xtensor._tensors import (
    Coordinate,
    XTensor,
    _coords_dropping,
    _slice_coordinate,
)

# ======================================================================
#
#                       S L I C E   /   S P L I T
#
# ======================================================================
#
# `narrow` / `select` / `split` / `chunk` are expressed as `__getitem__` on a
# single axis, so both axis names and coordinate labels are tracked for free.
# `flip` / `roll` keep the rank, but reorder the labels of the axes they touch.


def _slice_axis(input: XTensor, dim: int, index: tx.Any) -> tx.Any:
    """Index a single axis (`input[:, ..., index, ..., :]`)."""
    slicer = [slice(None)] * input.ndim
    slicer[dim] = index
    return input[tuple(slicer)]


@XTensor.overrides(_torch_func("narrow"))
def _(input: XTensor, dim: int | str, start: int, length: int) -> tx.Any:
    dim = _resolve_axis(input.names, dim) % input.ndim
    return _slice_axis(input, dim, slice(start, start + length))


@XTensor.overrides(_torch_func("select"))
def _(input: XTensor, dim: int | str, index: int) -> tx.Any:
    # `select(dim, i)` == `x[..., i, ...]`: the integer index drops the axis.
    dim = _resolve_axis(input.names, dim) % input.ndim
    return _slice_axis(input, dim, index)


@XTensor.overrides(_torch_func("unbind"))
def _(input: XTensor, dim: int | str = 0) -> tuple:
    dim = _resolve_axis(input.names, dim) % input.ndim
    return tuple(_slice_axis(input, dim, i) for i in range(input.shape[dim]))


@XTensor.overrides(_torch_func("split"))
def _(
    input: XTensor,
    split_size_or_sections: int | tx.Sequence,
    dim: int | str = 0,
) -> tuple:
    dim = _resolve_axis(input.names, dim) % input.ndim
    size = input.shape[dim]
    if isinstance(split_size_or_sections, int):
        step = split_size_or_sections
        sections = [step] * (size // step)
        if size % step:
            sections.append(size % step)
    else:
        sections = list(split_size_or_sections)
    pieces, start = [], 0
    for length in sections:
        pieces.append(_slice_axis(input, dim, slice(start, start + length)))
        start += length
    return tuple(pieces)


@XTensor.overrides(_torch_func("chunk"))
def _(input: XTensor, chunks: int, dim: int | str = 0) -> tuple:
    dim = _resolve_axis(input.names, dim) % input.ndim
    size = input.shape[dim]
    # `torch.chunk(n, chunks)` splits into pieces of ceil(n / chunks); the
    # last piece may be smaller (and there may be fewer than `chunks`).
    step = max(1, -(-size // chunks))
    return input.split(step, dim)


@XTensor.overrides(_torch_func("flip"))
def _(input: XTensor, dims: int | str | tx.Sequence) -> XTensor:
    resolved = _resolve_dims(input.names, dims)
    dlist = resolved if isinstance(resolved, (tuple, list)) else (resolved,)
    result = Tensor.flip(input, list(dlist))
    # Rank and axis positions are unchanged; the labels of a flipped axis are
    # reversed, and a flipped axis' `orientation` descriptor reverses too
    # ("left-to-right" -> "right-to-left").
    flipped = {input.names[d % input.ndim] for d in dlist}
    coords = _coords_dropping(input, *flipped)
    for name in flipped:
        labels = input.coords.get(name)
        if labels is None:
            continue
        if isinstance(labels, Coordinate):
            # A compact coordinate flips exactly by negating its spacing
            # (`_slice_coordinate`'s basic-slice path, `slice(None,None,-1)`
            # -- stays compact, no materialisation). An explicit one can't
            # use that same slice object: PyTorch tensors reject a negative
            # step (`t[::-1]` itself raises "step must be greater than
            # zero"), so it goes through the advanced-index path instead
            # (an explicit reversed position list) -- either way, never
            # `reversed()`/indexed as if it were a plain dict (#85).
            size = input.shape[input.names.index(name)]
            reverser = (
                slice(None, None, -1)
                if labels._compact()
                else list(range(size - 1, -1, -1))
            )
            reversed_coord = _slice_coordinate(labels, reverser, size)
            if reversed_coord is not None:
                coords[name] = (name,), reversed_coord
        else:
            coords[name] = (name,), tuple(reversed(labels))
    overrides = {"_coords": coords}
    meta = input._valid_axis_meta()
    if any("orientation" in meta.get(name, {}) for name in flipped):
        meta = {name: dict(extra) for name, extra in meta.items()}
        for name in flipped:
            if name in meta and "orientation" in meta[name]:
                meta[name]["orientation"] = _flip_orientation(
                    meta[name]["orientation"]
                )
        overrides["_axis_meta"] = meta
    return _carry(input, result, **overrides)


@XTensor.overrides(_torch_func("roll"))
def _(
    input: XTensor,
    shifts: int | tx.Sequence,
    dims: int | str | tx.Sequence | None = None,
) -> XTensor:
    if dims is None:
        # Flattened roll: axis names are unchanged, but per-axis label order
        # can no longer be tracked, so coordinates are dropped.
        result = Tensor.roll(input, shifts)
        return _carry(input, result, _coords={})

    dims = _resolve_dims(input.names, dims)
    result = Tensor.roll(input, shifts, dims)
    slist = shifts if isinstance(shifts, (tuple, list)) else (shifts,)
    dlist = dims if isinstance(dims, (tuple, list)) else (dims,)
    shift_by_name: dict = {}
    for shift, dim in zip(slist, dlist):
        name = input.names[dim % input.ndim]
        if name is not None:
            shift_by_name[name] = shift_by_name.get(name, 0) + shift
    coords = _coords_dropping(input, *shift_by_name)
    for name, shift in shift_by_name.items():
        labels = input.coords.get(name)
        if labels is None:
            continue
        if isinstance(labels, Coordinate):
            # a roll is a cyclic permutation, not a `slice`; give
            # `_slice_coordinate` the equivalent advanced index instead of
            # treating the coordinate as if it were a plain dict (#85).
            size = input.shape[input.names.index(name)]
            shift %= size or 1
            order = [(i - shift) % size for i in range(size)]
            rolled = _slice_coordinate(labels, order, size)
            if rolled is not None:
                coords[name] = (name,), rolled
        else:
            n = len(labels)
            shift %= n or 1
            coords[name] = (
                (name,),
                tuple(labels[(i - shift) % n] for i in range(n)),
            )
    return _carry(input, result, _coords=coords)
