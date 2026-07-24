from __future__ import annotations

# stdlib
from collections.abc import Sequence
from dataclasses import dataclass

# dependencies
import torch
import typing_extensions as tx
from torch import Tensor

# EllipsisType lives in `types` only since Python 3.10; fall back to the
# concrete type of `...` on older interpreters. Re-exported for _tensors.py.
try:
    from types import EllipsisType
except ImportError:  # Python < 3.10
    EllipsisType = type(...)

# torch.broadcast_shapes was added in torch 1.8; provide a small fallback so
# the utilities work across a wider PyTorch range.
try:
    from torch import broadcast_shapes
except ImportError:  # torch < 1.8

    def broadcast_shapes(*shapes):  # noqa: ANN001, ANN201
        return tuple(
            torch.broadcast_tensors(
                *[torch.empty(s, device="meta") for s in shapes]
            )[0].shape
        )


# typing (evaluated at import time -> use tx, never abc/builtin subscription)
_SlicerT = tx.Union[int, slice, EllipsisType, None]
_SmartSlicerT = tx.Union[_SlicerT, tx.List[int], Tensor]
CardinalSlicerT = tx.Union[_SlicerT, tx.Tuple[_SlicerT, ...]]
SmartSlicerT = tx.Union[_SmartSlicerT, tx.Tuple[_SmartSlicerT, ...]]

# constants
_UNSET: tx.Final[object] = object()


def _is_basic_index(value: tx.Any) -> bool:
    return isinstance(value, (int, slice, type(...), type(None)))


def _is_boolean_index(value: tx.Any) -> bool:
    if torch.is_tensor(value):
        return value.dtype == torch.bool
    elif isinstance(value, Sequence) and not isinstance(value, str):
        return all(isinstance(v, bool) for v in value)
    return False


def _is_advanced_index(value: tx.Any) -> bool:
    if torch.is_tensor(value):
        return value.dtype == torch.long
    elif isinstance(value, Sequence) and not isinstance(value, str):
        return all(isinstance(v, int) for v in value)
    return False


def _is_valid_index(value: tx.Any) -> bool:
    return (
        _is_basic_index(value)
        or _is_boolean_index(value)
        or _is_advanced_index(value)
    )


def _count_input_axes(values: SmartSlicerT) -> int:
    """
    Predict the number of axes in the input of an array slicing operation.
    """
    if not isinstance(values, tuple):
        values = (values,)

    count = 0
    for value in values:
        if value is ...:
            continue

        elif value is None:
            continue

        elif torch.is_tensor(value):
            if value.dtype == torch.bool:
                count += value.ndim
            else:
                count += 1

        elif isinstance(value, (int, slice, Sequence)):
            count += 1

        else:
            raise ValueError(f"Invalid slicer value: {value}")

    return count


def _count_output_axes(values: SmartSlicerT) -> int:
    """
    Predict the number of axes in the output of an array slicing operation.
    """
    if not isinstance(values, tuple):
        raise ValueError("Slicer must have been unrolled first.")

    count = 0
    shapes = []
    for value in values:
        if value is ...:
            raise ValueError("Slicer must have been unrolled first.")

        elif value is None:
            count += 1

        elif isinstance(value, int):
            continue

        elif isinstance(value, slice):
            count += 1

        elif torch.is_tensor(value):
            if value.dtype == torch.bool:
                count += 1
            else:
                shapes += [value.shape]

        elif isinstance(value, Sequence):
            shapes += [(len(value),)]

        else:
            raise ValueError(f"Invalid slicer value: {value}")

    if shapes:
        count += len(broadcast_shapes(*shapes))
    return count


def _unroll(
    values: Sequence,
    nb_values: int,
    side: tx.Literal["left", "right"] = "right",
    insert: tx.Any = None,
    ignore: tx.Any = _UNSET,
) -> Sequence:
    """
    Unroll a list by replacing an ellipsis with as many `insert` values
    as needed to reach a target number of values.

    If no ellipsis is present, an ellipsis is inserted on the left or
    right side of the list, depending on the `side` argument. Note that
    the default (`'right'`) corresponds to the common slicing convention
    in NumPy and PyTorch.

    The value `ignore` is ignored when counting the number of values in
    the input (or output) sequence.

    !!! example "Expand an array slicer"
        ```python
        _unroll((..., 0), 3, insert=slice(None), ignore=None)
        # Output: (slice(None), slice(None), 0)
        ```

    Parameters
    ----------
    values : sequence
        The input sequence, possibly containing an ellipsis (`...`).
    nb_values : int
        The target number of values in the output sequence.
    side : {'left', 'right'}, default 'right'
        The side on which to insert the ellipsis if it is not present in
        the input sequence.
    insert : any, default None
        The value to insert in place of the ellipsis.
    ignore : any, default _UNSET
        A value to ignore when counting the number of values in the input
        sequence. If set to `_UNSET`, all values are counted.
        If a callable, private behavior (see `_unroll_slicer`).

    Returns
    -------
    sequence
        The unrolled sequence.

    """
    otype = type(values)
    values = tuple(values)

    # Ensure an ellipsis is present
    side = side[:1].upper()
    if ... not in values:
        if side == "R":
            values += (...,)
        else:
            values = (...,) + values
    elif values.count(...) > 1:
        raise ValueError("Only one ellipsis is allowed")
    ellipsis_index = values.index(...)

    # Count the number of axes to unroll
    if ignore is not _UNSET:
        # If a callable, it counts the number of valid values.
        # I.e., it is the opposite behavior to "ignore".
        if not callable(ignore):
            ignore_value = ignore
            ignore = lambda x: int(x != ignore_value)  # noqa: E731
        nb_axes_current = sum(map(ignore, values))
    else:
        nb_axes_current = len(values) - 1

    if nb_axes_current > nb_values:
        raise ValueError(
            f"Too many axes ({nb_axes_current}) to unroll into {nb_values}"
        )

    # Insert as many anonymous axes as needed
    nb_axes_missing = max(0, nb_values - nb_axes_current)
    values = (
        values[:ellipsis_index]
        + (insert,) * nb_axes_missing
        + values[ellipsis_index + 1 :]
    )

    # Create new list of axes
    return otype(values)


def _unroll_slicer(
    values: SmartSlicerT,
    nb_values: int,
    side: tx.Literal["left", "right"] = "right",
    insert: tx.Any = slice(None),
    ignore: tx.Any = _count_input_axes,
) -> Sequence:
    """Specialized version of `_unroll` for array slicers."""
    if not isinstance(values, tuple):
        values = (values,)
    return _unroll(values, nb_values, side, insert, ignore)


@dataclass
class _AxisMapping:
    """Represents one output axis, or a group of advanced output axes."""

    axes: int | tuple[int, ...] | None
    """
    Input axes that end up in this output axis.
    If `None`, this is a new axis.
    """

    shape: tuple[int, ...] | None = None
    """
    Shape of the advanced index.
    Should only be set if index is a tensor or list of integers.
    """

    consumed: bool = False
    """
    Whether the input axis is consumed by the slicing operation.
    """


def _parse_slicer(
    slicer: SmartSlicerT,
    ndim: int | None = None,
) -> tuple[_AxisMapping, ...]:
    """
    Parse a slicer into a tuple of _AxisMapping objects.
    """
    if ndim is not None:
        slicer = _unroll_slicer(slicer, ndim)

    if not isinstance(slicer, tuple):
        raise ValueError("Slicer must have been unrolled first.")

    # List of objects representing each output axis.
    output_axes = []

    # Counter that count the number of input axes that have been
    # processed so far.
    input_axis = 0

    # Flags to check whether advanced axes are contiguous in the slicer.
    adv_is_contiguous = True
    previous_axis_was_adv = None

    # Iterate over the slicer.
    for value in slicer:
        if _is_advanced_index(value):
            # Check whether advanced axes are contiguous in the slicer
            adv_is_contiguous &= previous_axis_was_adv is not False
            previous_axis_was_adv = True

            # Store mapping
            shape = getattr(value, "shape", (len(value),))
            output_axes.append(_AxisMapping(input_axis, shape))

            # Increment counter
            input_axis += 1
            continue

        # Not an advanced axis -> mark it as such for the next iteration
        if previous_axis_was_adv is True:
            previous_axis_was_adv = False

        if _is_boolean_index(value):
            # Store mapping
            ndim = getattr(value, "ndim", 1)
            if ndim == 1:
                axes = input_axis
            else:
                axes = tuple(range(input_axis, input_axis + ndim))
            output_axes.append(_AxisMapping(axes))

            # Increment counter
            input_axis += ndim
            continue

        if isinstance(value, slice):
            output_axes.append(_AxisMapping(input_axis))
            input_axis += 1
            continue

        if isinstance(value, int):
            output_axes.append(_AxisMapping(input_axis, consumed=True))
            input_axis += 1
            continue

        if value is None:
            output_axes.append(_AxisMapping(None))
            continue

        raise ValueError(f"Invalid slicer value: {value}")

    # Compute brodcasted shape of all advanced indices.
    adv_shape = broadcast_shapes(
        *[axis.shape for axis in output_axes if axis.shape is not None]
    )
    adv_ndim = len(adv_shape)

    # Compute indices of all input axes that were advanced-indexed.
    adv_axes = tuple(
        set(axis.axes for axis in output_axes if axis.shape is not None)
    )
    adv_axes = [_AxisMapping(adv_axes)] * adv_ndim

    # Reorder axes
    reordered_axes = []
    if not adv_is_contiguous:
        reordered_axes += adv_axes

    has_seen_advanced = not adv_is_contiguous
    for axis in output_axes:
        if axis.shape is None:
            # Output axes of non-advanced (basic + boolean) indices
            reordered_axes.append(axis)

        elif not has_seen_advanced:
            # Output axes of advanced indices,
            reordered_axes += adv_axes
            has_seen_advanced = True

    return tuple(reordered_axes)


def _map_axes(
    slicer: SmartSlicerT, ndim: int | None = None, inverse: bool = False
) -> tuple[int | tuple[int, ...] | None, ...]:
    """
    Map output axis indices to input axis indices, given a slicer.

    If `ndim` is provided, the slicer is unrolled first.

    This function returns a tuple of the same length as the number of
    output axes, where each element is the index of the corresponding
    input axis, or `None` if it is a new axis.

    If dimensions are indexed using advanced indices (one boolean tensor
    with >= 2 dimensions, or >= 2 integer tensors), each advanced output
    axis will contain all matching input axes in a tuple.

    !!! example "Dropped axis"
        ```python
        _map_axes((slice(None), 0, slice(None)))
        # Output: (0, 2)
        ```

    !!! example "New axis"
        ```python
        _map_axes((slice(None), None, slice(None)))
        # Output: (0, None, 1)
        ```

    !!! example "Boolean indexing"
        ```python
        _map_axes((slice(None), torch.ones([5, 6], dtype=torch.bool)))
        # Output: (0, (1, 2))
        # NOTE: two dimensions are indexed by the same boolean tensor.
        ```

    !!! example "Advanced indexing"
        ```python
        _map_axes((slice(None), range(5), range(5)))
        # Output: (0, (1, 2))
        # NOTE: the two advanced indices (range) broadcast together.
        ```

    !!! example "Advanced multidimensional indexing"
        ```python
        i = torch.arange(5).view(5, 1)
        j = torch.arange(6).view(1, 6)
        _map_axes((slice(None), i, j))
        # Output: (0, (1, 2), (1, 2))
        # NOTE: the two advanced indices (tensors) broadcast together.
        ```

    !!! example "Non-contiguous advanced indexing"
        ```python
        _map_axes((slice(None), range(5), None, range(5)))
        # Output: ((1, 2), 0, None)
        # NOTE: advanced axes end up on the left of the output tensor.
        ```
    """
    if inverse:
        return _map_axes_inverse(slicer, ndim)

    output_axes = _parse_slicer(slicer, ndim)
    output_axes = tuple(axis.axes for axis in output_axes if not axis.consumed)
    return output_axes


def _map_axes_inverse(
    slicer: SmartSlicerT, ndim: int | None = None
) -> tuple[int | tuple[int, ...] | None, ...]:
    """
    Map input axis indices to output axis indices, given a slicer.

    If `ndim` is provided, the slicer is unrolled first.

    This function returns a tuple of the same length as the number of
    input axes, where each element is the index of the corresponding
    output axis, or `None` if the axis was dropped/consumed.

    If a dimension is indexed with an advanced index (a integer tensor
    with >= 2 dimensions), it will map to multiple output axes.

    !!! example "Dropped axis"
        ```python
        _map_axes_inverse((slice(None), 0, slice(None)))
        # Output: (0, None, 1)
        ```

    !!! example "New axis"
        ```python
        _map_axes_inverse((slice(None), None, slice(None)))
        # Output: (0, 2)
        ```

    !!! example "Boolean indexing"
        ```python
        _map_axes_inverse((slice(None), torch.ones([5, 6], dtype=torch.bool)))
        # Output: (0, 1, 1)
        # NOTE: two dimensions are indexed by the same boolean tensor.
        ```

    !!! example "Advanced indexing"
        ```python
        _map_axes_inverse((slice(None), range(5), range(5)))
        # Output: (0, 1, 1)
        # NOTE: the two advanced indices (range) broadcast together.
        ```

    !!! example "Advanced multidimensional indexing"
        ```python
        i = torch.arange(5).view(5, 1)
        j = torch.arange(6).view(1, 6)
        _map_axes_inverse((slice(None), i, j))
        # Output: (0, (1, 2), (1, 2))
        # NOTE: the two advanced indices (tensors) broadcast together.
        ```

    !!! example "Non-contiguous advanced indexing"
        ```python
        _map_axes_inverse((slice(None), range(5), None, range(5)))
        # Output: (1, 0, 0)
        # NOTE: advanced axes end up on the left of the output tensor.
        ```
    """
    if ndim is not None:
        slicer = _unroll_slicer(slicer, ndim)

    output_axes = _parse_slicer(slicer)
    input_axes = [None] * _count_input_axes(slicer)

    output_axis = -1
    for axis in output_axes:
        if axis.consumed:
            input_axes[axis.axes] = None
            continue

        output_axis += 1

        if axis.axes is None:
            continue

        if isinstance(axis.axes, int):
            input_axes[axis.axes] = output_axis

        else:
            for input_axis in axis.axes:
                if input_axes[input_axis] is None:
                    input_axes[input_axis] = output_axis
                    continue

                if not isinstance(input_axes[input_axis], tuple):
                    input_axes[input_axis] = (input_axes[input_axis],)

                if isinstance(output_axis, tuple):
                    input_axes[input_axis] += output_axis
                else:
                    input_axes[input_axis] += (output_axis,)

    return tuple(input_axes)


def _map_axis_index(
    axis: int, slicer: SmartSlicerT, ndim: int | None = None
) -> int | None:
    """
    Map an axis index in the input array to an axis index in the output
    array, given a slicer.

    If the output axis is dropped by the slicer (either because it has
    been indexed by an integer, or because it has been masked by a
    boolean tensor), return `None` instead.

    If the number of input dimensions `ndim` is not provided, the slicer
    must have been unrolled first, and the axis index must be non-negative.
    """
    if ndim is not None:
        slicer = _unroll_slicer(slicer, ndim)
        if axis < 0:
            axis += ndim

    return _map_axes_inverse(slicer, ndim)[axis]


def _get_slicer_by_index(
    slicer: SmartSlicerT, index: int, ndim: int | None = None
) -> SmartSlicerT:
    """
    Get the slicer value corresponding to a given input axis index.

    If the number of input dimensions `ndim` is not provided, the slicer
    must have been unrolled first, and the axis index must be non-negative.
    """
    if ndim is not None:
        slicer = _unroll_slicer(slicer, ndim)
        if index < 0:
            index += ndim

    for value in slicer:
        if value is ...:
            raise ValueError("Slicer must have been unrolled first.")

        elif value is None:
            continue

        elif isinstance(value, (int, slice)):
            if index == 0:
                return value
            index -= 1

        elif torch.is_tensor(value):
            if value.dtype == torch.bool:
                if index < value.ndim:
                    return value
                index -= value.ndim
            else:
                if index == 0:
                    return value
                index -= 1

        elif isinstance(value, Sequence):
            if index == 0:
                return value
            index -= 1

        else:
            raise ValueError(f"Invalid slicer value: {value}")

    raise IndexError(f"Input axis {index} out of range")
