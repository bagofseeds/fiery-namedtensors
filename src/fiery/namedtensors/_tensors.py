# NOTE: `from __future__ import annotations` makes every annotation a lazy
# string, so modern typing syntax (`str | None`, `list[int]`, `Self`, ...)
# is safe in signatures even on old interpreters. Only values *evaluated at
# runtime* (the type aliases below) must stay old-syntax; hence `typing`
# generics and `typing_extensions` rather than PEP 585/604/695.
from __future__ import annotations

# stdlib
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from typing import Any, List, Optional, Tuple, TypeVar, Union
from warnings import filterwarnings

# dependencies
import torch
from torch import Tensor
from typing_extensions import Self

# internals
from fiery.namedtensors import _arrayutils as arrayutils
from fiery.namedtensors._arrayutils import EllipsisType

# typing (evaluated at import time -> keep old, runtime-valid syntax)
_SlicerT = Union[int, slice, EllipsisType, None]
_SmartSlicerT = Union[_SlicerT, List[int], Tensor]
CardinalSlicerT = Union[_SlicerT, Tuple[_SlicerT, ...]]
SmartSlicerT = Union[_SmartSlicerT, Tuple[_SmartSlicerT, ...]]
ArgIndexNameT = Union[str, EllipsisType, None]
ArgIndexNamesT = Sequence[Union[ArgIndexNameT, Sequence[ArgIndexNameT]]]
ChannelNameT = Optional[str]
ChannelNamesT = Tuple[ChannelNameT, ...]
T = TypeVar("T")

# warnings
filterwarnings("ignore", ".*(Named tensors).*", UserWarning)


def _torch_func(name: str) -> Optional[Callable]:
    """
    Resolve a torch callable by name, preferring the functional form
    (`torch.<name>`) and falling back to the tensor method
    (`torch.Tensor.<name>`).

    Returns `None` if neither exists in the running PyTorch version, so
    that overrides can be registered conditionally and we only ever
    overload functions that actually exist.
    """
    func = getattr(torch, name, None)
    if func is None:
        func = getattr(torch.Tensor, name, None)
    return func


class ExtendedTensorMeta(type(Tensor)):
    # We need a metaclass so that each subclass has its own registry

    def __new__(
        cls, name: str, bases: tuple[type, ...], classdict: Mapping
    ) -> type:
        kls = super().__new__(cls, name, bases, classdict)
        kls._OVERRIDES = {}
        return kls


class ExtendedTensor(Tensor, metaclass=ExtendedTensorMeta):
    """
    A tensor with extended functionality, represented as a PyTorch
    tensor subclass.

    This tensor overrides some PyTorch builtin functions using the
    `__torch_function__` protocol. Function overrides are saved in a
    registry.
    """

    @classmethod
    def overrides(cls, func: Optional[Callable]) -> Callable:
        """
        Decorator to register a function override.

        `func` may be `None` (e.g. when resolved through
        [`_torch_func`][fiery.namedtensors._tensors._torch_func] for an op
        that does not exist in the running PyTorch version); in that case
        the override is silently skipped so that we never overload a
        function that is missing from this PyTorch build.
        """

        def decorator(newfunc: Callable) -> Callable:
            if func is None:
                # Target op absent in this PyTorch version: do not register.
                return newfunc
            newfunc = wraps(func)(newfunc)
            # Register as a public torch function
            cls._OVERRIDES[func] = newfunc
            # Register as a torch.Tensor method
            setattr(cls, func.__name__, newfunc)
            return newfunc

        return decorator

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        # Lookup the function in the registry, with some sort of
        # inheritance logic.
        kwargs = kwargs or {}
        for base in cls.__mro__:
            OVERRIDES = getattr(base, "_OVERRIDES", {})
            if func in OVERRIDES:
                func = OVERRIDES[func]
                break
        out = super().__torch_function__(func, types, args, kwargs)
        # Propagate subclass attributes (e.g. named-index metadata) from the
        # first tensor argument onto the output. Only do this when the output
        # is an actual tensor: many functions (in-place ops, property
        # setters such as `.names = ...`) return `None`, and some take no
        # tensor as their first argument.
        if isinstance(out, Tensor) and args:
            source = args[0]
            attrs = set()
            for base in cls.__mro__:
                attrs |= set(getattr(base, "_ATTRS", set()))
            for attr in attrs:
                if not hasattr(out, attr) and hasattr(source, attr):
                    setattr(out, attr, getattr(source, attr))
        return out


# ======================================================================
#
#                           N A M E D   A X E S
#
# ======================================================================


class NamedTensor(ExtendedTensor):
    """
    A tensor with named axes, represented as a PyTorch tensor subclass.

    This class leverages PyTorch's builin "names" feature, but extends
    it to support additional methods that the builtin implementation
    does not (e.g. `permute`).
    """

    def __new__(cls, *args, **kwargs) -> Self:
        # NOTE: remove arguments that `Tensor.__new__` does not support.
        kwargs.pop("names", None)
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        # NOTE: Tensor does not implement `__init__` (only `__new__`),
        # but we add support for the `names` argument here.
        super().__init__()  # This actually calls `object.__init__`
        if "names" in kwargs:
            self.names = kwargs.pop("names")

    def __getitem__(self, slicer: SmartSlicerT) -> Self:
        # NOTE: when newaxes are present in a slicer, Tensor.__getitem__
        # falls back to torch.unsqueeze, which is not implemented for
        # named tensors. This section handles newaxes manually.
        out = self
        if not isinstance(slicer, tuple):
            slicer = (slicer,)

        # Insert new axes using unsqueeze
        new_axes = (i for i, index in enumerate(slicer) if index is None)
        for i, dim in enumerate(new_axes):
            out = out.unsqueeze(dim + i)

        # Keep all other types of indices
        slicer = tuple(
            index if index is not None else slice(None) for index in slicer
        )

        # Slice tensor
        return Tensor.__getitem__(out, slicer)

    @property
    def T(self) -> Self:
        dims = reversed(range(self.ndim))
        return self.permute(*dims)


@NamedTensor.overrides(_torch_func("permute"))
def _(input: NamedTensor, *dims: int | tuple[int, ...]) -> NamedTensor:
    if len(dims) == 1:
        dims = dims[0]
    names = input.names
    out = Tensor.permute(input.rename(None), dims)
    if names:
        out.names = tuple(names[dim] for dim in dims)
    return out


@NamedTensor.overrides(_torch_func("unsqueeze"))
def _(input: NamedTensor, dim: int) -> NamedTensor:
    names = list(input.names)
    out = Tensor.unsqueeze(input.rename(None), dim)
    names.insert(dim, None)
    out.names = tuple(names)
    return out


@NamedTensor.overrides(_torch_func("squeeze"))
def _(input: NamedTensor, dim: int | list[int] | None = None) -> NamedTensor:
    ndim = input.ndim
    names = list(input.names)
    bare = input.rename(None)
    # `Tensor.squeeze(t, None)` is rejected on some PyTorch versions; when
    # no dim is given, squeeze all singleton dimensions.
    out = Tensor.squeeze(bare) if dim is None else Tensor.squeeze(bare, dim)
    if dim is None:
        names = [name for name, size in zip(names, input.shape) if size != 1]
    else:
        if isinstance(dim, int):
            dim = (dim,)
        dim = [d + ndim if d < 0 else d for d in dim]
        for d in sorted(dim, reverse=True):
            names.pop(d)
    out.names = tuple(names)
    return out


@NamedTensor.overrides(_torch_func("view"))
def _(input: NamedTensor, *shape: int | tuple[int, ...]) -> NamedTensor:
    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]
    shape = list(shape)
    if -1 in shape:
        known_numel = torch.Size([s for s in shape if s != -1]).numel()
        shape[shape.index(-1)] = input.numel() // known_numel

    # Name-tracking through an arbitrary reshape is inherently ambiguous
    # (a dimension may be split or merged). We take the conservative and
    # predictable rule: a name is preserved only for output dimensions that
    # align exactly with an input dimension in an unbroken run from either
    # the front or the back. Every reshaped axis becomes unnamed.
    old_shape = list(input.shape)
    old_names = list(input.names)
    n_new, n_old = len(shape), len(old_shape)
    new_names = [None] * n_new

    # Leading run of exactly-matching dimensions.
    i = 0
    while i < n_new and i < n_old and shape[i] == old_shape[i]:
        new_names[i] = old_names[i]
        i += 1

    # Trailing run of exactly-matching dimensions (stopping before the
    # already-matched leading run on either side).
    j = 0
    while (
        j < n_new - i
        and j < n_old - i
        and shape[n_new - 1 - j] == old_shape[n_old - 1 - j]
    ):
        new_names[n_new - 1 - j] = old_names[n_old - 1 - j]
        j += 1

    out = Tensor.view(input.rename(None), *shape)
    out.names = tuple(new_names)
    return out


# ======================================================================
#
#                       N A M E D   I N D I C E S
#
# ======================================================================


class TensorWithNamedIndices(NamedTensor):
    """
    A tensor with axes that can be indexed by strings, rather than
    integers.

    The names of the indices are stored in the `index_names` attribute.
    """

    _ATTRS = {"_index_names", "_index_dims"}

    def __new__(cls, *args, **kwargs) -> Self:
        # NOTE: remove arguments that Tensor.__new__ does not support.
        kwargs.pop("index_names", None)
        kwargs.pop("index_dims", None)
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        data: Tensor,
        *,
        index_names: ArgIndexNamesT = (...,),
        index_dims: int | Sequence[int] = -1,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        data: Tensor
            The tensor data.
        index_names: [sequence of] sequence[str | {...} | None]
            The names of the indices in the tensor.
            Can contain `None` for unnamed indices, or `...` for any
            number of unnamed indices.
        index_dims: int | sequence[int], optional
            The dimensions of the tensor that are named.
            If not a sequence:
            * If positive, it is the starting dimension of the named axes.
            * If negative, it is the ending dimension of the named axes.

        Other Parameters
        ----------------
        **kwargs
            See `torch.Tensor` for other parameters.
        """
        super().__init__(data, **kwargs)
        self._index_names, self._index_dims = _prepare_index_names(
            index_names, index_dims, self.shape
        )

    @property
    def index_names(self) -> tuple[tuple[str | None, ...], ...] | None:
        # Read straight from __dict__ so that a tensor produced by an
        # auto-wrapped op (which may not carry the metadata) reports `None`
        # instead of triggering `__getattr__`.
        return self.__dict__.get("_index_names", None)

    @index_names.setter
    def index_names(self, value: ArgIndexNamesT) -> None:
        self._index_names, self._index_dims = _prepare_index_names(
            value, self.index_dims, self.shape
        )

    @property
    def index_dims(self) -> tuple[int, ...] | None:
        return self.__dict__.get("_index_dims", None)

    @index_dims.setter
    def index_dims(self, value: int | Sequence[int]) -> None:
        self._index_names, self._index_dims = _prepare_index_names(
            self.index_names, value, self.shape
        )

    def __getattr__(self, name: str) -> Self:
        # Private / dunder attributes are never index-name lookups. Raising
        # `AttributeError` here (rather than returning `None`) is what lets
        # `hasattr(out, "_index_names")` be False on a freshly-wrapped
        # tensor, so `__torch_function__` can copy the metadata across.
        if name.startswith("_"):
            raise AttributeError(name)

        def _error(name: str) -> None:
            raise AttributeError(f"No such index: {name}")

        # No named indices at all -> nothing to look up.
        if self.index_names is None:
            _error(name)

        # Convert name to indices
        names = name.split(".")
        indices = [
            axis_names.index(name) if name in axis_names else _error(name)
            for name, axis_names in zip(names, self.index_names)
        ]

        # Build slicer
        slicer = [slice(None)] * self.ndim
        for dim, index in zip(self.index_dims, indices):
            slicer[dim] = index

        return self[tuple(slicer)]

    @wraps(Tensor.__getitem__)
    def __getitem__(self, index: SmartSlicerT) -> Self:
        # Slice tensor
        out = super().__getitem__(index)
        # If there are no named indices, leave whatever metadata the
        # auto-wrapping propagated (i.e. nothing) untouched.
        if self.index_names is None:
            return out
        # Compute new index names and dims
        idx = arrayutils._unroll_slicer(index, self.ndim)
        index_names, index_dims = self.index_names, self.index_dims
        index_names, index_dims = _slice_names_nd(index_names, index_dims, idx)
        # Assign the already-canonical (names, dims) directly to the private
        # attributes. Going through the public setters would re-run
        # `_prepare_index_names`, which needs both values at once and would
        # therefore see a stale/`None` counterpart mid-assignment.
        out._index_names, out._index_dims = index_names, index_dims
        return out

    def index(
        self, positions: SmartSlicerT, dims: int | tuple[int, ...]
    ) -> Self:
        """
        Index positions along one or more dimensions.

        This is a bespoke method (not a PyTorch op), so it is defined
        directly rather than through the version-guarded override
        mechanism: it must be available on every supported PyTorch
        version regardless of whether `torch.Tensor.index` exists.

        Parameters
        ----------
        positions : slicer or tuple of slicers
            The index (or indices) to select along each dimension in
            `dims`.
        dims : int or tuple of int
            The dimension (or dimensions) to index into. Must have the
            same length as `positions`.
        """
        if not isinstance(positions, tuple):
            positions = (positions,)
        if not isinstance(dims, tuple):
            dims = (dims,)
        if len(dims) != len(positions):
            raise ValueError(
                f"Number of dimensions ({len(dims)}) does not match "
                f"number of positions ({len(positions)})"
            )

        # Normalize negative dims.
        dims = tuple(d + self.ndim if d < 0 else d for d in dims)

        # Build a fully specified slicer, placing each position at its dim
        # and taking everything along the other dimensions, then delegate
        # to __getitem__ (which propagates the index names and dims).
        position_by_dim = dict(zip(dims, positions))
        slicer = tuple(
            position_by_dim.get(dim, slice(None)) for dim in range(self.ndim)
        )
        return self[slicer]


class NamedVector(TensorWithNamedIndices):
    """
    A vector with named axes, represented as a PyTorch tensor subclass.

    The names of the axes are stored in the `channels` attribute, which is
    a tuple of strings or `None` values. The order of the names matches
    the order of the values in the channel axis.
    """

    def __new__(cls, *args, **kwargs) -> Self:
        # NOTE: remove arguments that Tensor.__new__ does not support.
        kwargs.pop("channels", None)
        kwargs.pop("channel_dim", None)
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        data: Tensor,
        *,
        channels: Sequence[str | EllipsisType | None] = (...,),
        channel_dim: int = -1,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        data: Tensor
            The data of the (batched) matrix.
        channels: sequence[str | {...} | None]
            The names of the channels in the matrix.
            Can contain `None` for unnamed channels, or `...` for any
            number of unnamed channels.
        channel_dim: int, optional
            The index of the channel dimensions.
        """
        super().__init__(
            data, **kwargs, index_names=channels, index_dims=channel_dim
        )

    @property
    def channels(self) -> tuple[str | None, ...]:
        return self.index_names[0]

    @channels.setter
    def channels(self, value: Sequence[str | None]) -> None:
        self.index_names = (tuple(value),)

    @property
    def channel_dim(self) -> int:
        return self.index_dims[0]

    @channel_dim.setter
    def channel_dim(self, value: int) -> None:
        self.index_dims = (value,)


class NamedMatrix(TensorWithNamedIndices):
    """
    A matrix with named axes, represented as a PyTorch tensor subclass.
    """

    def __new__(cls, *args, **kwargs) -> Self:
        # NOTE: remove arguments that Tensor.__new__ does not support.
        kwargs.pop("channels", None)
        kwargs.pop("channel_dims", None)
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        data: Tensor,
        *,
        channels: tuple[Sequence[ArgIndexNameT], Sequence[ArgIndexNameT]] = (
            (...,),
            (...,),
        ),
        channel_dims: int | tuple[int, int] = -1,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        data: Tensor
            The data of the (batched) matrix.
        channels: pair of sequence[str | {...} | None]
            The names of the channels in the matrix.
            Can contain `None` for unnamed channels, or `...` for any
            number of unnamed channels.
        channel_dims: int | tuple[int, int], optional
            The indices of the channel dimensions.
        """
        super().__init__(
            data, **kwargs, index_names=channels, index_dims=channel_dims
        )

    @property
    def channels(self) -> tuple[str | None, ...]:
        return self.index_names

    @channels.setter
    def channels(self, value: Sequence[Sequence[str | None]]) -> None:
        self.index_names = tuple(map(tuple, value))

    @property
    def channel_dims(self) -> tuple[int, ...]:
        return self.index_dims

    @channel_dims.setter
    def channel_dims(self, value: Sequence[int]) -> None:
        self.index_dims = tuple(value)


# ======================================================================
#
#                           O V E R R I D E S
#
# ======================================================================


@TensorWithNamedIndices.overrides(_torch_func("permute"))
def _(
    input: TensorWithNamedIndices, *dims: int | tuple[int, ...]
) -> TensorWithNamedIndices:
    # Accept both `x.permute(0, 2, 1)` and `x.permute((0, 2, 1))`.
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = tuple(dims[0])
    dims = tuple(d + input.ndim if d < 0 else d for d in dims)

    out = NamedTensor.permute(input, *dims)
    # Permuting only reorders axes: the per-axis index names are unchanged,
    # but each named axis moves to its new position.
    if input.index_names is not None:
        out._index_names = input.index_names
        out._index_dims = tuple(dims.index(d) for d in input.index_dims)
    return out


@TensorWithNamedIndices.overrides(_torch_func("index_select"))
def _(input: TensorWithNamedIndices, dim: int, index: Tensor) -> Tensor:
    if dim < 0:
        dim += input.ndim

    out = NamedTensor.index_select(input, dim, index)
    # index_select keeps ndim; only the selected axis' names are re-sliced.
    if input.index_names is not None:
        names = list(input.index_names)
        dims = input.index_dims
        if dim in dims:
            k = dims.index(dim)
            names[k] = _slice_names(names[k], index)
        out._index_names = tuple(names)
        out._index_dims = dims
    return out


# ======================================================================
#
#                               U T I L S
#
# ======================================================================


def _get_sequence_depth(seq: Sequence) -> int:
    """Compute the depth of a nested sequence."""
    if not isinstance(seq, Sequence) or isinstance(seq, (str, bytes)):
        return 0
    elif not seq:
        return 1
    else:
        return 1 + max(_get_sequence_depth(item) for item in seq)


def _prepare_index_names(
    index_names: ArgIndexNamesT,
    index_dims: int | Sequence[int],
    shape: tuple[int, ...],  # Shape of the tensor
) -> tuple[tuple[ArgIndexNameT, ...], tuple[int, ...]]:
    """Ensure names and dims are tuples with the same length."""

    depth = _get_sequence_depth(index_names)
    if depth not in (1, 2):
        raise ValueError(
            f"Invalid index_names: {index_names}. "
            f"Must be a sequence of strings or a sequence of "
            f"sequences of strings."
        )

    # Ensure names is a sequence of sequence
    if depth == 1:
        index_names = (index_names,)
    ndim = len(index_names)

    # Ensure dims is a tuple of integers
    if isinstance(index_dims, int):
        if index_dims > 0:
            index_dims = tuple(range(index_dims, index_dims + ndim))
        else:
            index_dims += len(shape)
            index_dims = tuple(range(index_dims - ndim + 1, index_dims + 1))

    # Unroll index names
    sizes = tuple(shape[dim] for dim in index_dims)
    index_names = tuple(
        tuple(arrayutils._unroll(names, size))
        for names, size in zip(index_names, sizes)
    )

    return index_names, index_dims


def _slice_names(
    names: tuple[str | None, ...], slicer: _SmartSlicerT
) -> ChannelNamesT | None:
    """Apply a 1D slicer to a tuple of names."""
    if isinstance(slicer, int):
        return names[slicer]
    if isinstance(slicer, slice):
        return names[slicer]
    if arrayutils._is_boolean_index(slicer):
        return tuple(channel for channel, keep in zip(names, slicer) if keep)
    if arrayutils._is_advanced_index(slicer):
        return tuple(names[i] for i in slicer)
    return None


def _slice_names_nd(
    index_names: tuple[tuple[str | None, ...], ...],
    index_dims: tuple[int, ...],
    slicer: SmartSlicerT,  # must be unrolled
) -> tuple[
    tuple[tuple[str | None, ...], ...],  # new index names
    tuple[int, ...],  # new index dims
]:
    if index_names is None or index_dims is None:
        return None, None

    axis_map = arrayutils._map_axes_inverse(slicer)

    new_names, new_dims = [], []
    for dim, names in zip(index_dims, index_names):
        new_dim = axis_map[dim]
        if isinstance(new_dim, int):
            dim_slicer = arrayutils._get_slicer_by_index(slicer, dim)
            new_names.append(_slice_names(names, dim_slicer))
            new_dims.append(new_dim)

    return tuple(new_names) or None, tuple(new_dims) or None
