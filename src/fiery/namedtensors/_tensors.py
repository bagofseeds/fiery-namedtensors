# `from __future__ import annotations` keeps every annotation a lazy string,
# so only values *evaluated at runtime* -- the type aliases below -- must avoid
# PEP 585/604/695. All typing goes through `typing_extensions` (imported as
# `tx`), never abc/builtin subscription (e.g. `collections.abc.Sequence[...]`,
# which is not subscriptable before Python 3.9). `tx.Sequence` also works for
# the runtime isinstance checks.
from __future__ import annotations

# stdlib
from functools import wraps

# dependencies
import torch
import typing_extensions as tx
from torch import Tensor

# internals
from fiery.namedtensors import _arrayutils as arrayutils
from fiery.namedtensors._arrayutils import SmartSlicerT, _SmartSlicerT
from fiery.namedtensors._compat import EllipsisType
from fiery.namedtensors._compat import no_dispatch as _no_dispatch
from fiery.namedtensors._compat import torch_func as _torch_func

# typing (evaluated at import time -> use tx, never abc/builtin subscription).
# The slicer aliases (`SmartSlicerT`, ...) are shared from `_arrayutils`.
ArgIndexNameT = tx.Union[str, EllipsisType, None]
"""An index name: a string, `...` (any run of unnamed indices), or `None`."""

ArgIndexNamesT = tx.Sequence[
    tx.Union[ArgIndexNameT, tx.Sequence[ArgIndexNameT]]
]
"""Index names for one axis, or a sequence of such (one per named axis)."""

ChannelNameT = tx.Optional[str]
"""One channel name (`None` if unnamed)."""

ChannelNamesT = tx.Tuple[ChannelNameT, ...]
"""The ordered channel names of a `NamedVector` / `NamedMatrix` axis."""


def _resolve_axis(names: tuple[str | None, ...], dim: tx.Any) -> tx.Any:
    """
    Resolve one axis specifier to an integer position.

    A `str` is looked up in `names` (raising if absent); an `int` (possibly
    negative) or `None` passes through unchanged, as does anything else (e.g.
    a `Tensor`), so callers can share this with ops whose first argument is
    not always a dimension.
    """
    if isinstance(dim, str):
        try:
            return names.index(dim)
        except ValueError:
            raise ValueError(
                f"no axis named {dim!r} in {tuple(names)}"
            ) from None
    return dim


def _resolve_dims(names: tuple[str | None, ...], dim: tx.Any) -> tx.Any:
    """
    Resolve an axis specifier, or a sequence of them, to integer position(s).

    Wraps [`_resolve_axis`][fiery.namedtensors._tensors._resolve_axis]: a
    single specifier is resolved directly; a `tuple`/`list` is resolved
    element-wise (keeping its container type); anything else passes through.
    """
    if isinstance(dim, str):
        return _resolve_axis(names, dim)
    if isinstance(dim, (tuple, list)):
        return type(dim)(_resolve_axis(names, d) for d in dim)
    return dim


def _carry(source: Tensor, result: Tensor, **overrides: tx.Any) -> Tensor:
    """
    Return `result` as `source`'s subclass, carrying `source`'s subclass
    metadata (its `__dict__`, e.g. `_axis_names` / `_index_names`) and then
    applying `overrides` on top.

    Name-aware overrides return `_carry(input, <op>(input, ...), **new_meta)`
    so the *same* metadata lands on the result whether the op was reached as a
    method (`x.op(...)`) or as a function (`torch.op(x, ...)`).
    """
    cls = type(source)
    out = result if type(result) is cls else result.as_subclass(cls)
    out.__dict__.update(source.__dict__)
    out.__dict__.update(overrides)
    return out


class ExtendedTensorMeta(type(Tensor)):
    # We need a metaclass so that each subclass has its own registry

    def __new__(
        cls, name: str, bases: tuple[type, ...], classdict: tx.Mapping
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
    def overrides(cls, func: tx.Optional[tx.Callable]) -> tx.Callable:
        """
        Decorator to register a function override.

        `func` may be `None` (e.g. when resolved through
        [`torch_func`][fiery.namedtensors._compat.torch_func] for an op
        that does not exist in the running PyTorch version); in that case
        the override is silently skipped so that we never overload a
        function that is missing from this PyTorch build.
        """

        def decorator(newfunc: tx.Callable) -> tx.Callable:
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
        func: tx.Callable,
        types: tuple[type, ...],
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> tx.Any:
        # Look up a name-aware override for this op (most-derived first).
        kwargs = kwargs or {}
        override = None
        for base in cls.__mro__:
            registry = getattr(base, "_OVERRIDES", {})
            if func in registry:
                override = registry[func]
                break

        if override is not None:
            # The override computes the result AND its name metadata (via
            # `_carry`), so the same metadata lands for `x.op(...)` and
            # `torch.op(x, ...)`. Run it with subclass dispatch disabled so the
            # plain torch ops it calls do not recurse back here, and return its
            # result directly (routing it back through the default
            # `__torch_function__` re-wraps and drops the metadata on some
            # PyTorch versions).
            with _no_dispatch():
                return override(*args, **kwargs)

        out = super().__torch_function__(func, types, args, kwargs)
        # Ops without a name-aware override: carry subclass attributes (axis
        # names, named-index metadata) from the first tensor argument onto the
        # output, when the output is a real tensor (many ops return `None`).
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

    Axis names are **self-managed**: they live in the `_axis_names`
    attribute (propagated through `__torch_function__` via `_ATTRS`) and are
    exposed through the `names` property, which shadows PyTorch's builtin
    named-tensor attribute. The underlying tensor is never given builtin
    names, so the class does not depend on the experimental named-tensor API
    (`.rename` / builtin `.names`), which has been removed in some PyTorch
    builds.
    """

    _ATTRS = {"_axis_names"}

    def __new__(cls, *args, **kwargs) -> tx.Self:
        # NOTE: remove arguments that `Tensor.__new__` does not support.
        kwargs.pop("names", None)
        # Wrapping an existing tensor via `Tensor.__new__(cls, t)` is not
        # portable: some PyTorch versions reject a non-default dtype there
        # (e.g. a Long tensor raises "expected Float"). `as_subclass` re-tags
        # an existing tensor as this subclass across versions without a copy.
        if len(args) == 1 and not kwargs and isinstance(args[0], Tensor):
            return args[0].as_subclass(cls)
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        # NOTE: Tensor does not implement `__init__` (only `__new__`),
        # but we add support for the `names` argument here.
        super().__init__()  # This actually calls `object.__init__`
        if "names" in kwargs:
            self.names = kwargs.pop("names")

    @property
    def names(self) -> tuple[str | None, ...]:
        """The name of each axis (`None` for unnamed axes)."""
        names = self.__dict__.get("_axis_names", None)
        # Fall back to all-unnamed if metadata is missing or stale (e.g. it
        # was propagated onto the output of an op that changed the rank but
        # has no name-aware override yet).
        if names is None or len(names) != self.ndim:
            return (None,) * self.ndim
        return names

    @names.setter
    def names(self, value: tx.Optional[tx.Sequence[str | None]]) -> None:
        if value is None:
            self.__dict__.pop("_axis_names", None)
            return
        value = tuple(value)
        if len(value) != self.ndim:
            raise ValueError(
                f"Expected {self.ndim} names, got {len(value)}: {value}"
            )
        self._axis_names = value

    def _resolve_rename(
        self, names: tuple, rename_map: dict
    ) -> tuple[str | None, ...]:
        """Compute the new axis-name tuple for `rename` / `rename_`."""
        if rename_map:
            if names:
                raise ValueError(
                    "rename: cannot mix positional names and a rename map"
                )
            current = list(self.names)
            for old, new in rename_map.items():
                if old not in current:
                    raise ValueError(
                        f"rename: no axis named {old!r} in {tuple(current)}"
                    )
                current[current.index(old)] = new
            return tuple(current)
        if len(names) == 1 and names[0] is None:
            return (None,) * self.ndim
        if len(names) == 1 and isinstance(names[0], (tuple, list)):
            names = tuple(names[0])
        new_names = tuple(names)
        if len(new_names) != self.ndim:
            raise ValueError(
                f"rename: expected {self.ndim} names, got {len(new_names)}"
            )
        return new_names

    def rename(self, *names: str | None, **rename_map: str) -> tx.Self:
        """
        Return a view with renamed axes (self-managed; not the builtin op).

        Call positionally (`x.rename("a", "b")`), with `None` to clear all
        names (`x.rename(None)`), or with a mapping to rename specific axes
        (`x.rename(old="new")`). Other subclass metadata is preserved.
        """
        new_names = self._resolve_rename(names, rename_map)
        # `as_subclass` returns a view but does not copy `__dict__`, so carry
        # the subclass metadata (e.g. named-index dims) over explicitly.
        out = self.as_subclass(type(self))
        out.__dict__.update(self.__dict__)
        out._axis_names = new_names
        return out

    def rename_(self, *names: str | None, **rename_map: str) -> tx.Self:
        """In-place variant of `rename`."""
        self._axis_names = self._resolve_rename(names, rename_map)
        return self

    def __getitem__(self, slicer: SmartSlicerT) -> tx.Self:
        # The underlying tensor carries no builtin names, so basic indexing
        # (including newaxis via `None`) works directly.
        out = Tensor.__getitem__(self, slicer)
        # Map each output axis back to its source axis to carry names across.
        in_names = self.names
        axis_map = arrayutils._map_axes(slicer, self.ndim)
        out._axis_names = tuple(
            in_names[src] if isinstance(src, int) else None for src in axis_map
        )
        return out

    @property
    def T(self) -> tx.Self:
        dims = reversed(range(self.ndim))
        return self.permute(*dims)

    @property
    def mT(self) -> tx.Self:
        """Transpose of the last two dimensions (names included)."""
        return self.transpose(-2, -1)


@NamedTensor.overrides(_torch_func("permute"))
def _(input: NamedTensor, *dims: int | str | tuple) -> NamedTensor:
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = tuple(dims[0])
    names = input.names
    dims = tuple(_resolve_axis(names, dim) for dim in dims)
    result = Tensor.permute(input, dims)
    return _carry(input, result, _axis_names=tuple(names[dim] for dim in dims))


@NamedTensor.overrides(_torch_func("unsqueeze"))
def _(input: NamedTensor, dim: int) -> NamedTensor:
    names = list(input.names)
    result = Tensor.unsqueeze(input, dim)
    names.insert(dim, None)
    return _carry(input, result, _axis_names=tuple(names))


@NamedTensor.overrides(_torch_func("squeeze"))
def _(
    input: NamedTensor, dim: int | str | tx.Sequence | None = None
) -> NamedTensor:
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
        names = [name for name, size in zip(names, input.shape) if size != 1]
    else:
        if isinstance(dim, int):
            dim = (dim,)
        dim = [d + ndim if d < 0 else d for d in dim]
        for d in sorted(dim, reverse=True):
            names.pop(d)
    return _carry(input, result, _axis_names=tuple(names))


def _normalize_shape(input: NamedTensor, shape: tuple) -> list:
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


@NamedTensor.overrides(_torch_func("view"))
def _(input: NamedTensor, *shape: int | tuple[int, ...]) -> NamedTensor:
    shape = _normalize_shape(input, shape)
    names = _reshape_names(list(input.shape), list(input.names), shape)
    return _carry(input, Tensor.view(input, *shape), _axis_names=names)


@NamedTensor.overrides(_torch_func("reshape"))
def _(input: NamedTensor, *shape: int | tuple[int, ...]) -> NamedTensor:
    shape = _normalize_shape(input, shape)
    names = _reshape_names(list(input.shape), list(input.names), shape)
    return _carry(input, Tensor.reshape(input, shape), _axis_names=names)


# Reorder ops are all special cases of `permute`; delegating to it means they
# inherit correct axis-name AND named-index behaviour (and functional/method
# parity) for free.


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


@NamedTensor.overrides(_torch_func("transpose"))
def _(input: NamedTensor, dim0: int | str, dim1: int | str) -> NamedTensor:
    names = input.names
    dim0, dim1 = _resolve_axis(names, dim0), _resolve_axis(names, dim1)
    return input.permute(*_transpose_order(input.ndim, dim0, dim1))


@NamedTensor.overrides(_torch_func("swapaxes"))
def _(input: NamedTensor, dim0: int | str, dim1: int | str) -> NamedTensor:
    names = input.names
    dim0, dim1 = _resolve_axis(names, dim0), _resolve_axis(names, dim1)
    return input.permute(*_transpose_order(input.ndim, dim0, dim1))


@NamedTensor.overrides(_torch_func("swapdims"))
def _(input: NamedTensor, dim0: int | str, dim1: int | str) -> NamedTensor:
    names = input.names
    dim0, dim1 = _resolve_axis(names, dim0), _resolve_axis(names, dim1)
    return input.permute(*_transpose_order(input.ndim, dim0, dim1))


@NamedTensor.overrides(_torch_func("movedim"))
def _(input: NamedTensor, source, destination) -> NamedTensor:
    # `source` names an existing axis (resolvable); `destination` is a target
    # position, so it stays an integer.
    source = _resolve_dims(input.names, source)
    return input.permute(*_movedim_order(input.ndim, source, destination))


@NamedTensor.overrides(_torch_func("moveaxis"))
def _(input: NamedTensor, source, destination) -> NamedTensor:
    source = _resolve_dims(input.names, source)
    return input.permute(*_movedim_order(input.ndim, source, destination))


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

    def __new__(cls, *args, **kwargs) -> tx.Self:
        # NOTE: remove arguments that Tensor.__new__ does not support.
        kwargs.pop("index_names", None)
        kwargs.pop("index_dims", None)
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        data: Tensor,
        *,
        index_names: ArgIndexNamesT = (...,),
        index_dims: int | tx.Sequence[int] = -1,
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
    def index_dims(self, value: int | tx.Sequence[int]) -> None:
        self._index_names, self._index_dims = _prepare_index_names(
            self.index_names, value, self.shape
        )

    def __getattr__(self, name: str) -> tx.Self:
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
    def __getitem__(self, index: SmartSlicerT) -> tx.Self:
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
    ) -> tx.Self:
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

    def __new__(cls, *args, **kwargs) -> tx.Self:
        # NOTE: remove arguments that Tensor.__new__ does not support.
        kwargs.pop("channels", None)
        kwargs.pop("channel_dim", None)
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        data: Tensor,
        *,
        channels: tx.Sequence[str | EllipsisType | None] = (...,),
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
    def channels(self, value: tx.Sequence[str | None]) -> None:
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

    def __new__(cls, *args, **kwargs) -> tx.Self:
        # NOTE: remove arguments that Tensor.__new__ does not support.
        kwargs.pop("channels", None)
        kwargs.pop("channel_dims", None)
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        data: Tensor,
        *,
        channels: tuple[
            tx.Sequence[ArgIndexNameT], tx.Sequence[ArgIndexNameT]
        ] = (
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
    def channels(self, value: tx.Sequence[tx.Sequence[str | None]]) -> None:
        self.index_names = tuple(map(tuple, value))

    @property
    def channel_dims(self) -> tuple[int, ...]:
        return self.index_dims

    @channel_dims.setter
    def channel_dims(self, value: tx.Sequence[int]) -> None:
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
    # Accept both `x.permute(0, 2, 1)` and `x.permute((0, 2, 1))`, and axis
    # names in place of integers.
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = tuple(dims[0])
    dims = tuple(_resolve_axis(input.names, d) for d in dims)
    dims = tuple(d + input.ndim if d < 0 else d for d in dims)

    out = NamedTensor.permute(input, *dims)
    # Permuting only reorders axes: the per-axis index names are unchanged,
    # but each named axis moves to its new position.
    if input.index_names is not None:
        out._index_names = input.index_names
        out._index_dims = tuple(dims.index(d) for d in input.index_dims)
    return out


@TensorWithNamedIndices.overrides(_torch_func("index_select"))
def _(input: TensorWithNamedIndices, dim: int | str, index: Tensor) -> Tensor:
    dim = _resolve_axis(input.names, dim)
    if dim < 0:
        dim += input.ndim

    result = Tensor.index_select(input, dim, index)
    # index_select keeps ndim; only the selected axis' names are re-sliced.
    meta = {}
    if input.index_names is not None:
        names = list(input.index_names)
        dims = input.index_dims
        if dim in dims:
            k = dims.index(dim)
            names[k] = _slice_names(names[k], index)
        meta = {"_index_names": tuple(names), "_index_dims": dims}
    return _carry(input, result, **meta)


# ======================================================================
#
#                           R E D U C T I O N S
#
# ======================================================================
#
# Dimension-reducing ops (`sum`, `mean`, `amax`, ...) drop the reduced axis'
# name (or keep it as a size-1 axis under `keepdim=True`), and accept a name
# in place of an integer `dim=`. They share one factory: the ops below all
# take `dim` as their first optional positional argument and either remove
# the reduced axes or keep them as size-1.


def _reduce_index_meta(
    input: TensorWithNamedIndices, removed: set, overrides: dict
) -> None:
    """Drop/shift named-index metadata for axes removed by a reduction."""
    idx_names = input.__dict__.get("_index_names")
    idx_dims = input.__dict__.get("_index_dims")
    if idx_names is None or idx_dims is None:
        return
    ndim = input.ndim
    new_names, new_dims = [], []
    for names, dim in zip(idx_names, idx_dims):
        dim %= ndim
        if dim in removed:
            continue
        # each surviving named axis shifts left by the removed axes before it
        new_names.append(names)
        new_dims.append(dim - sum(1 for r in removed if r < dim))
    overrides["_index_names"] = tuple(new_names) or None
    overrides["_index_dims"] = tuple(new_dims) or None


def _reduce_names(input: NamedTensor, result: tx.Any, dim: tx.Any) -> tx.Any:
    """Recompute the name metadata for a dimension-reducing op's result."""
    if not isinstance(result, Tensor):
        # e.g. a (values, indices) namedtuple: left to a bespoke override.
        return result
    in_names = input.names
    ndim = input.ndim
    # `keepdim` is inferable from the output rank: a reduction either removes
    # the reduced axes or keeps them as size-1.
    if dim is not None and result.ndim == ndim:
        return _carry(input, result, _axis_names=in_names)
    if dim is None:
        removed = set(range(ndim))
    else:
        dims = dim if isinstance(dim, (tuple, list)) else (dim,)
        removed = {d % ndim for d in dims}
    overrides = {
        "_axis_names": tuple(
            name for i, name in enumerate(in_names) if i not in removed
        )
    }
    _reduce_index_meta(input, removed, overrides)
    return _carry(input, result, **overrides)


def _make_reduction(name: str) -> None:
    """Register a name-aware override for a dimension-reducing torch op."""
    base = _torch_func(name)

    def _reduction(input: NamedTensor, *args, **kwargs) -> tx.Any:
        names = input.names
        # Resolve a name given for `dim` (positional arg 0 or keyword) and
        # remember the (resolved) value so the output names can be computed.
        if "dim" in kwargs:
            dim = kwargs["dim"] = _resolve_dims(names, kwargs["dim"])
        elif args:
            dim = _resolve_dims(names, args[0])
            args = (dim,) + args[1:]
        else:
            dim = None
        return _reduce_names(input, base(input, *args, **kwargs), dim)

    # `overrides(None)` is a no-op, so ops missing from this torch are skipped.
    NamedTensor.overrides(base)(_reduction)


# `dim` is the first optional positional for each; version-guarded via
# `_torch_func`, so absent ops (e.g. `nanmean` on very old torch) are skipped.
_REDUCTIONS = (
    "sum",
    "mean",
    "nansum",
    "nanmean",
    "prod",
    "amax",
    "amin",
    "all",
    "any",
    "argmax",
    "argmin",
    "logsumexp",
    "count_nonzero",
)
for _reduction_name in _REDUCTIONS:
    _make_reduction(_reduction_name)


# ======================================================================
#
#                               U T I L S
#
# ======================================================================


def _get_sequence_depth(seq: tx.Sequence) -> int:
    """Compute the depth of a nested sequence."""
    if not isinstance(seq, tx.Sequence) or isinstance(seq, (str, bytes)):
        return 0
    elif not seq:
        return 1
    else:
        return 1 + max(_get_sequence_depth(item) for item in seq)


def _prepare_index_names(
    index_names: ArgIndexNamesT,
    index_dims: int | tx.Sequence[int],
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
