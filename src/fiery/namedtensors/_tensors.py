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
LabelT = tx.Optional[str]
"""One coordinate label (`None` for an unlabelled position)."""

LabelsT = tx.Tuple[LabelT, ...]
"""The ordered labels of a single (named) dimension."""

ArgLabelsT = tx.Sequence[tx.Union[str, EllipsisType, None]]
"""Labels as passed in: a sequence that may hold `...` for a run of `None`s."""

CoordsT = tx.Mapping[str, ArgLabelsT]
"""A mapping *dimension name -> its labels* (xarray-style coordinates)."""


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
    metadata (its `__dict__`, e.g. `_axis_names` / `_coords`) and then applying
    `overrides` on top.

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
        # names, coordinate labels) from the first tensor argument onto the
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
#                           N A M E D   T E N S O R
#
# ======================================================================


class NamedTensor(ExtendedTensor):
    """
    A tensor with named dimensions and, optionally, per-dimension coordinate
    **labels** -- an [xarray](https://docs.xarray.dev)-like `DataArray` over a
    live `torch.Tensor`.

    - **Dimensions** are named through `names` (self-managed in `_axis_names`,
      independent of PyTorch's experimental builtin named-tensor feature, so
      the class works even where that API has been removed).
    - **Coordinates** label the positions along a named dimension. They live
      in `coords` -- a mapping *dim name -> labels* -- keyed by dimension
      **name**, so they follow their dimension through reshaping/reordering
      with no positional bookkeeping. A labelled dimension must be named.

    Select by label with `sel`, by integer position with `isel`, or reach a
    single label by attribute (`x.red`).
    """

    _ATTRS = {"_axis_names", "_coords"}

    def __new__(cls, *args, **kwargs) -> tx.Self:
        # NOTE: remove arguments that `Tensor.__new__` does not support.
        kwargs.pop("names", None)
        kwargs.pop("coords", None)
        # Wrapping an existing tensor via `Tensor.__new__(cls, t)` is not
        # portable: some PyTorch versions reject a non-default dtype there
        # (e.g. a Long tensor raises "expected Float"). `as_subclass` re-tags
        # an existing tensor as this subclass across versions without a copy.
        if len(args) == 1 and not kwargs and isinstance(args[0], Tensor):
            return args[0].as_subclass(cls)
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        # NOTE: Tensor does not implement `__init__` (only `__new__`), but we
        # add support for the `names` / `coords` arguments here.
        super().__init__()  # This actually calls `object.__init__`
        names = kwargs.pop("names", None)
        coords = kwargs.pop("coords", None)
        if names is not None:
            self.names = names
        if coords is not None:
            self.coords = coords

    # -- dimensions --------------------------------------------------------

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

    # -- coordinates -------------------------------------------------------

    @property
    def coords(self) -> dict[str, LabelsT]:
        """
        The coordinate labels, as a `{dim name: labels}` dict.

        Only entries that are still valid are returned -- their dimension must
        be named on this tensor and its size must match the number of labels
        -- so stale metadata propagated onto a shape-changing op is hidden.
        """
        stored = self.__dict__.get("_coords") or {}
        names = self.names
        valid = {}
        for dim, labels in stored.items():
            if dim in names and len(labels) == self.shape[names.index(dim)]:
                valid[dim] = labels
        return valid

    @coords.setter
    def coords(self, value: tx.Optional[CoordsT]) -> None:
        if value is None:
            self.__dict__.pop("_coords", None)
            return
        names = self.names
        normalized = {}
        for dim, labels in dict(value).items():
            if dim not in names:
                raise ValueError(
                    f"coords: no axis named {dim!r} in {tuple(names)}"
                )
            size = self.shape[names.index(dim)]
            labels = tuple(labels)
            # `...` fills the middle with unlabelled positions.
            if Ellipsis in labels:
                labels = tuple(arrayutils._unroll(labels, size))
            if len(labels) != size:
                raise ValueError(
                    f"coords: dim {dim!r} has {len(labels)} labels "
                    f"for size {size}"
                )
            normalized[dim] = labels
        self._coords = normalized

    # -- renaming ----------------------------------------------------------

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

    def _remap_coords(self, new_names: tuple) -> dict:
        """Coordinates re-keyed from the current names to `new_names`."""
        coords = self.__dict__.get("_coords") or {}
        if not coords:
            return {}
        remapped = {}
        for old, new in zip(self.names, new_names):
            if old in coords and new is not None:
                remapped[new] = coords[old]
        return remapped

    def rename(self, *names: str | None, **rename_map: str) -> tx.Self:
        """
        Return a view with renamed axes (self-managed; not the builtin op).

        Call positionally (`x.rename("a", "b")`), with `None` to clear all
        names (`x.rename(None)`), or with a mapping to rename specific axes
        (`x.rename(old="new")`). Coordinates follow their (renamed) dimension.
        """
        new_names = self._resolve_rename(names, rename_map)
        # `as_subclass` returns a view but does not copy `__dict__`, so carry
        # the subclass metadata over explicitly.
        out = self.as_subclass(type(self))
        out.__dict__.update(self.__dict__)
        out._coords = self._remap_coords(new_names)
        out._axis_names = new_names
        return out

    def rename_(self, *names: str | None, **rename_map: str) -> tx.Self:
        """In-place variant of `rename`."""
        new_names = self._resolve_rename(names, rename_map)
        self._coords = self._remap_coords(new_names)
        self._axis_names = new_names
        return self

    # -- indexing / selection ---------------------------------------------

    def __getitem__(self, slicer: SmartSlicerT) -> tx.Self:
        # The underlying tensor carries no builtin names, so basic indexing
        # (including newaxis via `None`) works directly.
        out = Tensor.__getitem__(self, slicer)
        # Re-tag as this subclass: normal dispatch preserves it, but when
        # __getitem__ is reached from inside a `_no_dispatch` override (e.g.
        # `select` / `narrow` / `split`) the raw op returns a plain Tensor.
        if type(out) is not type(self):
            out = out.as_subclass(type(self))
        # Map each output axis back to its source axis to carry names across.
        # A single advanced index reports its source as a length-1 tuple; a
        # broadcast of several input axes reports a longer tuple (ambiguous ->
        # unnamed).
        in_names = self.names
        axis_map = arrayutils._map_axes(slicer, self.ndim)
        sources = [_single_source(src) for src in axis_map]
        out_names = tuple(
            in_names[src] if src is not None else None for src in sources
        )
        out._axis_names = out_names
        # Slice the labels of every kept axis that carries coordinates.
        coords = self.__dict__.get("_coords") or {}
        if coords:
            unrolled = arrayutils._unroll_slicer(slicer, self.ndim)
            new_coords = {}
            for out_axis, src in enumerate(sources):
                name = out_names[out_axis]
                if src is not None and name is not None:
                    labels = coords.get(in_names[src])
                    if labels is not None:
                        piece = arrayutils._get_slicer_by_index(unrolled, src)
                        sliced = _slice_labels(labels, piece)
                        if sliced is not None:
                            new_coords[name] = tuple(sliced)
            out._coords = new_coords
        return out

    def isel(self, **indexers: tx.Any) -> tx.Self:
        """
        Select by integer position along **named** dimensions.

        `x.isel(row=0, col=slice(1, 3))` indexes `row` at position 0 and `col`
        at positions 1..2, leaving the other axes untouched.
        """
        slicer = [slice(None)] * self.ndim
        for name, index in indexers.items():
            slicer[_resolve_axis(self.names, name)] = index
        return self[tuple(slicer)]

    def sel(self, **indexers: tx.Any) -> tx.Self:
        """
        Select by coordinate **label** along named dimensions.

        `x.sel(channel="red")` selects the position whose label is `"red"`. A
        list of labels selects several positions; a single label drops the
        dimension (like integer indexing).
        """
        coords = self.coords
        positional = {}
        for name, label in indexers.items():
            if name not in coords:
                raise ValueError(f"sel: dim {name!r} has no coordinates")
            labels = coords[name]
            is_many = isinstance(label, (list, tuple))
            wanted = list(label) if is_many else [label]
            positions = []
            for one in wanted:
                try:
                    positions.append(labels.index(one))
                except ValueError:
                    raise ValueError(
                        f"sel: no label {one!r} on dim {name!r}"
                    ) from None
            positional[name] = positions if is_many else positions[0]
        return self.isel(**positional)

    def __getattr__(self, name: str) -> tx.Self:
        # Only consulted when normal attribute lookup fails, so real methods
        # and attributes always win. Private / dunder names are never labels.
        if name.startswith("_"):
            raise AttributeError(name)
        coords = self.coords
        hits = [dim for dim, labels in coords.items() if name in labels]
        if len(hits) == 1:
            return self.sel(**{hits[0]: name})
        if len(hits) > 1:
            raise AttributeError(
                f"label {name!r} is ambiguous across dims {hits}"
            )
        raise AttributeError(name)

    # -- transposes --------------------------------------------------------

    @property
    def T(self) -> tx.Self:
        dims = reversed(range(self.ndim))
        return self.permute(*dims)

    @property
    def mT(self) -> tx.Self:
        """Transpose of the last two dimensions (names included)."""
        return self.transpose(-2, -1)

    # -- builtin named-tensor API, re-implemented self-managed -------------

    def refine_names(self, *names: str | None) -> tx.Self:
        """
        Return a view with names assigned to (only) the unnamed axes.

        Naming an already-named axis to a *different* name is an error; a
        given `None` keeps the current name. A single `...` keeps the names
        of the axes it spans. Self-managed (not the builtin op).
        """
        current = list(self.names)
        if Ellipsis in names:
            i = names.index(Ellipsis)
            n_explicit = len(names) - 1
            span = self.ndim - n_explicit
            if span < 0:
                raise ValueError(
                    f"refine_names: too many names for {self.ndim} axes"
                )
            names = names[:i] + tuple(current[i : i + span]) + names[i + 1 :]
        if len(names) != self.ndim:
            raise ValueError(
                f"refine_names: expected {self.ndim} names, got {len(names)}"
            )
        refined = []
        for cur, given in zip(current, names):
            if given is None:
                refined.append(cur)
            elif cur is not None and cur != given:
                raise ValueError(
                    f"refine_names: cannot rename axis {cur!r} to {given!r}"
                )
            else:
                refined.append(given)
        return self.rename(*refined)

    def _align_order(self, names: tuple) -> list:
        """Resolve a name order (possibly with `...`) to a permutation."""
        current = list(self.names)
        if Ellipsis in names:
            explicit = [n for n in names if n is not Ellipsis]
            rest = [n for n in current if n not in explicit]
            i = names.index(Ellipsis)
            names = names[:i] + tuple(rest) + names[i + 1 :]
        if len(names) != self.ndim:
            raise ValueError(
                f"align: expected {self.ndim} names, got {len(names)}"
            )
        return [_resolve_axis(self.names, n) for n in names]

    def align_to(self, *names: str) -> tx.Self:
        """
        Return a view with the axes permuted into the given name order.

        A single `...` stands for all the other axes, in their current
        order (e.g. `x.align_to(..., "channel")`). Self-managed.
        """
        return self.permute(*self._align_order(names))

    def align_as(self, other: NamedTensor) -> tx.Self:
        """
        Return a view aligned to `other`'s named axes.

        This tensor's axes are permuted into `other`'s order, and a size-1
        axis is inserted for every name that only `other` has. Every axis of
        `self` must be named and present in `other`. Self-managed.
        """
        target = _names_of(other)
        mine = self.names
        for name in mine:
            if name is None or name not in target:
                raise ValueError(
                    f"align_as: axis {name!r} is not in the target names "
                    f"{tuple(target)}"
                )
        # permute self's axes into the order they appear in `other`
        order = [mine.index(name) for name in target if name in mine]
        out = self.permute(*order)
        # insert size-1 axes for the names only `other` has
        for pos, name in enumerate(target):
            if name not in mine:
                out = out.unsqueeze(pos)
        return out.rename(*target)


# ---- coordinate helpers ---------------------------------------------------


def _coords_of(tensor: tx.Any) -> dict:
    """The coordinate labels of `tensor` (empty for a plain / non tensor)."""
    if isinstance(tensor, NamedTensor):
        return tensor.coords
    return {}


def _coords_for(input: NamedTensor, result_names: tuple) -> dict:
    """
    Keep only the coordinates whose dimension survives (by name) into
    `result_names`. Merged / split / removed axes lose their name and so drop
    their coordinates automatically.
    """
    kept = {name for name in result_names if name is not None}
    return {k: v for k, v in _coords_of(input).items() if k in kept}


def _single_source(src: tx.Any) -> tx.Optional[int]:
    """
    The single input axis an output axis came from, or `None` when it is a
    new axis or a broadcast of several input axes (`_map_axes` reports those
    as `None` / a multi-element tuple).
    """
    if isinstance(src, int):
        return src
    if isinstance(src, tuple) and len(src) == 1:
        return src[0]
    return None


def _slice_labels(labels: LabelsT, slicer: _SmartSlicerT) -> LabelsT | None:
    """Apply a 1-D slicer to a tuple of labels (see `__getitem__`)."""
    if isinstance(slicer, int):
        return (labels[slicer],)
    if isinstance(slicer, slice):
        return labels[slicer]
    if arrayutils._is_boolean_index(slicer):
        return tuple(x for x, keep in zip(labels, slicer) if keep)
    if arrayutils._is_advanced_index(slicer):
        return tuple(labels[int(i)] for i in slicer)
    return None


# ======================================================================
#
#                       R E S H A P E   /   R E O R D E R
#
# ======================================================================
#
# Reorder ops (permute and its special cases) never change a dimension's name
# or size, so coordinates -- keyed by name -- are carried through untouched by
# `_carry`; no per-op coordinate bookkeeping is needed.


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
    names = tuple(names)
    return _carry(
        input, result, _axis_names=names, _coords=_coords_for(input, names)
    )


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


def _reshape(input: NamedTensor, result: Tensor, shape: list) -> NamedTensor:
    names = _reshape_names(list(input.shape), list(input.names), shape)
    return _carry(
        input, result, _axis_names=names, _coords=_coords_for(input, names)
    )


@NamedTensor.overrides(_torch_func("view"))
def _(input: NamedTensor, *shape: int | tuple[int, ...]) -> NamedTensor:
    shape = _normalize_shape(input, shape)
    return _reshape(input, Tensor.view(input, *shape), shape)


@NamedTensor.overrides(_torch_func("reshape"))
def _(input: NamedTensor, *shape: int | tuple[int, ...]) -> NamedTensor:
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


# -- rank-changing reshape --------------------------------------------------


def _prepend_axes_meta(input: NamedTensor, n_new: int) -> dict:
    """`_carry` overrides for an op that prepends `n_new` unnamed axes."""
    # Existing axes keep their name and size, so their coordinates stay valid.
    return {"_axis_names": (None,) * n_new + input.names}


@NamedTensor.overrides(_torch_func("flatten"))
def _(
    input: NamedTensor,
    start_dim: int | str = 0,
    end_dim: int | str = -1,
) -> NamedTensor:
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


@NamedTensor.overrides(_torch_func("unflatten"))
def _(input: NamedTensor, dim: int | str, sizes: tx.Sequence) -> NamedTensor:
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


@NamedTensor.overrides(_torch_func("expand"))
def _(input: NamedTensor, *sizes: int | tx.Sequence) -> NamedTensor:
    if len(sizes) == 1 and isinstance(sizes[0], (tuple, list, torch.Size)):
        sizes = tuple(sizes[0])
    result = Tensor.expand(input, *sizes)
    return _carry(
        input, result, **_prepend_axes_meta(input, result.ndim - input.ndim)
    )


@NamedTensor.overrides(_torch_func("broadcast_to"))
def _(input: NamedTensor, shape: tx.Sequence) -> NamedTensor:
    result = Tensor.broadcast_to(input, shape)
    return _carry(
        input, result, **_prepend_axes_meta(input, result.ndim - input.ndim)
    )


@NamedTensor.overrides(_torch_func("diagonal"))
def _(
    input: NamedTensor,
    offset: int = 0,
    dim1: int | str = 0,
    dim2: int | str = 1,
) -> NamedTensor:
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


# ======================================================================
#
#                           R E D U C T I O N S
#
# ======================================================================
#
# Dimension-reducing ops (`sum`, `mean`, `amax`, ...) drop the reduced axis'
# name (and its coordinates), or keep it as a size-1 axis under `keepdim`, and
# accept a name in place of an integer `dim=`. They share one factory: the ops
# below take `dim` as their first optional positional argument and either
# remove the reduced axes or keep them as size-1.


def _reduce_names(input: NamedTensor, result: tx.Any, dim: tx.Any) -> tx.Any:
    """Recompute the name metadata for a dimension-reducing op's result."""
    if not isinstance(result, Tensor):
        # e.g. a (values, indices) namedtuple: left to a bespoke override.
        return result
    ndim = input.ndim
    # `keepdim` is inferable from the output rank: a reduction either removes
    # the reduced axes or keeps them as size-1.
    if dim is not None and result.ndim == ndim:
        return _carry(input, result, _axis_names=input.names)
    if dim is None:
        removed = set(range(ndim))
    else:
        dims = dim if isinstance(dim, (tuple, list)) else (dim,)
        removed = {d % ndim for d in dims}
    names = tuple(n for i, n in enumerate(input.names) if i not in removed)
    return _carry(
        input, result, _axis_names=names, _coords=_coords_for(input, names)
    )


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
#                       S L I C E   /   S P L I T
#
# ======================================================================
#
# `narrow` / `select` / `split` / `chunk` are expressed as `__getitem__` on a
# single axis, so both axis names and coordinate labels are tracked for free.
# `flip` / `roll` keep the rank, but reorder the labels of the axes they touch.


def _slice_axis(input: NamedTensor, dim: int, index: tx.Any) -> tx.Any:
    """Index a single axis (`input[:, ..., index, ..., :]`)."""
    slicer = [slice(None)] * input.ndim
    slicer[dim] = index
    return input[tuple(slicer)]


@NamedTensor.overrides(_torch_func("narrow"))
def _(input: NamedTensor, dim: int | str, start: int, length: int) -> tx.Any:
    dim = _resolve_axis(input.names, dim) % input.ndim
    return _slice_axis(input, dim, slice(start, start + length))


@NamedTensor.overrides(_torch_func("select"))
def _(input: NamedTensor, dim: int | str, index: int) -> tx.Any:
    # `select(dim, i)` == `x[..., i, ...]`: the integer index drops the axis.
    dim = _resolve_axis(input.names, dim) % input.ndim
    return _slice_axis(input, dim, index)


@NamedTensor.overrides(_torch_func("unbind"))
def _(input: NamedTensor, dim: int | str = 0) -> tuple:
    dim = _resolve_axis(input.names, dim) % input.ndim
    return tuple(_slice_axis(input, dim, i) for i in range(input.shape[dim]))


@NamedTensor.overrides(_torch_func("split"))
def _(
    input: NamedTensor,
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


@NamedTensor.overrides(_torch_func("chunk"))
def _(input: NamedTensor, chunks: int, dim: int | str = 0) -> tuple:
    dim = _resolve_axis(input.names, dim) % input.ndim
    size = input.shape[dim]
    # `torch.chunk(n, chunks)` splits into pieces of ceil(n / chunks); the
    # last piece may be smaller (and there may be fewer than `chunks`).
    step = max(1, -(-size // chunks))
    return input.split(step, dim)


@NamedTensor.overrides(_torch_func("flip"))
def _(input: NamedTensor, dims: int | str | tx.Sequence) -> NamedTensor:
    resolved = _resolve_dims(input.names, dims)
    dlist = resolved if isinstance(resolved, (tuple, list)) else (resolved,)
    result = Tensor.flip(input, list(dlist))
    # Rank and axis positions are unchanged; the labels of a flipped axis are
    # reversed too.
    flipped = {input.names[d % input.ndim] for d in dlist}
    coords = dict(input.coords)
    for name in flipped:
        if name in coords:
            coords[name] = tuple(reversed(coords[name]))
    return _carry(input, result, _coords=coords)


@NamedTensor.overrides(_torch_func("roll"))
def _(
    input: NamedTensor,
    shifts: int | tx.Sequence,
    dims: int | str | tx.Sequence | None = None,
) -> NamedTensor:
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
    coords = dict(input.coords)
    for name, shift in shift_by_name.items():
        if name in coords:
            labels = coords[name]
            n = len(labels)
            shift %= n or 1
            coords[name] = tuple(labels[(i - shift) % n] for i in range(n))
    return _carry(input, result, _coords=coords)


# ======================================================================
#
#                             C O M B I N E
#
# ======================================================================
#
# Multi-operand ops whose first argument is a *sequence* of tensors. Axis
# names are reconciled positionally across the operands: an axis keeps the
# unique non-`None` name its operands agree on, and is unnamed on conflict
# (a stricter conflict policy is left to the broadcasting-by-name work).


def _operand_axis_names(tensors: tx.Sequence) -> list:
    """The axis names of each operand (all-`None` for a plain tensor)."""
    return [_names_of(t) for t in tensors]


def _reconcile_axis_names(all_names: list, ndim: int) -> tuple:
    """Per-axis reconciled name: the agreed non-`None` name, else `None`."""
    reconciled = []
    for axis in range(ndim):
        distinct = {names[axis] for names in all_names} - {None}
        reconciled.append(distinct.pop() if len(distinct) == 1 else None)
    return tuple(reconciled)


@NamedTensor.overrides(_torch_func("cat"))
def _(tensors: tx.Sequence, dim: int | str = 0, **kwargs) -> NamedTensor:
    tensors = list(tensors)
    ref = tensors[0]
    dim = _resolve_axis(ref.names, dim) % ref.ndim
    result = torch.cat(tensors, dim, **kwargs)
    names = _reconcile_axis_names(_operand_axis_names(tensors), ref.ndim)
    cat_name = names[dim]
    coords = {}
    for pos, name in enumerate(names):
        if name is None:
            continue
        parts = [_coords_of(t).get(name) for t in tensors]
        if pos == dim:
            # concatenate the labels of the axis we join along
            if all(p is not None for p in parts):
                coords[name] = tuple(x for part in parts for x in part)
        elif parts[0] is not None and all(p == parts[0] for p in parts):
            # a non-join axis keeps its labels only if the operands agree
            coords[name] = parts[0]
    del cat_name
    return _carry(ref, result, _axis_names=names, _coords=coords)


@NamedTensor.overrides(_torch_func("stack"))
def _(tensors: tx.Sequence, dim: int = 0, **kwargs) -> NamedTensor:
    tensors = list(tensors)
    ref = tensors[0]
    out_ndim = ref.ndim + 1
    dim %= out_ndim
    result = torch.stack(tensors, dim, **kwargs)
    reconciled = _reconcile_axis_names(_operand_axis_names(tensors), ref.ndim)
    # A brand-new (unnamed) axis is inserted at `dim`.
    names = reconciled[:dim] + (None,) + reconciled[dim:]
    # Existing axes keep their name and size; keep the labels the operands
    # agree on.
    coords = {}
    for name in names:
        if name is None:
            continue
        parts = [_coords_of(t).get(name) for t in tensors]
        if parts[0] is not None and all(p == parts[0] for p in parts):
            coords[name] = parts[0]
    return _carry(ref, result, _axis_names=names, _coords=coords)


# ---- matrix multiplication ------------------------------------------------
#
# `matmul` / `mm` / `bmm` (and the `@` operator, which dispatches as `matmul`)
# follow torch's broadcasting rules: the contracted axes vanish, the batch
# axes broadcast, and the result's trailing axes are `(a[-2], b[-1])`.


def _names_of(tensor: tx.Any) -> tuple:
    """
    Axis names of a tensor: its `names` if a `NamedTensor`, all-`None` for a
    plain tensor, and `()` for a non-tensor (e.g. a Python scalar operand).
    """
    if isinstance(tensor, NamedTensor):
        return tensor.names
    if isinstance(tensor, Tensor):
        return (None,) * tensor.ndim
    return ()


def _broadcast_batch_names(x: tuple, y: tuple) -> tuple:
    """Reconcile two batch-name tuples under right-aligned broadcasting."""
    width = max(len(x), len(y))
    x = (None,) * (width - len(x)) + tuple(x)
    y = (None,) * (width - len(y)) + tuple(y)
    reconciled = []
    for xn, yn in zip(x, y):
        distinct = {xn, yn} - {None}
        reconciled.append(distinct.pop() if len(distinct) == 1 else None)
    return tuple(reconciled)


def _matmul_names(a: tuple, b: tuple) -> tuple:
    """Result axis names for `matmul(a, b)` given each operand's names."""
    na, nb = len(a), len(b)
    if na == 1 and nb == 1:
        return ()  # dot product -> scalar
    if na == 1:  # [k] @ [..., k, n] -> [..., n]
        return _broadcast_batch_names((), b[:-2]) + (b[-1],)
    if nb == 1:  # [..., m, k] @ [k] -> [..., m]
        return _broadcast_batch_names(a[:-2], ()) + (a[-2],)
    return _broadcast_batch_names(a[:-2], b[:-2]) + (a[-2], b[-1])


def _make_matmul(name: str) -> None:
    """Register a name-aware override for a matrix-multiplication op."""
    base = _torch_func(name)

    def _matmul(input: tx.Any, other: tx.Any, **kwargs) -> tx.Any:
        result = base(input, other, **kwargs)
        ref = input if isinstance(input, NamedTensor) else other
        # The contraction invalidates the coordinate layout.
        return _carry(
            ref,
            result,
            _axis_names=_matmul_names(_names_of(input), _names_of(other)),
            _coords={},
        )

    registered = NamedTensor.overrides(base)(_matmul)
    # The `@` operator dispatches with the *bound method* `Tensor.matmul`,
    # a different callable than the function `torch.matmul`, so register the
    # method too (when it exists and differs) or `a @ b` would miss it.
    method = getattr(Tensor, name, None)
    if base is not None and method is not None and method is not base:
        NamedTensor._OVERRIDES[method] = registered


for _matmul_name in ("matmul", "mm", "bmm"):
    _make_matmul(_matmul_name)


# ======================================================================
#
#                     G A T H E R   /   S C A T T E R
#
# ======================================================================


@NamedTensor.overrides(_torch_func("index_select"))
def _(input: NamedTensor, dim: int | str, index: Tensor) -> tx.Any:
    dim = _resolve_axis(input.names, dim) % input.ndim
    result = Tensor.index_select(input, dim, index)
    # Rank is unchanged; only the selected axis' labels are re-sliced.
    coords = dict(input.coords)
    name = input.names[dim]
    if name in coords:
        coords[name] = tuple(_slice_labels(coords[name], index))
    return _carry(input, result, _coords=coords)


@NamedTensor.overrides(_torch_func("gather"))
def _(input: NamedTensor, dim: int | str, index: Tensor, **kwargs) -> tx.Any:
    dim = _resolve_axis(input.names, dim) % input.ndim
    result = torch.gather(input, dim, index, **kwargs)
    # Rank (and each axis' name) is preserved; the gathered positions change
    # per-slice, so the gathered axis' labels are dropped.
    coords = dict(input.coords)
    coords.pop(input.names[dim], None)
    return _carry(input, result, _coords=coords)


@NamedTensor.overrides(_torch_func("take_along_dim"))
def _(
    input: NamedTensor, indices: Tensor, dim: int | str = None, **kwargs
) -> tx.Any:
    result = torch.take_along_dim(
        input, indices, _resolve_axis(input.names, dim), **kwargs
    )
    coords = dict(input.coords)
    if dim is not None:
        coords.pop(
            input.names[_resolve_axis(input.names, dim) % input.ndim], None
        )
    else:
        coords = {}
    return _carry(input, result, _coords=coords)


@NamedTensor.overrides(_torch_func("scatter"))
def _(
    input: NamedTensor, dim: int | str, index: Tensor, *args, **kwargs
) -> tx.Any:
    dim = _resolve_axis(input.names, dim)
    result = torch.scatter(input, dim, index, *args, **kwargs)
    # Positions and sizes are unchanged, so names and coordinates survive.
    return _carry(input, result)


@NamedTensor.overrides(_torch_func("scatter_add"))
def _(
    input: NamedTensor, dim: int | str, index: Tensor, src: Tensor, **kwargs
) -> tx.Any:
    dim = _resolve_axis(input.names, dim)
    result = torch.scatter_add(input, dim, index, src, **kwargs)
    return _carry(input, result)


@NamedTensor.overrides(_torch_func("where"))
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
        (t for t in (condition, x, y) if isinstance(t, NamedTensor)),
        condition,
    )
    # Reconciling coordinates across broadcast operands is out of scope; drop.
    return _carry(ref, result, _axis_names=names, _coords={})


@NamedTensor.overrides(_torch_func("masked_select"))
def _(input: NamedTensor, mask: Tensor, **kwargs) -> tx.Any:
    result = torch.masked_select(input, mask, **kwargs)
    ref = input if isinstance(input, NamedTensor) else mask
    # The result is 1-D and its length is data-dependent: a single unnamed
    # axis, and no coordinates.
    return _carry(ref, result, _axis_names=(None,), _coords={})


# ======================================================================
#
#             C O N V E N I E N C E   S P E C I A L I Z A T I O N S
#
# ======================================================================


class NamedVector(NamedTensor):
    """
    A vector with a single labelled **channel** axis.

    Convenience over `NamedTensor`: names one axis (default `"channel"`) and
    labels it. `x.channels` reads those labels; `x.<label>` and
    `x.sel(channel=...)` select by them.
    """

    _CHANNEL = "channel"

    def __new__(cls, *args, **kwargs) -> tx.Self:
        for key in ("channels", "channel_dim"):
            kwargs.pop(key, None)
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        data: Tensor,
        *,
        channels: ArgLabelsT = (...,),
        channel_dim: int = -1,
        **kwargs,
    ) -> None:
        super().__init__(data, **kwargs)
        names = list(self.names)
        names[channel_dim % self.ndim] = self._CHANNEL
        self.names = tuple(names)
        self.coords = {self._CHANNEL: channels}

    @property
    def channels(self) -> LabelsT | None:
        """The labels of the channel axis (`None` if it was dropped)."""
        return self.coords.get(self._CHANNEL)

    @channels.setter
    def channels(self, value: ArgLabelsT) -> None:
        coords = dict(self.coords)
        coords[self._CHANNEL] = value
        self.coords = coords


class NamedMatrix(NamedTensor):
    """
    A matrix with two labelled axes, `"row"` and `"col"`.

    Convenience over `NamedTensor`, analogous to `NamedVector`.
    """

    _ROW, _COL = "row", "col"

    def __new__(cls, *args, **kwargs) -> tx.Self:
        for key in ("channels", "channel_dims"):
            kwargs.pop(key, None)
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        data: Tensor,
        *,
        channels: tuple[ArgLabelsT, ArgLabelsT] = ((...,), (...,)),
        channel_dims: tuple[int, int] = (-2, -1),
        **kwargs,
    ) -> None:
        super().__init__(data, **kwargs)
        names = list(self.names)
        d0, d1 = (d % self.ndim for d in channel_dims)
        names[d0], names[d1] = self._ROW, self._COL
        self.names = tuple(names)
        self.coords = {self._ROW: channels[0], self._COL: channels[1]}

    @property
    def channels(self) -> tuple[LabelsT | None, LabelsT | None]:
        """The `(row, col)` labels."""
        coords = self.coords
        return (coords.get(self._ROW), coords.get(self._COL))

    @channels.setter
    def channels(self, value: tuple[ArgLabelsT, ArgLabelsT]) -> None:
        coords = dict(self.coords)
        coords[self._ROW], coords[self._COL] = value
        self.coords = coords
