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
from fiery.xtensor import _arrayutils as arrayutils
from fiery.xtensor._arrayutils import SmartSlicerT, _SmartSlicerT
from fiery.xtensor._compat import EllipsisType
from fiery.xtensor._compat import no_dispatch as _no_dispatch
from fiery.xtensor._compat import torch_func as _torch_func
from fiery.xtensor._options import get_option as _get_option

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

AxisMetaT = tx.Mapping[str, tx.Any]
"""Extra axis-descriptor fields (OME-NGFF): `type`/`unit`/`orientation`."""

AxisT = tx.Union[str, None, tx.Mapping[str, tx.Any]]
"""One axis as given: a bare name, `None`, or a descriptor dict + `name`."""


def _validate_orientation(orientation: tx.Any) -> None:
    """An `orientation` descriptor field must have the form ``{a}-to-{b}``."""
    if not isinstance(orientation, str) or "-to-" not in orientation:
        raise ValueError(
            "orientation must have the form '{a}-to-{b}', got "
            f"{orientation!r}"
        )


def _flip_orientation(orientation: str) -> str:
    """Reverse ``{a}-to-{b}`` into ``{b}-to-{a}`` (what a flip does)."""
    a, _, b = orientation.partition("-to-")
    return b + "-to-" + a


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

    Wraps [`_resolve_axis`][fiery.xtensor._tensors._resolve_axis]: a
    single specifier is resolved directly; a `tuple`/`list` is resolved
    element-wise (keeping its container type); anything else passes through.
    """
    if isinstance(dim, str):
        return _resolve_axis(names, dim)
    if isinstance(dim, (tuple, list)):
        return type(dim)(_resolve_axis(names, d) for d in dim)
    return dim


def _match_axes(input: XTensor, query: tx.Mapping) -> list:
    """
    Positions whose axis **descriptor** matches every key/value in `query`
    (a descriptor query like `{"type": "space"}`), in current axis order.
    """
    return [
        i
        for i, axis in enumerate(input.axes)
        if axis is not None and all(axis.get(k) == v for k, v in query.items())
    ]


def _query_positions(input: XTensor, dim: tx.Any) -> list:
    """
    Resolve a dim spec to a flat list of positions. Accepts an `int`, a name
    (`str`), a **descriptor query** (a dict matching zero-or-more axes), or a
    sequence mixing those. A query expands to *all* matching axes, in order.
    """
    if isinstance(dim, dict):
        return _match_axes(input, dim)
    if isinstance(dim, (tuple, list)):
        positions = []
        for one in dim:
            positions.extend(_query_positions(input, one))
        return positions
    return [_resolve_axis(input.names, dim) % input.ndim]


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
        [`torch_func`][fiery.xtensor._compat.torch_func] for an op
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


class XTensor(ExtendedTensor):
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
    - **Axis descriptors** may enrich a name with extra (OME-NGFF-style)
      fields -- `type`, `unit`, `orientation` -- passed as a dict in place of
      a bare name (`{"name": "x", "type": "space"}`). `names` stays the
      ergonomic view (bare names); `axes` returns the full descriptors. The
      extra fields live in `_axis_meta`, keyed by dimension name, so they
      follow the dimension like coordinates do.

    Select by label with `sel`, by integer position with `isel`, or reach a
    single label by attribute (`x.red`).
    """

    _ATTRS = {"_axis_names", "_coords", "_axis_meta"}

    def __new__(cls, *args, **kwargs) -> tx.Self:
        # NOTE: remove arguments that `Tensor.__new__` does not support.
        kwargs.pop("names", None)
        kwargs.pop("coords", None)
        kwargs.pop("axes", None)
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
        # `axes=` is an alias for `names=` that reads better for descriptors;
        # both accept bare names, `None`, or descriptor dicts.
        names = kwargs.pop("axes", None)
        if names is None:
            names = kwargs.pop("names", None)
        else:
            kwargs.pop("names", None)
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
    def names(self, value: tx.Optional[tx.Sequence[AxisT]]) -> None:
        if value is None:
            self.__dict__.pop("_axis_names", None)
            self.__dict__.pop("_axis_meta", None)
            return
        value = tuple(value)
        if len(value) != self.ndim:
            raise ValueError(
                f"Expected {self.ndim} names, got {len(value)}: {value}"
            )
        # Each item may be a bare name / `None`, or a descriptor dict carrying
        # extra fields (`type`/`unit`/`orientation`) alongside its `name`.
        names, meta = [], {}
        has_descriptor = False
        for item in value:
            if isinstance(item, dict):
                has_descriptor = True
                if "name" not in item:
                    raise ValueError(
                        f"axis descriptor must have a 'name': {item!r}"
                    )
                name = item["name"]
                extra = {k: v for k, v in item.items() if k != "name"}
                if "orientation" in extra:
                    _validate_orientation(extra["orientation"])
                names.append(name)
                if extra and name is not None:
                    meta[name] = extra
            else:
                names.append(item)
        self._axis_names = tuple(names)
        # Only touch `_axis_meta` when descriptors were actually supplied, so a
        # plain `x.names = (...)` keeps any existing metadata (the getter hides
        # entries whose dim is no longer present).
        if has_descriptor:
            self._axis_meta = meta

    # -- axis descriptors --------------------------------------------------

    @property
    def axes(self) -> tuple[dict | None, ...]:
        """
        Each axis as a descriptor dict ``{"name": ..., **extra}`` (or `None`
        for an unnamed axis). The extra OME-NGFF-style fields (`type`, `unit`,
        `orientation`) come from `_axis_meta`, keyed by dimension name.
        """
        meta = self._valid_axis_meta()
        return tuple(
            None if name is None else {"name": name, **meta.get(name, {})}
            for name in self.names
        )

    def _valid_axis_meta(self) -> dict[str, dict]:
        """`_axis_meta` filtered to dimensions still named on this tensor."""
        stored = self.__dict__.get("_axis_meta") or {}
        names = self.names
        return {name: extra for name, extra in stored.items() if name in names}

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

    def _remap_named(self, attr: str, new_names: tuple) -> dict:
        """Re-key a `{dim name: ...}` dict attribute from current names."""
        stored = self.__dict__.get(attr) or {}
        if not stored:
            return {}
        remapped = {}
        for old, new in zip(self.names, new_names):
            if old in stored and new is not None:
                remapped[new] = stored[old]
        return remapped

    def _remap_coords(self, new_names: tuple) -> dict:
        """Coordinates re-keyed from the current names to `new_names`."""
        return self._remap_named("_coords", new_names)

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
        out._axis_meta = self._remap_named("_axis_meta", new_names)
        out._axis_names = new_names
        return out

    def rename_(self, *names: str | None, **rename_map: str) -> tx.Self:
        """In-place variant of `rename`."""
        new_names = self._resolve_rename(names, rename_map)
        coords = self._remap_coords(new_names)
        meta = self._remap_named("_axis_meta", new_names)
        self._coords = coords
        self._axis_meta = meta
        self._axis_names = new_names
        return self

    # -- indexing / selection ---------------------------------------------

    def _resolve_label_slicer(self, slicer: SmartSlicerT) -> SmartSlicerT:
        # Resolve any *positional* coordinate-label index to an integer (or a
        # list of integers) against the axis it sits on, so `x[..., "y", "z"]`
        # addresses the last two axes by label. A bare `str` or list-of-`str`
        # element is a label index; everything else is left untouched.
        items = slicer if isinstance(slicer, tuple) else (slicer,)
        if not any(_is_label_index(v) for v in items):
            return slicer
        # Axes consumed by the explicit (non-newaxis, non-ellipsis) items; the
        # ellipsis, if any, fills the remaining axes in the middle.
        consumed = arrayutils._count_input_axes(items)
        gap = self.ndim - consumed
        resolved, axis = [], 0
        for value in items:
            if value is ...:
                axis += gap
                resolved.append(value)
            elif value is None:
                resolved.append(value)
            elif _is_label_index(value):
                resolved.append(self._label_to_index(axis, value))
                axis += 1
            else:
                resolved.append(value)
                axis += 1
        return tuple(resolved)

    def _label_to_index(self, axis: int, value: tx.Any) -> tx.Any:
        """One label -> its integer position on `axis` (a list of labels ->
        a list of positions); raises if the axis is unlabelled or the label
        is absent."""
        name = self.names[axis % self.ndim]
        labels = self.coords.get(name) if name is not None else None
        if labels is None:
            raise KeyError(
                f"axis {name!r} has no coordinates for label {value!r}"
            )

        def _one(label: str) -> int:
            try:
                return labels.index(label)
            except ValueError:
                raise KeyError(
                    f"no label {label!r} on axis {name!r}"
                ) from None

        if isinstance(value, str):
            return _one(value)
        return [_one(label) for label in value]

    def __getitem__(self, slicer: SmartSlicerT) -> tx.Self:
        # A positional coordinate label (`x[..., "y"]`) resolves to an integer
        # index against the axis it indexes before ordinary indexing runs.
        slicer = self._resolve_label_slicer(slicer)
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

    def _dims_with_label(self, label: str) -> list:
        """Named dims whose coordinates include `label` (usually 0 or 1)."""
        return [dim for dim, labels in self.coords.items() if label in labels]

    def __getattr__(self, name: str) -> tx.Self:
        # Only consulted when normal attribute lookup fails, so real methods
        # and attributes always win. Private / dunder names are never labels.
        if name.startswith("_"):
            raise AttributeError(name)
        hits = self._dims_with_label(name)
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

    def align_as(self, other: XTensor) -> tx.Self:
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
    if isinstance(tensor, XTensor):
        return tensor.coords
    return {}


def _coords_for(input: XTensor, result_names: tuple) -> dict:
    """
    Keep only the coordinates whose dimension survives (by name) into
    `result_names`. Merged / split / removed axes lose their name and so drop
    their coordinates automatically.
    """
    kept = {name for name in result_names if name is not None}
    return {k: v for k, v in _coords_of(input).items() if k in kept}


def _is_label_index(value: tx.Any) -> bool:
    """
    Whether a slicer element is a **coordinate label** index: a bare `str`, or
    a non-empty **list** of `str` (an advanced index by label). A *tuple* is
    not, so a top-level `x["y", "z"]` stays one label per axis rather than a
    single advanced index. Plain ints, slices, `None`, ellipsis and tensors
    are not labels either.
    """
    if isinstance(value, str):
        return True
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(item, str) for item in value)
    )


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


@XTensor.overrides(_torch_func("permute"))
def _(input: XTensor, *dims: int | str | tuple) -> XTensor:
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = tuple(dims[0])
    names = input.names
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


def _prepend_axes_meta(input: XTensor, n_new: int) -> dict:
    """`_carry` overrides for an op that prepends `n_new` unnamed axes."""
    # Existing axes keep their name and size, so their coordinates stay valid.
    return {"_axis_names": (None,) * n_new + input.names}


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
    return _carry(
        input, result, **_prepend_axes_meta(input, result.ndim - input.ndim)
    )


@XTensor.overrides(_torch_func("broadcast_to"))
def _(input: XTensor, shape: tx.Sequence) -> XTensor:
    result = Tensor.broadcast_to(input, shape)
    return _carry(
        input, result, **_prepend_axes_meta(input, result.ndim - input.ndim)
    )


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


def _resolve_reduce_dim(input: XTensor, dim: tx.Any) -> tx.Any:
    """
    Resolve a reduction's `dim`, expanding any **descriptor query** to the axes
    it matches. A query hitting a single axis collapses to a bare `int` (so
    single-`dim`-only reducers like `prod`/`argmax` keep working); one hitting
    several yields a list of positions. Non-query specs pass through
    [`_resolve_dims`][fiery.xtensor._tensors._resolve_dims] unchanged.
    """
    has_query = isinstance(dim, dict) or (
        isinstance(dim, (tuple, list))
        and any(isinstance(d, dict) for d in dim)
    )
    if not has_query:
        return _resolve_dims(input.names, dim)
    positions = _query_positions(input, dim)
    return positions[0] if len(positions) == 1 else positions


def _reduce_names(input: XTensor, result: tx.Any, dim: tx.Any) -> tx.Any:
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

    def _reduction(input: XTensor, *args, **kwargs) -> tx.Any:
        # Resolve a name/query for `dim` (positional arg 0 or keyword) and
        # remember the (resolved) value so the output names can be computed.
        if "dim" in kwargs:
            dim = kwargs["dim"] = _resolve_reduce_dim(input, kwargs["dim"])
        elif args:
            dim = _resolve_reduce_dim(input, args[0])
            args = (dim,) + args[1:]
        else:
            dim = None
        return _reduce_names(input, base(input, *args, **kwargs), dim)

    # `overrides(None)` is a no-op, so ops missing from this torch are skipped.
    XTensor.overrides(base)(_reduction)


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


# ---- irregular signatures & (values, indices) reducers --------------------


def _rebuild(namedtuple: tx.Any, fn: tx.Callable) -> tx.Any:
    """Apply `fn` to every member of a torch return-type namedtuple."""
    return type(namedtuple)(tuple(fn(member) for member in namedtuple))


def _make_std_var(name: str) -> None:
    """`std` / `var`: `dim` is the first positional, but a bool there is
    `unbiased`, not a dim."""
    base = _torch_func(name)

    def _op(input: XTensor, *args, **kwargs) -> tx.Any:
        names = input.names
        dim = None
        if "dim" in kwargs:
            dim = kwargs["dim"] = _resolve_dims(names, kwargs["dim"])
        elif args and not isinstance(args[0], bool):
            dim = _resolve_dims(names, args[0])
            args = (dim,) + args[1:]
        return _reduce_names(input, base(input, *args, **kwargs), dim)

    XTensor.overrides(base)(_op)


for _std_var_name in ("std", "var"):
    _make_std_var(_std_var_name)


@XTensor.overrides(_torch_func("norm"))
def _(input: XTensor, *args, **kwargs) -> tx.Any:
    # `norm(input, p, dim, keepdim, ...)`: `dim` is the *second* positional
    # (after `p`) or a keyword.
    names = input.names
    dim = None
    if "dim" in kwargs:
        dim = kwargs["dim"] = _resolve_dims(names, kwargs["dim"])
    elif len(args) >= 2:
        dim = _resolve_dims(names, args[1])
        args = (args[0], dim) + args[2:]
    return _reduce_names(input, torch.norm(input, *args, **kwargs), dim)


def _make_minmax(name: str) -> None:
    """`max` / `min`: overloaded — `x.max()` (scalar), `x.max(dim)` (a
    `(values, indices)` namedtuple that reduces `dim`), and `torch.max(a, b)`
    (elementwise)."""
    base = _torch_func(name)

    def _op(input: XTensor, *args, **kwargs) -> tx.Any:
        names = input.names
        if args and isinstance(args[0], Tensor):
            # elementwise max/min(a, b): reconcile names, drop coordinates
            result = base(input, *args, **kwargs)
            out = _broadcast_batch_names(names, _names_of(args[0]))
            return _carry(input, result, _axis_names=out, _coords={})
        dim = None
        if "dim" in kwargs:
            dim = kwargs["dim"] = _resolve_dims(names, kwargs["dim"])
        elif args:
            dim = _resolve_dims(names, args[0])
            args = (dim,) + args[1:]
        result = base(input, *args, **kwargs)
        if isinstance(result, Tensor):
            return _reduce_names(input, result, dim)  # scalar (no dim)
        return _rebuild(result, lambda m: _reduce_names(input, m, dim))

    XTensor.overrides(base)(_op)


for _minmax_name in ("max", "min"):
    _make_minmax(_minmax_name)


@XTensor.overrides(_torch_func("median"))
def _(input: XTensor, *args, **kwargs) -> tx.Any:
    names = input.names
    dim = None
    if "dim" in kwargs:
        dim = kwargs["dim"] = _resolve_dims(names, kwargs["dim"])
    elif args:
        dim = _resolve_dims(names, args[0])
        args = (dim,) + args[1:]
    result = torch.median(input, *args, **kwargs)
    if isinstance(result, Tensor):
        return _reduce_names(input, result, dim)  # median(x) -> scalar
    return _rebuild(result, lambda m: _reduce_names(input, m, dim))


def _make_dim_default_reduction(name: str, dim_pos: int) -> None:
    """`mode` / `kthvalue`: always return a `(values, indices)` namedtuple
    reducing one dim (default -1). `dim_pos` is where `dim` sits positionally
    (0 for `mode`, 1 for `kthvalue`, which takes `k` first)."""
    base = _torch_func(name)

    def _op(input: XTensor, *args, **kwargs) -> tx.Any:
        names = input.names
        if "dim" in kwargs:
            dim = kwargs["dim"] = _resolve_dims(names, kwargs["dim"])
        elif len(args) > dim_pos:
            dim = _resolve_dims(names, args[dim_pos])
            args = args[:dim_pos] + (dim,) + args[dim_pos + 1 :]
        else:
            dim = -1  # torch's default reduced dim
        result = base(input, *args, **kwargs)
        return _rebuild(result, lambda m: _reduce_names(input, m, dim))

    XTensor.overrides(base)(_op)


_make_dim_default_reduction("mode", 0)
_make_dim_default_reduction("kthvalue", 1)


def _make_sorting(name: str, k_arg: bool) -> None:
    """`sort` (rank- and size-preserving) / `topk` (keeps rank, resizes the
    sorted dim). Both return a `(values, indices)` namedtuple; the sorted dim's
    labels no longer match positions, so its coordinates are dropped."""
    base = _torch_func(name)
    dim_pos = 1 if k_arg else 0

    def _op(input: XTensor, *args, **kwargs) -> tx.Any:
        names = input.names
        if "dim" in kwargs:
            dim = kwargs["dim"] = _resolve_axis(names, kwargs["dim"])
        elif len(args) > dim_pos:
            dim = _resolve_axis(names, args[dim_pos])
            args = args[:dim_pos] + (dim,) + args[dim_pos + 1 :]
        else:
            dim = -1
        result = base(input, *args, **kwargs)
        coords = dict(input.coords)
        coords.pop(names[dim % input.ndim], None)
        return _rebuild(result, lambda m: _carry(input, m, _coords=coords))

    XTensor.overrides(base)(_op)


_make_sorting("sort", k_arg=False)
_make_sorting("topk", k_arg=True)


# ======================================================================
#
#                               S C A N S
#
# ======================================================================
#
# Unlike REDUCTIONS above, these ops are dimension-*preserving*: rank, sizes,
# names and coordinates are all unchanged, so `_carry(input, ...)` after
# resolving a name given for `dim` is all that's needed.


def _make_scan(name: str) -> None:
    """
    Register a name-aware override for a dim-preserving scan/activation op
    (`cumsum`, `softmax`, ...): `dim` is the op's first positional argument
    (or a keyword), and it may be given as a name (method form only, see
    `_resolve_axis`). Rank/size/names/coords are unchanged, so the result
    just needs `input`'s metadata carried onto it.
    """
    base = _torch_func(name)

    def _scan(input: XTensor, *args, **kwargs) -> XTensor:
        names = input.names
        if "dim" in kwargs:
            kwargs["dim"] = _resolve_axis(names, kwargs["dim"])
        elif args:
            args = (_resolve_axis(names, args[0]),) + args[1:]
        return _carry(input, base(input, *args, **kwargs))

    # `overrides(None)` is a no-op, so ops missing from this torch are skipped
    # (e.g. `logcumsumexp`, added in torch 1.9).
    XTensor.overrides(base)(_scan)


# `dim` is the first positional for each; `softmax`/`log_softmax` also take a
# keyword-only `dtype`, same as `cumsum`/`cumprod`/`logcumsumexp`.
_SCANS = ("cumsum", "cumprod", "softmax", "log_softmax", "logcumsumexp")
for _scan_name in _SCANS:
    _make_scan(_scan_name)


def _make_cum_extremum(name: str) -> None:
    """
    Register a name-aware override for `cummax` / `cummin`: a
    `(values, indices)` namedtuple, dim-preserving like the scans above, so
    `input`'s names+coords are carried onto *both* members via `_rebuild`.
    """
    base = _torch_func(name)

    def _op(input: XTensor, *args, **kwargs) -> tx.Any:
        names = input.names
        if "dim" in kwargs:
            kwargs["dim"] = _resolve_axis(names, kwargs["dim"])
        elif args:
            args = (_resolve_axis(names, args[0]),) + args[1:]
        result = base(input, *args, **kwargs)
        return _rebuild(result, lambda m: _carry(input, m))

    XTensor.overrides(base)(_op)


for _cum_extremum_name in ("cummax", "cummin"):
    _make_cum_extremum(_cum_extremum_name)


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
    coords = dict(input.coords)
    for name in flipped:
        if name in coords:
            coords[name] = tuple(reversed(coords[name]))
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


@XTensor.overrides(_torch_func("cat"))
def _(tensors: tx.Sequence, dim: int | str = 0, **kwargs) -> XTensor:
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
    meta = _merge_axis_meta(tensors, names)
    return _carry(
        ref, result, _axis_names=names, _coords=coords, _axis_meta=meta
    )


@XTensor.overrides(_torch_func("stack"))
def _(tensors: tx.Sequence, dim: int = 0, **kwargs) -> XTensor:
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
    meta = _merge_axis_meta(tensors, names)
    return _carry(
        ref, result, _axis_names=names, _coords=coords, _axis_meta=meta
    )


# ---- promoting stacks (hstack / vstack / dstack) ---------------------------
#
# Unlike `cat`/`stack`, these promote lower-rank operands first (`hstack`
# treats 1-D tensors specially; `vstack`/`dstack` reshape via
# `atleast_2d`/`atleast_3d`), which can shift a promoted operand's axes
# relative to the joined result. Positional name reconciliation is only
# sound when *every* operand already has the result's rank -- i.e. nothing
# was promoted -- so that is the only case handled; otherwise the result is
# left fully unnamed. Coordinate labels are always dropped: even in the
# aligned case, the join axis' positions are data-dependent per operand and
# the promotion rules make general label tracking unsafe.


def _promoted_stack_names(tensors: tx.Sequence, out_ndim: int) -> tuple:
    """
    Positional axis-name reconciliation for `hstack`/`vstack`/`dstack`.

    Reuses `_operand_axis_names` / `_reconcile_axis_names` (the same
    machinery `cat`/`stack` use), but only when every operand already has
    `out_ndim` dimensions -- i.e. the op didn't need to promote any operand's
    rank to join them, so each axis position lines up across operands. When
    ranks differ, promotion may have inserted axes ahead of an operand's own,
    so positional alignment can't be trusted: names are dropped (all `None`).
    """
    if all(getattr(t, "ndim", None) == out_ndim for t in tensors):
        return _reconcile_axis_names(_operand_axis_names(tensors), out_ndim)
    return (None,) * out_ndim


def _make_promoting_stack(name: str) -> None:
    """Register a conservative name-aware override for a promoting stack op
    (`hstack`/`vstack`/`dstack`): reconcile names positionally when ranks
    already align, else leave the result unnamed; coordinates always drop."""
    base = _torch_func(name)
    if base is None:
        return

    def _stack(tensors: tx.Sequence, **kwargs) -> tx.Any:
        tensors = list(tensors)
        ref = tensors[0]
        result = base(tensors, **kwargs)
        names = _promoted_stack_names(tensors, result.ndim)
        meta = _merge_axis_meta(tensors, names)
        return _carry(
            ref, result, _axis_names=names, _coords={}, _axis_meta=meta
        )

    XTensor.overrides(base)(_stack)


for _promoting_stack_name in ("hstack", "vstack", "dstack"):
    _make_promoting_stack(_promoting_stack_name)


# ---- matrix multiplication ------------------------------------------------
#
# `matmul` / `mm` / `bmm` (and the `@` operator, which dispatches as `matmul`)
# follow torch's broadcasting rules: the contracted axes vanish, the batch
# axes broadcast, and the result's trailing axes are `(a[-2], b[-1])`.


def _names_of(tensor: tx.Any) -> tuple:
    """
    Axis names of a tensor: its `names` if a `XTensor`, all-`None` for a
    plain tensor, and `()` for a non-tensor (e.g. a Python scalar operand).
    """
    if isinstance(tensor, XTensor):
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
        ref = input if isinstance(input, XTensor) else other
        names = _matmul_names(_names_of(input), _names_of(other))
        # The contraction invalidates the coordinate layout; surviving axes
        # keep their (merged) descriptors.
        return _carry(
            ref,
            result,
            _axis_names=names,
            _coords={},
            _axis_meta=_merge_axis_meta((input, other), names),
        )

    registered = XTensor.overrides(base)(_matmul)
    # The `@` operator dispatches with the *bound method* `Tensor.matmul`,
    # a different callable than the function `torch.matmul`, so register the
    # method too (when it exists and differs) or `a @ b` would miss it.
    method = getattr(Tensor, name, None)
    if base is not None and method is not None and method is not base:
        XTensor._OVERRIDES[method] = registered


for _matmul_name in ("matmul", "mm", "bmm"):
    _make_matmul(_matmul_name)


# ---- einsum / tensordot ----------------------------------------------------
#
# Both contract axes across operands in a way that's driven by an equation
# string (`einsum`) or explicit axis positions (`tensordot`), rather than by
# position/broadcasting like `matmul`. Neither has a `Tensor` method form, so
# (unlike `_make_matmul`) only the free function needs registering. A
# contraction invalidates the coordinate layout, so both drop coords.


def _einsum_output_names(
    equation: str, operand_names: tx.Sequence[tuple], out_ndim: int
) -> tuple:
    """
    Best-effort output axis names for `torch.einsum(equation, *operands)`.

    Parses both the explicit (`"ij,jk->ik"`) and implicit (no `->`; the
    output is whichever subscripts appear exactly once across all input
    operands, sorted alphabetically) forms. For each output subscript, the
    names of every operand axis bound to that subscript are reconciled via
    `_reconcile_axis_names` (unique non-`None` agreed name, else `None`).

    Falls back to an all-`None` tuple of length `out_ndim` for anything this
    simple parser can't confidently handle -- most notably an ellipsis
    (`"..."`, whose expanded rank depends on the operand shapes) -- so a
    name-aware `einsum` never raises where a plain `torch.einsum` would not.
    """
    fallback = (None,) * out_ndim
    if "." in equation:  # ellipsis ("...") -> width depends on operand shapes
        return fallback

    if "->" in equation:
        parts = equation.split("->")
        if len(parts) != 2:
            return fallback
        in_part, out_part = parts
    else:
        in_part, out_part = equation, None

    in_subscripts = [s.strip() for s in in_part.split(",")]
    if len(in_subscripts) != len(operand_names):
        return fallback
    for subscript, names in zip(in_subscripts, operand_names):
        if len(subscript) != len(names):
            return fallback
        if subscript and not subscript.isalpha():
            return fallback

    if out_part is None:
        counts: dict = {}
        for subscript in in_subscripts:
            for letter in subscript:
                counts[letter] = counts.get(letter, 0) + 1
        out_subscript = "".join(sorted(c for c, n in counts.items() if n == 1))
    else:
        out_subscript = out_part.strip()
        if out_subscript and not out_subscript.isalpha():
            return fallback

    if len(out_subscript) != out_ndim:
        return fallback

    names_by_letter: dict = {}
    for subscript, names in zip(in_subscripts, operand_names):
        for letter, name in zip(subscript, names):
            names_by_letter.setdefault(letter, []).append(name)

    output_names = []
    for letter in out_subscript:
        matches = names_by_letter.get(letter, [])
        reconciled = _reconcile_axis_names([(name,) for name in matches], 1)
        output_names.append(reconciled[0])
    return tuple(output_names)


def _einsum_operands(args: tuple) -> list:
    """
    The operand tensors, whether passed as varargs (`einsum(eq, a, b)`) or
    as the older single-list form (`einsum(eq, [a, b])`).
    """
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return list(args[0])
    return list(args)


@XTensor.overrides(_torch_func("einsum"))
def _(equation: str, *operands: tx.Any, **kwargs) -> tx.Any:
    flat = _einsum_operands(operands)
    result = torch.einsum(equation, *flat, **kwargs)
    ref = next((t for t in flat if isinstance(t, XTensor)), None)
    if ref is None:
        return result
    names = _einsum_output_names(
        equation, [_names_of(t) for t in flat], getattr(result, "ndim", 0)
    )
    meta = _merge_axis_meta(flat, names)
    return _carry(ref, result, _axis_names=names, _coords={}, _axis_meta=meta)


@XTensor.overrides(_torch_func("tensordot"))
def _(a: tx.Any, b: tx.Any, dims: tx.Any = 2, **kwargs) -> tx.Any:
    result = torch.tensordot(a, b, dims=dims, **kwargs)
    ref = a if isinstance(a, XTensor) else b
    a_names, b_names = _names_of(a), _names_of(b)
    if isinstance(dims, int):
        a_contracted = set(range(len(a_names) - dims, len(a_names)))
        b_contracted = set(range(dims))
    else:
        a_dims, b_dims = dims
        a_contracted = {d % len(a_names) for d in a_dims}
        b_contracted = {d % len(b_names) for d in b_dims}
    names = tuple(
        n for i, n in enumerate(a_names) if i not in a_contracted
    ) + tuple(n for i, n in enumerate(b_names) if i not in b_contracted)
    meta = _merge_axis_meta((a, b), names)
    return _carry(ref, result, _axis_names=names, _coords={}, _axis_meta=meta)


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
    coords = dict(input.coords)
    name = input.names[dim]
    if name in coords:
        coords[name] = tuple(_slice_labels(coords[name], index))
    return _carry(input, result, _coords=coords)


@XTensor.overrides(_torch_func("gather"))
def _(input: XTensor, dim: int | str, index: Tensor, **kwargs) -> tx.Any:
    dim = _resolve_axis(input.names, dim) % input.ndim
    result = torch.gather(input, dim, index, **kwargs)
    # Rank (and each axis' name) is preserved; the gathered positions change
    # per-slice, so the gathered axis' labels are dropped.
    coords = dict(input.coords)
    coords.pop(input.names[dim], None)
    return _carry(input, result, _coords=coords)


@XTensor.overrides(_torch_func("take_along_dim"))
def _(
    input: XTensor, indices: Tensor, dim: int | str = None, **kwargs
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


# ======================================================================
#
#                     P O I N T W I S E   ( B Y   N A M E )
#
# ======================================================================
#
# Binary/pointwise ops (`+`, `*`, comparisons, ...) combine names the
# xarray way: when **both** operands are fully-named `XTensor`s, their axes are
# aligned **by name** (union of dims, shared names broadcast together, axes
# transposed to match) rather than by position. Any unnamed axis (or a plain
# tensor / scalar operand) falls back to positional broadcasting.


def _reshape_to_order(x: XTensor, order: list) -> XTensor:
    """Permute/expand `x`'s named axes onto `order` (size-1 where absent)."""
    x_names = x.names
    present = [n for n in order if n in x_names]
    out = x.permute(*[x_names.index(n) for n in present])
    for pos, name in enumerate(order):
        if name not in x_names:
            out = out.unsqueeze(pos)
    return out


def _reindex_axis(x: XTensor, name: str, old: tuple, new: tuple) -> XTensor:
    """
    Select the positions of `x`'s `name` axis whose labels are `new` (a subset
    of `old`, in the wanted order) -- the reindex step of coordinate alignment.
    Operates on the tensor data only; the caller re-derives the metadata.
    """
    axis = x.names.index(name)
    index = torch.as_tensor(
        [old.index(label) for label in new], dtype=torch.long, device=x.device
    )
    return x.index_select(axis, index)


def _align_by_name(a: XTensor, b: XTensor) -> tuple:
    """
    Align two fully-named tensors by dim name; return `(a', b', names, coords)`
    ready for a positional (now name-matched) op.

    A shared dim that is **labelled on both operands** but whose labels differ
    is aligned xarray-style (`join="inner"`): both operands are reindexed to
    the intersection of their labels -- in `a`'s order -- before the op, so
    positions are matched by *label*, not by position. Identical label sets
    skip the reindex; a dim labelled on only one side keeps those labels.
    """
    a_names, b_names = a.names, b.names
    order = list(a_names) + [n for n in b_names if n not in a_names]
    coords = {}
    for name in order:
        ca, cb = a.coords.get(name), b.coords.get(name)
        if ca is not None and cb is not None and ca != cb:
            shared = set(cb)
            common = tuple(label for label in ca if label in shared)
            a = _reindex_axis(a, name, ca, common)
            b = _reindex_axis(b, name, cb, common)
            coords[name] = common
        elif ca is not None and cb is not None:  # identical labels
            coords[name] = ca
        elif ca is not None:
            coords[name] = ca
        elif cb is not None:
            coords[name] = cb
    return (
        _reshape_to_order(a, order),
        _reshape_to_order(b, order),
        tuple(order),
        coords,
    )


def _distinct(values: list) -> list:
    """The distinct `values`, order-preserving and tolerant of unhashables."""
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _merge_axis_meta(sources: tx.Sequence, result_names: tuple) -> dict:
    """
    Combine several operands' axis **descriptors** into one `_axis_meta` for a
    result whose dims are `result_names`, per the `combine_axes` option:

    - `"drop"` -- no descriptors on the result;
    - `"override"` -- the left-most operand's fields win on conflict;
    - `"strict"` -- a conflicting field raises `ValueError`;
    - `"drop_conflicts"` *(default)* -- union the axes, and for a shared dim
      keep the fields the operands agree on while dropping the ones that
      conflict (the rule coordinates already follow).

    A field present on only one operand is never a conflict; it is kept.
    """
    policy = _get_option("combine_axes")
    if policy == "drop":
        return {}
    wanted = {name for name in result_names if name is not None}
    # For each result dim, the extra-field dicts of the operands that name it.
    per_dim = {}
    for source in sources:
        if not isinstance(source, XTensor):
            continue
        meta = source._valid_axis_meta()
        for name in source.names:
            if name in wanted:
                per_dim.setdefault(name, []).append(meta.get(name, {}))
    merged = {}
    for name, dicts in per_dim.items():
        if policy == "override":
            extra = {}
            for one in reversed(dicts):  # left-most wins
                extra.update(one)
            if extra:
                merged[name] = extra
            continue
        extra = {}
        for key in {k for one in dicts for k in one}:
            present = [one[key] for one in dicts if key in one]
            distinct = _distinct(present)
            if len(distinct) == 1:
                extra[key] = distinct[0]
            elif policy == "strict":
                raise ValueError(
                    f"conflicting {key!r} for axis {name!r}: {distinct}"
                )
            # drop_conflicts: a conflicting field is simply omitted
        if extra:
            merged[name] = extra
    return merged


def _binary(a: tx.Any, b: tx.Any, base: tx.Callable, args, kwargs) -> tx.Any:
    a_named = isinstance(a, XTensor) and None not in a.names
    b_named = isinstance(b, XTensor) and None not in b.names
    if a_named and b_named:
        a2, b2, names, coords = _align_by_name(a, b)
        result = base(a2, b2, *args, **kwargs)
        meta = _merge_axis_meta((a, b), names)
        return _carry(
            a, result, _axis_names=names, _coords=coords, _axis_meta=meta
        )
    # positional fallback (unnamed axis, plain tensor, or scalar operand)
    result = base(a, b, *args, **kwargs)
    if not isinstance(result, Tensor):
        return result
    ref = a if isinstance(a, XTensor) else b
    names = _broadcast_batch_names(_names_of(a), _names_of(b))
    coords = _coords_of(ref) if result.ndim == getattr(ref, "ndim", -1) else {}
    meta = _merge_axis_meta((a, b), names)
    return _carry(
        ref, result, _axis_names=names, _coords=coords, _axis_meta=meta
    )


def _make_pointwise(name: str) -> None:
    """Register a broadcast-by-name override for a binary/pointwise op."""
    base = _torch_func(name)

    def _op(a: tx.Any, b: tx.Any, *args, **kwargs) -> tx.Any:
        return _binary(a, b, base, args, kwargs)

    registered = XTensor.overrides(base)(_op)
    # Operators (`a + b`, `a == b`, ...) dispatch with the bound method
    # `Tensor.<name>` -- a different callable than the function `torch.<name>`
    # -- so register both (as for `matmul`).
    method = getattr(Tensor, name, None)
    if base is not None and method is not None and method is not base:
        XTensor._OVERRIDES[method] = registered


# Elementwise ops whose result should align by name. `dim`-less, two-operand.
_POINTWISE = (
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "remainder",
    "floor_divide",
    "atan2",
    "hypot",
    "maximum",
    "minimum",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "logical_and",
    "logical_or",
    "logical_xor",
)
for _pointwise_name in _POINTWISE:
    _make_pointwise(_pointwise_name)


# ======================================================================
#
#             C O N V E N I E N C E   S P E C I A L I Z A T I O N S
#
# ======================================================================


class XVector(XTensor):
    """
    A vector with a single labelled **channel** axis.

    Convenience over `XTensor`: names one axis (default `"channel"`) and
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


class XMatrix(XTensor):
    """
    A matrix with two labelled axes, `"row"` and `"col"`.

    Convenience over `XTensor`, analogous to `XVector`.
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
