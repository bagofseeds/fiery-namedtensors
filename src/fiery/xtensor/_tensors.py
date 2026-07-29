# `from __future__ import annotations` keeps every annotation a lazy string,
# so only values *evaluated at runtime* -- the type aliases below -- must avoid
# PEP 585/604/695. All typing goes through `typing_extensions` (imported as
# `tx`), never abc/builtin subscription (e.g. `collections.abc.Sequence[...]`,
# which is not subscriptable before Python 3.9). `tx.Sequence` also works for
# the runtime isinstance checks.
from __future__ import annotations

# stdlib
import copy
import enum
import math
from functools import wraps

# dependencies
import torch
import typing_extensions as tx
from torch import Tensor

# internals
from fiery.xtensor import _arrayutils as arrayutils
from fiery.xtensor import _units
from fiery.xtensor._arrayutils import SmartSlicerT, _SmartSlicerT
from fiery.xtensor._compat import EllipsisType
from fiery.xtensor._compat import no_dispatch as _no_dispatch
from fiery.xtensor._compat import torch_func as _torch_func
from fiery.xtensor._options import combine_axes_policy as _combine_axes_policy
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
"""Free-form axis-descriptor fields (e.g. `type`, `orientation`, or any custom
key); only `orientation` carries built-in behaviour."""

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


def _either_dict_or_kwargs(
    positional: tx.Optional[tx.Mapping], kwargs: dict, funcname: str
) -> dict:
    """
    Merge an optional positional indexer mapping with `**kwargs`, xarray's own
    escape hatch for `.sel`/`.interp`-style calls: a dim whose name collides
    with one of the method's own keyword parameters (`.sel`'s `mode`/
    `tolerance`/`method`, `.interp`'s `method`/`bound`/`extrapolate`/`name`)
    can never be passed as `**kwargs` -- Python binds a matching keyword to
    the named parameter first, so it never reaches the catch-all -- but it
    can always be spelled out in an explicit dict instead
    (`x.sel({"method": 5.0})`). Passing both raises, rather than silently
    preferring one.
    """
    if positional is None:
        return dict(kwargs)
    if kwargs:
        raise ValueError(
            f"{funcname}: pass indexers as a dict OR as keyword arguments, "
            "not both"
        )
    return dict(positional)


def _expand_name_ellipsis(names: tuple, ndim: int, fill: tuple) -> tuple:
    """
    Expand a single `...` in a name tuple into the run of axes it stands for,
    so the tuple reaches `ndim` -- `...` means "the axes not named here". Each
    spanned position takes its value from `fill` (the same length as `ndim`):
    `(None,) * ndim` leaves the run **unnamed** (assignment), while the current
    names keep the run **unchanged** (modification). A tuple with no `...` is
    returned as-is; more than one `...` is an error.
    """
    if Ellipsis not in names:
        return names
    if names.count(Ellipsis) > 1:
        raise ValueError("only one '...' is allowed in a name list")
    i = names.index(Ellipsis)
    span = ndim - (len(names) - 1)
    if span < 0:
        raise ValueError(f"too many names for {ndim} axes: {names}")
    return names[:i] + tuple(fill[i : i + span]) + names[i + 1 :]


def _parse_axes(value: tuple, ndim: int) -> tx.Tuple[tuple, dict, dict]:
    """
    Parse an `axes=` spec into `(names, axis_meta, coord_specs)`. Each item is
    a bare name, `None`, or a **descriptor** dict with a required `name`, the
    coordinate keys `coord`/`labels` (→ `coord_specs`, handed to the `coords`
    setter), and any number of free-form metadata keys (e.g. `type`,
    `orientation`; → `axis_meta`). A single `...` fills the middle with unnamed
    axes.
    """
    value = _expand_name_ellipsis(value, ndim, (None,) * ndim)
    if len(value) != ndim:
        raise ValueError(f"Expected {ndim} axes, got {len(value)}: {value}")
    names, meta, coord_specs = [], {}, {}
    for item in value:
        if item is None or isinstance(item, str):
            names.append(item)
            continue
        if not isinstance(item, dict):
            raise TypeError(
                "axes= items must be a name, None, or a descriptor dict; "
                f"got {item!r}"
            )
        if "name" not in item:
            raise ValueError(f"axis descriptor must have a 'name': {item!r}")
        name = item["name"]
        names.append(name)
        extra = {
            k: v
            for k, v in item.items()
            if k not in ("name", "coord", "labels")
        }
        if "orientation" in extra:
            _validate_orientation(extra["orientation"])
        if extra and name is not None:
            meta[name] = extra
        if name is not None:
            if "coord" in item:
                coord_specs[name] = item["coord"]
            elif "labels" in item:
                coord_specs[name] = item["labels"]
    return tuple(names), meta, coord_specs


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

    def __deepcopy__(self, memo: dict) -> tx.Self:
        # `Tensor.__deepcopy__`'s default implementation is itself a
        # dispatched torch op (it starts with `has_torch_function_unary`),
        # so calling it on a subclass with a custom `__torch_function__`
        # re-enters that machinery under a disabled-dispatch context -- in
        # which `self.new_empty([])` (what it uses internally) returns a
        # plain `Tensor`, not this subclass, and it then raises rather than
        # silently mismatching. Defined as a plain method here (not a
        # registered override), so Python's normal attribute lookup finds
        # this directly and the dispatch-based default body never runs.
        if id(self) in memo:
            return memo[id(self)]
        # Unlike vanilla `Tensor.__deepcopy__`, this doesn't restrict itself
        # to graph leaves, and doesn't preserve the autograd graph either way
        # -- the result below is always a fresh, detached snapshot of the
        # current values, re-marked to require grad if the original did (if
        # you need the copy to stay attached to the original computation for
        # a later `.backward()`, deepcopy is the wrong tool regardless --
        # `.clone()` directly, without detaching, is). A strict leaf-only
        # check would fail even the ordinary case of wrapping an existing
        # `requires_grad=True` tensor: `as_subclass` (needed for the
        # zero-copy retag `XTensor(t)` does) is itself a *view* op under
        # PyTorch's own autograd rules, and any view of a grad-requiring leaf
        # is non-leaf -- true of a plain `Tensor.as_subclass`/`.view()` too,
        # not specific to this subclass -- so almost every `XTensor` wrapping
        # a grad-requiring input would already fail that check before any
        # arithmetic is even involved.
        data = self.as_subclass(Tensor).detach().clone()
        out = data.as_subclass(type(self))
        # `as_subclass` on a tensor that already requires grad returns a
        # tracked *view* (non-leaf) -- setting `requires_grad_()` only
        # afterwards, on the already-retagged (and by now grad-free) `out`,
        # is what keeps the result a genuine leaf.
        if self.requires_grad:
            out.requires_grad_()
        memo[id(self)] = out
        out.__dict__ = copy.deepcopy(self.__dict__, memo)
        if self.is_leaf and self.grad is not None:
            out.grad = copy.deepcopy(self.grad, memo)
        return out

    def __format__(self, format_spec: str) -> str:
        # `Tensor.__format__`'s own body checks `type(self) is Tensor` and
        # falls back to `object.__format__` (== `str(self)`) for any
        # subclass -- including this one. That's silently fine for most
        # calls, but fatal for a 0-dim tensor specifically: an int-dtype
        # tensor's `repr` (`torch._tensor_str._Formatter`) formats each
        # element via `f"{value}"`, where `value` is itself a 0-dim slice
        # of this subclass -- so `str(self)` on THAT slice re-enters the
        # very same tensor-printing machinery, on a tensor that is still
        # 0-dim and still this subclass, forever (issue #118). A plain
        # `Tensor` never hits this because its `__format__` extracts
        # `.item()` directly instead of recursing back into `repr`.
        # Defined as a plain method (not a registered override, matching
        # `__deepcopy__` above) so Python's normal dunder lookup finds
        # this directly, without going through `__torch_function__`.
        if self.dim() == 0 and not self.is_meta:
            return self.as_subclass(Tensor).item().__format__(format_spec)
        return object.__format__(self, format_spec)


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
    - **Axis descriptors** may enrich a name with extra fields -- any custom
      key you like (`type` is the OME-NGFF convention shown in examples;
      `orientation` is the one field with built-in behaviour) -- passed as a
      dict in place of a bare name (`{"name": "x", "type": "space"}`). `names`
      stays the ergonomic view (bare names); `axes` returns the full
      descriptors. The extra fields live in `_axis_meta`, keyed by dimension
      name, so they follow the dimension like coordinates do.

    Select by label with `sel`, by integer position with `isel`, or reach a
    single label by attribute (`x.red`).
    """

    _ATTRS = {
        "_axis_names",
        "_coords",
        "_axis_meta",
        "_data_unit",
    }

    def __new__(cls, *args, **kwargs) -> tx.Self:
        # NOTE: remove arguments that `Tensor.__new__` does not support.
        kwargs.pop("names", None)
        kwargs.pop("coords", None)
        kwargs.pop("axes", None)
        kwargs.pop("unit", None)
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
        # `names=` takes bare strings; `axes=` is the general per-axis
        # container (descriptor dicts: name + coord/labels + free-form fields).
        axes = kwargs.pop("axes", None)
        names = kwargs.pop("names", None)
        coords = kwargs.pop("coords", None)
        unit = kwargs.pop("unit", None)
        coord_specs = {}
        if axes is not None:
            axis_names, meta, coord_specs = _parse_axes(tuple(axes), self.ndim)
            self._axis_names = axis_names
            self._axis_meta = meta
        if names is not None:
            self.names = names
        if coords is not None:
            # an explicit `coords=` merges onto (and overrides) any coordinates
            # embedded in `axes=` descriptors.
            coord_specs = {**coord_specs, **dict(coords)}
        if coord_specs:
            self.coords = coord_specs
        if unit is not None:
            self.unit = unit

    # -- dimensions --------------------------------------------------------

    @property
    def names(self) -> tuple[str | None, ...]:
        """
        The name of each axis (`None` for unnamed axes). On assignment a single
        `...` expands to a run of unnamed axes, so `x.names = ("b", ..., "w")`
        names only the ends and leaves the middle unnamed.
        """
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
        # A single `...` fills the unspecified middle with unnamed axes, so
        # `names=("b", ..., "x")` on a 4-D tensor -> ("b", None, None, "x").
        value = _expand_name_ellipsis(value, self.ndim, (None,) * self.ndim)
        if len(value) != self.ndim:
            raise ValueError(
                f"Expected {self.ndim} names, got {len(value)}: {value}"
            )
        # `names=` takes bare strings (or `None`); richer axis descriptors --
        # `coord`/`labels` and free-form fields -- go through `axes=` instead.
        for item in value:
            if not (item is None or isinstance(item, str)):
                raise TypeError(
                    "names= takes strings (or None); pass a descriptor dict "
                    f"through axes= instead of {item!r}"
                )
        self._axis_names = value

    # -- axis descriptors --------------------------------------------------

    @property
    def axes(self) -> tuple[dict | None, ...]:
        """
        Each axis as a descriptor dict ``{"name": ..., **extra}`` (or `None`
        for an unnamed axis). The extra fields -- any custom key (`type` by
        OME-NGFF convention, `orientation`, ...) -- come from `_axis_meta`,
        keyed by dimension name.
        """
        meta = self._valid_axis_meta()
        return tuple(
            None if name is None else {"name": name, **meta.get(name, {})}
            for name in self.names
        )

    @axes.setter
    def axes(self, value: tx.Optional[tx.Sequence]) -> None:
        if value is None:
            for attr in ("_axis_names", "_axis_meta"):
                self.__dict__.pop(attr, None)
            return
        names, meta, coord_specs = _parse_axes(tuple(value), self.ndim)
        self._axis_names = names
        self._axis_meta = meta
        # A descriptor may embed its coordinate under `coord` (numeric) or
        # `labels` (categorical); apply those, leaving other coords untouched.
        if coord_specs:
            self.coords = coord_specs

    def _valid_axis_meta(self) -> dict[str, dict]:
        """`_axis_meta` filtered to dimensions still named on this tensor."""
        stored = self.__dict__.get("_axis_meta") or {}
        names = self.names
        return {name: extra for name, extra in stored.items() if name in names}

    # -- coordinates -------------------------------------------------------

    @property
    def coords(self) -> dict[str, LabelsT]:
        """
        The coordinates, as a `{dim name: coordinate}` dict. A coordinate is a
        tuple of **labels**, or a compact numeric [`Coordinate`][fiery.xtensor.
        _tensors.Coordinate] (`{spacing[, origin]}`, whose `["values"]` key
        materialises the positions; Proposal 0001).

        Only entries that are still valid are returned -- every dim in their
        `dims` must still be named on this tensor (and, for labels, its size
        must match the label count) -- so stale metadata propagated onto a
        shape-changing op is hidden.

        Stored internally as `{name: (dims, coord)}` (Proposal 0005): a
        **dimension** coordinate has `dims == (name,)` (it *is* the dim's
        index, so `.sel(name=...)` works); a **non-dimension** coordinate
        (disambiguated by key: `name` is not itself a dim) rides along some
        other dim(s), and is not an index. A non-dimension coordinate may span
        **several** dims (a compact **affine** coordinate, Proposal 0005 step
        3 -- `spacing` is a vector, one component per dim in `dims`, `origin`
        a single scalar shared across them); a general multi-dim *explicit*
        (curvilinear) coordinate isn't implemented yet.
        """
        names = self.names
        valid = {}
        stored = self.__dict__.get("_coords") or {}
        for name, (dims, coord) in stored.items():
            if any(dim not in names for dim in dims):
                continue
            if len(dims) > 1:
                # multi-dim / affine coordinate (Proposal 0005 step 3) -- only
                # the compact form is implemented; anything else is unreachable
                # (rejected at input time) so it is simply skipped here.
                # The grid is laid out in **this tensor's** axis order, not in
                # `dims` order: the two differ when `dims` was given in
                # another order, or once an axis-reordering op (`permute` /
                # `transpose` / `movedim`) has moved them, and `["values"]` is
                # a bare array with no dims of its own -- so materialising in
                # `dims` order would silently misalign it with the data.
                if isinstance(coord, Coordinate) and coord._compact():
                    axes = [names.index(dim) for dim in dims]
                    valid[name] = coord._bound_axes(
                        tuple(
                            (component, self.shape[axes[component]])
                            for component in sorted(
                                range(len(dims)), key=axes.__getitem__
                            )
                        )
                    )
                continue
            size = self.shape[names.index(dims[0])]
            if isinstance(coord, Coordinate):
                if coord._compact():
                    valid[name] = coord._bound(size)
                elif len(dict.__getitem__(coord, "values")) == size:
                    valid[name] = coord  # explicit: kept if length matches
            elif len(coord) == size:
                valid[name] = coord
        return valid

    @coords.setter
    def coords(self, value: tx.Optional[CoordsT]) -> None:
        if value is None:
            self.__dict__.pop("_coords", None)
            return
        names = self.names
        unified = {}
        for key, spec in dict(value).items():
            if key not in names:
                # a coord keyed by a non-axis name is a **non-dimension**
                # coordinate (Proposal 0005): given as `(dim, values)`, it
                # rides along `dim` rather than indexing it -- or, spanning
                # several dims, `(dims, {spacing, ...})` is a compact
                # **affine** coordinate (step 3); that form materialises and
                # re-slices exactly, so (unlike a single-dim non-dimension
                # coordinate) it has no fixed "length" to check on input.
                dims, coord = _parse_nondim_coord(key, spec, names)
                if len(dims) == 1:
                    size = self.shape[names.index(dims[0])]
                    _check_nondim_len(key, dims[0], coord, size)
                unified[key] = dims, coord
                continue
            if _is_compact_coord(spec) or _is_explicit_coord(spec):
                unified[key] = _pack_coord(key, _make_coordinate(spec))
                continue
            size = self.shape[names.index(key)]
            labels = tuple(spec)
            # a bare sequence of plain numbers is a numeric coordinate, not
            # labels that happen to be numbers (issue #107) -- auto-promote
            # it through the same explicit-coordinate path a tensor spec
            # already takes, so it gets real `.sel` support (mode/tolerance/
            # units) instead of becoming a silently uncomparable label. The
            # length check applies here too -- promotion must not bypass the
            # #95/#97 validation a plain label sequence already gets below.
            promoted = _promote_numeric_labels(key, labels)
            if isinstance(promoted, Coordinate):
                if len(labels) != size:
                    raise ValueError(
                        f"coords: dim {key!r} has {len(labels)} values "
                        f"for size {size}"
                    )
                unified[key] = _pack_coord(key, promoted)
                continue
            # `...` fills the middle with unlabelled positions.
            if Ellipsis in labels:
                labels = tuple(arrayutils._unroll(labels, size))
            if len(labels) != size:
                raise ValueError(
                    f"coords: dim {key!r} has {len(labels)} labels "
                    f"for size {size}"
                )
            unified[key] = _pack_coord(key, labels)
        self._coords = unified

    # -- data unit ---------------------------------------------------------

    @property
    def unit(self) -> tx.Optional[str]:
        """
        The physical unit of the tensor's **values** (the *data* unit, Proposal
        0003), or `None`. Assigning *annotates* (it never changes the data);
        `to_unit` converts. Under `unit_backend="pint"` the unit is validated
        and normalised on set; with the default `unit_backend=None` it is an
        opaque string that is simply carried through operations.
        """
        return self.__dict__.get("_data_unit")

    @unit.setter
    def unit(self, value: tx.Optional[str]) -> None:
        if value is None:
            self.__dict__.pop("_data_unit", None)
            return
        self._data_unit = _units.normalise(value)

    def to_unit(self, unit: str) -> tx.Self:
        """
        Convert the data to `unit`, rescaling the values by the conversion
        factor (requires a unit already set and `unit_backend="pint"`).
        """
        current = self.unit
        if current is None:
            raise ValueError("to_unit: this tensor has no unit to convert")
        unit = _units.normalise(unit)
        scaled = Tensor.mul(self, _units.factor(current, unit))
        return _carry(self, scaled, _data_unit=unit)

    @property
    def magnitude(self) -> tx.Self:
        """
        The tensor with its **data unit dropped** (Proposal 0003 §7.1) -- the
        bare values, still an `XTensor` with the same names and coordinates.
        A view (no data copy); the original is unchanged. `x.magnitude.unit`
        is always `None`. (To get a plain `torch.Tensor`, use
        `x.as_subclass(torch.Tensor)`.)
        """
        return _carry(self, self.as_subclass(type(self)), _data_unit=None)

    # -- attaching a unit by multiplication (Proposal 0003 §2.4) -----------
    #
    # `x * u.mm` / `x / u.s`: a backend `Unit`/`Quantity` operand attaches or
    # derives a data unit. This must be caught at the operator dunder, because
    # Python's protocol otherwise lets the unit library's reflected `__rmul__`
    # intercept `x * <unit>` first (yielding a wrapped object, never an
    # `XTensor`). A non-unit operand falls straight back to the normal path,
    # so name/unit algebra for ordinary operands is untouched.

    def __mul__(self, other: tx.Any) -> tx.Any:
        if _units.is_unit_like(other):
            return _attach_unit(self, other, "mul")
        return Tensor.__mul__(self, other)

    def __rmul__(self, other: tx.Any) -> tx.Any:
        if _units.is_unit_like(other):
            return _attach_unit(self, other, "mul")
        return Tensor.__rmul__(self, other)

    def __truediv__(self, other: tx.Any) -> tx.Any:
        if _units.is_unit_like(other):
            return _attach_unit(self, other, "div")
        return Tensor.__truediv__(self, other)

    def __rtruediv__(self, other: tx.Any) -> tx.Any:
        # `unit / x` is normally handled by the unit library itself before we
        # are consulted; this only fires for e.g. a scalar left operand.
        return Tensor.__rtruediv__(self, other)

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
        # A single `...` keeps the axes it spans unchanged (`rename` modifies,
        # so an unspecified run is left as-is, not unnamed).
        new_names = _expand_name_ellipsis(
            tuple(names), self.ndim, tuple(self.names)
        )
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
        """
        Coordinates re-keyed from the current names to `new_names`. A
        **dimension** coordinate (its key is one of `self.names`) is re-keyed
        like the axis, and dropped if that axis is unnamed; a
        **non-dimension** coordinate (Proposal 0005 -- its key is its own
        name, not a dim) keeps its key. Either way, every dim in its `dims`
        is remapped the same way, and the coordinate drops if any of them is
        unnamed.
        """
        stored = self.__dict__.get("_coords") or {}
        if not stored:
            return {}
        current = self.names
        rename_of = {
            old: new for old, new in zip(current, new_names) if new is not None
        }
        remapped = {}
        for old_key, (dims, coord) in stored.items():
            if old_key in current:
                new_key = rename_of.get(old_key)
                if new_key is None:
                    continue
            else:
                new_key = old_key
            if any(dim not in rename_of for dim in dims):
                continue
            if new_key in remapped:
                raise ValueError(
                    f"rename: coordinate name collision on {new_key!r} "
                    "(a renamed axis now matches an existing coordinate's "
                    "name); choose a name that doesn't collide"
                )
            new_dims = tuple(rename_of[dim] for dim in dims)
            if new_key in new_names and new_dims != (new_key,):
                # Renaming an axis onto a **multi-dim** coordinate's key would
                # leave an entry whose key *is* a dim but which is not that
                # dim's index -- breaking the `dims == (name,)` <=> dimension
                # coordinate invariant every consumer relies on (`sel`,
                # `__getitem__`'s dimension-coordinate pass, `flip`/`roll`
                # would then treat the vector `spacing` as a 1-D one and
                # corrupt it). Refuse, like any other name collision.
                raise ValueError(
                    f"rename: coordinate name collision on {new_key!r} "
                    "(a renamed axis now matches a multi-dim coordinate's "
                    f"name, which spans {new_dims}); choose a name that "
                    "doesn't collide"
                )
            remapped[new_key] = (new_dims, coord)
        return remapped

    def rename(self, *names: str | None, **rename_map: str) -> tx.Self:
        """
        Return a view with renamed axes (self-managed; not the builtin op).

        Call positionally (`x.rename("a", "b")`), with `None` to clear all
        names (`x.rename(None)`), or with a mapping to rename specific axes
        (`x.rename(old="new")`). A single `...` keeps the axes it spans
        unchanged (`x.rename("A", ..., "Z")`). Coordinates follow their
        (renamed) dimension.
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

    def _swap_dims_state(self, dims_map: dict) -> tuple:
        """
        Validate a `swap_dims` mapping and compute its `(new_names,
        new_coords)`. Each `{old_dim: new_name}` pair promotes `new_name` --
        an existing non-dimension coordinate riding `old_dim` **alone** -- to
        be `old_dim`'s replacement index, renaming the axis to `new_name` in
        the process (xarray's `swap_dims`).

        This is *not* a rename with extra steps: `rename` would re-key
        `old_dim`'s own dimension coordinate onto `new_name` too, colliding
        with the very coordinate being promoted (the same collision `rename`
        already raises on). `swap_dims` never re-keys a coordinate -- it only
        remaps `dims` tuples through the axis substitution, exactly like
        `rename` does for a coordinate's `dims` -- so which entry counts as
        *the* dimension coordinate falls out structurally afterwards (`dims
        == (key,)`): `new_name`'s entry becomes `(new_name,)` (now the
        index), and `old_dim`'s former entry becomes `(new_name,)` too but
        keyed `old_dim` (now a rider, since its key no longer matches its own
        `dims`) -- exactly xarray's "old index survives under its old name,
        riding the renamed axis" behaviour.
        """
        names = self.names
        stored = self.__dict__.get("_coords") or {}
        for old_dim, new_name in dims_map.items():
            if old_dim not in names:
                raise ValueError(f"swap_dims: no axis named {old_dim!r}")
            if new_name in names and new_name != old_dim:
                raise ValueError(
                    f"swap_dims: {new_name!r} is already an axis name"
                )
            entry = stored.get(new_name)
            if entry is None or entry[0] != (old_dim,):
                raise ValueError(
                    f"swap_dims: {new_name!r} must be an existing "
                    f"non-dimension coordinate riding {old_dim!r} alone, "
                    "to be promoted to its index"
                )
        new_names = tuple(dims_map.get(n, n) for n in names)
        seen = [n for n in new_names if n is not None]
        if len(set(seen)) != len(seen):
            raise ValueError(
                "swap_dims: the result would have duplicate axis names "
                f"{tuple(new_names)}"
            )
        new_coords = {
            key: (tuple(dims_map.get(d, d) for d in dims), coord)
            for key, (dims, coord) in stored.items()
        }
        return new_names, new_coords

    def swap_dims(
        self, dims_map: tx.Optional[dict] = None, **kwargs: str
    ) -> tx.Self:
        """
        Promote a non-dimension coordinate to be its dim's index, demoting
        the previous index to ride along under its old key -- xarray's
        `swap_dims`. `{old_dim: new_name}` (positionally or as keywords):
        `new_name` must already be a non-dimension coordinate riding
        `old_dim` alone (`coords={..., new_name: (old_dim, values)}`).

        ```python
        da.swap_dims({"time": "label"}).sel(label="c")   # promote, then select
        ```

        The axis itself is renamed `old_dim -> new_name` (so `.names` and any
        axis descriptor follow, like [`rename`][fiery.xtensor.XTensor.rename]);
        every other coordinate riding `old_dim` keeps its own key and simply
        rides the renamed axis.
        """
        mapping = dict(dims_map or {})
        mapping.update(kwargs)
        if not mapping:
            return self
        new_names, new_coords = self._swap_dims_state(mapping)
        out = self.as_subclass(type(self))
        out.__dict__.update(self.__dict__)
        out._coords = new_coords
        out._axis_meta = self._remap_named("_axis_meta", new_names)
        out._axis_names = new_names
        return out

    def swap_dims_(
        self, dims_map: tx.Optional[dict] = None, **kwargs: str
    ) -> tx.Self:
        """In-place variant of `swap_dims`."""
        mapping = dict(dims_map or {})
        mapping.update(kwargs)
        if not mapping:
            return self
        new_names, new_coords = self._swap_dims_state(mapping)
        self._coords = new_coords
        self._axis_meta = self._remap_named("_axis_meta", new_names)
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
        # ellipsis, if any, fills the remaining axes in the middle. A label
        # index (str / list-of-str / query dict) consumes exactly one axis.
        consumed = sum(
            1 if _is_label_index(v) else arrayutils._count_input_axes((v,))
            for v in items
            if v is not None and v is not ...
        )
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
        """
        Resolve a positional label index against `axis`:

        - a `str` -> the integer position whose label **identity** matches
          (drops the axis, like an int);
        - a list of `str` -> the list of such positions;
        - a `dict` -> a *query* over structured labels, giving a `slice`
          (contiguous) or index list of the matches (keeps the axis).

        Raises if the axis is unlabelled or a named label is absent.
        """
        name = self.names[axis % self.ndim]
        labels = self.coords.get(name) if name is not None else None
        if labels is None:
            raise KeyError(
                f"axis {name!r} has no coordinates for label {value!r}"
            )
        if isinstance(value, dict):
            return _positions_to_index(_match_positions(labels, value))

        identities = [_label_name(label) for label in labels]

        def _one(label: str) -> int:
            try:
                return identities.index(label)
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
        # A 0-D integer tensor index (`x[torch.tensor(1)]`) behaves exactly
        # like the plain `int` it's equivalent to; normalising it up front
        # means the slicer-classification helpers below never have to
        # special-case a tensor with no `len()`.
        slicer = arrayutils._normalize_scalar_tensor_index(slicer)
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
        # Carry every coordinate through by default (dimension or
        # non-dimension, unified `{name: (dims, coord)}` storage, Proposal
        # 0005): the loop below overwrites each *surviving* axis' own
        # **dimension** coordinate with its properly sliced value (or drops
        # it if the slicer can't be applied). A **non-dimension** coordinate
        # is never explicitly re-sliced here -- it rides through unchanged,
        # and the `coords` property's own dim/size validation drops it once
        # the dim it rides on is removed or resized.
        stored = self.__dict__.get("_coords") or {}
        if stored:
            unrolled = arrayutils._unroll_slicer(slicer, self.ndim)
            new_stored = dict(stored)
            for out_axis, src in enumerate(sources):
                name = out_names[out_axis]
                if src is None or name is None:
                    continue
                in_name = in_names[src]
                entry = stored.get(in_name)
                if entry is None:
                    continue
                _, coord = entry
                piece = arrayutils._get_slicer_by_index(unrolled, src)
                if isinstance(coord, Coordinate):
                    adjusted = _slice_coordinate(coord, piece, self.shape[src])
                    if adjusted is not None:
                        new_stored[name] = (name,), adjusted
                    else:
                        new_stored.pop(in_name, None)
                else:
                    sliced = _slice_labels(coord, piece)
                    if sliced is not None:
                        new_stored[name] = (name,), tuple(sliced)
                    else:
                        new_stored.pop(in_name, None)
            # A compact non-dimension coordinate -- one whose key is not the
            # dim it rides on, so the loop above never looks it up -- *is*
            # explicitly re-sliced when it spans a multi-dim compact
            # **affine** coordinate (Proposal 0005 step 3): exact per-
            # component, like a dimension coordinate's own basic-slice
            # update, just applied once per spanned dim.
            for key, (dims, coord) in stored.items():
                if len(dims) == 1 and dims[0] == key:
                    continue  # a dimension coordinate; handled above
                if not (isinstance(coord, Coordinate) and coord._compact()):
                    continue  # labels / explicit: ride through unchanged
                if any(dim not in in_names for dim in dims):
                    continue  # already invalid; the coords property drops it
                pieces = {}
                sizes = {}
                for dim in dims:
                    src = in_names.index(dim)
                    pieces[dim] = arrayutils._get_slicer_by_index(
                        unrolled, src
                    )
                    sizes[dim] = self.shape[src]
                result = _slice_affine_coordinate(coord, dims, pieces, sizes)
                if result is None:
                    new_stored.pop(key, None)
                else:
                    new_stored[key] = result
            # Selecting a single position on a unit-carrying axis collapses
            # that axis away; its per-position data unit folds into the base
            # data unit (effective unit = base * product of coord units).
            if _units.active():
                folded = self.__dict__.get("_data_unit")
                kept = {src for src in sources if src is not None}
                changed = False
                for ax, in_name in enumerate(in_names):
                    if ax in kept or in_name is None:
                        continue
                    piece = arrayutils._get_slicer_by_index(unrolled, ax)
                    if not isinstance(piece, int):
                        continue
                    entry = stored.get(in_name)
                    labels = (
                        entry[1]
                        if entry is not None
                        and not isinstance(entry[1], Coordinate)
                        else None
                    )
                    unit = _label_unit(labels[piece]) if labels else None
                    if unit is not None:
                        folded = _units.mul(folded, unit)
                        changed = True
                if changed:
                    out._data_unit = folded
            out._coords = new_stored
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

    def sel(
        self,
        indexers: tx.Optional[tx.Mapping] = None,
        mode: tx.Optional[str] = None,
        tolerance: tx.Any = None,
        method: tx.Optional[str] = None,
        **indexers_kwargs: tx.Any,
    ) -> tx.Self:
        """
        Select by coordinate **label** (or numeric value) along named dims.

        `x.sel(channel="red")` selects the position whose label is `"red"`. A
        list of labels selects several positions; a single label drops the
        dimension (like integer indexing). For **structured** coordinates, a
        `str` matches a label's `"name"`, and a **dict** queries the labels'
        fields (`x.sel(channel={"type": "signal"})`), keeping the axis and
        selecting every match.

        On a **numeric** coordinate (Proposal 0001), the selector is a value
        (`x.sel(t="2s")`, Proposal 0004). `mode` chooses which tick an inexact
        value snaps to:

        - `"round"` *(default)* — the nearest tick by value;
        - `"floor"` / `"ceil"` — the largest tick `<=` / smallest tick `>=`
          the value (**value** space, robust to a descending coordinate);
        - `"prev"` / `"next"` — the neighbouring tick at the lower / higher
          **index** (tick order; needs a monotonic coordinate).

        `tolerance` (a value in the position unit) caps the allowed gap. A
        **bare** `.sel(t=v)` is **exact** (`tolerance=0`); passing a `mode`
        implies an unbounded snap unless a `tolerance` is given.

        A **`slice(lo, hi)`** on a numeric coordinate is a **value range**
        (issue #109), unit-aware, resolving to a contiguous integer `slice` —
        half-open like ordinary Python indexing (`lo <= value < hi`), **not**
        xarray's inclusive-both-ends convention (see `vs-xarray.md`). Bounds
        are compared numerically regardless of order or of the coordinate's
        own direction: `t=slice(1, 5)` and `t=slice(5, 1)` select the same
        range. A single bound is positional (`slice(1, None)` -> `value >=
        1`; `slice(None, 5)` -> `value < 5`); an out-of-range or empty result
        is a well-formed empty axis, not an error. `slice.step` is not
        supported (`mode`/`tolerance` don't apply to a range either).

        A **joint query over an affine coordinate** (Proposal 0005 step 3 --
        `lat`/`lon`-style, spanning several dims at once) picks the dims'
        integer positions in one shot from a closed-form inverse (issue
        #82 phase 1): pass a value for *every* coordinate name that spans
        the same `dims` (e.g. `x.sel(lat=52.1, lon=4.3)` for a 2-D affine
        `lat`/`lon`) -- no dedicated syntax, ordinary keyword arguments that
        happen to share `dims` are recognised as one joint system. Only a
        **square, invertible** map is supported (exactly one coordinate
        value per spanned dim); an under- or over-determined query raises
        rather than falling back to a least-squares fit. Only `mode="round"`
        (the default) applies -- `floor`/`ceil`/`prev`/`next` have no
        well-defined meaning jointly across several coupled dims. `tolerance`
        applies per queried coordinate name, same as a 1-D numeric `.sel`
        (a bare query is exact by default) -- checked against the *rounded*
        position's own value, since a joint query never has one "the" gap
        the way a single coordinate does.

        Pass `indexers` as an explicit mapping (`x.sel({"mode": "red"})`)
        instead of keyword arguments when a dim's name collides with one
        of `sel`'s own keyword parameters (`mode`, `tolerance`, `method`)
        -- xarray's own escape hatch for exactly this, since a keyword
        argument matching one of those names is always bound to the
        parameter, never reaching the indexers. Passing both raises.
        """
        indexers = _either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
        if mode is not None and method is not None:
            raise ValueError("sel: pass either 'mode' or 'method', not both")
        raw = mode if mode is not None else method
        sel_mode = _resolve_sel_mode(raw)
        if tolerance is None:
            # a bare sel is exact; asking for a mode implies an unbounded snap
            tolerance = 0 if raw is None else None
        elif isinstance(tolerance, float) and tolerance == float("inf"):
            tolerance = None  # explicit unbounded
        positional, consumed = self._affine_sel_groups(
            indexers, sel_mode, tolerance
        )
        coords = self.coords
        for name, label in indexers.items():
            if name in consumed:
                continue
            if name in positional:
                # `name` is itself a dim already resolved by a joint affine
                # query over a *different* coordinate group spanning it --
                # e.g. `x.sel(lat=.., lon=.., y=..)` where `lat`/`lon` span
                # `y` too. Silently letting this loop overwrite that result
                # would discard the joint solve without any signal (#82
                # phase 1 review); pick one or the other instead.
                raise ValueError(
                    f"sel: dim {name!r} is set both by a joint affine "
                    "query over its coordinate group and directly in the "
                    "same call -- pass one or the other"
                )
            if name not in coords:
                raise ValueError(f"sel: dim {name!r} has no coordinates")
            if name not in self.names:
                # a non-dimension coordinate (Proposal 0005) is not an index
                raise ValueError(
                    f"sel: {name!r} is not an index coordinate; "
                    "promote it with swap_dims first"
                )
            labels = coords[name]
            if isinstance(labels, Coordinate):
                if isinstance(label, slice):
                    positional[name] = _numeric_select_range(
                        labels, label, name
                    )
                    continue
                positional[name] = _numeric_select(
                    labels, label, sel_mode, tolerance, name
                )
                continue
            if isinstance(label, dict):
                positional[name] = _positions_to_index(
                    _match_positions(labels, label)
                )
                continue
            identities = [_label_name(one) for one in labels]
            is_many = isinstance(label, (list, tuple))
            wanted = list(label) if is_many else [label]
            positions = []
            for one in wanted:
                # the selector needs the same identity extraction the
                # stored labels already got, so `.sel(season=Season.WINTER)`
                # and `.sel(season="WINTER")` resolve identically (#107) --
                # falling back to the raw selector only when it doesn't
                # resolve to an identity of its own (e.g. it's already a
                # plain string, where extraction is a no-op).
                target = _label_name(one)
                if target is None:
                    target = one
                try:
                    positions.append(identities.index(target))
                except ValueError:
                    raise ValueError(
                        f"sel: no label {one!r} on dim {name!r}"
                    ) from None
            positional[name] = positions if is_many else positions[0]
        return self.isel(**positional)

    def _affine_sel_groups(
        self,
        indexers: tx.Mapping[str, tx.Any],
        sel_mode: str,
        tolerance: tx.Optional[float],
    ) -> tuple:
        """
        Resolve every **joint affine query** among `.sel`'s `indexers` --
        `{dim: integer position}` for each spanned dim, plus the set of
        indexer names consumed this way (issue #82 phase 1). A coordinate
        NAME present in `indexers` that spans several dims at once
        (Proposal 0005 step 3) is grouped with every other queried name
        sharing the exact same `dims`; each group must supply exactly
        `len(dims)` values (one per dim) to be square and invertible.
        """
        stored = self.__dict__.get("_coords") or {}
        groups: dict = {}
        for name in indexers:
            entry = stored.get(name)
            if entry is None:
                continue
            dims, coord = entry
            if (
                len(dims) > 1
                and isinstance(coord, Coordinate)
                and coord._compact()
            ):
                groups.setdefault(dims, []).append(name)
        positional: dict = {}
        consumed: set = set()
        for dims, names_in_group in groups.items():
            if len(names_in_group) != len(dims):
                raise ValueError(
                    f"sel: a joint affine query over {dims!r} needs "
                    f"exactly {len(dims)} coordinate value(s) (one per "
                    f"dim), got {len(names_in_group)} "
                    f"({sorted(names_in_group)!r}) -- square systems only "
                    "(#82 phase 1), no least-squares fallback"
                )
            positional.update(
                _affine_sel_indices(
                    self,
                    dims,
                    names_in_group,
                    indexers,
                    sel_mode,
                    tolerance,
                )
            )
            consumed.update(names_in_group)
        return positional, consumed

    def interp(
        self,
        method: tx.Any = "linear",
        bound: tx.Any = None,
        extrapolate: tx.Any = None,
        **indexers: tx.Any,
    ) -> tx.Self:
        """
        Interpolate onto new coordinate values along named dims (Prop. 0004).

        Where [`sel`][fiery.xtensor.XTensor.sel] *picks* existing positions,
        `interp` *computes* values at arbitrary positions of a **numeric**
        coordinate, the xarray way::

            x.interp(t=2.5)                   # one point -> drops the axis
            x.interp(t=[0.0, 0.5, 1.0])       # several  -> keeps the axis
            x.interp(t="2.5s")                # unitful (backend converts)
            x.interp(t=q, method="cubic")     # a query tensor (grads flow)

        `method` is the interpolation order -- ``"nearest"`` (built in) or a
        higher order (``"linear"`` *(default)*, ``"quadratic"``, ``"cubic"``,
        or an int), which needs the optional `fiery.interpol` backend
        (``pip install fiery-xtensor[interp]``). An out-of-range query follows
        `bound` (default: the `interp_bound` option -- ``"replicate"`` clamps
        to the edge) and `extrapolate` (default: the `interp_extrapolate`
        option); both can be set with
        [`set_options`][fiery.xtensor.set_options].

        A **scalar** query drops the axis (like `sel`); a **list**/tensor keeps
        it, its coordinate becoming the queried positions. A **regular**
        (compact `spacing`/`origin`) coordinate supports every `method`; an
        **irregular** (explicit values) one only supports `"nearest"`/
        `"linear"` (issue #73, via a monotonic `searchsorted` inversion) --
        both are exact because the map between value space and index space is
        locally affine between two bracketing ticks. A higher order needs a
        true non-uniform spline in *value* space, which this architecture
        cannot provide (see issue #81); it is not a currently-missing
        feature.
        """
        out = self
        for name, target in indexers.items():
            out = out._interp_axis(name, target, method, bound, extrapolate)
        return out

    def _interp_axis(
        self,
        name: str,
        target: tx.Any,
        method: tx.Any,
        bound: tx.Any,
        extrapolate: tx.Any,
    ) -> tx.Self:
        """Interpolate a single named axis onto `target` (see `interp`)."""
        axis = _resolve_axis(self.names, name)
        coord = self.coords.get(name)
        if not isinstance(coord, Coordinate):
            raise ValueError(f"interp: dim {name!r} has no numeric coordinate")
        order = _interp_order(method)
        if coord._compact():
            spacing = dict.__getitem__(coord, "spacing")
            origin = dict.get(coord, "origin")
            unit = spacing["unit"]
            step = spacing["value"]
            base = origin["value"] if origin is not None else 0
            query, is_many = _query_values(target, unit)
            frac = (query - base) / step
        else:
            if order >= 2:
                raise NotImplementedError(
                    f"interp(method={method!r}) on the irregular coordinate "
                    f"{name!r}: only nearest/linear are supported on an "
                    "irregular coordinate (#73) -- a higher order would need "
                    "a true non-uniform spline in value space, which this "
                    "architecture cannot provide (not a missing feature, "
                    "see #81)"
                )
            stored_values = dict.__getitem__(coord, "values")
            unit = stored_values.unit
            query, is_many = _query_values(target, unit)
            frac = _irregular_frac(
                stored_values.as_subclass(Tensor), query, name
            )
        if frac.numel() == 0:
            # an empty query -> an empty axis, the same way an empty
            # advanced index (`x[[]]`) already behaves, rather than the
            # backend's internal reshape choking on a zero-sized grid (#96).
            empty_index = torch.empty(0, dtype=torch.long, device=self.device)
            raw = self.as_subclass(Tensor).index_select(axis, empty_index)
        else:
            eff_bound = _get_option("interp_bound") if bound is None else bound
            eff_extrap = (
                _get_option("interp_extrapolate")
                if extrapolate is None
                else extrapolate
            )
            raw = _interp_pull(
                self.as_subclass(Tensor),
                axis,
                frac,
                order,
                eff_bound,
                eff_extrap,
            )
        out = _carry(self, raw)
        # the interpolated axis now sits at the queried positions: give it an
        # explicit coordinate (dropping whatever `name` held before -- labels
        # or numeric -- plus any non-dimension coordinate riding on it, since
        # neither corresponds to the new positions; Proposal 0005).
        new_coords = _coords_dropping(self, name)
        explicit = Coordinate(values=XTensor(query, unit=unit))
        new_coords[name] = (name,), explicit
        out._coords = new_coords
        if not is_many:
            # a scalar query drops the axis (like integer indexing / sel)
            out = out.isel(**{name: 0})
        return out

    def _dims_with_label(self, label: str) -> list:
        """Named dims a label **identity** appears on (usually 0 or 1)."""
        return [
            dim
            for dim, labels in self.coords.items()
            if any(_label_name(one) == label for one in labels)
        ]

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
        # A single `...` keeps the names of the axes it spans (refine only
        # touches the *unnamed* axes; the spanned run rides through unchanged).
        names = _expand_name_ellipsis(names, self.ndim, tuple(current))
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


# ---- numeric coordinates (Proposal 0001) ----------------------------------


class Coordinate(_units.MagicDict):
    """
    A **numeric coordinate** (Proposal 0001) -- a magic dict in one of two
    forms:

    - **compact / regular** -- `{spacing[, origin]}` (each a
      [`Unitful`][fiery.xtensor._units.Unitful]); `["values"]` is a **derived**
      key materialising `origin + i * spacing` **fresh each access** (no cache,
      so a learnable spacing never goes stale and gradients flow back);
    - **explicit / irregular** -- `{"values": <unitful 1-D tensor>}`;
      `["values"]` returns the stored array.

    The **position** unit (`["values"].unit`) is distinct from the tensor's
    data unit (Proposal 0003).
    """

    def _compact(self) -> bool:
        """Whether this is a compact (spacing/origin) coordinate."""
        return "spacing" in self or "origin" in self

    def _bound(self, size: int) -> "Coordinate":
        """A copy that knows its axis `size`, so `["values"]` materialises."""
        out = Coordinate(self)
        out._size = size
        return out

    def _bound_axes(self, axes: tuple) -> "Coordinate":
        """
        A copy bound to several axes -- `((spacing component, axis size),
        ...)`, **in the host tensor's axis order** -- so `["values"]`
        materialises an N-D **affine** grid laid out like the tensor
        (Proposal 0005 step 3: `spacing` is a vector with one component per
        spanned dim, but `dims` need not be in the tensor's own axis order).
        """
        out = Coordinate(self)
        out._axes = tuple(axes)
        return out

    def __getitem__(self, key: tx.Any) -> tx.Any:
        if key == "values" and self._compact():
            if "_axes" in self.__dict__:
                return self._materialise_axes()
            return self._materialise()
        return dict.__getitem__(self, key)

    def _materialise(self) -> "XTensor":
        spacing = dict.__getitem__(self, "spacing")
        origin = dict.get(self, "origin")
        step = spacing["value"]
        start = origin["value"] if origin is not None else 0
        index = torch.arange(self._size)
        if isinstance(step, Tensor):
            index = index.to(step)
        values = start + index * step
        return XTensor(values, unit=spacing["unit"])

    def _materialise_axes(self) -> "XTensor":
        """
        Materialise a compact **affine** coordinate (Proposal 0005 step 3)
        over its bound `_axes`: `origin + sum_d spacing[d] * index_d`, via a
        broadcast `arange` per spanned dim -- an N-D grid, still
        differentiable w.r.t. `spacing`/`origin` (no dense grid is stored,
        only assembled fresh on each access, exactly like the 1-D case). The
        axes come in the host tensor's order (see `_bound_axes`), each paired
        with the `spacing` component it draws on, so the grid lines up with
        the data whatever order `dims` is in.
        """
        spacing = dict.__getitem__(self, "spacing")
        origin = dict.get(self, "origin")
        components = spacing["value"]
        total = origin["value"] if origin is not None else 0
        ndim = len(self._axes)
        for axis, (component_index, size) in enumerate(self._axes):
            component = components[component_index]
            index = torch.arange(size)
            if isinstance(component, Tensor):
                index = index.to(component)
            shape = [1] * ndim
            shape[axis] = size
            total = total + index.view(shape) * component
        return XTensor(total, unit=spacing["unit"])

    def to(self, unit: tx.Any) -> "Coordinate":
        """
        Convert the coordinate's **position** unit, rescaling
        `spacing`/`origin` (compact) or the stored `values` (explicit). Needs a
        backend. Carries over the axis-size binding (`_bound`/`_bound_axes`)
        if this coordinate already had one, so `coords[name].to(unit)
        ["values"]` still materialises instead of raising for lack of a
        bound size.
        """
        if self._compact():
            out = Coordinate()
            out["spacing"] = dict.__getitem__(self, "spacing").to(unit)
            if "origin" in self:
                out["origin"] = dict.__getitem__(self, "origin").to(unit)
        else:
            out = Coordinate(
                values=dict.__getitem__(self, "values").to_unit(unit)
            )
        if "_size" in self.__dict__:
            out._size = self._size
        if "_axes" in self.__dict__:
            out._axes = self._axes
        return out


def as_xtensor(
    value: tx.Any,
    *,
    dtype: tx.Any = None,
    device: tx.Any = None,
    unit: tx.Any = arrayutils._UNSET,
    names: tx.Any = arrayutils._UNSET,
    coords: tx.Any = arrayutils._UNSET,
) -> XTensor:
    """
    Coerce `value` (a bare Python number, a plain `Tensor`, or an `XTensor`)
    into an `XTensor` -- the `XTensor` analogue of `torch.as_tensor` (issue
    #114, generalising #112's `_coerce_unitful_tensor`): **graph-safe**
    (`torch.as_tensor(value)` with no `dtype=`/`device=` is a strict identity
    passthrough for an already-a-tensor `value` -- the *same object*, never a
    detaching copy, unlike `torch.tensor(existing_tensor)`'s well-known
    footgun of silently returning a fresh, non-differentiable leaf), and
    metadata-preserving: `unit`/`names`/`coords` ride through untouched
    unless a keyword **explicitly** overrides them -- mirroring how
    `torch.as_tensor(t, dtype=..., device=...)` only converts what you pass.
    A given override **replaces wholesale**, never merges (`coords={...}`
    discards whatever coordinates `value` already had, rather than combining
    the two) -- simpler to specify and implement than a per-key merge, and
    there's no established "merge coords" semantics to fall back on anyway.

    `dtype=`/`device=` extend `torch.as_tensor`'s own conversion, applied
    *before* the metadata is settled (so e.g. an axis-typed vs. numeric
    dtype affects nothing about the labels themselves). `None` (the default
    for both) means "leave as is" -- the same convention `torch.as_tensor`
    and `.to()` use -- so, unlike `unit`/`names`/`coords`, there's no
    separate "not overridden" sentinel needed here: a dtype/device override
    has no meaningful "clear" value the way `unit=None` does.

    A genuine dtype/device conversion is applied via `base.to(dtype=...,
    device=...)`, *not* by forwarding `dtype=`/`device=` straight to
    `torch.as_tensor` itself: `torch.as_tensor(an_xtensor, dtype=...)`
    silently degrades to a **plain `Tensor`** whenever it actually has to
    convert something (verified empirically -- a genuine PyTorch quirk, not
    a hypothetical), stripping every bit of metadata in the process. `.to()`
    on the subclass instead goes through its own `__torch_function__` (which
    already carries the axis names/coords/unit onto ops it doesn't otherwise
    special-case). `.to()` isn't called at all when `dtype`/`device` already
    match `value`'s own -- rather than relying on `.to()`'s own no-op-
    returns-self behaviour, which isn't consistent across the range of
    torch versions this library supports (verified: old torch does *not*
    short-circuit when `device=` is passed explicitly alongside `dtype=`,
    even when both already match).

    `value`'s own tensor is never mutated: when nothing is overridden and
    `value` is already an `XTensor`, it is returned as-is (the same object,
    metadata included); otherwise the result is always a **fresh** subclass
    view (`Tensor.as_subclass`, no data copy) before any override is
    applied, so overriding e.g. `unit=` never reaches back and changes
    `value`'s own unit as a side effect.
    """
    base = torch.as_tensor(value)
    # Skip `.to()` entirely when neither actually changes anything, rather
    # than trusting its own no-op-returns-self behaviour: passing `device=`
    # explicitly (even as `None`) alongside `dtype=` defeats that fast path
    # on old torch (verified on 1.7/1.8 CI) even when both already match --
    # this way, identity is guaranteed by construction, not by a version-
    # dependent internal optimisation.
    if (dtype is not None and dtype != base.dtype) or (
        device is not None and torch.device(device) != base.device
    ):
        base = base.to(dtype=dtype, device=device)
    if isinstance(base, XTensor) and (
        unit is arrayutils._UNSET
        and names is arrayutils._UNSET
        and coords is arrayutils._UNSET
    ):
        return base
    out_cls = type(base) if isinstance(base, XTensor) else XTensor
    out = base.as_subclass(out_cls)
    if isinstance(base, XTensor):
        # copy the *raw* stored metadata directly, not through the
        # `names`/`coords` property setters -- those validate against
        # `out`'s already-current names/shape, which is premature when
        # `names` is being overridden in the same call: a coordinate keyed
        # by the *old* name would fail that validation outright instead of
        # simply going stale (`.coords`'s own getter already treats a
        # coordinate whose dim isn't a current name as invalid and silently
        # drops it -- direct assignment to `.names` on an existing `XTensor`
        # has exactly this same behaviour today, so this matches it rather
        # than introducing a new failure mode).
        out.__dict__.update(base.__dict__)
    if names is not arrayutils._UNSET:
        out.names = names
    if unit is not arrayutils._UNSET:
        out.unit = unit
    if coords is not arrayutils._UNSET:
        out.coords = coords
    return out


def _as_unitful(obj: tx.Any) -> tx.Any:
    """Coerce a spacing/origin input to a `Unitful`, preserving a tensor."""
    if isinstance(obj, XTensor):
        unit = obj.unit
        if unit is None:
            return _units.Unitful(value=obj, unit=_units.normalise(""))
        return _units.Unitful(value=obj.magnitude, unit=unit)
    return _units.as_unitful(obj)


def _as_unitful_vector(obj: tx.Any, ndims: int) -> tx.Any:
    """
    Coerce an affine coordinate's `spacing` input (Proposal 0005 step 3) to a
    `Unitful` wrapping a 1-D tensor of `ndims` components -- one per spanned
    dim. Any component given as a tensor (e.g. a learnable 0-rank tensor) is
    preserved via `torch.stack` rather than `torch.as_tensor`, so it keeps its
    autograd graph; a component given as a bare number is not learnable.
    """
    unitful = _as_unitful(obj)
    value = unitful["value"]
    if isinstance(value, Tensor):
        vec = value
    elif isinstance(value, (list, tuple)) and any(
        isinstance(v, Tensor) for v in value
    ):
        vec = torch.stack(
            [
                v
                if isinstance(v, Tensor)
                else torch.as_tensor(v, dtype=torch.get_default_dtype())
                for v in value
            ]
        )
    else:
        vec = torch.as_tensor(value, dtype=torch.get_default_dtype())
    if vec.ndim != 1 or vec.shape[0] != ndims:
        raise ValueError(
            "coords: an affine coordinate's spacing must have one component "
            f"per dim ({ndims} here), got shape {tuple(vec.shape)}"
        )
    return _units.Unitful(value=vec, unit=unitful["unit"])


def _as_unitful_origin(obj: tx.Any) -> tx.Any:
    """
    Coerce a coordinate's `origin` input to a `Unitful`, requiring it be a
    **scalar** -- unlike `spacing`, `origin` is always a single value shared
    across every spanned dim, even for a multi-dim affine coordinate (step
    3). Catches a non-scalar origin here with a clear message instead of
    deferring to an opaque broadcast-shape error at materialisation time.
    """
    unitful = _as_unitful(obj)
    value = unitful["value"]
    if isinstance(value, Tensor):
        shape = tuple(value.shape)
    elif isinstance(value, (list, tuple)):
        shape = (len(value),)
    else:
        shape = ()
    if shape:
        raise ValueError(
            f"coords: a coordinate's origin must be a scalar, got shape "
            f"{shape}"
        )
    return unitful


def _reconcile_origin_unit(coord: Coordinate) -> None:
    """
    Make a compact coordinate's `origin` share `spacing`'s **unit**, in
    place. `_materialise`/`_materialise_axes` add `origin`'s raw magnitude
    directly onto the `spacing`-scaled index and label the *result* with
    `spacing`'s unit alone -- `origin`'s own declared unit is never
    otherwise consulted, so if it differs from `spacing`'s the two
    magnitudes would silently get mixed as if they were the same unit.

    A no-op if either is missing, or they already agree. If `origin`'s unit
    wasn't specified at all (a bare number, defaulting to `""`) it simply
    *inherits* `spacing`'s unit -- that default shouldn't read as a real,
    conflicting "dimensionless" declaration. Otherwise (`origin` was given
    an explicit unit that differs from `spacing`'s) it's converted into
    `spacing`'s unit if compatible (needs a backend for the actual
    conversion), or raises if the two are declared in incompatible units.
    """
    if "spacing" not in coord or "origin" not in coord:
        return
    spacing_unit = dict.__getitem__(coord, "spacing")["unit"]
    origin = dict.__getitem__(coord, "origin")
    origin_unit = origin["unit"]
    if _units.equal(spacing_unit, origin_unit):
        return
    # an omitted unit defaults to `""` with no backend, or normalises to the
    # real string `"dimensionless"` under pint -- either way, that default
    # shouldn't read as a deliberate, conflicting declaration.
    if not origin_unit or _units.dimensionless(origin_unit):
        dict.__setitem__(
            coord,
            "origin",
            _units.Unitful(value=origin["value"], unit=spacing_unit),
        )
        return
    if not _units.compatible(spacing_unit, origin_unit):
        raise ValueError(
            f"coords: origin's unit {origin_unit!r} is not compatible with "
            f"spacing's unit {spacing_unit!r}"
        )
    dict.__setitem__(coord, "origin", origin.to(spacing_unit))


def _is_compact_coord(spec: tx.Any) -> bool:
    """Whether a `coords[dim]` value is a compact numeric coordinate (a mapping
    with `spacing`/`origin`) rather than a sequence of labels."""
    return isinstance(spec, tx.Mapping) and (
        "spacing" in spec or "origin" in spec
    )


def _is_explicit_coord(spec: tx.Any) -> bool:
    """Whether a `coords[dim]` value is an **explicit** numeric coordinate -- a
    tensor of positions -- rather than a sequence of labels."""
    return isinstance(spec, Tensor)


def _is_pure_number(label: tx.Any) -> bool:
    """
    Whether a bare label is a **position**, not a category (issue #107) --
    a plain `int`/`float`, never a `bool` or an `enum.Enum` member (an
    `IntEnum`/`IntFlag` member *is* an actual `int` -- Python's own
    `class IntEnum(int, Enum)` -- but being an Enum member at all is a
    deliberate "this is a named category" signal, so it's checked first and
    always wins over the numeric check, the same role pandas' explicit
    `Categorical` dtype plays).
    """
    if isinstance(label, (bool, enum.Enum)):
        return False
    return isinstance(label, (int, float))


def _promote_numeric_labels(key: str, labels: tuple) -> tx.Any:
    """
    Auto-promote an all-numeric label sequence to a numeric `Coordinate`
    (issue #107) -- shared by dimension and non-dimension coordinate
    parsing, so a bare tuple/list of plain numbers is a numeric coordinate
    everywhere it can appear, not just on a dim's own index. Raise if
    numbers are mixed with categorical values (`bool`/`Enum`/`str`/dict/
    `None`/`...` are always categorical, never a position) -- otherwise
    return `labels` unchanged, for the caller's own label handling
    (Ellipsis-unroll, length check, ...).
    """
    if labels and all(_is_pure_number(label) for label in labels):
        try:
            values = torch.as_tensor(labels)
        except RuntimeError as exc:
            raise ValueError(
                f"coords: {key!r} -- {labels!r} isn't representable as a "
                f"numeric coordinate: {exc}"
            ) from None
        return _make_coordinate(values)
    if any(_is_pure_number(label) for label in labels):
        raise ValueError(
            f"coords: {key!r} mixes numeric values with categorical ones "
            f"in {labels!r} -- bool/Enum/str/dict/None/Ellipsis are "
            "always categorical (never a position); give a compact "
            "{'spacing': ...}/explicit tensor for a numeric coordinate, "
            "or make every value a label"
        )
    return labels


def _make_coordinate(spec: tx.Any) -> Coordinate:
    """Build a `Coordinate` from a compact spec or an explicit tensor."""
    if _is_explicit_coord(spec):
        if isinstance(spec, XTensor) and spec.unit is not None:
            values = as_xtensor(spec)  # preserve its own unit, graph-safe
        else:
            # force dimensionless -- via the override kwarg, not a post-hoc
            # mutation, so a unit-less `spec` is never changed in place.
            # (Benign behaviour change vs. the old `XTensor(spec, unit=...)`
            # call this replaced: if `spec` is itself an XTensor with
            # `unit=None`, its own `names`/`coords` now ride along instead of
            # being silently dropped -- verified inert for every existing
            # caller, since a bare Tensor/number spec has none to preserve.)
            values = as_xtensor(spec, unit=_units.normalise(""))
        if values.ndim != 1:
            raise ValueError(
                "coords: a numeric coordinate must be 1-D, got shape "
                f"{tuple(values.shape)}"
            )
        return Coordinate(values=values)
    coord = Coordinate()
    if "origin" in spec:
        coord["origin"] = _as_unitful_origin(spec["origin"])
    if "spacing" in spec:
        coord["spacing"] = _as_unitful(spec["spacing"])
    else:
        # symmetric to an omitted `origin` defaulting to 0 in `spacing`'s
        # unit (see `_materialise`): an omitted `spacing` defaults to 1 in
        # `origin`'s unit -- `_is_compact_coord` guarantees at least one of
        # the two is present, so this only runs with `origin` given.
        origin_unit = coord["origin"]["unit"] if "origin" in coord else ""
        coord["spacing"] = _units.Unitful(value=1, unit=origin_unit)
    _reconcile_origin_unit(coord)
    return coord


def _make_affine_coordinate(spec: tx.Mapping, ndims: int) -> Coordinate:
    """
    Build a compact **affine** `Coordinate` spanning `ndims` dims (Proposal
    0005 step 3) -- a generalisation of the 1-D compact form (0001) where
    `spacing` is a **vector**, one component per dim, and `origin` stays a
    single scalar shared across them: `value[i_0,...] = origin +
    sum_d spacing[d] * i_d`.
    """
    if "spacing" not in spec:
        raise ValueError(
            "coords: an affine (multi-dim) coordinate requires 'spacing'"
        )
    coord = Coordinate()
    coord["spacing"] = _as_unitful_vector(spec["spacing"], ndims)
    if "origin" in spec:
        coord["origin"] = _as_unitful_origin(spec["origin"])
    _reconcile_origin_unit(coord)
    return coord


def _affine_sel_indices(
    tensor: "XTensor",
    dims: tuple,
    names_in_group: list,
    indexers: tx.Mapping[str, tx.Any],
    sel_mode: str,
    tolerance: tx.Optional[float],
) -> dict:
    """
    Solve the closed-form affine inverse for one joint `.sel` query (issue
    #82 phase 1): given a target world value for each of `len(dims)`
    coordinate names spanning the same `dims`, `index = A^-1 (world -
    origin)`, then snap to the nearest integer position along each dim --
    never materialising the affine grid (`spacing`/`origin` alone are
    enough, mirroring the 1-D compact `.sel` fast path, #110).

    `A`'s rows are each queried coordinate's `spacing` vector (already
    ordered along `dims`, Proposal 0005 step 3); `names_in_group`'s order
    only has to line up between `A`'s rows and the right-hand side, not
    match any particular canonical order.
    """
    if sel_mode != "round":
        raise NotImplementedError(
            f"sel: mode={sel_mode!r} isn't supported for a joint affine "
            "query over several coupled dims (#82 phase 1) -- only the "
            "default 'round' is; floor/ceil/prev/next don't have a "
            "well-defined meaning jointly across several dims"
        )
    stored = tensor.__dict__.get("_coords") or {}
    per_name = []
    for name in names_in_group:
        _, coord = stored[name]
        spacing = dict.__getitem__(coord, "spacing")
        origin = dict.get(coord, "origin")
        vec = spacing["value"]
        # solved in float64 regardless of the spacing's own/default dtype --
        # matching `_numeric_select_compact`'s closed-form convention -- so a
        # genuinely float64-precision-dependent spacing (or an int one) isn't
        # silently downcast to float32 and solved wrong (review finding #3).
        if isinstance(vec, Tensor):
            vec = vec.to(torch.float64)
        else:
            vec = torch.as_tensor(vec, dtype=torch.float64)
        base = float(origin["value"]) if origin is not None else 0.0
        target = _selector_value(indexers[name], spacing["unit"])
        per_name.append((name, vec, base, target, spacing["unit"]))
    matrix = torch.stack([vec for _, vec, _, _, _ in per_name])
    # built on `matrix`'s own device (not implicitly CPU): a spacing tensor
    # that lives off-CPU must not force a cross-device op here (review
    # finding #2).
    vector = torch.tensor(
        [target - base for _, _, base, target, _ in per_name],
        dtype=matrix.dtype,
        device=matrix.device,
    )
    try:
        index = torch.inverse(matrix) @ vector
    except RuntimeError as exc:
        # narrowed to the actual singular-matrix message, so an unrelated
        # failure (e.g. a device mismatch) isn't misattributed as "not
        # invertible" (review finding #2).
        if "singular" not in str(exc).lower():
            raise
        raise ValueError(
            f"sel: the affine map over {dims!r} ({sorted(names_in_group)!r}) "
            f"isn't invertible: {exc}"
        ) from None
    rounded = index.round().long().tolist()
    result = {}
    for dim, position in zip(dims, rounded):
        size = tensor.shape[_resolve_axis(tensor.names, dim)]
        if not 0 <= position < size:
            raise ValueError(
                f"sel: the joint affine query resolves dim {dim!r} to "
                f"index {position}, out of range for size {size}"
            )
        result[dim] = position
    # `tolerance` was silently ignored for a joint query (review finding #4)
    # -- re-evaluate the forward map at the rounded index for each queried
    # coordinate NAME (not dim: the gap is meaningful per coordinate, since
    # several can share the same dims) and enforce it the same way the 1-D
    # path does, so a bare `.sel(lat=.., lon=..)` stays exact by default too.
    # Built once on `matrix`'s own device/dtype (not implicitly CPU) -- a
    # per-name `torch.tensor(rounded, ...)` with no `device=` would silently
    # reintroduce the exact cross-device bug fix #2 (above) already closed.
    rounded_t = torch.tensor(rounded, dtype=matrix.dtype, device=matrix.device)
    for name, vec, base, target, unit in per_name:
        predicted = base + float(vec @ rounded_t)
        gap = abs(predicted - target)
        tol = None if tolerance is None else _selector_value(tolerance, unit)
        _check_sel_tolerance(gap, tol, target, sel_mode, indexers[name], name)
    return result


# ---- non-dimension coordinates (Proposal 0005) -----------------------------


def _nondim_coord_len(coord: tx.Any) -> int:
    """The number of positions in a non-dimension coordinate's values."""
    if isinstance(coord, Coordinate):
        return len(dict.__getitem__(coord, "values"))
    return len(coord)


def _parse_nondim_coord(key: str, spec: tx.Any, names: tuple) -> tuple:
    """
    Parse a `(dim(s), values)` non-dimension coordinate spec into `(dims,
    coord)`. `dim(s)` is a single dim name (1-D, rides along that one dim), or
    a sequence of several dim names -- only supported for a **compact**
    (`spacing`/`origin`) coordinate: a multi-dim **affine** coordinate
    (Proposal 0005 step 3, `spacing` a vector, one component per dim).

    A **1-D compact** spec isn't supported: unlike a dimension coordinate, a
    single-dim non-dimension one isn't re-sliced when its dim is (there is no
    per-component affine to update against just one dim's slicer the way
    step 3's multi-dim form is) -- for an explicit or label coordinate
    that's caught by the length check on resize, but a compact coordinate
    binds to *any* size, so it would silently rebind to the wrong affine
    after a non-trivial slice instead of raising or dropping. Rejecting it
    here avoids that silent-wrong-values trap.

    A **multi-dim explicit** (general curvilinear, e.g. arbitrary `lat(y,x)`
    values) spec isn't implemented yet either -- only the compact affine form
    is (step 3); curvilinear arrays are future work (#82).
    """
    if not (isinstance(spec, tuple) and len(spec) == 2):
        raise ValueError(
            f"coords: {key!r} is not an axis; a non-dimension coordinate must "
            "be given as (dim, values) or (dims, values) for a multi-dim "
            "coordinate"
        )
    dims_spec, raw = spec
    if isinstance(dims_spec, str):
        dims = (dims_spec,)
    elif (
        isinstance(dims_spec, (list, tuple))
        and dims_spec
        and all(isinstance(d, str) for d in dims_spec)
    ):
        dims = tuple(dims_spec)
    else:
        raise ValueError(
            f"coords: {key!r} -- expected a dim name or a sequence of dim "
            f"names, got {dims_spec!r}"
        )
    for dim in dims:
        if dim not in names:
            raise ValueError(
                f"coords: no axis named {dim!r} in {tuple(names)}"
            )
    if len(set(dims)) != len(dims):
        raise ValueError(f"coords: {key!r} repeats a dim in {dims!r}")
    if _is_compact_coord(raw):
        if len(dims) == 1:
            raise NotImplementedError(
                f"coords: {key!r} -- a compact (spacing/origin) "
                "non-dimension coordinate over a single dim isn't supported "
                "yet (it wouldn't survive slicing its dim correctly); use "
                "an explicit tensor of values instead"
            )
        return dims, _make_affine_coordinate(raw, len(dims))
    if len(dims) > 1:
        raise NotImplementedError(
            f"coords: {key!r} -- a multi-dim non-dimension coordinate is "
            "only supported in compact (spacing/origin) affine form for "
            "now; explicit per-position values over several dims "
            "(curvilinear) isn't implemented yet"
        )
    if _is_explicit_coord(raw):
        coord = _make_coordinate(raw)
    else:
        coord = _promote_numeric_labels(key, tuple(raw))
    return dims, coord


def _check_nondim_len(key: str, dim: str, coord: tx.Any, size: int) -> None:
    """Validate a non-dimension coordinate's length against its dim's size."""
    length = _nondim_coord_len(coord)
    if length != size:
        raise ValueError(
            f"coords: non-dimension coordinate {key!r} has {length} values "
            f"for dim {dim!r} of size {size}"
        )


# ---- coordinate helpers ---------------------------------------------------


def _coords_of(tensor: tx.Any) -> dict:
    """The coordinate labels of `tensor` (empty for a plain / non tensor)."""
    if isinstance(tensor, XTensor):
        return tensor.coords
    return {}


def _pack_coord(name: str, coord: tx.Any) -> tuple:
    """
    Wrap one plain coordinate value into the unified `_coords` storage entry,
    `(dims, coord)` (Proposal 0005). Every coordinate is a **dimension**
    coordinate for now, so `dims == (name,)`; non-dimension / multi-dim
    coordinates widen `dims` in a later slice.
    """
    return (name,), coord


def _pack_coords(flat: tx.Mapping) -> dict:
    """`{name: coord}` -> the unified `{name: (dims, coord)}` storage shape."""
    return {name: _pack_coord(name, coord) for name, coord in flat.items()}


def _coords_for(input: XTensor, result_names: tuple) -> dict:
    """
    Keep the coordinates (dimension or non-dimension, labels or numeric) all
    of whose `dims` survive (by name) into `result_names`. A merged / split /
    removed axis loses its name, so any coordinate keyed on it -- or merely
    *riding* on it (Proposal 0005) -- drops automatically. `key in valid`
    first restricts to `input`'s own currently-valid coordinates (already
    filtered by size/name on `input`); the raw (unbound) coordinate is kept
    rather than `valid`'s bound copy, since a survivor's dim size is by
    construction unchanged, so the result rebinds it identically on read.
    """
    kept = {name for name in result_names if name is not None}
    valid = _coords_of(input)
    stored = input.__dict__.get("_coords") or {}
    return {
        key: (dims, coord)
        for key, (dims, coord) in stored.items()
        if key in valid and all(dim in kept for dim in dims)
    }


def _coords_dropping(input: XTensor, *dims: tx.Optional[str]) -> dict:
    """
    `input`'s coordinates, packed into unified storage, minus every
    coordinate that touches any of `dims` -- its own dimension coordinate, or
    a non-dimension coordinate (Proposal 0005) *riding* on it. For ops whose
    positions along `dims` no longer correspond to the stored ones (sort,
    flip, roll, gather, index_select, ...): the caller re-adds a transformed
    dimension coordinate for a touched dim itself when it can track one (e.g.
    flip reverses, roll rotates); a rider is conservatively dropped outright.
    """
    touched = set(dims)
    valid = _coords_of(input)
    stored = input.__dict__.get("_coords") or {}
    return {
        key: (entry_dims, coord)
        for key, (entry_dims, coord) in stored.items()
        if key in valid and not touched & set(entry_dims)
    }


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


def _is_label_index(value: tx.Any) -> bool:
    """
    Whether a slicer element is a **coordinate label** index: a bare `str`, a
    non-empty **list** of `str` (an advanced index by label), or a **dict**
    (a structured-coordinate *query* selecting the matching positions). A
    *tuple* is not, so a top-level `x["y", "z"]` stays one label per axis
    rather than a single advanced index. Plain ints, slices, `None`, ellipsis
    and tensors are not labels either.
    """
    if isinstance(value, (str, dict)):
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


def _label_name(label: tx.Any) -> tx.Any:
    """
    A label's **identity** for name-based selection: an `enum.Enum` member
    (`Enum`/`IntEnum`/`IntFlag`/`StrEnum`) is its `.name` -- checked
    **before** the `str` case, so a `str`-mixin enum (`class X(str, Enum)`,
    or `StrEnum`) resolves by name too, not by falling through to the `str`
    branch and matching on its *value* instead -- so `.sel(season="WINTER")`
    and `.sel(season=Season.WINTER)` resolve to the same identity (issue
    #107). A composite `Flag`/`IntFlag` value can have no single matching
    member name (`.name` is `None` -- observed on Python <= 3.10; 3.11+
    synthesises a `"A|B"` spelling), in which case the member itself is the
    identity instead -- still comparable by equality, just not selectable
    by a string name. A plain `str` is itself; a **structured** label (dict)
    is its `"name"` field; `bool` is its own identity too (`True`/`False`
    are a fixed two-value category, same reasoning as an `Enum` member,
    even though `bool` is technically an `int` subclass); a bare `int`/
    `float` is `None`: numbers are never labels, they're routed to a
    numeric `Coordinate` instead (`_is_pure_number`), so treating one as
    its own identity here would only paper over that split rather than
    respect it. Any other non-numeric object is its own identity.
    """
    if isinstance(label, enum.Enum):
        name = label.name
        return name if isinstance(name, str) else label
    if isinstance(label, str):
        return label
    if isinstance(label, dict):
        return label.get("name")
    if isinstance(label, bool):
        return label
    if isinstance(label, (int, float)):
        return None
    return label


def _label_unit(label: tx.Any) -> tx.Optional[str]:
    """
    A structured label's **per-position data unit** (its `"unit"` field), or
    `None` (Proposal 0003 phase 3 — heterogeneous, per-axis data units).
    """
    if isinstance(label, dict):
        return label.get("unit")
    return None


def _label_matches(label: tx.Any, query: tx.Mapping) -> bool:
    """Whether a **structured** `label` contains every key/value in `query`."""
    return isinstance(label, dict) and all(
        label.get(key) == value for key, value in query.items()
    )


def _match_positions(labels: LabelsT, query: tx.Mapping) -> list:
    """Positions whose structured label matches `query`, in axis order."""
    return [
        i for i, label in enumerate(labels) if _label_matches(label, query)
    ]


def _positions_to_index(positions: list) -> tx.Any:
    """
    Turn matched positions into an index that **keeps the axis**: a `slice`
    when they are contiguous (stays a basic index), else the position list (an
    advanced index). An empty match yields an empty list (a size-0 axis).
    """
    if positions and positions == list(range(positions[0], positions[-1] + 1)):
        return slice(positions[0], positions[-1] + 1)
    return positions


#: Relative tolerance for an "exact" numeric-coordinate match (floats).
_EXACT_MATCH_REL = 1e-6


def _selector_value(selector: tx.Any, unit: tx.Optional[str]) -> float:
    """
    A numeric selector as a plain float in the coordinate's position `unit`. A
    bare number is taken as already in that unit; a unitful selector (`"2mm"`,
    `(2, "mm")`, a pint quantity, ...) has its magnitude/unit split regardless
    of a backend -- only the actual **conversion** into a *different* unit
    (e.g. `"2000ms"` onto a `"s"` coordinate) needs one active.
    """
    if isinstance(selector, (int, float)):
        return float(selector)
    quantity = _as_unitful(selector)
    value, sel_unit = quantity["value"], quantity["unit"]
    if (
        unit
        and sel_unit
        and _units.active()
        and not _units.equal(sel_unit, unit)
    ):
        value = value * _units.factor(sel_unit, unit)
    return float(value)


#: `sel` modes -> canonical name. `round`/`floor`/`ceil` act on **values**;
#: `prev`/`next` on **tick order**. xarray's fill methods are positional, so
#: they alias onto `prev`/`next`.
_SEL_MODE_ALIASES = {
    "round": "round",
    "nearest": "round",
    "floor": "floor",
    "ceil": "ceil",
    "prev": "prev",
    "pad": "prev",
    "ffill": "prev",
    "next": "next",
    "backfill": "next",
    "bfill": "next",
}


def _resolve_sel_mode(mode: tx.Optional[str]) -> str:
    """The canonical `sel` mode for `mode`/`method` (`None` -> `"round"`)."""
    if mode is None:
        return "round"
    try:
        return _SEL_MODE_ALIASES[mode]
    except (KeyError, TypeError):
        raise ValueError(
            f"sel: unknown mode {mode!r}; use one of "
            "round/floor/ceil/prev/next (or the xarray aliases "
            "nearest/pad/ffill/backfill/bfill)"
        ) from None


def _check_sel_tolerance(
    gap: float,
    tol: tx.Optional[float],
    target: float,
    mode: str,
    one: tx.Any,
    name: str,
) -> None:
    """Raise if `gap` (the chosen tick's distance from `target`) is over
    `tolerance` -- shared by the compact and explicit `.sel` paths."""
    if tol is None:
        return
    cap = tol if tol > 0 else _EXACT_MATCH_REL * max(1.0, abs(target))
    if gap > cap:
        raise ValueError(
            f"sel: {mode} tick for {one!r} on {name!r} is {gap} away, "
            f"over tolerance {tol}"
        )


def _pick_sel_index(
    values: Tensor, target: float, mode: str, ascending: bool
) -> tx.Optional[int]:
    """
    The index of the tick `mode` selects for `target`, or `None` if there is
    none on the required side. `round` is nearest by value; `floor`/`ceil` are
    value-space; `prev`/`next` are tick-order (they resolve to `floor`/`ceil`
    per the coordinate's direction).
    """
    if mode == "round":
        return int((values - target).abs().argmin())
    if mode == "prev":
        mode = "floor" if ascending else "ceil"
    elif mode == "next":
        mode = "ceil" if ascending else "floor"
    if mode == "floor":  # largest value <= target
        mask = values <= target
        if not bool(mask.any()):
            return None
        cand = torch.where(
            mask, values, torch.full_like(values, float("-inf"))
        )
        return int(cand.argmax())
    # ceil: smallest value >= target
    mask = values >= target
    if not bool(mask.any()):
        return None
    cand = torch.where(mask, values, torch.full_like(values, float("inf")))
    return int(cand.argmin())


class _ClosedFormMiss(Exception):
    """
    Internal signal: the O(1) closed-form `.sel` search couldn't resolve a
    target within its bounded local walk. Only possible for an
    astronomically large `|origin/spacing|` ratio, where the
    `(target - origin) / spacing` seed's cancellation error spans more than
    `_CLOSED_FORM_MAX_STEPS` ticks -- the caller falls back to materialising
    and searching (the always-correct path) for just that one target,
    rather than risk a wrong answer.
    """


#: Cap on the local-walk correction steps in `_closed_form_sel_index`
#: before giving up on the O(1) shortcut for one target (see
#: `_ClosedFormMiss`). Generous relative to the walk actually needed for any
#: realistic `origin`/`spacing` ratio -- the division's cancellation error,
#: even scaled by a large ratio, is still a small fraction of one tick
#: spacing outside truly pathological (near float64 precision-limit) input.
_CLOSED_FORM_MAX_STEPS = 64


def _closed_form_sel_index(
    base: float,
    step: float,
    target: float,
    mode: str,
    ascending: bool,
    size: int,
) -> tx.Optional[int]:
    """
    The integer index `mode` selects for `target` on a compact coordinate
    (`value(k) = base + k*step`, `step != 0`) -- **exact**, matching
    `_pick_sel_index` (the search-based path) for any coordinate whose ticks
    are distinct in float64, in O(1) for any realistic input (issue #110).
    (At an astronomical `|base/step|` -- beyond float64's practical
    precision -- ticks can literally collide to the same float64 value;
    this still picks a tick with the identical *value* as the search path
    would, just not necessarily the identical *index* among duplicates --
    a degenerate-input edge case, not a real divergence.) `target` must not
    be `nan` -- checked by the caller, not here, so the check applies
    uniformly including the `spacing == 0` case this function never sees.

    - The two array **endpoints** (`k=0`, `k=size-1`) are compared against
      `target` directly, so "target is beyond (or exactly at) the whole
      coordinate" resolves exactly regardless of scale -- clamping to the
      last/first tick, or `None`, per `_pick_sel_index`'s semantics -- with
      no reliance on a noisy index estimate. This also makes an infinite
      `target` fall out correctly with no special-casing: comparing a
      finite endpoint against `+inf`/`-inf` is always well-defined.
    - Otherwise `target` lies strictly inside the coordinate's value range,
      so the tick `mode` wants is resolved by walking from a seed index --
      `(target - base) / step`, rounded -- toward the boundary, comparing
      **actual tick values** (`base + k*step`, a stable multiply-add) at
      each step, never trusting the division's result directly. This is
      exact (no epsilon guessing) because `value(k)` is monotonic in `k`,
      so the "satisfies" predicate `floor`/`ceil` cares about is a simple
      step function of `k` (true on a prefix or a suffix, depending on
      `mode`/`ascending`), and `|value(k) - target|` (for `round`) is
      unimodal in `k` -- a local walk that only stops on failing to
      strictly improve is guaranteed to reach the true global answer,
      *provided* the seed is within a bounded number of ticks of it. The
      division's cancellation error (which the PR #115 review found scales
      with `|base/step|`, not a fixed few ULPs) can violate that only at
      ratios far beyond realistic use -- `_ClosedFormMiss` is the safety
      net for that case.
    - `round`'s exact-tie tie-break favours the **lower** index (matching
      `argmin`'s first-occurrence rule): the walk only ever moves *left* on
      a tie (`<=`), never *right* (`<`), so it converges to the lower of
      two tied ticks regardless of which side the seed started on.
    - `prev`/`next` resolve to `floor`/`ceil` per direction, same as the
      search-based path.
    """
    if size == 0:
        return None
    if mode == "prev":
        mode = "floor" if ascending else "ceil"
    elif mode == "next":
        mode = "ceil" if ascending else "floor"

    def value(k: int) -> float:
        return base + k * step

    v_first, v_last = value(0), value(size - 1)
    lo_end, hi_end = (v_first, v_last) if ascending else (v_last, v_first)

    if mode == "round":
        if math.isinf(target):
            # matches `(values - (+/-inf)).abs().argmin()`: every entry
            # becomes `+inf`, so the first occurrence (index 0) wins.
            return 0
        idx = (target - base) / step

        def gap(k: int) -> float:
            return abs(value(k) - target)

        j = min(size - 1, max(0, int(round(idx))))
        for _ in range(_CLOSED_FORM_MAX_STEPS + 1):
            if j > 0 and gap(j - 1) <= gap(j):
                j -= 1
            elif j < size - 1 and gap(j + 1) < gap(j):
                j += 1
            else:
                return j
        raise _ClosedFormMiss

    if mode == "floor":  # largest tick value <= target
        if target < lo_end:
            return None
        if target >= hi_end:
            return (size - 1) if ascending else 0
        satisfies = lambda k: value(k) <= target  # noqa: E731
    else:  # ceil: smallest tick value >= target
        if target > hi_end:
            return None
        if target <= lo_end:
            return 0 if ascending else (size - 1)
        satisfies = lambda k: value(k) >= target  # noqa: E731

    # `target` is strictly between the endpoints, so a genuine boundary
    # exists in [0, size). `want_higher`: which way the "satisfying" step
    # function's true region extends (see docstring) -- ascending/floor and
    # descending/ceil want the *largest* satisfying k; the other two want
    # the *smallest*.
    want_higher = (mode == "floor") == ascending
    step_dir = 1 if want_higher else -1
    idx = (target - base) / step
    j = min(size - 1, max(0, int(round(idx))))
    steps = 0
    while not satisfies(j):
        j -= step_dir
        steps += 1
        if not (0 <= j < size) or steps > _CLOSED_FORM_MAX_STEPS:
            raise _ClosedFormMiss
    steps = 0
    while 0 <= j + step_dir < size and satisfies(j + step_dir):
        j += step_dir
        steps += 1
        if steps > _CLOSED_FORM_MAX_STEPS:
            raise _ClosedFormMiss
    return j


def _first_index_ge(
    base: float, step: float, size: int, threshold: float
) -> int:
    """
    The smallest `k` in `[0, size]` with `base + k*step >= threshold` --
    `size` itself means "no tick satisfies" (issue #109's range `.sel`
    shares this and `_first_index_lt` with #110's `_closed_form_sel_index`:
    same exact, no-materialisation technique -- endpoints checked directly,
    otherwise a division-seeded walk on real tick values).

    For an **ascending** `value(k)` (`step > 0`) this predicate is false
    then true (a suffix) -- the transition point needs an actual walk. For
    a **descending** one (`step < 0`) it's true then false (a prefix), so
    the smallest satisfying `k`, if any, is trivially `0`.
    """
    if size == 0:
        return 0

    def value(k: int) -> float:
        return base + k * step

    if step > 0:
        if value(size - 1) < threshold:
            return size
        if value(0) >= threshold:
            return 0
        idx = (threshold - base) / step
        j = min(size - 1, max(0, int(round(idx))))
        steps = 0
        while not value(j) >= threshold:
            j += 1
            steps += 1
            if steps > _CLOSED_FORM_MAX_STEPS:
                raise _ClosedFormMiss
        steps = 0
        while j > 0 and value(j - 1) >= threshold:
            j -= 1
            steps += 1
            if steps > _CLOSED_FORM_MAX_STEPS:
                raise _ClosedFormMiss
        return j
    return 0 if value(0) >= threshold else size


def _first_index_lt(
    base: float, step: float, size: int, threshold: float
) -> int:
    """
    The smallest `k` in `[0, size]` with `base + k*step < threshold` --
    `size` means "no tick satisfies". The mirror image of `_first_index_ge`
    -- trivial (`0`/`size`) for an **ascending** `value(k)` (a prefix
    predicate), a real walk for a **descending** one (a suffix).
    """
    if size == 0:
        return 0

    def value(k: int) -> float:
        return base + k * step

    if step < 0:
        if value(size - 1) >= threshold:
            return size
        if value(0) < threshold:
            return 0
        idx = (threshold - base) / step
        j = min(size - 1, max(0, int(round(idx))))
        steps = 0
        while not value(j) < threshold:
            j += 1
            steps += 1
            if steps > _CLOSED_FORM_MAX_STEPS:
                raise _ClosedFormMiss
        steps = 0
        while j > 0 and value(j - 1) < threshold:
            j -= 1
            steps += 1
            if steps > _CLOSED_FORM_MAX_STEPS:
                raise _ClosedFormMiss
        return j
    return 0 if value(0) < threshold else size


def _numeric_select_compact(
    coord: "Coordinate",
    selector: tx.Any,
    mode: str,
    tolerance: tx.Any,
    name: str,
) -> tx.Any:
    """
    `_numeric_select` for a **compact** coordinate: `origin`/`spacing` give an
    O(1) closed-form inverse (`index = (value - origin) / spacing`), so this
    never materialises `["values"]` or searches it (issue #110) -- the whole
    point of the compact representation is to avoid exactly that for a large
    regular grid. `coord` must already be size-bound (`coord._bound(size)`,
    what `.coords` always returns), so `coord._size` is available. Falls back
    to materialising and searching (still correct, just not O(1)) only for
    the rare target `_closed_form_sel_index` can't resolve locally
    (`_ClosedFormMiss`) -- see that function's docstring.
    """
    spacing = dict.__getitem__(coord, "spacing")
    origin = dict.get(coord, "origin")
    unit = spacing["unit"]
    step = float(spacing["value"])
    base = float(origin["value"]) if origin is not None else 0.0
    size = coord._size
    is_many = isinstance(selector, list)
    wanted = list(selector) if is_many else [selector]
    tol = None if tolerance is None else _selector_value(tolerance, unit)
    # a single-tick (or empty) coordinate has no direction of its own to
    # speak of -- match `_numeric_select`'s explicit-coordinate convention
    # of defaulting to ascending in that case, rather than trusting a
    # declared negative spacing that has nothing to actually order.
    ascending = True if size <= 1 else step > 0
    materialised_values = None  # lazily materialised only on a fallback
    positions = []
    for one in wanted:
        target = _selector_value(one, unit)
        if math.isnan(target):
            raise ValueError(f"sel: target {target!r} is not a number")
        if step == 0:
            # degenerate: every tick sits at `base` -- round always matches
            # it (index 0, the same tie-break `argmin` gives an all-equal
            # array, which is ascending per `(diffs >= 0).all()`, so
            # prev->floor, next->ceil); floor/ceil are valid only from the
            # matching side.
            if mode == "prev":
                eff_mode = "floor"
            elif mode == "next":
                eff_mode = "ceil"
            else:
                eff_mode = mode
            if (
                eff_mode == "round"
                or (eff_mode == "floor" and base <= target)
                or (eff_mode == "ceil" and base >= target)
            ):
                j = 0
            else:
                j = None
        else:
            try:
                j = _closed_form_sel_index(
                    base, step, target, mode, ascending, size
                )
            except _ClosedFormMiss:
                if materialised_values is None:
                    # built directly in float64 -- matching the closed-form
                    # walk's own arithmetic -- rather than materialising
                    # via `coord["values"]` (which computes in the tensor's
                    # default, float32, dtype: `torch.arange(size)*step`
                    # already loses precision there) and upcasting
                    # afterwards, which cannot recover what's already lost.
                    # This regime is exactly where that precision gap
                    # matters (an astronomically large `|base/step|`, the
                    # only way this fallback is ever reached).
                    materialised_values = (
                        base + torch.arange(size, dtype=torch.float64) * step
                    )
                j = _pick_sel_index(
                    materialised_values, target, mode, ascending
                )
        if j is None:
            raise ValueError(f"sel: no {mode} tick for {one!r} on {name!r}")
        gap = abs(base + j * step - target)
        _check_sel_tolerance(gap, tol, target, mode, one, name)
        positions.append(j)
    return positions if is_many else positions[0]


def _numeric_select(
    coord: "Coordinate",
    selector: tx.Any,
    mode: str,
    tolerance: tx.Any,
    name: str,
) -> tx.Any:
    """
    Resolve a value-based selector against a numeric `Coordinate` to integer
    position(s) (Proposal 0004), snapping per `mode` (see `sel`). `tolerance`
    (a delta in the position unit) caps the gap; `None` is unbounded, `0` is
    exact (up to float epsilon). A **compact** coordinate resolves in closed
    form (`_numeric_select_compact`, issue #110); an **explicit** one
    materialises and searches, below.
    """
    if coord._compact():
        return _numeric_select_compact(coord, selector, mode, tolerance, name)
    materialised = coord["values"]
    values = materialised.as_subclass(Tensor)
    unit = materialised.unit
    # a `list` selects several positions; a `tuple` is a unitful (value, unit)
    is_many = isinstance(selector, list)
    wanted = list(selector) if is_many else [selector]
    tol = None if tolerance is None else _selector_value(tolerance, unit)
    ascending = True
    if mode in ("prev", "next") and values.numel() > 1:
        diffs = values[1:] - values[:-1]
        if bool((diffs >= 0).all()):
            ascending = True
        elif bool((diffs <= 0).all()):
            ascending = False
        else:
            raise ValueError(
                f"sel: mode={mode!r} needs a monotonic coordinate on {name!r}"
            )
    positions = []
    for one in wanted:
        target = _selector_value(one, unit)
        j = _pick_sel_index(values, target, mode, ascending)
        if j is None:
            raise ValueError(f"sel: no {mode} tick for {one!r} on {name!r}")
        gap = float((values[j] - target).abs())
        _check_sel_tolerance(gap, tol, target, mode, one, name)
        positions.append(j)
    return positions if is_many else positions[0]


def _numeric_select_range(
    coord: "Coordinate", selector: slice, name: str
) -> slice:
    """
    Resolve a `slice(lo, hi)` value-range selector against a numeric
    `Coordinate` to an integer position `slice` (#109) -- half-open
    (`lo <= value < hi`), unit-aware, on both compact and explicit
    coordinates. Bounds are compared **numerically**, independent of the
    order they're given in or of the coordinate's own direction:
    `slice(lo, hi)` and `slice(hi, lo)` are the same request. A single bound
    is positional (`slice.start` alone -> `value >= start`; `slice.stop`
    alone -> `value < stop`); an out-of-range (including `+/-inf`) bound
    clamps to an empty or full range rather than raising (#96's empty-axis
    precedent); a `nan` bound raises, since no comparison to it is
    well-formed. `step` has no value-range meaning here and is rejected
    outright, rather than repurposed to signal an open/closed bound -- that
    would overload one field with two unrelated meanings, the trap #93
    already fixed.
    """
    if selector.step is not None:
        raise ValueError(
            f"sel: a range selector on {name!r} does not take a step "
            f"({selector.step!r}) -- slice(lo, hi) only"
        )
    if coord._compact():
        spacing = dict.__getitem__(coord, "spacing")
        unit = spacing["unit"]
        size = coord._size
    else:
        materialised = coord["values"]
        values = materialised.as_subclass(Tensor)
        unit = materialised.unit
        size = values.numel()
    start = (
        None
        if selector.start is None
        else _selector_value(selector.start, unit)
    )
    stop = (
        None if selector.stop is None else _selector_value(selector.stop, unit)
    )
    for bound in (start, stop):
        if bound is not None and math.isnan(bound):
            raise ValueError(
                f"sel: a range selector on {name!r} has a NaN bound"
            )
    if start is not None and stop is not None:
        lo, hi = (start, stop) if start <= stop else (stop, start)
    else:
        lo, hi = start, stop
    if coord._compact():
        return _compact_range_slice(coord, lo, hi, size)
    return _explicit_range_slice(values, lo, hi, name)


def _compact_range_slice(
    coord: "Coordinate",
    lo: tx.Optional[float],
    hi: tx.Optional[float],
    size: int,
) -> slice:
    """
    The compact-coordinate half of `_numeric_select_range` -- never
    materialises `["values"]` (issue #110's O(1) property extends to range
    selection too), sharing `_first_index_ge`/`_first_index_lt` with
    point-selection's closed-form path (`_numeric_select_compact`) rather
    than a second, independently-epsilon-tuned implementation.
    """
    spacing = dict.__getitem__(coord, "spacing")
    step = float(spacing["value"])
    origin = dict.get(coord, "origin")
    base = float(origin["value"]) if origin is not None else 0.0

    def resolve(fn, threshold, default):
        if threshold is None:
            return default
        try:
            return fn(base, step, size, threshold)
        except _ClosedFormMiss:
            full = base + torch.arange(size, dtype=torch.float64) * step
            side = "ge" if fn is _first_index_ge else "lt"
            mask = full >= threshold if side == "ge" else full < threshold
            return int(mask.long().argmax()) if bool(mask.any()) else size

    if step == 0:
        # every tick sits at `base` -- a plain value comparison decides
        # whether the whole axis is in range, or none of it is.
        included = (lo is None or base >= lo) and (hi is None or base < hi)
        return slice(0, size) if included else slice(0, 0)
    if step > 0:
        i_start = resolve(_first_index_ge, lo, 0)
        i_stop = resolve(_first_index_ge, hi, size)
    else:
        i_start = resolve(_first_index_lt, hi, 0)
        i_stop = resolve(_first_index_lt, lo, size)
    return slice(i_start, i_stop)


def _explicit_range_slice(
    values: Tensor, lo: tx.Optional[float], hi: tx.Optional[float], name: str
) -> slice:
    """The explicit half of `_numeric_select_range` (searchsorted-based)."""
    n = values.numel()
    if n == 0:
        return slice(0, 0)
    if n == 1:
        v = float(values[0])
        included = (lo is None or v >= lo) and (hi is None or v < hi)
        return slice(0, 1) if included else slice(0, 0)
    ticks = values.detach()
    diffs = ticks[1:] - ticks[:-1]
    if bool((diffs >= 0).all()):
        ascending, ordered = True, ticks
    elif bool((diffs <= 0).all()):
        ascending, ordered = False, ticks.flip(0)
    else:
        wanted = diffs >= 0 if bool(diffs[0] >= 0) else diffs <= 0
        j = int(wanted.logical_not().long().argmax())
        raise ValueError(
            f"sel: a range selector on {name!r} needs a monotonic "
            f"coordinate; ticks {j} and {j + 1} are {float(ticks[j])} and "
            f"{float(ticks[j + 1])}"
        )
    ordered = ordered.contiguous()
    if not ordered.is_floating_point():
        # an integer-dtype coordinate must not truncate a fractional bound
        # (10.5 silently becoming 10) when the needle is cast to match --
        # and float64, not the tensor default (float32), since an int64
        # coordinate can hold values (e.g. epoch timestamps) well past
        # float32's 2**24 exact-integer limit, where float32 would collapse
        # distinct ticks together just as badly as the truncation this
        # guards against.
        ordered = ordered.to(torch.float64)

    def _bracket(value: float) -> int:
        needle = torch.tensor(
            value, dtype=ordered.dtype, device=ordered.device
        )
        return int(torch.searchsorted(ordered, needle))

    k_start = 0 if lo is None else _bracket(lo)
    k_stop = n if hi is None else _bracket(hi)
    if ascending:
        return slice(k_start, k_stop)
    return slice(n - k_stop, n - k_start)


#: `interp` method names -> integer spline order (mirrors `fiery.interpol`).
_INTERP_ORDERS = {
    "nearest": 0,
    "zeroth": 0,
    "linear": 1,
    "first": 1,
    "quadratic": 2,
    "second": 2,
    "cubic": 3,
    "third": 3,
}


def _interp_order(method: tx.Any) -> int:
    """The integer spline order for an `interp` `method` (a name or an int)."""
    if isinstance(method, int) and not isinstance(method, bool):
        return method
    try:
        return _INTERP_ORDERS[method]
    except (KeyError, TypeError):
        raise ValueError(
            f"interp: unknown method {method!r}; use an int order or one of "
            f"{sorted(_INTERP_ORDERS)}"
        ) from None


def _query_values(target: tx.Any, unit: tx.Optional[str]) -> tx.Any:
    """
    A numeric `interp` query as a 1-D float tensor in the position `unit`, plus
    whether it **keeps** the axis (a list / 1-D tensor) or **drops** it (a
    scalar). A bare tensor is taken as already in the position unit (and its
    gradient rides through); everything else goes through `_selector_value`, so
    a unitful query (`"2s"`, `(2, "s")`, ...) is converted first.
    """
    if isinstance(target, Tensor):
        flat = target.reshape(-1)
        if not flat.is_floating_point():
            flat = flat.to(torch.get_default_dtype())
        return flat, target.ndim > 0
    is_many = isinstance(target, list)
    items = target if is_many else [target]
    values = [_selector_value(one, unit) for one in items]
    query = torch.tensor(values, dtype=torch.get_default_dtype())
    return query, is_many


def _irregular_frac(values: Tensor, query: Tensor, name: str) -> Tensor:
    """
    Fractional index for `query` against an **irregular** (non-uniform,
    strictly monotonic) 1-D coordinate `values`, via `torch.searchsorted` +
    a local linear inverse (issue #73): `searchsorted` brackets each query
    between two adjacent ticks `values[k] <= query <= values[k+1]` (ascending
    order; a descending coordinate is bracketed the same way by searching
    its reverse), then `k + (query - values[k]) / (values[k+1] - values[k])`
    is the exact fractional index -- it inverts the same piecewise-linear
    map the nearest/linear pull already samples between those two ticks, so
    the round trip is exact (unlike a higher order, whose spline basis is
    uniform in index space -- see #81). Differentiable w.r.t. both `query`
    and `values`: only the *search* (which bracket a query falls in) runs on
    detached copies, since an index has no useful gradient; the returned
    fraction is computed from the original tensors. `values` is guaranteed
    1-D here -- `_make_coordinate` rejects a non-1-D coordinate at
    construction (#97), so there's no need to re-check it per consumer.
    """
    n = values.numel()
    if n < 2:
        raise ValueError(
            f"interp: irregular coordinate {name!r} needs at least 2 points"
        )
    ticks = values.detach()  # the check is a predicate: no graph needed
    diffs = ticks[1:] - ticks[:-1]
    if bool((diffs > 0).all()):
        ascending, ordered = True, values
    elif bool((diffs < 0).all()):
        ascending, ordered = False, values.flip(0)
    else:
        # point at the first offending pair -- a tie (a repeated tick, easy to
        # hit by accident in float32) reads as "not monotonic" otherwise, with
        # nothing to say *where*.
        wanted = diffs > 0 if bool(diffs[0] > 0) else diffs < 0
        j = int(wanted.logical_not().long().argmax())
        raise ValueError(
            f"interp: irregular coordinate {name!r} must be strictly "
            f"monotonic (ascending or descending); ticks {j} and {j + 1} "
            f"are {float(ticks[j])} and {float(ticks[j + 1])}"
        )
    # a coordinate sliced with a step (`x[::2]`) is a strided view, which
    # `searchsorted` copies (and warns about) -- do it once, quietly.
    k = (
        torch.searchsorted(
            ordered.detach().contiguous(), query.detach(), right=False
        )
        - 1
    )
    k = k.clamp(0, n - 2)
    v0, v1 = ordered[k], ordered[k + 1]
    frac = k.to(query.dtype) + (query - v0) / (v1 - v0)
    return frac if ascending else (n - 1) - frac


def _interpol() -> tx.Any:
    """The optional `fiery.interpol` backend, or `None` if not installed."""
    try:
        from fiery import interpol
    except ImportError:
        return None
    return interpol


def _nearest_gather(
    moved: Tensor, frac: Tensor, length: int, bound: tx.Any
) -> Tensor:
    """
    Built-in nearest-neighbour pull along the **last** axis of `moved` (no
    backend). The fractional indices `frac` round to the closest tick; an
    out-of-range index is resolved by `bound` -- clamp for
    ``"replicate"``/``"nearest"``, wrap for ``"dft"``/``"wrap"``. Any other
    boundary needs the `fiery.interpol` backend.
    """
    idx = frac.round().long()
    if bound in ("replicate", "nearest", 1):
        idx = idx.clamp(0, length - 1)
    elif bound in ("dft", "wrap", 6):
        idx = idx.remainder(length)
    else:
        raise ImportError(
            f"interp method='nearest' with bound {bound!r} needs the "
            "fiery.interpol backend; install fiery-xtensor[interp]"
        )
    return moved.index_select(-1, idx)


def _interp_pull(
    raw: Tensor,
    axis: int,
    frac: Tensor,
    order: int,
    bound: tx.Any,
    extrapolate: tx.Any,
) -> Tensor:
    """
    Interpolate `raw` along `axis` at fractional indices `frac` (see `interp`).

    Order 0 (nearest) is done in-package -- a gather -- so it needs no backend;
    order >= 1 delegates to `fiery.interpol.grid_pull`, the optional
    `fiery-xtensor[interp]` dependency.
    """
    n = int(frac.shape[0])
    moved = torch.movedim(raw, axis, -1)  # (*rest, length)
    rest = moved.shape[:-1]
    length = int(moved.shape[-1])
    interpol = _interpol()
    if order == 0 and interpol is None:
        out = _nearest_gather(moved, frac, length, bound)
    else:
        if interpol is None:
            raise ImportError(
                "interp with order >= 1 needs the fiery.interpol backend; "
                "install fiery-xtensor[interp]"
            )
        flat = moved.reshape(-1, 1, length)
        if not flat.is_floating_point():
            flat = flat.to(torch.get_default_dtype())
        grid = frac.reshape(1, n, 1).to(flat).expand(flat.shape[0], n, 1)
        pulled = interpol.grid_pull(
            flat,
            grid,
            interpolation=order,
            bound=bound,
            extrapolate=extrapolate,
        )  # (batch, 1, n)
        out = pulled.reshape(*rest, n)
    return torch.movedim(out, -1, axis)


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


def _slice_coordinate(
    coord: Coordinate, slicer: _SmartSlicerT, size: int
) -> tx.Optional[Coordinate]:
    """
    Apply a 1-D `slicer` to a numeric `Coordinate` on an axis of `size`. A
    **basic slice** stays exact: a compact coordinate updates its affine
    (`spacing *= step`, `origin += start * spacing`); an explicit one slices
    its values. An **advanced** index materialises a compact coordinate to
    explicit first. Returns `None` for a slicer that cannot be applied (the
    coordinate then drops).
    """
    if isinstance(slicer, slice):
        start, stop, step = slicer.indices(size)
        if start == 0 and step == 1 and stop >= size:
            return coord  # a full slice leaves the coordinate untouched
        if coord._compact():
            spacing = dict.__getitem__(coord, "spacing")
            origin = dict.get(coord, "origin")
            base = origin["value"] if origin is not None else 0
            out = Coordinate()
            out["spacing"] = _units.Unitful(
                value=spacing["value"] * step, unit=spacing["unit"]
            )
            out["origin"] = _units.Unitful(
                value=base + start * spacing["value"], unit=spacing["unit"]
            )
            return out
        return Coordinate(values=dict.__getitem__(coord, "values")[slicer])
    if arrayutils._is_boolean_index(slicer) or arrayutils._is_advanced_index(
        slicer
    ):
        if coord._compact():
            values = coord._bound(size)["values"]
        else:
            values = dict.__getitem__(coord, "values")
        return Coordinate(values=values[slicer])
    return None


def _slice_affine_coordinate(
    coord: Coordinate, dims: tuple, pieces: dict, sizes: dict
) -> tx.Optional[tuple]:
    """
    Apply one slicer per spanned dim to a compact coordinate that may span
    **several** dims (a non-dimension coordinate, `len(dims) >= 1`; the
    genuinely multi-dim case is Proposal 0005 step 3's affine coordinate --
    `spacing` a vector, one component per dim, `origin` a single shared
    scalar). Exact per-component, 0001's trick generalised:

    - a **basic slice** on a dim updates that dim's component exactly
      (`origin += start * component`, `component *= step`) and keeps the dim;
    - an **integer** index folds that dim out entirely (`origin += index *
      component`), dropping it from `dims`/`spacing` -- the coordinate
      survives with one fewer dim (possibly collapsing to an ordinary 1-D
      compact non-dimension coordinate);
    - anything else (boolean / advanced indexing) can't stay affine, so the
      *whole* coordinate is dropped.

    Returns `(new_dims, new_coord)`, or `None` to drop the coordinate
    (either because an unsupported indexer touched one of its dims, or
    because every dim it spanned was folded away by integer indices, leaving
    no axis for it to ride on).
    """
    if all(
        isinstance(pieces[dim], slice)
        and pieces[dim].indices(sizes[dim]) == (0, sizes[dim], 1)
        for dim in dims
    ):
        return dims, coord  # every dim is a full no-op slice; nothing to do
    spacing = dict.__getitem__(coord, "spacing")
    origin = dict.get(coord, "origin")
    unit = spacing["unit"]
    components = spacing["value"]
    # a coordinate already collapsed to a single dim stores a bare scalar
    # `spacing` (the ordinary 1-D compact form), not a length-1 vector -- only
    # index into it when there is more than one component to pick from.
    is_vector = len(dims) > 1
    base = origin["value"] if origin is not None else 0
    new_dims = []
    new_components = []
    for i, dim in enumerate(dims):
        piece = pieces[dim]
        size = sizes[dim]
        component = components[i] if is_vector else components
        if isinstance(piece, slice):
            start, _stop, step = piece.indices(size)
            base = base + start * component
            new_dims.append(dim)
            new_components.append(component * step)
        elif isinstance(piece, int):
            index = piece + size if piece < 0 else piece
            base = base + index * component
        else:
            return None  # boolean / advanced index: can't stay affine
    if not new_dims:
        return None  # every spanned dim was folded away; no axis left to ride
    new_coord = Coordinate()
    new_coord["spacing"] = _units.Unitful(
        value=new_components[0]
        if len(new_components) == 1
        else torch.stack(new_components),
        unit=unit,
    )
    new_coord["origin"] = _units.Unitful(value=base, unit=unit)
    return tuple(new_dims), new_coord


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


#: Sentinel: an axis's per-position units disagree (dimensionally invalid).
_INCOMPATIBLE = object()


def _uniform_unit(labels: LabelsT) -> tx.Any:
    """
    The single per-position data unit shared by every label on an axis:
    `None` if the axis carries no units, the common unit if they all agree
    (under the backend), or `_INCOMPATIBLE` when they differ or only some
    positions carry one.
    """
    units = [_label_unit(one) for one in labels]
    present = [u for u in units if u is not None]
    if not present:
        return None
    first = present[0]
    if len(present) != len(units):
        return _INCOMPATIBLE
    if any(not _units.equal(first, other) for other in present[1:]):
        return _INCOMPATIBLE
    return first


def _reduce_unit(input: XTensor, removed: tx.Set) -> dict:
    """
    Fold the per-position units of any reduced unit-carrying axis into the base
    data unit (a reduction sums positions, so their unit must be uniform).
    Incompatible units are dimensionally invalid: drop the unit (default) or
    raise under `unit_policy="strict"`. Returns an override for `_carry` (empty
    when nothing changes, so the base unit propagates untouched).
    """
    if not _units.active():
        return {}
    coords = input.coords
    if not coords:
        return {}
    names = input.names
    base = input.__dict__.get("_data_unit")
    changed = False
    for ax in removed:
        name = names[ax] if ax < len(names) else None
        labels = coords.get(name) if name is not None else None
        if not labels:
            continue
        unit = _uniform_unit(labels)
        if unit is _INCOMPATIBLE:
            _unit_strict(True, f"reducing incompatible units on axis {name!r}")
            return {"_data_unit": None}
        if unit is not None:
            base = _units.mul(base, unit)
            changed = True
    return {"_data_unit": base} if changed else {}


def _reduce_names(input: XTensor, result: tx.Any, dim: tx.Any) -> tx.Any:
    """Recompute the name metadata for a dimension-reducing op's result."""
    if not isinstance(result, Tensor):
        # e.g. a (values, indices) namedtuple: left to a bespoke override.
        return result
    ndim = input.ndim
    if dim is None:
        removed = set(range(ndim))
    else:
        dims = dim if isinstance(dim, (tuple, list)) else (dim,)
        removed = {d % ndim for d in dims}
    unit_kw = _reduce_unit(input, removed)
    # `keepdim` is inferable from the output rank: a reduction either removes
    # the reduced axes or keeps them as size-1. Either way the reduced axis's
    # coordinates go, so its folded unit still applies. Dropped explicitly
    # (issue #90): the reduced axis's *name* is unchanged under `keepdim`, so
    # a compact coordinate -- which has no length of its own to invalidate,
    # unlike a label/explicit one -- would otherwise silently rebind to the
    # new size-1 axis as if it described "position 0" of the original extent.
    if dim is not None and result.ndim == ndim:
        in_names = input.names
        reduced = {in_names[ax] for ax in removed if ax < len(in_names)}
        return _carry(
            input,
            result,
            _axis_names=in_names,
            _coords=_coords_dropping(input, *reduced),
            **unit_kw,
        )
    names = tuple(n for i, n in enumerate(input.names) if i not in removed)
    return _carry(
        input,
        result,
        _axis_names=names,
        _coords=_coords_for(input, names),
        **unit_kw,
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
        coords = _coords_dropping(input, names[dim % input.ndim])
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
        ref,
        result,
        _axis_names=names,
        _coords=_pack_coords(coords),
        _axis_meta=meta,
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
        ref,
        result,
        _axis_names=names,
        _coords=_pack_coords(coords),
        _axis_meta=meta,
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
        # A contraction is a sum of products: fold each side's contracted-axis
        # unit into its base and multiply (heterogeneous units require the
        # contracted axis to be unit-uniform per side).
        unit_kw = {}
        if _units.active():
            axa, axb = _matmul_contracted_axes(
                getattr(input, "ndim", 0), getattr(other, "ndim", 0)
            )
            unit_kw["_data_unit"] = _contraction_unit(
                (input, other), ([axa], [axb])
            )
        # The contraction invalidates the coordinate layout; surviving axes
        # keep their (merged) descriptors.
        return _carry(
            ref,
            result,
            _axis_names=names,
            _coords={},
            _axis_meta=_merge_axis_meta((input, other), names),
            **unit_kw,
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
    unit_kw = {}
    if _units.active():
        axes = _einsum_contracted_axes(equation, flat)
        if axes is None:
            # unparsable (e.g. ellipsis): fall back to the product of bases
            base = None
            for operand in flat:
                base = _units.mul(base, _unit_of(operand))
            unit_kw["_data_unit"] = base
        else:
            unit_kw["_data_unit"] = _contraction_unit(flat, axes)
    return _carry(
        ref, result, _axis_names=names, _coords={}, _axis_meta=meta, **unit_kw
    )


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
    unit_kw = {}
    if _units.active():
        unit_kw["_data_unit"] = _contraction_unit(
            (a, b), (sorted(a_contracted), sorted(b_contracted))
        )
    return _carry(
        ref, result, _axis_names=names, _coords={}, _axis_meta=meta, **unit_kw
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
    if labels is not None:
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
            # list membership (not a set) so unhashable structured labels align
            common = tuple(label for label in ca if label in cb)
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


def _leading_none(names: tuple) -> int:
    """The length of the leading run of `None` axes (the anonymous prefix)."""
    count = 0
    for name in names:
        if name is not None:
            break
        count += 1
    return count


def _anon_leading(names: tuple) -> bool:
    """
    Whether every unnamed axis is in the **leading** run -- no `None` after a
    named axis. This is the layout partial-name alignment can handle (issue
    #75); an interleaved/trailing `None` is ambiguous and rejected.
    """
    seen_named = False
    for name in names:
        if name is None and seen_named:
            return False
        if name is not None:
            seen_named = True
    return True


def _reconcile_coords(a: XTensor, b: XTensor, names: tx.Iterable) -> tuple:
    """
    Reconcile the coordinates of the shared axes in `names`, returning
    `(a', b', coords)`. Two differing **categorical** label sets are
    inner-joined (both operands reindexed to the intersection, in `a`'s order);
    an agreeing coordinate is kept; a coordinate present on only one side rides
    along; a differing **numeric** coordinate or a **kind mismatch** is a
    conflict and is dropped (issue #72).
    """
    coords: dict = {}
    for name in names:
        ca, cb = a.coords.get(name), b.coords.get(name)
        if ca is None and cb is None:
            continue
        if ca is None:
            coords[name] = cb
        elif cb is None:
            coords[name] = ca
        elif isinstance(ca, tuple) and isinstance(cb, tuple) and ca != cb:
            common = tuple(label for label in ca if label in cb)
            a = _reindex_axis(a, name, ca, common)
            b = _reindex_axis(b, name, cb, common)
            coords[name] = common
        elif ca == cb:  # agree (identical labels or numeric coordinate)
            coords[name] = ca
        # else: differing numeric / kind mismatch -> conflict, drop
    return a, b, coords


def _reshape_partitioned(
    x: XTensor, anon: int, named: list, max_anon: int, order: list
) -> XTensor:
    """
    Reshape `x` -- a leading anonymous run of length `anon` then the all-named
    suffix `named` -- onto `[None]*max_anon + order`: permute the named suffix
    into `order`, insert a size-1 axis for each name it lacks, and left-pad the
    anonymous run to `max_anon` (so anonymous axes broadcast positionally,
    right-aligned).
    """
    present = [n for n in order if n in named]
    perm = list(range(anon)) + [anon + named.index(n) for n in present]
    out = x.permute(*perm)
    for pos, name in enumerate(order):
        if name not in named:
            out = out.unsqueeze(anon + pos)
    for _ in range(max_anon - anon):
        out = out.unsqueeze(0)
    return out


def _align_partitioned(a: XTensor, b: XTensor) -> tuple:
    """
    Align two operands whose unnamed axes are all **leading** (issue #75): the
    trailing **named** suffixes align by name (union, transpose-to-match,
    broadcast a missing axis, inner-join differing categorical labels), while
    the leading **anonymous** runs broadcast **positionally** (right-aligned,
    like torch batch dims). Returns `(a', b', names, coords)`.
    """
    ka, kb = _leading_none(a.names), _leading_none(b.names)
    an = list(a.names[ka:])  # named suffix of a (no None)
    bn = list(b.names[kb:])  # named suffix of b
    order = an + [n for n in bn if n not in an]  # named union, a first
    max_anon = max(ka, kb)
    a, b, coords = _reconcile_coords(a, b, order)
    a2 = _reshape_partitioned(a, ka, an, max_anon, order)
    b2 = _reshape_partitioned(b, kb, bn, max_anon, order)
    names = (None,) * max_anon + tuple(order)
    return a2, b2, names, coords


def _align_identical(a: XTensor, b: XTensor) -> tuple:
    """
    Align two operands with the **same** `names` tuple. Their axes already
    correspond 1:1 by name-and-position, so no reshape is needed (positional is
    name-aligned) -- this stays unambiguous even when a `None` is not leading.
    Only the coordinates of the named axes are reconciled. Returns
    `(a', b', coords)`.
    """
    named = dict.fromkeys(n for n in a.names if n is not None)
    return _reconcile_coords(a, b, named)


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
    result whose dims are `result_names`. Each descriptor field is resolved
    independently under its `combine_axes` policy (see `set_options`):

    - `"drop"` -- always drop the field;
    - `"override"` -- keep the left-most operand's value;
    - `"strict"` -- raise `ValueError` on a conflict;
    - `"drop_conflicts"` *(default)* -- keep the value the operands agree on,
      drop it where they conflict (the rule coordinates already follow).

    A field present on only one operand is never a conflict; it is kept
    (unless its policy is `"drop"`).
    """
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
        extra = {}
        for key in {k for one in dicts for k in one}:
            policy = _combine_axes_policy(key)
            if policy == "drop":
                continue
            present = [one[key] for one in dicts if key in one]
            if policy == "override":
                extra[key] = present[0]  # left-most operand naming the field
                continue
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


# -- data-unit algebra (Proposal 0003) ---------------------------------------
#
# Under an active `unit_backend`, a pointwise op transforms the operands' data
# units per its rule below; a dimensionally invalid/ambiguous step drops the
# unit (default) or raises (`unit_policy="strict"`). With no backend it is
# skipped and the unit rides along opaquely via `_carry`.

_UNIT_RULE = {
    "mul": "mul",
    "div": "div",
    "floor_divide": "div",
    "pow": "pow",
    "add": "add",
    "sub": "add",
    "remainder": "add",
    "maximum": "add",
    "minimum": "add",
    "hypot": "add",
    "eq": "cmp",
    "ne": "cmp",
    "lt": "cmp",
    "le": "cmp",
    "gt": "cmp",
    "ge": "cmp",
    "atan2": "drop",
    "logical_and": "drop",
    "logical_or": "drop",
    "logical_xor": "drop",
}


def _unit_of(x: tx.Any) -> tx.Optional[str]:
    """The data unit of `x`, or `None` (a plain tensor/scalar is unitless)."""
    return x.__dict__.get("_data_unit") if isinstance(x, XTensor) else None


def _attach_unit(x: XTensor, operand: tx.Any, op: str) -> XTensor:
    """
    Combine a backend `Unit`/`Quantity` `operand` into `x` (Proposal 0003
    §2.4): its magnitude scales the data, its unit multiplies (`op="mul"`) or
    divides (`op="div"`) `x`'s data unit. A bare `Unit` has magnitude 1, so the
    data is untouched -- but through a fresh view, never `x` itself, so
    `_carry` cannot annotate the original in place.
    """
    magnitude, unit = _units.split_quantity(operand)
    if magnitude == 1.0:
        scaled = x.as_subclass(type(x))
    else:
        scaled = Tensor.mul(x, magnitude)
    current = _unit_of(x)
    combined = (
        _units.mul(current, unit) if op == "mul" else _units.div(current, unit)
    )
    return _carry(x, scaled, _data_unit=combined)


def _unit_strict(invalid: bool, detail: str) -> None:
    """Raise on an invalid unit step under `unit_policy="strict"`."""
    if invalid and _get_option("unit_policy") == "strict":
        raise ValueError(detail)


def _binary_unit(a: tx.Any, b: tx.Any, rule: str) -> tx.Optional[str]:
    """Result data unit for a pointwise op under `rule` (honours policy)."""
    ua, ub = _unit_of(a), _unit_of(b)
    if rule == "mul":
        return _units.mul(ua, ub)
    if rule == "div":
        return _units.div(ua, ub)
    if rule == "pow":
        if isinstance(b, (int, float)):
            return _units.pow_(ua, b)
        _unit_strict(
            ua is not None, "pow: non-scalar exponent on a united value"
        )
        return None
    if rule == "add":
        if _units.equal(ua, ub):
            return ua
        _unit_strict(True, f"incompatible units {ua!r} and {ub!r}")
        return None
    if rule == "cmp":
        _unit_strict(
            not _units.equal(ua, ub), f"comparing units {ua!r} and {ub!r}"
        )
        return None
    return None  # "drop": result is unitless


def _reconcile_units(
    a: tx.Any, b: tx.Any, rule: tx.Optional[str]
) -> tx.Tuple[tx.Any, tx.Any, dict]:
    """
    Apply the data-unit algebra to a pointwise op's operands. For `add`/`cmp`
    of **compatible-but-different** units (e.g. `V` and `mV`), implicitly
    convert the *right* operand to the left's unit (Proposal 0003 §7.2) so the
    values line up before the op; then compute the result unit per `rule` and
    policy. Returns the (possibly rescaled) operands and the `_data_unit`
    override for `_carry`. Inert with no backend / no unit rule.
    """
    if not (_units.active() and rule is not None):
        return a, b, {}
    if rule in ("add", "cmp"):
        ua, ub = _unit_of(a), _unit_of(b)
        if (
            ua is not None
            and ub is not None
            and not _units.equal(ua, ub)
            and _units.compatible(ua, ub)
        ):
            converted = Tensor.mul(b, _units.factor(ub, ua))
            b = _carry(b, converted, _data_unit=ua)
    return a, b, {"_data_unit": _binary_unit(a, b, rule)}


# -- contraction (matmul / einsum / tensordot) unit algebra ------------------
#
# A contraction is a sum of products over one or more axes. For the sum to be
# dimensionally valid each contracted axis must be **unit-uniform** per side;
# its uniform per-position unit then folds into that operand's base, and the
# operands' effective units multiply (Proposal 0003 §4). A non-uniform
# contracted axis is invalid -> drop (default) / raise (strict).


def _axis_uniform_unit(x: tx.Any, axis: int) -> tx.Any:
    """
    The single per-position data unit of `x`'s axis `axis` (`None` when it
    carries no coordinate units), or `_INCOMPATIBLE` when the positions
    disagree -- contracting such an axis is dimensionally invalid.
    """
    if not isinstance(x, XTensor):
        return None
    ndim = x.ndim
    if not -ndim <= axis < ndim:
        return None
    name = x.names[axis]
    if name is None:
        return None
    labels = x.coords.get(name)
    if not labels:
        return None
    return _uniform_unit(labels)


def _contraction_unit(
    operands: tx.Sequence, contracted_axes: tx.Sequence
) -> tx.Optional[str]:
    """
    Base data unit for a contraction: the product over `operands` of each
    operand's base unit and the uniform per-position unit of each of its
    contracted axes (`contracted_axes[i]` lists the summed axes of
    `operands[i]`). A non-uniform contracted axis drops the unit (default) or
    raises (`unit_policy="strict"`).
    """
    total = None
    for operand, axes in zip(operands, contracted_axes):
        effective = _unit_of(operand)
        for axis in axes:
            unit = _axis_uniform_unit(operand, axis)
            if unit is _INCOMPATIBLE:
                _unit_strict(
                    True, "contracting an axis with non-uniform units"
                )
                return None
            effective = _units.mul(effective, unit)
        total = _units.mul(total, effective)
    return total


def _matmul_contracted_axes(na: int, nb: int) -> tx.Tuple[int, int]:
    """The contracted axis of each operand under `matmul` broadcasting."""
    if na == 1 and nb == 1:
        return 0, 0  # dot product
    if na == 1:
        return 0, -2  # [k] @ [..., k, n]
    if nb == 1:
        return -1, 0  # [..., m, k] @ [k]
    return -1, -2  # [..., m, k] @ [..., k, n]


def _einsum_contracted_axes(
    equation: str, operands: tx.Sequence
) -> tx.Optional[list]:
    """
    Per-operand lists of contracted (summed) axis indices for
    `einsum(equation, *operands)` -- a subscript that does **not** appear in
    the output. Returns `None` for anything this simple parser can't handle
    (most notably an ellipsis), so the caller falls back to base units only.
    """
    if "." in equation:
        return None
    if "->" in equation:
        parts = equation.split("->")
        if len(parts) != 2:
            return None
        in_part, out_part = parts
    else:
        in_part, out_part = equation, None
    in_subscripts = [s.strip() for s in in_part.split(",")]
    if len(in_subscripts) != len(operands):
        return None
    for subscript, operand in zip(in_subscripts, operands):
        if subscript and not subscript.isalpha():
            return None
        if len(subscript) != getattr(operand, "ndim", len(subscript)):
            return None
    if out_part is None:
        counts: dict = {}
        for subscript in in_subscripts:
            for letter in subscript:
                counts[letter] = counts.get(letter, 0) + 1
        out_letters = {c for c, n in counts.items() if n == 1}
    else:
        out_subscript = out_part.strip()
        if out_subscript and not out_subscript.isalpha():
            return None
        out_letters = set(out_subscript)
    return [
        [i for i, letter in enumerate(subscript) if letter not in out_letters]
        for subscript in in_subscripts
    ]


def _binary(
    a: tx.Any, b: tx.Any, base: tx.Callable, args, kwargs, rule=None
) -> tx.Any:
    # `x * u.mm` (a unit operand) is handled earlier, at the operator dunders
    # (§2.4); here both operands are ordinary values. Reconcile units first --
    # this may rescale `b` (implicit V->mV-style conversion) -- then run the op
    # on the reconciled operands.
    a, b, unit_kw = _reconcile_units(a, b, rule)
    if isinstance(a, XTensor) and isinstance(b, XTensor):
        a_names, b_names = a.names, b.names
        a_has = any(n is not None for n in a_names)
        b_has = any(n is not None for n in b_names)
        # Both carry names -> align by name. An all-unnamed operand has nothing
        # to align on and behaves like a plain tensor (positional, below).
        if a_has and b_has:
            if a_names == b_names:
                # identical layout -> axes already correspond 1:1; positional
                # is name-aligned, unambiguous even with a non-leading `None`.
                a2, b2, coords = _align_identical(a, b)
                names = a_names
            elif not (_anon_leading(a_names) and _anon_leading(b_names)):
                # a `None` sits after a named axis: aligning by name is
                # ambiguous and silent positional would mis-pair (issue #75).
                raise ValueError(
                    "pointwise op on partially-named tensors whose unnamed "
                    "axes are not all leading is ambiguous; name every axis "
                    "(refine_names) or move the unnamed axes to the front"
                )
            elif None in a_names or None in b_names:
                a2, b2, names, coords = _align_partitioned(a, b)
            else:
                a2, b2, names, coords = _align_by_name(a, b)
            result = base(a2, b2, *args, **kwargs)
            meta = _merge_axis_meta((a, b), names)
            return _carry(
                a,
                result,
                _axis_names=names,
                _coords=_pack_coords(coords),
                _axis_meta=meta,
                **unit_kw,
            )
    # positional fallback (a plain tensor / scalar operand, or an all-unnamed
    # XTensor -- which behaves like a plain tensor)
    result = base(a, b, *args, **kwargs)
    if not isinstance(result, Tensor):
        return result
    ref = a if isinstance(a, XTensor) else b
    names = _broadcast_batch_names(_names_of(a), _names_of(b))
    coords = (
        _coords_for(ref, names)
        if result.ndim == getattr(ref, "ndim", -1)
        else {}
    )
    meta = _merge_axis_meta((a, b), names)
    return _carry(
        ref,
        result,
        _axis_names=names,
        _coords=coords,
        _axis_meta=meta,
        **unit_kw,
    )


def _make_pointwise(name: str) -> None:
    """Register a broadcast-by-name override for a binary/pointwise op."""
    base = _torch_func(name)
    rule = _UNIT_RULE.get(name)

    def _op(a: tx.Any, b: tx.Any, *args, **kwargs) -> tx.Any:
        return _binary(a, b, base, args, kwargs, rule)

    registered = XTensor.overrides(base)(_op)
    # Operators (`a + b`, `a == b`, ...) dispatch with the bound method
    # `Tensor.<name>` -- a different callable than the function `torch.<name>`
    # -- so register both (as for `matmul`).
    method = getattr(Tensor, name, None)
    if base is not None and method is not None and method is not base:
        XTensor._OVERRIDES[method] = registered
    # `**` dispatches `Tensor.__pow__`, which is *not* `Tensor.pow`, so the
    # operator would otherwise miss the override (unlike `+`/`*`/...).
    if name == "pow":
        dunder = getattr(Tensor, "__pow__", None)
        if base is not None and dunder is not None:
            XTensor._OVERRIDES[dunder] = registered


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


# -- transcendental functions (require a dimensionless argument) --------------
#
# `exp`/`log`/`sin`/... are only defined on dimensionless numbers, so under an
# active backend a united argument drops its unit (default) or raises
# (`unit_policy="strict"`); the result is dimensionless. With no backend the
# unit rides along opaquely, unchanged. (These are elementwise, so names and
# coordinates carry through as usual.)


def _make_transcendental(name: str) -> None:
    base = _torch_func(name)
    if base is None:
        return

    def _op(input: tx.Any, *args, **kwargs) -> tx.Any:
        result = base(input, *args, **kwargs)
        if not _units.active():
            return _carry(input, result)
        unit = _unit_of(input)
        _unit_strict(
            not _units.dimensionless(unit),
            f"{name}: expected a dimensionless argument, got unit {unit!r}",
        )
        return _carry(input, result, _data_unit=None)

    XTensor.overrides(base)(_op)


_TRANSCENDENTAL = (
    "exp", "expm1", "log", "log2", "log10", "log1p",
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "sigmoid", "erf", "erfc",
)  # fmt: skip
for _transcendental_name in _TRANSCENDENTAL:
    _make_transcendental(_transcendental_name)
