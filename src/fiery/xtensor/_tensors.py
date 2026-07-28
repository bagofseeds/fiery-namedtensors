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

        Only entries that are still valid are returned -- their dimension must
        be named on this tensor (and, for labels, its size must match the label
        count) -- so stale metadata propagated onto a shape-changing op is
        hidden.

        Stored internally as `{name: (dims, coord)}` (Proposal 0005): a
        **dimension** coordinate has `dims == (name,)` (it *is* the dim's
        index, so `.sel(name=...)` works); a **non-dimension** coordinate
        (disambiguated by key: `name` is not itself a dim) rides along some
        other dim, `dims == (dim,)`, and is not an index. A wider `dims`
        (multi-dim / affine coordinates) is a later slice.
        """
        names = self.names
        valid = {}
        stored = self.__dict__.get("_coords") or {}
        for name, (dims, coord) in stored.items():
            if any(dim not in names for dim in dims):
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
                # rides along `dim` rather than indexing it.
                dim, coord = _parse_nondim_coord(key, spec, names)
                size = self.shape[names.index(dim)]
                _check_nondim_len(key, dim, coord, size)
                unified[key] = (dim,), coord
                continue
            if _is_compact_coord(spec) or _is_explicit_coord(spec):
                unified[key] = _pack_coord(key, _make_coordinate(spec))
                continue
            size = self.shape[names.index(key)]
            labels = tuple(spec)
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
        mode: tx.Optional[str] = None,
        tolerance: tx.Any = None,
        method: tx.Optional[str] = None,
        **indexers: tx.Any,
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
        implies an unbounded snap unless a `tolerance` is given. `method=` is
        an xarray-compatible alias for `mode` (`nearest`→round, `pad`/`ffill`→
        prev, `backfill`/`bfill`→next); pass one of `mode`/`method`, not both.
        """
        if mode is not None and method is not None:
            raise ValueError("sel: pass either 'mode' or 'method', not both")
        raw = mode if mode is not None else method
        sel_mode = _resolve_sel_mode(raw)
        if tolerance is None:
            # a bare sel is exact; asking for a mode implies an unbounded snap
            tolerance = 0 if raw is None else None
        elif isinstance(tolerance, float) and tolerance == float("inf"):
            tolerance = None  # explicit unbounded
        coords = self.coords
        positional = {}
        for name, label in indexers.items():
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
                try:
                    positions.append(identities.index(one))
                except ValueError:
                    raise ValueError(
                        f"sel: no label {one!r} on dim {name!r}"
                    ) from None
            positional[name] = positions if is_many else positions[0]
        return self.isel(**positional)

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
        it, its coordinate becoming the queried positions. Only **regular**
        (compact `spacing`/`origin`) coordinates are supported for now.
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
        if not coord._compact():
            raise NotImplementedError(
                f"interp on the irregular coordinate {name!r} is not "
                "supported yet (regular spacing/origin only for now; see #73)"
            )
        spacing = dict.__getitem__(coord, "spacing")
        origin = dict.get(coord, "origin")
        unit = spacing["unit"]
        step = spacing["value"]
        base = origin["value"] if origin is not None else 0
        query, is_many = _query_values(target, unit)
        frac = (query - base) / step
        order = _interp_order(method)
        eff_bound = _get_option("interp_bound") if bound is None else bound
        eff_extrap = (
            _get_option("interp_extrapolate")
            if extrapolate is None
            else extrapolate
        )
        raw = _interp_pull(
            self.as_subclass(Tensor), axis, frac, order, eff_bound, eff_extrap
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

    def __getitem__(self, key: tx.Any) -> tx.Any:
        if key == "values" and self._compact():
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

    def to(self, unit: tx.Any) -> "Coordinate":
        """
        Convert the coordinate's **position** unit, rescaling
        `spacing`/`origin` (compact) or the stored `values` (explicit). Needs a
        backend.
        """
        if self._compact():
            out = Coordinate()
            out["spacing"] = dict.__getitem__(self, "spacing").to(unit)
            if "origin" in self:
                out["origin"] = dict.__getitem__(self, "origin").to(unit)
            return out
        return Coordinate(
            values=dict.__getitem__(self, "values").to_unit(unit)
        )


def _as_unitful(obj: tx.Any) -> tx.Any:
    """Coerce a spacing/origin input to a `Unitful`, preserving a tensor."""
    if isinstance(obj, XTensor):
        unit = obj.unit
        if unit is None:
            return _units.Unitful(value=obj, unit=_units.normalise(""))
        return _units.Unitful(value=obj.magnitude, unit=unit)
    return _units.as_unitful(obj)


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


def _make_coordinate(spec: tx.Any) -> Coordinate:
    """Build a `Coordinate` from a compact spec or an explicit tensor."""
    if _is_explicit_coord(spec):
        if isinstance(spec, XTensor) and spec.unit is not None:
            values = spec
        else:
            values = XTensor(spec, unit=_units.normalise(""))
        return Coordinate(values=values)
    coord = Coordinate()
    if "spacing" in spec:
        coord["spacing"] = _as_unitful(spec["spacing"])
    if "origin" in spec:
        coord["origin"] = _as_unitful(spec["origin"])
    return coord


# ---- non-dimension coordinates (Proposal 0005) -----------------------------


def _nondim_coord_len(coord: tx.Any) -> int:
    """The number of positions in a non-dimension coordinate's values."""
    if isinstance(coord, Coordinate):
        return len(dict.__getitem__(coord, "values"))
    return len(coord)


def _parse_nondim_coord(key: str, spec: tx.Any, names: tuple) -> tuple:
    """
    Parse a `(dim, values)` non-dimension coordinate spec into `(dim, coord)`,
    where `coord` is an **explicit** numeric `Coordinate` or a tuple of
    labels. A **compact** (`spacing`/`origin`) spec isn't supported here yet:
    unlike a dimension coordinate, a non-dimension one isn't re-sliced when
    its dim is (Proposal 0005's "no slice-tracking yet") -- for an explicit
    or label coordinate that's caught by the length check on resize, but a
    compact coordinate binds to *any* size, so it would silently rebind to
    the wrong affine after a non-trivial slice instead of raising or
    dropping. Rejecting it here avoids that silent-wrong-values trap; lift
    the restriction once non-dimension coordinates are re-sliced properly.
    """
    if not (
        isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str)
    ):
        raise ValueError(
            f"coords: {key!r} is not an axis; a non-dimension coordinate must "
            "be given as (dim, values)"
        )
    dim, raw = spec
    if dim not in names:
        raise ValueError(f"coords: no axis named {dim!r} in {tuple(names)}")
    if _is_compact_coord(raw):
        raise NotImplementedError(
            f"coords: {key!r} -- a compact (spacing/origin) non-dimension "
            "coordinate isn't supported yet (it wouldn't survive slicing its "
            "dim correctly); use an explicit tensor of values instead"
        )
    coord = _make_coordinate(raw) if _is_explicit_coord(raw) else tuple(raw)
    return dim, coord


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


def _label_name(label: tx.Any) -> tx.Optional[str]:
    """
    A label's **identity** for name-based selection: a `str` is itself, a
    **structured** label (dict) is its `"name"` field, anything else `None`.
    """
    if isinstance(label, str):
        return label
    if isinstance(label, dict):
        return label.get("name")
    return None


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
    `(2, "mm")`, a pint quantity, ...) is converted under an active backend.
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
    exact (up to float epsilon).
    """
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
        if tol is not None:
            cap = tol if tol > 0 else _EXACT_MATCH_REL * max(1.0, abs(target))
            if gap > cap:
                raise ValueError(
                    f"sel: {mode} tick for {one!r} on {name!r} is {gap} away, "
                    f"over tolerance {tol}"
                )
        positions.append(j)
    return positions if is_many else positions[0]


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
    # coordinates go, so its folded unit still applies.
    if dim is not None and result.ndim == ndim:
        return _carry(input, result, _axis_names=input.names, **unit_kw)
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
