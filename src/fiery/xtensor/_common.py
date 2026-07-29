"""Name/axis plumbing shared across `fiery.xtensor`, with no torch-subclass
knowledge of its own -- dim-name resolution, axis descriptors, and the
`_carry` metadata-propagation helper every override relies on.
"""

from __future__ import annotations

# dependencies
import typing_extensions as tx
from torch import Tensor

# internals
from fiery.xtensor._compat import EllipsisType

if tx.TYPE_CHECKING:
    from fiery.xtensor._tensors import XTensor

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

    Wraps [`_resolve_axis`][fiery.xtensor._common._resolve_axis]: a
    single specifier is resolved directly; a `tuple`/`list` is resolved
    element-wise (keeping its container type); anything else passes through.
    """
    if isinstance(dim, str):
        return _resolve_axis(names, dim)
    if isinstance(dim, (tuple, list)):
        return type(dim)(_resolve_axis(names, d) for d in dim)
    return dim


def _either_dict_or_kwargs(
    positional: tuple, kwargs: dict, funcname: str
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

    `positional` is captured via the caller's own `*args` (not a named
    `indexers=` parameter) so this mapping itself can never collide with a
    query name either -- `x.sel(indexers=5.0)` on a dim literally called
    "indexers" reaches `**kwargs` exactly as before this escape hatch
    existed, since a bare `*args` slot can only ever be filled positionally.
    """
    if len(positional) > 1:
        raise TypeError(
            f"{funcname}: at most one positional argument (an indexers "
            "mapping) is accepted"
        )
    if not positional:
        return dict(kwargs)
    if kwargs:
        raise ValueError(
            f"{funcname}: pass indexers as a dict OR as keyword arguments, "
            "not both"
        )
    return dict(positional[0])


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
