"""REDUCTIONS and SCANS: `_make_reduction` (`sum`/`mean`/`amax`/...) and
friends drop the reduced axis' name+coords (keeping them under `keepdim`);
a reduce-all drops to an unnamed scalar. Irregular/namedtuple reducers
(`std`/`var`, `min`/`max`, `sort`/`topk`, `cumsum`/`cummax`/...) share the
same axis bookkeeping.

Importing this module registers its overrides onto `XTensor` (see
CLAUDE.md, "How the subclassing works").
"""

from __future__ import annotations

import torch
import typing_extensions as tx
from torch import Tensor

from fiery.xtensor import _units
from fiery.xtensor._common import (
    _carry,
    _query_positions,
    _resolve_axis,
    _resolve_dims,
)
from fiery.xtensor._compat import torch_func as _torch_func
from fiery.xtensor._meta import (
    _INCOMPATIBLE,
    _broadcast_batch_names,
    _uniform_unit,
    _unit_strict,
)
from fiery.xtensor._tensors import (
    XTensor,
    _coords_dropping,
    _coords_for,
    _names_of,
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
    [`_resolve_dims`][fiery.xtensor._common._resolve_dims] unchanged.
    """
    has_query = isinstance(dim, dict) or (
        isinstance(dim, (tuple, list))
        and any(isinstance(d, dict) for d in dim)
    )
    if not has_query:
        return _resolve_dims(input.names, dim)
    positions = _query_positions(input, dim)
    return positions[0] if len(positions) == 1 else positions


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
