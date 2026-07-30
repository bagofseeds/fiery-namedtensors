"""POINTWISE (BY NAME): binary arithmetic/comparison operators and
dimensionless transcendental functions. Two fully-named operands align by
dim name (xarray-style, `join="inner"` on shared labelled dims) rather than
position; an unnamed or non-`XTensor` operand falls back to positional
broadcasting.

Importing this module registers its overrides onto `XTensor` (see
CLAUDE.md, "How the subclassing works").
"""

from __future__ import annotations

import torch
import typing_extensions as tx
from torch import Tensor

from fiery.xtensor import _units
from fiery.xtensor._common import _carry
from fiery.xtensor._compat import torch_func as _torch_func
from fiery.xtensor._meta import (
    _broadcast_batch_names,
    _merge_axis_meta,
    _unit_strict,
)
from fiery.xtensor._selection import _pack_coords
from fiery.xtensor._tensors import XTensor, _coords_for, _names_of, _unit_of

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
    convert the *right* operand to the left's unit (Proposal 0003 §7) so the
    values line up before the op; then compute the result unit per `rule` and
    policy. Returns the (possibly rescaled) operands and the `_data_units`
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
            b = _carry(b, converted, _data_units=ua)
    return a, b, {"_data_units": _binary_unit(a, b, rule)}


def _has_names(x: tx.Any) -> bool:
    """Whether `x` is an `XTensor` carrying at least one named axis."""
    return isinstance(x, XTensor) and any(n is not None for n in x.names)


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
    # `ref` stays the plain isinstance-based pick: it's the donor `_carry`
    # uses for dtype/subclass/`__dict__` (data unit included, when no
    # backend override applies), and `a`'s claim to that role has nothing to
    # do with whether it happens to carry names.
    ref = a if isinstance(a, XTensor) else b
    # `cref` -- the coordinate/name source -- prefers whichever operand
    # actually carries names: an all-unnamed `XTensor` (e.g. a plain tensor
    # merely wrapped bare) has nothing more to offer there than a plain
    # tensor, so it shouldn't out-rank a named operand just for being `a`
    # (issue #157). The `a_has and b_has` gate above guarantees at most one
    # of `a`/`b` has real names here, so there's no reconciliation to do --
    # just pick the informative side.
    if _has_names(a):
        cref = a
    elif _has_names(b):
        cref = b
    else:
        cref = ref
    names = _broadcast_batch_names(_names_of(a), _names_of(b))
    coords = (
        _coords_for(cref, names)
        if result.ndim == getattr(cref, "ndim", -1)
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
        return _carry(input, result, _data_units=None)

    XTensor.overrides(base)(_op)


_TRANSCENDENTAL = (
    "exp", "expm1", "log", "log2", "log10", "log1p",
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "sigmoid", "erf", "erfc",
)  # fmt: skip
for _transcendental_name in _TRANSCENDENTAL:
    _make_transcendental(_transcendental_name)
