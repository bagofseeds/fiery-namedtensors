"""COMBINE: multi-operand ops whose first argument is a *sequence* of
tensors (`cat`/`stack`, the promoting `hstack`/`vstack`/`dstack`,
`matmul`/`mm`/`bmm`, `einsum`/`tensordot`). Axis names are reconciled
positionally across the operands (`_reconcile_axis_names`), and axis
**descriptors** are merged via `_merge_axis_meta`. The contraction-specific
unit algebra (`matmul`/`einsum`'s dimensional validity check) lives here
too, since it has no other caller.

Importing this module registers its overrides onto `XTensor` (see
CLAUDE.md, "How the subclassing works").
"""

from __future__ import annotations

import torch
import typing_extensions as tx
from torch import Tensor

from fiery.xtensor import _units
from fiery.xtensor._common import _carry, _resolve_axis
from fiery.xtensor._compat import torch_func as _torch_func
from fiery.xtensor._meta import (
    _INCOMPATIBLE,
    _broadcast_batch_names,
    _merge_axis_meta,
    _uniform_unit,
    _unit_strict,
)
from fiery.xtensor._selection import _pack_coords
from fiery.xtensor._tensors import XTensor, _coords_of, _names_of, _unit_of

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
            unit_kw["_data_units"] = _contraction_unit(
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
            unit_kw["_data_units"] = base
        else:
            unit_kw["_data_units"] = _contraction_unit(flat, axes)
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
        unit_kw["_data_units"] = _contraction_unit(
            (a, b), (sorted(a_contracted), sorted(b_contracted))
        )
    return _carry(
        ref, result, _axis_names=names, _coords={}, _axis_meta=meta, **unit_kw
    )


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
