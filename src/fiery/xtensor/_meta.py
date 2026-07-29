"""Name/axis-descriptor/unit reconciliation shared by two or more operator
modules (`_reduce`, `_combine`, `_gather`, `_pointwise`) -- nothing here is
specific to any one op family.
"""

from __future__ import annotations

import typing_extensions as tx

from fiery.xtensor import _units
from fiery.xtensor._common import LabelsT
from fiery.xtensor._options import combine_axes_policy as _combine_axes_policy
from fiery.xtensor._options import get_option as _get_option
from fiery.xtensor._selection import _label_unit
from fiery.xtensor._tensors import XTensor

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


def _unit_strict(invalid: bool, detail: str) -> None:
    """Raise on an invalid unit step under `unit_policy="strict"`."""
    if invalid and _get_option("unit_policy") == "strict":
        raise ValueError(detail)
