"""Selection math with no `XTensor`/`Coordinate` dependency of its own:
label matching, numeric tolerance/mode resolution, the O(1) closed-form
`.sel` fast path (with its exact-search fallback), and value-range slicing.
Kept separate from `_tensors.py` because it is provably acyclic against the
core class -- everything here works on plain labels/tensors/floats.
"""

from __future__ import annotations

import enum
import math

import torch
import typing_extensions as tx
from torch import Tensor

from fiery.xtensor import _arrayutils as arrayutils
from fiery.xtensor import _units
from fiery.xtensor._common import LabelsT

if tx.TYPE_CHECKING:
    from fiery.xtensor._arrayutils import _SmartSlicerT
    from fiery.xtensor._tensors import Coordinate


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
    """Whether a `coords[dim]` value is an **explicit** numeric coordinate --
    a tensor of positions, or a `{"value": ...}` mapping wrapping one --
    rather than a sequence of labels."""
    if isinstance(spec, Tensor):
        return True
    return isinstance(spec, tx.Mapping) and "value" in spec


def _check_unambiguous_coord_spec(spec: tx.Any) -> None:
    """Raise if a mapping spec mixes an explicit `value` with a compact
    `spacing`/`origin` -- ambiguous, never silently pick one over the
    other."""
    if (
        isinstance(spec, tx.Mapping)
        and "value" in spec
        and ("spacing" in spec or "origin" in spec)
    ):
        raise ValueError(
            "coords: a spec cannot mix an explicit 'value' with a compact "
            "'spacing'/'origin' -- give one or the other"
        )


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


def _check_curvilinear_shape(
    key: str, coord: Coordinate, dims: tuple, shape: tuple, names: tuple
) -> None:
    """
    Validate an explicit **curvilinear** coordinate's stored shape (issue
    #82) against its spanned dims' current sizes, at construction time --
    the multi-dim counterpart of `_check_nondim_len`.
    """
    raw = dict.__getitem__(coord, "value")
    expected = tuple(shape[names.index(dim)] for dim in dims)
    if tuple(raw.shape) != expected:
        raise ValueError(
            f"coords: {key!r} spans dims {dims!r} of shape {expected}, but "
            f"its values have shape {tuple(raw.shape)}"
        )


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


#: Relative tolerance for an "exact" numeric-coordinate match (floats).
_EXACT_MATCH_REL = 1e-6

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


def _compact_range_slice(
    coord: "Coordinate",
    lo: tx.Optional[float],
    hi: tx.Optional[float],
    size: int,
) -> slice:
    """
    The compact-coordinate half of `_numeric_select_range` -- never
    materialises `["value"]` (issue #110's O(1) property extends to range
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
