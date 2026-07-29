"""Physical-unit support (Proposal 0003 — data units).

A **backend** provides the unit *algebra*; the tensor data is never wrapped in
a `pint.Quantity` (that is not a `torch.Tensor`, so it would not survive
`torch.*`/`nn`/autograd) — we annotate the `XTensor` subclass and feed the
backend `Unit` objects instead. Units are stored as **canonical strings**, so
the metadata is backend-independent and picklable.

Selected by the `unit_backend` option (`None` by default → units are inert
opaque strings; `"pint"` → validation/algebra/conversion). This module is a
thin dispatch over the active backend, so adding another library (astropy,
unyt) is a matter of teaching these functions a new name.
"""

from __future__ import annotations

import re

import typing_extensions as tx

from fiery.xtensor._options import get_option

#: Cached `(pint_module, registry)`; built lazily on first use.
_PINT = None


def _pint() -> tx.Any:
    global _PINT
    if _PINT is None:
        import pint

        _PINT = (pint, pint.UnitRegistry())
    return _PINT


def active() -> tx.Optional[str]:
    """The current unit backend name (`None` when units are inert)."""
    return get_option("unit_backend")


def normalise(unit: tx.Any) -> tx.Any:
    """
    Canonical string form of `unit`; raise `ValueError` if it cannot be parsed.
    Identity (returns `unit` unchanged) when no backend is active.
    """
    if unit is None or active() != "pint":
        return unit
    _, ureg = _pint()
    try:
        return str(ureg.Unit(unit))
    except Exception as exc:
        raise ValueError(f"invalid unit {unit!r}: {exc}") from None


def equal(a: tx.Any, b: tx.Any) -> bool:
    """Whether two units are the same (normalised equality under a backend)."""
    if a is None or b is None:
        return a is b
    if active() != "pint":
        return a == b
    _, ureg = _pint()
    try:
        return ureg.Unit(a) == ureg.Unit(b)
    except Exception:
        return a == b


def factor(frm: str, to: str) -> float:
    """
    The scalar to multiply data by to convert it from unit `frm` to unit `to`.
    Requires an active backend; raises if the units are incompatible.
    """
    if active() != "pint":
        raise ValueError("unit conversion requires unit_backend='pint'")
    _, ureg = _pint()
    return ureg.Quantity(1.0, frm).to(to).magnitude


# -- unit algebra (used by the arithmetic wiring; Proposal 0003 phase 2) ------


def mul(a: tx.Any, b: tx.Any) -> tx.Any:
    """Product of two units (`None` = dimensionless)."""
    if a is None:
        return b
    if b is None:
        return a
    _, ureg = _pint()
    return str(ureg.Unit(a) * ureg.Unit(b))


def div(a: tx.Any, b: tx.Any) -> tx.Any:
    """Quotient of two units (`None` = dimensionless)."""
    if b is None:
        return a
    _, ureg = _pint()
    numerator = ureg.dimensionless if a is None else ureg.Unit(a)
    return str(numerator / ureg.Unit(b))


def compatible(a: tx.Any, b: tx.Any) -> bool:
    """Whether two units share a dimensionality (are inter-convertible)."""
    if a is None or b is None:
        return a is None and b is None
    if active() != "pint":
        return a == b
    _, ureg = _pint()
    return ureg.Unit(a).dimensionality == ureg.Unit(b).dimensionality


def pow_(a: tx.Any, n: tx.Any) -> tx.Any:
    """A unit raised to the (scalar) power `n`; `None` stays `None`."""
    if a is None:
        return None
    _, ureg = _pint()
    return str(ureg.Unit(a) ** n)


def dimensionless(a: tx.Any) -> bool:
    """Whether a unit is dimensionless (`None` counts as dimensionless)."""
    if a is None:
        return True
    if active() != "pint":
        return False
    _, ureg = _pint()
    return ureg.Unit(a).dimensionless


# -- recognising the backend's own unit / quantity objects (for `x * mm`) -----


def is_unit_like(obj: tx.Any) -> bool:
    """Whether `obj` is a `Unit`/`Quantity` of the active backend."""
    if active() != "pint":
        return False
    pint, _ = _pint()
    return isinstance(obj, (pint.Unit, pint.Quantity))


def looks_like_unit(obj: tx.Any) -> bool:
    """
    Whether `obj` could be a **unit** spec on its own: `None`, a string, or
    the active backend's own `Unit`/`Quantity` object -- never a bare number.
    A unit is one of those three things or nothing at all, so this fully
    disambiguates a 2-tuple: `(value, unit)` iff the second element looks
    like a unit, otherwise it is a raw (possibly vector) value, not a pair.
    """
    return obj is None or isinstance(obj, str) or is_unit_like(obj)


def split_quantity(obj: tx.Any) -> tx.Tuple[tx.Any, str]:
    """
    Decompose a backend unit/quantity into `(magnitude, unit_string)`: a bare
    `Unit` is `(1.0, "unit")`; a `Quantity` is `(its magnitude, its units)`.
    """
    pint, _ = _pint()
    if isinstance(obj, pint.Unit):
        return 1.0, str(obj)
    return obj.magnitude, str(obj.units)


# -- the "magic dict" family (Proposal 0001) ---------------------------------
#
# Metadata dicts (a unitful value, a coordinate, ...) are plain `dict`s that
# also expose *whitelisted* keys as attributes and preserve tensor values
# untouched. Item access is canonical for every key; attribute sugar covers
# only keys that do not shadow the `dict` API (never `values`/`keys`/`items`).

#: Keys reachable as attributes (they don't collide with `dict` methods).
_MAGIC_ATTR_KEYS = frozenset(
    {"value", "unit", "name", "spacing", "origin", "type", "orientation"}
)


class MagicDict(dict):
    """A `dict` whose whitelisted keys are also attribute-accessible."""

    def __getattr__(self, name: str) -> tx.Any:
        if name in _MAGIC_ATTR_KEYS and name in self:
            return self[name]
        raise AttributeError(name)


class Unitful(MagicDict):
    """
    A `{value, unit}` pair. `value` may be a scalar **or** a live 0-rank
    tensor (kept untouched, so a learnable value keeps its autograd graph);
    `unit` is a canonical string. Read as `q["value"]` / `q.unit`; convert
    with `.to(unit)`.
    """

    def to(self, unit: tx.Any) -> "Unitful":
        """Convert to `unit` (needs a backend), rescaling value and unit."""
        target = normalise(unit)
        scale = factor(self["unit"], target)
        return Unitful(value=self["value"] * scale, unit=target)


#: A leading numeric literal (optional sign, decimal, exponent) followed by
#: whatever's left (the unit suffix, possibly empty/whitespace-only).
_VALUE_UNIT_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(.*)$"
)


def _parse_unit_string(text: str) -> tx.Tuple[tx.Any, str]:
    """
    `"0.5mm"` -> `(0.5, "mm")`, `"mm"` -> `(1, "mm")`. With a backend, pint
    parses the full value+unit grammar (algebra, aliases, ...). Without one,
    a regex splits a leading numeric literal from the unit suffix -- not a
    real unit parser, but enough to not silently discard the magnitude the
    way unconditionally returning `(1, text)` used to: `"0.5mm"` no longer
    becomes `(1, "0.5mm")`, just `(1, text)` for a string with no leading
    number at all (a bare unit, e.g. `"mm"`).
    """
    if active() != "pint":
        match = _VALUE_UNIT_RE.match(text)
        if match:
            return float(match.group(1)), match.group(2).strip()
        return 1, text
    _, ureg = _pint()
    try:
        quantity = ureg.Quantity(text)
        return quantity.magnitude, str(quantity.units)
    except Exception:
        return 1, normalise(text)


def as_unitful(obj: tx.Any) -> Unitful:
    """
    Coerce an accepted *unitful* input into a `Unitful`: a `Unitful`, a
    `{"value", "unit"}` dict, a `(value, unit)` tuple, a backend `Unit`/
    `Quantity`, a unit string, or a bare value (dimensionless). A united
    `XTensor` value is handled by the caller (it needs the tensor type).

    A 2-tuple is `(value, unit)` only when its second element `looks_like_unit`
    (`None`/a string/a backend `Unit`/`Quantity`) -- a 2-tuple of bare numbers
    (e.g. a 2-component vector `spacing`) is never mistaken for `(value,
    unit)`; it falls through as a raw value instead (issue #93).
    """
    if isinstance(obj, Unitful):
        return obj
    if isinstance(obj, dict) and "value" in obj:
        return Unitful(value=obj["value"], unit=normalise(obj.get("unit", "")))
    if isinstance(obj, tuple) and len(obj) == 2 and looks_like_unit(obj[1]):
        return Unitful(value=obj[0], unit=normalise(obj[1]))
    if is_unit_like(obj):
        magnitude, unit = split_quantity(obj)
        return Unitful(value=magnitude, unit=normalise(unit))
    if isinstance(obj, str):
        value, unit = _parse_unit_string(obj)
        return Unitful(value=value, unit=unit)
    return Unitful(value=obj, unit=normalise(""))
