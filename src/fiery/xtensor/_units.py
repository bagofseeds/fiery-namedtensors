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


def units_available() -> bool:
    """Whether a real (non-`None`) unit backend is active."""
    return active() is not None


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


def split_quantity(obj: tx.Any) -> tx.Tuple[tx.Any, str]:
    """
    Decompose a backend unit/quantity into `(magnitude, unit_string)`: a bare
    `Unit` is `(1.0, "unit")`; a `Quantity` is `(its magnitude, its units)`.
    """
    pint, _ = _pint()
    if isinstance(obj, pint.Unit):
        return 1.0, str(obj)
    return obj.magnitude, str(obj.units)
