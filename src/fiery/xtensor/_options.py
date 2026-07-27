"""Global options for `fiery.xtensor`, settable as a context manager.

`set_options` doubles as a permanent setter (`set_options(combine_axes=...)`)
and a scoped context manager (`with set_options(combine_axes="strict"): ...`),
mirroring `xarray.set_options`.
"""

from __future__ import annotations

import typing_extensions as tx

#: The descriptor-merge policies. `"raise"` is an alias for `"strict"`.
_POLICIES = ("drop_conflicts", "strict", "raise", "override", "drop")

#: Reserved key: the default policy for descriptor fields not named explicitly.
_DEFAULT_KEY = "*"

#: The known unit backends (Proposal 0003). `None` = no unit semantics.
_UNIT_BACKENDS = (None, "pint")

#: What happens on a dimensionally invalid/ambiguous unit step (Proposal 0003).
_UNIT_POLICIES = ("drop", "strict")

#: Boundary conditions understood by `.interp` (Proposal 0004). These mirror
#: `fiery.interpol`'s vocabulary (name -> out-of-bound behaviour); the default,
#: ``"replicate"``, clamps to the edge value.
_INTERP_BOUNDS = (
    "zero",
    "zeros",
    "replicate",
    "nearest",
    "border",
    "dct1",
    "mirror",
    "dct2",
    "reflect",
    "dst1",
    "antimirror",
    "dst2",
    "antireflect",
    "dft",
    "wrap",
)

#: Live option values. Read through `get_option`; write only via `set_options`.
_OPTIONS = {
    "combine_axes": "drop_conflicts",
    "unit_backend": None,
    "unit_policy": "drop",
    "interp_bound": "replicate",
    "interp_extrapolate": True,
}


def _validate_combine_axes(value: tx.Any) -> None:
    if isinstance(value, str):
        policies = {value}
    elif isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(
                "combine_axes keys must be strings (descriptor field names, "
                f"or {_DEFAULT_KEY!r} for the default)"
            )
        policies = set(value.values())
    else:
        raise ValueError(
            "combine_axes must be a policy str or a {field: policy} dict, "
            f"got {type(value).__name__}"
        )
    unknown = policies - set(_POLICIES)
    if unknown:
        raise ValueError(
            f"invalid combine_axes policy {sorted(unknown)}; "
            f"valid: {list(_POLICIES)}"
        )


def _validate_unit_backend(value: tx.Any) -> None:
    if value not in _UNIT_BACKENDS:
        raise ValueError(
            f"invalid unit_backend {value!r}; valid: {list(_UNIT_BACKENDS)}"
        )
    if value == "pint":
        # Fail at *set* time (deterministic) rather than on first use.
        try:
            import pint  # noqa: F401
        except ImportError:
            raise ValueError(
                "unit_backend='pint' requires pint; install "
                "fiery-xtensor[units] (or `pip install pint`)"
            ) from None


def _validate_unit_policy(value: tx.Any) -> None:
    if value not in _UNIT_POLICIES:
        raise ValueError(
            f"invalid unit_policy {value!r}; valid: {list(_UNIT_POLICIES)}"
        )


def _validate_interp_bound(value: tx.Any) -> None:
    # An int order is passed straight through to the backend; a string is
    # checked against the known names so a typo fails at set time.
    if isinstance(value, int):
        return
    if value not in _INTERP_BOUNDS:
        raise ValueError(
            f"invalid interp_bound {value!r}; valid: {list(_INTERP_BOUNDS)}"
        )


def _validate_interp_extrapolate(value: tx.Any) -> None:
    if not isinstance(value, (bool, int)):
        raise ValueError(
            f"invalid interp_extrapolate {value!r}; expected a bool or int"
        )


#: Per-option validators (options without one accept any value).
_VALIDATORS = {
    "combine_axes": _validate_combine_axes,
    "unit_backend": _validate_unit_backend,
    "unit_policy": _validate_unit_policy,
    "interp_bound": _validate_interp_bound,
    "interp_extrapolate": _validate_interp_extrapolate,
}


def get_option(name: str) -> tx.Any:
    """The current value of option `name`."""
    return _OPTIONS[name]


def combine_axes_policy(field: str) -> str:
    """
    The effective `combine_axes` policy for a single descriptor `field`.

    With a plain-string option every field shares it; with a `{field: policy}`
    dict a field uses its own entry, else the `"*"` default, else
    `"drop_conflicts"`. `"raise"` is normalised to `"strict"`.
    """
    spec = _OPTIONS["combine_axes"]
    if isinstance(spec, str):
        policy = spec
    else:
        policy = spec.get(field, spec.get(_DEFAULT_KEY, "drop_conflicts"))
    return "strict" if policy == "raise" else policy


class set_options:
    """
    Set one or more options, globally or for a `with` block.

    Options:

    - **`combine_axes`** -- how axis **descriptors** (`type`/`orientation`/any
      custom field) combine when two operands meet in a name-aware op
      (broadcast, alignment, `cat`/`stack`/`matmul`/`einsum`/…). Either a
      single policy applied to every field, or a `{field: policy}` dict for
      per-field control (with `"*"` as the default for unlisted fields).
      Policies:

        - `"drop_conflicts"` *(default)* -- keep a field the operands agree on,
          drop it where they conflict;
        - `"strict"` (alias `"raise"`) -- raise `ValueError` on a conflict;
        - `"override"` -- keep the left operand's value;
        - `"drop"` -- always drop the field.

    Used as a permanent setter or a scoped context manager::

        set_options(combine_axes="strict")                    # until changed
        with set_options(combine_axes="strict"):              # for this block
            ...
        # per-field: drop everything, but a clashing `type` is an error
        with set_options(combine_axes={"*": "drop", "type": "raise"}):
            ...

    - **`unit_backend`** -- the physical-unit engine for **data units**
      (Proposal 0003): `None` *(default)* means units are inert opaque strings;
      `"pint"` enables validation/algebra/conversion (and is rejected at set
      time if pint is not installed).
    - **`unit_policy`** -- what a dimensionally invalid/ambiguous step does:
      `"drop"` *(default)* silently drops the unit, `"strict"` raises.
    - **`interp_bound`** -- the default boundary condition for
      [`interp`][fiery.xtensor.XTensor.interp] (Proposal 0004), i.e. how an
      out-of-range query is resolved. `"replicate"` *(default)* clamps to the
      edge value; other names (`"wrap"`, `"reflect"`, `"mirror"`, `"zero"`, …)
      mirror `fiery.interpol`. A per-call `bound=` overrides it.
    - **`interp_extrapolate`** -- whether `interp` extrapolates past the ends
      (`True` *(default)*; with `interp_bound="replicate"` this is the clamp).
      A per-call `extrapolate=` overrides it.
    """

    def __init__(self, **options: tx.Any) -> None:
        self._previous = {}
        for name, value in options.items():
            if name not in _OPTIONS:
                raise ValueError(
                    f"unknown option {name!r}; "
                    f"valid options: {sorted(_OPTIONS)}"
                )
            validator = _VALIDATORS.get(name)
            if validator is not None:
                validator(value)
            self._previous[name] = _OPTIONS[name]
        _OPTIONS.update(options)

    def __enter__(self) -> set_options:
        return self

    def __exit__(self, *exc: tx.Any) -> None:
        _OPTIONS.update(self._previous)
