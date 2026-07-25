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

#: Live option values. Read through `get_option`; write only via `set_options`.
_OPTIONS = {
    "combine_axes": "drop_conflicts",
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


#: Per-option validators (options without one accept any value).
_VALIDATORS = {
    "combine_axes": _validate_combine_axes,
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

    - **`combine_axes`** -- how axis **descriptors** (`type`/`unit`/
      `orientation`/custom) combine when two operands meet in a name-aware op
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
        # per-field: drop everything, but a clashing `unit` is an error
        with set_options(combine_axes={"*": "drop", "unit": "raise"}):
            ...
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
