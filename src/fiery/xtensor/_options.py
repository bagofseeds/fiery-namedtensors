"""Global options for `fiery.xtensor`, settable as a context manager.

`set_options` doubles as a permanent setter (`set_options(combine_axes=...)`)
and a scoped context manager (`with set_options(combine_axes="strict"): ...`),
mirroring `xarray.set_options`.
"""

from __future__ import annotations

import typing_extensions as tx

#: Live option values. Read through `get_option`; write only via `set_options`.
_OPTIONS = {
    "combine_axes": "drop_conflicts",
}

#: Allowed values per option (options with a free value are absent here).
_CHOICES = {
    "combine_axes": ("drop_conflicts", "strict", "override", "drop"),
}


def get_option(name: str) -> tx.Any:
    """The current value of option `name`."""
    return _OPTIONS[name]


class set_options:
    """
    Set one or more options, globally or for a `with` block.

    Options:

    - **`combine_axes`** -- how axis **descriptors** (`type`/`unit`/
      `orientation`) combine when two operands meet in a name-aware op
      (broadcast-by-name, alignment):

        - `"drop_conflicts"` *(default)* -- union the axes; per shared dim keep
          the descriptor fields the operands agree on and drop the ones that
          conflict (the same rule coordinates already follow);
        - `"strict"` -- raise `ValueError` on any conflicting field;
        - `"override"` -- keep the left operand's fields, ignore conflicts;
        - `"drop"` -- drop all descriptor fields from the result.

    Used as a permanent setter or a scoped context manager::

        set_options(combine_axes="strict")            # until changed
        with set_options(combine_axes="strict"):      # for this block
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
            choices = _CHOICES.get(name)
            if choices is not None and value not in choices:
                raise ValueError(
                    f"invalid value {value!r} for {name!r}; "
                    f"valid values: {list(choices)}"
                )
            self._previous[name] = _OPTIONS[name]
        _OPTIONS.update(options)

    def __enter__(self) -> set_options:
        return self

    def __exit__(self, *exc: tx.Any) -> None:
        _OPTIONS.update(self._previous)
