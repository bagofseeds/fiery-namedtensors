"""Named factory helpers.

Thin wrappers around the `torch.*` construction functions that return a
[`NamedTensor`][fiery.namedtensors.NamedTensor] directly, so callers can name
the axes at creation time instead of wrapping every call by hand::

    named_zeros(2, 3, names=("row", "col"))

Each wrapper forwards all positional and keyword arguments to the matching
`torch.*` function and accepts an extra ``names=`` keyword.
"""

from __future__ import annotations

import torch
import typing_extensions as tx

from fiery.namedtensors._tensors import NamedTensor


def _make_factory(name: str) -> tx.Optional[tx.Callable]:
    """Build a `named_<name>` wrapper around `torch.<name>` (or `None`)."""
    base = getattr(torch, name, None)
    if base is None:  # pragma: no cover - all wrapped ops are very old
        return None

    def factory(
        *args: tx.Any,
        names: tx.Optional[tx.Sequence[str | None]] = None,
        **kwargs: tx.Any,
    ) -> NamedTensor:
        return NamedTensor(base(*args, **kwargs), names=names)

    factory.__name__ = "named_" + name
    factory.__qualname__ = "named_" + name
    factory.__doc__ = (
        f"Like `torch.{name}`, but returns a `NamedTensor`.\n\n"
        f"All arguments are forwarded to `torch.{name}`; pass `names=(...)`\n"
        f"to name the axes of the result."
    )
    return factory


named_zeros = _make_factory("zeros")
named_ones = _make_factory("ones")
named_empty = _make_factory("empty")
named_full = _make_factory("full")
named_arange = _make_factory("arange")
named_rand = _make_factory("rand")
named_randn = _make_factory("randn")
named_eye = _make_factory("eye")


__all__ = [
    "named_zeros",
    "named_ones",
    "named_empty",
    "named_full",
    "named_arange",
    "named_rand",
    "named_randn",
    "named_eye",
]
