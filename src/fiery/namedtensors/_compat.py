"""Version-compatibility helpers for `fiery.namedtensors`.

Small shims that let the package span a wide range of Python and PyTorch
versions. Keep anything version-conditional here so the rest of the package
can import a stable surface.
"""

from __future__ import annotations

import torch
import typing_extensions as tx

# EllipsisType lives in `types` only since Python 3.10; fall back to the
# concrete type of `...` on older interpreters.
try:
    from types import EllipsisType
except ImportError:  # Python < 3.10
    EllipsisType = type(...)

# Context manager that disables (subclass) torch-function dispatch while a
# name-aware override runs, so the plain torch ops it calls do not recurse.
# It was renamed `DisableTorchFunction` -> `DisableTorchFunctionSubclass`
# around torch 1.13; support both.
try:
    from torch._C import (
        DisableTorchFunctionSubclass as no_dispatch,  # noqa: F401
    )
except ImportError:  # torch < ~1.13
    from torch._C import DisableTorchFunction as no_dispatch  # noqa: F401


def broadcast_shape(*shapes: tx.Sequence[int]) -> torch.Size:
    """
    Broadcast several shapes into one, using pure shape arithmetic.

    Equivalent to `torch.broadcast_shapes` but implemented without
    allocating any tensors, so it is cheap and works on PyTorch versions
    that predate `torch.broadcast_shapes` (added in 1.8).

    Parameters
    ----------
    *shapes : sequence of int
        The shapes to broadcast together.

    Returns
    -------
    torch.Size
        The broadcasted shape.

    Raises
    ------
    RuntimeError
        If the shapes are not broadcastable.
    """
    ndim = max((len(s) for s in shapes), default=0)
    result = [1] * ndim
    for shape in shapes:
        offset = ndim - len(shape)
        for i, size in enumerate(shape):
            j = offset + i
            if size == 1 or size == result[j]:
                continue
            if result[j] == 1:
                result[j] = size
            else:
                raise RuntimeError(
                    "shape mismatch: objects cannot be broadcast to a "
                    f"single shape: {list(shapes)}"
                )
    return torch.Size(result)


def torch_func(name: str) -> tx.Optional[tx.Callable]:
    """
    Resolve a torch callable by name, preferring the functional form
    (`torch.<name>`) and falling back to the tensor method
    (`torch.Tensor.<name>`).

    Returns `None` if neither exists in the running PyTorch version, so
    that overrides can be registered conditionally and we only ever
    overload functions that actually exist.
    """
    func = getattr(torch, name, None)
    if func is None:
        func = getattr(torch.Tensor, name, None)
    return func
