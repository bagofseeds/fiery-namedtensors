"""The generic name-aware tensor-subclass base (see CLAUDE.md, "How the
subclassing works"): a per-subclass override registry plus the
`__torch_function__` hook that runs a registered override or propagates
metadata attributes through an unregistered op.
"""

from __future__ import annotations

import copy
from functools import wraps

import typing_extensions as tx
from torch import Tensor

from fiery.xtensor._compat import no_dispatch as _no_dispatch


class ExtendedTensorMeta(type(Tensor)):
    # We need a metaclass so that each subclass has its own registry

    def __new__(
        cls, name: str, bases: tuple[type, ...], classdict: tx.Mapping
    ) -> type:
        kls = super().__new__(cls, name, bases, classdict)
        kls._OVERRIDES = {}
        return kls


class ExtendedTensor(Tensor, metaclass=ExtendedTensorMeta):
    """
    A `torch.Tensor` subclass with extended, name-aware behaviour.

    Selected torch functions are overridden through the `__torch_function__`
    protocol; the overrides live in a per-subclass registry, populated by the
    [`overrides`][fiery.xtensor.ExtendedTensor.overrides] decorator. Any op
    without an override still propagates the subclass's own metadata
    attributes from its first tensor argument onto the result.
    """

    @classmethod
    def overrides(cls, func: tx.Optional[tx.Callable]) -> tx.Callable:
        """
        Decorator to register a function override.

        `func` may be `None` (an op that does not exist in the running
        PyTorch version); in that case the override is silently skipped so
        that we never overload a function that is missing from this
        PyTorch build.
        """

        def decorator(newfunc: tx.Callable) -> tx.Callable:
            if func is None:
                # Target op absent in this PyTorch version: do not register.
                return newfunc
            newfunc = wraps(func)(newfunc)
            # Register as a public torch function
            cls._OVERRIDES[func] = newfunc

            # Register as a torch.Tensor method. `newfunc`'s own body calls
            # `base(input, ...)` to get the real op's result -- when reached
            # via `__torch_function__` (the functional form, `torch.op(x)`),
            # that inner call is made under `_no_dispatch()` below, so it
            # runs the plain op directly instead of recursing back here. The
            # method form (`x.op(...)`) is instead resolved by ordinary
            # Python attribute lookup, which never goes through
            # `__torch_function__` at all -- so without this wrapper,
            # `newfunc`'s inner `base(...)` call would re-enter
            # `__torch_function__`, find this same override, and run the
            # entire body a second time (issue #160). Disabling dispatch
            # here mirrors what `__torch_function__` already does for the
            # functional form, so both forms run the override exactly once.
            @wraps(newfunc)
            def method_slot(*args: tx.Any, **kwargs: tx.Any) -> tx.Any:
                with _no_dispatch():
                    return newfunc(*args, **kwargs)

            setattr(cls, func.__name__, method_slot)
            return newfunc

        return decorator

    @classmethod
    def __torch_function__(
        cls,
        func: tx.Callable,
        types: tuple[type, ...],
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> tx.Any:
        # Look up a name-aware override for this op (most-derived first).
        kwargs = kwargs or {}
        override = None
        for base in cls.__mro__:
            registry = getattr(base, "_OVERRIDES", {})
            if func in registry:
                override = registry[func]
                break

        if override is not None:
            # The override computes the result AND its name metadata (via
            # `_carry`), so the same metadata lands for `x.op(...)` and
            # `torch.op(x, ...)`. Run it with subclass dispatch disabled so the
            # plain torch ops it calls do not recurse back here, and return its
            # result directly (routing it back through the default
            # `__torch_function__` re-wraps and drops the metadata on some
            # PyTorch versions).
            with _no_dispatch():
                return override(*args, **kwargs)

        out = super().__torch_function__(func, types, args, kwargs)
        # Ops without a name-aware override: carry subclass attributes (axis
        # names, coordinate labels) from the first tensor argument onto the
        # output, when the output is a real tensor (many ops return `None`).
        if isinstance(out, Tensor) and args:
            source = args[0]
            attrs = set()
            for base in cls.__mro__:
                attrs |= set(getattr(base, "_ATTRS", set()))
            for attr in attrs:
                if not hasattr(out, attr) and hasattr(source, attr):
                    setattr(out, attr, getattr(source, attr))
        return out

    def __deepcopy__(self, memo: dict) -> tx.Self:
        # `Tensor.__deepcopy__`'s default implementation is itself a
        # dispatched torch op (it starts with `has_torch_function_unary`),
        # so calling it on a subclass with a custom `__torch_function__`
        # re-enters that machinery under a disabled-dispatch context -- in
        # which `self.new_empty([])` (what it uses internally) returns a
        # plain `Tensor`, not this subclass, and it then raises rather than
        # silently mismatching. Defined as a plain method here (not a
        # registered override), so Python's normal attribute lookup finds
        # this directly and the dispatch-based default body never runs.
        if id(self) in memo:
            return memo[id(self)]
        # Unlike vanilla `Tensor.__deepcopy__`, this doesn't restrict itself
        # to graph leaves, and doesn't preserve the autograd graph either way
        # -- the result below is always a fresh, detached snapshot of the
        # current values, re-marked to require grad if the original did (if
        # you need the copy to stay attached to the original computation for
        # a later `.backward()`, deepcopy is the wrong tool regardless --
        # `.clone()` directly, without detaching, is). A strict leaf-only
        # check would fail even the ordinary case of wrapping an existing
        # `requires_grad=True` tensor: `as_subclass` (needed for the
        # zero-copy retag `XTensor(t)` does) is itself a *view* op under
        # PyTorch's own autograd rules, and any view of a grad-requiring leaf
        # is non-leaf -- true of a plain `Tensor.as_subclass`/`.view()` too,
        # not specific to this subclass -- so almost every `XTensor` wrapping
        # a grad-requiring input would already fail that check before any
        # arithmetic is even involved.
        data = self.as_subclass(Tensor).detach().clone()
        out = data.as_subclass(type(self))
        # `as_subclass` on a tensor that already requires grad returns a
        # tracked *view* (non-leaf) -- setting `requires_grad_()` only
        # afterwards, on the already-retagged (and by now grad-free) `out`,
        # is what keeps the result a genuine leaf.
        if self.requires_grad:
            out.requires_grad_()
        memo[id(self)] = out
        out.__dict__ = copy.deepcopy(self.__dict__, memo)
        if self.is_leaf and self.grad is not None:
            out.grad = copy.deepcopy(self.grad, memo)
        return out

    def __format__(self, format_spec: str) -> str:
        # `Tensor.__format__`'s own body checks `type(self) is Tensor` and
        # falls back to `object.__format__` (== `str(self)`) for any
        # subclass -- including this one. That's silently fine for most
        # calls, but fatal for a 0-dim tensor specifically: an int-dtype
        # tensor's `repr` (`torch._tensor_str._Formatter`) formats each
        # element via `f"{value}"`, where `value` is itself a 0-dim slice
        # of this subclass -- so `str(self)` on THAT slice re-enters the
        # very same tensor-printing machinery, on a tensor that is still
        # 0-dim and still this subclass, forever (issue #118). A plain
        # `Tensor` never hits this because its `__format__` extracts
        # `.item()` directly instead of recursing back into `repr`.
        # Defined as a plain method (not a registered override, matching
        # `__deepcopy__` above) so Python's normal dunder lookup finds
        # this directly, without going through `__torch_function__`.
        if self.dim() == 0 and not self.is_meta:
            return self.as_subclass(Tensor).item().__format__(format_spec)
        return object.__format__(self, format_spec)
