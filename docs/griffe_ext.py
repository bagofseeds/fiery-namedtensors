"""Griffe extension exposing runtime-registered `XTensor` members.

`fiery.xtensor` attaches most of `XTensor`'s torch-op overrides (and the
`x*` factories in `_factories.py`) to their class/module at **import time**
(`setattr(cls, ...)`, `xzeros = _make_factory("zeros")`, ...), for wide
PyTorch-version compatibility (see CLAUDE.md). Griffe's static analysis
sees the *assignment* (so a bare, undocumented placeholder member already
exists) but never the docstring or signature `functools.wraps`/direct
assignment attaches at runtime -- so without this extension, these members
render as empty stubs (or are hidden entirely by `show_if_no_docstring`).

This extension runs once griffe's static object tree is fully built
(`on_class`/`on_module`), imports the real module, and replaces any
still-undocumented member with a synthetic `Function` built from the
*live* object's docstring and signature -- inspected with
`follow_wrapped=False`, since these overrides carry `functools.wraps`
metadata from the wrapped `torch.*` function, which would otherwise send
`inspect.signature` chasing a signature-less C builtin instead of the
real Python-level implementation.

`XTensor` itself additionally inherits hundreds of plain (un-overridden)
`torch.Tensor` methods that have nothing to do with this library -- only
the names actually present in its `_OVERRIDES` registry (the ones this
library customises) are added; everything else is left to
`inherited_members = false` to hide, same as before this extension.
"""

from __future__ import annotations

import importlib
import inspect

import griffe


def _griffe_parameters(func: object) -> griffe.Parameters:
    """Build griffe `Parameters` from `func`'s real (non-wrapped) signature."""
    try:
        signature = inspect.signature(func, follow_wrapped=False)
    except (TypeError, ValueError):
        return griffe.Parameters()
    params = []
    for param in signature.parameters.values():
        kind = griffe.ParameterKind(param.kind.description)
        default = (
            repr(param.default)
            if param.default is not inspect.Parameter.empty
            else None
        )
        params.append(griffe.Parameter(param.name, kind=kind, default=default))
    return griffe.Parameters(*params)


def _add_runtime_members(obj: griffe.Object, real: object, names) -> None:
    """Replace each still-undocumented member of `obj` (by `name`, from the
    live `real` object) with a synthetic, fully-documented `Function`."""
    for name in names:
        existing = obj.members.get(name)
        if existing is not None:
            try:
                documented = existing.docstring is not None
            except griffe.AliasResolutionError:
                # a re-exported name from elsewhere (e.g. `from __future__
                # import annotations`) that griffe can't (and doesn't need
                # to) resolve here -- nothing of ours to add.
                continue
            if documented:
                continue  # already documented statically -- leave it alone
        target = getattr(real, name, None)
        if not callable(target):
            continue
        doc = inspect.getdoc(target)
        if not doc:
            continue
        obj.set_member(
            name,
            griffe.Function(
                name,
                parameters=_griffe_parameters(target),
                docstring=griffe.Docstring(doc),
                parent=obj,
            ),
        )


class RuntimeMembersExtension(griffe.Extension):
    """Expose `XTensor`/`_factories` members only added via runtime
    `setattr`."""

    def on_class(
        self,
        *,
        cls: griffe.Class,
        loader: griffe.GriffeLoader,
        **kwargs: object,
    ) -> None:
        if cls.path != "fiery.xtensor._tensors.XTensor":
            return
        real = importlib.import_module("fiery.xtensor._tensors").XTensor
        names = sorted({func.__name__ for func in real._OVERRIDES.values()})
        _add_runtime_members(cls, real, names)

    def on_module(
        self,
        *,
        mod: griffe.Module,
        loader: griffe.GriffeLoader,
        **kwargs: object,
    ) -> None:
        if mod.path != "fiery.xtensor._factories":
            return
        real = importlib.import_module("fiery.xtensor._factories")
        names = [name for name in dir(real) if not name.startswith("_")]
        _add_runtime_members(mod, real, names)
