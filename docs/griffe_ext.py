"""Griffe extension exposing runtime-registered `XTensor` members.

`fiery.xtensor` attaches most of `XTensor`'s torch-op overrides (and the
`x*` factories in `_factories.py`) to their class/module at **import time**
(`setattr(cls, ...)`, `xzeros = _make_factory("zeros")`, ...), for wide
PyTorch-version compatibility (see CLAUDE.md). Griffe's static analysis
sees the *assignment* for the `_factories` case (so a bare, undocumented
placeholder `Attribute` already exists) but for `XTensor`'s overrides --
attached via a decorator's `setattr` inside a function body, not a
module-level assignment -- the member is simply **absent** from the
static model. Either way, the real docstring/signature only exists once
the module actually runs, so without this extension most of `XTensor`'s
public surface renders as empty or missing entirely.

This extension runs once griffe's static object tree is fully built
(`on_class`/`on_module`), imports the real module, and replaces any
still-undocumented member with a synthetic `Function` built from the
*live* object's signature -- inspected with `follow_wrapped=False`, since
these overrides carry `functools.wraps` metadata from the wrapped
`torch.*` function, which would otherwise send `inspect.signature`
chasing a signature-less C builtin instead of the real Python-level
implementation.

The two targets get different docstring treatment:

- `_factories`' `x*` wrappers carry their own hand-written docstring
  (assigned in `_make_factory`/`_make_like_factory`), so it's used as-is.
- `XTensor`'s overrides carry `functools.wraps(original_torch_func)`,
  which unconditionally copies the *wrapped* function's docstring --
  PyTorch's own reST-formatted docs, which don't parse cleanly under this
  site's numpy-style docstring renderer (stray Sphinx roles/directives)
  and don't mention this library's own name-propagation behaviour at all.
  A short, generated one-line description is used instead; issue #130
  tracks writing real per-method docstrings.

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


def _griffe_signature(func: object) -> tuple:
    """`(Parameters, returns)` from `func`'s real (non-wrapped) signature."""
    try:
        signature = inspect.signature(func, follow_wrapped=False)
    except (TypeError, ValueError):
        return griffe.Parameters(), None
    params = []
    for param in signature.parameters.values():
        kind = griffe.ParameterKind(param.kind.description)
        default = (
            repr(param.default)
            if param.default is not inspect.Parameter.empty
            else None
        )
        annotation = (
            str(param.annotation)
            if param.annotation is not inspect.Parameter.empty
            else None
        )
        params.append(
            griffe.Parameter(
                param.name, annotation=annotation, kind=kind, default=default
            )
        )
    returns = (
        str(signature.return_annotation)
        if signature.return_annotation is not inspect.Signature.empty
        else None
    )
    return griffe.Parameters(*params), returns


def _own_doc(name: str, target: object) -> str:
    """A wrapper's own hand-written docstring (`_factories.py`'s `x*`s)."""
    return inspect.getdoc(target) or ""


def _override_doc(name: str, target: object) -> str:
    """A short generated description for an `XTensor` torch-op override --
    `inspect.getdoc` would return the *wrapped* `torch.<name>`'s own raw
    reST docstring (copied in by `functools.wraps`), which renders with
    broken markup here and never mentions this library's own behaviour."""
    return (
        f"Name-aware `torch.{name}`: behaves like `torch.{name}`, but "
        "this tensor's names (and coordinates, where applicable) "
        f"propagate onto the result. See `torch.{name}` for the full "
        "numerical behaviour."
    )


def _add_runtime_members(
    obj: griffe.Object, real: object, names, doc_for
) -> None:
    """Replace each still-undocumented member of `obj` (by `name`, from the
    live `real` object) with a synthetic, fully-documented `Function`."""
    for name in names:
        existing = obj.members.get(name)
        if existing is not None:
            try:
                documented = existing.docstring is not None
            except griffe.AliasResolutionError:
                # a re-exported name from elsewhere (`tx`, `torch`, `from
                # __future__ import annotations`, ...) that griffe can't
                # (and doesn't need to) resolve here -- nothing of ours to
                # add for it.
                continue
            if documented:
                continue  # already documented statically -- leave it alone
        target = getattr(real, name, None)
        if not inspect.isroutine(target):
            continue
        doc = doc_for(name, target)
        if not doc:
            continue
        parameters, returns = _griffe_signature(target)
        obj.set_member(
            name,
            griffe.Function(
                name,
                parameters=parameters,
                returns=returns,
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
        try:
            # Import the top-level package, not `_tensors` directly: the
            # override registry is only fully populated once all six leaf
            # operator modules (`_combine`, `_gather`, `_pointwise`,
            # `_reduce`, `_shape`, `_slice`) have run their
            # `XTensor.overrides(...)`-decorated registrations, which
            # `fiery.xtensor.__init__` triggers as an import side effect.
            real = importlib.import_module("fiery.xtensor").XTensor
        except ImportError:
            # `fiery.xtensor` (and its `torch` dependency) isn't installed
            # in this docs-build environment -- fall back to whatever
            # static analysis alone could discover, rather than failing
            # the whole build.
            return
        names = sorted({func.__name__ for func in real._OVERRIDES.values()})
        _add_runtime_members(cls, real, names, _override_doc)

    def on_module(
        self,
        *,
        mod: griffe.Module,
        loader: griffe.GriffeLoader,
        **kwargs: object,
    ) -> None:
        if mod.path != "fiery.xtensor._factories":
            return
        try:
            real = importlib.import_module("fiery.xtensor._factories")
        except ImportError:
            return
        names = list(getattr(real, "__all__", ()))
        _add_runtime_members(mod, real, names, _own_doc)
