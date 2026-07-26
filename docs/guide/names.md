# Names & dimensions

`XTensor` carries **named dimensions** that follow the tensor through reshaping
and reductions, and it lets you refer to any dimension by its name instead of an
integer position.

## Named dimensions

Names propagate through reshaping and reordering, and a name tuple can use `...`
to stand for a run of axes you don't name here (name just the ends):

```python
import torch
from fiery.xtensor import XTensor

# Named dimensions
x = XTensor(torch.zeros(2, 3, 4), names=("batch", "height", "width"))
x.T.names                       # ('width', 'height', 'batch')
x.unsqueeze(1).names            # ('batch', None, 'height', 'width')

# `...` stands for a run of axes you don't name here (name just the ends)
XTensor(torch.zeros(2, 3, 4, 5), names=("batch", ..., "width")).names
#                               # ('batch', None, None, 'width')

# Refer to a dimension by name (method form)
x.transpose("height", "width").names   # ('batch', 'width', 'height')
x.sum(dim="batch").names               # ('height', 'width')
x.mean(dim="height", keepdim=True).names  # ('batch', 'height', 'width')
```

## Referring to a dimension by name

Anywhere an operation takes a `dim` (or `dim0`/`dim1`, `source`/`destination`,
…), you can pass an axis **name** instead of an integer — **on the method
form**:

```python
x.transpose("height", "width")   # ok
x.sum(dim="height")              # ok
x.movedim("batch", -1)           # ok (names or ints, mixed)
```

Name-as-dim is **not** available on the *functional* form
(`torch.transpose(x, "height", "width")`, `torch.sum(x, dim="height")`). This
is a **PyTorch limitation**, not a preference of ours — we would happily accept
a string `dim` on the functional forms if we could:

- The **method** form (`x.op(...)`) resolves to a function this package
  installs on the tensor subclass, so a name is turned into an integer in
  Python *before* PyTorch ever sees the arguments.
- The **functional** form (`torch.op(x, ...)`) goes straight into PyTorch's
  C-level argument parser, which validates that `dim` is an integer *before*
  the `__torch_function__` hook that would let us intercept the call runs. On
  recent PyTorch a string `dim` therefore raises `TypeError` from PyTorch
  itself, before this package is consulted. Older PyTorch happened to dispatch
  first, so the behaviour was version-dependent and is not relied upon.

Intercepting the functional form would require monkey-patching the `torch.*`
functions globally, which this package deliberately does not do. The functional
form still works perfectly with an **integer** `dim` — and still carries names
through the result (`torch.sum(x, 1).names == x.sum(dim=1).names`); only the
name-*as*-dim convenience is method-only. Operations that have no method form
at all (`torch.cat`, `torch.stack`) take an integer `dim` only.
