# Broadcasting & alignment

When **both** operands of a pointwise op are fully-named `XTensor`s, their axes
are aligned **by name** — and, where a shared dimension is labelled on both
sides, **by label** — the xarray way.

## Broadcast by name

A pointwise op (`+`, `-`, `*`, `/`, a comparison, …) between two fully-named
`XTensor`s pairs their axes up by name rather than by position:

```python
a = xtensor(torch.arange(6).reshape(2, 3), names=("x", "y"))
b = xtensor(torch.arange(6).reshape(3, 2), names=("y", "x"))  # transposed
(a + b).names            # ('x', 'y')  — b is transposed to match, then added

c = xtensor(torch.arange(2), names=("x",))
d = xtensor(torch.arange(3), names=("y",))
(c + d).shape            # (2, 3)  — disjoint dims broadcast to the outer grid
```

The result's dimensions are the **union** of the operands' names; a shared name
is broadcast together (its sizes must match, or one must be 1) and coordinates
that agree are carried through.

### Partially-named operands

An `XTensor` may have **unnamed** (`None`) axes — a plain torch tensor usually
has an anonymous batch axis. Name-alignment still applies as long as the
unnamed axes are all **leading** (the common "a few batch dims, then named
axes" layout): the **named suffix aligns by name** (union, transpose-to-match,
broadcast a missing axis) while the **leading anonymous run broadcasts
positionally**, right-aligned like ordinary torch batch dims.

```python
a = xtensor(torch.zeros(3, 3), names=("x", "y"))
b = xtensor(torch.zeros(4, 3, 3), names=(None, "x", "y"))  # a batch of them
(a + b).names            # (None, 'x', 'y')  — 'x'/'y' aligned, batch broadcast
```

Two more cases:

- **Identical names** align 1:1 by position, so a non-leading `None` is fine
  when both operands share the exact same `names` (`x(a, None) + y(a, None)`).
- An operand that is **all-unnamed** (every axis `None`), a **plain tensor**,
  or a **scalar** has nothing to align on and broadcasts **positionally** — the
  plain-torch rule (this matches xarray, which broadcasts a bare array against
  the trailing axes).

What is **not** allowed is a `None` sitting *after* a named axis on operands
whose names differ — e.g. `x(a, None) + y(b, None)`. There, aligning by name is
ambiguous and silent positional broadcasting could pair the wrong axes, so the
op **raises**. Name every axis (`refine_names`) or move the unnamed axes to the
front. (This is the resolution of
[#75](https://github.com/bagofseeds/fiery-xtensor/issues/75).)

## Coordinate alignment

When a shared dimension is **labelled on both operands** but the labels are in a
different order — or only partly overlap — the operands are aligned **by label**
before the op, the xarray `join="inner"` way: both are reindexed to the
intersection of their labels (in the left operand's order), so positions are
matched by *label*, not by position.

```python
a = xtensor(torch.tensor([1., 2., 3.]), names=("x",), coords={"x": ("A", "B", "C")})
b = xtensor(torch.tensor([10., 20., 30.]), names=("x",), coords={"x": ("C", "B", "A")})
(a + b).coords            # {'x': ('A', 'B', 'C')}
(a + b).tolist()          # [31.0, 22.0, 13.0]  — A+A, B+B, C+C

c = xtensor(torch.tensor([1., 2., 3.]), names=("x",), coords={"x": ("A", "B", "C")})
d = xtensor(torch.tensor([10., 20., 30.]), names=("x",), coords={"x": ("B", "C", "D")})
(c + d).coords            # {'x': ('B', 'C')}    — inner join to the overlap
```

A dimension labelled on only one side has nothing to align against, so its
labels simply ride along and the op stays positional.
