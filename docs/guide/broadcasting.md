# Broadcasting & alignment

When **both** operands of a pointwise op are fully-named `XTensor`s, their axes
are aligned **by name** — and, where a shared dimension is labelled on both
sides, **by label** — the xarray way.

## Broadcast by name

When **both** operands of a pointwise op (`+`, `-`, `*`, `/`, comparisons, …)
are fully-named `XTensor`s, their axes are aligned **by name** — the xarray way
— instead of by position:

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

### When name-alignment does *not* apply

Name-alignment kicks in only when **both** operands are fully-named `XTensor`s.
Otherwise the op uses ordinary **positional** broadcasting (the plain-torch
rule):

- an operand is a **plain tensor** or a **scalar** — it has no names to align
  against, so it broadcasts positionally (this matches xarray, which also
  broadcasts a bare array against the trailing axes);
- **any** axis on **either** operand is **unnamed** (`None`).

That second case is stricter than xarray, which has no unnamed dimensions and
so *always* aligns by name. Here, a single unnamed axis on one operand drops
the **whole** op to positional — even the axes that *are* named. This is a
deliberately conservative first cut (positional broadcasting is unambiguous);
whether a *partially* named operand should still align its named axes by name,
and only treat the unnamed ones positionally, is an open design question. If you
rely on name-alignment, name **every** axis of both operands (`refine_names`
fills the gaps).

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
