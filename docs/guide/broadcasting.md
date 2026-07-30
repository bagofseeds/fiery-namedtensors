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

### Each axis's pairing key is its own operand's business

An axis pairs up **by name** if it has one, and **by position** if it doesn't
— and which of those applies is decided **per operand, per axis**, never by
looking at what the *other* operand's names happen to be:

```python
X = xtensor(torch.zeros(2, 3), names=("a", "b"))
Y = xtensor(torch.zeros(3, 2), names=("b", "a"))
(X + Y).names            # ('a', 'b')  — paired by NAME: X's 'a' meets Y's 'a',
                          # even though it sits at a different position in Y

X2 = xtensor(torch.zeros(2, 3), names=(None, None))
Y2 = xtensor(torch.zeros(2, 3), names=(None, None))
(X2 + Y2).names          # (None, None) — paired by POSITION: first meets first
```

This can look inconsistent at first — "does an axis pair by name or by
position?" — but it isn't a case-by-case rule, it's one rule applied locally:
*an unnamed axis always broadcasts positionally against whatever sits in the
corresponding slot on the other side, regardless of whether that other axis
happens to be named.* The consequence is the sharp edge below.

### Sharp edge: an anonymous operand is not the same as a partially-named one

Because pairing is decided per axis, giving one more axis a name can change
*which values land where* — even when the result's `shape` and `names` come
out identical either way:

```python
a = xtensor(torch.zeros(3, 3, 5), names=(None, "x", "y"))   # batch=3, x=3, y=5
d = torch.arange(15.).reshape(3, 5)

r1 = a + xtensor(d)                            # right operand: fully anonymous
r2 = a + xtensor(d, names=(None, "y"))          # right operand: partially named
# r1.shape == r2.shape == (3, 3, 5)
# r1.names == r2.names == (None, 'x', 'y')

r1[:, 0, 0].tolist()      # [0.0, 0.0, 0.0]   -- d broadcasts positionally: its
                          # axis 0 (size 3) lines up with a's trailing axis 'x'
r2[:, 0, 0].tolist()      # [0.0, 5.0, 10.0]  -- d's axis 0 is now anonymous
                          # too, but 'y' on the right pins d's axis 1 to a's
                          # 'y', which pushes d's anonymous axis to align with
                          # a's anonymous batch axis instead of its 'x' axis
```

`r1` and `r2` are unequal element-wise despite sharing a shape and a `names`
tuple — the fully-anonymous `d` broadcasts against `a`'s two *trailing* axes
(the ordinary torch rule), while naming just one of `d`'s axes anchors it to
that specific dimension and lets its remaining anonymous axis fall back
against `a`'s *leading* one instead. Naming every axis removes the ambiguity
entirely (`xtensor(d, names=("x", "y"))` agrees with `r1` here, because now
there is a name to pair on for both):

```python
r3 = a + xtensor(d, names=("x", "y"))
r3[:, 0, 0].tolist()      # [0.0, 0.0, 0.0]  -- agrees with r1
```

The practical rule: on a mixed anonymous/named operation, either name **every**
axis that has a same-sized counterpart on the other side, or leave **all** of
them unnamed — don't name only *some* of an operand's axes when the other
operand is fully anonymous, since which axis absorbs the "leftover" anonymous
dimension depends on that choice. (Investigated as
[#145](https://github.com/bagofseeds/fiery-xtensor/issues/145); the current
behavior is intentional — see the issue for the fuller design discussion — and
this section is the documentation half of that resolution.)

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
