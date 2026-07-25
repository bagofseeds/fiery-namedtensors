# Proposal 0003 — Data units

| | |
| --- | --- |
| **Status** | Draft — exploring (design not settled; no implementation) |
| **Author** | (proposed) |
| **Created** | 2026-07-25 |
| **Tracking** | part of [#3](https://github.com/bagofseeds/fiery-xtensor/issues/3); the *other* meaning of "unit" from Proposal 0001; builds on Proposal 0002 (structured coordinates) |

## Abstract

A **data unit** is the physical unit of the tensor's *values* ("this element is
3.2 **volts**"), as opposed to a **coordinate** unit, which qualifies the ticks
along an axis (Proposal 0001). An `XTensor` with a data unit is a **view-based
`Quantity`**: a plain `torch.Tensor` plus a unit annotation that rides through
ops and drives dimensional analysis — but never a `pint.Quantity` wrapping the
data, so autograd/GPU/dispatch stay intact.

This document is deliberately exploratory. Data units are the *hard* unit
problem — the one with real design tension — so the goal here is to map the use
cases, fix a model, work through propagation op-by-op, and surface the problems
before committing to anything.

## 1. Why bother — use cases

- **Bug-catching arithmetic.** `voltage + current` should raise, not silently
  add. Dimensional consistency is the cheapest class of correctness check.
- **Conversions at boundaries.** `x.to_unit("mV")` rescales the data once, at
  the edge of a pipeline, instead of scattering `* 1e3` through the code.
- **Heterogeneous stacks (the driving case).** A `channel` axis whose positions
  hold *different quantities* — `["voltage" (V), "current" (A), "power" (W)]`.
  Selecting a channel yields a view in that channel's unit; this is exactly the
  "the unit of a coordinate is the unit of the data at that coordinate" idea.
- **Physics / simulation fields.** Velocity `m/s`, pressure `Pa`, mass `kg`;
  operators that combine them (`ρ·v²`) get their units checked and derived.
- **Interop.** Attach a unit when ingesting data, detach (`.magnitude`) when
  handing off to a plain-tensor API.

## 2. The model

### 2.1 Uniform data unit

The whole tensor shares one unit `U`. Stored as a new whole-tensor attribute
`_data_unit` (added to `_ATTRS`, so it propagates by default like the others).
The element unit is `U` everywhere. This is the simple, common case.

### 2.2 Heterogeneous data unit (units that vary along an axis)

The driving case. Along **one** axis, each position carries its own unit — which
is precisely a **structured coordinate** (Proposal 0002) whose labels have a
`unit` field:

```python
meas = xtensor(data, names=("q", "t"), coords={"q": [
    {"name": "voltage", "unit": "V"},
    {"name": "current", "unit": "A"},
    {"name": "power",   "unit": "W"},
]})
```

So heterogeneity **is** a coordinate `unit` (0002), reused here to mean *the data
unit contributed by that position*. No new storage — B rides on the structured
coordinate substrate.

### 2.3 The effective-unit rule (the unifying formula)

The unit of a single element is the product of a whole-tensor base unit and one
factor per axis that carries per-position units:

```
unit(x[i₀, i₁, …]) = base_unit · Π_k  coord_unit(axis k, position i_k)
```

where an axis with no per-position units contributes a dimensionless factor.
This subsumes both regimes: **uniform** = base unit only; **heterogeneous** =
one axis contributes; and it *defines* what happens when **more than one** axis
carries units — the factors **multiply** (this is the crux problem, §4.1).

## 3. Propagation semantics (op by op)

Let `U`, `V` be operand units (each possibly a per-position map).

| Op family | Result data unit | Constraint / notes |
| --- | --- | --- |
| `add` / `sub`, `maximum`/`minimum`, comparisons | `U` (after converting `V`→`U`) | **`U` and `V` must be compatible** (same dimension) elementwise, else raise |
| `mul` / `div` / `floor_divide` | `U·V` / `U∕V` | always defined; grows/​shrinks the unit map |
| `pow(n)` | `Uⁿ` | scalar rational `n` only; tensor/​unknown exponent ⇒ require `U` dimensionless |
| `matmul` / `einsum` / `tensordot` | `U·V` | the **contracted** axis' per-position units must be *uniform* on each side, else the summed terms differ in unit → raise (§4.4) |
| `sum` / `mean` / `cumsum` over axis *a* | unit unchanged | if *a* carries per-position units they must be **compatible**; convert-then-reduce, else raise (§4.2) |
| `prod` over a size-*k* axis | `Uᵏ` (or Π of the axis' per-position units) | dimensionally valid but rarely intended |
| `std` / `var` | `U` / `U²` | over a unit-carrying axis: as `sum` |
| `norm` (p-norm) | `U` | requires a single compatible unit along the reduced axes |
| `exp`/`log`/`sin`/`sigmoid`/… (transcendental) | dimensionless | **input must be dimensionless**, else raise (§4.3) |
| `abs`/`neg`/`clamp`/`sort`/`flip`/reshape/… | `U` unchanged | pure structure/sign |
| a plain tensor / Python scalar operand | dimensionless | scalars are unitless |
| `sel`/`isel`/`[]` down to one position on a unit axis | that position's unit folds into `base_unit` | heterogeneity collapses to uniform |

## 4. The hard problems (this is the point of the proposal)

### 4.1 More than one axis carrying units → multiplication

The effective-unit rule says the factors multiply: a channel axis in `V` and a
time axis in `s` make each element `V·s`. That is *consistent* but almost never
what a user means, and it makes the element unit depend on **two** coordinates at
once. Options:

- **(a) Allow it** — full multiplicative composition; maximally general, but the
  unit of `x` is now a rank-N map and every op must track products.
- **(b) Restrict to one** — at most one axis may carry per-position units;
  constructing a second raises. Simple, covers the driving case, and can be
  relaxed later. **Leaning (b) for a first cut.**
- **(c) Forbid heterogeneity entirely at first** — ship only the *uniform*
  data unit (§2.1); revisit heterogeneity once uniform is proven. Smallest.

### 4.2 Reducing across incompatible units must raise

`sum` over `["voltage", "current", "power"]` is `V + A + W` — dimensionally
meaningless. So a reduction over a unit-carrying axis must **require compatible
units** (all convertible to one), convert, then reduce; incompatible ⇒ a hard
error. This is a *feature* (it catches the mistake), but it means reductions
gain a unit-compatibility pre-check, and the "reduce everything" ops
(`x.sum()` with no dim, `norm`, `flatten`-then-reduce) are only valid on
uniform, single-dimension data.

### 4.3 "Is it even one quantity?" — global ops and transcendentals

A tensor whose values carry a unit is not an arbitrary array. `exp(x)` is
undefined unless `x` is dimensionless; so are `log`, `sin`, `softmax`,
activation functions, etc. A united tensor therefore **cannot pass through most
of a neural network** without first being made dimensionless. Do we:

- **enforce** it (raise on `exp` of a non-dimensionless tensor) — safe, but
  intrusive and easy to trip over; or
- **warn / auto-strip** the unit on such ops — permissive, but silently drops
  the guarantee.

This tension (safety vs. it-just-works) is the single biggest UX question for
data units and probably decides how widely they get used.

### 4.4 Contraction validity

`matmul`/`einsum` sum products over a contracted axis; every summed term must
share a unit, so the contracted axis must be **uniform** on each operand.
Heterogeneous contraction is generally invalid — another compatibility
pre-check, and a reason heterogeneous units interact badly with linear algebra.

### 4.5 Cost

A per-position unit map is `O(size of the unit axis)` metadata and needs a
product/compat check per relevant op. Uniform units are `O(1)`. If heterogeneity
is in scope, ops in the hot path pay a (small, symbolic) tracking cost — worth
measuring, and a reason `unit_backend=None` must remain truly zero-overhead.

## 5. Relationship to the other layers

- **Orthogonal to coordinate units (0001).** A tensor may have *both*: a `t`
  axis in seconds (tick metric) **and** data in volts (value unit). They never
  interact — one describes where a sample sits, the other what it measures.
- **Reuses structured coordinates (0002)** for the heterogeneous case: a
  per-position data unit is a coordinate `unit` field. No new storage there.
- **Reuses the `unit_backend`** option (0001): data units are inert with
  `unit_backend=None`, and all parsing/compat/conversion goes through the same
  backend interface (`normalise`, `equal`, `convert`, plus unit **algebra**:
  `mul`, `div`, `pow`, `dimensionless?`, `compatible?`).

## 6. Suggested phasing

1. **Uniform data unit only** (§2.1): `_data_unit`, `unit=`/`.unit`,
   `to_unit`, `+`/`-` compat, `*`/`/` products, matmul, dimensionless-guarded
   transcendentals. Proves the view-`Quantity` model end to end.
2. **Heterogeneous, restricted to one unit-axis** (§2.2 + §4.1b): the driving
   channel-stack case, with reduction/contraction compat checks (§4.2, §4.4).
3. **Full multiplicative composition** (§4.1a) — only if a real need appears.

## 7. Open questions

1. **§4.3 is the big one:** enforce dimensionless-input for transcendentals, or
   auto-strip with a warning? (Decides NN-friendliness.)
2. **§4.1:** one unit-axis (b) vs full multiplication (a) vs uniform-only (c) for
   the first cut.
3. **Detach ergonomics:** `.magnitude` / `.to("V").tensor()` to drop to a plain
   tensor — how explicit must it be.
4. **Equality/compat granularity:** exact dimension match, or allow implicit
   conversion on `add` when dimensions agree (V + mV → V)?
5. **Serialisation / `repr`:** how a (possibly heterogeneous) unit shows up.
6. **Which ops silently pass vs. must be annotated** — the full allow/deny list
   beyond §3.

## References

- pint unit algebra — <https://pint.readthedocs.io>
- Proposal 0001 (coordinate units) — the *other* unit
- Proposal 0002 (structured coordinates) — storage for heterogeneous units
