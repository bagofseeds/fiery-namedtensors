# Proposal 0003 — Data units

| | |
| --- | --- |
| **Status** | Draft — converging (design decided; no implementation) |
| **Author** | (proposed) |
| **Created** | 2026-07-25 |
| **Tracking** | part of [#3](https://github.com/bagofseeds/fiery-xtensor/issues/3); the *other* meaning of "unit" from Proposal 0001; builds on Proposal 0002 (structured coordinates) |

## Abstract

A **data unit** is the physical unit of the tensor's *values* ("this element is
3.2 **volts**"), as opposed to a **coordinate** unit, which qualifies the ticks
along an axis (Proposal 0001). An `XTensor` with a data unit is a **view-based
`Quantity`**: a plain `torch.Tensor` plus a unit annotation that rides through
ops — never a `pint.Quantity` wrapping the data, so autograd/GPU/dispatch stay
intact.

The design follows **one general principle**, not a pile of special cases:

> **Units combine algebraically. Wherever a computation is dimensionally
> invalid or ambiguous, the unit is *dropped* by default; a `unit_policy`
> setting can make it *raise* instead.**

So units are a zero-friction, best-effort annotation out of the box, and a
dimensional-safety net when you opt in.

## 1. Use cases

- **Bug-catching arithmetic** (opt-in strict): `voltage + current` raises.
- **Conversions at boundaries**: `x.to_unit("mV")` rescales the data once.
- **Heterogeneous stacks**: a `channel` axis whose positions hold different
  quantities — `["voltage" (V), "current" (A), "power" (W)]`; selecting a
  channel yields a view in that channel's unit. (This *is* "the unit of a
  coordinate is the unit of the data there".)
- **Physics / simulation**: velocity `m/s`, pressure `Pa`; derived units
  (`ρ·v²`) fall out of the algebra.
- **Interop**: attach on ingest, detach (`.magnitude`) on handoff.

## 1a. Why *annotate* a tensor, not *wrap* it in `pint.Quantity`

We attach a unit to a `torch.Tensor` **subclass** and use the backend for the
unit **algebra only** — we do **not** store the data inside a `pint.Quantity`.
The reason is concrete (measured against pint 0.25 + torch 2.2):

- A `pint.Quantity` is **not** a `torch.Tensor` (`isinstance(q, Tensor)` is
  `False`). Its Python arithmetic operators *do* defer to the wrapped tensor and
  **autograd survives** them (`(q + q).sum().backward()` works) — so the
  intuition that it "just defers" holds for `+`/`*`/… .
- But the **torch functional API and `nn` reject it**: `torch.sin(q)`,
  `torch.matmul(q, q)`, `F.linear(q, …)` all raise *"Multiple dispatch failed …
  returned NotImplemented"* — pint intercepts *NumPy*'s protocol, not torch's
  `__torch_function__`, so `torch.*` calls fall through. A Quantity therefore
  can't traverse a model.
- Worse, forwarding a non-arithmetic attribute (`q.reshape(…)`) routes through
  pint's numpy-duck-array coercion, which calls `.numpy()` — and that **fails on
  a grad-requiring or CUDA tensor**.

So wrapping forces an unwrap/rewrap at every torch boundary, dropping the unit
each time. Our `XTensor` **is** a `Tensor`, so `torch.*`, `nn`, autograd, GPU,
and `__torch_function__` all work unchanged — and we still get pint's unit
algebra by feeding it `Unit` objects, never the data. Best of both.

## 2. The model

### 2.1 The general rule (no special cases)

Every element's unit is a product of a whole-tensor **base unit** and one factor
per axis that carries per-position units:

```
unit(x[i₀, i₁, …]) = base_unit · Π_k  unit_k(i_k)
```

An axis with no per-position units contributes a dimensionless factor. That
single formula covers **every** case:

- **uniform** — only `base_unit` is non-trivial;
- **one heterogeneous axis** — the channel-stack case (one `unit_k` non-trivial);
- **several heterogeneous axes** — the factors simply **multiply** (a `V` axis
  and an `s` axis make each element `V·s`). This is embraced, not restricted:
  it's the same rule scaling up, and it is what makes the model composable.

### 2.2 Storage

- **`base_unit`** — a new whole-tensor attribute `_data_unit` (added to
  `_ATTRS`, so it propagates by default).
- **per-axis units** — a `unit` field on a **structured coordinate** (Proposal
  0002): the coordinate `unit` at a position *is* `unit_k(i_k)`. No new storage;
  heterogeneous data units ride entirely on 0002.

The **effective** unit of the tensor is derived on demand from `base_unit` and
whichever coordinates carry `unit`s — the formula in §2.1.

### 2.3 Assigning and reading a unit

The **base** unit is a whole-tensor property, spelled exactly like `names` /
`coords` — a constructor kwarg *and* a settable property, both backed by
`_data_unit`:

```python
x = xtensor(data, names=("b", "c", "t"), unit="V")   # at construction
x.unit = "V"                                          # or after the fact
x.unit                                                # -> "V"  (None when unset)
```

- **Assign ≠ convert.** `x.unit = "V"` (and `unit=`) *annotates* — it declares
  the data is in volts, changing **no** data. Conversion is a separate verb,
  `x.to_unit("mV")`, which rescales the data (×1000) and updates the unit
  (needs a unit already set and a backend).
- **Base vs. per-position.** `.unit` is the whole-tensor **base**; the
  *heterogeneous* case is assigned through the structured-coordinate `unit`
  field (§2.2), not `.unit`. Effective element unit = `base · Π(coord units)`.
- **Always storable, backend-gated.** Like axis/coordinate units today,
  `.unit = "V"` is stored and carried even with `unit_backend=None`; it simply
  does not drive validation/algebra/conversion until a backend is selected — so
  annotating never fails.
- **Detach** drops back to a plain, unit-free tensor (`.magnitude`, spelling TBD
  — §7).

*(Naming note: `.unit` is the **data** unit; an axis' tick unit is
`x.axes[i]["unit"]` (Proposal 0001). Different levels — whole-tensor vs
per-axis — but if the shared word is a concern, `.data_unit` is the fallback
spelling. §7.)*

## 3. The `unit_policy` (drop by default, strict on request)

A single option decides what happens when the algebra hits a dimensionally
invalid or ambiguous step:

| `unit_policy` | on an invalid/ambiguous step |
| --- | --- |
| **`"drop"`** *(default)* | silently produce a result with **no** unit |
| `"strict"` | raise `ValueError` |

Set it like every other option — permanently *or* scoped (`set_options` is both
a setter and a context manager):

```python
set_options(unit_policy="strict")            # session-wide
with set_options(unit_policy="strict"):      # for a block
    ...
```

`"drop"` is deliberately symmetric to `combine_axes`'s conflict-dropping: the
non-strict path never blocks a computation, it just forgets the unit it can no
longer justify.

## 4. Propagation (all one rule + the policy)

Let `U`, `V` be operand units.

| Op family | Result | Invalid/ambiguous ⇒ policy applies when… |
| --- | --- | --- |
| `mul` / `div` / `floor_divide` | `U·V` / `U∕V` | never — always defined |
| `pow(n)` | `Uⁿ` (scalar rational `n`) | tensor/unknown exponent, or non-dimensionless base |
| `matmul` / `einsum` / `tensordot` | `U·V` | the **contracted** axis is not unit-uniform per side (summed terms differ) |
| `add`/`sub`, `maximum`/`minimum`, comparisons | `U` (convert `V`→`U`) | `U`,`V` **incompatible** dimensions |
| `sum`/`mean`/`cumsum`/`std`/`var`/`norm` over axis *a* | `U` (`U²` for `var`) | axis *a* carries **incompatible** per-position units |
| `prod` over size-*k* axis | `Uᵏ` / Π of the axis units | never — always defined |
| `exp`/`log`/`sin`/`sigmoid`/softmax/… (transcendental) | dimensionless | input is **not dimensionless** |
| `abs`/`neg`/`clamp`/`sort`/`flip`/reshape/… | `U` | never — pure structure/sign |
| plain-tensor / scalar operand | dimensionless | — |
| `sel`/`isel`/`[]` to one position on a unit axis | that unit folds into `base_unit` | — |

"Policy applies" = **drop** the unit by default, or **raise** under
`unit_policy="strict"`. Everything else is the plain algebra of §2.1.

### Worked examples

```python
# default policy = "drop"
(volts + amps)            # incompatible -> unitless result (silently)
torch.exp(volts)          # non-dimensionless input -> unitless result
channels.sum(dim="q")     # [V, A, W] incompatible -> unitless result
(volts * amps)            # -> W          (always; pure algebra)
(volts / seconds)         # -> V/s        (always)

with set_options(unit_policy="strict"):
    volts + amps          # ValueError: incompatible units 'V' and 'A'
    torch.exp(volts)      # ValueError: exp expects a dimensionless argument
```

## 5. Relationship to the other layers

- **Orthogonal to coordinate units (0001).** A tensor may have both — a `t` axis
  in seconds (tick metric) *and* data in volts (value unit); they never interact.
- **Reuses structured coordinates (0002)** for per-axis (heterogeneous) units.
- **Reuses `unit_backend` (0001)** for *all* the unit **algebra**, so we
  implement almost none of it ourselves. pint supplies: parse/normalise
  (`ureg.Unit(s)`), multiply/divide/power (`Unit * Unit`, `Unit ** n`),
  compatibility & dimensionless checks (`.dimensionality`, `.dimensionless`),
  and the conversion **factor** (`ureg.Quantity(1, a).to(b).magnitude` — a
  scalar we multiply into the tensor). What *this package* implements is only
  the thin wiring: which op maps to which unit transform (§4), the `unit_policy`
  drop/strict switch, and the metadata plumbing (`_data_unit` + coordinate
  `unit`s). With `unit_backend=None` the whole layer is inert (no base unit,
  coordinate `unit`s stay opaque strings) — zero overhead by default.

## 6. Suggested phasing

The general rule is adopted from the start; phasing is about **op coverage**,
not restricting the model:

1. **base unit + core algebra** — `unit=`/`.unit`/`to_unit`, `*`/`/`/`pow`,
   `add`/`sub` compat, matmul, transcendental drop, the `unit_policy` switch.
2. **per-axis (heterogeneous) units** on structured coordinates + the reduction/
   contraction compat handling (all via the same drop/strict policy).
3. **conveniences** — `.magnitude`/detach, unit-aware `repr`, richer conversions.

## 7. Open questions

1. **Detach ergonomics** — `.magnitude` / `.to("V").tensor()` to drop to a plain
   tensor: spelling and how explicit.
2. **Implicit conversion on `add`** — when dimensions agree (V + mV), auto-
   convert to the left unit (proposed) vs require exact match.
3. **`repr` / serialisation** of a (possibly heterogeneous) unit.
4. **Per-op policy override** — is one global `unit_policy` enough, or do some
   ops want their own (mirroring `combine_axes`'s per-field dict)? Deferred
   until a need appears.

## Related note

`combine_axes` already accepts both a permanent `set_options(...)` call *and* a
`with` block (the same object is setter and context manager), so it is already
package-wide-settable; if a broader configuration surface (env var / config
file) is wanted for *all* options, that's a separate issue — happy to file it.

## References

- pint unit algebra — <https://pint.readthedocs.io>
- Proposal 0001 (coordinate units) — the *other* unit
- Proposal 0002 (structured coordinates) — storage for heterogeneous units
