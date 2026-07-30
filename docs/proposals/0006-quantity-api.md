# Proposal 0006 — A `pint.Quantity`-shaped API on `XTensor`

| | |
| --- | --- |
| **Status** | Proposed — not yet implemented |
| **Author** | (proposed) |
| **Created** | 2026-07-30 |
| **Tracking** | [#143](https://github.com/bagofseeds/fiery-xtensor/issues/143); builds on Proposal 0003 (data units) |

## Abstract

Proposal 0003 gave `XTensor` a data unit (`.units`, `.to_units()`, `.magnitude`)
and the arithmetic to go with it (unit algebra on `*`/`/`/`pow`, compatible-unit
conversion on `add`/`sub`/compare, drop-or-raise via `unit_policy`). That is
already most of what makes a united `XTensor` feel like "a `pint.Quantity` that
is also a named tensor" — but `XTensor` doesn't yet expose a few small,
frequently-reached-for corners of `pint.Quantity`'s read-only query surface:
dimensionality, compatibility checks, and a convert-and-drop shortcut. This
proposal scopes exactly those additions, resolves the `.magnitude` vs.
complex-`abs` naming question raised alongside #143, and specifies the
graceful-degradation contract for every new member when no unit backend is
selected.

## 0. What's already there (0003, landed)

- `.units` (get/set, constructor kwarg), `.to_units(unit)` (convert), `.magnitude`
  (drop, still an `XTensor`).
- Arithmetic: `*`/`/`/`pow` combine units; `add`/`sub`/comparisons convert a
  compatible right operand to the left's unit, or drop/raise (`unit_policy`) on
  an incompatible one; `matmul`/`einsum`/`tensordot` fold contracted-axis units.
- `x * u.mm` attaches a unit from the backend's own `Unit`/`Quantity` objects.
- Heterogeneous (per-axis) units via structured coordinates (0002).

None of that is revisited here. The gap is specifically the small *query*
surface pint exposes beyond arithmetic — the bit a user reaches for when they
want to ask a united tensor a question rather than compute with it.

## 1. Surveying `pint.Quantity`'s public API

Checked against pint 0.25 (`[m for m in dir(pint.Quantity(1, "V")) if not
m.startswith("_")]`), grouped by what to do with each:

| pint member | Disposition |
| --- | --- |
| `magnitude`, `units`, `to(unit)` | **have it** — `.magnitude`, `.units`, `.to_units(unit)` |
| arithmetic, comparisons | **have it** — 0003 §4 |
| `dimensionality` | **add** (§2.1) |
| `dimensionless` | **add** (§2.1) — thin wrapper, `_units.dimensionless` already exists internally |
| `check(unit)` / `is_compatible_with(unit)` | **add**, one spelling (§2.2) |
| `m_as(unit)` | **add** — convert-and-drop shortcut (§2.3) |
| `m`, `u` (short aliases for `magnitude`/`units`) | **skip** — too terse for a public API, no torch precedent |
| `ito`, `ito_base_units`, `ito_preferred`, `ito_reduced_units`, `ito_root_units`, `ito_unprefixed` | **skip, deliberately** (§3.1) — in-place conversion, contradicts the view/annotate design |
| `to_base_units`, `to_reduced_units`, `to_compact`, `to_preferred`, `to_root_units`, `to_unprefixed` | **defer** (§3.2) — unit-simplification family, real but lower-value; open question |
| `plus_minus` | **skip** — uncertainty propagation, no analog in scope |
| `to_timedelta`, `from_list`/`from_sequence`/`from_tuple`, `visualize`, `compute`, `persist`, `force_ndarray*`, `format_babel` | **skip** — Dask/pandas/babel-formatting interop, not applicable |
| `real`, `imag`, `T`, `dot`, `prod`, `clip`, `fill`, `flat`, `searchsorted`, `tolist`, `shape`, `ndim`, `dtype`, `compare`, `unit_items`, `compatible_units`, `UnitsContainer` | **already inherited** — these are `torch.Tensor`/array-protocol members pint re-exposes for numpy interop; `XTensor` already has them natively as a `Tensor` subclass, carrying the unit through per 0003 §4's catch-all structure-preserving row |

So the actual proposal is three additions (§2) plus two explicit non-goals worth
recording so they aren't re-litigated later (§3).

## 2. What to add

### 2.1 `.dimensionality` / `.dimensionless`

```python
@property
def dimensionality(self) -> str:
    """
    The physical dimensionality of this tensor's unit (e.g. `"[length]"`,
    `"[mass] * [length] ** 2 / [time] ** 3"`), or `""` if unitless.
    Requires `unit_backend="pint"`.
    """

@property
def dimensionless(self) -> bool:
    """Whether this tensor's unit is dimensionless (or unset). Always
    `True` with no unit backend, since an opaque unit carries no known
    dimensionality to check."""
```

`dimensionality` returns a **string** (`str(pint.Unit(u).dimensionality)`), not
a raw pint `UnitsContainer` — consistent with 0003's "units stored/returned as
canonical strings, never a raw backend object" rule (§5a), so the member stays
meaningful if a future backend adapter (astropy, unyt) is swapped in.
`dimensionless` is a thin wrapper over the `_units.dimensionless()` helper that
already backs the transcendental-function guard (0003 §4) — this just exposes
it.

### 2.2 `.is_compatible_with(unit)`

```python
def is_compatible_with(self, unit: str) -> bool:
    """
    Whether this tensor's unit shares a dimensionality with `unit` (so
    `to_units(unit)` would succeed). `False` if this tensor has no unit.
    Requires `unit_backend="pint"`.
    """
```

A thin wrapper over the existing `_units.compatible()` helper (already used
internally for the `add`/`sub`/compare conversion rule). pint offers this under
*two* names, `check()` and `is_compatible_with()` — pure aliases of each other.
Shipping one avoids doubling the API for no benefit; `is_compatible_with` reads
better standalone (`check` is ambiguous out of context). Open to bikeshedding
in review (§4.1).

### 2.3 `.m_as(unit)`

```python
def m_as(self, unit: str) -> tx.Self:
    """
    Convert to `unit` and drop the annotation in one step — sugar for
    `x.to_units(unit).magnitude`. Still an `XTensor` (see `.magnitude`'s
    own note on getting a plain `torch.Tensor`).
    """
    return self.to_units(unit).magnitude
```

Pure composition of two existing operations; no new mechanism. Matches pint's
`m_as` (their shorthand for "magnitude, as this unit").

## 2.4 Graceful degradation (the audit `#143` asked for)

Every method above needs a unit backend to do anything; the failure mode with
`unit_backend=None` must be as clear as `to_units()`'s existing one
(`_units.factor` raises `ValueError("unit conversion requires
unit_backend='pint'")`, not an `AttributeError` or a silent wrong answer).
Same contract for the new members:

| member | with no backend |
| --- | --- |
| `.dimensionality` | raise `ValueError("dimensionality requires unit_backend='pint'")` |
| `.dimensionless` | `True` (no known dimensionality to be non-trivial about — matches `_units.dimensionless(None)`'s existing behaviour, which already returns `True` for an unset/opaque unit) |
| `.is_compatible_with(unit)` | `_units.compatible()` already degrades to plain string equality without a backend (existing behaviour, unchanged) — no new failure mode needed |
| `.m_as(unit)` | inherits `to_units()`'s existing `ValueError` on `self.units is None` or no backend |

No other existing 0003 member was found to have a gap during this audit — the
constructor, `.units` get/set, and `.to_units()` all already fail with a
`ValueError` naming the missing backend rather than an opaque `AttributeError`.

## 3. Explicit non-goals (recorded so they aren't re-asked)

### 3.1 No in-place conversion (`ito*`)

pint's `ito()` family rescales the `Quantity`'s own storage in place. `XTensor`
never mutates the underlying tensor storage of an existing object as a side
effect of a unit operation — `.units = ...` mutates only the **metadata**
(`_data_units`, an annotation), never the data, and `to_units()` returns a
*new* tensor (0003 §2.3: "assign ≠ convert"). An in-place data rescale would
also silently break autograd for any tensor requiring grad (the graph would
point at values that no longer match forward-computed history) and has no
existing precedent among `XTensor`'s other ops, which are consistently
functional. Not shipping this family is a design commitment, not an oversight.

### 3.2 No unit-simplification family (`to_base_units`, `to_compact`, …) — yet

pint's `to_base_units`/`to_reduced_units`/`to_compact`/`to_preferred` rewrite a
compound unit into a canonical or "nicest" equivalent form (e.g. simplify
`kg*m/s**2` toward `N` under a preferred-unit table, or reduce to SI base
units). These are genuinely useful but orthogonal to the rest of this
proposal — they need a policy decision (which "preferred units" table? global
option or per-call?) that 0003 never had to make, since nothing before now
rewrote a unit's *form* rather than converting its *value*. Recording as an
open question (§4.2) rather than silently dropping it, since it's plausible a
user will want it once `.dimensionality` makes compound units more visible.

## 4. Open questions

1. **Naming**: `is_compatible_with` vs. `check`, or ship both as aliases? Leaning
   towards one name (`is_compatible_with`) per the no-API-bloat instinct that
   already dropped `m`/`u` as aliases — but pint ships both, so there's
   precedent either way.
2. **Unit-simplification family** (§3.2): worth a future issue once there's a
   concrete use case, or fold into this proposal's scope now?
3. Confirm `.magnitude` naming (§5) doesn't need revisiting given the pint
   precedent found below.

## 5. Resolving the `.magnitude` vs. complex-`abs` question

The issue raised: does `.magnitude` (0003's "drop the unit annotation" verb)
read confusingly against the mathematical sense of "magnitude of a complex
number" (its modulus, `|z|`)?

Checked directly against pint 0.25:

```python
>>> import pint
>>> u = pint.UnitRegistry()
>>> q = u.Quantity(-3 + 4j, "V")
>>> q.magnitude
(-3+4j)
>>> abs(q).magnitude
5.0
```

**`pint.Quantity.magnitude` itself returns the raw underlying value verbatim —
it never takes an absolute value, complex or otherwise.** `XTensor.magnitude`
already matches this exactly (it drops the unit annotation and returns the
tensor's actual values unchanged, negative or complex as they are). So there is
no real naming collision with the library this proposal mirrors — only a
possible first-read ambiguity for a user coming from pure-math terminology.

**Decision: keep `.magnitude` as named.** It is the correct pint-parity name.
Resolve the ambiguity with one clarifying line in the existing docstring (a
doc-only change, not an API change):

> `x.magnitude` drops the *unit annotation* — it is not the mathematical
> modulus. For the absolute value of a (possibly complex) tensor, use
> `x.abs()` / `torch.abs(x)`, which `XTensor` already inherits unchanged.

## References

- Proposal 0003 (data units) — the arithmetic and storage this proposal builds on
- pint `Quantity` API — <https://pint.readthedocs.io/en/stable/api/base.html>
- Issue #143
