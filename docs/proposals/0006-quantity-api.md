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
frequently-reached-for corners of `pint.Quantity`'s API: dimensionality,
compatibility checks, a convert-and-drop shortcut, in-place conversion, and
the unit-simplification family (`to_base_units`/`to_reduced_units`/
`to_compact`). This proposal scopes exactly those additions, resolves the
`.magnitude` vs. complex-`abs` naming question raised alongside #143, and
specifies the graceful-degradation contract for every new member when no unit
backend is selected.

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
| `ito` | **add**, as `.to_units_(unit)` (§2.4) — in-place conversion; `XTensor` already has this pattern (`rename_`/`swap_dims_`), see revised reasoning below |
| `to_base_units`, `to_reduced_units`, `to_compact` | **add** (§2.5) — deterministic, no policy needed; mimic pint directly |
| `to_preferred`, `to_root_units`, `to_unprefixed` | **defer** — `to_preferred` needs a preferred-units-table policy pint itself doesn't default (§3); `to_root_units`/`to_unprefixed` are niche pint-registry-system-specific variants, not proposed here |
| `ito_base_units`, `ito_preferred`, `ito_reduced_units`, `ito_root_units`, `ito_unprefixed` | **skip for now** — in-place variants of the §2.5 family; real but rarer than the plain conversions, not worth the surface area yet |
| `plus_minus` | **skip** — uncertainty propagation, no analog in scope |
| `to_timedelta`, `from_list`/`from_sequence`/`from_tuple`, `visualize`, `compute`, `persist`, `force_ndarray*`, `format_babel` | **skip** — Dask/pandas/babel-formatting interop, not applicable |
| `real`, `imag`, `T`, `dot`, `prod`, `clip`, `fill`, `flat`, `searchsorted`, `tolist`, `shape`, `ndim`, `dtype`, `compare`, `unit_items`, `compatible_units`, `UnitsContainer` | **already inherited** — these are `torch.Tensor`/array-protocol members pint re-exposes for numpy interop; `XTensor` already has them natively as a `Tensor` subclass, carrying the unit through per 0003 §4's catch-all structure-preserving row |

So the actual proposal is eight new members across §2.1–2.5 (`dimensionality`,
`dimensionless`, `is_compatible_with`, `m_as`, `to_units_`, `to_base_units`,
`to_reduced_units`, `to_compact`) plus one explicit non-goal worth recording
so it isn't re-litigated later (§3).

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
    """Whether this tensor's unit is dimensionless (or unset). With no
    unit backend this is `not bool(self.units)` -- `True` for no unit or
    an explicitly empty one, `False` for any opaque unit string, since
    there's no dimensionality system to consult otherwise."""
```

`dimensionality` returns a **string** (`str(pint.Unit(u).dimensionality)`), not
a raw pint `UnitsContainer` — consistent with 0003's "units stored/returned as
canonical strings, never a raw backend object" rule (§5a), so the member stays
meaningful if a future backend adapter (astropy, unyt) is swapped in.

`dimensionless` is meant as a thin wrapper over the existing
`_units.dimensionless()` helper (already backing the transcendental-function
guard, 0003 §4) — but that helper's own no-backend behaviour turns out to be
inconsistent with what this property should do: it returns `False` for
`_units.dimensionless("")` today (an explicitly-empty unit is *not* treated as
dimensionless without a backend), where `not bool("")` is `True`. Caught during
review — the property's contract (`not bool(self.units)`) is correct here,
and `_units.dimensionless()` itself should be fixed to match as part of the
implementation, not worked around at the property level.

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
internally for the `add`/`sub`/compare conversion rule).

Correction from an earlier draft: `check()` and `is_compatible_with()` are
**not** pure aliases in pint — checked against pint 0.25's source directly.
`check(dimension)` compares against a **dimension expression**
(`self.dimensionality == registry.get_dimensionality(dimension)`), while
`is_compatible_with(other)` compares against a **unit/quantity/string** and
additionally supports pint's *context* system (`*contexts` — temporary
conversion rules, e.g. spectroscopy's frequency↔wavelength). For a plain unit
string the two usually agree, but `is_compatible_with` is the more general,
more directly useful one for `XTensor`'s case (checking against another
unit string, which is all `_units.compatible()` does today) — keeping that
name and skipping `check` isn't just avoiding a redundant alias, it's picking
the method whose actual semantics match what `_units.compatible()` provides.
`XTensor`'s version won't take pint's `*contexts`/`**ctx_kwargs` (no context
system exists in this codebase), so document it as the simpler, unit-only
form.

On which predates the other: inconclusive from pint's own changelog —
`is_compatible_with` bugfixes are on record as far back as pint 0.19, and
`Quantity.check`'s own addition isn't separately dated there (a same-named
`UnitRegistry.check` *decorator*, a different feature, dates to 0.7). Not
worth more digging; the semantic difference above is the more useful signal
for this decision anyway.

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

Raised in review: could `m_as` be misread as related to `torch.Tensor.mT`/`.mH`
(PyTorch's batch-matrix transpose/conjugate-transpose properties), since both
start with a bare `m`? Low risk in practice — `mT`/`mH` are no-argument
*properties*, `m_as(unit)` is a *method* that requires a unit argument, so the
call shapes don't collide (`x.mT` vs. `x.m_as("V")`), and there's no shared
word beyond the single letter. Keeping the pint-literal name here, same as
elsewhere in this proposal, rather than inventing an `XTensor`-specific
spelling for a low-value convenience method that exists mainly for pint
familiarity in the first place.

### 2.4 `.to_units_(unit)` — in-place conversion

**Revised from the first draft**, which proposed skipping `ito()` entirely on
the theory that in-place unit conversion has no precedent and would break
autograd. Both premises don't hold up:

- **There *is* precedent.** `XTensor` already ships in-place metadata mutators
  with the trailing-underscore convention — `rename_` and `swap_dims_`
  (Proposal 0005) mutate axis metadata on `self` and return it, exactly the
  shape an in-place unit conversion would take.
- **Autograd doesn't silently break.** Verified directly: an in-place rescale
  (`y.mul_(factor)`) on a **non-leaf** tensor computes the correct gradient,
  identically to any other in-place torch op. The only failure is PyTorch's
  own standard restriction on **leaf** tensors requiring grad (`a view of a
  leaf Variable that requires grad is being used in an in-place operation`) —
  the exact same `RuntimeError` any `mul_`/`add_`/etc. already raises on a
  leaf today. Nothing unit-specific about it.

So the honest constraint on an in-place conversion isn't "autograd" or "no
precedent" — it's the same constraint every in-place torch op already has:
the operation must not need to change dtype (an int-dtype tensor scaled by a
non-integer factor already fails on plain `mul_` today, not just here) or
device (moving data is definitionally not an in-place operation). Given that,
adding it is straightforward:

```python
def to_units_(self, unit: str) -> tx.Self:
    """
    Convert to `unit` in place -- rescales the data and updates the unit
    annotation on `self`, returning `self`. Same restrictions as any other
    in-place op (`mul_`, `add_`, ...): raises on a leaf tensor that requires
    grad, and raises if the scale factor can't be applied without changing
    dtype. Requires a unit already set and `unit_backend="pint"`.
    """
    current = self.units
    if current is None:
        raise ValueError("to_units_: this tensor has no unit to convert")
    unit = _units.normalise(unit)
    self.mul_(_units.factor(current, unit))
    self._data_units = unit
    return self
```

**Naming**: spelled `to_units_` (trailing underscore), not pint's bare `ito` —
consistency with `rename_`/`swap_dims_`'s already-established in-place
convention wins here over literal pint-mirroring, since this one has a local
precedent to match and pint's own naming (`to_units` → `ito`, not `ito_units`)
doesn't transfer cleanly anyway.

**Deliberately not proposed here**: in-place counterparts of the §2.5 family
below (`ito_base_units` etc.) — real, but rarer than plain conversion, and
easy to add later following the same pattern once there's demand.

### 2.5 The unit-simplification family: `to_base_units`, `to_reduced_units`, `to_compact`

Revised from the first draft, which deferred this whole family behind a
"needs a policy decision" concern. Checked directly against pint: only
`to_preferred` actually needs one (an explicit preferred-units table — it
raises `'default_preferred_units' is not defined in the unit registry` with
none supplied). `to_base_units`, `to_reduced_units`, and `to_compact` are all
**deterministic** — pint computes them from the unit registry alone, no
table or config required:

```python
>>> q = ureg.Quantity(5000, "g*mm/s**2")
>>> q.to_base_units()      # -> 0.005 kilogram * meter / second ** 2
>>> q.to_reduced_units()   # -> 5000 gram * millimeter / second ** 2  (this case, no-op)
>>> q.to_compact()         # -> 5.0 gram * meter / second ** 2
```

So these three are in scope now, each a thin wrapper mirroring `to_units`'s
own shape (compute a target unit string via the backend, then reuse the same
convert-and-carry machinery `to_units` already has):

```python
def to_base_units(self) -> tx.Self: ...    # convert to SI base units
def to_reduced_units(self) -> tx.Self: ... # convert to the reduced/simplified form
def to_compact(self) -> tx.Self: ...       # pick a "nice"-scale prefix
```

Each requires `unit_backend="pint"` and a unit already set, same failure mode
as `to_units()`. `to_preferred(units)` (the one that genuinely needs a table)
stays out of scope for now — see §3.

Implementation note: since this is composition on top of `to_units`, it may
land as a follow-up PR after the core §2.1–2.4 additions rather than in the
same one — the design is settled either way.

### 2.6 Graceful degradation (the audit `#143` asked for)

Every method above needs a unit backend to do anything; the failure mode with
`unit_backend=None` must be as clear as `to_units()`'s existing one
(`_units.factor` raises `ValueError("unit conversion requires
unit_backend='pint'")`, not an `AttributeError` or a silent wrong answer).
Same contract for the new members:

| member | with no backend |
| --- | --- |
| `.dimensionality` | raise `ValueError("dimensionality requires unit_backend='pint'")` |
| `.dimensionless` | `not bool(self.units)` — `True` for no unit or an explicitly empty one, `False` for any opaque unit string (§2.1) |
| `.is_compatible_with(unit)` | `_units.compatible()` already degrades to plain string equality without a backend (existing behaviour, unchanged) — no new failure mode needed |
| `.m_as(unit)` | inherits `to_units()`'s existing `ValueError` on `self.units is None` or no backend |
| `.to_units_(unit)` | same `ValueError` contract as `.to_units()` (§2.4) |
| `.to_base_units()` / `.to_reduced_units()` / `.to_compact()` | same `ValueError` contract as `.to_units()` (§2.5) |

No other existing 0003 member was found to have a gap during this audit — the
constructor, `.units` get/set, and `.to_units()` all already fail with a
`ValueError` naming the missing backend rather than an opaque `AttributeError`.

## 3. Explicit non-goal: `to_preferred` (needs a policy this proposal doesn't set)

pint's `to_preferred(units)` rewrites a compound unit toward whichever member
of a caller-supplied (or registry-default) **preferred-units table** matches
its dimensionality (e.g. simplify `kg*m/s**2` toward `N` because `N` was
listed as preferred for that dimensionality) — unlike `to_base_units`/
`to_reduced_units`/`to_compact` (§2.5), it has no sensible parameter-free
default; pint itself raises without an explicit or registry-configured table.
Adding it means deciding *where that table lives* (a `set_options(...)`
global, a per-call argument, both?) — a genuinely new kind of policy decision
0003 never had to make, since nothing before now let a unit's *form* (as
opposed to its *value*) be configured. Recording as an open question (§4)
rather than silently dropping it.

## 4. Open questions

1. **`to_preferred`** (§3): worth adding now with a `set_options`-scoped
   preferred-units table, or a future issue once there's a concrete use case?
   `to_base_units`/`to_reduced_units`/`to_compact` (§2.5) don't wait on this
   either way, since they need no such table.
2. Confirm `.magnitude` naming (§5) doesn't need revisiting given the pint
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
