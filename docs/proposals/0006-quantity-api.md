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
compatibility checks, a convert-and-drop shortcut, in-place conversion, the
unit-simplification family (`to_base_units`/`to_reduced_units`/`to_compact`/
`to_preferred`), and letting `.to()` itself recognize a backend unit (plus a
general in-place `.to_()`). This proposal scopes exactly those additions,
works out how unit conversion should interact with **heterogeneous per-axis
units** (§3, raised in review — a real gap `to_units()` doesn't cover today),
resolves the `.magnitude` vs. complex-`abs` naming question raised alongside
#143, and specifies the graceful-degradation contract for every new member
when no unit backend is selected.

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
| `m`, `u` (short aliases for `magnitude`/`units`) | **add** — reversed from an earlier draft; see §2.3 |
| `ito` | **add**, as `.to_units_(unit)` (§2.4) — in-place conversion; `XTensor` already has this pattern (`rename_`/`swap_dims_`), see revised reasoning below |
| `to(unit)` | **extend** `XTensor.to()` itself to recognize a backend `Unit`/`Quantity` positional argument (§2.6) |
| — (no pint equivalent; new) | `.to_()`, a general in-place counterpart of `.to()` (§2.7) |
| `to_base_units`, `to_reduced_units`, `to_compact`, `to_preferred` | **add** (§2.8) — all four are a thin pass-through to the backend, no XTensor-level policy needed even for `to_preferred` (see revised reasoning below) |
| `to_root_units`, `to_unprefixed` | **skip** — niche pint-registry-system-specific variants, not proposed here |
| `ito_base_units`, `ito_preferred`, `ito_reduced_units`, `ito_root_units`, `ito_unprefixed` | **skip for now** — in-place variants of the §2.8 family; real but rarer than the plain conversions, not worth the surface area yet |
| `plus_minus` | **skip** — uncertainty propagation, no analog in scope |
| `to_timedelta`, `from_list`/`from_sequence`/`from_tuple`, `visualize`, `compute`, `persist`, `force_ndarray*`, `format_babel` | **skip** — Dask/pandas/babel-formatting interop, not applicable |
| `real`, `imag`, `T`, `dot`, `prod`, `clip`, `fill`, `flat`, `searchsorted`, `tolist`, `shape`, `ndim`, `dtype`, `compare`, `unit_items`, `compatible_units`, `UnitsContainer` | **already inherited** — these are `torch.Tensor`/array-protocol members pint re-exposes for numpy interop; `XTensor` already has them natively as a `Tensor` subclass, carrying the unit through per 0003 §4's catch-all structure-preserving row |

So the actual proposal is thirteen new members across §2.1–2.8
(`dimensionality`, `dimensionless`, `is_compatible_with`, `m_as`, `m`, `u`,
`to_units_`, `.to()`'s unit dispatch, `to_()`, `to_base_units`,
`to_reduced_units`, `to_compact`, `to_preferred`), plus a design extension to
`to_units()` itself for heterogeneous per-axis units (§3) — no non-goals
remain.

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

### 2.3 `.m_as(unit)`, and reversing course on `.m`/`.u`

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
word beyond the single letter.

**Reversed from the first draft**: `.m`/`.u` (pint's own short aliases for
`.magnitude`/`.units`) were dropped there as "too terse for a public API, no
torch precedent." Fair pushback in review: that reasoning doesn't survive
`m_as` being added right next to them under the identical logic ("matches
pint's literal name"). Verified there's no collision — neither `m` nor `u` is
an existing `torch.Tensor` attribute (`hasattr(torch.zeros(3), "m")` and
`"u"` are both `False`), so nothing is shadowed. Adding both as plain
one-line property aliases:

```python
@property
def m(self) -> tx.Self:
    """Alias for `.magnitude`."""
    return self.magnitude

@property
def u(self) -> tx.Optional[str]:
    """Alias for `.units`."""
    return self.units
```

`.u` is read-only (an alias, not a second spelling of the setter — `.units =
...` stays the one way to annotate, avoiding two equally-valid assignment
spellings for the same thing).

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

**Deliberately not proposed here**: in-place counterparts of the §2.7 family
below (`ito_base_units` etc.) — real, but rarer than plain conversion, and
easy to add later following the same pattern once there's demand.

### 2.5 `.to()` learns to recognize a backend unit

Raised in review: does `.to()` already dispatch on its positional argument's
*type* (dtype vs. device vs. tensor), the way core PyTorch does — and if so,
can that dispatch also recognize a pint unit?

Checked directly: `XTensor` doesn't currently override `.to()` at all (no
override registered anywhere in the codebase) — it's the plain inherited
`torch.Tensor.to()`, with names/coords/units carried onto the result only via
`ExtendedTensor`'s default fallback (any unregistered op propagates the
subclass's metadata attributes from its first tensor argument, per
`_extended.py`). The type-based dispatch among dtype/device/tensor/
`memory_format` isn't something `XTensor`'s own code inspects — it's torch's
own C-level overload resolution, and it does **not** recognize a pint object
today:

```python
>>> t = torch.zeros(3)
>>> t.to(ureg.mm)
TypeError: to() received an invalid combination of arguments - got (Unit), ...
```

So adding pint-unit recognition means intercepting *before* that dispatch —
exactly the pattern `__mul__`/`__rmul__` already use for `x * u.mm` (0003
§2.4), just simpler here since `.to()` is an ordinary method (no reflected-
operator precedence issue to route around):

```python
def to(self, *args: tx.Any, **kwargs: tx.Any) -> tx.Self:
    """
    Same as `torch.Tensor.to`, plus one extra form: pass a backend
    `Unit`/`Quantity` as the sole positional argument to convert the data
    unit instead (`x.to(ureg.mm)` is `x.to_units("mm")`). A `Quantity`'s
    own magnitude is ignored, matching pint's own `Quantity.to(quantity)`
    (verified: `q.to(3 * ureg.mV)` converts to `mV`, the `3` is not
    applied). Never triggers on a plain string, so it can't collide with
    torch's own device-string overload (`x.to("cuda")`).
    """
    if args and _units.is_unit_like(args[0]) and not kwargs and len(args) == 1:
        _, unit = _units.split_quantity(args[0])
        return self.to_units(unit)
    return Tensor.to(self, *args, **kwargs)
```

Restricted to a backend **object** (`_units.is_unit_like`, which is already
`False` for plain strings and already gated on `unit_backend="pint"`), never
a string — a bare unit *string* stays firmly torch's own device-string
territory (`"cuda"`, `"cpu"`), which would be genuinely ambiguous to
reinterpret. With no backend active, `is_unit_like` is always `False`, so
`.to()` falls straight through to today's plain-torch behaviour — no
degradation table entry needed here beyond what `.to_units()` already has.

### 2.6 `.to_()` — a general in-place `.to()`

Raised in review, once in-place conversion (§2.4) and unit-aware `.to()`
(§2.5) were both on the table: can `.to()` have a general in-place
counterpart too, one that *forbids* an actual dtype/device change (raising
if the request would require one) while still allowing other metadata (the
unit) to mutate?

That composes cleanly with what's already here — `.to_()` is `.to()`'s
in-place counterpart the same way `to_units_` is `to_units`'s:

```python
def to_(self, *args: tx.Any, **kwargs: tx.Any) -> tx.Self:
    """
    In-place `.to()`. A backend `Unit`/`Quantity` argument converts the
    unit in place (delegates to `to_units_`). A dtype/device argument
    that would actually change dtype or device raises -- that's not
    something an in-place op can do -- but one that resolves to a no-op
    (already the requested dtype/device) succeeds trivially, matching
    the spirit of `rename_`/`swap_dims_`: mutate what can be mutated,
    refuse what can't.
    """
    if args and _units.is_unit_like(args[0]) and not kwargs and len(args) == 1:
        _, unit = _units.split_quantity(args[0])
        return self.to_units_(unit)
    result = Tensor.to(self, *args, **kwargs)
    if result.dtype != self.dtype or result.device != self.device:
        raise ValueError(
            "to_: in-place .to() cannot change dtype or device "
            f"(would produce dtype={result.dtype}, device={result.device})"
        )
    return self
```

Checking `result.dtype`/`result.device` against `self`'s (rather than, say,
`result is self`) is deliberate: torch's own `.to()` returns a genuinely new
tensor even for a no-op conversion when `copy=True` is explicitly passed, and
that shouldn't be treated as a rejected dtype/device change — only an actual
mismatch should raise.

### 2.7 The unit-simplification family: `to_base_units`, `to_reduced_units`, `to_compact`, `to_preferred`

Revised twice now. First draft deferred this whole family behind a "needs a
policy decision" concern; second draft split it, moving in the three
deterministic members but keeping `to_preferred` out, reasoning that an
explicit preferred-units *table* needed a home (a `set_options` global?
Per-call?) that 0003 never had to provide. Raised in review: why not just
defer entirely to whatever pint does there, the same way `to_base_units`
et al. already do? Checked pint's own signature directly —
`to_preferred(preferred_units: list[UnitLike] | None = None)` — and that
settles it: it's a **per-call optional argument**, not a registry-wide
setting `XTensor` would need to own. Passing nothing reproduces pint's exact
behaviour (raises `'default_preferred_units' is not defined in the unit
registry'` if the backend registry has no default configured); passing a
list is the caller supplying the table **at the call site**, same as pint
itself. There's no policy for this package to invent — it's a pure
pass-through:

```python
>>> q = ureg.Quantity(5000, "g*mm/s**2")
>>> q.to_base_units()             # -> 0.005 kilogram * meter / second ** 2
>>> q.to_reduced_units()          # -> 5000 gram * millimeter / second ** 2 (this case, no-op)
>>> q.to_compact()                # -> 5.0 gram * meter / second ** 2
>>> q.to_preferred([ureg.N])      # -> 0.005 newton  (only with an explicit or registry-default table)
```

All four are in scope, each a thin wrapper mirroring `to_units`'s own shape
(compute a target unit string via the backend, then reuse the same
convert-and-carry machinery `to_units` already has):

```python
def to_base_units(self) -> tx.Self: ...    # convert to SI base units
def to_reduced_units(self) -> tx.Self: ... # convert to the reduced/simplified form
def to_compact(self) -> tx.Self: ...       # pick a "nice"-scale prefix
def to_preferred(
    self, preferred_units: tx.Optional[tx.List[str]] = None
) -> tx.Self:
    """
    Convert to whichever unit the backend's preferred-units logic picks.
    `preferred_units` is a list of unit strings to guide it (mirrors
    pint's `Quantity.to_preferred`); omit it to use the backend
    registry's own default, which raises if none is configured -- exactly
    pint's behaviour, not a new one. Requires a unit already set and
    `unit_backend="pint"`.
    """
```

pint's own `to_preferred` wants `Unit` objects in the list, not strings
(`AttributeError: 'str' object has no attribute 'dimensionality'` if you pass
plain strings) — so the wrapper normalises each string in `preferred_units`
to `ureg.Unit(s)` before delegating, keeping the string-only input contract
the rest of this API already has (same shape as `x * u.mm` already accepting
a backend-native operand at the call boundary, 0003 §2.4, while storage stays
canonical strings).

Each of the four requires `unit_backend="pint"` and a unit already set, same
failure mode as `to_units()`.

Implementation note: since this is composition on top of `to_units`, it may
land as a follow-up PR after the core §2.1–2.6 additions rather than in the
same one — the design is settled either way.

### 2.8 Graceful degradation (the audit `#143` asked for)

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
| `.m` / `.u` | no special case — plain aliases of `.magnitude`/`.units`, which already work with no backend (§2.3) |
| `.to(unit)` | `_units.is_unit_like` is already `False` without a backend, so the unit-recognition branch never triggers; falls straight through to plain `Tensor.to()` (§2.5) |
| `.to_units_(unit)` | same `ValueError` contract as `.to_units()` (§2.4) |
| `.to_(unit)` | same as `.to_units_()` for the unit path; the dtype/device path never needs a backend at all (§2.6) |
| `.to_base_units()` / `.to_reduced_units()` / `.to_compact()` / `.to_preferred(...)` | same `ValueError` contract as `.to_units()` (§2.7) |

No other existing 0003 member was found to have a gap during this audit — the
constructor, `.units` get/set, and `.to_units()` all already fail with a
`ValueError` naming the missing backend rather than an opaque `AttributeError`.

## 3. Heterogeneous per-axis units and `to_units()`

Raised in review: when a tensor has **per-axis (heterogeneous) units** (0003
§2.2/§4, structured-coordinate `unit` fields), what does `to_units()` actually
do? Does it only touch the whole-tensor base unit? Do we need a way to force
the *entire* tensor into one uniform unit?

**Current behaviour, verified directly:**

```python
# A fully heterogeneous tensor (no base unit at all -- V/A/W channels):
x = xtensor(data, names=("q", "t"), coords={"q": [
    {"name": "voltage", "unit": "V"}, {"name": "current", "unit": "A"},
    {"name": "power", "unit": "W"}]})
x.units            # None -- no base unit was ever set
x.to_units("mV")   # ValueError: to_units: this tensor has no unit to convert

# A tensor with BOTH a base unit and heterogeneous per-axis units:
y = xtensor(data, names=("q", "t"), units="m", coords={"q": [...same 3...]})
y.units            # "meter"
z = y.to_units("mm")
z.units            # "millimeter" -- base converted correctly
z.coords["q"]      # UNCHANGED -- still {"unit": "V"/"A"/"W"} per position
```

The second case isn't a bug — `to_units()` rescales only the **base** factor
of `unit(x[i,j]) = base · Π(coord units)` (0003 §2.1); every position's data
gets multiplied by the *same* base conversion factor regardless of what
per-position unit rides alongside it, which is dimensionally consistent by
construction. The real gap is the **first** case: there is no way today to
take a heterogeneous axis and fold it into one uniform unit, because
`to_units()` requires a base to already exist.

**Where "fold to one unit" is well-defined vs. not.** The `V`/`A`/`W` example
above can *never* have a single meaningful target unit — voltage, current,
and power aren't the same dimension, so "make it all volts" is not just
unimplemented, it's dimensionally nonsensical (exactly the case
`unit_policy` already exists to drop/raise on). The case where folding to one
unit **is** well-defined is a per-position axis that's **compatible but
differently scaled** — e.g. `coords={"q": [{"unit": "mm"}, {"unit": "cm"},
{"unit": "m"}]}`, three positions of the same dimension (length) at different
prefixes. There, "convert the whole tensor to mm" has one obvious answer:
rescale each position's data by its own mm-conversion factor, then the
per-position `unit` field becomes redundant (every position is now mm, same
as the new base) and should be **cleared** — leaving it set would double-count
under the `base · Π(coord units)` formula (base becomes `"mm"` *and* every
position still carries `"unit": "mm"` → effective unit reads as `mm²`,
which is wrong).

**Proposed extension**: let `to_units(unit)` handle this case directly,
rather than adding a separate verb — when a per-position axis is fully
dimensionally **compatible** with `unit`, fold it in (rescale per-position,
clear that axis's `unit` field, done in the same call); when a per-position
axis has any **incompatible** position, that axis is left as today (only the
base converts) — consistent with the existing drop-on-incompatible
philosophy, applied per-axis rather than introducing a new whole-tensor
failure mode.

**Open question this doesn't fully resolve**: what about a single axis with
**mixed compatibility** — some positions compatible with `unit`, others not
(e.g. `["mm", "s"]` on the same axis, converting to `"mm"`)? Two honest
options, both defensible:

1. **Axis-level all-or-nothing**: any incompatible position anywhere on the
   axis leaves the *whole axis* untouched (only the base converts) — simplest
   to reason about, matches how `unit_policy` already treats a whole
   operation as compatible-or-not rather than element-by-element.
2. **Position-level**: convert whichever positions are compatible, leave the
   rest as-is (now genuinely mixed: some positions folded into the base,
   others still carrying their own incompatible unit) — more capable, but a
   materially more complex model (the per-position `unit` field can no
   longer be uniformly cleared, since some positions still need it).

Leaning towards (1) for a first pass — it composes with everything else in
this proposal without inventing a new partial-clear state for structured
coordinates — but this needs your call before implementation, since it's a
real behavioural choice, not just a naming one.

## 4. Open questions

1. Confirm `.magnitude` naming (§5) doesn't need revisiting given the pint
   precedent found below.
2. **Mixed-compatibility axis** (§3): axis-level all-or-nothing, or
   position-level partial fold?

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
