# Proposal 0006 — A `pint.Quantity`-shaped API on `XTensor`

| | |
| --- | --- |
| **Status** | Accepted — implemented (all eighteen members of §2.1–2.7, plus §2.8's degradation contract and §4's docstring clarification) |
| **Author** | (proposed) |
| **Created** | 2026-07-30 |
| **Updated** | 2026-07-30 — implemented |
| **Tracking** | [#143](https://github.com/bagofseeds/fiery-xtensor/issues/143); builds on Proposal 0003 (data units) |

## Abstract

Proposal 0003 gave `XTensor` a data unit (`.units`, `.to_units()`, `.magnitude`)
and the arithmetic to go with it (unit algebra on `*`/`/`/`pow`, compatible-unit
conversion on `add`/`sub`/compare, drop-or-raise via `unit_policy`). That is
already most of what makes a united `XTensor` feel like "a `pint.Quantity` that
is also a named tensor" — but `XTensor` doesn't yet expose a few small,
frequently-reached-for corners of `pint.Quantity`'s API: dimensionality
queries (`.dimensionality`/`.dimensionless`/`.unitless`), compatibility
checks, the `.m`/`.u`/`.m_as` short forms, in-place conversion (`.to_units_`
and, for the unit-simplification family, `to_base_units`/`to_reduced_units`/
`to_compact`/`to_preferred` and their four in-place counterparts, all newly
in scope here, not just their `_`-suffixed forms), and letting `.to()`/a new
`.to_()` recognize a backend unit via a `units=` keyword (string or backend
object, not just the positional-object form) plus `names=`/`coords=`
overrides (mirroring `as_xtensor`). This proposal scopes exactly those
additions, describes how they all behave on a tensor with **heterogeneous
per-axis units** (§3 — folding those into `to_units()` itself is explicitly
out of scope, see below), resolves the `.magnitude` vs. complex-`abs` naming
question raised alongside #143, and specifies the graceful-degradation
contract for every new member when no unit backend is selected.

## 0. What's already there (0003, landed)

- `.units` (get/set, constructor kwarg), `.to_units(unit)` (convert), `.magnitude`
  (drop, still an `XTensor`).
- Arithmetic: `*`/`/`/`pow` combine units; `add`/`sub`/comparisons convert a
  compatible right operand to the left's unit, or drop/raise (`unit_policy`) on
  an incompatible one; `matmul`/`einsum`/`tensordot` fold contracted-axis units.
- `x * u.mm` attaches a unit from the backend's own `Unit`/`Quantity` objects.
- Heterogeneous (per-axis) units via structured coordinates (0002).

None of that is revisited here. What's missing spans three shapes: **queries**
(`.dimensionality`, `.dimensionless`, `.unitless`, `.is_compatible_with`),
**short forms** (`.m`/`.u`/`.m_as`), and **conversion variants** `to_units()`
doesn't yet have — in place, via `.to()`/`.to_()`, or into a
backend-computed target unit (the simplification family) rather than one you
name yourself.

## 1. Surveying `pint.Quantity`'s public API

Checked against pint 0.25 (`[m for m in dir(pint.Quantity(1, "V")) if not
m.startswith("_")]`), grouped by what to do with each:

| pint member | Disposition |
| --- | --- |
| `magnitude`, `units`, `to(unit)` | **have it** — `.magnitude`, `.units`, `.to_units(unit)` |
| arithmetic, comparisons | **have it** — 0003 §4 |
| `dimensionality` | **add** (§2.1) |
| `dimensionless` | **add** (§2.1) — thin wrapper, `_units.dimensionless` already exists internally |
| `unitless` | **add** (§2.1) — a *different* predicate from `dimensionless` (a radian-valued quantity is dimensionless but not unitless), worth disposing of explicitly rather than silently dropping |
| `check(dimension)` / `is_compatible_with(unit)` | **add**, one spelling (§2.2) |
| `m_as(unit)` | **add** — convert-and-drop shortcut (§2.3) |
| `m`, `u` (short aliases for `magnitude`/`units`) | **add** — reversed from an earlier draft; see §2.3 |
| `ito` | **add**, as `.to_units_(unit)` (§2.4) — in-place conversion; `XTensor` already has this pattern (`rename_`/`swap_dims_`), see revised reasoning below |
| `to(unit)` | **extend** `XTensor.to()` itself: recognize a backend `Unit`/`Quantity` positional argument, plus new `units=`/`names=`/`coords=` keyword overrides (§2.5) |
| — (no pint equivalent; new) | `.to_()`, a general in-place counterpart of `.to()` with the same overrides (§2.6); `to_compact_` (§2.7) — pint's own `ito*` family has no in-place `to_compact`, but there's no reason `XTensor`'s shouldn't |
| `to_base_units`, `to_reduced_units`, `to_compact`, `to_preferred` | **add** (§2.7) — all four are a thin pass-through to the backend, no XTensor-level policy needed even for `to_preferred` (see revised reasoning below) |
| `to_root_units`, `to_unprefixed` | **skip** — niche pint-registry-system-specific variants, not proposed here |
| `ito_base_units`, `ito_preferred`, `ito_reduced_units` | **add**, as `to_base_units_`/`to_preferred_`/`to_reduced_units_` (§2.7) — reversed from an earlier draft; same trailing-underscore pattern as `to_units_` |
| `ito_root_units`, `ito_unprefixed` | **skip** — in-place variants of the two members already skipped above |
| `plus_minus` | **skip** — uncertainty propagation, no analog in scope |
| `to_timedelta`, `from_list`/`from_sequence`/`from_tuple`/`to_tuple`, `visualize`, `compute`, `persist`, `force_ndarray*`, `format_babel` | **skip** — Dask/pandas/babel-formatting interop, not applicable |
| `real`, `imag`, `T`, `dot`, `prod`, `clip`, `tolist`, `shape`, `ndim`, `dtype`, `put` | **already inherited** — genuine `torch.Tensor` methods/properties (verified: `hasattr(torch.zeros(3), name)` is `True` for each); `XTensor` already has them natively as a `Tensor` subclass, carrying the unit through per 0003 §4's catch-all structure-preserving row |
| `fill`, `flat`, `searchsorted`, `compare` | **skip** — despite looking numpy-inherited, these are **not** `torch.Tensor` members under those names (`hasattr` is `False`) — torch spells the first three `fill_`/`flatten`/`torch.searchsorted`; `compare` has no direct analogue and isn't needed |
| `unit_items`, `compatible_units`, `UnitsContainer` | **skip** — pint's own internal unit-registry machinery, not numpy/array interop at all; `compatible_units` (list every unit `.to()` could reach) is the one a reader might actually want — explicit scoping call to leave it out of this proposal, not an oversight |

So the actual proposal is eighteen new members across §2.1–2.7
(`dimensionality`, `dimensionless`, `unitless`, `is_compatible_with`, `m_as`,
`m`, `u`, `to_units_`, `.to()`'s extended dispatch, `to_()`, `to_base_units`,
`to_reduced_units`, `to_compact`, `to_preferred`, and their four in-place
counterparts) — no non-goals remain; §3 documents (rather than changes)
behaviour on heterogeneous per-axis units.

## 2. What to add

### 2.1 `.dimensionality` / `.dimensionless` / `.unitless`

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

@property
def unitless(self) -> bool:
    """
    Whether this tensor has literally **no unit at all** -- a stricter
    check than `.dimensionless`. Angle units are the case that splits
    them: `xtensor(x, units="rad").dimensionless` is `True` (an angle is
    dimensionally trivial) but `.unitless` is `False` (it still names a
    unit). With no backend the two coincide exactly (`not bool(self.units)`
    for both), since there's no dimensionality system to tell them apart.
    Requires `unit_backend="pint"` for the two to actually differ.
    """
```

`dimensionality` returns a **string** (`str(pint.Unit(u).dimensionality)`), not
a raw pint `UnitsContainer` — consistent with 0003's "units stored/returned as
canonical strings, never a raw backend object" rule (0003 §5a), so the member
stays meaningful if a future backend adapter (astropy, unyt) is swapped in.

`dimensionless` is meant as a thin wrapper over the existing
`_units.dimensionless()` helper (already backing the transcendental-function
guard, 0003 §4) — but that helper's own no-backend behaviour turns out to be
inconsistent with what this property should do: it returns `False` for
`_units.dimensionless("")` today (an explicitly-empty unit is *not* treated as
dimensionless without a backend), where `not bool("")` is `True`. Caught during
review — the property's contract (`not bool(self.units)`) is correct here,
and `_units.dimensionless()` itself should be fixed to match as part of the
implementation, not worked around at the property level. This is a real
(if tiny) behaviour change to a landed 0003 helper, so its impact is worth
stating rather than assuming: checked both existing call sites, and neither
is affected. `_pointwise.py`'s transcendental-function guard short-circuits
on `if not _units.active(): return ...` *before* it ever calls
`_units.dimensionless()`, so the no-backend branch there is unreachable
either way; `_selection.py`'s `if not origin_unit or
_units.dimensionless(origin_unit)` short-circuits on `not origin_unit` first,
so `_units.dimensionless()` is only ever reached with a **truthy** (i.e.
non-empty) `origin_unit` — the exact input the fix doesn't change behaviour
for.

`.unitless` needs a fifth: verified directly against pint 0.25's source
(`Quantity.unitless` is `not bool(self.to_root_units()._units)`, distinct
from `Quantity.dimensionless`, which is `self.dimensionality == {}`-ish and
is `True` for a radian-valued quantity) — worth adding for exactly this
reason, not silently folded into `dimensionless`.

**Backend-protocol note** (applies to `.dimensionality` here and to the whole
§2.7 family below): 0003 §5a's adapter protocol (`normalise`, `equal`/
`compatible`/`dimensionless`, `mul`/`div`/`pow_`, `factor`, `unit_of`/
`magnitude_of`) has no operation for dimensionality, or for computing a
base/reduced/compact/preferred target unit. Implementing `.dimensionality`
and the §2.7 family means growing that protocol with roughly five new
entries (a `dimensionality(unit) -> str`, and `base_units`/`reduced_units`/
`compact_units`/`preferred_units(unit, preferred) -> str`), so the
pint-specific calls this proposal's own snippets show for illustration
(`str(pint.Unit(u).dimensionality)`, `ureg.Unit(s)`) end up living inside
`_units.py`'s pint adapter, not hard-coded in `_tensors.py` — exactly what
0003 §5a exists to prevent. `.is_compatible_with` is the one new member here
that needs no protocol growth: it reuses the existing `compatible()` entry
as-is.

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
  with the trailing-underscore convention — `rename_` (pre-existing) and
  `swap_dims_` (Proposal 0005, which itself cites `rename_`/`rename` as the
  precedent it follows) mutate axis metadata on `self` and return it, exactly
  the shape an in-place unit conversion would take.
- **Autograd doesn't silently break.** Verified directly: an in-place rescale
  (`y.mul_(factor)`) on a **non-leaf** tensor computes the correct gradient,
  identically to any other in-place torch op. The only failure is PyTorch's
  own standard restriction on a **leaf** tensor requiring grad — verified on
  an actual `XTensor` leaf, which raises `a view of a leaf Variable that
  requires grad is being used in an in-place operation` (the "view of"
  wording specifically because `XTensor` construction goes through
  `as_subclass`, which torch treats as a view internally; a bare
  `torch.Tensor` leaf gives the shorter `a leaf Variable that requires
  grad...`). Either way it's the exact same `RuntimeError` any `mul_`/`add_`/
  etc. already raises on a leaf today. Nothing unit-specific about it.

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

**Revised from an earlier draft**, which skipped in-place counterparts of the
§2.7 family below as "rarer, add later." Raised in review: there's no reason
not to, once `to_units_` establishes the pattern — see §2.7.

### 2.5 `.to()` learns to recognize a backend unit — and `units=`/`names=`/`coords=`

Raised in review: does `.to()` already dispatch on its positional argument's
*type* (dtype vs. device vs. tensor), the way core PyTorch does — and if so,
can that dispatch also recognize a pint unit? Follow-up, once that was
settled: `.to()` should also take a `units=` **keyword**, not just a
positional form (mirroring how `dtype=`/`device=` already work both ways) —
and, going further, should it also let you override `names=`/`coords=` in the
same call?

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
t = torch.zeros(3)
t.to(ureg.mm)
# TypeError: to() received an invalid combination of arguments - got (Unit), ...
```

So adding pint-unit recognition means intercepting *before* that dispatch —
exactly the pattern `__mul__`/`__rmul__` already use for `x * u.mm` (0003
§2.4), just simpler here since `.to()` is an ordinary method (no reflected-
operator precedence issue to route around).

**The `names=`/`coords=` idea turns out to already exist**, just not on
`.to()`: `as_xtensor(value, dtype=, device=, units=, names=, coords=)` (#114)
*is* exactly "coerce, optionally convert dtype/device, optionally override
metadata" — the free-function form of what `.to()` with overrides would be.
Rather than reinvent that logic, `.to()` becomes a thin front end that parses
its own positional dtype/device/unit forms, then delegates the metadata part
to `as_xtensor`, reusing its already-reviewed no-op/copy handling instead of
duplicating it:

```python
def to(
    self,
    *args: tx.Any,
    units: tx.Any = arrayutils._UNSET,
    names: tx.Any = arrayutils._UNSET,
    coords: tx.Any = arrayutils._UNSET,
    **kwargs: tx.Any,
) -> tx.Self:
    """
    Same as `torch.Tensor.to` (dtype/device, positional or keyword), plus
    `units=`/`names=`/`coords=` overrides. `names=`/`coords=` are the
    instance-method form of `as_xtensor`'s override kwargs of the same
    name (delegated to it directly, below). `units=` is **not** the same
    as `as_xtensor`'s `units=` -- `as_xtensor(x, units=...)` *annotates*
    (assigns through the `.units` setter, changing no data), matching its
    own "override replaces wholesale" contract; `.to(units=...)`
    **converts** (like `to_units`, requiring a unit already set),
    consistent with `.to()` always meaning conversion, never blind
    reassignment. A bare positional backend `Unit`/`Quantity`
    (`x.to(ureg.mm)`) is sugar for `units=` with that same object; unlike
    the positional form, `units=` also accepts a plain unit string
    directly (a keyword can't collide with torch's own device-string
    overload the way a positional string would).
    """
    if (
        args
        and _units.is_unit_like(args[0])
        and units is arrayutils._UNSET
        and len(args) == 1
        and not kwargs
    ):
        units, args = args[0], ()
    if _units.is_unit_like(units):
        _, units = _units.split_quantity(units)
    result = Tensor.to(self, *args, **kwargs)
    if units is arrayutils._UNSET and names is arrayutils._UNSET and coords is arrayutils._UNSET:
        return result
    if units is not arrayutils._UNSET:
        result = result.to_units(units)
    return as_xtensor(result, names=names, coords=coords)
```

The positional-unit branch is restricted to a backend **object**
(`_units.is_unit_like`, already `False` for plain strings and already gated
on `unit_backend="pint"`), never a string — a bare unit *string* stays
firmly torch's own device-string territory (`"cuda"`, `"cpu"`), which would
be genuinely ambiguous to reinterpret positionally. `units=` the *keyword*
has no such ambiguity (torch's `.to()` has no keyword by that name today), so
it accepts a plain string too. With no backend active, `is_unit_like` is
always `False`, so `.to()` falls straight through to today's plain-torch
behaviour when `units=` isn't a string either — no new degradation-table
entry needed beyond what `.to_units()` already has.

An edge case worth flagging rather than silently deciding: `as_xtensor(x,
units=None)` is a valid call that **clears** the unit (any `_UNSET`-eligible
override accepts `None` as "set it to nothing"), but `.to(units=None)`
inherits `to_units`'s convert semantics, and `to_units(None)` isn't a
sensible conversion target — this should raise a clear error at
implementation time rather than being worked out ad hoc.

The positional branch also requires `not kwargs` — `x.to(ureg.mm,
copy=True)` therefore does **not** take the unit-recognition path; it falls
straight through to `Tensor.to(self, ureg.mm, copy=True)`, which raises
`TypeError` from torch itself (the same error as today, just with an extra
argument torch also doesn't recognise). Combining a positional unit with
other `.to()` options isn't supported — only `units=` the keyword composes
with `names=`/`coords=`/dtype/device kwargs in the same call.

This also closes a sub-question issue #142 (closed) left open: whether
`to_unit(...)` should become `.to(...)` instead of a separately-named
method. The answer lands as **both**: `.to_units()` stays the explicit,
`.to_units`-shaped spelling, and `.to(units=...)` is now equally valid
sugar for the same conversion — not a replacement, matching this codebase's
existing habit of having more than one valid spelling for a common operation
where each serves a different calling style (compare `x.sum(dim="q")` vs.
`torch.sum(x, dim="q")`).

`Coordinate.to(unit)` and `_units.Unitful.to(unit)` (both pre-existing) are
worth citing too: `.to(unit)` already means "convert this unit" *within this
codebase*, not just by analogy to pint — so `.to()` gaining that meaning for
`XTensor`'s own data unit isn't a new idiom, just extending an existing one
to a new class.

**Not proposed here**: an `axes=` override (the axis-descriptor metadata from
#39a/#39b, distinct from `names`/`coords`). `as_xtensor` itself doesn't have
one either, so adding it to `.to()` alone would be asymmetric — better as its
own follow-up (add it to `as_xtensor` first; `.to()`'s delegation picks it up
for free) than solved ad hoc inside this proposal.

### 2.6 `.to_()` — a general in-place `.to()`

Raised in review, once in-place conversion (§2.4) and the extended `.to()`
(§2.5) were both on the table: can `.to()` have a general in-place
counterpart too, one that *forbids* an actual dtype/device change (raising if
the request would require one) while still allowing other metadata (units,
names, coords) to mutate?

That composes cleanly with what's already here — `.to_()` mirrors `.to()`'s
own parsing, but every metadata override goes through an already-in-place
path (`to_units_`, and the `names`/`coords` **property setters**, which
already mutate `self.__dict__` directly today — `x.names = (...)` is already
an in-place assignment, nothing new needed there):

```python
def to_(
    self,
    *args: tx.Any,
    units: tx.Any = arrayutils._UNSET,
    names: tx.Any = arrayutils._UNSET,
    coords: tx.Any = arrayutils._UNSET,
    **kwargs: tx.Any,
) -> tx.Self:
    """
    In-place `.to()`, with the same `units=`/`names=`/`coords=` overrides
    as the non-in-place form (§2.5), each going through an already
    in-place path -- but "in place" doesn't mean "always succeeds": a
    dtype/device change raises unless the request already matches
    `self`'s current dtype/device (a no-op), even if forced through
    `copy=True`; `units=` inherits `to_units_`'s own `ValueError`/
    `RuntimeError` restrictions (§2.4: no unit set, or a leaf tensor
    requiring grad); `names=`/`coords=` inherit their setters' own
    validation (a length mismatch on `names=`, an invalid coordinate
    spec on `coords=`).
    """
    if (
        args
        and _units.is_unit_like(args[0])
        and units is arrayutils._UNSET
        and len(args) == 1
        and not kwargs
    ):
        units, args = args[0], ()
    if _units.is_unit_like(units):
        _, units = _units.split_quantity(units)
    result = Tensor.to(self, *args, **kwargs)
    if result.dtype != self.dtype or result.device != self.device:
        raise ValueError(
            "to_: in-place .to() cannot change dtype or device "
            f"(would produce dtype={result.dtype}, device={result.device})"
        )
    if units is not arrayutils._UNSET:
        self.to_units_(units)
    if names is not arrayutils._UNSET:
        self.names = names
    if coords is not arrayutils._UNSET:
        self.coords = coords
    return self
```

Checking `result.dtype`/`result.device` against `self`'s (rather than, say,
`result is self`) is deliberate: torch's own `.to()` returns a genuinely new
tensor even for a no-op conversion when `copy=True` is explicitly passed, and
that shouldn't be treated as a rejected dtype/device change — only an actual
mismatch should raise.

Implementer's note: as written, `Tensor.to(self, *args, **kwargs)` always
runs first, so a dtype/device change that's going to be rejected still pays
for the (potentially real, allocating) conversion before the check discards
it and raises. That's fine for a proposal-level sketch — checking the
*parsed request* against `self.dtype`/`self.device` before calling
`Tensor.to()` at all would avoid the wasted allocation, left as an
implementation detail rather than a design decision.

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
q = ureg.Quantity(5000, "g*mm/s**2")
q.to_base_units()             # -> 0.005 kilogram * meter / second ** 2
q.to_reduced_units()          # -> 5000 gram * millimeter / second ** 2 (this case, no-op)
q.to_compact()                # -> 5.0 gram * meter / second ** 2
q.to_preferred([ureg.N])      # -> 0.005 newton  (only with an explicit or registry-default table)
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

**In-place counterparts**: raised in review — since `to_units_` (§2.4)
establishes the pattern, add `to_base_units_`/`to_reduced_units_`/
`to_compact_`/`to_preferred_` too, rather than deferring them as "add later."
Each computes its target unit string exactly like its non-in-place sibling,
then delegates to `to_units_` for the actual in-place rescale:

```python
def to_base_units_(self) -> tx.Self: ...
def to_reduced_units_(self) -> tx.Self: ...
def to_compact_(self) -> tx.Self: ...
def to_preferred_(
    self, preferred_units: tx.Optional[tx.List[str]] = None
) -> tx.Self: ...
```

Same `ValueError` contract as `to_units_()` (no backend, or no unit set), and
the same leaf/dtype-mismatch restrictions any in-place op already has.

Implementation note: since this is composition on top of `to_units`/
`to_units_`, it may land as a follow-up PR after the core §2.1–2.6 additions
rather than in the same one — the design is settled either way.

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
| `.unitless` | same fallback as `.dimensionless` (`not bool(self.units)`) — the two coincide exactly with no backend, since there's no dimensionality system to distinguish "no unit" from "a dimensionless unit" (§2.1) |
| `.is_compatible_with(unit)` | `_units.compatible()` already degrades to plain string equality without a backend (existing behaviour, unchanged) — no new failure mode needed |
| `.m_as(unit)` | inherits `to_units()`'s existing `ValueError` on `self.units is None` or no backend |
| `.m` / `.u` | no special case — plain aliases of `.magnitude`/`.units`, which already work with no backend (§2.3) |
| `.to(...)` | a backend-object positional/`units=` argument is already `False` under `is_unit_like` with no backend, so that branch never triggers; a plain `units=` string still goes through `to_units()`'s own `ValueError`; `names=`/`coords=` overrides need no backend at all (§2.5) |
| `.to_units_(unit)` | same `ValueError` contract as `.to_units()` (§2.4) |
| `.to_(...)` | same as `.to()` above for `units=`/positional-unit; the dtype/device and `names=`/`coords=` paths never need a backend at all (§2.6) |
| `.to_base_units()` / `.to_reduced_units()` / `.to_compact()` / `.to_preferred(...)` and their `_`-suffixed in-place counterparts | same `ValueError` contract as `.to_units()` (§2.7) |

No other existing 0003 member was found to have a gap during this audit — the
constructor, `.units` get/set, and `.to_units()` all already fail with a
`ValueError` naming the missing backend rather than an opaque `AttributeError`.

## 3. Heterogeneous per-axis units: how everything above behaves (no change here)

Raised in review: when a tensor has **per-axis (heterogeneous) units** (0003
§2.2/§4, structured-coordinate `unit` fields), what does `to_units()` actually
do — and does anything in this proposal need to change that? Explicitly
**out of scope**: folding a heterogeneous axis into one uniform unit is a
separate, harder design question (raised, then deliberately parked). This
section only **describes** the current, unchanged behavior, since every new
member above (`to_units_`, `.to()`/`.to_()`'s `units=`, the whole §2.7 family
and its in-place counterparts, `m_as`) is built directly on `to_units()`/
`.units`, so they all inherit exactly this behavior — nothing new to design.

**Verified directly:**

```python
# A fully heterogeneous tensor (no base unit at all -- V/A/W channels):
x = xtensor(data, names=("q", "t"), coords={"q": [
    {"name": "voltage", "unit": "V"}, {"name": "current", "unit": "A"},
    {"name": "power", "unit": "W"}]})
x.units            # None -- no base unit was ever set
x.to_units("mV")   # ValueError: to_units: this tensor has no unit to convert
x.to_units_("mV")  # same ValueError -- delegates straight to to_units's check
x.m_as("mV")       # same ValueError -- composes to_units + magnitude

# A tensor with BOTH a base unit and heterogeneous per-axis units:
y = xtensor(data, names=("q", "t"), units="m", coords={"q": [...same 3...]})
y.units            # "meter"
z = y.to_units("mm")
z.units            # "millimeter" -- base converted correctly
z.coords["q"]      # UNCHANGED -- still {"unit": "V"/"A"/"W"} per position
```

Neither case is a bug: `to_units()` (and everything above that funnels
through it) rescales only the **base** factor of `unit(x[i,j]) = base ·
Π(coord units)` (0003 §2.1) — a fully heterogeneous tensor has no base to
convert, so it raises immediately; a tensor with both converts the base
uniformly, correctly, leaving whatever per-position units ride alongside it
untouched. Folding a per-axis unit into the base (e.g. so a `mm`/`cm`/`m`
per-position axis becomes uniformly `mm`) is a real, separate feature with
its own design questions (what about an axis with *mixed* compatibility to
the target unit?) — deliberately not tackled by this proposal; every member
here composes on top of whatever `to_units()` does today, unchanged, and
picks up that future work automatically if `to_units()` itself ever grows it.

## 4. Resolving the `.magnitude` vs. complex-`abs` question

The issue raised: does `.magnitude` (0003's "drop the unit annotation" verb)
read confusingly against the mathematical sense of "magnitude of a complex
number" (its modulus, `|z|`)?

Checked directly against pint 0.25:

```python
import pint
u = pint.UnitRegistry()
q = u.Quantity(-3 + 4j, "V")
q.magnitude       # -> (-3+4j)
abs(q).magnitude  # -> 5.0
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
