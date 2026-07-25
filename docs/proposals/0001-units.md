# Proposal 0001 — Axis units

| | |
| --- | --- |
| **Status** | Draft — under discussion (implementation not started) |
| **Author** | (proposed) |
| **Created** | 2026-07-25 |
| **Tracking** | part of [#3](https://github.com/bagofseeds/fiery-xtensor/issues/3); supersedes the units sketch in [#39](https://github.com/bagofseeds/fiery-xtensor/issues/39) / [#48](https://github.com/bagofseeds/fiery-xtensor/issues/48) |

## Abstract

This proposal asks what — if anything — the axis-descriptor `unit` field (from
#39) should *mean*. It is the **axis-name** unit, one of three distinct places a
unit can attach (see *Scope*); the other two are separate proposals. **Nothing
here is decided** — it is written to frame the discussion.

## Scope — the three unit loci

"Unit" attaches in three different places, each with its own owner and
conversion story. This proposal is **only about the first**:

| # | Unit on… | Example | Proposal |
| --- | --- | --- | --- |
| **1** | **axis name** (descriptor) | `{"name": "x", "unit": "um"}` | **this one** |
| 2 | **coordinate labels** | `coords={"t": [{"name": "t0", "value": 0.0, "unit": "s"}, …]}` | 0003 |
| 3 | **tensor data** | `xtensor(volts, …)` (dimensional analysis) | later |

Why keep them apart: an **axis** unit is metadata on the dimension (there are no
values to convert unless the axis also carries numeric coordinates — that's
locus 2); a **data** unit drives dimensional analysis through arithmetic (locus
3). Bundling them forced the hard locus-3 questions onto the small locus-1 one.

## What an axis unit is (today)

The `unit` field on an axis descriptor is meant to state the physical unit of
that **dimension** — an `x` axis in micrometres, a `t` axis in seconds. Today it
is stored and carried like any other descriptor field (`_axis_meta`, keyed by
dim name; merged across operands by `_merge_axis_meta`), but **never inspected**
— it is an opaque string. It is *not* transformed by tensor arithmetic
(multiplying two images does not make the `x` axis µm²).

```python
x = xtensor(data, names=[{"name": "x", "type": "space", "unit": "um"}, "y"])
x.axes[0]          # {'name': 'x', 'type': 'space', 'unit': 'um'}  (works now)
x.T.axes[-1]       # unchanged — the unit follows its dim
```

So the axis unit already *rides along*. The only thing a proposal could add is
**validation** (and maybe normalisation). That is the whole question.

## The question to settle

**Should an axis `unit` be validated/normalised, and if so, how?** The options,
smallest to largest commitment:

1. **Leave it free-form** — a `unit` is documentation, never checked. Zero deps,
   zero risk, but `"metre"`, `"metres"`, `"m"`, and `"meter"` are all "valid"
   and mutually unequal, and a typo (`"metr"`) is silent.
2. **Validate when a unit library is present** — parse the string through
   `pint` (an optional `[units]` extra) and raise on failure; stay free-form
   when the extra is absent. Catches typos where the extra is installed; keeps
   the zero-dep default and the 3.7 floor. *(Sketched below.)*
3. **Validate *and* normalise** — canonicalise to a single spelling
   (`"micron"` → `"micrometer"`), so equality (and the `combine_axes` merge)
   compares units meaningfully. More surface, and it rewrites the author's
   spelling.

Sub-questions that hang off the choice:

- **Library** — `pint` (domain-neutral de-facto standard) vs `astropy.units`
  (heavier) vs a bespoke string-unit vs none.
- **Optional vs hard dep** — an optional extra preserves the zero-dep default
  and wide-Python floor; a hard dep simplifies the code but raises the floor
  (recent pint needs Python ≥ 3.8/3.9).
- **Never wrap the data** — whatever the choice, a `pint.Quantity` around the
  tensor is off the table: it would forfeit autograd/GPU/dispatch (the reason
  this package is native over torch). A library would parse the unit *symbol*
  only.
- **Unit equality under merge** — with option 3, should `_merge_axis_meta`
  compare *parsed* units so `"um"` and `"micrometer"` agree instead of
  conflicting as strings?

## Sketch (if option 2 is chosen)

A `_units` module with a lazily-built, cached registry: `units_available()` and
`validate_unit(unit)` (parse via pint when present, raise `ValueError` on
failure; a no-op otherwise). The `names`/`axes` setter calls it alongside
`_validate_orientation`. `pyproject` grows a `units = ["pint>=0.18"]` extra;
CI installs pint where the interpreter supports it so the path is exercised,
while the 3.7 job keeps units opaque. *(Prototyped, then set aside pending this
discussion.)*

## Backwards compatibility

Any option is compatible with today's stored `unit` strings. Option 2/3 add one
observable change **only where the extra is installed**: an unparsable unit
would newly raise at construction.

## Open questions

1. Validate at all, or stay free-form? (The core decision above.)
2. If validating: which library, optional vs hard dep, normalise or not?
3. Should axis units even exist independently of coordinate values (locus 2), or
   is an axis unit only meaningful once the axis has numeric coordinates?
4. Unit equality in the descriptor merge (string vs parsed).

## References

- OME-NGFF axes — <https://ngff.openmicroscopy.org/latest/#axes-md>
- pint — <https://pint.readthedocs.io>
- Proposal 0002 (structured coordinates) — the substrate for locus 2
