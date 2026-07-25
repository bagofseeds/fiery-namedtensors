# Proposal 0001 — Axis units

| | |
| --- | --- |
| **Status** | Draft — converging (design below reflects discussion) |
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

So the axis unit already *rides along*. The only thing worth adding for an axis
unit is **normalisation and normalised equality** — there is nothing to
*convert* (a bare dimension has no values). This is deliberately modest; the
interesting unit work is locus 2 (coordinate units), where values exist.

## Design — an explicit `unit backend`

Behaviour must **not** depend on whether pint happens to be importable; it is
opt-in through a package **option**, so it is deterministic and reproducible.

- A new option, **`unit_backend`**, **default `None`** — today's behaviour: a
  `unit` is a free-form string, never inspected, string-compared.
- Set it explicitly to opt into unit semantics — reusing the existing
  `set_options` (which is *both* a permanent setter and a context manager, so no
  new machinery):

  ```python
  from fiery.xtensor import set_options

  set_options(unit_backend="pint")                 # for the session
  with set_options(unit_backend="pint"):           # for a block
      ...
  ```

- A backend is a small interface — `normalise(unit) -> str` (canonical form;
  raising on an unparsable unit) and, from it, `equal(a, b)` (normalised
  equality). Built-ins: `None` (identity + string equality) and `"pint"`;
  the seam leaves room for `"astropy"` or a custom backend later.
- Setting `unit_backend="pint"` when pint is not installed **raises at set
  time** — a clear, immediate error rather than silent degradation.
- **Never wraps the data.** The backend touches unit *symbols* only; the tensor
  stays a plain `torch.Tensor` (autograd/GPU/dispatch intact).

### What a backend changes

With a non-`None` backend:

1. a `unit` descriptor field is **normalised** when set (so `.axes` shows a
   canonical spelling, and typos raise);
2. `_merge_axis_meta` compares units by **normalised equality**, so `"um"` and
   `"micrometer"` agree instead of conflicting as strings.

No conversion, no arithmetic — that is all an *axis* unit can meaningfully do.

## Backwards compatibility

Fully compatible and **behaviour is unchanged by default** (`unit_backend` is
`None`): a `unit` stays a free-form string. Semantics appear only when a backend
is explicitly selected.

## Open questions

1. **Normalise-on-store vs on-compare** — rewrite the stored `unit` to the
   canonical spelling (so `.axes` is canonical, but the author's `"um"` becomes
   `"micrometer"`), or keep the original and normalise only for equality?
   *(Leaning: normalise on store — one source of truth.)*
2. **Backend names / packaging** — ship `"pint"` as an optional `[units]` extra
   (import lazily, error if selected-but-absent); reserve room for other
   backends.
3. This is the small half. **Locus 2 (coordinate units) is the priority** —
   see Proposal 0003 — because that is where conversion actually applies.

## References

- OME-NGFF axes — <https://ngff.openmicroscopy.org/latest/#axes-md>
- pint — <https://pint.readthedocs.io>
- Proposal 0002 (structured coordinates) — the substrate for locus 2
