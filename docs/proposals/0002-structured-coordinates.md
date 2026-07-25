# Proposal 0002 — Structured coordinates

| | |
| --- | --- |
| **Status** | Accepted — implementing |
| **Author** | (proposed) |
| **Created** | 2026-07-25 |
| **Tracking** | part of [#3](https://github.com/bagofseeds/fiery-xtensor/issues/3) |

## Abstract

Today a coordinate label is a bare `str` (or `None`). This proposal lets a label
also be a **structured** value — a dict carrying fields — so a position along an
axis can be described (`{"name": "GFP", "type": "signal"}`), and so positions can
be **selected by a field query** (`x[:, {"type": "signal"}]`). It is the coords
counterpart of the axis **descriptors** from #39: descriptors describe the *whole
axis*, structured coordinates describe *each position along it*.

## Motivation

The immediate driver is label indexing by attribute: `x[..., {"type": "space"}]`
should pick the positions whose coordinate matches the query — resolving to a
`slice` when they are contiguous, or an index list when they are not. That has no
meaning while a coordinate is a plain string: there is nothing on a position to
match `{"type": ...}` against. The canonical use case is an OME-NGFF-style
channel axis:

```python
img = xtensor(data, names=("c", "y", "x"), coords={"c": [
    {"name": "DAPI", "type": "nucleus"},
    {"name": "GFP",  "type": "signal"},
    {"name": "RFP",  "type": "signal"},
]})
img[{"type": "signal"}]        # the GFP+RFP sub-stack (a slice here)
img.sel(c="GFP")               # still selects by name
```

## Data model

A coordinate label becomes `str | None | dict` (a **structured label**). Storage
is unchanged: `_coords[dim]` stays a tuple of labels, one per position; some may
now be dicts. The propagation machinery (`_slice_labels`, the length-guarding
`coords` getter, `cat`'s label concatenation) treats a label as an **opaque
value**, so structured labels ride through slicing/reordering/concatenation with
**no change**.

A structured label's **identity** is its `"name"` field (mirroring axis
descriptors): `_label_name(label)` returns `label["name"]` for a dict, the string
itself for a `str`, and `None` otherwise. Identity is what name-based selection
matches on; a structured label without a `"name"` is still queryable but not
name-selectable.

## Selection semantics

Two selection modes, unified across `.sel`, attribute access, and positional
`[]`:

1. **By name** — `x.sel(c="GFP")`, `x.GFP`, `x[:, "GFP"]` — matches the position
   whose **identity** equals the string. A single match **drops** the axis (as
   an integer index does). Unchanged for plain-string coordinates.
2. **By query** *(new)* — a **dict** `{"type": "signal"}` matches **every**
   position whose structured label contains those key/value pairs. A query is a
   filter, so it always **keeps** the axis, and resolves to:
   - a **`slice`** when the matches are contiguous (cheap, stays a basic index);
   - an **index list** otherwise (advanced index).
   A query that matches nothing yields a size-0 axis (no error), like an empty
   inner join.

Everywhere a positional label is accepted (`_is_label_index`), a dict is now a
**query index**: `x[:, {"type": "signal"}]`, composing with ints/slices/`...`
exactly like the string form (#55). `.sel(c={"type": "signal"})` is the keyword
spelling of the same thing.

## Interaction with axis descriptors (#39)

Orthogonal, and deliberately parallel in spelling:

| | describes | lives in | query spelling |
| --- | --- | --- | --- |
| axis **descriptor** | the whole axis | `_axis_meta` | `x.sum(dim={"type": "space"})` (selects **axes**) |
| structured **coordinate** | one position | `_coords` | `x[:, {"type": "signal"}]` (selects **positions**) |

A `{"type": ...}` dict selects *axes* when it lands in a `dim=`/`movedim`
position (descriptor query, #39b) and *positions* when it lands in a `[]`/`sel`
slot (this proposal). The two never collide because they occupy different slots.

## Specification

- **Storage / constructor** — the `coords` setter already accepts any per-
  position value; structured labels need no storage change. `...` fill and the
  length check are unchanged.
- **Helpers** — `_label_name(label)` (identity); `_label_matches(label, query)`
  (all key/values present); `_match_positions(labels, query)` (matching indices);
  `_positions_to_index(positions)` (contiguous → `slice`, else list).
- **`sel`** — a `str`/list value matches by identity; a `dict` value queries
  (keeps the axis).
- **`_label_to_index`** (positional `[]`) — `str` → identity → int; list of
  `str` → list of ints; `dict` → query → `slice`/list.
- **`_is_label_index`** — now also `True` for a `dict`; `_resolve_label_slicer`
  counts a query element as one axis.
- **`_dims_with_label`** (attribute access) — matches by identity.
- **Alignment** — `_align_by_name` compares labels by value; switched from a
  hashed set to list membership so **unhashable** dict labels align too.

## Backwards compatibility

Fully compatible. Plain-string coordinates behave exactly as before: identity of
a `str` is itself, so `sel`/attribute/positional-string are unchanged, and no
existing call passes a dict where a label is expected. The only new behaviour is
that a dict is now accepted as a coordinate value and as a query slicer.

## Open questions

1. **Require `"name"` on a structured label?** Proposed: **no** — optional, but
   needed for name-selection. (Consistent with descriptors, which *do* require
   `name`; coordinates are looser because a position may be purely descriptive.)
2. **Zero-match query:** size-0 axis (proposed) vs raise. Size-0 is consistent
   with an empty inner join and with boolean-mask indexing.
3. **`.sel` query dropping vs keeping:** a query always keeps the axis (it is a
   filter); a *single-name* select still drops. Should a name given as a
   1-element list keep it? Yes — matches today's `sel(c=["GFP"])` behaviour.
4. **Nested / typed queries** (ranges, predicates) are out of scope; equality on
   provided keys only.

## References

- OME-NGFF axes / `omero:channels` — <https://ngff.openmicroscopy.org/latest/>
- Proposal 0001 (units) — the other structured-metadata layer
- Axis descriptors — #39 (the axis-level analogue)
