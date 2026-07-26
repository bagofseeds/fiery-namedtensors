# Axis descriptors

A dimension name can be enriched into an OME-NGFF-style **descriptor** — a dict
carrying a `type`, `unit`, and `orientation` — and those fields follow the
dimension the way coordinates do, merge across name-aware ops, and can address
whole groups of axes at once.

A name can be enriched into an [OME-NGFF](https://ngff.openmicroscopy.org)-style
**descriptor** — a dict with a required `name` plus optional `type`, `unit`,
`orientation` (and a `coord`/`labels`, see [Coordinates](coordinates.md)) — by
passing it through **`axes=`**. (`names=` takes bare strings only; descriptors
go through `axes=`.)

```python
x = xtensor(
    torch.zeros(2, 3, 4),
    axes=[
        {"name": "b", "type": "batch"},
        "h",
        {"name": "w", "type": "space", "orientation": "left-to-right"},
    ],
)
x.names          # ('b', 'h', 'w')          — the bare, ergonomic view
x.axes           # full descriptors, one dict per axis
x.flip("w").axes[2]["orientation"]   # 'right-to-left'  — flip reverses it
```

`axes=` is the general per-axis container; `names=` (bare strings) and
`coords=` are shortcuts into it.

Descriptor fields are keyed by dimension name, so — like coordinates — they
follow their dimension through `permute`, reductions, `rename`, etc. An
`orientation` must read `"{a}-to-{b}"`; flipping the axis rewrites it to
`"{b}-to-{a}"`.

When two operands meet in a name-aware op (broadcast, alignment), their
descriptors are **merged by dim name** the same way coordinates are: the result
is the union of the axes, and for a shared dim the fields the operands **agree**
on are kept while **conflicting** fields are dropped. That policy is
configurable via `set_options(combine_axes=...)`, usable globally or as a
context manager:

```python
from fiery.xtensor import set_options

a = xtensor(torch.ones(3), axes=[{"name": "x", "type": "space"}])
b = xtensor(torch.ones(3), axes=[{"name": "x", "type": "time"}])
(a + b).axes                      # ({'name': 'x'},)  — the clash drops 'type'

with set_options(combine_axes="strict"):
    a + b                         # raises ValueError: conflicting 'type' …
```

The policy is one of `"drop_conflicts"` (default), `"strict"` (alias
`"raise"`), `"override"` (keep the left operand's value) or `"drop"` (always
drop the field). Pass a `{field: policy}` dict to set it **per descriptor
field** — `"*"` is the default for fields you don't name:

```python
# drop everything by default, but a clashing unit is an error
with set_options(combine_axes={"*": "drop", "unit": "raise"}):
    ...
```

`combine_axes` accepts `"drop_conflicts"` (default), `"strict"` (raise on any
clash), `"override"` (keep the left operand's fields), or `"drop"` (discard all
descriptors).

A descriptor is also a way to **address** axes. Anywhere you can pass a `dim`
(method form), a query dict selects *every* axis whose descriptor matches — so
one call can act on a whole group of axes at once:

```python
x.movedim({"type": "space"}, -1)   # all space axes to the back, order kept
x.sum(dim={"type": "channel"})     # reduce every channel axis
```

A query that matches a single axis behaves exactly like naming it (so it still
works with single-`dim`-only ops such as `prod`).
