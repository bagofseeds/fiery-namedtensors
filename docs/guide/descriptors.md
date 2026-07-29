# Axis descriptors

A dimension name can be enriched into a **descriptor** — a dict carrying any
extra fields you like — and those fields follow the dimension the way
coordinates do, merge across name-aware ops, and can address whole groups of
axes at once.

A name is enriched by passing a dict through **`axes=`** (`names=` takes bare
strings only). The dict needs a `name`; every other key is up to you:

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

## Which keys are special

A descriptor is an **open** dict — attach whatever metadata suits your data.
Only a few keys are reserved by the library:

- **`name`** *(required)* — becomes the dimension's name.
- **`coord`** / **`labels`** — the dimension's coordinate (see
  [Coordinates](coordinates.md)); handed straight to `coords`.
- **`orientation`** — the one otherwise-free field with built-in behaviour: it
  must read `"{a}-to-{b}"`, and flipping the axis rewrites it to `"{b}-to-{a}"`.

**Everything else is free-form.** The `type` in the examples is just the
[OME-NGFF](https://ngff.openmicroscopy.org) convention — it is **not** a
first-class key, and nothing in the library privileges it. Use
`{"name": "c", "modality": "MRI", "role": "readout"}` or whatever keys your
pipeline cares about; they are stored, carried, merged, and queryable exactly
the way `type` is.

`axes=` is the general per-axis container; `names=` (bare strings) and
`coords=` are shortcuts into it. Descriptor fields are keyed by dimension name,
so — like coordinates — they follow their dimension through `permute`,
reductions, `rename`, etc.

!!! note "Physical units are not an axis descriptor"
    There is no blessed axis `unit`. An axis's physical unit is carried by its
    coordinate values (a **position unit**, see [Coordinates](coordinates.md)),
    while a tensor's values carry a **data unit** (see
    [Data units](data-units.md)). A `unit` key placed in a descriptor is just
    inert free-form metadata — it gets no unit semantics.

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
`"raise"`), `"override"` (keep the left operand's value), or `"drop"` (always
drop the field). Pass a `{field: policy}` dict to set it **per descriptor
field** — `"*"` is the default for fields you don't name:

```python
# drop everything by default, but a clashing type is an error
with set_options(combine_axes={"*": "drop", "type": "raise"}):
    ...
```

A descriptor is also a way to **address** axes. Anywhere you can pass a `dim`
(method form), a query dict selects *every* axis whose descriptor matches — so
one call can act on a whole group of axes at once:

```python
x.movedim({"type": "space"}, -1)   # all space axes to the back, order kept
x.sum(dim={"type": "channel"})     # reduce every channel axis
```

A query that matches a single axis behaves exactly like naming it (so it still
works with single-`dim`-only ops such as `prod`).
