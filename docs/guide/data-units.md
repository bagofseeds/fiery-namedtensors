# Data units

A tensor's **values** can carry a physical unit — the `.units` property (a
constructor `units=` kwarg or a settable attribute) — that rides through
operations the way names and coordinates do. By default a unit is an opaque
string; with a units backend it also gains validation, conversion, and
dimensional algebra:

```python
v = xtensor(data, names=("b", "t"), units="V")
v.units                # "V"
v.T.units              # "V"  — carried through reshaping/reduction/indexing
v.units = "mV"         # annotate: never changes the data
```

By default (`unit_backend=None`) a unit is an **opaque string** — stored and
carried, never inspected. Selecting a backend turns on validation, conversion,
and **dimensional algebra**:

```python
from fiery.xtensor import set_options
with set_options(unit_backend="pint"):     # needs fiery-xtensor[units]
    volts = xtensor(v, units="V")
    amps  = xtensor(a, units="A")
    secs  = xtensor(t, units="s")
    volts.to_units("mV")          # converts: rescales the data ×1000
    (volts * amps).units          # "ampere * volt"  — units multiply
    (volts / secs).units          # "volt / second"
    (volts ** 2).units            # "volt ** 2"
    (volts @ amps).units          # "ampere * volt"  — matmul multiplies too
```

You can also **attach** a unit by multiplying with the backend's own unit
objects, the way pint builds a `Quantity` from `5 * ureg.metre`:

```python
import pint
u = pint.UnitRegistry()
with set_options(unit_backend="pint"):
    (x * u.mm).units             # "millimeter"          — bare unit; data unchanged
    (x * (3 * u.mm)).units       # "millimeter", data ×3 — a quantity scales too
    (v / u.s).units              # "volt / second"
```

(Write the unit on the **right** — `x * u.mm`, not `u.mm * x` — so the
`XTensor` handles it before pint's own reflected operator does.)

Whenever a step is dimensionally invalid or ambiguous — adding incompatible
units, or a transcendental like `exp`/`log` of a united value — the result
silently **drops** the unit; `set_options(unit_policy="strict")` makes those
same steps **raise** instead:

```python
(volts + amps).units              # None    — incompatible, dropped
torch.exp(volts).units            # None    — exp needs a dimensionless argument
with set_options(unit_policy="strict"):
    volts + amps                  # ValueError: incompatible units 'volt' and 'ampere'
```

**Compatible** units are reconciled automatically: adding or comparing `V` and
`mV` converts the right operand to the left's unit first (only *incompatible*
dimensions drop/raise). And `.magnitude` drops the unit when you want the bare
values:

```python
with set_options(unit_backend="pint"):
    (xtensor(v, units="V") + xtensor(mv, units="mV")).units  # "volt" — mV converted
    xtensor(v, units="V").magnitude.units                    # None   — unit dropped, still an XTensor
```

## Asking about a unit

Under a backend you can ask what a unit *is*, and whether a conversion would
work, without converting anything:

```python
with set_options(unit_backend="pint"):
    power = xtensor(data, units="W")
    power.dimensionality             # "[mass] * [length] ** 2 / [time] ** 3"
    power.dimensionless              # False
    power.is_compatible_with("kW")   # True   — so to_units("kW") would work
    power.is_compatible_with("V")    # False

    angle = xtensor(data, units="rad")
    angle.dimensionless              # True   — an angle has no dimensions
    angle.unitless                   # False  — but it still names a unit
```

`.u` and `.m` are short aliases for `.units` and `.magnitude`, and `.m_as`
converts and drops the annotation in one step:

```python
with set_options(unit_backend="pint"):
    volts = xtensor(data, units="V")
    volts.u                # "volt"
    volts.m.units          # None
    volts.m_as("mV")       # values ×1000, no unit
```

## More ways to convert

Every conversion has an **in-place** twin, spelled with a trailing underscore
(the same convention as `rename_`): it rescales the data and updates the
annotation on the tensor itself instead of returning a new one.

```python
with set_options(unit_backend="pint"):
    x = xtensor(data, units="V")
    x.to_units_("mV")      # x is now in millivolts
```

`.to()` converts too, alongside its usual dtype/device job — by keyword, or
straight from one of the backend's own unit objects:

=== "By keyword"

    ```python
    x.to(units="mm")
    ```

=== "From a backend unit"

    ```python
    x.to(u.mm)
    ```

=== "Alongside dtype and names"

    ```python
    x.to(torch.float64, units="mm", names=("b", "t"))
    ```

`units=` **converts** (so a unit must already be set) — annotating is still
`x.units = ...`. `names=`/`coords=` replace that metadata wholesale, exactly
as they do for `as_xtensor`. Write the unit as a keyword to pass it as a
plain string: a positional string is torch's own device spelling (`"cuda"`),
so only a backend unit object is recognised there.

`.to_()` does all of the same in place, and refuses a real dtype or device
change (moving the data is not an in-place operation):

```python
with set_options(unit_backend="pint"):
    y = xtensor(data, names=("b", "t"), units="m")
    y.to_(units="mm", names=("b", "time"))   # fine
    y.to_(torch.float64)                     # ValueError
```

When you would rather not name the target unit at all, let the backend pick
one:

```python
with set_options(unit_backend="pint"):
    force = xtensor(torch.tensor([5000.0]), units="g*mm/s**2")
    force.to_base_units()        # 0.005 kilogram * meter / second ** 2
    force.to_reduced_units()     # 5000 gram * millimeter / second ** 2
    force.to_compact()           # 5.0 gram * meter / second ** 2
    force.to_preferred(["N"])    # 0.005 newton
```

`to_compact` reads the values as well as the unit — it picks the prefix that
keeps them near 1 — and `to_preferred` needs a list of units to choose from
(or a default configured on the backend registry). All four have an in-place
twin as well (`to_base_units_`, `to_compact_`, …).

## Heterogeneous (per-axis) units

Units may also **vary along an axis**: give a structured coordinate
([Proposal 0002](../proposals/0002-structured-coordinates.md)) a `unit` field
per position, and each position stacks a different quantity (a
`voltage`/`current`/`power` channel stack). The effective element unit is
`base · Π(coord units)`:

```python
with set_options(unit_backend="pint"):
    x = xtensor(data, names=("q", "t"), coords={"q": [
        {"name": "voltage", "unit": "V"},
        {"name": "current", "unit": "A"},
        {"name": "power",   "unit": "W"},
    ]})
    x.units                     # None    — no single base unit (heterogeneous)
    x.sel(q="voltage").units    # "V"     — selecting one position folds its unit in
    x.sum(dim="q").units        # None    — V/A/W incompatible -> dropped

    # a *uniform* axis (every position in the same unit) folds cleanly instead:
    uniform = xtensor(data, names=("q", "t"), coords={"q": [
        {"name": "vx", "unit": "V"},
        {"name": "vy", "unit": "V"},
        {"name": "vz", "unit": "V"},
    ]})
    uniform.sum(dim="q").units  # "V"     — the shared unit folds into the base
```

Reducing over such an axis folds a uniform unit into the base or, on
incompatible per-position units, drops it (or raises under
`unit_policy="strict"`).

The data itself is never wrapped in a `pint.Quantity`, so autograd, GPU, and
`__torch_function__` keep working throughout.
