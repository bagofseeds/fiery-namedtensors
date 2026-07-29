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
