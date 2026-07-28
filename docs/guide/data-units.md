# Data units

A tensor's **values** can carry a physical unit — the `.unit` property (a
constructor `unit=` kwarg or a settable attribute) — that rides through
operations the way names and coordinates do. By default a unit is an opaque
string; with a units backend it also gains validation, conversion, and
dimensional algebra:

```python
v = xtensor(data, names=("b", "t"), unit="V")
v.unit                 # "V"
v.T.unit               # "V"  — carried through reshaping/reduction/indexing
v.unit = "mV"          # annotate: never changes the data
```

By default (`unit_backend=None`) a unit is an **opaque string** — stored and
carried, never inspected. Selecting a backend turns on validation, conversion,
and **dimensional algebra**:

```python
from fiery.xtensor import set_options
with set_options(unit_backend="pint"):     # needs fiery-xtensor[units]
    volts = xtensor(v, unit="V")
    amps  = xtensor(a, unit="A")
    volts.to_unit("mV")            # converts: rescales the data ×1000
    (volts * amps).unit           # "ampere * volt"  — units multiply
    (volts / secs).unit           # "volt / second"
    (volts ** 2).unit             # "volt ** 2"
    (volts @ amps).unit           # "ampere * volt"  — matmul multiplies too
```

You can also **attach** a unit by multiplying with the backend's own unit
objects, the way pint builds a `Quantity` from `5 * ureg.metre`:

```python
import pint
u = pint.UnitRegistry()
with set_options(unit_backend="pint"):
    (x * u.mm).unit              # "millimeter"          — bare unit; data unchanged
    (x * (3 * u.mm)).unit        # "millimeter", data ×3 — a quantity scales too
    (v / u.s).unit               # "volt / second"
```

(Write the unit on the **right** — `x * u.mm`, not `u.mm * x` — so the
`XTensor` handles it before pint's own reflected operator does.)

Whenever a step is dimensionally invalid or ambiguous — adding incompatible
units, or a transcendental like `exp`/`log` of a united value — the result
silently **drops** the unit; `set_options(unit_policy="strict")` makes those
same steps **raise** instead:

```python
(volts + amps).unit               # None    — incompatible, dropped
torch.exp(volts).unit             # None    — exp needs a dimensionless argument
with set_options(unit_policy="strict"):
    volts + amps                  # ValueError: incompatible units 'volt' and 'ampere'
```

**Compatible** units are reconciled automatically: adding or comparing `V` and
`mV` converts the right operand to the left's unit first (only *incompatible*
dimensions drop/raise). And `.magnitude` drops the unit when you want the bare
values:

```python
with set_options(unit_backend="pint"):
    (xtensor(v, unit="V") + xtensor(mv, unit="mV")).unit   # "volt" — mV converted
    xtensor(v, unit="V").magnitude.unit                    # None   — unit dropped, still an XTensor
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
    x.unit                     # None    — no single base unit (heterogeneous)
    x.sel(q="voltage").unit    # "V"     — selecting one position folds its unit in
    x.sum(dim="q").unit        # None    — V/A/W incompatible -> dropped
    # a *uniform* axis (all positions same unit) folds cleanly on reduction:
    y.sum(dim="q").unit        # "V"
```

Reducing over such an axis folds a uniform unit into the base or, on
incompatible per-position units, drops it (or raises under
`unit_policy="strict"`).

The data itself is never wrapped in a `pint.Quantity`, so autograd, GPU, and
`__torch_function__` keep working throughout.
