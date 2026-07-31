---
icon: fontawesome/brands/python
---

# Data units

[`XTensor`][fiery.xtensor.XTensor]'s physical-unit annotation on a tensor's
*values* — assigning/reading `.units`, converting with `to_units`, and the
`pint.Quantity`-shaped simplification/conversion helpers. See also the
[Data units guide](../guide/data-units.md), and
[Proposal 0003](../proposals/0003-data-units.md) /
[Proposal 0006](../proposals/0006-quantity-api.md).

::: fiery.xtensor.XTensor
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - units
        - dimensionality
        - dimensionless
        - unitless
        - is_compatible_with
        - to_units
        - to_units_
        - magnitude
        - m_as
        - to_base_units
        - to_base_units_
        - to_reduced_units
        - to_reduced_units_
        - to_compact
        - to_compact_
        - to_preferred
        - to_preferred_
