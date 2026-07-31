---
icon: fontawesome/brands/python
---

# Core

The data model: [`XTensor`][fiery.xtensor.XTensor] itself (dimension names,
axis descriptors, coordinates, renaming, and dtype/device conversion), and
[`ExtendedTensor`][fiery.xtensor.ExtendedTensor], the generic name-aware
`torch.Tensor` subclass base it's built on.

Everything else `XTensor` supports — reductions, combining, gather/scatter/
indexing, shape, data units — has its own page; see the [API overview](index.md).

::: fiery.xtensor.XTensor
    options:
      members:
        - names
        - axes
        - coords
        - rename
        - rename_
        - refine_names
        - swap_dims
        - swap_dims_
        - align_to
        - align_as
        - to
        - to_

::: fiery.xtensor.ExtendedTensor
