---
icon: fontawesome/brands/python
---

# API reference

`fiery.xtensor` makes names a first-class citizen of `torch.Tensor`, in the
spirit of [xarray](https://docs.xarray.dev). Most of its public surface lives
on one class, [`XTensor`][fiery.xtensor.XTensor] (also available lowercase as
`xtensor`) — an xarray-like `DataArray` over a live `torch.Tensor`, carrying
self-managed named dimensions and, optionally, per-dimension coordinate
labels. The reference is split by what each group of methods/functions does,
rather than one page for the whole package:

- **[Core](core.md)** — `XTensor` itself: names, axis descriptors,
  coordinates, renaming, and dtype/device conversion; plus
  `ExtendedTensor`, the generic name-aware tensor-subclass base it's built on.
- **[Factories](factories.md)** — build an `XTensor` directly (`xzeros`,
  `xones`, `xarange`, ...), or coerce something else into one (`as_xtensor`,
  `is_xtensor`).
- **[Reductions](reductions.md)** — sums, means, extrema, sorts, and their
  `nan`-aware / cumulative variants.
- **[Combining](combining.md)** — concatenation/stacking and the
  matrix-product family.
- **[Pointwise operations](pointwise.md)** — elementwise arithmetic,
  comparison, logical, and trigonometric operations.
- **[Gather, scatter & indexing](indexing.md)** — `sel`/`isel`/`interp`,
  PyTorch's gather/scatter family, and slicing/splitting.
- **[Shape](shape.md)** — reshaping, (un)squeezing, transposing/permuting,
  and broadcasting.
- **[Data units](units.md)** — the physical unit of a tensor's *values*
  (see also the [Data units guide](../guide/data-units.md)).
- **[Options](options.md)** — library-wide behaviour switches.

Every method lives on exactly one of these pages (`XTensor` itself is
introduced on [Core](core.md) and referenced from the others).
