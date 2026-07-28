# Differences from xarray

`fiery.xtensor` is an [xarray](https://docs.xarray.dev)-like library, and it
follows xarray's conventions wherever it can: named **dimensions**, coordinate
**labels** keyed by dimension name, selection with `.sel` / `.isel`,
broadcasting and alignment **by name**, and a `set_options` context manager.
If you know xarray, `XTensor` should feel familiar.

The one structural difference colours everything else: an `XTensor` **is a
`torch.Tensor`** (a subclass), not a container that wraps an array. Autograd,
device, dtype, and `__torch_function__` all keep working, and every `torch.*`
op accepts an `XTensor`. xarray, by contrast, wraps a NumPy/Dask array and
exposes a curated API over it.

This page lists where the two diverge, in both directions.

## Added — things xarray does not have

- **A real tensor.** Gradients flow through named ops; `.cuda()`, `.double()`,
  `torch.stack`, `@`, and friends all work, because an `XTensor` *is* a tensor.
- **Numeric coordinates with an affine.** A dimension coordinate can be stored
  **compactly** as `spacing`/`origin` (materialised on demand), so a regular
  grid costs O(1) and its spacing can be a **learnable** tensor that gradients
  flow back to ([Proposal 0001](../proposals/0001-units.md)).
- **Two kinds of unit.** Beyond coordinate **position** units, a tensor's
  **values** carry a physical **data unit** with dimensional algebra and
  conversion, via an optional pint backend
  ([Proposal 0003](../proposals/0003-data-units.md)). xarray leaves units to
  the separate `pint-xarray` project.
- **Axis descriptors.** A name can be enriched with OME-NGFF-style `type`,
  `unit`, and `orientation` fields, and ops can resolve a `dim` by *type*
  ([Axis descriptors](descriptors.md)).
- **Unnamed dimensions.** A torch tensor often has anonymous axes (a batch dim,
  say); `XTensor` allows a dim to be unnamed (`None`). xarray requires every
  dimension to be named.
- **Attribute access to a label.** `x.red` reaches the position labelled
  `"red"` when that label is unambiguous — sugar over `.sel`.
- **Ergonomic constructors.** `xvector` / `xmatrix` and the `x*` factories
  (`xzeros`, `xones`, `xfill`, …) name and label axes at creation time.

## Missing — xarray features not (yet) here

- **No `Dataset`.** `XTensor` is the `DataArray` analogue only; there is no
  multi-variable container.
- **No split-apply-combine.** `groupby`, `resample`, `rolling`, `weighted`,
  and the rest of that suite are not implemented.
- **No IO / plotting / pandas bridge.** No NetCDF/Zarr readers, no `.plot`, no
  `.to_pandas()` / `MultiIndex`.
- **Curvilinear (arbitrary multi-dim) coordinates aren't implemented yet.** A
  dim can already carry several coordinates — a **non-dimension** coordinate
  rides along it, and a compact **affine** coordinate can span several dims at
  once ([Proposal 0005](../proposals/0005-multiple-coordinates.md)) — but a
  general `lat(y, x)`-style array of arbitrary values over multiple dims is
  not yet landed.
- **Alignment is inner-join only.** There is no `join="outer"/"left"/"right"`
  (which would need NaN fill); operands align to the **intersection** of their
  labels. (xarray's *default* `arithmetic_join` is also `"inner"`, so the
  common case matches.)

## Behaviour & vocabulary differences

- **`names`, not `dims`.** The dimension names live on `.names` (xarray:
  `.dims`); the values are a `torch.Tensor`, not a NumPy array.
- **Unnamed axes are allowed.** xarray has no unnamed dims; xtensor does (a
  plain tensor's anonymous batch axis). A pointwise op still aligns **by name**
  when the unnamed axes are all **leading** — the named suffix aligns by name,
  the anonymous prefix broadcasts positionally — and when both operands share
  the same `names`. An all-unnamed operand, a plain tensor, or a scalar
  broadcasts positionally; a `None` *after* a named axis on differently-named
  operands raises (ambiguous). See [broadcasting](broadcasting.md).
- **Selection vs interpolation are separate verbs.** Like xarray, `.sel` picks
  an existing position (exact, or snapped by `mode`/`method` with a
  `tolerance`) and `.interp` computes values off the grid, on both regular
  and irregular numeric coordinates
  ([Proposal 0004](../proposals/0004-numeric-selection.md)).
- **Name-as-`dim` is method-only.** `x.sum(dim="height")` works; the functional
  form `torch.sum(x, dim="height")` does not — a **PyTorch limitation** (its
  C-level argument parser rejects a string `dim` before our hook runs), not a
  choice of ours (see [Names & dimensions](names.md)).
