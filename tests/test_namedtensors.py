"""Tests for fiery.namedtensors."""

import pytest
import torch

from fiery.namedtensors import (
    NamedTensor,
    TensorWithNamedIndices,
)
from fiery.namedtensors._tensors import (
    _get_sequence_depth,
    _prepare_index_names,
    _slice_names,
    _slice_names_nd,
)


def test_named_tensor_getitem_with_new_axis_keeps_names():
    x = NamedTensor(torch.arange(6).reshape(2, 3), names=("row", "col"))

    y = x[:, None, 1:]

    assert isinstance(y, NamedTensor)
    assert y.shape == (2, 1, 2)
    assert y.names == ("row", None, "col")


def test_named_tensor_T_reverses_axis_order_and_names():
    x = NamedTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("batch", "height", "width"),
    )

    y = x.T

    assert y.shape == (4, 3, 2)
    assert y.names == ("width", "height", "batch")


def test_unsqueeze_and_squeeze_round_trip_axis_names():
    x = NamedTensor(torch.arange(6).reshape(2, 3), names=("row", "col"))

    y = x.unsqueeze(1)
    z = y.squeeze(1)

    assert y.names == ("row", None, "col")
    assert z.names == ("row", "col")


def test_squeeze_without_dim_removes_singleton_axis_names():
    x = NamedTensor(torch.ones(1, 3, 1), names=("left", "mid", "right"))

    y = x.squeeze()

    assert y.shape == (3,)
    assert y.names == ("mid",)


def test_view_keeps_matching_leading_name_and_marks_reshaped_axes_unnamed():
    x = NamedTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("batch", "height", "width"),
    )

    y = x.view(2, 12)
    z = x.view(2, -1)

    assert y.names == ("batch", None)
    assert z.names == ("batch", None)


def test_tensor_with_named_indices_attribute_lookup_by_name_path():
    x = TensorWithNamedIndices(
        torch.arange(6).reshape(2, 3),
        names=("row", "col"),
        index_names=(("r0", "r1"), ("c0", "c1", "c2")),
        index_dims=(0, 1),
    )

    out = x.r1.c2

    assert out.ndim == 0
    assert out.item() == 5


def test_tensor_with_named_indices_unknown_attribute_raises():
    x = TensorWithNamedIndices(
        torch.arange(6).reshape(2, 3),
        names=("row", "col"),
        index_names=(("r0", "r1"), ("c0", "c1", "c2")),
        index_dims=(0, 1),
    )

    with pytest.raises(AttributeError, match="No such index"):
        _ = x.r2


def test_tensor_with_named_indices_getitem_updates_index_metadata():
    x = TensorWithNamedIndices(
        torch.arange(24).reshape(2, 3, 4),
        names=("batch", "channel", "coord"),
        index_names=(("c0", "c1", "c2"), ("x", "y", "z", "t")),
        index_dims=(1, 2),
    )

    y = x[:, 1, :2]

    assert y.shape == (2, 2)
    assert y.index_names == (("x", "y"),)
    assert y.index_dims == (1,)


def test_tensor_with_named_indices_index_selects_positions_and_updates_meta():
    x = TensorWithNamedIndices(
        torch.arange(24).reshape(2, 3, 4),
        names=("batch", "channel", "coord"),
        index_names=(("c0", "c1", "c2"), ("x", "y", "z", "t")),
        index_dims=(1, 2),
    )

    # Select channel c1 (position 1 on dim 1); dim 2 kept in full.
    y = x.index(1, dims=1)

    assert y.shape == (2, 4)
    # Dim 1 was consumed; the coord index moves up to dim 1.
    assert y.index_names == (("x", "y", "z", "t"),)
    assert y.index_dims == (1,)
    assert torch.equal(y, x.as_subclass(torch.Tensor)[:, 1, :])


def test_tensor_with_named_indices_index_mismatched_lengths_raises():
    x = TensorWithNamedIndices(
        torch.arange(24).reshape(2, 3, 4),
        names=("batch", "channel", "coord"),
        index_names=(("c0", "c1", "c2"), ("x", "y", "z", "t")),
        index_dims=(1, 2),
    )

    with pytest.raises(ValueError, match="does not match"):
        x.index((0, 1), dims=(1,))


def test_get_sequence_depth_handles_nested_sequences_and_strings():
    assert _get_sequence_depth(["a", "b"]) == 1
    assert _get_sequence_depth([["a"], ["b", "c"]]) == 2
    assert _get_sequence_depth("abc") == 0
    assert _get_sequence_depth([]) == 1


def test_prepare_index_names_unrolls_ellipsis_and_normalizes_dims():
    names, dims = _prepare_index_names(
        index_names=(("r0", "r1"), (...,)),
        index_dims=(0, 2),
        shape=(2, 3, 4),
    )

    assert names[0] == ("r0", "r1")
    assert names[1] == (None, None, None, None)
    assert dims == (0, 2)


def test_prepare_index_names_rejects_invalid_depth():
    with pytest.raises(ValueError, match="Invalid index_names"):
        _prepare_index_names(index_names=123, index_dims=0, shape=(2, 3))


def test_slice_names_supports_int_slice_bool_and_advanced_indices():
    names = ("a", "b", "c", "d")

    assert _slice_names(names, 1) == "b"
    assert _slice_names(names, slice(1, 3)) == ("b", "c")
    assert _slice_names(names, [True, False, True, False]) == ("a", "c")
    assert _slice_names(names, [3, 0]) == ("d", "a")


def test_slice_names_nd_tracks_output_dims_after_slicing():
    index_names = (("c0", "c1", "c2"), ("x", "y", "z", "t"))
    index_dims = (1, 2)
    slicer = (slice(None), 1, [0, 2])

    new_names, new_dims = _slice_names_nd(index_names, index_dims, slicer)

    assert new_names == (("x", "z"),)
    assert new_dims == (1,)


# --- regression tests for the fiery port -----------------------------------


def test_index_metadata_survives_auto_wrapped_ops():
    # `clone()` has no name-aware override, so it is auto-wrapped by
    # __torch_function__; the index metadata must be propagated onto the
    # result rather than dropped.
    x = TensorWithNamedIndices(
        torch.arange(6).reshape(2, 3),
        names=("row", "col"),
        index_names=(("c0", "c1", "c2"),),
        index_dims=(1,),
    )

    y = x.clone()

    assert y.index_names == (("c0", "c1", "c2"),)
    assert y.index_dims == (1,)


def test_tensor_without_named_indices_is_still_sliceable():
    x = TensorWithNamedIndices(torch.arange(6).reshape(2, 3))
    # Strip metadata, then slice: must not raise and must report no names.
    bare = x.clone()
    bare.__dict__.pop("_index_names", None)
    bare.__dict__.pop("_index_dims", None)

    y = bare[:, 1]

    assert y.index_names is None
    assert y.shape == (2,)


def test_view_preserves_trailing_unchanged_axis_name():
    x = NamedTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("batch", "height", "width"),
    )

    y = x.view(6, 4)

    assert y.shape == (6, 4)
    assert y.names == (None, "width")


def test_torch_func_returns_none_for_missing_op():
    from fiery.namedtensors._tensors import _torch_func

    assert _torch_func("permute") is not None
    assert _torch_func("this_op_does_not_exist_anywhere") is None


def test_tensor_with_named_indices_permute_reorders_index_dims():
    x = TensorWithNamedIndices(
        torch.arange(24).reshape(2, 3, 4),
        index_names=(("c0", "c1", "c2"), ("x", "y", "z", "t")),
        index_dims=(1, 2),
    )

    y = x.permute(2, 0, 1)

    assert y.shape == (4, 2, 3)
    # dim 1 -> position 2, dim 2 -> position 0; per-axis names unchanged.
    assert y.index_names == (("c0", "c1", "c2"), ("x", "y", "z", "t"))
    assert y.index_dims == (2, 0)
    assert torch.equal(y, x.as_subclass(torch.Tensor).permute(2, 0, 1))


def test_tensor_with_named_indices_index_select_reslices_axis_names():
    x = TensorWithNamedIndices(
        torch.arange(24).reshape(2, 3, 4),
        index_names=(("c0", "c1", "c2"), ("x", "y", "z", "t")),
        index_dims=(1, 2),
    )

    # Method form is the documented API and carries the metadata correctly.
    # (Functional `torch.index_select(x, ...)` does not yet propagate
    # override-set metadata through the outer dispatch -- tracked in the
    # roadmap under unifying propagation.)
    y = x.index_select(2, torch.tensor([0, 2]))

    assert y.shape == (2, 3, 2)
    # Only the dim-2 axis names are re-sliced; dim-1 names untouched.
    assert y.index_names == (("c0", "c1", "c2"), ("x", "z"))
    assert y.index_dims == (1, 2)


def test_rename_out_of_place_sets_and_clears_names():
    x = NamedTensor(torch.zeros(2, 3), names=("row", "col"))

    y = x.rename("a", "b")

    assert y.names == ("a", "b")
    assert x.names == ("row", "col")  # out-of-place: x unchanged
    assert y.rename(None).names == (None, None)
    plain = torch.Tensor
    assert torch.equal(y.as_subclass(plain), x.as_subclass(plain))


def test_rename_by_map():
    x = NamedTensor(torch.zeros(2, 3), names=("row", "col"))

    assert x.rename(col="C").names == ("row", "C")
    with pytest.raises(ValueError, match="no axis named"):
        x.rename(nope="X")


def test_rename_in_place_returns_self():
    x = NamedTensor(torch.zeros(2, 3), names=("row", "col"))

    out = x.rename_("a", "b")

    assert out is x
    assert x.names == ("a", "b")


def test_rename_preserves_index_metadata():
    x = TensorWithNamedIndices(
        torch.arange(6).reshape(2, 3),
        index_names=(("c0", "c1", "c2"),),
        index_dims=(1,),
    )

    y = x.rename("row", "col")

    assert y.names == ("row", "col")
    assert y.index_names == (("c0", "c1", "c2"),)
    assert y.index_dims == (1,)


@pytest.mark.skipif(
    not hasattr(torch.Tensor, "names"),
    reason="builtin named-tensor API not present in this torch build",
)
def test_names_do_not_use_builtin_named_tensors():
    # The self-managed names must not set the underlying tensor's builtin
    # (C-level) names -- that is what keeps us portable across torch versions.
    x = NamedTensor(torch.zeros(2, 3), names=("row", "col"))
    assert torch.Tensor.names.__get__(x) == (None, None)


# --- R4: functional-form (torch.op(x, ...)) metadata parity -----------------


@pytest.mark.skipif(
    not hasattr(torch, "permute"),
    reason="torch.permute (functional) was added in torch 1.9",
)
def test_functional_permute_matches_method_form():
    x = NamedTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    assert torch.permute(x, (2, 0, 1)).names == x.permute(2, 0, 1).names
    assert torch.permute(x, (2, 0, 1)).names == ("c", "a", "b")


def test_functional_unsqueeze_squeeze_match_method_form():
    x = NamedTensor(torch.zeros(2, 3), names=("row", "col"))
    assert torch.unsqueeze(x, 1).names == ("row", None, "col")

    y = NamedTensor(torch.zeros(1, 3), names=("s", "col"))
    assert torch.squeeze(y).names == ("col",)


def test_functional_index_select_matches_method_form():
    x = TensorWithNamedIndices(
        torch.arange(24).reshape(2, 3, 4),
        index_names=(("c0", "c1", "c2"), ("x", "y", "z", "t")),
        index_dims=(1, 2),
    )
    idx = torch.tensor([0, 2])
    func = torch.index_select(x, 2, idx)
    meth = x.index_select(2, idx)
    assert (
        func.index_names
        == meth.index_names
        == (("c0", "c1", "c2"), ("x", "z"))
    )
    assert func.index_dims == meth.index_dims == (1, 2)


@pytest.mark.skipif(
    not hasattr(torch, "permute"),
    reason="torch.permute (functional) was added in torch 1.9",
)
def test_functional_permute_carries_index_dims():
    x = TensorWithNamedIndices(
        torch.arange(24).reshape(2, 3, 4),
        index_names=(("c0", "c1", "c2"), ("x", "y", "z", "t")),
        index_dims=(1, 2),
    )
    y = torch.permute(x, (2, 0, 1))
    assert y.index_dims == (2, 0)
    assert y.index_names == (("c0", "c1", "c2"), ("x", "y", "z", "t"))


# --- reshape / reorder op family --------------------------------------------


def test_transpose_family_swaps_axis_names():
    x = NamedTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    assert x.transpose(0, 2).names == ("c", "b", "a")
    assert x.swapaxes(0, 1).names == ("b", "a", "c")
    assert x.swapdims(1, 2).names == ("a", "c", "b")
    # functional form matches the method form
    assert torch.transpose(x, 0, 2).names == ("c", "b", "a")


def test_mT_transposes_last_two_axis_names():
    x = NamedTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    assert x.mT.names == ("a", "c", "b")
    assert x.mT.shape == (2, 4, 3)


def test_movedim_reorders_axis_names_like_torch():
    x = NamedTensor(torch.zeros(2, 3, 4, 5), names=("a", "b", "c", "d"))
    y = x.movedim(0, 2)
    assert y.names == ("b", "c", "a", "d")
    assert y.shape == tuple(torch.movedim(torch.zeros(2, 3, 4, 5), 0, 2).shape)
    assert x.moveaxis((0, 1), (2, 3)).names == ("c", "d", "a", "b")


def test_reshape_uses_same_name_rule_as_view():
    x = NamedTensor(torch.arange(24).reshape(2, 3, 4), names=("b", "h", "w"))
    assert x.reshape(2, 12).names == ("b", None)
    assert x.reshape(6, 4).names == (None, "w")
    assert torch.reshape(x, (2, -1)).names == ("b", None)


def test_transpose_carries_index_dims():
    x = TensorWithNamedIndices(
        torch.arange(24).reshape(2, 3, 4),
        index_names=(("c0", "c1", "c2"), ("x", "y", "z", "t")),
        index_dims=(1, 2),
    )
    y = x.transpose(1, 2)
    assert y.index_dims == (2, 1)
    assert y.index_names == (("c0", "c1", "c2"), ("x", "y", "z", "t"))


# ----------------------------------------------------------------------
# name-as-dim: a name may stand in for an integer `dim=`
# ----------------------------------------------------------------------


def test_permute_accepts_axis_names():
    x = NamedTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    y = x.permute("c", "a", "b")
    assert y.names == ("c", "a", "b")
    assert y.shape == (4, 2, 3)
    # matches the equivalent integer permutation
    assert y.shape == x.permute(2, 0, 1).shape


def test_transpose_family_accepts_axis_names():
    # Name-as-dim is a method-form feature: the functional form
    # (`torch.transpose(x, "a", ...)`) cannot be relied on because newer
    # PyTorch rejects a non-int dim at the C dispatcher before
    # `__torch_function__` runs.
    x = NamedTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    assert x.transpose("a", "c").names == ("c", "b", "a")
    assert x.swapaxes("a", "b").names == ("b", "a", "c")
    assert x.swapdims("b", "c").names == ("a", "c", "b")


def test_movedim_accepts_axis_names_for_source():
    x = NamedTensor(torch.zeros(2, 3, 4, 5), names=("a", "b", "c", "d"))
    assert x.movedim("a", 2).names == ("b", "c", "a", "d")
    assert x.moveaxis(("a", "b"), (2, 3)).names == ("c", "d", "a", "b")


def test_squeeze_accepts_axis_name():
    x = NamedTensor(torch.ones(2, 1, 3), names=("a", "one", "b"))
    y = x.squeeze("one")
    assert y.names == ("a", "b")
    assert y.shape == (2, 3)


def test_index_select_accepts_axis_name():
    x = TensorWithNamedIndices(
        torch.arange(24).reshape(2, 3, 4),
        index_names=(("w", "x", "y", "z"),),
        index_dims=(2,),
    )
    x.names = ("batch", "feat", "chan")
    # select along the axis addressed by its (axis) name
    by_name = x.index_select("chan", torch.tensor([0, 2]))
    by_int = x.index_select(2, torch.tensor([0, 2]))
    assert by_name.shape == by_int.shape == (2, 3, 2)
    # the sliced index names match, whichever form was used
    assert by_name.index_names == by_int.index_names == (("w", "y"),)


def test_name_as_dim_unknown_name_raises():
    x = NamedTensor(torch.zeros(2, 3), names=("a", "b"))
    with pytest.raises(ValueError, match="no axis named 'z'"):
        x.transpose("a", "z")


# ----------------------------------------------------------------------
# reductions: drop / keep the reduced axis' name, accept a name for `dim`
# ----------------------------------------------------------------------


def test_sum_drops_reduced_axis_name():
    x = NamedTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.sum(dim="b").names == ("a", "c")
    assert x.sum(dim="b").shape == (2, 4)
    assert x.sum(dim=1).names == ("a", "c")  # int still works


def test_sum_keepdim_preserves_reduced_axis_name():
    x = NamedTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    y = x.sum(dim="b", keepdim=True)
    assert y.names == ("a", "b", "c")
    assert y.shape == (2, 1, 4)


def test_reduction_over_multiple_named_axes():
    x = NamedTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.mean(dim=("a", "c")).names == ("b",)


def test_reduce_all_yields_scalar_with_no_names():
    x = NamedTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    s = x.sum()
    assert s.names == ()
    assert s.ndim == 0


def test_functional_reduction_carries_names_like_method():
    x = NamedTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert torch.mean(x, 1).names == x.mean(dim=1).names == ("a", "c")


def test_argmax_and_amax_track_names():
    x = NamedTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.amax(dim="c").names == ("a", "b")
    assert x.argmax(dim="a").names == ("b", "c")


def test_negative_dim_reduction():
    x = NamedTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.sum(dim=-1).names == ("a", "b")


def test_reduction_drops_named_index_metadata_for_reduced_axis():
    t = TensorWithNamedIndices(
        torch.arange(24).reshape(2, 3, 4),
        index_names=(("p", "q", "r"), ("w", "x", "y", "z")),
        index_dims=(1, 2),
    )
    t.names = ("a", "b", "c")
    # reduce the axis carrying the first index group
    r = t.sum(dim="b")
    assert r.names == ("a", "c")
    assert r.index_dims == (1,)
    assert r.index_names == (("w", "x", "y", "z"),)
    # reduce a plain axis: index dims shift down, index names unchanged
    r2 = t.sum(dim="a")
    assert r2.index_dims == (0, 1)
    assert r2.index_names == (("p", "q", "r"), ("w", "x", "y", "z"))


# ----------------------------------------------------------------------
# slice / split ops: select / narrow / unbind / split / chunk / flip / roll
# ----------------------------------------------------------------------


def test_select_drops_axis_name():
    x = NamedTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.select("b", 0).names == ("a", "c")
    assert x.select("b", 0).shape == (2, 4)
    # functional form matches
    assert torch.select(x, 1, 0).names == ("a", "c")


def test_narrow_keeps_names_and_accepts_name_dim():
    x = NamedTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    y = x.narrow("c", 1, 2)
    assert y.names == ("a", "b", "c")
    assert y.shape == (2, 3, 2)


def test_unbind_returns_pieces_without_the_unbound_axis():
    x = NamedTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    pieces = x.unbind("a")
    assert len(pieces) == 2
    assert all(p.names == ("b", "c") for p in pieces)
    assert all(p.shape == (3, 4) for p in pieces)


def test_split_and_chunk_keep_all_names():
    x = NamedTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    parts = x.split(2, dim="c")
    assert [p.shape for p in parts] == [(2, 3, 2), (2, 3, 2)]
    assert all(p.names == ("a", "b", "c") for p in parts)
    chunks = x.chunk(2, dim="b")
    assert [p.shape for p in chunks] == [(2, 2, 4), (2, 1, 4)]
    assert all(p.names == ("a", "b", "c") for p in chunks)


def test_flip_and_roll_preserve_names():
    x = NamedTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.flip(("a", "c")).names == ("a", "b", "c")
    assert x.roll(1, dims="b").names == ("a", "b", "c")
    assert x.roll(2).names == ("a", "b", "c")  # flattened roll


def test_select_drops_named_index_group():
    t = TensorWithNamedIndices(
        torch.arange(24).reshape(2, 3, 4),
        index_names=(("p", "q", "r"), ("w", "x", "y", "z")),
        index_dims=(1, 2),
    )
    # select the axis carrying the first index group
    r = t.select(1, 0)
    assert r.index_dims == (1,)
    assert r.index_names == (("w", "x", "y", "z"),)


def test_narrow_and_split_slice_named_indices():
    t = TensorWithNamedIndices(
        torch.arange(24).reshape(2, 3, 4),
        index_names=(("w", "x", "y", "z"),),
        index_dims=(2,),
    )
    assert t.narrow(2, 1, 2).index_names == (("x", "y"),)
    parts = t.split(2, dim=2)
    assert [p.index_names for p in parts] == [(("w", "x"),), (("y", "z"),)]


def test_flip_reverses_named_indices_on_flipped_axis():
    t = TensorWithNamedIndices(
        torch.arange(24).reshape(2, 3, 4),
        index_names=(("w", "x", "y", "z"),),
        index_dims=(2,),
    )
    assert t.flip(2).index_names == (("z", "y", "x", "w"),)


def test_roll_rolls_named_indices_on_rolled_axis():
    t = TensorWithNamedIndices(
        torch.arange(12).reshape(3, 4),
        index_names=(("w", "x", "y", "z"),),
        index_dims=(1,),
    )
    # a right shift of 1 moves each index name one step forward (cyclically)
    assert t.roll(1, dims=1).index_names == (("z", "w", "x", "y"),)


# ----------------------------------------------------------------------
# reshape-family (rank-changing): flatten / unflatten / expand / diagonal
# ----------------------------------------------------------------------


def test_flatten_marks_merged_axis_unnamed():
    x = NamedTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.flatten(0, 1).names == (None, "c")
    assert x.flatten(0, 1).shape == (6, 4)
    assert x.flatten("a", "b").names == (None, "c")  # by name
    assert x.flatten(1, 1).names == ("a", "b", "c")  # no-op keeps names
    assert torch.flatten(x, 1, 2).names == ("a", None)  # functional form


def test_unflatten_marks_split_axes_unnamed():
    x = NamedTensor(torch.arange(24).reshape(6, 4), names=("a", "b"))
    y = x.unflatten("a", (2, 3))
    assert y.names == (None, None, "b")
    assert y.shape == (2, 3, 4)
    # a single-element split is a no-op and keeps the name
    assert x.unflatten(0, (6,)).names == ("a", "b")


def test_expand_and_broadcast_to_prepend_unnamed_axes():
    x = NamedTensor(torch.zeros(3, 4), names=("b", "c"))
    assert x.expand(2, 3, 4).names == (None, "b", "c")
    assert torch.broadcast_to(x, (2, 3, 4)).names == (None, "b", "c")


def test_diagonal_drops_the_two_axes_and_appends_unnamed():
    x = NamedTensor(torch.zeros(3, 3, 4), names=("a", "b", "c"))
    y = x.diagonal(0, "a", "b")
    assert y.names == ("c", None)
    assert y.shape == (4, 3)


def test_flatten_drops_named_index_metadata_in_merged_range():
    t = TensorWithNamedIndices(
        torch.arange(24).reshape(2, 3, 4),
        index_names=(("p", "q", "r"), ("w", "x", "y", "z")),
        index_dims=(1, 2),
    )
    # both named axes are inside the merged range -> index metadata dropped
    assert t.flatten(1, 2).index_names is None


def test_expand_shifts_named_index_dims():
    e = TensorWithNamedIndices(
        torch.zeros(3, 4),
        index_names=(("w", "x", "y", "z"),),
        index_dims=(1,),
    )
    out = e.expand(2, 3, 4)
    assert out.index_dims == (2,)
    assert out.index_names == (("w", "x", "y", "z"),)


def test_diagonal_shifts_surviving_named_index_dim():
    d = TensorWithNamedIndices(
        torch.zeros(3, 3, 4),
        index_names=(("w", "x", "y", "z"),),
        index_dims=(2,),
    )
    out = d.diagonal(0, 0, 1)
    assert out.index_dims == (0,)
    assert out.index_names == (("w", "x", "y", "z"),)


# ----------------------------------------------------------------------
# combine ops: cat / stack (name reconciliation across operands)
# ----------------------------------------------------------------------


def test_cat_reconciles_axis_names():
    a = NamedTensor(torch.zeros(2, 3), names=("r", "c"))
    b = NamedTensor(torch.zeros(4, 3), names=("r", "c"))
    out = torch.cat([a, b], 0)
    assert out.names == ("r", "c")
    assert out.shape == (6, 3)
    # a name is usable for the dim
    assert torch.cat([a, b], "r").names == ("r", "c")


def test_cat_conflicting_names_become_unnamed():
    a = NamedTensor(torch.zeros(2, 3), names=("r", "c"))
    conflicting = NamedTensor(torch.zeros(2, 3), names=("r", "x"))
    assert torch.cat([a, conflicting], 0).names == ("r", None)


def test_stack_inserts_an_unnamed_axis():
    a = NamedTensor(torch.zeros(2, 3), names=("r", "c"))
    assert torch.stack([a, a], 0).names == (None, "r", "c")
    assert torch.stack([a, a], 1).names == ("r", None, "c")
    assert torch.stack([a, a], 0).shape == (2, 2, 3)


def test_cat_concatenates_index_names_along_index_axis():
    t1 = TensorWithNamedIndices(
        torch.zeros(2, 2), index_names=(("p", "q"),), index_dims=(1,)
    )
    t2 = TensorWithNamedIndices(
        torch.zeros(2, 3), index_names=(("r", "s", "u"),), index_dims=(1,)
    )
    out = torch.cat([t1, t2], 1)
    assert out.index_names == (("p", "q", "r", "s", "u"),)
    assert out.index_dims == (1,)
    assert out.shape == (2, 5)


def test_cat_with_a_plain_operand_drops_index_metadata():
    t1 = TensorWithNamedIndices(
        torch.zeros(2, 2), index_names=(("p", "q"),), index_dims=(1,)
    )
    out = torch.cat([t1, NamedTensor(torch.zeros(2, 2))], 1)
    assert out.index_names is None


def test_stack_shifts_index_dims_past_the_new_axis():
    t1 = TensorWithNamedIndices(
        torch.zeros(2, 2), index_names=(("p", "q"),), index_dims=(1,)
    )
    out = torch.stack([t1, t1], 0)
    assert out.index_dims == (2,)
    assert out.index_names == (("p", "q"),)
    assert out.shape == (2, 2, 2)
