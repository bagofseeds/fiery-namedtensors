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
    assert torch.equal(y.rename(None), x.rename(None)[:, 1, :])


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
    # `rename(None)` is auto-wrapped by __torch_function__; the index
    # metadata must be propagated onto the result rather than dropped.
    x = TensorWithNamedIndices(
        torch.arange(6).reshape(2, 3),
        names=("row", "col"),
        index_names=(("c0", "c1", "c2"),),
        index_dims=(1,),
    )

    y = x.rename(None)

    assert y.index_names == (("c0", "c1", "c2"),)
    assert y.index_dims == (1,)


def test_tensor_without_named_indices_is_still_sliceable():
    x = TensorWithNamedIndices(torch.arange(6).reshape(2, 3))
    # Strip metadata, then slice: must not raise and must report no names.
    bare = x.rename(None)
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
    assert torch.equal(y.rename(None), x.rename(None).permute(2, 0, 1))


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
