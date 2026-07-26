"""Tests for fiery.xtensor._compat."""

import pytest
import torch

from fiery.xtensor._compat import (
    EllipsisType,
    broadcast_shape,
    torch_func,
)


def test_ellipsis_type_matches_ellipsis():
    assert isinstance(..., EllipsisType)
    assert EllipsisType is type(...)


@pytest.mark.parametrize(
    "shapes, expected",
    [
        (((5, 1), (1, 6)), (5, 6)),
        (((3,), (2, 3)), (2, 3)),
        (((5, 1), (1, 6), (5, 6)), (5, 6)),
        (((), (4,)), (4,)),
        (((1, 1), (1, 1)), (1, 1)),
    ],
)
def test_broadcast_shape_matches_torch(shapes, expected):
    result = broadcast_shape(*shapes)
    assert isinstance(result, torch.Size)
    assert tuple(result) == expected
    # Agrees with PyTorch's own broadcasting, where available (torch >= 1.8;
    # `broadcast_shape` exists precisely to cover the torch that predate it).
    if hasattr(torch, "broadcast_shapes"):
        assert tuple(result) == tuple(torch.broadcast_shapes(*shapes))


def test_broadcast_shape_rejects_incompatible():
    with pytest.raises(RuntimeError):
        broadcast_shape((2, 3), (4, 3))


def test_broadcast_shape_allocates_nothing_for_huge_shapes():
    # Pure shape arithmetic must not allocate: enormous shapes are fine.
    result = broadcast_shape((10**9, 1), (1, 10**9))
    assert tuple(result) == (10**9, 10**9)


def test_torch_func_resolves_existing_and_missing():
    assert torch_func("permute") is not None
    assert torch_func("index_select") is not None
    assert torch_func("this_op_does_not_exist_anywhere") is None
