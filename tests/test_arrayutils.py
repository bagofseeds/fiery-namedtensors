"""Tests for fiery.xtensor._arrayutils."""

# dependencies
import pytest
import torch

# fiery
from fiery.xtensor._arrayutils import (
    _count_output_axes,
    _map_axes,
    _map_axes_inverse,
    _unroll,
    _unroll_slicer,
)


class TestUnroll:
    """Tests for `_unroll` function."""

    def test_unroll_no_ellipsis(self) -> None:
        assert _unroll((0, 1), 3, insert=None) == (0, 1, None)
        assert _unroll((0, 1), 3, insert=None, side="left") == (None, 0, 1)

    def test_unroll_with_ellipsis(self) -> None:
        assert _unroll((..., 0), 3, insert=None) == (None, None, 0)
        assert _unroll((0, ...), 3, insert=None) == (0, None, None)
        assert _unroll((0, ..., 1), 4, insert=None) == (0, None, None, 1)

    def test_unroll_too_many_values(self) -> None:
        with pytest.raises(ValueError):
            _unroll((0, 1, 2), 2)

    def test_unroll_too_many_ellipsis(self) -> None:
        with pytest.raises(ValueError):
            _unroll((..., ...), 3)


count_output_axes_data = [
    (0, (10,)),
    (slice(None), (10,)),
    (..., (10,)),
    (None, (10,)),
    ([0, 1], (10,)),
    (torch.tensor([0, 1]), (10,)),
    (torch.tensor([True, False] * 5), (10,)),
    ((0, 1), (10, 20)),
    ((0, slice(None)), (10, 20)),
    ((slice(None), 0), (10, 20)),
    ((0, ...), (10, 20)),
    ((..., 0), (10, 20)),
    ((None, 0, ...), (10, 20)),
    ((..., None, 0), (10, 20)),
    (([0, 1], [0, 1]), (10, 20)),
    ((torch.tensor([0, 1]), torch.tensor([0, 1])), (10, 20)),
    ((slice(None), [0, 1]), (10, 20)),
    ((slice(None), torch.tensor([0, 1])), (10, 20)),
    ((slice(None), torch.tensor([True, False] * 10)), (10, 20)),
    ((torch.tensor([True, False] * 5), slice(None)), (10, 20)),
    ((torch.tensor([[True, False] * 10] * 10),), (10, 20)),
    (torch.tensor([[0, 1], [2, 3]]), (10,)),
    # 3 dimensions
    ((0, 1, 2), (10, 20, 30)),
    ((slice(None), 0, 1), (10, 20, 30)),
    ((0, slice(None), 1), (10, 20, 30)),
    ((0, 1, slice(None)), (10, 20, 30)),
    ((..., 0, 1), (10, 20, 30)),
    ((0, ..., 1), (10, 20, 30)),
    ((0, 1, ...), (10, 20, 30)),
    ((None, ..., 0, 1), (10, 20, 30)),
    ((0, None, ..., 1), (10, 20, 30)),
    ((0, 1, None, ...), (10, 20, 30)),
    ((slice(None), [0, 1], [0, 1]), (10, 20, 30)),
    ((slice(None), torch.tensor([0, 1]), torch.tensor([0, 1])), (10, 20, 30)),
    # 4 dimensions
    ((0, 1, 2, 3), (10, 20, 30, 40)),
    ((slice(None), 0, 1, 2), (10, 20, 30, 40)),
    ((0, slice(None), 1, 2), (10, 20, 30, 40)),
    ((0, 1, slice(None), 2), (10, 20, 30, 40)),
    ((0, 1, 2, slice(None)), (10, 20, 30, 40)),
    ((..., 0, 1, 2), (10, 20, 30, 40)),
    ((0, ..., 1, 2), (10, 20, 30, 40)),
    ((0, 1, ..., 2), (10, 20, 30, 40)),
    ((0, 1, 2, ...), (10, 20, 30, 40)),
]


@pytest.mark.parametrize("slicer, shape_in", count_output_axes_data)
def test_count_output_axes(slicer, shape_in) -> None:
    """Test `_count_output_axes` function."""
    x = torch.empty(shape_in)
    slicer_unrolled = _unroll_slicer(slicer, x.ndim)
    assert _count_output_axes(slicer_unrolled) == x[slicer].ndim


i = torch.arange(5).view(5, 1)
j = torch.arange(6).view(1, 6)
map_axes_data = [
    ((slice(None), 0, slice(None)), (0, 2)),
    ((slice(None), None, slice(None)), (0, None, 1)),
    ((slice(None), torch.ones([5, 6], dtype=torch.bool)), (0, (1, 2))),
    ((slice(None), range(5), range(5)), (0, (1, 2))),
    ((slice(None), i, j), (0, (1, 2), (1, 2))),
    ((slice(None), range(5), None, range(5)), ((1, 2), 0, None)),
]
map_axes_inverse_data = [
    ((slice(None), 0, slice(None)), (0, None, 1)),
    ((slice(None), None, slice(None)), (0, 2)),
    ((slice(None), torch.ones([5, 6], dtype=torch.bool)), (0, 1, 1)),
    ((slice(None), range(5), range(5)), (0, 1, 1)),
    ((slice(None), i, j), (0, (1, 2), (1, 2))),
    ((slice(None), range(5), None, range(5)), (1, 0, 0)),
]


@pytest.mark.parametrize("slicer, output", map_axes_data)
def test_map_axes(slicer, output) -> None:
    """Test `_map_axes` function."""
    assert _map_axes(slicer) == output


@pytest.mark.parametrize("slicer, output", map_axes_inverse_data)
def test_map_axes_inverse(slicer, output) -> None:
    """Test `_map_axes_inverse` function."""
    assert _map_axes_inverse(slicer) == output
