"""Tests for the named factory helpers."""

import torch

from fiery.xtensor import (
    XTensor,
    named_arange,
    named_eye,
    named_full,
    named_ones,
    named_zeros,
)


def test_named_zeros_sets_names_and_returns_named_tensor():
    x = named_zeros(2, 3, names=("row", "col"))
    assert isinstance(x, XTensor)
    assert x.names == ("row", "col")
    assert x.shape == (2, 3)


def test_factory_accepts_a_size_tuple_and_forwards_dtype():
    x = named_ones((2, 3), names=("a", "b"), dtype=torch.float64)
    assert x.names == ("a", "b")
    assert x.dtype == torch.float64
    assert torch.equal(x, torch.ones(2, 3, dtype=torch.float64))


def test_named_full_and_arange_and_eye():
    assert named_full((2, 2), 7, names=("a", "b"))[0, 0].item() == 7
    assert named_arange(5, names=("x",)).names == ("x",)
    assert named_arange(5).shape == (5,)
    assert named_eye(3, names=("r", "c")).names == ("r", "c")


def test_names_are_optional():
    assert named_zeros(2, 3).names == (None, None)


def test_factory_result_tracks_names_through_ops():
    x = named_zeros(2, 3, names=("a", "b"))
    assert x.sum(dim="a").names == ("b",)
    assert x.T.names == ("b", "a")
