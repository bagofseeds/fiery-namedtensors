"""Tests for the `x*` factory helpers."""

import torch

from fiery.xtensor import (
    XTensor,
    xarange,
    xempty_like,
    xeye,
    xfill,
    xfull,
    xlinspace,
    xones,
    xones_like,
    xzeros,
    xzeros_like,
)


def test_xzeros_sets_names_and_returns_named_tensor():
    x = xzeros(2, 3, names=("row", "col"))
    assert isinstance(x, XTensor)
    assert x.names == ("row", "col")
    assert x.shape == (2, 3)


def test_factory_accepts_a_size_tuple_and_forwards_dtype():
    x = xones((2, 3), names=("a", "b"), dtype=torch.float64)
    assert x.names == ("a", "b")
    assert x.dtype == torch.float64
    assert torch.equal(x, torch.ones(2, 3, dtype=torch.float64))


def test_xfull_and_arange_and_eye():
    assert xfull((2, 2), 7, names=("a", "b"))[0, 0].item() == 7
    assert xarange(5, names=("x",)).names == ("x",)
    assert xarange(5).shape == (5,)
    assert xeye(3, names=("r", "c")).names == ("r", "c")


def test_xlinspace_names_a_coordinate_axis():
    t = xlinspace(0.0, 1.0, 5, names=("t",))
    assert t.names == ("t",)
    assert t.shape == (5,)
    assert t[0].item() == 0.0 and t[-1].item() == 1.0


def test_xfill_is_xfull():
    assert xfill is xfull
    assert xfill((2,), 3.0, names=("x",)).tolist() == [3.0, 3.0]


def test_names_are_optional():
    assert xzeros(2, 3).names == (None, None)


def test_factory_result_tracks_names_through_ops():
    x = xzeros(2, 3, names=("a", "b"))
    assert x.sum(dim="a").names == ("b",)
    assert x.T.names == ("b", "a")


def test_factory_understands_axes_descriptors():
    x = xzeros(2, 3, axes=[{"name": "y", "type": "space"}, "x"])
    assert x.names == ("y", "x")
    assert x.axes[0]["type"] == "space"


def test_factory_understands_coords_and_unit():
    x = xzeros(3, names=("c",), coords={"c": ("r", "g", "b")}, unit="V")
    assert x.coords["c"] == ("r", "g", "b")
    assert x.unit == "V"
    assert x.sel(c="g").item() == 0.0


def test_like_inherits_names_and_coords():
    x = xzeros(2, 3, names=("row", "col"), coords={"col": ("a", "b", "c")})
    y = xones_like(x)
    assert y.names == ("row", "col")
    assert y.coords["col"] == ("a", "b", "c")
    assert torch.equal(y, torch.ones(2, 3))


def test_like_can_override_metadata():
    x = xzeros(2, 3, names=("row", "col"))
    y = xzeros_like(x, names=("a", "b"))
    assert y.names == ("a", "b")


def test_like_on_a_plain_tensor_returns_an_xtensor():
    y = xempty_like(torch.zeros(2, 2))
    assert isinstance(y, XTensor)
    assert y.shape == (2, 2)


def test_like_forwards_dtype():
    x = xones(2, names=("x",))
    y = xzeros_like(x, dtype=torch.int32)
    assert y.dtype == torch.int32
    assert y.names == ("x",)


def test_named_factories_are_gone():
    import fiery.xtensor as ns

    for old in ("named_zeros", "named_ones", "named_full"):
        assert not hasattr(ns, old)
