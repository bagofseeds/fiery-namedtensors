"""Tests for fiery.xtensor."""

import pytest
import torch

from fiery.xtensor import (
    XTensor,
    set_options,
    xmatrix,
    xvector,
)
from fiery.xtensor._tensors import _slice_labels, _torch_func

# Ops added in torch 1.8; the package registers them only when present, so on
# an older torch (the 1.7 floor) the corresponding tests are skipped.
_HAS_SWAPAXES = hasattr(torch, "swapaxes")  # swapaxes / swapdims
_HAS_MOVEAXIS = hasattr(torch, "moveaxis")
_HAS_BROADCAST_TO = hasattr(torch, "broadcast_to")

# ----------------------------------------------------------------------
# dimensions (names)
# ----------------------------------------------------------------------


def test_named_tensor_getitem_with_new_axis_keeps_names():
    x = XTensor(torch.arange(6).reshape(2, 3), names=("row", "col"))
    y = x[:, None, 1:]
    assert isinstance(y, XTensor)
    assert y.shape == (2, 1, 2)
    assert y.names == ("row", None, "col")


def test_named_tensor_T_reverses_axis_order_and_names():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("batch", "height", "width"),
    )
    y = x.T
    assert y.shape == (4, 3, 2)
    assert y.names == ("width", "height", "batch")


def test_unsqueeze_and_squeeze_round_trip_axis_names():
    x = XTensor(torch.arange(6).reshape(2, 3), names=("row", "col"))
    y = x.unsqueeze(1)
    z = y.squeeze(1)
    assert y.names == ("row", None, "col")
    assert z.names == ("row", "col")


def test_squeeze_without_dim_removes_singleton_axis_names():
    x = XTensor(torch.ones(1, 3, 1), names=("left", "mid", "right"))
    y = x.squeeze()
    assert y.shape == (3,)
    assert y.names == ("mid",)


def test_view_keeps_matching_leading_name_and_marks_reshaped_axes_unnamed():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("batch", "height", "width"),
    )
    assert x.view(2, 12).names == ("batch", None)
    assert x.view(2, -1).names == ("batch", None)


def test_view_preserves_trailing_unchanged_axis_name():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("batch", "height", "width"),
    )
    y = x.view(6, 4)
    assert y.shape == (6, 4)
    assert y.names == (None, "width")


# ----------------------------------------------------------------------
# coordinates (labels keyed by dim name)
# ----------------------------------------------------------------------


def _labelled():
    return XTensor(
        torch.arange(12).reshape(3, 4),
        names=("row", "col"),
        coords={"col": ("w", "x", "y", "z")},
    )


def test_coords_are_keyed_by_dim_name():
    x = _labelled()
    assert x.coords == {"col": ("w", "x", "y", "z")}


def test_coords_constructor_requires_a_named_axis():
    # a coord keyed by a non-axis name is read as a non-dimension coordinate
    # (Proposal 0005), which must be a (dim, values) tuple -- `()` is not one
    with pytest.raises(ValueError, match="not an axis"):
        XTensor(torch.zeros(2, 3), names=("row", None), coords={"col": ()})


# -- non-dimension coordinates (Proposal 0005, first slice) -------------------


def test_non_dimension_coordinate_rides_along_a_dim():
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={
            "t": ["a", "b", "c", "d"],  # dimension coord (the index)
            "season": ("t", ["w", "w", "sp", "sp"]),  # non-dim coord along t
        },
    )
    assert sorted(x.coords) == ["season", "t"]
    assert x.coords["season"] == ("w", "w", "sp", "sp")
    assert x.sel(t="b").item() == 1.0  # the index is selectable
    with pytest.raises(ValueError, match="not an index coordinate"):
        x.sel(season="sp")  # a non-dimension coordinate is not an index


def test_non_dimension_coordinate_propagates_and_drops():
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"season": ("t", ["w", "w", "sp", "sp"])},
    )
    assert x.rename(t="time").coords["season"] == ("w", "w", "sp", "sp")
    assert "season" not in x.sum(dim="t").coords  # dim removed -> dropped
    assert (
        "season" not in x[:2].coords
    )  # dim resized -> dropped (conservative)


def test_non_dimension_coordinate_survives_slicing_an_unrelated_axis():
    x = XTensor(
        torch.arange(12.0).reshape(3, 4),
        names=("t", "u"),
        coords={"season": ("t", ["w", "w", "sp"])},
    )
    out = x[:, :2]  # "u" is resized, "t" (and its rider) is untouched
    assert out.coords["season"] == ("w", "w", "sp")


def test_non_dimension_numeric_coordinate():
    x = XTensor(
        torch.arange(3.0),
        names=("i",),
        coords={
            "wl": (
                "i",
                XTensor(torch.tensor([400.0, 500.0, 600.0]), unit="nm"),
            )
        },
    )
    assert x.coords["wl"]["values"].tolist() == [400.0, 500.0, 600.0]
    assert x.coords["wl"]["values"].unit == "nm"


def test_non_dimension_compact_coordinate_is_rejected():
    # a compact (spacing/origin) non-dimension coordinate isn't re-sliced
    # when its dim is (no slice-tracking yet) and would silently rebind to
    # the wrong affine after a resize -- rejected rather than allowed to
    # silently misbehave; an explicit tensor of values works (see above).
    with pytest.raises(NotImplementedError, match="isn't supported yet"):
        XTensor(
            torch.arange(4.0),
            names=("i",),
            coords={"wl": ("i", {"spacing": 10.0, "origin": 400.0})},
        )


def test_non_dimension_coordinate_length_is_checked():
    with pytest.raises(ValueError, match="has 2 values for dim"):
        XTensor(
            torch.arange(4.0), names=("t",), coords={"s": ("t", ["a", "b"])}
        )


def test_non_dimension_coordinate_drops_when_its_dim_is_reordered():
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"season": ("t", ["w", "w", "sp", "sp"])},
    )
    # a rider's positions no longer correspond once its dim is
    # sorted/flipped/rolled/gathered/index_selected -- conservatively dropped,
    # same as the dimension coordinate itself would be
    assert "season" not in x.flip("t").coords
    assert "season" not in x.roll(1, "t").coords
    assert "season" not in x.sort("t").values.coords
    idx = torch.tensor([0, 1, 2, 3])
    assert "season" not in x.gather("t", idx).coords
    assert "season" not in x.index_select("t", idx[:2]).coords


def test_rename_raises_on_a_coordinate_name_collision():
    # renaming an axis onto an existing non-dimension coordinate's name would
    # otherwise silently drop one of the two entries (dict key collision)
    x = XTensor(
        torch.arange(6.0).reshape(2, 3),
        names=("t", "u"),
        coords={"season": ("t", ["w", "sp"]), "u": ["x", "y", "z"]},
    )
    with pytest.raises(ValueError, match="coordinate name collision"):
        x.rename(u="season")
    # a non-colliding rename is unaffected
    renamed = x.rename(u="month")
    assert renamed.coords["season"] == ("w", "sp")
    assert renamed.coords["month"] == ("x", "y", "z")


def test_coords_constructor_checks_label_count():
    with pytest.raises(ValueError, match="labels for size"):
        XTensor(
            torch.zeros(2, 3), names=("row", "col"), coords={"col": ("a", "b")}
        )


def test_coords_ellipsis_fills_unlabelled_positions():
    x = XTensor(
        torch.zeros(2, 4),
        names=("row", "col"),
        coords={"col": ("a", ..., "z")},
    )
    assert x.coords["col"] == ("a", None, None, "z")


# -- compact numeric coordinates (Proposal 0001 phase 1) ----------------------


def test_compact_numeric_coordinate_stores_and_materialises():
    x = XTensor(
        torch.zeros(2, 4),
        names=("y", "x"),
        coords={"x": {"spacing": (0.5, "mm"), "origin": (-1.0, "mm")}},
    )
    cx = x.coords["x"]
    assert cx["spacing"]["value"] == 0.5
    assert cx["spacing"].unit == "mm"  # attribute sugar for a safe key
    # `["values"]` is a derived key: origin + i*spacing
    assert cx["values"].tolist() == [-1.0, -0.5, 0.0, 0.5]
    assert cx["values"].unit == "mm"  # the POSITION unit


def test_numeric_and_categorical_coords_coexist():
    x = XTensor(
        torch.zeros(3, 4),
        names=("c", "x"),
        coords={"c": ["r", "g", "b"], "x": {"spacing": (0.5, "mm")}},
    )
    assert x.coords["c"] == ("r", "g", "b")  # labels
    assert x.coords["x"]["values"].tolist() == [0.0, 0.5, 1.0, 1.5]


def test_numeric_coord_propagates_and_drops_with_its_axis():
    x = XTensor(
        torch.zeros(2, 4),
        names=("y", "x"),
        coords={"x": {"spacing": (0.5, "mm")}},
    )
    assert x.T.coords["x"]["spacing"]["value"] == 0.5  # transpose keeps it
    reduced = x.sum(dim="y")
    assert reduced.coords["x"]["spacing"]["value"] == 0.5  # reduce other axis
    assert "x" not in x.sum(dim="x").coords  # reducing its axis drops it
    assert x.rename(x="u").coords["u"]["spacing"]["value"] == 0.5  # rename


def test_numeric_coord_attribute_sugar_does_not_shadow_dict_api():
    x = XTensor(
        torch.zeros(4), names=("x",), coords={"x": {"spacing": (2, "m")}}
    )
    sp = x.coords["x"]["spacing"]
    assert callable(sp.values)  # `values` stays the dict method...
    assert sp["value"] == 2  # ...the key is item-access only
    assert sp.unit == "m"  # safe key reachable as an attribute


def test_learnable_spacing_keeps_its_gradient():
    pytest.importorskip("pint")
    leaf = torch.tensor(0.5, requires_grad=True)
    with set_options(unit_backend="pint"):
        step = XTensor(leaf, unit="mm")
        x = XTensor(
            torch.zeros(5), names=("t",), coords={"t": {"spacing": step}}
        )
        x.coords["t"]["values"].sum().backward()
        assert leaf.grad.item() == 10.0  # d/dstep sum(i*step) = 0+1+2+3+4


def test_spacing_unitful_converts():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(
            torch.zeros(4),
            names=("x",),
            coords={"x": {"spacing": (2.0, "mm")}},
        )
        converted = x.coords["x"]["spacing"].to("um")
        assert converted["value"] == 2000.0
        assert converted["unit"] == "micrometer"


# -- explicit numeric coords + slicing + conversion (0001 phase 2) ------------


def test_explicit_numeric_coordinate_stores_positions():
    t = XTensor(torch.tensor([0.0, 0.5, 2.0, 4.0]), unit="s")
    x = XTensor(torch.arange(4.0), names=("t",), coords={"t": t})
    assert x.coords["t"]["values"].tolist() == [0.0, 0.5, 2.0, 4.0]
    assert x.coords["t"]["values"].unit == "s"  # position unit


def test_compact_coord_slices_affinely():
    x = XTensor(
        torch.arange(8.0),
        names=("x",),
        coords={"x": {"spacing": (0.5, "mm"), "origin": (0.0, "mm")}},
    )
    # an offset slice shifts origin; a strided slice scales spacing
    off = x[2:].coords["x"]["values"].tolist()
    assert off == [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    assert x[::2].coords["x"]["values"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert x[1:6:2].coords["x"]["values"].tolist() == [0.5, 1.5, 2.5]
    assert "x" not in x[3].coords  # integer index drops the axis + coord


def test_explicit_coord_slices_including_advanced():
    t = XTensor(torch.tensor([0.0, 0.5, 2.0, 4.0]), unit="s")
    x = XTensor(torch.arange(4.0), names=("t",), coords={"t": t})
    assert x[1:3].coords["t"]["values"].tolist() == [0.5, 2.0]
    assert x[[0, 2]].coords["t"]["values"].tolist() == [0.0, 2.0]


def test_compact_advanced_index_materialises_to_explicit():
    x = XTensor(
        torch.arange(8.0),
        names=("x",),
        coords={"x": {"spacing": (0.5, "mm")}},
    )
    picked = x[[1, 3]].coords["x"]
    assert not picked._compact()  # became an explicit coordinate
    assert picked["values"].tolist() == [0.5, 1.5]


def test_coordinate_converts_its_position_unit():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        compact = XTensor(
            torch.arange(4.0),
            names=("x",),
            coords={"x": {"spacing": (2.0, "mm")}},
        )
        assert compact.coords["x"].to("um")["spacing"]["value"] == 2000.0
        explicit = XTensor(
            torch.arange(3.0),
            names=("t",),
            coords={"t": XTensor(torch.tensor([1.0, 2.0, 3.0]), unit="mm")},
        )
        got = explicit.coords["t"].to("um")["values"].tolist()
        assert got == [1000.0, 2000.0, 3000.0]


def test_sel_by_numeric_coordinate_value():
    # values 0, 0.5, 1.0, ... 3.5 along t
    x = XTensor(
        torch.arange(8.0),
        names=("t",),
        coords={"t": {"spacing": (0.5, "s"), "origin": (0.0, "s")}},
    )
    assert x.sel(t=1.5).item() == 3.0  # exact value -> index 3
    assert x.sel(t=[0.5, 2.0]).tolist() == [1.0, 4.0]  # a list keeps the axis
    with pytest.raises(ValueError, match="over tolerance"):
        x.sel(t=1.7)  # bare sel is exact (tolerance 0)
    # a mode implies an unbounded snap -> nearest tick is 1.5 (index 3)
    assert x.sel(t=1.7, mode="round").item() == 3.0
    assert x.sel(t=1.7, method="nearest").item() == 3.0  # xarray alias
    with pytest.raises(ValueError, match="over tolerance"):
        x.sel(t=1.7, mode="round", tolerance=0.1)


def test_sel_modes_round_floor_ceil_prev_next():
    # ascending ticks 0,2,4,6,8 ; data 0,10,20,30,40
    x = XTensor(
        torch.arange(5.0) * 10,
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 0.0}},
    )
    assert x.sel(t=5.0, mode="floor").item() == 20.0  # value 4
    assert x.sel(t=5.0, mode="ceil").item() == 30.0  # value 6
    # ascending: prev == floor, next == ceil
    assert x.sel(t=5.0, mode="prev").item() == 20.0
    assert x.sel(t=5.0, mode="next").item() == 30.0


def test_sel_modes_split_value_vs_tickorder_on_descending():
    # descending ticks 8,6,4,2,0 ; data 0,10,20,30,40
    d = XTensor(
        torch.arange(5.0) * 10,
        names=("t",),
        coords={"t": XTensor(torch.tensor([8.0, 6.0, 4.0, 2.0, 0.0]))},
    )
    # value-space floor/ceil are orientation-robust
    assert d.sel(t=5.0, mode="floor").item() == 20.0  # value 4
    assert d.sel(t=5.0, mode="ceil").item() == 10.0  # value 6
    # tick-order prev/next SWAP vs floor/ceil on a descending coordinate
    assert d.sel(t=5.0, mode="prev").item() == 10.0  # == ceil here
    assert d.sel(t=5.0, mode="next").item() == 20.0  # == floor here


def test_sel_mode_and_method_are_exclusive_and_validated():
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 1.0, "origin": 0.0}},
    )
    with pytest.raises(ValueError, match="either 'mode' or 'method'"):
        x.sel(t=1.0, mode="round", method="nearest")
    with pytest.raises(ValueError, match="unknown mode"):
        x.sel(t=1.0, mode="bogus")


def test_sel_numeric_is_unit_aware():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(
            torch.arange(4.0),  # positions 0,1,2,3 mm
            names=("x",),
            coords={"x": {"spacing": (1.0, "mm")}},
        )
        assert x.sel(x="2000um", method="nearest").item() == 2.0  # converted
        assert x.sel(x=(2, "mm")).item() == 2.0  # a (value, unit) tuple


def test_sel_explicit_numeric_coordinate():
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": XTensor(torch.tensor([0.0, 0.5, 2.0, 4.0]), unit="s")},
    )
    assert x.sel(t=2.0).item() == 2.0
    assert x.sel(t=1.0, method="nearest").item() == 1.0  # nearest tick is 0.5


def test_interp_nearest_is_builtin_without_the_backend(monkeypatch):
    # order-0 (nearest) needs no fiery.interpol; force the backend absent.
    from fiery.xtensor import _tensors

    monkeypatch.setattr(_tensors, "_interpol", lambda: None)
    x = XTensor(  # ticks 0,2,4,6,8
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 0.0}},
    )
    got = x.interp(t=[3.0, 5.0], method="nearest")
    assert got.tolist() == [2.0, 2.0]  # round(1.5)=2, round(2.5)=2 (half-even)
    assert got.coords["t"]["values"].as_subclass(torch.Tensor).tolist() == [
        3.0,
        5.0,
    ]
    # out-of-range clamps (replicate, the default) ...
    assert x.interp(t=[-4.0, 20.0], method="nearest").tolist() == [0.0, 4.0]
    # ... or wraps with bound="wrap"
    assert x.interp(t=10.0, method="nearest", bound="wrap").item() == 0.0


def test_interp_higher_order_needs_the_backend(monkeypatch):
    from fiery.xtensor import _tensors

    monkeypatch.setattr(_tensors, "_interpol", lambda: None)
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 0.0}},
    )
    with pytest.raises(ImportError, match="fiery-xtensor\\[interp\\]"):
        x.interp(t=[1.0], method="linear")


def test_interp_linear_computes_new_values():
    pytest.importorskip("fiery.interpol")
    x = XTensor(  # value == index (ticks 0,2,4,6,8)
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 0.0}},
    )
    got = x.interp(t=[1.0, 3.0, 5.0])  # halfway ticks -> half-index values
    assert got.tolist() == [0.5, 1.5, 2.5]
    assert got.names == ("t",)
    assert got.coords["t"]["values"].as_subclass(torch.Tensor).tolist() == [
        1.0,
        3.0,
        5.0,
    ]


def test_interp_scalar_drops_the_axis():
    pytest.importorskip("fiery.interpol")
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 0.0}},
    )
    got = x.interp(t=3.0)  # a scalar query drops the axis, like sel
    assert got.ndim == 0
    assert got.item() == 1.5


def test_interp_keeps_other_axes_and_names():
    pytest.importorskip("fiery.interpol")
    x = XTensor(
        torch.arange(10.0).reshape(2, 5),
        names=("b", "t"),
        coords={"t": {"spacing": 2.0, "origin": 0.0}},
    )
    got = x.interp(t=[1.0, 3.0])
    assert got.shape == (2, 2)
    assert got.names == ("b", "t")
    assert got.tolist() == [[0.5, 1.5], [5.5, 6.5]]


def test_interp_is_unit_aware():
    pytest.importorskip("fiery.interpol")
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(
            torch.arange(5.0),
            names=("t",),
            coords={"t": {"spacing": (2.0, "s"), "origin": (0.0, "s")}},
        )
        assert x.interp(t="3000ms").item() == 1.5  # 3000 ms -> 3 s -> 1.5


def test_interp_query_gradients_flow():
    pytest.importorskip("fiery.interpol")
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 0.0}},
    )
    q = torch.tensor([3.0], requires_grad=True)
    x.interp(t=q).sum().backward()
    assert q.grad.tolist() == [0.5]  # d(value)/d(query) = 1 / spacing


def test_interp_bound_option_and_override():
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 0.0}},
    )
    with set_options(interp_bound="wrap"):
        assert x.interp(t=10.0, method="nearest").item() == 0.0  # wraps
        # a per-call bound overrides the global option
        got = x.interp(t=10.0, method="nearest", bound="replicate")
        assert got.item() == 4.0  # clamps


def test_interp_irregular_nearest_is_builtin_without_the_backend(monkeypatch):
    # order-0 (nearest) needs no fiery.interpol, same as the regular case.
    from fiery.xtensor import _tensors

    monkeypatch.setattr(_tensors, "_interpol", lambda: None)
    x = XTensor(  # value == index**2, ticks 0, 1, 4, 9 (irregular)
        torch.tensor([0.0, 1.0, 4.0, 9.0]),
        names=("t",),
        coords={"t": torch.tensor([0.0, 1.0, 4.0, 9.0])},
    )
    got = x.interp(t=[2.5, 8.0], method="nearest")
    assert got.tolist() == [4.0, 9.0]  # nearest tick to 2.5 is 4, to 8 is 9
    assert got.coords["t"]["values"].as_subclass(torch.Tensor).tolist() == [
        2.5,
        8.0,
    ]


def test_interp_irregular_linear_computes_new_values():
    pytest.importorskip("fiery.interpol")
    # data 0, 10, 20, 30 at irregular ticks 0, 1, 4, 9 -- deliberately *not*
    # the identity map, so a bracket-and-interpolate bug can't hide behind
    # "the query happened to equal the answer".
    x = XTensor(
        torch.tensor([0.0, 10.0, 20.0, 30.0]),
        names=("t",),
        coords={"t": torch.tensor([0.0, 1.0, 4.0, 9.0])},
    )
    # 0.5 is halfway between ticks 0 and 1 (positions 0 and 1) -> data 0..10
    # 2.5 is 1.5/3 of the way between ticks 1 and 4 (positions 1 and 2)
    got = x.interp(t=[0.5, 2.5])
    assert got.tolist() == pytest.approx([5.0, 15.0])
    assert got.coords["t"]["values"].as_subclass(torch.Tensor).tolist() == [
        0.5,
        2.5,
    ]
    # exact match on a tick is exact, not just "close"
    assert x.interp(t=4.0).item() == pytest.approx(20.0)
    # out-of-range clamps to the edge value (the "replicate" default bound,
    # same as the regular-coordinate case) -- the fractional index itself
    # still extrapolates past the end segment's slope (frac -1 and 3.2 for
    # these queries), it's `bound` that then clamps it into range.
    got_extrap = x.interp(t=[-1.0, 10.0])
    assert got_extrap.tolist() == pytest.approx([0.0, 30.0])


def test_interp_irregular_descending_coordinate():
    pytest.importorskip("fiery.interpol")
    x = XTensor(
        torch.tensor([0.0, 10.0, 20.0, 30.0]),
        names=("t",),
        coords={"t": torch.tensor([9.0, 4.0, 1.0, 0.0])},  # descending
    )
    # 2.5 is 1.5/3 of the way between ticks 4 (index 1) and 1 (index 2)
    assert x.interp(t=2.5).item() == pytest.approx(15.0)
    assert x.interp(t=9.0).item() == pytest.approx(0.0)
    assert x.interp(t=0.0).item() == pytest.approx(30.0)


def test_interp_irregular_coordinate_must_be_monotonic():
    x = XTensor(
        torch.zeros(4),
        names=("t",),
        coords={"t": torch.tensor([0.0, 2.0, 1.0, 3.0])},
    )
    with pytest.raises(ValueError, match="strictly monotonic"):
        x.interp(t=1.5)


def test_interp_irregular_higher_order_is_not_implemented():
    # higher orders need a true non-uniform spline (a uniform-index-space
    # spline basis isn't cubic-in-value on non-uniform nodes); see #81.
    x = XTensor(
        torch.tensor([0.0, 1.0, 4.0, 9.0]),
        names=("t",),
        coords={"t": torch.tensor([0.0, 1.0, 4.0, 9.0])},
    )
    with pytest.raises(NotImplementedError, match="#81"):
        x.interp(t=2.5, method="cubic")


def test_interp_irregular_gradients_flow_through_query_and_values():
    pytest.importorskip("fiery.interpol")
    values = torch.tensor([0.0, 1.0, 4.0, 9.0], requires_grad=True)
    x = XTensor(
        torch.tensor([0.0, 1.0, 4.0, 9.0]), names=("t",), coords={"t": values}
    )
    q = torch.tensor(2.5, requires_grad=True)
    x.interp(t=q).backward()
    assert q.grad is not None and q.grad.item() != 0.0
    assert values.grad is not None
    # only the bracketing ticks (indices 1 and 2, ticks 1 and 4) get a
    # nonzero gradient -- the query never touches ticks 0 or 3
    assert values.grad[0].item() == 0.0
    assert values.grad[3].item() == 0.0
    assert values.grad[1].item() != 0.0
    assert values.grad[2].item() != 0.0


def test_interp_irregular_gradcheck_against_finite_differences():
    # "gradients flow" is not the same as "gradients are right": check the
    # analytic gradient of the whole value -> frac -> pull chain numerically.
    pytest.importorskip("fiery.interpol")
    data = torch.tensor([3.0, -1.0, 7.0, 2.0, 11.0], dtype=torch.float64)

    def pull(vals, query):
        x = XTensor(data, names=("t",), coords={"t": vals})
        return x.interp(t=query).as_subclass(torch.Tensor)

    ticks = torch.tensor(
        [0.0, 1.0, 4.0, 9.0, 25.0], dtype=torch.float64, requires_grad=True
    )
    # in-range brackets only (a query sitting exactly on a tick is a kink of
    # the piecewise-linear map, where a one-sided derivative is expected)
    q = torch.tensor([0.5, 2.5, 12.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(pull, (ticks, q), eps=1e-6, atol=1e-6)
    # ... and the same for a descending coordinate (the flipped search path)
    down = torch.tensor(
        [25.0, 9.0, 4.0, 1.0, 0.0], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(pull, (down, q), eps=1e-6, atol=1e-6)


def test_interp_irregular_matches_the_regular_path_when_ticks_are_uniform():
    # an explicit coordinate that *happens* to be uniform must agree with the
    # compact one describing the same ticks -- bit for bit, for every bound.
    pytest.importorskip("fiery.interpol")
    data = torch.tensor([3.0, -1.0, 7.0, 2.0, 11.0])
    query = [-3.0, 0.25, 1.7, 3.9, 4.5, 11.0]  # ticks are 1, 3, 5, 7, 9
    explicit = XTensor(
        data, names=("t",), coords={"t": torch.arange(5.0) * 2.0 + 1.0}
    )
    compact = XTensor(
        data, names=("t",), coords={"t": {"spacing": 2.0, "origin": 1.0}}
    )
    for method in ("nearest", "linear"):
        got = explicit.interp(t=query, method=method)
        want = compact.interp(t=query, method=method)
        assert got.tolist() == want.tolist()
    for bound in ("replicate", "zero", "dft"):
        got = explicit.interp(t=query, bound=bound)
        want = compact.interp(t=query, bound=bound)
        assert got.tolist() == want.tolist()


def test_interp_irregular_matches_piecewise_linear_over_a_batch():
    # one query per bracket plus both out-of-range ends, all in one call --
    # the brackets differ across the batch, so a non-vectorised (or
    # first-bracket-for-everyone) inversion would show up here.
    pytest.importorskip("fiery.interpol")
    ticks = [0.0, 1.0, 4.0, 9.0, 25.0]
    data = [3.0, -1.0, 7.0, 2.0, 11.0]
    x = XTensor(
        torch.tensor(data), names=("t",), coords={"t": torch.tensor(ticks)}
    )
    query = [0.5, 2.5, 6.5, 17.0]
    expected = [1.0, 3.0, 4.5, 6.5]  # hand-computed segment midpoints
    assert x.interp(t=query).tolist() == pytest.approx(expected)
    # every tick reproduces its own value exactly (not just "close")
    assert x.interp(t=ticks).tolist() == data
    # both ends clamp under the default "replicate" bound
    assert x.interp(t=[-100.0, 100.0]).tolist() == pytest.approx([3.0, 11.0])


def test_interp_irregular_descending_out_of_range_both_ends():
    pytest.importorskip("fiery.interpol")
    x = XTensor(
        torch.tensor([11.0, 2.0, 7.0, -1.0, 3.0]),
        names=("t",),
        coords={"t": torch.tensor([25.0, 9.0, 4.0, 1.0, 0.0])},
    )
    # past the *high* end is index 0, past the *low* end is index -1
    got = x.interp(t=[-100.0, 0.5, 17.0, 100.0])
    assert got.tolist() == pytest.approx([3.0, 1.0, 6.5, 11.0])


def test_interp_irregular_two_point_coordinate():
    pytest.importorskip("fiery.interpol")
    x = XTensor(  # the smallest coordinate an inversion can bracket
        torch.tensor([1.0, 5.0]),
        names=("t",),
        coords={"t": torch.tensor([0.0, 4.0])},
    )
    got = x.interp(t=[-1.0, 0.0, 2.0, 4.0, 6.0])
    assert got.tolist() == pytest.approx([1.0, 1.0, 3.0, 5.0, 5.0])


def test_interp_irregular_is_unit_aware():
    pytest.importorskip("fiery.interpol")
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(
            torch.tensor([0.0, 10.0, 20.0, 30.0]),
            names=("t",),
            coords={
                "t": XTensor(torch.tensor([0.0, 1.0, 4.0, 9.0]), unit="s")
            },
        )
        assert x.interp(t="2500ms").item() == pytest.approx(15.0)
        got = x.interp(t=["500ms", "2500ms"])
        assert got.tolist() == pytest.approx([5.0, 15.0])
        # the new coordinate carries the *position* unit, not the query's
        coord = got.coords["t"]["values"]
        assert coord.unit == "second"
        assert coord.as_subclass(torch.Tensor).tolist() == [0.5, 2.5]


def test_interp_irregular_higher_order_needs_the_backend(monkeypatch):
    # the backend guard is the same one the regular path uses: order >= 1
    # still needs fiery.interpol on an irregular coordinate.
    from fiery.xtensor import _tensors

    monkeypatch.setattr(_tensors, "_interpol", lambda: None)
    x = XTensor(
        torch.tensor([0.0, 1.0, 4.0, 9.0]),
        names=("t",),
        coords={"t": torch.tensor([0.0, 1.0, 4.0, 9.0])},
    )
    with pytest.raises(ImportError, match="fiery-xtensor\\[interp\\]"):
        x.interp(t=[2.5], method="linear")


def test_interp_irregular_repeated_ticks_name_the_offending_pair():
    # a repeated tick is a zero-width cell: there is no inverse, and it is
    # easy to hit by accident (a float32 cumsum of many small steps), so the
    # error says *where* rather than just "not monotonic".
    x = XTensor(
        torch.zeros(4),
        names=("t",),
        coords={"t": torch.tensor([0.0, 1.0, 1.0, 2.0])},
    )
    with pytest.raises(ValueError, match="ticks 1 and 2 are 1.0 and 1.0"):
        x.interp(t=1.5)


def test_interp_irregular_needs_at_least_two_ticks():
    x = XTensor(
        torch.tensor([5.0]), names=("t",), coords={"t": torch.tensor([2.0])}
    )
    with pytest.raises(ValueError, match="at least 2 points"):
        x.interp(t=2.0)


def test_interp_irregular_needs_a_1d_coordinate():
    # `coords={dim: <tensor>}` accepts any tensor whose *first* axis matches,
    # so a 2-D one is storable; interp must say so rather than fall through
    # to an opaque searchsorted shape error.
    x = XTensor(
        torch.arange(6.0).reshape(3, 2),
        names=("t", "u"),
        coords={"t": torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])},
    )
    with pytest.raises(ValueError, match="must be 1-D"):
        x.interp(t=1.5)


def test_interp_irregular_on_a_sliced_coordinate():
    # `x[::2]` leaves the coordinate a strided view; the inversion must still
    # bracket against the *sliced* ticks.
    pytest.importorskip("fiery.interpol")
    x = XTensor(
        torch.tensor([3.0, -1.0, 7.0, 2.0, 11.0]),
        names=("t",),
        coords={"t": torch.tensor([0.0, 1.0, 4.0, 9.0, 25.0])},
    )
    sliced = x[::2]  # ticks 0, 4, 25 / data 3, 7, 11
    assert sliced.coords["t"]["values"].as_subclass(torch.Tensor).tolist() == [
        0.0,
        4.0,
        25.0,
    ]
    assert sliced.interp(t=2.0).item() == pytest.approx(5.0)


def test_interp_needs_a_numeric_coordinate():
    x = _labelled()
    with pytest.raises(ValueError, match="no numeric coordinate"):
        x.interp(col=1.0)


def test_sel_selects_a_labelled_position_and_drops_the_axis():
    x = _labelled()
    y = x.sel(col="y")
    assert y.shape == (3,)
    assert y.names == ("row",)
    assert torch.equal(y, x.as_subclass(torch.Tensor)[:, 2])


def test_sel_with_a_list_of_labels_keeps_the_axis():
    x = _labelled()
    y = x.sel(col=["w", "y"])
    assert y.shape == (3, 2)
    assert y.coords == {"col": ("w", "y")}


def test_sel_unknown_label_or_dim_raises():
    x = _labelled()
    with pytest.raises(ValueError, match="no label 'nope'"):
        x.sel(col="nope")
    with pytest.raises(ValueError, match="no coordinates"):
        x.sel(row="anything")


def test_isel_selects_by_integer_position_along_a_named_dim():
    x = _labelled()
    assert x.isel(col=slice(1, 3)).coords == {"col": ("x", "y")}
    assert x.isel(row=0).names == ("col",)


def test_attribute_access_selects_a_single_label():
    x = _labelled()
    out = x.x  # label "x" on dim "col"
    assert out.shape == (3,)
    assert torch.equal(out, x.as_subclass(torch.Tensor)[:, 1])


def test_attribute_access_can_be_chained_across_dims():
    x = XTensor(
        torch.arange(6).reshape(2, 3),
        names=("row", "col"),
        coords={"row": ("r0", "r1"), "col": ("c0", "c1", "c2")},
    )
    out = x.r1.c2
    assert out.ndim == 0
    assert out.item() == 5


def test_attribute_access_unknown_label_raises():
    with pytest.raises(AttributeError):
        _ = _labelled().nope


def test_getitem_slices_the_labels_of_kept_axes():
    x = _labelled()
    y = x[:, 1:3]
    assert y.coords == {"col": ("x", "y")}
    # an integer index drops the axis and its labels
    assert x[:, 1].coords == {}


def test_getitem_resolves_a_positional_label_against_its_axis():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("b", "row", "col"),
        coords={"row": ("r0", "r1", "r2"), "col": ("w", "x", "y", "z")},
    )
    # a bare label indexes the axis it sits on (like an int there): drops it
    out = x[:, "r1", "y"]
    assert out.names == ("b",)
    assert torch.equal(out, x[:, 1, 2])


def test_getitem_labels_address_trailing_axes_through_ellipsis():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("b", "row", "col"),
        coords={"row": ("r0", "r1", "r2"), "col": ("w", "x", "y", "z")},
    )
    # ellipsis fills the leading axes; the labels address the last two
    assert torch.equal(x[..., "r2", "z"], x[:, 2, 3])
    assert x[..., "r2", "z"].names == ("b",)
    # a label mixes with ints and slices
    assert torch.equal(x[0, "r1", :], x[0, 1, :])


def test_getitem_list_of_labels_is_an_advanced_index_keeping_the_axis():
    x = _labelled()  # names ("row", "col"), coords col=("w","x","y","z")
    out = x[:, ["w", "y"]]
    assert out.names == ("row", "col")
    assert out.coords == {"col": ("w", "y")}


def test_getitem_top_level_label_tuple_is_one_label_per_axis():
    x = XTensor(
        torch.arange(6).reshape(2, 3),
        names=("row", "col"),
        coords={"row": ("r0", "r1"), "col": ("c0", "c1", "c2")},
    )
    # x["r1", "c2"] -> label per axis, not a single advanced index
    assert x["r1", "c2"].item() == 5


def test_getitem_label_on_an_unlabelled_or_missing_axis_raises():
    x = _labelled()  # only "col" is labelled
    with pytest.raises(KeyError, match="no coordinates"):
        _ = x["w"]  # axis 0 ("row") has no coordinates
    with pytest.raises(KeyError, match="no label"):
        _ = x[:, "nope"]


# ----------------------------------------------------------------------
# structured coordinates (dict-valued labels + field queries)
# ----------------------------------------------------------------------


def _channels():
    return XTensor(
        torch.arange(6).reshape(3, 2).float(),
        names=("c", "x"),
        coords={
            "c": [
                {"name": "DAPI", "type": "nucleus"},
                {"name": "GFP", "type": "signal"},
                {"name": "RFP", "type": "signal"},
            ]
        },
    )


def test_structured_labels_select_by_name_everywhere():
    img = _channels()
    expected = img.as_subclass(torch.Tensor)[1]
    assert torch.equal(img.sel(c="GFP"), expected)  # keyword
    assert torch.equal(img.GFP, expected)  # attribute
    assert torch.equal(img["GFP"], expected)  # positional string
    assert img.sel(c="GFP").names == ("x",)  # a single name drops the axis


def test_structured_query_selects_contiguous_matches_as_a_slice():
    img = _channels()
    sig = img[{"type": "signal"}]  # positions 1,2 -> a slice, keeps the axis
    assert sig.names == ("c", "x")
    assert sig.shape == (2, 2)
    assert [label["name"] for label in sig.coords["c"]] == ["GFP", "RFP"]
    # the sel keyword spelling is equivalent
    assert torch.equal(img.sel(c={"type": "signal"}), sig)


def test_structured_query_selects_non_contiguous_matches_as_a_list():
    img = XTensor(
        torch.arange(6).reshape(3, 2).float(),
        names=("c", "x"),
        coords={
            "c": [
                {"name": "A", "type": "signal"},
                {"name": "B", "type": "nucleus"},
                {"name": "C", "type": "signal"},
            ]
        },
    )
    out = img[{"type": "signal"}]  # positions 0,2 -> advanced index
    assert [label["name"] for label in out.coords["c"]] == ["A", "C"]
    assert out.shape == (2, 2)


def test_structured_query_composes_positionally_and_can_be_empty():
    vol = XTensor(
        torch.zeros(4, 3, 2),
        names=("z", "c", "x"),
        coords={
            "c": [
                {"name": "A", "type": "s"},
                {"name": "B", "type": "s"},
                {"name": "C", "type": "t"},
            ]
        },
    )
    # a query addresses the axis it sits on, mixing with an int index
    assert vol[0, {"type": "s"}].names == ("c", "x")
    assert vol[0, {"type": "s"}].shape == (2, 2)
    # a query matching nothing gives a size-0 axis (no error)
    assert vol[:, {"type": "nope"}].shape == (4, 0, 2)


def test_permute_carries_coordinates_unchanged():
    x = _labelled()
    assert x.T.coords == {"col": ("w", "x", "y", "z")}
    assert x.transpose(0, 1).coords == {"col": ("w", "x", "y", "z")}


def test_coords_getter_hides_stale_metadata():
    # A shape-changing op without coordinate handling must not report labels
    # of the wrong length; the getter guards this.
    x = _labelled()
    y = x.clone()
    y._coords = {"col": ("w", "x")}  # wrong length for size-4 axis
    assert y.coords == {}


# ----------------------------------------------------------------------
# renaming
# ----------------------------------------------------------------------


def test_rename_out_of_place_sets_and_clears_names():
    x = XTensor(torch.zeros(2, 3), names=("row", "col"))
    y = x.rename("a", "b")
    assert y.names == ("a", "b")
    assert x.names == ("row", "col")  # out-of-place: x unchanged
    assert y.rename(None).names == (None, None)
    plain = torch.Tensor
    assert torch.equal(y.as_subclass(plain), x.as_subclass(plain))


def test_rename_by_map():
    x = XTensor(torch.zeros(2, 3), names=("row", "col"))
    assert x.rename(col="C").names == ("row", "C")
    with pytest.raises(ValueError, match="no axis named"):
        x.rename(nope="X")


def test_rename_in_place_returns_self():
    x = XTensor(torch.zeros(2, 3), names=("row", "col"))
    out = x.rename_("a", "b")
    assert out is x
    assert x.names == ("a", "b")


def test_rename_moves_coordinates_to_the_new_dim_name():
    x = _labelled()
    # positional rename sets the whole name tuple; the labelled axis keeps its
    # name here, so the coordinates stay under "col"
    y = x.rename("batch", "col")
    assert y.coords == {"col": ("w", "x", "y", "z")}
    # renaming the labelled axis moves its coordinates to the new name
    z = x.rename(col="feature")
    assert z.coords == {"feature": ("w", "x", "y", "z")}


# ----------------------------------------------------------------------
# builtin named-tensor API, re-implemented self-managed
# ----------------------------------------------------------------------


def test_refine_names_only_names_unnamed_axes():
    x = XTensor(torch.zeros(2, 3, 4), names=(None, "b", None))
    assert x.refine_names("a", "b", "c").names == ("a", "b", "c")
    assert x.refine_names(None, "b", None).names == (None, "b", None)


def test_refine_names_ellipsis_keeps_spanned_names():
    x = XTensor(torch.zeros(2, 3, 4, 5), names=(None, "b", "c", None))
    assert x.refine_names("a", ..., "d").names == ("a", "b", "c", "d")


def test_names_ellipsis_fills_the_middle_with_unnamed_axes():
    # a single `...` in `names=` stands for a run of unnamed (None) axes
    t = torch.zeros(2, 3, 4, 5)
    assert XTensor(t, names=("b", ..., "x")).names == ("b", None, None, "x")
    assert XTensor(t, names=(..., "x")).names == (None, None, None, "x")
    assert XTensor(t, names=("b", ...)).names == ("b", None, None, None)
    assert XTensor(t, names=(...,)).names == (None,) * 4


def test_names_setter_accepts_ellipsis():
    x = XTensor(torch.zeros(2, 3, 4))
    x.names = ("b", ..., "w")
    assert x.names == ("b", None, "w")


def test_names_ellipsis_with_a_descriptor_dict():
    x = XTensor(
        torch.zeros(2, 3, 4),
        axes=[{"name": "b", "type": "batch"}, ..., "w"],
    )
    assert x.names == ("b", None, "w")
    assert x.axes[0] == {"name": "b", "type": "batch"}


def test_names_ellipsis_rejects_more_than_one_and_overflow():
    t = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="only one '...'"):
        XTensor(t, names=("a", ..., "b", ...))
    with pytest.raises(ValueError, match="too many names"):
        XTensor(t, names=("a", "b", "c", ...))


def test_rename_ellipsis_keeps_the_spanned_names():
    # `rename` modifies, so `...` leaves the spanned axes unchanged
    x = XTensor(
        torch.zeros(2, 3, 4, 5),
        names=("b", "c", "h", "w"),
        coords={"w": ("a", "b", "c", "d", "e")},
    )
    renamed = x.rename("B", ..., "W")
    assert renamed.names == ("B", "c", "h", "W")
    assert renamed.coords == {"W": ("a", "b", "c", "d", "e")}


def test_permute_ellipsis_stands_for_the_remaining_axes():
    x = XTensor(
        torch.zeros(2, 3, 4, 5),
        names=("b", "c", "h", "w"),
        coords={"w": ("a", "b", "c", "d", "e")},
    )
    assert x.permute("w", ...).names == ("w", "b", "c", "h")
    assert x.permute(..., "b").names == ("c", "h", "w", "b")
    assert x.permute("w", ...).coords == {"w": ("a", "b", "c", "d", "e")}


def test_refine_names_rejects_renaming_a_named_axis():
    x = XTensor(torch.zeros(2, 3), names=("a", "b"))
    with pytest.raises(ValueError, match="cannot rename"):
        x.refine_names("a", "X")


def test_align_to_permutes_into_the_given_order():
    x = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    y = x.align_to("c", "a", "b")
    assert y.names == ("c", "a", "b")
    assert y.shape == (4, 2, 3)
    assert x.align_to(..., "a").names == ("b", "c", "a")


def test_align_as_reorders_and_inserts_size_one_axes():
    x = XTensor(torch.zeros(4, 2), names=("c", "a"))
    other = XTensor(torch.zeros(2, 3, 4, 5), names=("a", "b", "c", "d"))
    y = x.align_as(other)
    assert y.names == ("a", "b", "c", "d")
    assert y.shape == (2, 1, 4, 1)


def test_align_as_requires_every_axis_present_in_target():
    other = XTensor(torch.zeros(2, 3), names=("a", "b"))
    with pytest.raises(ValueError, match="not in the target"):
        XTensor(torch.zeros(2), names=("q",)).align_as(other)


@pytest.mark.skipif(
    not hasattr(torch.Tensor, "names"),
    reason="builtin named-tensor API not present in this torch build",
)
def test_names_do_not_use_builtin_named_tensors():
    # The self-managed names must not set the underlying tensor's builtin
    # (C-level) names -- that is what keeps us portable across torch versions.
    x = XTensor(torch.zeros(2, 3), names=("row", "col"))
    assert torch.Tensor.names.__get__(x) == (None, None)


def test_torch_func_returns_none_for_missing_op():
    assert _torch_func("permute") is not None
    assert _torch_func("this_op_does_not_exist_anywhere") is None


# ----------------------------------------------------------------------
# functional-form (torch.op(x, ...)) metadata parity
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(torch, "permute"),
    reason="torch.permute (functional) was added in torch 1.9",
)
def test_functional_permute_matches_method_form():
    x = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    assert torch.permute(x, (2, 0, 1)).names == x.permute(2, 0, 1).names
    assert torch.permute(x, (2, 0, 1)).names == ("c", "a", "b")


def test_functional_unsqueeze_squeeze_match_method_form():
    x = XTensor(torch.zeros(2, 3), names=("row", "col"))
    assert torch.unsqueeze(x, 1).names == ("row", None, "col")
    y = XTensor(torch.zeros(1, 3), names=("s", "col"))
    assert torch.squeeze(y).names == ("col",)


@pytest.mark.skipif(
    not hasattr(torch, "permute"),
    reason="torch.permute (functional) was added in torch 1.9",
)
def test_functional_permute_carries_coordinates():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("a", "b", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    y = torch.permute(x, (2, 0, 1))
    assert y.names == ("c", "a", "b")
    assert y.coords == {"c": ("w", "x", "y", "z")}


# ----------------------------------------------------------------------
# reshape / reorder op family (axis names)
# ----------------------------------------------------------------------


def test_transpose_family_swaps_axis_names():
    x = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    assert x.transpose(0, 2).names == ("c", "b", "a")
    assert torch.transpose(x, 0, 2).names == ("c", "b", "a")


@pytest.mark.skipif(
    not _HAS_SWAPAXES, reason="torch.swapaxes/swapdims added in torch 1.8"
)
def test_swapaxes_family_swaps_axis_names():
    x = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    assert x.swapaxes(0, 1).names == ("b", "a", "c")
    assert x.swapdims(1, 2).names == ("a", "c", "b")


def test_mT_transposes_last_two_axis_names():
    x = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    assert x.mT.names == ("a", "c", "b")
    assert x.mT.shape == (2, 4, 3)


def test_movedim_reorders_axis_names_like_torch():
    x = XTensor(torch.zeros(2, 3, 4, 5), names=("a", "b", "c", "d"))
    y = x.movedim(0, 2)
    assert y.names == ("b", "c", "a", "d")
    assert y.shape == tuple(torch.movedim(torch.zeros(2, 3, 4, 5), 0, 2).shape)


@pytest.mark.skipif(
    not _HAS_MOVEAXIS, reason="torch.moveaxis added in torch 1.8"
)
def test_moveaxis_reorders_axis_names_like_torch():
    x = XTensor(torch.zeros(2, 3, 4, 5), names=("a", "b", "c", "d"))
    assert x.moveaxis((0, 1), (2, 3)).names == ("c", "d", "a", "b")


def test_reshape_uses_same_name_rule_as_view():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("b", "h", "w"))
    assert x.reshape(2, 12).names == ("b", None)
    assert x.reshape(6, 4).names == (None, "w")
    assert torch.reshape(x, (2, -1)).names == ("b", None)


# ----------------------------------------------------------------------
# name-as-dim: a name may stand in for an integer `dim=`
# ----------------------------------------------------------------------


def test_permute_accepts_axis_names():
    x = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    y = x.permute("c", "a", "b")
    assert y.names == ("c", "a", "b")
    assert y.shape == (4, 2, 3)
    assert y.shape == x.permute(2, 0, 1).shape


def test_transpose_family_accepts_axis_names():
    # Name-as-dim is a method-form feature: the functional form
    # (`torch.transpose(x, "a", ...)`) cannot be relied on because newer
    # PyTorch rejects a non-int dim at the C dispatcher before
    # `__torch_function__` runs.
    x = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    assert x.transpose("a", "c").names == ("c", "b", "a")


@pytest.mark.skipif(
    not _HAS_SWAPAXES, reason="torch.swapaxes/swapdims added in torch 1.8"
)
def test_swapaxes_family_accepts_axis_names():
    x = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    assert x.swapaxes("a", "b").names == ("b", "a", "c")
    assert x.swapdims("b", "c").names == ("a", "c", "b")


def test_movedim_accepts_axis_names_for_source():
    x = XTensor(torch.zeros(2, 3, 4, 5), names=("a", "b", "c", "d"))
    assert x.movedim("a", 2).names == ("b", "c", "a", "d")


@pytest.mark.skipif(
    not _HAS_MOVEAXIS, reason="torch.moveaxis added in torch 1.8"
)
def test_moveaxis_accepts_axis_names_for_source():
    x = XTensor(torch.zeros(2, 3, 4, 5), names=("a", "b", "c", "d"))
    assert x.moveaxis(("a", "b"), (2, 3)).names == ("c", "d", "a", "b")


def test_squeeze_accepts_axis_name():
    x = XTensor(torch.ones(2, 1, 3), names=("a", "one", "b"))
    y = x.squeeze("one")
    assert y.names == ("a", "b")
    assert y.shape == (2, 3)


def test_index_select_accepts_axis_name_and_slices_coords():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("batch", "feat", "chan"),
        coords={"chan": ("w", "x", "y", "z")},
    )
    by_name = x.index_select("chan", torch.tensor([0, 2]))
    by_int = x.index_select(2, torch.tensor([0, 2]))
    assert by_name.shape == by_int.shape == (2, 3, 2)
    assert by_name.coords == by_int.coords == {"chan": ("w", "y")}


def test_name_as_dim_unknown_name_raises():
    x = XTensor(torch.zeros(2, 3), names=("a", "b"))
    with pytest.raises(ValueError, match="no axis named 'z'"):
        x.transpose("a", "z")


# ----------------------------------------------------------------------
# reductions: drop / keep the reduced axis' name (and its coords)
# ----------------------------------------------------------------------


def test_sum_drops_reduced_axis_name():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.sum(dim="b").names == ("a", "c")
    assert x.sum(dim="b").shape == (2, 4)
    assert x.sum(dim=1).names == ("a", "c")  # int still works


def test_sum_keepdim_preserves_reduced_axis_name():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    y = x.sum(dim="b", keepdim=True)
    assert y.names == ("a", "b", "c")
    assert y.shape == (2, 1, 4)


def test_reduction_over_multiple_named_axes():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.mean(dim=("a", "c")).names == ("b",)


def test_reduce_all_yields_scalar_with_no_names():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    s = x.sum()
    assert s.names == ()
    assert s.ndim == 0


def test_functional_reduction_carries_names_like_method():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert torch.mean(x, 1).names == x.mean(dim=1).names == ("a", "c")


def test_argmax_and_amax_track_names():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.amax(dim="c").names == ("a", "b")
    assert x.argmax(dim="a").names == ("b", "c")


def test_negative_dim_reduction():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.sum(dim=-1).names == ("a", "b")


def test_reduction_drops_coordinates_of_the_reduced_axis():
    x = XTensor(
        torch.arange(24.0).reshape(2, 3, 4),
        names=("a", "b", "c"),
        coords={"b": ("p", "q", "r"), "c": ("w", "x", "y", "z")},
    )
    # reduce the axis carrying the "b" labels
    r = x.sum(dim="b")
    assert r.names == ("a", "c")
    assert r.coords == {"c": ("w", "x", "y", "z")}
    # reduce a plain axis: the other labels are unchanged
    assert x.sum(dim="a").coords == {
        "b": ("p", "q", "r"),
        "c": ("w", "x", "y", "z"),
    }


# ----------------------------------------------------------------------
# irregular / (values, indices) reducers
# ----------------------------------------------------------------------


def _named():
    return XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))


def test_std_var_reduce_and_accept_a_name():
    x = _named()
    assert x.std(dim="b").names == ("a", "c")
    assert x.var(dim="c").names == ("a", "b")
    # a positional bool is `unbiased`, not a dim -> reduces everything
    assert x.std(False).names == ()
    assert x.std(False).ndim == 0


def test_norm_reduces_the_named_dim():
    x = _named()
    assert x.norm(dim="b").names == ("a", "c")
    assert torch.norm(x, 2, 1).names == ("a", "c")


def test_max_min_return_named_values_and_indices():
    x = _named()
    out = x.max(dim="b")
    assert out.values.names == out.indices.names == ("a", "c")
    assert x.max().names == ()  # scalar form
    # the two-tensor (elementwise) form reconciles names, keeps rank
    assert torch.max(x, x * 2).names == ("a", "b", "c")


def test_median_mode_kthvalue_track_names():
    x = _named()
    assert torch.median(x).names == ()  # global median -> scalar
    assert torch.median(x, 1).values.names == ("a", "c")
    assert x.mode(dim="b").values.names == ("a", "c")
    assert x.kthvalue(2, dim="b").values.names == ("a", "c")


def test_sort_preserves_names_and_drops_the_sorted_coordinate():
    x = XTensor(
        torch.randn(2, 3, 4),
        names=("a", "b", "c"),
        coords={"b": ("p", "q", "r"), "c": ("w", "x", "y", "z")},
    )
    s = x.sort(dim="c")
    assert s.values.names == s.indices.names == ("a", "b", "c")
    # the sorted axis' labels no longer match positions -> dropped
    assert s.values.coords == {"b": ("p", "q", "r")}


def test_topk_keeps_names_resizes_and_drops_the_dim_coordinate():
    x = XTensor(
        torch.randn(2, 3, 4),
        names=("a", "b", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    t = x.topk(2, dim="c")
    assert t.values.names == ("a", "b", "c")
    assert t.values.shape == (2, 3, 2)
    assert t.values.coords == {}


# ----------------------------------------------------------------------
# scans: cumsum / cumprod / softmax / log_softmax / logcumsumexp / cummax /
# cummin -- dimension-preserving (rank, sizes, names, coords all unchanged)
# ----------------------------------------------------------------------


def test_cumsum_keeps_names_and_accepts_a_name_dim():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    y = x.cumsum("a")
    assert isinstance(y, XTensor)
    assert y.names == ("a", "b", "c")
    assert y.shape == (2, 3, 4)
    assert y[1, 0, 0].item() == x[0, 0, 0].item() + x[1, 0, 0].item()
    assert x.cumsum(1).names == ("a", "b", "c")  # int still works


def test_functional_softmax_keeps_names():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    y = torch.softmax(x, 1)
    assert isinstance(y, XTensor)
    assert y.names == ("a", "b", "c")
    assert y.shape == (2, 3, 4)


def test_log_softmax_keeps_names():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.log_softmax("c").names == ("a", "b", "c")


@pytest.mark.skipif(
    not hasattr(torch, "logcumsumexp"),
    reason="logcumsumexp added in a newer torch",
)
def test_logcumsumexp_keeps_names():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.logcumsumexp("b").names == ("a", "b", "c")


def test_cumsum_keeps_the_labelled_dims_coordinates():
    x = XTensor(
        torch.arange(24.0).reshape(2, 3, 4),
        names=("a", "b", "c"),
        coords={"b": ("p", "q", "r"), "c": ("w", "x", "y", "z")},
    )
    y = x.cumsum("b")
    assert y.names == ("a", "b", "c")
    assert y.coords == {"b": ("p", "q", "r"), "c": ("w", "x", "y", "z")}


@pytest.mark.skipif(
    not hasattr(torch, "cummax"), reason="cummax added in a newer torch"
)
def test_cummax_keeps_names_and_coords_on_both_members():
    x = XTensor(
        torch.arange(24.0).reshape(2, 3, 4),
        names=("a", "b", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    out = x.cummax("a")
    assert out.values.names == out.indices.names == ("a", "b", "c")
    assert out.values.shape == out.indices.shape == (2, 3, 4)
    assert out.values.coords == {"c": ("w", "x", "y", "z")}
    assert out.indices.coords == {"c": ("w", "x", "y", "z")}


# ----------------------------------------------------------------------
# slice / split ops: select / narrow / unbind / split / chunk / flip / roll
# ----------------------------------------------------------------------


def test_select_drops_axis_name():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.select("b", 0).names == ("a", "c")
    assert x.select("b", 0).shape == (2, 4)
    assert torch.select(x, 1, 0).names == ("a", "c")


def test_narrow_keeps_names_and_accepts_name_dim():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    y = x.narrow("c", 1, 2)
    assert y.names == ("a", "b", "c")
    assert y.shape == (2, 3, 2)


def test_unbind_returns_pieces_without_the_unbound_axis():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    pieces = x.unbind("a")
    assert len(pieces) == 2
    assert all(p.names == ("b", "c") for p in pieces)
    assert all(p.shape == (3, 4) for p in pieces)


def test_split_and_chunk_keep_all_names():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    parts = x.split(2, dim="c")
    assert [p.shape for p in parts] == [(2, 3, 2), (2, 3, 2)]
    assert all(p.names == ("a", "b", "c") for p in parts)
    chunks = x.chunk(2, dim="b")
    assert [p.shape for p in chunks] == [(2, 2, 4), (2, 1, 4)]
    assert all(p.names == ("a", "b", "c") for p in chunks)


def test_flip_and_roll_preserve_names():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.flip(("a", "c")).names == ("a", "b", "c")
    assert x.roll(1, dims="b").names == ("a", "b", "c")
    assert x.roll(2).names == ("a", "b", "c")  # flattened roll


def test_select_drops_the_coordinates_of_the_selected_axis():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("a", "b", "c"),
        coords={"b": ("p", "q", "r"), "c": ("w", "x", "y", "z")},
    )
    r = x.select("b", 0)
    assert r.names == ("a", "c")
    assert r.coords == {"c": ("w", "x", "y", "z")}


def test_narrow_and_split_slice_coordinates():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("a", "b", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    assert x.narrow("c", 1, 2).coords == {"c": ("x", "y")}
    parts = x.split(2, dim="c")
    assert [p.coords for p in parts] == [{"c": ("w", "x")}, {"c": ("y", "z")}]


def test_flip_reverses_coordinates_on_the_flipped_axis():
    x = XTensor(
        torch.arange(12).reshape(3, 4),
        names=("a", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    assert x.flip("c").coords == {"c": ("z", "y", "x", "w")}


def test_roll_rolls_coordinates_on_the_rolled_axis():
    x = XTensor(
        torch.arange(12).reshape(3, 4),
        names=("a", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    # a right shift of 1 moves each label one step forward (cyclically)
    assert x.roll(1, dims="c").coords == {"c": ("z", "w", "x", "y")}


def test_flip_reverses_a_compact_numeric_coordinate_and_stays_compact():
    # #85: reversed()/indexing a Coordinate like a plain dict silently
    # dropped it (flip) or crashed (roll); it must materialise/negate the
    # numeric *values* instead.
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": {"spacing": 1.0}}
    )
    out = x.flip("t")
    assert out.tolist() == [3.0, 2.0, 1.0, 0.0]
    coord = out.coords["t"]
    assert coord._compact()  # negating spacing keeps it exact + compact
    assert coord["values"].tolist() == [3.0, 2.0, 1.0, 0.0]
    assert out.flip("t").coords["t"]["values"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_flip_reverses_an_explicit_numeric_coordinate():
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": torch.tensor([10.0, 20.0, 30.0, 40.0])},
    )
    assert x.flip("t").coords["t"]["values"].tolist() == [
        40.0,
        30.0,
        20.0,
        10.0,
    ]


def test_roll_rolls_a_compact_numeric_coordinate():
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": {"spacing": 1.0}}
    )
    # a right shift of 1: same cyclic convention as the label test above
    out = x.roll(1, dims="t")
    assert out.tolist() == [3.0, 0.0, 1.0, 2.0]
    assert out.coords["t"]["values"].tolist() == [3.0, 0.0, 1.0, 2.0]


def test_roll_rolls_an_explicit_numeric_coordinate():
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": torch.tensor([10.0, 20.0, 30.0, 40.0])},
    )
    assert x.roll(-1, dims="t").coords["t"]["values"].tolist() == [
        20.0,
        30.0,
        40.0,
        10.0,
    ]


def test_flip_of_a_compact_numeric_coordinate_keeps_gradients_flowing():
    spacing = torch.tensor(2.0, requires_grad=True)
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": {"spacing": spacing}}
    )
    x.flip("t").coords["t"]["values"].sum().backward()
    assert spacing.grad is not None


# ----------------------------------------------------------------------
# reshape-family (rank-changing): flatten / unflatten / expand / diagonal
# ----------------------------------------------------------------------


def test_flatten_marks_merged_axis_unnamed():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.flatten(0, 1).names == (None, "c")
    assert x.flatten(0, 1).shape == (6, 4)
    assert x.flatten("a", "b").names == (None, "c")  # by name
    assert x.flatten(1, 1).names == ("a", "b", "c")  # no-op keeps names
    assert torch.flatten(x, 1, 2).names == ("a", None)  # functional form


def test_unflatten_marks_split_axes_unnamed():
    x = XTensor(torch.arange(24).reshape(6, 4), names=("a", "b"))
    y = x.unflatten("a", (2, 3))
    assert y.names == (None, None, "b")
    assert y.shape == (2, 3, 4)
    assert x.unflatten(0, (6,)).names == ("a", "b")  # single split = no-op


def test_expand_prepends_unnamed_axes():
    x = XTensor(torch.zeros(3, 4), names=("b", "c"))
    assert x.expand(2, 3, 4).names == (None, "b", "c")


@pytest.mark.skipif(
    not _HAS_BROADCAST_TO, reason="torch.broadcast_to added in torch 1.8"
)
def test_broadcast_to_prepends_unnamed_axes():
    x = XTensor(torch.zeros(3, 4), names=("b", "c"))
    assert torch.broadcast_to(x, (2, 3, 4)).names == (None, "b", "c")


def test_diagonal_drops_the_two_axes_and_appends_unnamed():
    x = XTensor(torch.zeros(3, 3, 4), names=("a", "b", "c"))
    y = x.diagonal(0, "a", "b")
    assert y.names == ("c", None)
    assert y.shape == (4, 3)


def test_flatten_drops_coordinates_in_the_merged_range():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("a", "b", "c"),
        coords={"b": ("p", "q", "r"), "c": ("w", "x", "y", "z")},
    )
    assert x.flatten(1, 2).coords == {}  # both labelled axes merged away


def test_expand_keeps_coordinates_of_existing_axes():
    x = XTensor(
        torch.zeros(3, 4),
        names=("b", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    out = x.expand(2, 3, 4)
    assert out.names == (None, "b", "c")
    assert out.coords == {"c": ("w", "x", "y", "z")}


def test_diagonal_keeps_the_surviving_axis_coordinates():
    x = XTensor(
        torch.zeros(3, 3, 4),
        names=("a", "b", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    out = x.diagonal(0, 0, 1)
    assert out.names == ("c", None)
    assert out.coords == {"c": ("w", "x", "y", "z")}


# ----------------------------------------------------------------------
# combine ops: cat / stack (name reconciliation across operands)
# ----------------------------------------------------------------------


def test_cat_reconciles_axis_names():
    a = XTensor(torch.zeros(2, 3), names=("r", "c"))
    b = XTensor(torch.zeros(4, 3), names=("r", "c"))
    out = torch.cat([a, b], 0)
    assert out.names == ("r", "c")
    assert out.shape == (6, 3)


def test_cat_conflicting_names_become_unnamed():
    a = XTensor(torch.zeros(2, 3), names=("r", "c"))
    conflicting = XTensor(torch.zeros(2, 3), names=("r", "x"))
    assert torch.cat([a, conflicting], 0).names == ("r", None)


def test_stack_inserts_an_unnamed_axis():
    a = XTensor(torch.zeros(2, 3), names=("r", "c"))
    assert torch.stack([a, a], 0).names == (None, "r", "c")
    assert torch.stack([a, a], 1).names == ("r", None, "c")
    assert torch.stack([a, a], 0).shape == (2, 2, 3)


def test_cat_concatenates_coordinates_along_the_join_axis():
    a = XTensor(torch.zeros(2, 2), names=("r", "c"), coords={"c": ("p", "q")})
    b = XTensor(
        torch.zeros(2, 3), names=("r", "c"), coords={"c": ("x", "y", "z")}
    )
    out = torch.cat([a, b], 1)
    assert out.coords == {"c": ("p", "q", "x", "y", "z")}
    assert out.shape == (2, 5)


def test_cat_with_a_plain_operand_drops_coordinates():
    a = XTensor(torch.zeros(2, 2), names=("r", "c"), coords={"c": ("p", "q")})
    out = torch.cat([a, XTensor(torch.zeros(2, 2))], 1)
    assert out.coords == {}


def test_stack_keeps_the_agreed_coordinates():
    a = XTensor(torch.zeros(2, 2), names=("r", "c"), coords={"c": ("p", "q")})
    out = torch.stack([a, a], 0)
    assert out.names == (None, "r", "c")
    assert out.coords == {"c": ("p", "q")}
    assert out.shape == (2, 2, 2)


# ----------------------------------------------------------------------
# combine ops: hstack / vstack / dstack (conservative: names reconciled only
# when every operand already has the result's rank; coordinates always drop)
# ----------------------------------------------------------------------


def test_hstack_reconciles_names_when_ranks_already_align():
    a = XTensor(torch.zeros(2, 3), names=("r", "c"))
    b = XTensor(torch.zeros(2, 3), names=("r", "c"))
    out = torch.hstack([a, b])
    assert out.names == ("r", "c")
    assert out.coords == {}
    assert out.shape == (2, 6)


def test_vstack_reconciles_names_when_ranks_already_align():
    a = XTensor(torch.zeros(2, 3), names=("r", "c"))
    b = XTensor(torch.zeros(2, 3), names=("r", "c"))
    out = torch.vstack([a, b])
    assert out.names == ("r", "c")
    assert out.coords == {}
    assert out.shape == (4, 3)


def test_dstack_reconciles_names_when_ranks_already_align():
    a = XTensor(torch.zeros(2, 3, 5), names=("r", "c", "d"))
    b = XTensor(torch.zeros(2, 3, 5), names=("r", "c", "d"))
    out = torch.dstack([a, b])
    assert out.names == ("r", "c", "d")
    assert out.coords == {}
    assert out.shape == (2, 3, 10)


def test_vstack_drops_names_when_an_operand_is_promoted():
    # a plain 1-D vector is promoted to 2-D by `vstack`, shifting its axis
    # relative to the already-2-D operand -- positional alignment is not
    # trustworthy here, so the conservative result is fully unnamed.
    a = XTensor(torch.zeros(3), names=("c",))
    b = XTensor(torch.zeros(2, 3), names=("r", "c"))
    out = torch.vstack([a, b])
    assert out.names == (None, None)
    assert out.coords == {}
    assert out.shape == (3, 3)


def test_dstack_conflicting_names_become_unnamed():
    a = XTensor(torch.zeros(2, 3, 5), names=("r", "c", "d"))
    conflicting = XTensor(torch.zeros(2, 3, 5), names=("r", "c", "x"))
    out = torch.dstack([a, conflicting])
    assert out.names == ("r", "c", None)


# ----------------------------------------------------------------------
# matmul family: matmul / mm / bmm / @  (axis names)
# ----------------------------------------------------------------------


def test_matmul_2d_names_rows_from_a_cols_from_b():
    a = XTensor(torch.zeros(2, 3), names=("m", "k"))
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    assert torch.matmul(a, b).names == ("m", "n")
    assert torch.mm(a, b).names == ("m", "n")
    assert (a @ b).names == ("m", "n")  # `@` dispatches Tensor.matmul
    assert (a @ b).shape == (2, 4)


def test_matmul_vector_cases():
    a = XTensor(torch.zeros(2, 3), names=("m", "k"))
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    v = XTensor(torch.zeros(3), names=("k",))
    assert (v @ b).names == ("n",)
    assert (a @ v).names == ("m",)
    assert (v @ v).names == ()


def test_matmul_batches_broadcast_and_reconcile_names():
    a = XTensor(torch.zeros(5, 2, 3), names=("b", "m", "k"))
    b = XTensor(torch.zeros(5, 3, 4), names=("b", "k", "n"))
    assert torch.bmm(a, b).names == ("b", "m", "n")
    assert (a @ b).names == ("b", "m", "n")
    flat = XTensor(torch.zeros(3, 4), names=("k", "n"))
    assert (a @ flat).names == ("b", "m", "n")
    a2 = XTensor(torch.zeros(5, 2, 3), names=("x", "m", "k"))
    assert (a2 @ b).names == (None, "m", "n")


def test_matmul_with_one_plain_operand():
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    assert torch.matmul(torch.zeros(2, 3), b).names == (None, "n")
    a = XTensor(torch.zeros(2, 3), names=("m", "k"))
    assert torch.matmul(a, torch.zeros(3, 4)).names == ("m", None)


def test_matmul_drops_coordinates():
    a = XTensor(
        torch.zeros(2, 3), names=("m", "k"), coords={"k": ("a", "b", "c")}
    )
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    out = a @ b
    assert out.coords == {}
    assert out.names == ("m", "n")


# ----------------------------------------------------------------------
# einsum / tensordot
# ----------------------------------------------------------------------


def test_einsum_explicit_equation_names_output_from_operands():
    a = XTensor(torch.zeros(2, 3), names=("m", "k"))
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    out = torch.einsum("ij,jk->ik", a, b)
    assert out.names == ("m", "n")
    assert out.shape == (2, 4)


def test_einsum_operands_as_a_single_list():
    a = XTensor(torch.zeros(2, 3), names=("m", "k"))
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    out = torch.einsum("ij,jk->ik", [a, b])
    assert out.names == ("m", "n")


def test_einsum_batched_equation():
    a = XTensor(torch.zeros(5, 2, 3), names=("b", "i", "j"))
    b = XTensor(torch.zeros(5, 3, 4), names=("b", "j", "k"))
    out = torch.einsum("bij,bjk->bik", a, b)
    assert out.names == ("b", "i", "k")
    assert out.shape == (5, 2, 4)


def test_einsum_summed_axis_is_dropped_from_the_output():
    a = XTensor(torch.zeros(2, 3), names=("row", "col"))
    out = torch.einsum("ij->i", a)
    assert out.names == ("row",)
    assert out.shape == (2,)


def test_einsum_implicit_output_matches_explicit():
    a = XTensor(torch.zeros(2, 3), names=("m", "k"))
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    assert torch.einsum("ij,jk", a, b).names == ("m", "n")


def test_einsum_ellipsis_falls_back_to_unnamed():
    a = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    b = XTensor(torch.zeros(2, 4, 5), names=("a", "c", "d"))
    out = torch.einsum("...bc,...cd->...bd", a, b)
    assert out.names == (None, None, None)
    assert out.shape == (2, 3, 5)


def test_einsum_with_a_plain_operand():
    a = XTensor(torch.zeros(2, 3), names=("m", "k"))
    out = torch.einsum("ij,jk->ik", a, torch.zeros(3, 4))
    assert out.names == ("m", None)


def test_einsum_drops_coordinates():
    a = XTensor(
        torch.zeros(2, 3), names=("m", "k"), coords={"k": ("x", "y", "z")}
    )
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    out = torch.einsum("ij,jk->ik", a, b)
    assert out.coords == {}


def test_tensordot_int_dims_contracts_trailing_leading_axes():
    a = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    b = XTensor(torch.zeros(3, 4, 5), names=("b", "c", "d"))
    out = torch.tensordot(a, b, dims=2)
    assert out.names == ("a", "d")
    assert out.shape == (2, 5)


def test_tensordot_dims_as_a_pair_of_axis_lists():
    a = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    b = XTensor(torch.zeros(3, 4, 5), names=("b", "c", "d"))
    out = torch.tensordot(a, b, dims=([1, 2], [0, 1]))
    assert out.names == ("a", "d")
    assert out.shape == (2, 5)


def test_tensordot_with_a_plain_operand():
    a = XTensor(torch.zeros(2, 3), names=("a", "k"))
    out = torch.tensordot(a, torch.zeros(3, 4), dims=1)
    assert out.names == ("a", None)
    b = XTensor(torch.zeros(3, 2), names=("k", "a2"))
    out2 = torch.tensordot(torch.zeros(2, 3), b, dims=1)
    assert out2.names == (None, "a2")


def test_tensordot_drops_coordinates():
    a = XTensor(
        torch.zeros(2, 3), names=("a", "k"), coords={"k": ("p", "q", "r")}
    )
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    out = torch.tensordot(a, b, dims=1)
    assert out.coords == {}
    assert out.names == ("a", "n")


def test_einsum_keeps_descriptors_of_surviving_axes_only():
    # a preserved axis keeps its descriptor (type/…); a contracted one's
    # descriptor is dropped with the axis — no leak onto the survivor.
    a = XTensor(
        torch.zeros(5, 2, 3),
        axes=[
            {"name": "b", "type": "batch"},
            "i",
            {"name": "j", "type": "k"},
        ],
    )
    b = XTensor(torch.zeros(5, 3, 4), names=("b", "j", "k"))
    out = torch.einsum("bij,bjk->bik", a, b)
    assert out.names == ("b", "i", "k")
    assert out.axes == (
        {"name": "b", "type": "batch"},
        {"name": "i"},
        {"name": "k"},
    )


def test_tensordot_keeps_descriptors_of_surviving_axes_only():
    a = XTensor(
        torch.zeros(2, 3),
        axes=[{"name": "a", "type": "space"}, {"name": "k", "type": "chan"}],
    )
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    out = torch.tensordot(a, b, dims=1)
    assert out.names == ("a", "n")
    assert out.axes == ({"name": "a", "type": "space"}, {"name": "n"})


def test_matmul_keeps_each_operands_surviving_descriptor():
    # each result axis keeps the descriptor of the operand it came from; the
    # right operand's trailing axis used to lose it.
    a = XTensor(
        torch.zeros(2, 3),
        axes=[{"name": "m", "type": "space"}, {"name": "k", "type": "chan"}],
    )
    b = XTensor(
        torch.zeros(3, 4),
        axes=[{"name": "k", "type": "chan"}, {"name": "n", "type": "time"}],
    )
    out = a @ b
    assert out.names == ("m", "n")
    assert out.axes == (
        {"name": "m", "type": "space"},
        {"name": "n", "type": "time"},
    )


def test_cat_merges_descriptors_keeping_agreement_dropping_conflicts():
    a = XTensor(
        torch.zeros(2, 3),
        axes=[{"name": "r", "type": "space"}, {"name": "c", "type": "chan"}],
    )
    b = XTensor(
        torch.zeros(2, 3),
        axes=[{"name": "r", "type": "space"}, {"name": "c", "type": "time"}],
    )
    out = torch.cat([a, b], dim=0)
    # "r" agrees -> kept; "c" conflicts (chan vs time) -> field dropped
    assert out.axes == ({"name": "r", "type": "space"}, {"name": "c"})


def test_stack_keeps_existing_descriptors_new_axis_unnamed():
    a = XTensor(
        torch.zeros(2, 3),
        axes=[{"name": "r", "type": "space"}, {"name": "c", "type": "chan"}],
    )
    out = torch.stack([a, a], dim=0)
    assert out.axes == (
        None,
        {"name": "r", "type": "space"},
        {"name": "c", "type": "chan"},
    )


def test_combine_op_strict_policy_raises_on_conflicting_descriptor():
    a = XTensor(torch.zeros(2, 3), axes=["r", {"name": "c", "type": "chan"}])
    b = XTensor(torch.zeros(2, 3), axes=["r", {"name": "c", "type": "time"}])
    with set_options(combine_axes="strict"):
        with pytest.raises(ValueError, match="conflicting 'type'"):
            torch.cat([a, b], dim=0)


# ----------------------------------------------------------------------
# gather / scatter / where / masked_select
# ----------------------------------------------------------------------


def test_gather_keeps_axis_names_and_accepts_name_dim():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    idx = torch.zeros(2, 3, 4, dtype=torch.long)
    assert x.gather("c", idx).names == ("a", "b", "c")
    assert torch.gather(x, 2, idx).names == ("a", "b", "c")


@pytest.mark.skipif(
    not hasattr(torch, "take_along_dim"),
    reason="take_along_dim not in this torch",
)
def test_take_along_dim_keeps_names():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    idx = torch.zeros(2, 3, 1, dtype=torch.long)
    assert x.take_along_dim(idx, "c").names == ("a", "b", "c")


def test_scatter_preserves_names_and_coordinates():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("a", "b", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    idx = torch.zeros(2, 3, 4, dtype=torch.long)
    src = torch.zeros(2, 3, 4, dtype=x.dtype)
    out = x.scatter("c", idx, src)
    assert out.names == ("a", "b", "c")
    # positions/sizes are unchanged, so the labels survive
    assert out.coords == {"c": ("w", "x", "y", "z")}


def test_gather_drops_the_gathered_axis_coordinates():
    x = XTensor(
        torch.arange(12).reshape(3, 4),
        names=("a", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    idx = torch.zeros(3, 4, dtype=torch.long)
    assert x.gather("c", idx).coords == {}


def _where_supports_scalar():
    try:
        torch.where(torch.tensor([True]), torch.tensor([1.0]), 0.0)
    except (RuntimeError, TypeError):
        return False
    return True


def test_where_reconciles_operand_names():
    p = XTensor(torch.zeros(2, 3), names=("r", "k"))
    q = XTensor(torch.ones(2, 3), names=("r", "k"))
    assert torch.where(p > 0.5, p, q).names == ("r", "k")
    conflicting = XTensor(torch.ones(2, 3), names=("r", "z"))
    assert torch.where(p > 0.5, p, conflicting).names == ("r", None)


@pytest.mark.skipif(
    not _where_supports_scalar(),
    reason="scalar `where` operand not supported in this torch",
)
def test_where_with_a_scalar_operand_keeps_the_tensor_names():
    p = XTensor(torch.zeros(2, 3), names=("r", "k"))
    assert torch.where(p > 0.5, p, 0.0).names == ("r", "k")


def test_masked_select_collapses_to_one_unnamed_axis():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    out = torch.masked_select(x, x > 5)
    assert out.names == (None,)
    assert out.ndim == 1


def test_masked_fill_keeps_names_and_coordinates():
    x = XTensor(
        torch.arange(6.0).reshape(2, 3),
        names=("r", "c"),
        coords={"c": ("a", "b", "d")},
    )
    # shape-preserving, so names and coordinates ride through (both forms)
    out = x.masked_fill(x > 2, 0.0)
    assert out.names == ("r", "c")
    assert out.coords == {"c": ("a", "b", "d")}
    assert torch.masked_fill(x, x > 2, 0.0).names == ("r", "c")
    x.masked_fill_(x > 2, 0.0)  # in-place keeps them too
    assert x.names == ("r", "c")


def test_nonzero_drops_names_the_output_axes_are_not_the_inputs():
    x = XTensor(
        torch.tensor([[0, 5], [7, 0]]),
        names=("r", "c"),
        coords={"c": ("a", "b")},
    )
    # default: a (nnz, ndim) index tensor -- its axes are NOT r/c
    idx = x.nonzero()
    assert idx.shape == (2, 2)
    assert idx.names == (None, None)
    assert idx.coords == {}
    assert torch.nonzero(x).names == (None, None)
    # as_tuple: one 1-D index tensor per input dim, each unnamed
    parts = x.nonzero(as_tuple=True)
    assert len(parts) == 2
    assert all(p.names == (None,) for p in parts)


# ----------------------------------------------------------------------
# index_add / index_copy / index_fill (name-as-dim, method-form only)
# ----------------------------------------------------------------------


def test_index_fill_keeps_names_and_coordinates():
    x = XTensor(
        torch.zeros(2, 4),
        names=("a", "b"),
        coords={"b": ("w", "x", "y", "z")},
    )
    idx = torch.tensor([1, 3])
    out = x.index_fill("b", idx, 5.0)
    assert out.names == ("a", "b")
    # positions/sizes are unchanged, so the labels survive
    assert out.coords == {"b": ("w", "x", "y", "z")}
    assert torch.equal(out[:, 1], torch.full((2,), 5.0))
    assert torch.equal(out[:, 3], torch.full((2,), 5.0))


def test_index_add_keeps_names_and_coordinates():
    x = XTensor(
        torch.zeros(2, 4),
        names=("a", "b"),
        coords={"b": ("w", "x", "y", "z")},
    )
    idx = torch.tensor([1, 3])
    src = torch.ones(2, 2)
    out = x.index_add("b", idx, src)
    assert out.names == ("a", "b")
    assert out.coords == {"b": ("w", "x", "y", "z")}
    assert torch.equal(out[:, 1], torch.ones(2))
    assert torch.equal(out[:, 0], torch.zeros(2))


def test_index_copy_keeps_names_and_coordinates():
    x = XTensor(
        torch.zeros(2, 4),
        names=("a", "b"),
        coords={"b": ("w", "x", "y", "z")},
    )
    idx = torch.tensor([1, 3])
    src = torch.full((2, 2), 7.0)
    out = x.index_copy("b", idx, src)
    assert out.names == ("a", "b")
    assert out.coords == {"b": ("w", "x", "y", "z")}
    assert torch.equal(out[:, 1], torch.full((2,), 7.0))
    assert torch.equal(out[:, 3], torch.full((2,), 7.0))


# ----------------------------------------------------------------------
# internal helpers
# ----------------------------------------------------------------------


def test_slice_labels_supports_int_slice_bool_and_advanced_indices():
    labels = ("a", "b", "c", "d")
    assert _slice_labels(labels, 1) == ("b",)
    assert _slice_labels(labels, slice(1, 3)) == ("b", "c")
    assert _slice_labels(labels, [True, False, True, False]) == ("a", "c")
    assert _slice_labels(labels, [3, 0]) == ("d", "a")


# ----------------------------------------------------------------------
# convenience specializations
# ----------------------------------------------------------------------


def test_xvector_names_and_labels_the_channel_axis():
    v = xvector(torch.zeros(2, 3), channels=("x", "y", "z"))
    assert type(v) is XTensor  # a plain XTensor, not a distinct subclass
    assert v.names == (None, "channel")
    assert v.coords == {"channel": ("x", "y", "z")}
    assert v.sel(channel="y").shape == (2,)
    assert torch.equal(v.y, v.as_subclass(torch.Tensor)[:, 1])


def test_xvector_channel_dim_and_default_unlabelled():
    v = xvector(torch.zeros(3, 2), channel_dim=0)
    assert v.names == ("channel", None)
    # the default `channels=(...,)` names the axis without labelling it
    assert v.coords == {"channel": (None, None, None)}


def test_xmatrix_labels_row_and_col():
    m = xmatrix(torch.zeros(2, 3), rows=("r0", "r1"), cols=("c0", "c1", "c2"))
    assert type(m) is XTensor
    assert m.names == ("row", "col")
    assert m.coords == {"row": ("r0", "r1"), "col": ("c0", "c1", "c2")}
    assert m.sel(row="r1", col="c2").ndim == 0


def test_xvector_reduction_returns_a_plain_xtensor():
    # dropping the channel axis yields a normal XTensor -- the type never
    # outlives its meaning (the old XVector subclass did not maintain this)
    v = xvector(torch.zeros(2, 3), channels=("x", "y", "z"))
    reduced = v.sum(dim="channel")
    assert type(reduced) is XTensor
    assert reduced.names == (None,)


# ----------------------------------------------------------------------
# pointwise: broadcast-by-name (xarray-style)
# ----------------------------------------------------------------------


def test_add_aligns_transposed_operands_by_name():
    a = XTensor(torch.arange(6.0).reshape(2, 3), names=("x", "y"))
    b = XTensor(
        torch.arange(6.0).reshape(3, 2), names=("y", "x")
    )  # transposed
    out = a + b
    assert out.names == ("x", "y")
    assert out.shape == (2, 3)
    # aligning b to (x, y) is b.T; the sum matches
    assert torch.equal(out, a + b.rename("y", "x").T.rename("x", "y"))


def test_disjoint_named_dims_broadcast_to_the_outer_grid():
    c = XTensor(torch.arange(2.0), names=("x",))
    d = XTensor(torch.arange(3.0), names=("y",))
    out = c + d
    assert out.names == ("x", "y")
    assert out.shape == (2, 3)


def test_shared_name_broadcasts_size_one():
    e = XTensor(torch.arange(3.0).reshape(1, 3), names=("x", "y"))
    f = XTensor(torch.arange(6.0).reshape(2, 3), names=("x", "y"))
    assert (e + f).names == ("x", "y")
    assert (e + f).shape == (2, 3)


def test_scalar_and_plain_operands_keep_the_tensor_names():
    a = XTensor(torch.arange(6.0).reshape(2, 3), names=("x", "y"))
    assert (a + 1).names == ("x", "y")
    assert (a * 2).names == ("x", "y")
    assert (a + torch.ones(2, 3)).names == ("x", "y")
    assert (a == a).names == ("x", "y")  # comparisons too


def test_identical_names_align_positionally_even_with_nonleading_none():
    # same names tuple -> axes correspond 1:1, so a non-leading None is fine
    g = XTensor(torch.zeros(2, 3), names=("x", None))
    assert (g + g).names == ("x", None)
    h = XTensor(torch.zeros(2, 3, 4), names=("b", None, "c"))
    assert (h + h).names == ("b", None, "c")


def test_all_unnamed_operand_behaves_like_a_plain_tensor():
    # an all-None XTensor has nothing to align on -> positional, no raise
    a = XTensor(torch.zeros(2, 3), names=("x", "y"))
    u = XTensor(torch.zeros(2, 3))
    assert u.names == (None, None)
    assert (a + u).names == ("x", "y")


def test_partial_names_align_named_suffix_broadcast_leading_anon():
    # issue #75: unnamed axes all leading -> named suffix aligns by name,
    # anonymous prefix broadcasts positionally. The shared name is used, so
    # the old silent mis-pair (square shapes) is gone.
    a = XTensor(torch.arange(9.0).reshape(3, 3), names=("x", "y"))
    b = XTensor(torch.zeros(3, 3), names=(None, "x"))
    out = a + b
    assert out.names == (None, "x", "y")  # not the old ('x', None)
    assert out.shape == (3, 3, 3)
    # a missing named axis broadcasts; differing anon counts right-align
    c = XTensor(torch.zeros(2, 4, 3, 5), names=(None, None, "x", "y"))
    d = XTensor(torch.zeros(4, 5, 3), names=(None, "y", "x"))
    assert (c + d).names == (None, None, "x", "y")
    assert (c + d).shape == (2, 4, 3, 5)


def test_partial_names_reconcile_coordinates_on_the_named_suffix():
    a = XTensor(torch.zeros(3), names=("y",), coords={"y": ("A", "B", "C")})
    b = XTensor(
        torch.zeros(4, 3), names=(None, "y"), coords={"y": ("C", "B", "A")}
    )
    out = a + b
    # aligned by label (inner-join), anon axis broadcast in front
    assert out.names == (None, "y")
    assert out.coords == {"y": ("A", "B", "C")}


def test_partial_names_not_all_leading_raises():
    # a None after a named axis, with *different* names -> ambiguous -> raise
    a = XTensor(torch.zeros(2, 3), names=("x", None))
    b = XTensor(torch.zeros(2, 3), names=("y", None))
    with pytest.raises(ValueError, match="not all leading"):
        _ = a + b


def test_pointwise_aligns_coordinates_by_name():
    p = XTensor(
        torch.zeros(2, 3), names=("x", "y"), coords={"y": ("a", "b", "c")}
    )
    q = XTensor(
        torch.zeros(3, 2), names=("y", "x"), coords={"y": ("a", "b", "c")}
    )
    out = p + q
    assert out.names == ("x", "y")
    assert out.coords == {"y": ("a", "b", "c")}


def test_pointwise_reindexes_operands_to_align_reordered_labels():
    # a shared labelled dim in a different label order is aligned by *label*,
    # not by position (xarray semantics): A+A, B+B, C+C.
    a = XTensor(
        torch.tensor([1.0, 2.0, 3.0]),
        names=("x",),
        coords={"x": ("A", "B", "C")},
    )
    b = XTensor(
        torch.tensor([10.0, 20.0, 30.0]),
        names=("x",),
        coords={"x": ("C", "B", "A")},
    )
    out = a + b
    assert out.coords == {"x": ("A", "B", "C")}
    assert out.tolist() == [31.0, 22.0, 13.0]


def test_pointwise_inner_joins_partially_overlapping_labels():
    a = XTensor(
        torch.tensor([1.0, 2.0, 3.0]),
        names=("x",),
        coords={"x": ("A", "B", "C")},
    )
    b = XTensor(
        torch.tensor([10.0, 20.0, 30.0]),
        names=("x",),
        coords={"x": ("B", "C", "D")},
    )
    out = a + b
    # intersection in a's order: B, C
    assert out.coords == {"x": ("B", "C")}
    assert out.tolist() == [12.0, 23.0]


def test_pointwise_alignment_broadcasts_over_a_disjoint_dim():
    a = XTensor(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        names=("r", "x"),
        coords={"x": ("A", "B")},
    )
    b = XTensor(
        torch.tensor([10.0, 20.0]), names=("x",), coords={"x": ("B", "A")}
    )
    out = a + b
    assert out.names == ("r", "x")
    assert out.coords == {"x": ("A", "B")}
    # b reindexed to (A, B) = [20, 10], then broadcast over r
    assert out.tolist() == [[21.0, 12.0], [23.0, 14.0]]


def test_pointwise_one_sided_labels_stay_positional():
    # only one operand labels the shared dim: nothing to align against, so the
    # labels ride along and the op stays positional.
    a = XTensor(
        torch.tensor([1.0, 2.0]), names=("x",), coords={"x": ("A", "B")}
    )
    b = XTensor(torch.tensor([10.0, 20.0]), names=("x",))
    out = a + b
    assert out.coords == {"x": ("A", "B")}
    assert out.tolist() == [11.0, 22.0]


# ----------------------------------------------------------------------
# axis descriptors (OME-NGFF-style: type / unit / orientation)
# ----------------------------------------------------------------------


def test_descriptor_axes_expose_extra_fields_while_names_stay_bare():
    x = XTensor(
        torch.zeros(2, 3, 4),
        axes=[
            {"name": "b", "type": "batch"},
            "h",
            {"name": "w", "type": "space", "orientation": "left-to-right"},
        ],
    )
    assert x.names == ("b", "h", "w")  # ergonomic view stays bare
    assert x.axes == (
        {"name": "b", "type": "batch"},
        {"name": "h"},
        {"name": "w", "type": "space", "orientation": "left-to-right"},
    )


def test_axes_keyword_is_an_alias_for_names():
    x = XTensor(torch.zeros(2), axes=[{"name": "t", "type": "time"}])
    assert x.axes == ({"name": "t", "type": "time"},)


def test_descriptor_requires_a_name_and_valid_orientation():
    with pytest.raises(ValueError, match="must have a 'name'"):
        XTensor(torch.zeros(2), axes=[{"type": "space"}])
    with pytest.raises(ValueError, match="a}-to-{b"):
        XTensor(torch.zeros(2), axes=[{"name": "x", "orientation": "lr"}])


def test_names_takes_strings_only_descriptors_go_through_axes():
    with pytest.raises(TypeError, match="descriptor dict through axes="):
        XTensor(torch.zeros(2), names=[{"name": "x", "type": "space"}])


def test_axes_embeds_coordinates():
    x = XTensor(
        torch.zeros(3, 4),
        axes=[
            {"name": "c", "labels": ["r", "g", "b"]},
            {"name": "x", "type": "space", "coord": {"spacing": (0.5, "mm")}},
        ],
    )
    assert x.names == ("c", "x")
    assert x.coords["c"] == ("r", "g", "b")  # `labels` -> categorical coord
    assert x.coords["x"]["values"].tolist() == [0.0, 0.5, 1.0, 1.5]  # `coord`
    assert x.axes[1] == {"name": "x", "type": "space"}  # meta kept


def test_axes_and_coords_merge_at_construction():
    x = XTensor(
        torch.zeros(2, 3),
        axes=[{"name": "a"}, {"name": "b", "type": "chan"}],
        coords={"b": ["p", "q", "r"]},
    )
    assert x.coords["b"] == ("p", "q", "r")  # coords= merges onto axes=
    assert x.axes[1] == {"name": "b", "type": "chan"}


def test_axis_meta_follows_the_dimension_through_ops():
    x = XTensor(
        torch.zeros(2, 3, 4),
        axes=[
            {"name": "b", "type": "batch"},
            "h",
            {"name": "w", "type": "space"},
        ],
    )
    # permute reorders descriptors with their axes
    assert x.permute(2, 0, 1).axes[0] == {"name": "w", "type": "space"}
    # reducing a described axis drops its metadata (getter filters it)
    assert x.sum(dim="w").axes == (
        {"name": "b", "type": "batch"},
        {"name": "h"},
    )


def test_rename_moves_axis_metadata_to_the_new_name():
    x = XTensor(torch.zeros(2, 3), axes=[{"name": "w", "type": "space"}, "h"])
    assert x.rename(w="width").axes[0] == {"name": "width", "type": "space"}


def test_flip_reverses_the_orientation_of_a_flipped_axis():
    x = XTensor(
        torch.zeros(2, 3),
        axes=[{"name": "y", "orientation": "top-to-bottom"}, "x"],
    )
    assert x.flip("y").axes[0]["orientation"] == "bottom-to-top"
    # an axis without an orientation is untouched
    assert x.flip("x").axes[0] == {"name": "y", "orientation": "top-to-bottom"}


def test_descriptors_and_coordinates_coexist():
    z = XTensor(
        torch.zeros(3),
        axes=[{"name": "c", "type": "channel"}],
        coords={"c": ("r", "g", "b")},
    )
    assert z.axes == ({"name": "c", "type": "channel"},)
    assert z.coords == {"c": ("r", "g", "b")}
    assert z.sel(c="g").item() == 0


# ----------------------------------------------------------------------
# combining axis descriptors across operands (the `combine_axes` option)
# ----------------------------------------------------------------------


def _sp(name):
    return {"name": name, "type": "space"}


def _tm(name):
    return {"name": name, "type": "time"}


def test_broadcast_keeps_descriptors_of_each_disjoint_dim():
    # both operands contribute a dim; each keeps its own descriptor (the
    # right operand's used to be dropped).
    a = XTensor(torch.ones(2), axes=[_sp("x")])
    b = XTensor(torch.ones(3), axes=[_tm("y")])
    assert (a + b).axes == (
        {"name": "x", "type": "space"},
        {"name": "y", "type": "time"},
    )


def test_shared_dim_keeps_agreeing_fields_and_one_sided_fields():
    a = XTensor(torch.ones(3), axes=[_sp("x")])
    b = XTensor(torch.ones(3), axes=[_sp("x")])
    assert (a + b).axes == ({"name": "x", "type": "space"},)
    # a field on only one side is not a conflict -- it is kept
    c = XTensor(torch.ones(3), names=["x"])
    assert (a + c).axes == ({"name": "x", "type": "space"},)


def test_conflicting_field_is_dropped_and_order_independent():
    a = XTensor(torch.ones(3), axes=[_sp("x")])
    b = XTensor(torch.ones(3), axes=[_tm("x")])
    # type conflicts (space vs time) -> the field drops; the bare name stays
    assert (a + b).axes == ({"name": "x"},)
    assert (b + a).axes == ({"name": "x"},)  # no left-operand bias


def test_strict_policy_raises_on_conflict_and_restores_after_block():
    a = XTensor(torch.ones(3), axes=[_sp("x")])
    b = XTensor(torch.ones(3), axes=[_tm("x")])
    with set_options(combine_axes="strict"):
        with pytest.raises(ValueError, match="conflicting 'type'"):
            _ = a + b
        # compatible descriptors do not raise even under strict
        assert (a + a).axes == ({"name": "x", "type": "space"},)
    # the option is restored on exit -> conflict drops again
    assert (a + b).axes == ({"name": "x"},)


def test_override_policy_lets_the_left_operand_win():
    a = XTensor(torch.ones(3), axes=[_sp("x")])
    b = XTensor(torch.ones(3), axes=[_tm("x")])
    with set_options(combine_axes="override"):
        assert (a + b).axes == ({"name": "x", "type": "space"},)
        assert (b + a).axes == ({"name": "x", "type": "time"},)


def test_drop_policy_removes_all_descriptors():
    a = XTensor(torch.ones(2), axes=[_sp("x")])
    b = XTensor(torch.ones(3), axes=[_tm("y")])
    with set_options(combine_axes="drop"):
        assert (a + b).axes == ({"name": "x"}, {"name": "y"})


def test_set_options_rejects_unknown_option_or_value():
    with pytest.raises(ValueError, match="unknown option"):
        set_options(nope=1)
    with pytest.raises(ValueError, match="invalid combine_axes policy"):
        set_options(combine_axes="bogus")
    with pytest.raises(ValueError, match="invalid combine_axes policy"):
        set_options(combine_axes={"type": "bogus"})
    with pytest.raises(ValueError, match="must be a policy str or a"):
        set_options(combine_axes=5)
    with pytest.raises(ValueError, match="invalid unit_backend"):
        set_options(unit_backend="nope")
    with pytest.raises(ValueError, match="invalid unit_policy"):
        set_options(unit_policy="nope")


# ----------------------------------------------------------------------
# data units (Proposal 0003 — the .unit annotation)
# ----------------------------------------------------------------------


def test_data_unit_is_stored_carried_and_opaque_without_a_backend():
    # default backend is None: a unit is an opaque string, stored and carried
    x = XTensor(torch.ones(2, 3), names=("a", "b"), unit="V")
    assert x.unit == "V"
    assert x.T.unit == "V"  # rides through reshape/reorder
    assert x.sum(dim="a").unit == "V"  # ... and reductions
    assert x[0].unit == "V"  # ... and indexing
    x.unit = None
    assert x.unit is None
    x.unit = "not_a_real_unit"  # no backend -> no validation
    assert x.unit == "not_a_real_unit"


def test_data_unit_pint_backend_validates_and_normalises():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        y = XTensor(torch.ones(3), unit="mV")
        assert y.unit == "millivolt"  # canonicalised
        with pytest.raises(ValueError, match="invalid unit"):
            XTensor(torch.ones(2), unit="not_a_unit_zz")


def test_to_unit_converts_the_data_by_the_conversion_factor():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        y = XTensor(torch.ones(3), unit="mV")
        z = y.to_unit("V")
        assert z.unit == "volt"
        assert torch.allclose(
            z.as_subclass(torch.Tensor), torch.full((3,), 0.001)
        )
        with pytest.raises(ValueError, match="no unit to convert"):
            XTensor(torch.ones(2)).to_unit("V")


def _united(**kw):
    return {name: XTensor(torch.ones(3), unit=u) for name, u in kw.items()}


def test_data_unit_algebra_mul_div_pow():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        u = _united(V="V", A="A", s="s")
        V, A, s = u["V"], u["A"], u["s"]
        assert (V * A).unit == "ampere * volt"  # product
        assert (V / s).unit == "volt / second"  # quotient
        assert (V * 2).unit == "volt"  # a scalar is dimensionless
        assert (V**2).unit == "volt ** 2"  # power (via the `**` operator)


def test_data_unit_algebra_add_and_compare():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        V = XTensor(torch.ones(3), unit="V")
        A = XTensor(torch.ones(3), unit="A")
        assert (V + V).unit == "volt"  # same unit kept
        assert (V + A).unit is None  # incompatible -> dropped (default policy)
        assert (V < V).unit is None  # comparison result is unitless
        with set_options(unit_policy="strict"):
            with pytest.raises(ValueError, match="incompatible units"):
                _ = V + A


def test_data_unit_matmul_multiplies_units():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        M = XTensor(torch.ones(2, 2), unit="V")
        N = XTensor(torch.ones(2, 2), unit="A")
        assert (M @ N).unit == "ampere * volt"


def test_data_unit_transcendental_requires_dimensionless():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        V = XTensor(torch.ones(3), unit="V")
        assert torch.exp(V).unit is None  # drop the unit (default policy)
        assert torch.log(XTensor(torch.ones(3))).unit is None  # unitless
        with set_options(unit_policy="strict"):
            with pytest.raises(ValueError, match="dimensionless"):
                torch.exp(V)


def test_data_unit_algebra_preserves_autograd():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        leaf = torch.ones(3, requires_grad=True)
        A = XTensor(torch.ones(3), unit="A")
        (XTensor(leaf, unit="V") * A).sum().backward()
        assert leaf.grad is not None


def test_data_unit_algebra_is_inert_without_a_backend():
    # no backend: no algebra, the unit just rides along opaquely
    V = XTensor(torch.ones(3), unit="V")
    A = XTensor(torch.ones(3), unit="A")
    assert (V * A).unit == "V"  # carried from the left operand, not combined
    assert torch.exp(V).unit == "V"  # not dropped


# -- attaching a unit by multiplication (Proposal 0003 phase 4, §2.4) ---------


def test_multiplying_by_a_unit_attaches_it():
    pint = pytest.importorskip("pint")
    u = pint.UnitRegistry()
    with set_options(unit_backend="pint"):
        x = XTensor(torch.arange(3.0), names=("t",))
        attached = x * u.mm  # a bare Unit: data unchanged, unit attached
        assert attached.unit == "millimeter"
        assert attached.tolist() == [0.0, 1.0, 2.0]
        assert attached.names == ("t",)  # names ride through
        assert x.unit is None  # the original is never annotated in place


def test_multiplying_by_a_quantity_scales_the_data():
    pint = pytest.importorskip("pint")
    u = pint.UnitRegistry()
    with set_options(unit_backend="pint"):
        x = XTensor(torch.arange(3.0))
        scaled = x * (3 * u.mm)  # a Quantity carries a magnitude
        assert scaled.unit == "millimeter"
        assert scaled.tolist() == [0.0, 3.0, 6.0]


def test_dividing_by_a_unit_derives_the_quotient_unit():
    pint = pytest.importorskip("pint")
    u = pint.UnitRegistry()
    with set_options(unit_backend="pint"):
        v = XTensor(torch.ones(3), unit="V")
        assert (v / u.s).unit == "volt / second"
        assert (v * u.ohm).unit == "ohm * volt"


def test_unit_multiplication_leaves_ordinary_operands_untouched():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(torch.arange(3.0), unit="m")
        assert (x * 2).tolist() == [0.0, 2.0, 4.0]  # scalar mul, not a unit
        assert (2 * x).tolist() == [0.0, 2.0, 4.0]  # reflected scalar
        assert (x * x).unit == "meter ** 2"  # two united tensors: algebra
        assert (x / 2).unit == "meter"  # scalar divide keeps the unit


# -- heterogeneous (per-axis) data units (Proposal 0003 phase 3) --------------


def _channel_stack():
    # a `q` axis whose positions carry different data units
    return XTensor(
        torch.arange(12.0).reshape(3, 4),
        names=("q", "t"),
        coords={
            "q": [
                {"name": "voltage", "unit": "V"},
                {"name": "current", "unit": "A"},
                {"name": "power", "unit": "W"},
            ]
        },
    )


def test_hetero_unit_folds_into_base_on_selection():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = _channel_stack()
        assert x.unit is None  # heterogeneous: no single base unit
        assert x.sel(q="voltage").unit == "V"  # fold the channel's unit
        assert x.sel(q="current").unit == "A"
        assert x.isel(q=2).unit == "W"  # isel folds too
        assert x[0].unit == "V"  # and plain integer indexing


def test_hetero_unit_selection_multiplies_into_an_existing_base():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(
            torch.arange(12.0).reshape(3, 4),
            names=("q", "t"),
            coords={
                "q": [
                    {"name": "a", "unit": "m"},
                    {"name": "b", "unit": "m"},
                    {"name": "c", "unit": "m"},
                ]
            },
            unit="s",
        )
        assert x.isel(q=0).unit == "meter * second"  # base * coord unit


def test_hetero_unit_slice_keeps_axis_and_units():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = _channel_stack()
        sl = x.isel(q=slice(0, 2))  # keeps the axis -> no fold
        assert sl.unit is None
        assert sl.coords["q"] == (
            {"name": "voltage", "unit": "V"},
            {"name": "current", "unit": "A"},
        )


def _uniform_stack(unit="V"):
    return XTensor(
        torch.arange(12.0).reshape(3, 4),
        names=("q", "t"),
        coords={"q": [{"name": n, "unit": unit} for n in "abc"]},
    )


def test_reduction_folds_a_uniform_axis_unit():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = _uniform_stack("V")
        assert x.sum(dim="q").unit == "V"  # uniform axis unit folds in
        assert x.mean(dim="q").unit == "V"
        assert x.sum(dim="q", keepdim=True).unit == "V"  # keepdim too
        assert x.sum().unit == "V"  # dim=None reduces the unit axis as well


def test_reduction_keeps_base_over_a_unitless_axis():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = _uniform_stack("V")
        assert x.sum(dim="t").unit is None  # `t` carries no units; base stays


def test_reduction_over_incompatible_units_drops_or_raises():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = _channel_stack()
        assert x.sum(dim="q").unit is None  # V/A/W incompatible -> dropped
        with set_options(unit_policy="strict"):
            with pytest.raises(ValueError, match="incompatible units"):
                x.sum(dim="q")


def test_hetero_units_are_inert_without_a_backend():
    x = _channel_stack()  # unit_backend=None
    assert x.sel(q="voltage").unit is None  # no fold when the layer is inert
    assert x.sum(dim="q").unit is None


# -- phase 4: detach, implicit conversion, heterogeneous contraction ----------


def test_magnitude_drops_the_data_unit_but_keeps_names():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(torch.arange(3.0), names=("t",), unit="V")
        m = x.magnitude
        assert m.unit is None  # the data unit is stripped
        assert m.names == ("t",)  # names/coords ride through
        assert m.tolist() == [0.0, 1.0, 2.0]
        assert x.unit == "volt"  # the original is unchanged (a view)


def test_add_implicitly_converts_compatible_units():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        volts = XTensor(torch.ones(3), names=("t",), unit="V")
        millivolts = XTensor(torch.full((3,), 500.0), names=("t",), unit="mV")
        # right operand converts to the left's unit before adding
        left = volts + millivolts
        assert left.unit == "volt"
        assert left.tolist() == [1.5, 1.5, 1.5]
        right = millivolts + volts
        assert right.unit == "millivolt"
        assert right.tolist() == [1500.0, 1500.0, 1500.0]
        # comparisons convert too
        assert (volts > millivolts).tolist() == [True, True, True]


def test_add_incompatible_units_still_drops_or_raises():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        volts = XTensor(torch.ones(3), unit="V")
        amps = XTensor(torch.ones(3), unit="A")
        assert (volts + amps).unit is None  # incompatible -> dropped
        with set_options(unit_policy="strict"):
            with pytest.raises(ValueError, match="incompatible units"):
                _ = volts + amps


def test_matmul_folds_uniform_contracted_axis_units():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        a = XTensor(
            torch.ones(2, 3),
            names=("i", "k"),
            coords={"k": [{"name": c, "unit": "m"} for c in "abc"]},
            unit="V",
        )
        b = XTensor(
            torch.ones(3, 2),
            names=("k", "j"),
            coords={"k": [{"name": c, "unit": "s"} for c in "abc"]},
            unit="A",
        )
        # (V·m) · (A·s) — each side's uniform contracted-axis unit folds in
        assert (a @ b).unit == "ampere * meter * second * volt"


def test_contraction_over_non_uniform_axis_drops_or_raises():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        a = XTensor(
            torch.ones(2, 3),
            names=("i", "k"),
            coords={
                "k": [
                    {"name": "x", "unit": "m"},
                    {"name": "y", "unit": "s"},
                    {"name": "z", "unit": "kg"},
                ]
            },
        )
        b = XTensor(torch.ones(3, 2), names=("k", "j"))
        assert (a @ b).unit is None  # non-uniform contracted axis -> dropped
        with set_options(unit_policy="strict"):
            with pytest.raises(ValueError, match="non-uniform"):
                _ = a @ b


def test_einsum_and_tensordot_multiply_operand_units():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        m = XTensor(torch.ones(2, 2), unit="V")
        n = XTensor(torch.ones(2, 2), unit="A")
        # einsum previously kept only the first operand's base unit
        assert torch.einsum("ij,jk->ik", m, n).unit == "ampere * volt"
        assert torch.tensordot(m, n, dims=1).unit == "ampere * volt"
        # an unparsable equation (ellipsis) falls back to the base product
        p = XTensor(torch.ones(2, 2, 2), unit="V")
        q = XTensor(torch.ones(2, 2, 2), unit="A")
        out = torch.einsum("...ij,...jk->...ik", p, q)
        assert out.unit == "ampere * volt"


def test_combine_axes_accepts_a_per_field_policy():
    # a custom, free-form descriptor field ("role") -- nothing is special
    # about it; the per-field policy applies to any key.
    a = XTensor(
        torch.ones(3),
        axes=[{"name": "x", "type": "space", "role": "readout"}],
    )
    b = XTensor(
        torch.ones(3),
        axes=[{"name": "x", "type": "time", "role": "readout"}],
    )
    # default drops every field, but an agreeing "role" is kept
    with set_options(combine_axes={"*": "drop", "role": "raise"}):
        assert (a + b).axes == ({"name": "x", "role": "readout"},)


def test_per_field_raise_policy_fires_only_for_that_field():
    a = XTensor(torch.ones(3), axes=[{"name": "x", "role": "readout"}])
    b = XTensor(torch.ones(3), axes=[{"name": "x", "role": "phase"}])
    with set_options(combine_axes={"*": "drop", "role": "raise"}):
        with pytest.raises(ValueError, match="conflicting 'role'"):
            _ = a + b


def test_per_field_override_keeps_the_left_value_for_one_field():
    a = XTensor(
        torch.ones(3),
        axes=[{"name": "x", "type": "space", "orientation": "left-to-right"}],
    )
    b = XTensor(
        torch.ones(3),
        axes=[{"name": "x", "type": "space", "orientation": "right-to-left"}],
    )
    with set_options(combine_axes={"orientation": "override"}):
        # "type" agrees (kept); "orientation" takes the left operand's value
        assert (a + b).axes[0]["orientation"] == "left-to-right"
        assert (b + a).axes[0]["orientation"] == "right-to-left"


def test_unlisted_fields_fall_back_to_drop_conflicts():
    a = XTensor(torch.ones(3), axes=[{"name": "x", "type": "space"}])
    b = XTensor(torch.ones(3), axes=[{"name": "x", "type": "time"}])
    # only "role" is customised; "type" uses the drop_conflicts default
    with set_options(combine_axes={"role": "strict"}):
        assert (a + b).axes == ({"name": "x"},)


def test_descriptor_keys_are_free_form_not_a_fixed_schema():
    # `type` is only a convention; any custom key is stored, carried through
    # ops, and queryable exactly the same way.
    x = XTensor(
        torch.zeros(2, 3),
        axes=[
            {"name": "c", "modality": "MRI", "role": "readout"},
            {"name": "t", "modality": "MRI"},
        ],
    )
    assert x.axes[0] == {"name": "c", "modality": "MRI", "role": "readout"}
    # a custom key follows the dim through an op ...
    assert x.T.axes[1]["role"] == "readout"
    # ... and addresses axes just like `type` would
    assert x.sum(dim={"modality": "MRI"}).ndim == 0
    assert x.sum(dim={"role": "readout"}).names == ("t",)


def test_movedim_by_type_moves_the_whole_block_to_the_back():
    x = XTensor(
        torch.zeros(2, 3, 4, 5),
        axes=[
            {"name": "b", "type": "batch"},
            {"name": "h", "type": "space"},
            {"name": "c", "type": "channel"},
            {"name": "w", "type": "space"},
        ],
    )
    moved = x.movedim({"type": "space"}, -1)
    # both space axes go to the back, keeping their relative order (h before w)
    assert moved.names == ("b", "c", "h", "w")
    assert tuple(moved.shape) == (2, 4, 3, 5)


@pytest.mark.skipif(
    not _HAS_MOVEAXIS, reason="torch.moveaxis added in torch 1.8"
)
def test_moveaxis_by_type_moves_the_block_to_the_front():
    x = XTensor(
        torch.zeros(2, 3, 4, 5),
        axes=[
            {"name": "b", "type": "batch"},
            {"name": "h", "type": "space"},
            {"name": "c", "type": "channel"},
            {"name": "w", "type": "space"},
        ],
    )
    moved = x.moveaxis({"type": "space"}, 0)
    assert moved.names == ("h", "w", "b", "c")
    assert tuple(moved.shape) == (3, 5, 2, 4)


def test_sum_by_type_reduces_every_matching_axis():
    y = XTensor(
        torch.ones(2, 3, 4),
        axes=[
            {"name": "b", "type": "batch"},
            {"name": "c1", "type": "channel"},
            {"name": "c2", "type": "channel"},
        ],
    )
    reduced = y.sum(dim={"type": "channel"})
    assert reduced.names == ("b",)
    assert reduced[0].item() == 12  # 3 * 4 ones summed


def test_single_match_query_stays_prod_safe():
    # a query hitting exactly one axis collapses to a bare int, so even
    # single-`dim`-only reducers (`prod`) accept it
    y = XTensor(
        torch.ones(2, 3, 4),
        axes=[
            {"name": "b", "type": "batch"},
            {"name": "c1", "type": "channel"},
            {"name": "c2", "type": "channel"},
        ],
    )
    assert y.prod(dim={"type": "batch"}).names == ("c1", "c2")
