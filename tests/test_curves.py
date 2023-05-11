import pytest
from datetime import datetime as dt
from pandas import DataFrame
import numpy as np
from math import log, exp

import context
from rateslib.curves import Curve, LineCurve, index_left, interpolate, IndexCurve
from rateslib.dual import Dual, Dual2
from rateslib.calendars import get_calendar


@pytest.fixture()
def curve():
    return Curve(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.99,
        },
        interpolation="linear",
        id="v",
        convention="Act360",
        ad=1,
    )


@pytest.fixture()
def line_curve():
    return LineCurve(
        nodes={
            dt(2022, 3, 1): 2.00,
            dt(2022, 3, 31): 2.01,
        },
        interpolation="linear",
        id="v",
        ad=1,
    )


@pytest.fixture()
def index_curve():
    return IndexCurve(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.999,
        },
        interpolation="linear_index",
        id="v",
        ad=1,
        index_base=110.0,
    )


@pytest.mark.parametrize("method", ["flat_forward", "flat_backward"])
def test_flat_interp(method):
    assert interpolate(1, 1, 5, 2, 10, method) == 5
    assert interpolate(2, 1, 5, 2, 10, method) == 10
    assert interpolate(1.5, 1, 5, 2, 10, "flat_forward") == 5
    assert interpolate(1.5, 1, 5, 2, 10, "flat_backward") == 10


@pytest.mark.parametrize("curve_style, expected", [("df", 0.995), ("line", 2.005)])
def test_linear_interp(curve_style, expected, curve, line_curve):
    if curve_style == "df":
        obj = curve
    else:
        obj = line_curve
    assert obj[dt(2022, 3, 16)] == Dual(expected, ["v0", "v1"], np.array([0.5, 0.5]))


def test_log_linear_interp():
    curve = Curve(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.99,
        },
        interpolation="log_linear",
        id="v",
        convention="Act360",
        ad=1,
    )
    val = exp((log(1.00) + log(0.99)) / 2)
    assert curve[dt(2022, 3, 16)] == Dual(val, ["v0", "v1"], np.array([0.49749372, 0.50251891]))


def test_linear_zero_rate_interp():
    # not tested
    pass


def test_line_curve_rate(line_curve):
    expected = Dual(2.005, ["v0", "v1"], np.array([0.5, 0.5]))
    assert line_curve.rate(effective=dt(2022, 3, 16)) == expected


@pytest.mark.parametrize("li, ll, val, expected", [
    ([0, 1, 2, 3, 4], 5, 0, 0),
    ([0, 1, 2, 3, 4], 5, 0.5, 0),
    ([0, 1, 2, 3, 4], 5, 1, 0),
    ([0, 1, 2, 3, 4], 5, 1.5, 1),
    ([0, 1, 2, 3, 4], 5, 2, 1),
    ([0, 1, 2, 3, 4], 5, 2.5, 2),
    ([0, 1, 2, 3, 4], 5, 3, 2),
    ([0, 1, 2, 3, 4], 5, 3.5, 3),
    ([0, 1, 2, 3, 4], 5, 4, 3),
    ([0, 1, 2, 3, 4], 5, 4.5, 3),  # extrapolate
    ([0, 1, 2, 3, 4], 5, -0.5, 0),  # extrapolate
])
def test_index_left(li, ll, val, expected):
    result = index_left(li, ll, val)
    assert result == expected


def test_zero_rate_plot():
    # test calcs without raise
    curve_zero = Curve(
        nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99,
               dt(2024, 1, 1): 0.979, dt(2025, 1, 1): 0.967},
        interpolation="linear_zero_rate",
    )
    curve_zero.plot("1d")


def test_curve_equality_type_differ(curve, line_curve):
    assert curve != line_curve


def test_serialization(curve):
    expected = (
        '{"nodes": {"2022-03-01": 1.0, "2022-03-31": 0.99}, '
        '"interpolation": "linear", "t": null, "c": null, "id": "v", '
        '"convention": "Act360", "endpoints": ["natural", "natural"], "modifier": "MF", '
        '"calendar_type": "null", "ad": 1, "calendar": null}'
    )
    result = curve.to_json()
    assert result == expected


def test_serialization_round_trip(curve, line_curve, index_curve):
    serial = curve.to_json()
    constructed = Curve.from_json(serial)
    assert constructed == curve

    serial = line_curve.to_json()
    constructed = LineCurve.from_json(serial)
    assert constructed == line_curve

    serial = index_curve.to_json()
    constructed = IndexCurve.from_json(serial)
    assert constructed == index_curve


def test_serialization_round_trip_spline():
    curve = Curve(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.99,
            dt(2022, 5, 1): 0.98,
            dt(2022, 6, 4): 0.97,
            dt(2022, 7, 4): 0.96,
        },
        interpolation="linear",
        id="v",
        convention="Act360",
        ad=1,
        t=[dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1),
           dt(2022, 6, 4),
           dt(2022, 7, 4), dt(2022, 7, 4), dt(2022, 7, 4), dt(2022, 7, 4)]
    )

    serial = curve.to_json()
    constructed = Curve.from_json(serial)
    assert constructed == curve


def test_serialization_curve_str_calendar():
    curve = Curve(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.99,
        },
        interpolation="linear",
        id="v",
        convention="Act360",
        modifier="F",
        calendar="LDN",
        ad=1,
    )
    serial = curve.to_json()
    constructed = Curve.from_json(serial)
    assert constructed == curve


def test_serialization_curve_custom_calendar():
    calendar = get_calendar("ldn")
    curve = Curve(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.99,
        },
        interpolation="linear",
        id="v",
        convention="Act360",
        modifier="F",
        calendar=calendar,
        ad=1,
    )
    serial = curve.to_json()
    constructed = Curve.from_json(serial)
    assert constructed == curve


def test_copy_curve(curve, line_curve):
    copied = curve.copy()
    assert copied == curve
    assert id(copied) != id(curve)

    copied = line_curve.copy()
    assert copied == line_curve
    assert id(copied) != id(line_curve)


@pytest.mark.parametrize("attr, val", [
    ("nodes", {dt(2022, 3, 1): 1.00}),
    ("interpolation", "log_linear"),
    ("id", "x"),
    ("ad", 0),
    ("convention", "actact"),
    ("t", [dt(2022, 1, 1)]),
    ("calendar_type", "bad")
])
def test_curve_equality_checks(attr, val, curve):
    copied_curve = curve.copy()
    setattr(copied_curve, attr, val)
    assert copied_curve != curve


def test_curve_equality_spline_coeffs():
    curve = Curve(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.99,
            dt(2022, 5, 1): 0.98,
            dt(2022, 6, 4): 0.97,
            dt(2022, 7, 4): 0.96,
        },
        interpolation="linear",
        id="v",
        convention="Act360",
        ad=0,
        t=[dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1),
           dt(2022, 6, 4),
           dt(2022, 7, 4), dt(2022, 7, 4), dt(2022, 7, 4), dt(2022, 7, 4)]
    )
    curve2 = Curve(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.99,
            dt(2022, 5, 1): 0.98,
            dt(2022, 6, 4): 0.97,
            dt(2022, 7, 4): 0.93,  # <- note generates different spline
        },
        interpolation="linear",
        id="v",
        convention="Act360",
        ad=0,
        t=[dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1), dt(2022, 5, 1),
           dt(2022, 6, 4),
           dt(2022, 7, 4), dt(2022, 7, 4), dt(2022, 7, 4), dt(2022, 7, 4)]
    )
    curve2.nodes[dt(2022, 7, 4)] = 0.96  # set a specific node without recalc spline
    assert curve2 != curve  # should detect on curve2.spline.c


def test_custom_interpolator():
    def interp(date, nodes):
        return date

    curve = Curve(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.99,
        },
        interpolation=interp,
        id="v",
        convention="Act360",
        ad=1,
    )

    assert curve[dt(2022, 3, 15)] == dt(2022, 3, 15)


def test_df_is_zero_in_past(curve):
    assert curve[dt(1999, 1, 1)] == 0.0


def test_curve_none_return(curve):
    result = curve.rate(dt(2022, 2, 1), dt(2022, 2, 2))
    assert result  is None


@pytest.mark.parametrize("endpoints, expected", [
    ("natural", [1.0, 0.995913396831872, 0.9480730429565414, 0.95]),
    ("not_a_knot", [1.0, 0.9967668788593117, 0.9461282456344617, 0.95]),
    (("not_a_knot", "natural"), [1.0, 0.9965809643843604, 0.9480575781858877, 0.95]),
    (("natural", "not_a_knot"), [1.0, 0.9959615881004005, 0.9461971628597721, 0.95])
])
def test_spline_endpoints(endpoints, expected):
    curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.99,
            dt(2024, 1, 1): 0.97,
            dt(2025, 1, 1): 0.95,
            dt(2026, 1, 1): 0.95,
        },
        endpoints=endpoints,
        t=[dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
           dt(2023, 1, 1),
           dt(2024, 1, 1),
           dt(2025, 1, 1),
           dt(2026, 1, 1), dt(2026, 1, 1), dt(2026, 1, 1), dt(2026, 1, 1)],
    )
    for i, date in enumerate(
        [dt(2022, 1, 1), dt(2022, 7, 1), dt(2025, 7, 1), dt(2026, 1, 1)]
    ):
        assert curve[date] == expected[i]


@pytest.mark.parametrize("endpoints", [("natural", "bad"), ("bad", "natural")])
def test_spline_endpoints_raise(endpoints):
    with pytest.raises(NotImplementedError, match="Endpoint method"):
        curve = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.99,
                dt(2024, 1, 1): 0.97,
                dt(2025, 1, 1): 0.95,
                dt(2026, 1, 1): 0.95,
            },
            endpoints=endpoints,
            t=[dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
               dt(2023, 1, 1),
               dt(2024, 1, 1),
               dt(2025, 1, 1),
               dt(2026, 1, 1), dt(2026, 1, 1), dt(2026, 1, 1), dt(2026, 1, 1)],
        )


def test_not_a_knot_raises():
    with pytest.raises(ValueError, match="`endpoints` cannot be 'not_a_knot'"):
        curve = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2024, 1, 1): 0.97,
                dt(2026, 1, 1): 0.95,
            },
            endpoints="not_a_knot",
            t=[dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
               dt(2024, 1, 1),
               dt(2026, 1, 1), dt(2026, 1, 1), dt(2026, 1, 1), dt(2026, 1, 1)],
        )


def test_set_ad_order_no_spline():
    curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.99,
        },
        id="v",
    )
    assert curve[dt(2022, 1, 1)] == 1.0
    assert curve.ad == 0

    curve._set_ad_order(1)
    assert curve[dt(2022, 1, 1)] == Dual(1.0, "v0")
    assert curve.ad == 1

    old_id = id(curve.nodes)
    curve._set_ad_order(2)
    assert curve[dt(2022, 1, 1)] == Dual2(1.0, "v0")
    assert curve.ad == 2
    assert id(curve.nodes) != old_id  # new nodes object thus a new id

    expected_id = id(curve.nodes)
    curve._set_ad_order(2)
    assert id(curve.nodes) == expected_id  # new objects not created when order unchged


def test_set_ad_order_raises(curve):
    with pytest.raises(ValueError, match="`order` can only be in {0, 1, 2}"):
        curve._set_ad_order(100)


def test_index_left_raises():
    with pytest.raises(ValueError, match="`index_left` designed for intervals."):
        index_left([1], 1, 100)


# def test_curve_shift():
#     curve = Curve(
#         nodes={
#             dt(2022, 1, 1): 1.0,
#             dt(2023, 1, 1): 0.988,
#             dt(2024, 1, 1): 0.975,
#             dt(2025, 1, 1): 0.965,
#             dt(2026, 1, 1): 0.955,
#             dt(2027, 1, 1): 0.9475
#         },
#         t=[
#             dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
#             dt(2025, 1, 1),
#             dt(2026, 1, 1),
#             dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
#         ],
#     )
#     result_curve = curve.shift(25)
#     diff = np.array([
#         result_curve.rate(_, "1D") - curve.rate(_, "1D") - 0.25 for _ in [
#             dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)
#         ]
#     ])
#     assert np.all(np.abs(diff) < 1e-7)


@pytest.mark.parametrize("ad_order", [0, 1, 2])
def test_curve_shift_ad_order(ad_order):
    curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.988,
            dt(2024, 1, 1): 0.975,
            dt(2025, 1, 1): 0.965,
            dt(2026, 1, 1): 0.955,
            dt(2027, 1, 1): 0.9475
        },
        t=[
            dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
        ],
        ad=ad_order,
    )
    result_curve = curve.shift(25)
    diff = np.array([
        result_curve.rate(_, "1D") - curve.rate(_, "1D") - 0.25 for _ in [
            dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)
        ]
    ])
    assert np.all(np.abs(diff) < 1e-7)


def test_curve_shift_dual_input():
    curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.988,
            dt(2024, 1, 1): 0.975,
            dt(2025, 1, 1): 0.965,
            dt(2026, 1, 1): 0.955,
            dt(2027, 1, 1): 0.9475
        },
        t=[
            dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
        ],
    )
    result_curve = curve.shift(Dual(25, "z"))
    diff = np.array([
        result_curve.rate(_, "1D") - curve.rate(_, "1D") - 0.25 for _ in [
            dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)
        ]
    ])
    assert np.all(np.abs(diff) < 1e-7)


@pytest.mark.parametrize("ad_order", [0, 1, 2])
def test_linecurve_shift(ad_order):
    curve = LineCurve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.988,
            dt(2024, 1, 1): 0.975,
            dt(2025, 1, 1): 0.965,
            dt(2026, 1, 1): 0.955,
            dt(2027, 1, 1): 0.9475,
        },
        t=[
            dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
        ],
        ad=ad_order
    )
    result_curve = curve.shift(25)
    diff = np.array([
        result_curve[_] - curve[_] - 0.25 for _ in [
            dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)
        ]
    ])
    assert np.all(np.abs(diff) < 1e-7)


def test_linecurve_shift_dual_input():
    curve = LineCurve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.988,
            dt(2024, 1, 1): 0.975,
            dt(2025, 1, 1): 0.965,
            dt(2026, 1, 1): 0.955,
            dt(2027, 1, 1): 0.9475,
        },
        t=[
            dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
        ],
    )
    result_curve = curve.shift(Dual(25, "z"))
    diff = np.array([
        result_curve[_] - curve[_] - 0.25 for _ in [
            dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)
        ]
    ])
    assert np.all(np.abs(diff) < 1e-7)


@pytest.mark.parametrize("ad_order", [0, 1, 2])
def test_indexcurve_shift(ad_order):
    curve = IndexCurve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.988,
            dt(2024, 1, 1): 0.975,
            dt(2025, 1, 1): 0.965,
            dt(2026, 1, 1): 0.955,
            dt(2027, 1, 1): 0.9475,
        },
        t=[
            dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
        ],
        ad=ad_order,
        index_base=110.,
        interpolation="log_linear",
    )
    result_curve = curve.shift(25)
    diff = np.array([
        result_curve.rate(_, "1D") - curve.rate(_, "1D") - 0.25 for _ in [
            dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)
        ]
    ])
    assert np.all(np.abs(diff) < 1e-7)
    assert result_curve.index_base == curve.index_base


def test_indexcurve_shift_dual_input():
    curve = IndexCurve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.988,
            dt(2024, 1, 1): 0.975,
            dt(2025, 1, 1): 0.965,
            dt(2026, 1, 1): 0.955,
            dt(2027, 1, 1): 0.9475,
        },
        t=[
            dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
        ],
        index_base=110.0,
        interpolation="log_linear",
    )
    result_curve = curve.shift(Dual(25, "z"))
    diff = np.array([
        result_curve.rate(_, "1D") - curve.rate(_, "1D") - 0.25 for _ in [
            dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)
        ]
    ])
    assert np.all(np.abs(diff) < 1e-7)
    assert result_curve.index_base == curve.index_base


@pytest.mark.parametrize("crv, t, tol", [
    (Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.988,
            dt(2024, 1, 1): 0.975,
            dt(2025, 1, 1): 0.965,
            dt(2026, 1, 1): 0.955,
            dt(2027, 1, 1): 0.9475
        },
        t=[
            dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
        ],
    ), False, 1e-8),
    (IndexCurve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.988,
            dt(2024, 1, 1): 0.975,
            dt(2025, 1, 1): 0.965,
            dt(2026, 1, 1): 0.955,
            dt(2027, 1, 1): 0.9475
        },
        t=[
            dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
        ],
        index_base=110.,
    ), False, 1e-8),
    (LineCurve(
        nodes={
            dt(2022, 1, 1): 1.7,
            dt(2023, 1, 1): 1.65,
            dt(2024, 1, 1): 1.4,
            dt(2025, 1, 1): 1.3,
            dt(2026, 1, 1): 1.25,
            dt(2027, 1, 1): 1.35
        },
        t=[
            dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
        ],
    ), False, 1e-8),
    (Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 2): 0.988,
            dt(2024, 1, 1): 0.975,
            dt(2025, 1, 1): 0.965,
            dt(2026, 1, 1): 0.955,
            dt(2027, 1, 1): 0.9475
        },
        t=[
            dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
            dt(2023, 1, 2),
            dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
        ],
    ), True, 1e-3),
])
def test_curve_translate(crv, t, tol):
    result_curve = crv.translate(dt(2023, 1, 1), t=t)
    diff = np.array([
        result_curve.rate(_, "1D") - crv.rate(_, "1D") for _ in [
            dt(2023, 1, 25), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)
        ]
    ])
    assert np.all(np.abs(diff) < tol)
    if type(crv) is IndexCurve:
        assert result_curve.index_base == crv.index_value(dt(2023, 1, 1))


@pytest.mark.parametrize("crv", [
    Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.988,
            dt(2024, 1, 1): 0.975,
            dt(2025, 1, 1): 0.965,
            dt(2026, 1, 1): 0.955,
            dt(2027, 1, 1): 0.9475
        },
        t=[
            dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
        ],
    ),
    LineCurve(
        nodes={
            dt(2022, 1, 1): 1.7,
            dt(2023, 1, 1): 1.65,
            dt(2024, 1, 1): 1.4,
            dt(2025, 1, 1): 1.3,
            dt(2026, 1, 1): 1.25,
            dt(2027, 1, 1): 1.35
        },
        t=[
            dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1), dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
        ],
    )
])
def test_curve_roll(crv):
    rolled_curve = crv.roll("10d")
    rolled_curve2 = crv.roll("-10d")

    expected = np.array([crv.rate(_, "1D") for _ in [
            dt(2023, 1, 15), dt(2023, 3, 15), dt(2024, 11, 15), dt(2026, 4, 15)
        ]
    ])
    result = np.array([rolled_curve.rate(_, "1D") for _ in [
            dt(2023, 1, 25), dt(2023, 3, 25), dt(2024, 11, 25), dt(2026, 4, 25)
        ]
    ])
    result2 = np.array([rolled_curve2.rate(_, "1D") for _ in [
            dt(2023, 1, 5), dt(2023, 3, 5), dt(2024, 11, 5), dt(2026, 4, 5)
        ]
    ])
    assert np.all(np.abs(result - expected) < 1e-7)
    assert np.all(np.abs(result2 - expected) < 1e-7)


def test_curve_roll_copy(curve):
    result = curve.roll("0d")
    assert result == curve


def test_curve_translate_raises(curve):
    with pytest.raises(ValueError, match="Cannot translate exactly for the given"):
        curve.translate(dt(2022, 4, 1))


def test_curve_translate_knots_raises(curve):
    curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.988,
            dt(2024, 1, 1): 0.975,
            dt(2025, 1, 1): 0.965,
            dt(2026, 1, 1): 0.955,
            dt(2027, 1, 1): 0.9475
        },
        t=[dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
           dt(2022, 12, 1),
           dt(2024, 1, 1),
           dt(2025, 1, 1),
           dt(2026, 1, 1),
           dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1), dt(2027, 1, 1),
        ],
    )
    with pytest.raises(ValueError, match="Cannot translate spline knots for given"):
        curve.translate(dt(2022, 12, 15))


def test_curve_index_linear_daily_interp():
    curve = IndexCurve(
        nodes={dt(2022, 1, 1): 1.0, dt(2022, 1, 5): 0.9999},
        index_base=200.0,
    )
    result = curve.index_value(dt(2022, 1, 5))
    expected = 200.020002002
    assert abs(result - expected) < 1e-7

    result = curve.index_value(dt(2022, 1, 3))
    expected = 200.010001001  # value is linearly interpolated between index values.
    assert abs(result - expected) < 1e-7


def test_indexcurve_raises():
    with pytest.raises(ValueError, match="`index_base` must be given"):
        curve = IndexCurve({dt(2022, 1, 1): 1.0})


class TestPlotCurve:

    def test_plot_curve(self, curve):
        fig, ax, lines = curve.plot("1d")
        result = lines[0].get_data()
        assert result[0][0] == dt(2022, 3, 1)
        assert abs(result[1][0].real - 12.004001333774994) < 1e-6

    def test_plot_linecurve(self, line_curve):
        fig, ax, lines = line_curve.plot("0d")
        result = lines[0].get_data()
        assert result[0][0] == dt(2022, 3, 1)
        assert abs(result[1][0].real - 2.0) < 1e-6

    @pytest.mark.parametrize("left", ["1d", dt(2022, 3, 2)])
    def test_plot_curve_left(self, curve, left):
        fig, ax, lines = curve.plot("1d", left=left)
        result = lines[0].get_data()
        assert result[0][0] == dt(2022, 3, 2)
        assert abs(result[1][0].real - 12.008005336896055) < 1e-6

    def test_plot_curve_left_raise(self, curve):
        with pytest.raises(ValueError, match="`left` must be supplied as"):
            fig, ax, lines = curve.plot("1d", left=100.3)

    @pytest.mark.parametrize("right", ["2d", dt(2022, 3, 3)])
    def test_plot_curve_right(self, curve, right):
        fig, ax, lines = curve.plot("1d", right=right)
        result = lines[0].get_data()
        assert result[0][-1] == dt(2022, 3, 2)
        assert abs(result[1][-1].real - 12.008005336896055) < 1e-6

    def test_plot_curve_right_raise(self, curve):
        with pytest.raises(ValueError, match="`right` must be supplied as"):
            fig, ax, lines = curve.plot("1d", right=100.3)

    def test_plot_comparators(self, curve):
        fig, ax, lines = curve.plot("1d", comparators=[curve])
        assert len(lines) == 2
        res1 = lines[0].get_data()
        res2 = lines[1].get_data()
        assert res1[0][0] == res2[0][0]
        assert res1[1][0] == res2[1][0]

    def test_plot_diff(self, curve):
        fig, ax, lines = curve.plot("1d", comparators=[curve], difference=True)
        assert len(lines) == 1
        result = lines[0].get_data()
        assert result[0][0] == dt(2022, 3, 1)
        assert result[1][0] == 0