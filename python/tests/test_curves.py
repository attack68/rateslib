from datetime import datetime as dt
from math import exp, log

import numpy as np
import pytest
from matplotlib import pyplot as plt
from rateslib import default_context
from rateslib.calendars import get_calendar
from rateslib.curves import (
    CompositeCurve,
    Curve,
    IndexCurve,
    LineCurve,
    MultiCsaCurve,
    index_left,
    interpolate,
)
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, gradient
from rateslib.fx import FXForwards, FXRates
from rateslib.instruments import IRS
from rateslib.solver import Solver


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
    assert obj[dt(2022, 3, 16)] == Dual(expected, ["v0", "v1"], [0.5, 0.5])


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
    result = curve[dt(2022, 3, 16)]
    expected = Dual(val, ["v0", "v1"], [0.49749372, 0.50251891])
    assert abs(result - expected) < 1e-15
    assert all(np.isclose(gradient(result, ["v0", "v1"]), expected.dual))


def test_linear_zero_rate_interp():
    # not tested
    pass


def test_line_curve_rate(line_curve):
    expected = Dual(2.005, ["v0", "v1"], [0.5, 0.5])
    assert line_curve.rate(effective=dt(2022, 3, 16)) == expected


@pytest.mark.parametrize(
    "scm, exp",
    [
        ("none_simple", 5.56617834937),
        ("isda_flat_compounding", 5.57234801943),
        ("isda_compounding", 5.58359355318),
    ],
)
def test_curve_rate_floating_spread(scm, exp):
    curve = Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.9985, dt(2022, 3, 1): 0.995})
    result = curve.rate(dt(2022, 1, 1), dt(2022, 3, 1), None, 250, scm)
    assert (result - exp) < 1e-8


def test_curve_rate_raises(curve):
    with pytest.raises(ValueError, match="Must supply a valid `spread_compound"):
        curve.rate(dt(2022, 3, 3), "7d", float_spread=10.0, spread_compound_method="bad")


@pytest.mark.parametrize(
    "li, ll, val, expected",
    [
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
    ],
)
def test_index_left(li, ll, val, expected):
    result = index_left(li, ll, val)
    assert result == expected


def test_zero_rate_plot():
    # test calcs without raise
    curve_zero = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.99,
            dt(2024, 1, 1): 0.979,
            dt(2025, 1, 1): 0.967,
        },
        interpolation="linear_zero_rate",
    )
    curve_zero.plot("1d")
    plt.close("all")


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
        t=[
            dt(2022, 5, 1),
            dt(2022, 5, 1),
            dt(2022, 5, 1),
            dt(2022, 5, 1),
            dt(2022, 6, 4),
            dt(2022, 7, 4),
            dt(2022, 7, 4),
            dt(2022, 7, 4),
            dt(2022, 7, 4),
        ],
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


@pytest.mark.parametrize(
    "attr, val",
    [
        ("nodes", {dt(2022, 3, 1): 1.00}),
        ("interpolation", "log_linear"),
        ("id", "x"),
        ("ad", 0),
        ("convention", "actact"),
        ("t", [dt(2022, 1, 1)]),
        ("calendar_type", "bad"),
    ],
)
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
        t=[
            dt(2022, 5, 1),
            dt(2022, 5, 1),
            dt(2022, 5, 1),
            dt(2022, 5, 1),
            dt(2022, 6, 4),
            dt(2022, 7, 4),
            dt(2022, 7, 4),
            dt(2022, 7, 4),
            dt(2022, 7, 4),
        ],
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
        t=[
            dt(2022, 5, 1),
            dt(2022, 5, 1),
            dt(2022, 5, 1),
            dt(2022, 5, 1),
            dt(2022, 6, 4),
            dt(2022, 7, 4),
            dt(2022, 7, 4),
            dt(2022, 7, 4),
            dt(2022, 7, 4),
        ],
    )
    curve2.nodes[dt(2022, 7, 4)] = 0.96  # set a specific node without recalc spline
    assert curve2 != curve  # should detect on curve2.spline.c


def test_curve_interp_raises():
    interp = "BAD"
    curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2022, 2, 1): 0.9,
        },
        id="curve",
        interpolation=interp,
    )

    err = '`interpolation` must be in {"linear", "log_linear", "linear_index'
    with pytest.raises(ValueError, match=err):
        curve[dt(2022, 1, 15)]


def test_curve_sorted_nodes_raises():
    err = "Curve node dates are not sorted or contain duplicates."
    with pytest.raises(ValueError, match=err):
        Curve(
            nodes={
                dt(2022, 2, 1): 0.9,
                dt(2022, 1, 1): 1.0,
            },
            id="curve",
        )


def test_interp_raises():
    interp = "linea"  # Wrongly spelled interpolation method
    err = '`interpolation` must be in {"linear", "log_linear", "linear_index'
    with pytest.raises(ValueError, match=err):
        interpolate(1.5, 1, 5, 2, 10, interp)


def test_curve_interp_case():
    curve_lower = Curve(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.99,
        },
        interpolation="log_linear",
        id="id",
        convention="Act360",
        ad=1,
    )
    curve_upper = Curve(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.99,
        },
        interpolation="LOG_LINEAR",
        id="id",
        convention="Act360",
        ad=1,
    )
    assert curve_lower[dt(2022, 3, 16)] == curve_upper[dt(2022, 3, 16)]


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
    assert result is None


@pytest.mark.parametrize(
    "endpoints, expected",
    [
        ("natural", [1.0, 0.995913396831872, 0.9480730429565414, 0.95]),
        ("not_a_knot", [1.0, 0.9967668788593117, 0.9461282456344617, 0.95]),
        (("not_a_knot", "natural"), [1.0, 0.9965809643843604, 0.9480575781858877, 0.95]),
        (("natural", "not_a_knot"), [1.0, 0.9959615881004005, 0.9461971628597721, 0.95]),
    ],
)
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
        t=[
            dt(2022, 1, 1),
            dt(2022, 1, 1),
            dt(2022, 1, 1),
            dt(2022, 1, 1),
            dt(2023, 1, 1),
            dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2026, 1, 1),
            dt(2026, 1, 1),
            dt(2026, 1, 1),
        ],
    )
    for i, date in enumerate([dt(2022, 1, 1), dt(2022, 7, 1), dt(2025, 7, 1), dt(2026, 1, 1)]):
        result = curve[date]
        assert (result - expected[i]) < 1e-12


@pytest.mark.parametrize("endpoints", [("natural", "bad"), ("bad", "natural")])
def test_spline_endpoints_raise(endpoints):
    with pytest.raises(NotImplementedError, match="Endpoint method"):
        Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.99,
                dt(2024, 1, 1): 0.97,
                dt(2025, 1, 1): 0.95,
                dt(2026, 1, 1): 0.95,
            },
            endpoints=endpoints,
            t=[
                dt(2022, 1, 1),
                dt(2022, 1, 1),
                dt(2022, 1, 1),
                dt(2022, 1, 1),
                dt(2023, 1, 1),
                dt(2024, 1, 1),
                dt(2025, 1, 1),
                dt(2026, 1, 1),
                dt(2026, 1, 1),
                dt(2026, 1, 1),
                dt(2026, 1, 1),
            ],
        )


def test_not_a_knot_raises():
    with pytest.raises(ValueError, match="`endpoints` cannot be 'not_a_knot'"):
        Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2024, 1, 1): 0.97,
                dt(2026, 1, 1): 0.95,
            },
            endpoints="not_a_knot",
            t=[
                dt(2022, 1, 1),
                dt(2022, 1, 1),
                dt(2022, 1, 1),
                dt(2022, 1, 1),
                dt(2024, 1, 1),
                dt(2026, 1, 1),
                dt(2026, 1, 1),
                dt(2026, 1, 1),
                dt(2026, 1, 1),
            ],
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
    assert curve[dt(2022, 1, 1)] == Dual(1.0, ["v0"], [])
    assert curve.ad == 1

    old_id = id(curve.nodes)
    curve._set_ad_order(2)
    assert curve[dt(2022, 1, 1)] == Dual2(1.0, ["v0"], [], [])
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
@pytest.mark.parametrize("composite", [True, False])
def test_curve_shift_ad_order(ad_order, composite):
    curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.988,
            dt(2024, 1, 1): 0.975,
            dt(2025, 1, 1): 0.965,
            dt(2026, 1, 1): 0.955,
            dt(2027, 1, 1): 0.9475,
        },
        t=[
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
        ],
        ad=ad_order,
    )
    result_curve = curve.shift(25, composite=composite)
    diff = np.array(
        [
            result_curve.rate(_, "1D") - curve.rate(_, "1D") - 0.25
            for _ in [dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)]
        ]
    )
    assert np.all(np.abs(diff) < 1e-7)


def test_curve_shift_association():
    args = (dt(2022, 2, 1), "1d")
    curve = Curve(
        nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.988},
    )
    solver = Solver(
        curves=[curve], instruments=[IRS(dt(2022, 1, 1), "1Y", "A", curves=curve)], s=[2.0]
    )
    base = curve.rate(*args)
    ass_shifted_curve = curve.shift(100)
    stat_shifted_curve = curve.shift(100, composite=False)
    assert abs(base - ass_shifted_curve.rate(*args) + 1.00) < 1e-5
    assert abs(base - stat_shifted_curve.rate(*args) + 1.00) < 1e-5

    solver.s = [3.0]
    solver.iterate()
    assert abs(ass_shifted_curve.rate(*args) - stat_shifted_curve.rate(*args)) > 0.95


def test_curve_shift_dual_input():
    curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.988,
            dt(2024, 1, 1): 0.975,
            dt(2025, 1, 1): 0.965,
            dt(2026, 1, 1): 0.955,
            dt(2027, 1, 1): 0.9475,
        },
        t=[
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
        ],
    )
    result_curve = curve.shift(Dual(25, ["z"], []))
    diff = np.array(
        [
            result_curve.rate(_, "1D") - curve.rate(_, "1D") - 0.25
            for _ in [dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)]
        ]
    )
    assert np.all(np.abs(diff) < 1e-7)


def test_composite_curve_shift():
    c1 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999})
    c2 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.998})
    cc = CompositeCurve([c1, c2])
    result = cc.shift(20, composite=False).rate(dt(2022, 1, 1), "1d")
    expected = c1.rate(dt(2022, 1, 1), "1d") + c2.rate(dt(2022, 1, 1), "1d") + 0.2
    assert abs(result - expected) < 1e-3


@pytest.mark.parametrize("ad_order", [0, 1, 2])
@pytest.mark.parametrize("composite", [True, False])
def test_linecurve_shift(ad_order, composite):
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
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
        ],
        ad=ad_order,
    )
    result_curve = curve.shift(25, composite=composite)
    diff = np.array(
        [
            result_curve[_] - curve[_] - 0.25
            for _ in [dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)]
        ]
    )
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
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
        ],
    )
    result_curve = curve.shift(Dual(25, ["z"], []))
    diff = np.array(
        [
            result_curve[_] - curve[_] - 0.25
            for _ in [dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)]
        ]
    )
    assert np.all(np.abs(diff) < 1e-7)


@pytest.mark.parametrize("ad_order", [0, 1, 2])
@pytest.mark.parametrize("composite", [True, False])
def test_indexcurve_shift(ad_order, composite):
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
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
        ],
        ad=ad_order,
        index_base=110.0,
        interpolation="log_linear",
    )
    result_curve = curve.shift(25, composite=composite)
    diff = np.array(
        [
            result_curve.rate(_, "1D") - curve.rate(_, "1D") - 0.25
            for _ in [dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)]
        ]
    )
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
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
        ],
        index_base=110.0,
        interpolation="log_linear",
    )
    result_curve = curve.shift(Dual(25, ["z"], []))
    diff = np.array(
        [
            result_curve.rate(_, "1D") - curve.rate(_, "1D") - 0.25
            for _ in [dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)]
        ]
    )
    assert np.all(np.abs(diff) < 1e-7)
    assert result_curve.index_base == curve.index_base


@pytest.mark.parametrize(
    "crv, t, tol",
    [
        (
            Curve(
                nodes={
                    dt(2022, 1, 1): 1.0,
                    dt(2023, 1, 1): 0.988,
                    dt(2024, 1, 1): 0.975,
                    dt(2025, 1, 1): 0.965,
                    dt(2026, 1, 1): 0.955,
                    dt(2027, 1, 1): 0.9475,
                },
                t=[
                    dt(2024, 1, 1),
                    dt(2024, 1, 1),
                    dt(2024, 1, 1),
                    dt(2024, 1, 1),
                    dt(2025, 1, 1),
                    dt(2026, 1, 1),
                    dt(2027, 1, 1),
                    dt(2027, 1, 1),
                    dt(2027, 1, 1),
                    dt(2027, 1, 1),
                ],
            ),
            False,
            1e-8,
        ),
        (
            IndexCurve(
                nodes={
                    dt(2022, 1, 1): 1.0,
                    dt(2023, 1, 1): 0.988,
                    dt(2024, 1, 1): 0.975,
                    dt(2025, 1, 1): 0.965,
                    dt(2026, 1, 1): 0.955,
                    dt(2027, 1, 1): 0.9475,
                },
                t=[
                    dt(2024, 1, 1),
                    dt(2024, 1, 1),
                    dt(2024, 1, 1),
                    dt(2024, 1, 1),
                    dt(2025, 1, 1),
                    dt(2026, 1, 1),
                    dt(2027, 1, 1),
                    dt(2027, 1, 1),
                    dt(2027, 1, 1),
                    dt(2027, 1, 1),
                ],
                index_base=110.0,
            ),
            False,
            1e-8,
        ),
        (
            LineCurve(
                nodes={
                    dt(2022, 1, 1): 1.7,
                    dt(2023, 1, 1): 1.65,
                    dt(2024, 1, 1): 1.4,
                    dt(2025, 1, 1): 1.3,
                    dt(2026, 1, 1): 1.25,
                    dt(2027, 1, 1): 1.35,
                },
                t=[
                    dt(2024, 1, 1),
                    dt(2024, 1, 1),
                    dt(2024, 1, 1),
                    dt(2024, 1, 1),
                    dt(2025, 1, 1),
                    dt(2026, 1, 1),
                    dt(2027, 1, 1),
                    dt(2027, 1, 1),
                    dt(2027, 1, 1),
                    dt(2027, 1, 1),
                ],
            ),
            False,
            1e-8,
        ),
        (
            Curve(
                nodes={
                    dt(2022, 1, 1): 1.0,
                    dt(2023, 1, 2): 0.988,
                    dt(2024, 1, 1): 0.975,
                    dt(2025, 1, 1): 0.965,
                    dt(2026, 1, 1): 0.955,
                    dt(2027, 1, 1): 0.9475,
                },
                t=[
                    dt(2022, 1, 1),
                    dt(2022, 1, 1),
                    dt(2022, 1, 1),
                    dt(2022, 1, 1),
                    dt(2023, 1, 2),
                    dt(2024, 1, 1),
                    dt(2025, 1, 1),
                    dt(2026, 1, 1),
                    dt(2027, 1, 1),
                    dt(2027, 1, 1),
                    dt(2027, 1, 1),
                    dt(2027, 1, 1),
                ],
            ),
            True,
            1e-3,
        ),
    ],
)
def test_curve_translate(crv, t, tol):
    result_curve = crv.translate(dt(2023, 1, 1), t=t)
    diff = np.array(
        [
            result_curve.rate(_, "1D") - crv.rate(_, "1D")
            for _ in [dt(2023, 1, 25), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)]
        ]
    )
    assert np.all(np.abs(diff) < tol)
    if type(crv) is IndexCurve:
        assert result_curve.index_base == crv.index_value(dt(2023, 1, 1))


@pytest.mark.parametrize(
    "crv",
    [
        Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.988,
                dt(2024, 1, 1): 0.975,
                dt(2025, 1, 1): 0.965,
                dt(2026, 1, 1): 0.955,
                dt(2027, 1, 1): 0.9475,
            },
            t=[
                dt(2024, 1, 1),
                dt(2024, 1, 1),
                dt(2024, 1, 1),
                dt(2024, 1, 1),
                dt(2025, 1, 1),
                dt(2026, 1, 1),
                dt(2027, 1, 1),
                dt(2027, 1, 1),
                dt(2027, 1, 1),
                dt(2027, 1, 1),
            ],
        ),
        LineCurve(
            nodes={
                dt(2022, 1, 1): 1.7,
                dt(2023, 1, 1): 1.65,
                dt(2024, 1, 1): 1.4,
                dt(2025, 1, 1): 1.3,
                dt(2026, 1, 1): 1.25,
                dt(2027, 1, 1): 1.35,
            },
            t=[
                dt(2024, 1, 1),
                dt(2024, 1, 1),
                dt(2024, 1, 1),
                dt(2024, 1, 1),
                dt(2025, 1, 1),
                dt(2026, 1, 1),
                dt(2027, 1, 1),
                dt(2027, 1, 1),
                dt(2027, 1, 1),
                dt(2027, 1, 1),
            ],
        ),
    ],
)
def test_curve_roll(crv):
    rolled_curve = crv.roll("10d")
    rolled_curve2 = crv.roll("-10d")

    expected = np.array(
        [
            crv.rate(_, "1D")
            for _ in [dt(2023, 1, 15), dt(2023, 3, 15), dt(2024, 11, 15), dt(2026, 4, 15)]
        ]
    )
    result = np.array(
        [
            rolled_curve.rate(_, "1D")
            for _ in [dt(2023, 1, 25), dt(2023, 3, 25), dt(2024, 11, 25), dt(2026, 4, 25)]
        ]
    )
    result2 = np.array(
        [
            rolled_curve2.rate(_, "1D")
            for _ in [dt(2023, 1, 5), dt(2023, 3, 5), dt(2024, 11, 5), dt(2026, 4, 5)]
        ]
    )
    assert np.all(np.abs(result - expected) < 1e-7)
    assert np.all(np.abs(result2 - expected) < 1e-7)


def test_curve_roll_copy(curve):
    result = curve.roll("0d")
    assert result == curve


def test_curve_spline_warning():
    curve = Curve(
        nodes={
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 0.99,
            dt(2025, 1, 1): 0.97,
            dt(2026, 1, 1): 0.94,
            dt(2027, 1, 1): 0.91,
        },
        t=[
            dt(2023, 1, 1),
            dt(2023, 1, 1),
            dt(2023, 1, 1),
            dt(2023, 1, 1),
            dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
        ],
    )
    with pytest.warns(UserWarning):
        curve[dt(2028, 1, 1)]


def test_index_curve_roll():
    crv = IndexCurve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.988,
            dt(2024, 1, 1): 0.975,
            dt(2025, 1, 1): 0.965,
            dt(2026, 1, 1): 0.955,
            dt(2027, 1, 1): 0.9475,
        },
        t=[
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2024, 1, 1),
            dt(2025, 1, 1),
            dt(2026, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
            dt(2027, 1, 1),
        ],
        index_base=110.0,
        interpolation="log_linear",
    )
    rolled_curve = crv.roll("10d")
    rolled_curve2 = crv.roll("-10d")

    expected = np.array(
        [
            crv.rate(_, "1D")
            for _ in [dt(2023, 1, 15), dt(2023, 3, 15), dt(2024, 11, 15), dt(2026, 4, 15)]
        ]
    )
    result = np.array(
        [
            rolled_curve.rate(_, "1D")
            for _ in [dt(2023, 1, 25), dt(2023, 3, 25), dt(2024, 11, 25), dt(2026, 4, 25)]
        ]
    )
    result2 = np.array(
        [
            rolled_curve2.rate(_, "1D")
            for _ in [dt(2023, 1, 5), dt(2023, 3, 5), dt(2024, 11, 5), dt(2026, 4, 5)]
        ]
    )
    assert np.all(np.abs(result - expected) < 1e-7)
    assert np.all(np.abs(result2 - expected) < 1e-7)
    assert rolled_curve.index_base == crv.index_base


def test_curve_translate_raises(curve):
    with pytest.raises(ValueError, match="Cannot translate into the past."):
        curve.translate(dt(2020, 4, 1))


def test_curve_zero_width_rate_raises(curve):
    with pytest.raises(ZeroDivisionError, match="effective:"):
        curve.rate(dt(2022, 3, 10), dt(2022, 3, 10))


class TestIndexCurve:
    def test_curve_translate_knots_raises(self, curve):
        curve = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.988,
                dt(2024, 1, 1): 0.975,
                dt(2025, 1, 1): 0.965,
                dt(2026, 1, 1): 0.955,
                dt(2027, 1, 1): 0.9475,
            },
            t=[
                dt(2022, 1, 1),
                dt(2022, 1, 1),
                dt(2022, 1, 1),
                dt(2022, 1, 1),
                dt(2022, 12, 1),
                dt(2024, 1, 1),
                dt(2025, 1, 1),
                dt(2026, 1, 1),
                dt(2027, 1, 1),
                dt(2027, 1, 1),
                dt(2027, 1, 1),
                dt(2027, 1, 1),
            ],
        )
        with pytest.raises(ValueError, match="Cannot translate spline knots for given"):
            curve.translate(dt(2022, 12, 15))

    def test_curve_index_linear_daily_interp(self):
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

    def test_indexcurve_raises(self):
        with pytest.raises(ValueError, match="`index_base` must be given"):
            IndexCurve({dt(2022, 1, 1): 1.0})

    def test_index_value_raises(self):
        curve = IndexCurve({dt(2022, 1, 1): 1.0}, index_base=100.0)
        with pytest.raises(ValueError, match="`interpolation` for `index_value`"):
            curve.index_value(dt(2022, 1, 1), interpolation="BAD")

    @pytest.mark.parametrize("ad", [0, 1, 2])
    def test_roll_preserves_ad(self, ad):
        curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_base=100.0,
            index_lag=3,
            id="tags_",
            ad=ad,
        )
        new_curve = curve.roll("1m")
        assert new_curve.ad == curve.ad

    def test_historic_rate_is_none(self):
        curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_base=100.0,
            index_lag=3,
            id="tags_",
        )
        assert curve.rate(dt(2021, 3, 4), "1b", "f") is None


class TestCompositeCurve:
    def test_curve_df_based(self):
        curve1 = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.98,
                dt(2024, 1, 1): 0.965,
                dt(2025, 1, 1): 0.955,
            },
            t=[
                dt(2023, 1, 1),
                dt(2023, 1, 1),
                dt(2023, 1, 1),
                dt(2023, 1, 1),
                dt(2024, 1, 1),
                dt(2025, 1, 1),
                dt(2025, 1, 1),
                dt(2025, 1, 1),
                dt(2025, 1, 1),
            ],
        )
        curve2 = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2022, 6, 30): 1.0,
                dt(2022, 7, 1): 0.999992,
                dt(2022, 12, 31): 0.999992,
                dt(2023, 1, 1): 0.999984,
                dt(2023, 6, 30): 0.999984,
                dt(2023, 7, 1): 0.999976,
                dt(2023, 12, 31): 0.999976,
                dt(2024, 1, 1): 0.999968,
                dt(2024, 6, 30): 0.999968,
                dt(2024, 7, 1): 0.999960,
                dt(2025, 1, 1): 0.999960,
            },
        )
        curve = CompositeCurve([curve1, curve2])

        result1 = curve.rate(dt(2022, 12, 30), "1d")
        result2 = curve.rate(dt(2022, 12, 31), "1d")
        result3 = curve.rate(dt(2023, 1, 1), "1d")

        expected1 = curve.rate(dt(2022, 12, 30), "1d", approximate=False)
        expected2 = curve.rate(dt(2022, 12, 31), "1d", approximate=False)
        expected3 = curve.rate(dt(2023, 1, 1), "1d", approximate=False)

        assert abs(result1 - expected1) < 1e-9
        assert abs(result2 - expected2) < 1e-9
        assert abs(result3 - expected3) < 1e-9

        result = curve.rate(dt(2022, 6, 1), "1Y")
        expected = curve.rate(dt(2022, 6, 1), "1Y", approximate=False)
        assert abs(result - expected) < 1e-4

    def test_composite_curve_translate(self):
        curve1 = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.98,
                dt(2024, 1, 1): 0.965,
                dt(2025, 1, 1): 0.955,
            },
            t=[
                dt(2023, 1, 1),
                dt(2023, 1, 1),
                dt(2023, 1, 1),
                dt(2023, 1, 1),
                dt(2024, 1, 1),
                dt(2025, 1, 1),
                dt(2025, 1, 1),
                dt(2025, 1, 1),
                dt(2025, 1, 1),
            ],
        )
        curve2 = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2022, 6, 30): 1.0,
                dt(2022, 7, 1): 0.999992,
                dt(2022, 12, 31): 0.999992,
                dt(2023, 1, 1): 0.999984,
                dt(2023, 6, 30): 0.999984,
                dt(2023, 7, 1): 0.999976,
                dt(2023, 12, 31): 0.999976,
                dt(2024, 1, 1): 0.999968,
                dt(2024, 6, 30): 0.999968,
                dt(2024, 7, 1): 0.999960,
                dt(2025, 1, 1): 0.999960,
            },
        )
        crv = CompositeCurve([curve1, curve2])

        result_curve = crv.translate(dt(2022, 3, 1))
        diff = np.array(
            [
                result_curve.rate(_, "1D") - crv.rate(_, "1D")
                for _ in [dt(2023, 1, 25), dt(2023, 3, 24), dt(2024, 11, 11)]
            ]
        )
        assert np.all(np.abs(diff) < 1e-5)

    def test_composite_curve_roll(self):
        curve1 = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.98,
                dt(2024, 1, 1): 0.965,
                dt(2025, 1, 1): 0.955,
            },
            t=[
                dt(2023, 1, 1),
                dt(2023, 1, 1),
                dt(2023, 1, 1),
                dt(2023, 1, 1),
                dt(2024, 1, 1),
                dt(2025, 1, 1),
                dt(2025, 1, 1),
                dt(2025, 1, 1),
                dt(2025, 1, 1),
            ],
        )
        curve2 = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2022, 6, 30): 1.0,
                dt(2022, 7, 1): 0.999992,
                dt(2022, 12, 31): 0.999992,
                dt(2023, 1, 1): 0.999984,
                dt(2023, 6, 30): 0.999984,
                dt(2023, 7, 1): 0.999976,
                dt(2023, 12, 31): 0.999976,
                dt(2024, 1, 1): 0.999968,
                dt(2024, 6, 30): 0.999968,
                dt(2024, 7, 1): 0.999960,
                dt(2025, 1, 1): 0.999960,
            },
        )
        crv = CompositeCurve([curve1, curve2])

        rolled_curve = crv.roll("10d")
        expected = np.array(
            [crv.rate(_, "1D") for _ in [dt(2023, 1, 15), dt(2023, 3, 15), dt(2024, 11, 15)]]
        )
        result = np.array(
            [
                rolled_curve.rate(_, "1D")
                for _ in [dt(2023, 1, 25), dt(2023, 3, 25), dt(2024, 11, 25)]
            ]
        )

        assert np.all(np.abs(result - expected) < 1e-7)

    def test_isinstance_raises(self):
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99})
        line_curve = LineCurve({dt(2022, 1, 1): 10.0, dt(2023, 1, 1): 12.0})
        with pytest.raises(TypeError, match="`curves` must be a list of"):
            CompositeCurve([curve, line_curve])

    @pytest.mark.parametrize(
        "attribute, val",
        [
            ("modifier", ["MF", "MP"]),
            ("calendar", ["ldn", "tgt"]),
            ("convention", ["act360", "act365f"]),
        ],
    )
    def test_attribute_error_raises(self, attribute, val):
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, **{attribute: val[0]})
        c2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, **{attribute: val[1]})
        with pytest.raises(ValueError, match="Cannot composite curves with dif"):
            CompositeCurve([c1, c2])

    def test_line_based(self):
        c1 = LineCurve({dt(2022, 1, 1): 1.5, dt(2022, 1, 3): 1.0})
        c2 = LineCurve({dt(2022, 1, 1): 2.0, dt(2022, 1, 3): 3.0})
        cc = CompositeCurve([c1, c2])
        expected = 3.75
        result = cc.rate(dt(2022, 1, 2))
        assert abs(result - expected) < 1e-8

        result = cc[dt(2022, 1, 2)]
        assert abs(result - expected) < 1e-8

    def test_initial_node_raises(self):
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99})
        c2 = Curve({dt(2022, 1, 2): 1.0, dt(2023, 1, 1): 0.99})
        with pytest.raises(ValueError, match="`curves` must share the same ini"):
            CompositeCurve([c1, c2])

    @pytest.mark.parametrize("lag, base", [([2, 3], [100.0, 100.0]), ([3, 3], [100.0, 100.001])])
    def test_index_curves_raises(self, lag, base):
        ic1 = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_lag=lag[0], index_base=base[0]
        )
        ic2 = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_lag=lag[1], index_base=base[1]
        )
        with pytest.raises(ValueError, match="Cannot composite curves with different"):
            CompositeCurve([ic1, ic2])

    def test_index_curves_attributes(self):
        ic1 = IndexCurve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_lag=3, index_base=101.1)
        ic2 = IndexCurve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_lag=3, index_base=101.1)
        cc = CompositeCurve([ic1, ic2])
        assert cc.index_lag == 3
        assert cc.index_base == 101.1

        result = cc.index_value(dt(2022, 1, 31), interpolation="monthly")
        expected = 101.1
        assert abs(result - expected) < 1e-5

        result = cc.index_value(dt(1999, 1, 1))
        expected = 0.0
        assert abs(result - expected) < 1e-5

        result = cc.index_value(dt(2022, 1, 1))
        expected = 101.1
        assert abs(result - expected) < 1e-5

    def test_index_curves_interp_raises(self):
        ic1 = IndexCurve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_lag=3, index_base=101.1)
        ic2 = IndexCurve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_lag=3, index_base=101.1)
        cc = CompositeCurve([ic1, ic2])
        with pytest.raises(ValueError, match="`interpolation` for `index_value` must"):
            cc.index_value(dt(2022, 1, 31), interpolation="bad interp")

    def test_multi_csa(self):
        c1 = Curve(
            {
                dt(2022, 1, 1): 1.0,
                dt(2022, 1, 2): 0.99997260,  # 1%
                dt(2022, 1, 3): 0.99991781,  # 2%
                dt(2022, 1, 4): 0.99983564,  # 3%
                dt(2022, 1, 5): 0.99972608,  # 4%
            },
            convention="Act365F",
        )
        c2 = Curve(
            {
                dt(2022, 1, 1): 1.0,
                dt(2022, 1, 2): 0.99989042,  # 4%
                dt(2022, 1, 3): 0.99980825,  # 3%
                dt(2022, 1, 4): 0.99975347,  # 2%
                dt(2022, 1, 5): 0.99972608,  # 1%
            },
            convention="Act365F",
        )
        c3 = Curve(
            {
                dt(2022, 1, 1): 1.0,
                dt(2022, 1, 2): 0.99989042,  # 4%
                dt(2022, 1, 3): 0.99979455,  # 3.5%
                dt(2022, 1, 4): 0.99969869,  # 3.5%
                dt(2022, 1, 5): 0.99958915,  # 4%
            },
            convention="Act365F",
        )
        cc = MultiCsaCurve([c1, c2, c3])
        with default_context("multi_csa_steps", [1, 1, 1, 1, 1, 1, 1]):
            r1 = cc.rate(dt(2022, 1, 1), "1d")
            r2 = cc.rate(dt(2022, 1, 2), "1d")
            r3 = cc.rate(dt(2022, 1, 3), "1d")
            r4 = cc.rate(dt(2022, 1, 4), "1d")

        assert abs(r1 - 4.0) < 1e-3
        assert abs(r2 - 3.5) < 1e-3
        assert abs(r3 - 3.5) < 1e-3
        assert abs(r4 - 4.0) < 1e-3

    def test_multi_csa_granularity(self):
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 0.9, dt(2072, 1, 1): 0.5})
        c2 = Curve({dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 0.8, dt(2072, 1, 1): 0.7})
        cc = MultiCsaCurve([c1, c2], multi_csa_max_step=182, multi_csa_min_step=182)

        r1 = cc.rate(dt(2052, 5, 24), "1d")
        # r2 = cc.rate(dt(2052, 5, 25), "1d")
        # r3 = cc.rate(dt(2052, 5, 26), "1d")

        assert abs(r1 - 1.448374) < 1e-3

    def test_composite_curve_proxies(self):
        uu = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="uu")
        ee = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.991}, id="ee")
        eu = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.992}, id="eu")
        fxf = FXForwards(
            fx_rates=FXRates({"eurusd": 1.1}, settlement=dt(2022, 1, 1)),
            fx_curves={
                "usdusd": uu,
                "eureur": ee,
                "eurusd": eu,
            },
        )
        pc = MultiCsaCurve([uu, fxf.curve("usd", "eur")])
        result = pc[dt(2023, 1, 1)]
        expected = 0.98900
        assert abs(result - expected) < 1e-4

        pc = MultiCsaCurve(
            [
                fxf.curve("usd", "eur"),
                uu,
            ],
        )
        result = pc[dt(2023, 1, 1)]
        assert abs(result - expected) < 1e-4

    def test_multi_raises(self, line_curve, curve):
        with pytest.raises(TypeError, match="Multi-CSA curves must"):
            MultiCsaCurve([line_curve])

        with pytest.raises(ValueError, match="`multi_csa_max_step` cannot be less "):
            MultiCsaCurve([curve], multi_csa_max_step=3, multi_csa_min_step=4)

    def test_multi_csa_shift(self):
        c1 = Curve(
            {
                dt(2022, 1, 1): 1.0,
                dt(2022, 1, 2): 0.99997260,  # 1%
                dt(2022, 1, 3): 0.99991781,  # 2%
                dt(2022, 1, 4): 0.99983564,  # 3%
                dt(2022, 1, 5): 0.99972608,  # 4%
            },
            convention="Act365F",
        )
        c2 = Curve(
            {
                dt(2022, 1, 1): 1.0,
                dt(2022, 1, 2): 0.99989042,  # 4%
                dt(2022, 1, 3): 0.99980825,  # 3%
                dt(2022, 1, 4): 0.99975347,  # 2%
                dt(2022, 1, 5): 0.99972608,  # 1%
            },
            convention="Act365F",
        )
        c3 = Curve(
            {
                dt(2022, 1, 1): 1.0,
                dt(2022, 1, 2): 0.99989042,  # 4%
                dt(2022, 1, 3): 0.99979455,  # 3.5%
                dt(2022, 1, 4): 0.99969869,  # 3.5%
                dt(2022, 1, 5): 0.99958915,  # 4%
            },
            convention="Act365F",
        )
        cc = MultiCsaCurve([c1, c2, c3])
        cc_shift = cc.shift(100, composite=False)
        with default_context("multi_csa_steps", [1, 1, 1, 1, 1, 1, 1]):
            r1 = cc_shift.rate(dt(2022, 1, 1), "1d")
            r2 = cc_shift.rate(dt(2022, 1, 2), "1d")
            r3 = cc_shift.rate(dt(2022, 1, 3), "1d")
            r4 = cc_shift.rate(dt(2022, 1, 4), "1d")

        assert abs(r1 - 5.0) < 1e-3
        assert abs(r2 - 4.5) < 1e-3
        assert abs(r3 - 4.5) < 1e-3
        assert abs(r4 - 5.0) < 1e-3

    def test_composite_curve_no_index_value_raises(self, curve):
        cc = CompositeCurve([curve])
        with pytest.raises(TypeError, match="`index_value` not available"):
            cc.index_value(dt(2022, 1, 1))

    def test_historic_rate_is_none(self):
        c1 = Curve(
            {
                dt(2022, 1, 1): 1.0,
                dt(2022, 1, 2): 0.99997260,  # 1%
                dt(2022, 1, 3): 0.99991781,  # 2%
                dt(2022, 1, 4): 0.99983564,  # 3%
                dt(2022, 1, 5): 0.99972608,  # 4%
            },
            convention="Act365F",
        )
        c2 = Curve(
            {
                dt(2022, 1, 1): 1.0,
                dt(2022, 1, 2): 0.99989042,  # 4%
                dt(2022, 1, 3): 0.99980825,  # 3%
                dt(2022, 1, 4): 0.99975347,  # 2%
                dt(2022, 1, 5): 0.99972608,  # 1%
            },
            convention="Act365F",
        )
        cc = CompositeCurve([c1, c2])
        assert cc.rate(dt(2021, 3, 4), "1b", "f") is None


class TestMultiCsaCurve:
    def test_historic_rate_is_none(self):
        c1 = Curve(
            {
                dt(2022, 1, 1): 1.0,
                dt(2022, 1, 2): 0.99997260,  # 1%
                dt(2022, 1, 3): 0.99991781,  # 2%
                dt(2022, 1, 4): 0.99983564,  # 3%
                dt(2022, 1, 5): 0.99972608,  # 4%
            },
            convention="Act365F",
        )
        c2 = Curve(
            {
                dt(2022, 1, 1): 1.0,
                dt(2022, 1, 2): 0.99989042,  # 4%
                dt(2022, 1, 3): 0.99980825,  # 3%
                dt(2022, 1, 4): 0.99975347,  # 2%
                dt(2022, 1, 5): 0.99972608,  # 1%
            },
            convention="Act365F",
        )
        cc = MultiCsaCurve([c1, c2])
        assert cc.rate(dt(2021, 3, 4), "1b", "f") is None


class TestPlotCurve:
    def test_plot_curve(self, curve):
        fig, ax, lines = curve.plot("1d")
        result = lines[0].get_data()
        assert result[0][0] == dt(2022, 3, 1)
        assert abs(result[1][0].real - 12.004001333774994) < 1e-6
        plt.close("all")

    def test_plot_linecurve(self, line_curve):
        fig, ax, lines = line_curve.plot("0d")
        result = lines[0].get_data()
        assert result[0][0] == dt(2022, 3, 1)
        assert abs(result[1][0].real - 2.0) < 1e-6
        plt.close("all")

    @pytest.mark.parametrize("left", ["1d", dt(2022, 3, 2)])
    def test_plot_curve_left(self, curve, left):
        fig, ax, lines = curve.plot("1d", left=left)
        result = lines[0].get_data()
        assert result[0][0] == dt(2022, 3, 2)
        assert abs(result[1][0].real - 12.008005336896055) < 1e-6
        plt.close("all")

    def test_plot_curve_left_raise(self, curve):
        with pytest.raises(ValueError, match="`left` must be supplied as"):
            fig, ax, lines = curve.plot("1d", left=100.3)
        plt.close("all")

    @pytest.mark.parametrize("right", ["2d", dt(2022, 3, 3)])
    def test_plot_curve_right(self, curve, right):
        fig, ax, lines = curve.plot("1d", right=right)
        result = lines[0].get_data()
        assert result[0][-1] == dt(2022, 3, 3)
        assert abs(result[1][-1].real - 12.012012012015738) < 1e-6
        plt.close("all")

    def test_plot_curve_right_raise(self, curve):
        with pytest.raises(ValueError, match="`right` must be supplied as"):
            fig, ax, lines = curve.plot("1d", right=100.3)
        plt.close("all")

    def test_plot_comparators(self, curve):
        fig, ax, lines = curve.plot("1d", comparators=[curve])
        assert len(lines) == 2
        res1 = lines[0].get_data()
        res2 = lines[1].get_data()
        assert res1[0][0] == res2[0][0]
        assert res1[1][0] == res2[1][0]
        plt.close("all")

    def test_plot_diff(self, curve):
        fig, ax, lines = curve.plot("1d", comparators=[curve], difference=True)
        assert len(lines) == 1
        result = lines[0].get_data()
        assert result[0][0] == dt(2022, 3, 1)
        assert result[1][0] == 0
        plt.close("all")

    @pytest.mark.parametrize("left", [NoInput(0), dt(2022, 1, 1), "0d"])
    @pytest.mark.parametrize("right", [NoInput(0), dt(2022, 2, 1), "0d"])
    def test_plot_index(self, left, right):
        i_curve = IndexCurve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 1.0}, index_base=2.0)
        fig, ax, lines = i_curve.plot_index(left=left, right=right)
        result = lines[0].get_data()
        assert result[0][0] == dt(2022, 1, 1)
        assert abs(result[1][0].real - 2.0) < 1e-6
        plt.close("all")

    def test_plot_index_comparators(self):
        i_curve = IndexCurve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 1.0}, index_base=2.0)
        i_curv2 = IndexCurve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 1.0}, index_base=2.0)
        fig, ax, lines = i_curve.plot_index(comparators=[i_curv2])
        assert len(lines) == 2
        res1 = lines[0].get_data()
        res2 = lines[1].get_data()
        assert res1[0][0] == res2[0][0]
        assert res1[1][0] == res2[1][0]
        plt.close("all")

    def test_plot_index_diff(self):
        i_curv = IndexCurve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 1.0}, index_base=2.0)
        i_curv2 = IndexCurve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 1.0}, index_base=2.0)
        fig, ax, lines = i_curv.plot_index("1d", comparators=[i_curv2], difference=True)
        assert len(lines) == 1
        result = lines[0].get_data()
        assert result[0][0] == dt(2022, 1, 1)
        assert result[1][0] == 0
        plt.close("all")

    def test_plot_index_raises(self):
        i_curve = IndexCurve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 1.0}, index_base=2.0)
        with pytest.raises(ValueError, match="`left` must be supplied as"):
            i_curve.plot_index(left=2.0)
        with pytest.raises(ValueError, match="`right` must be supplied as"):
            i_curve.plot_index(right=2.0)

    def test_composite_curve_plot(self):
        curve1 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 12, 1): 0.95}, modifier="MF", calendar="bus")
        curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 12, 1): 0.97}, modifier="MF", calendar="bus")
        cc = CompositeCurve(curves=[curve1, curve2])
        cc.plot("1m")

    def test_plot_a_rolled_spline_curve(self):
        curve = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.988,
                dt(2024, 1, 1): 0.975,
                dt(2025, 1, 1): 0.965,
                dt(2026, 1, 1): 0.955,
                dt(2027, 1, 1): 0.9475,
            },
            t=[
                dt(2024, 1, 1),
                dt(2024, 1, 1),
                dt(2024, 1, 1),
                dt(2024, 1, 1),
                dt(2025, 1, 1),
                dt(2026, 1, 1),
                dt(2027, 1, 1),
                dt(2027, 1, 1),
                dt(2027, 1, 1),
                dt(2027, 1, 1),
            ],
        )
        rolled_curve = curve.roll("6m")
        rolled_curve2 = curve.roll("-6m")
        curve.plot(
            "1d",
            comparators=[rolled_curve, rolled_curve2],
            labels=["orig", "rolled", "rolled2"],
            right=dt(2026, 6, 30),
        )
        usd_curve = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 7, 1): 0.98, dt(2023, 1, 1): 0.95},
            calendar="nyc",
            id="sofr",
        )
        usd_args = dict(effective=dt(2022, 1, 1), spec="usd_irs", curves="sofr")
        Solver(
            curves=[usd_curve],
            instruments=[
                IRS(**usd_args, termination="6M"),
                IRS(**usd_args, termination="1Y"),
            ],
            s=[4.35, 4.85],
            instrument_labels=["6M", "1Y"],
            id="us_rates",
        )
        usd_curve.plot("1b", labels=["SOFR o/n"])
