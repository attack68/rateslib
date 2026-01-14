# SPDX-License-Identifier: LicenseRef-Rateslib-Dual
#
# Copyright (c) 2026 Siffrorna Technology Limited
#
# Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
# Source-available, not open source.
#
# See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
# and/or contact info (at) rateslib (dot) com
####################################################################################################

from datetime import datetime as dt
from math import exp, log

import numpy as np
import pytest
from matplotlib import pyplot as plt
from pandas import Series
from rateslib import default_context, defaults, fixings
from rateslib.curves import (
    CompositeCurve,
    Curve,
    LineCurve,
    MultiCsaCurve,
    average_rate,
    index_left,
    index_value,
)
from rateslib.curves.curves import CreditImpliedCurve, _BaseCurve, _CurveMeta, _try_index_value
from rateslib.curves.utils import _CurveNodes, _CurveSpline
from rateslib.data.loader import FixingMissingDataError
from rateslib.dual import Dual, Dual2, Variable, gradient
from rateslib.dual.utils import _get_order_of
from rateslib.enums.generics import Err, NoInput, Ok
from rateslib.fx import FXForwards, FXRates
from rateslib.instruments import IRS
from rateslib.scheduling import Cal, dcf, get_calendar
from rateslib.solver import Solver


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
def index_curve():
    return Curve(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.999,
        },
        interpolation="linear_index",
        id="v",
        ad=1,
        index_base=110.0,
    )


def test_meta_attribute(curve, line_curve):
    assert isinstance(curve._meta, _CurveMeta)
    assert isinstance(line_curve._meta, _CurveMeta)


@pytest.mark.parametrize("method", ["flat_forward", "flat_backward"])
def test_flat_interp(method) -> None:
    curve = Curve(
        {dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.9, dt(2002, 1, 1): 0.8},
        interpolation=method,
    )
    assert curve[dt(2000, 1, 1)] == 1.0
    assert curve[dt(2001, 1, 1)] == 0.9
    assert curve[dt(2002, 1, 1)] == 0.8

    if method == "flat_forward":
        assert curve[dt(2000, 7, 1)] == 1.0
    else:
        assert curve[dt(2000, 7, 1)] == 0.9


@pytest.mark.parametrize(("curve_style", "expected"), [("df", 0.995), ("line", 2.005)])
def test_linear_interp(curve_style, expected, curve, line_curve) -> None:
    if curve_style == "df":
        obj = curve
    else:
        obj = line_curve
    result = obj[dt(2022, 3, 16)]
    assert abs(result - Dual(expected, ["v1", "v0"], [0.5, 0.5])) < 1e-10
    assert np.all(np.isclose(result.dual, np.array([0.5, 0.5])))


def test_log_linear_interp() -> None:
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


def test_linear_zero_rate_interp() -> None:
    # not tested
    pass


def test_line_curve_rate(line_curve) -> None:
    expected = Dual(2.005, ["v0", "v1"], [0.5, 0.5])
    result = line_curve.rate(effective=dt(2022, 3, 16))
    assert abs(result - expected) < 1e-10
    assert np.all(np.isclose(result.dual, np.array([0.5, 0.5])))


@pytest.mark.parametrize(
    ("scm", "exp"),
    [
        ("none_simple", 5.56617834937),
        ("isda_flat_compounding", 5.57234801943),
        ("isda_compounding", 5.58359355318),
    ],
)
def test_curve_rate_floating_spread(scm, exp) -> None:
    curve = Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.9985, dt(2022, 3, 1): 0.995})
    result = curve.rate(dt(2022, 1, 1), dt(2022, 3, 1), None, 250, scm)
    assert (result - exp) < 1e-8


def test_curve_rate_raises(curve) -> None:
    with pytest.raises(ValueError, match="Must supply a valid `spread_compound"):
        curve.rate(dt(2022, 3, 3), "7d", float_spread=10.0, spread_compound_method="bad")


@pytest.mark.parametrize(
    ("li", "ll", "val", "expected"),
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
def test_index_left(li, ll, val, expected) -> None:
    result = index_left(li, ll, val)
    assert result == expected


def test_zero_rate_plot() -> None:
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


def test_curve_equality_type_differ(curve, line_curve) -> None:
    assert curve != line_curve


def test_copy_curve(curve, line_curve) -> None:
    copied = curve.copy()
    assert copied == curve
    assert id(copied) != id(curve)

    copied = line_curve.copy()
    assert copied == line_curve
    assert id(copied) != id(line_curve)


@pytest.mark.parametrize(
    ("attr", "val"),
    [
        ("_nodes", _CurveNodes({dt(2000, 1, 1): 1.0})),
        ("_interpolator", "some_value"),
        ("_id", "x"),
        ("_ad", 0),
        ("_meta", "some_value"),
    ],
)
def test_curve_equality_checks(attr, val, curve) -> None:
    copied_curve = curve.copy()
    assert copied_curve == curve
    setattr(copied_curve, attr, val)
    assert copied_curve != curve


def test_curve_equality_spline_coeffs() -> None:
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
    assert curve2 != curve  # should detect on curve2.spline.c
    curve2.update_node(dt(2022, 7, 4), 0.96)
    assert curve2 == curve  # spline.c will be resolved on calculation to the same values


def test_curve_interp_raises() -> None:
    interp = "BAD"

    err = "Curve interpolation: 'bad' not ava"
    with pytest.raises(ValueError, match=err):
        Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2022, 2, 1): 0.9,
            },
            id="curve",
            interpolation=interp,
        )


def test_curve_sorted_nodes_raises() -> None:
    err = "Curve node dates are not sorted or contain duplicates."
    with pytest.raises(ValueError, match=err):
        Curve(
            nodes={
                dt(2022, 2, 1): 0.9,
                dt(2022, 1, 1): 1.0,
            },
            id="curve",
        )


def test_curve_interp_case() -> None:
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


def test_custom_interpolator() -> None:
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


def test_df_is_zero_in_past(curve) -> None:
    assert curve[dt(1999, 1, 1)] == 0.0


def test_curve_none_return(curve) -> None:
    result = curve.rate(dt(2022, 2, 1), dt(2022, 2, 2))
    assert result is None


@pytest.mark.parametrize(
    ("endpoints", "expected"),
    [
        ("natural", [1.0, 0.995913396831872, 0.9480730429565414, 0.95]),
        ("not_a_knot", [1.0, 0.9967668788593117, 0.9461282456344617, 0.95]),
        (("not_a_knot", "natural"), [1.0, 0.9965809643843604, 0.9480575781858877, 0.95]),
        (("natural", "not_a_knot"), [1.0, 0.9959615881004005, 0.9461971628597721, 0.95]),
    ],
)
def test_spline_endpoints(endpoints, expected) -> None:
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
def test_spline_endpoints_raise(endpoints) -> None:
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


def test_not_a_knot_raises() -> None:
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


def test_set_ad_order_no_spline() -> None:
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


def test_set_ad_order_raises(curve) -> None:
    with pytest.raises(ValueError, match="`order` can only be in {0, 1, 2}"):
        curve._set_ad_order(100)


def test_index_left_raises() -> None:
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
# @pytest.mark.parametrize("composite", [True, False])
def test_curve_shift_ad_order(ad_order) -> None:
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
    result_curve = curve.shift(25)
    diff = np.array(
        [
            result_curve.rate(_, "1D") - curve.rate(_, "1D") - 0.25
            for _ in [dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)]
        ],
    )
    assert np.all(np.abs(diff) < 1e-7)

    result_curve._set_ad_order((ad_order + 1) % 3)
    assert result_curve.ad == (ad_order + 1) % 3


@pytest.mark.skip(reason="composite argument removed from shift method in v2.1")
def test_curve_shift_association() -> None:
    # test a dynamic shift association with curves, active after a Solver mutation
    args = (dt(2022, 2, 1), "1d")
    curve = Curve(
        nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.988},
    )
    solver = Solver(
        curves=[curve],
        instruments=[IRS(dt(2022, 1, 1), "1Y", "A", curves=curve)],
        s=[2.0],
    )
    base = curve.rate(*args)
    ass_shifted_curve = curve.shift(100)
    stat_shifted_curve = curve.shift(100, composite=False)
    assert abs(base - ass_shifted_curve.rate(*args) + 1.00) < 1e-5
    assert abs(base - stat_shifted_curve.rate(*args) + 1.00) < 1e-5

    solver.s = [3.0]
    solver.iterate()
    base = curve.rate(*args)
    assert abs(base - ass_shifted_curve.rate(*args) + 1.00) < 1e-5
    assert abs(ass_shifted_curve.rate(*args) - stat_shifted_curve.rate(*args)) > 0.95


def test_curve_shift_dual_input() -> None:
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
        ],
    )
    assert np.all(np.abs(diff) < 1e-7)


def test_composite_curve_shift() -> None:
    c1 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999})
    c2 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.998})
    cc = CompositeCurve([c1, c2])
    result = cc.shift(20).rate(dt(2022, 1, 1), "1d")
    expected = c1.rate(dt(2022, 1, 1), "1d") + c2.rate(dt(2022, 1, 1), "1d") + 0.2
    assert abs(result - expected) < 1e-3


@pytest.mark.parametrize("ad_order", [0, 1, 2])
# @pytest.mark.parametrize("composite", [True, False])
def test_linecurve_shift(ad_order) -> None:
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
    result_curve = curve.shift(25)
    diff = np.array(
        [
            result_curve[_] - curve[_] - 0.25
            for _ in [dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)]
        ],
    )
    assert np.all(np.abs(diff) < 1e-7)


def test_linecurve_shift_dual_input() -> None:
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
        ],
    )
    assert np.all(np.abs(diff) < 1e-7)


@pytest.mark.parametrize("ad_order", [0, 1, 2])
# @pytest.mark.parametrize("composite", [True, False])
def test_indexcurve_shift(ad_order) -> None:
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
        index_base=110.0,
        interpolation="log_linear",
    )
    result_curve = curve.shift(25)
    diff = np.array(
        [
            result_curve.rate(_, "1D") - curve.rate(_, "1D") - 0.25
            for _ in [dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)]
        ],
    )
    assert np.all(np.abs(diff) < 1e-7)
    assert result_curve.meta.index_base == curve.meta.index_base


def test_indexcurve_shift_dual_input() -> None:
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
        index_base=110.0,
        interpolation="log_linear",
    )
    result_curve = curve.shift(Dual(25, ["z"], []))
    diff = np.array(
        [
            result_curve.rate(_, "1D") - curve.rate(_, "1D") - 0.25
            for _ in [dt(2022, 1, 10), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)]
        ],
    )
    assert np.all(np.abs(diff) < 1e-7)
    assert result_curve.meta.index_base == curve.meta.index_base


@pytest.mark.parametrize("c_obj", ["c", "l", "i"])
@pytest.mark.parametrize("ini_ad", [0, 1, 2])
@pytest.mark.parametrize(
    "spread", [1.0, Dual(1.0, ["z"], []), Dual2(1.0, ["z"], [], []), Variable(1.0, ["z"])]
)
# @pytest.mark.parametrize("composite", [False, True])
def test_curve_shift_ad_orders(curve, line_curve, index_curve, c_obj, ini_ad, spread):
    if c_obj == "c":
        c = curve
    elif c_obj == "l":
        c = line_curve
    else:
        c = index_curve
    c._set_ad_order(ini_ad)

    if ini_ad + _get_order_of(spread) == 3:
        with pytest.raises(TypeError, match="Cannot create a ShiftedCurve with mixed AD orders"):
            c.shift(spread)
        return None

    result = c.shift(spread)
    expected = max(_get_order_of(spread), ini_ad)
    assert result._ad == expected


@pytest.mark.parametrize(
    ("crv", "tol"),
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
            1e-8,
        ),
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
                index_base=110.0,
            ),
            1e-8,
        ),
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
                index_base=110.0,
                interpolation="linear_index",
            ),
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
            1e-3,
        ),
    ],
)
def test_curve_translate(crv, tol) -> None:
    result_curve = crv.translate(dt(2023, 1, 1))
    diff = np.array(
        [
            result_curve.rate(_, "1D") - crv.rate(_, "1D")
            for _ in [dt(2023, 1, 25), dt(2023, 3, 24), dt(2024, 11, 11), dt(2026, 4, 5)]
        ],
    )
    assert np.all(np.abs(diff) < tol)
    if not isinstance(result_curve.meta.index_base, NoInput):
        projected_base = crv.index_value(dt(2023, 1, 1), crv.meta.index_lag)
        assert abs(result_curve.meta.index_base - projected_base) < 1e-14

    # test date between original initial and translated initial is zero
    assert result_curve[dt(1900, 1, 1)] == 0.0
    assert result_curve[dt(2022, 12, 31)] == 0.0


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
@pytest.mark.parametrize(
    "dates",
    [
        ("10d", "-10d"),
        (dt(2022, 1, 11), dt(2021, 12, 22)),
        (10, -10),
    ],
)
def test_curve_roll(crv, dates) -> None:
    rolled_curve = crv.roll(dates[0])
    rolled_curve2 = crv.roll(dates[1])

    expected = np.array(
        [
            crv.rate(_, "1D")
            for _ in [dt(2023, 1, 15), dt(2023, 3, 15), dt(2024, 11, 15), dt(2026, 4, 15)]
        ],
    )
    result = np.array(
        [
            rolled_curve.rate(_, "1D")
            for _ in [dt(2023, 1, 25), dt(2023, 3, 25), dt(2024, 11, 25), dt(2026, 4, 25)]
        ],
    )
    result2 = np.array(
        [
            rolled_curve2.rate(_, "1D")
            for _ in [dt(2023, 1, 5), dt(2023, 3, 5), dt(2024, 11, 5), dt(2026, 4, 5)]
        ],
    )
    assert np.all(np.abs(result - expected) < 1e-7)
    assert np.all(np.abs(result2 - expected) < 1e-7)

    # value prior to initial node
    assert rolled_curve[dt(1900, 1, 1)] == 0.0


@pytest.mark.skip(reason="v2.1 uses a RolledCurve and does not return a compatible object for eq")
def test_curve_roll_copy(curve) -> None:
    result = curve.roll("0d")
    assert result == curve


def test_curve_spline_warning() -> None:
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


def test_index_curve_roll() -> None:
    crv = Curve(
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
        ],
    )
    result = np.array(
        [
            rolled_curve.rate(_, "1D")
            for _ in [dt(2023, 1, 25), dt(2023, 3, 25), dt(2024, 11, 25), dt(2026, 4, 25)]
        ],
    )
    result2 = np.array(
        [
            rolled_curve2.rate(_, "1D")
            for _ in [dt(2023, 1, 5), dt(2023, 3, 5), dt(2024, 11, 5), dt(2026, 4, 5)]
        ],
    )
    assert np.all(np.abs(result - expected) < 1e-7)
    assert np.all(np.abs(result2 - expected) < 1e-7)
    assert rolled_curve.meta.index_base == crv.meta.index_base


@pytest.mark.parametrize(
    "s",
    [
        Series(index=[dt(2000, 1, 1), dt(2000, 2, 1), dt(2000, 3, 1)], data=[100.0, 200, 300]),
        158.62068965517238,
        "KLMN",
    ],
)
def test_index_value_series(s) -> None:
    # test that a Series input to fixings works
    fixings.add(
        "KLMN",
        Series(index=[dt(2000, 1, 1), dt(2000, 2, 1), dt(2000, 3, 1)], data=[100.0, 200, 300]),
    )
    result = index_value(
        index_lag=1,
        index_method="daily",
        index_fixings=s,
        index_date=dt(2000, 2, 18),
        index_curve=NoInput(0),
    )
    expected = 12 / 29 * 100.0 + 17 / 29 * 200
    fixings.pop("KLMN")
    assert abs(result - expected) < 1e-10


def test_curve_translate_raises(curve) -> None:
    with pytest.raises(ValueError, match="Cannot translate into the past."):
        curve.translate(dt(2020, 4, 1))


def test_curve_zero_width_rate_raises(curve) -> None:
    with pytest.raises(ZeroDivisionError, match="effective:"):
        curve.rate(dt(2022, 3, 10), dt(2022, 3, 10))


def test_set_node_vector_updates_ad_attribute(curve) -> None:
    curve._set_node_vector([0.98], ad=2)
    assert curve.ad == 2


@pytest.mark.parametrize(
    ("convention", "expected"),
    [
        ("act360", 4.3652192566314705),
        ("30360", 4.372999441829487),
        ("act365f", 4.372518793743008),
        ("bus252", 4.354756779569957),
    ],
)
def test_average_rate(convention, expected):
    start = dt(2000, 1, 1)
    end = dt(2006, 1, 1)
    rate = 5.0
    d = dcf(start, end, convention, calendar="bus")
    result, d_, n_ = average_rate(start, end, convention, rate, d)

    assert abs(result - expected) < 1e-12
    assert abs((1 + d * rate / 100.0) - (1 + d_ * result / 100.0) ** n_) < 1e-12


@pytest.mark.parametrize("curve", [Curve, LineCurve])
def test_spline_interpolation_feature(curve):
    t = [dt(2000, 1, 1)] * 4 + [dt(2001, 1, 1)] + [dt(2002, 1, 1)] * 4
    original = curve(nodes={dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98, dt(2002, 1, 1): 0.975}, t=t)
    feature = curve(
        nodes={dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98, dt(2002, 1, 1): 0.975},
        interpolation="spline",
    )
    assert feature.interpolator.spline.t == t
    assert feature.interpolator.spline.spline.c == original.interpolator.spline.spline.c

    assert feature[dt(2000, 1, 1)] == original[dt(2000, 1, 1)]
    assert feature[dt(1999, 1, 1)] == original[dt(1999, 1, 1)]
    assert feature[dt(2001, 5, 1)] == original[dt(2001, 5, 1)]


class TestCurve:
    def test_repr(self):
        curve = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.98,
                dt(2024, 1, 1): 0.965,
                dt(2025, 1, 1): 0.955,
            },
            id="sofr",
        )
        expected = f"<rl.Curve:{curve.id} at {hex(id(curve))}>"
        assert expected == curve.__repr__()

    def test_cache_clear_and_defaults(self):
        curve = Curve({dt(2000, 1, 1): 1.0, dt(2002, 1, 1): 0.99})
        curve[dt(2001, 1, 1)]
        assert len(curve._cache) == 1
        curve._clear_cache()
        assert len(curve._cache) == 0
        v1 = curve[dt(2001, 1, 1)]
        curve.update_node(dt(2002, 1, 1), 0.98)
        # cache cleared by function
        assert len(curve._cache) == 0
        v2 = curve[dt(2001, 1, 1)]
        assert v2 != v1

        with default_context("curve_caching", False):
            curve.nodes.nodes[dt(2002, 1, 1)] = 0.90
            # no clear cache required, but value will re-calc anyway
            assert curve[dt(2001, 1, 1)] != v2

    def test_typing_as_base_curve(self):
        curve = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.98,
                dt(2024, 1, 1): 0.965,
                dt(2025, 1, 1): 0.955,
            },
            id="sofr",
        )
        assert isinstance(curve, _BaseCurve)

    @pytest.mark.skip(reason="TranslatedCurve was constructed in v2.1 and bypasses this.")
    def test_curve_translate_knots_raises(self) -> None:
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

    def test_calendar_passed_to_rate_dcf(self):
        # Holidays on which no overnight DI rate is published
        reserve_holidays = [
            "2025-01-01",
            "2025-03-03",
            "2025-03-04",
            "2025-04-18",
            "2025-04-21",
            "2025-05-01",
            "2025-06-19",
            "2025-09-07",
            "2025-10-12",
            "2025-11-02",
            "2025-11-15",
            "2025-11-20",
            "2025-12-25",
            "2026-01-01",
            "2026-02-16",
            "2026-02-17",
            "2026-04-03",
            "2026-04-21",
            "2026-05-01",
            "2026-06-04",
            "2026-09-07",
            "2026-10-12",
            "2026-11-02",
            "2026-11-15",
            "2026-11-20",
            "2026-12-25",
        ]
        bra = Cal(holidays=[dt.strptime(h, "%Y-%m-%d") for h in reserve_holidays], week_mask=[5, 6])

        curve = Curve(
            nodes={
                dt(2025, 5, 15): 1.0,
                dt(2026, 1, 2): 0.919218,
            },
            convention="bus252",
            calendar=bra,
        )
        d = dcf(dt(2025, 5, 15), dt(2026, 1, 2), "bus252", calendar=bra)
        expected = (1 + 0.14) ** -d
        assert abs(expected - curve[dt(2026, 1, 2)]) < 5e-7

        # period rate
        result = curve.rate(dt(2025, 5, 15), dt(2026, 1, 2))
        expected = (1 / 0.919218 - 1) * 100 / d
        assert abs(expected - result) < 5e-7

    @pytest.mark.parametrize("interpolation", ["linear", "log_linear"])
    def test_linear_bus_interpolation(self, interpolation) -> None:
        curve = Curve(
            nodes={dt(2000, 1, 3): 1.0, dt(2000, 1, 17): 0.9},
            calendar="bus",
            convention="act365f",
            interpolation=interpolation,
        )
        curve2 = Curve(
            nodes={dt(2000, 1, 3): 1.0, dt(2000, 1, 17): 0.9},
            calendar="bus",
            convention="bus252",
            interpolation=interpolation,
        )

        assert curve[dt(2000, 1, 17)] == curve2[dt(2000, 1, 17)]
        assert curve[dt(2000, 1, 3)] == curve2[dt(2000, 1, 3)]

        assert curve[dt(2000, 1, 5)] != curve2[dt(2000, 1, 5)]
        assert curve[dt(2000, 1, 10)] == curve2[dt(2000, 1, 10)]  #  half calendar and bus
        assert curve[dt(2000, 1, 13)] != curve2[dt(2000, 1, 13)]

    def test_update_meta(self, curve):
        curve.update_meta("credit_discretization", 101)
        assert curve.meta.credit_discretization == 101

    def test_no_termination(self, curve):
        with pytest.raises(ValueError, match="`termination` must be supplied"):
            curve.rate(dt(2022, 3, 2))

    def test_index_value_lag_mismatch(self, index_curve):
        with pytest.raises(ValueError, match="'curve' interpolation can only be used"):
            index_curve.index_value(
                index_date=dt(2022, 3, 4),
                index_lag=22,
                index_method="curve",
            )

    def test_update_node_raises(self, curve):
        with pytest.raises(KeyError, match="`key` is not in"):
            curve.update_node(dt(2000, 1, 1), 1.0)


class TestLineCurve:
    def test_repr(self):
        curve = LineCurve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.98,
                dt(2024, 1, 1): 0.965,
                dt(2025, 1, 1): 0.955,
            },
            id="libor1m",
        )
        expected = f"<rl.LineCurve:{curve.id} at {hex(id(curve))}>"
        assert expected == curve.__repr__()

    def test_typing_as_base_curve(self):
        curve = LineCurve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.98,
                dt(2024, 1, 1): 0.965,
                dt(2025, 1, 1): 0.955,
            },
            id="libor1m",
        )
        assert isinstance(curve, _BaseCurve)

    def test_index_values_raises(self, line_curve):
        with pytest.raises(TypeError, match="A 'values' type Curve cannot"):
            line_curve.index_value(dt(2022, 3, 3), index_lag=0)


class TestIndexCurve:
    def test_curve_index_linear_daily_interp(self) -> None:
        curve = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 1, 5): 0.9999},
            index_base=200.0,
            interpolation="linear_index",
            index_lag=2,
        )
        result = curve.index_value(dt(2022, 1, 5), 2)
        expected = 200.020002002
        assert abs(result - expected) < 1e-7

        result = curve.index_value(dt(2022, 1, 3), 2)
        expected = 200.010001001  # value is linearly interpolated between index values.
        assert abs(result - expected) < 1e-7

    # SKIP: with deprecation of IndexCurve errors must be deferred to price time.
    # def test_indexcurve_raises(self) -> None:
    #     with pytest.raises(ValueError, match="`index_base` must be given"):
    #         Curve({dt(2022, 1, 1): 1.0})

    def test_index_value_raises(self) -> None:
        curve = Curve({dt(2022, 1, 1): 1.0}, index_base=100.0)
        with pytest.raises(ValueError, match="`index_method` as string: 'BAD' is not a v"):
            curve.index_value(dt(2022, 1, 1), 3, index_method="BAD")

    @pytest.mark.parametrize("ad", [0, 1, 2])
    def test_roll_preserves_ad(self, ad) -> None:
        curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_base=100.0,
            index_lag=3,
            id="tags_",
            ad=ad,
        )
        new_curve = curve.roll("1m")
        assert new_curve.ad == curve.ad

    def test_historic_rate_is_none(self) -> None:
        curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_base=100.0,
            index_lag=3,
            id="tags_",
        )
        assert curve.rate(dt(2021, 3, 4), "1b", "f") is None

    def test_repr(self):
        curve = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 1, 5): 0.9999}, index_base=200.0, id="us_cpi"
        )
        expected = f"<rl.Curve:us_cpi at {hex(id(curve))}>"
        assert expected == curve.__repr__()

    def test_typing_as_base_curve(self):
        curve = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 1, 5): 0.9999}, index_base=200.0, id="us_cpi"
        )
        assert isinstance(curve, _BaseCurve)


class TestCompositeCurve:
    def test_long_1day_rate_captured(self):
        c1 = Curve({dt(2000, 1, 1): 1.0, dt(2030, 1, 1): 0.8, dt(2030, 1, 2): 0.7999})
        c2 = Curve({dt(2000, 1, 1): 1.0, dt(2030, 1, 1): 0.7, dt(2030, 1, 2): 0.6999})
        r1 = c1.rate(dt(2030, 1, 1), dt(2030, 1, 2))
        r2 = c2.rate(dt(2030, 1, 1), dt(2030, 1, 2))
        cc = CompositeCurve([c1, c2])
        result = cc.rate(dt(2030, 1, 1), dt(2030, 1, 2))
        assert abs(result - r1 - r2) < 5e-4

    def test_curve_df_based(self) -> None:
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

        for date in [dt(2022, 12, 30), dt(2022, 12, 31), dt(2023, 1, 1)]:
            result1 = curve.rate(date, "1d")
            expected1 = curve1.rate(date, "1d") + curve2.rate(date, "1d")
            assert abs(result1 - expected1) < 2e-8

        result = curve.rate(dt(2022, 6, 1), "1Y")
        expected = curve1.rate(dt(2022, 6, 1), "1Y") + curve2.rate(dt(2022, 6, 1), "1Y")
        assert abs(result - expected) < 1e-4

    def test_composite_curve_translate(self) -> None:
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
            ],
        )
        assert np.all(np.abs(diff) < 1e-5)

    def test_composite_curve_roll(self) -> None:
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
            [crv.rate(_, "1D") for _ in [dt(2023, 1, 15), dt(2023, 3, 15), dt(2024, 11, 15)]],
        )
        result = np.array(
            [
                rolled_curve.rate(_, "1D")
                for _ in [dt(2023, 1, 25), dt(2023, 3, 25), dt(2024, 11, 25)]
            ],
        )

        assert np.all(np.abs(result - expected) < 1e-7)

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("rate", (dt(2022, 1, 1), "1d")),
            ("roll", ("10d",)),
            ("translate", (dt(2022, 1, 10),)),
            ("shift", (10.0, "id")),
            ("__getitem__", (dt(2022, 1, 10),)),
            ("index_value", (dt(2022, 1, 10), 3)),
        ],
    )
    def test_composite_curve_precheck_cache(self, method, args) -> None:
        # test precache_check on shift
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999}, index_base=100.0, index_lag=3)
        c2 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.998})
        cc = CompositeCurve([c1, c2])
        cc._cache[dt(1980, 1, 1)] = 100.0

        # mutate a curve to trigger cache id clear
        c1._set_node_vector([0.99], 0)
        getattr(cc, method)(*args)
        assert dt(1980, 1, 1) not in cc._cache

    def test_isinstance_raises(self) -> None:
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99})
        line_curve = LineCurve({dt(2022, 1, 1): 10.0, dt(2023, 1, 1): 12.0})
        with pytest.raises(TypeError, match="CompositeCurve can only contain curves of the same t"):
            CompositeCurve([curve, line_curve])

    @pytest.mark.parametrize(
        ("attribute", "val"),
        [
            ("modifier", ["MF", "MP"]),
            ("calendar", ["ldn", "tgt"]),
            ("convention", ["act360", "act365f"]),
        ],
    )
    def test_attribute_error_raises(self, attribute, val) -> None:
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, **{attribute: val[0]})
        c2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, **{attribute: val[1]})
        with pytest.raises(ValueError, match="Cannot composite curves with dif"):
            CompositeCurve([c1, c2])

    def test_line_based(self) -> None:
        c1 = LineCurve({dt(2022, 1, 1): 1.5, dt(2022, 1, 3): 1.0})
        c2 = LineCurve({dt(2022, 1, 1): 2.0, dt(2022, 1, 3): 3.0})
        cc = CompositeCurve([c1, c2])
        expected = 3.75
        result = cc.rate(dt(2022, 1, 2))
        assert abs(result - expected) < 1e-8

        result = cc[dt(2022, 1, 2)]
        assert abs(result - expected) < 1e-8

    def test_initial_node_raises(self) -> None:
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99})
        c2 = Curve({dt(2022, 1, 2): 1.0, dt(2023, 1, 1): 0.99})
        with pytest.raises(ValueError, match="`curves` must share the same ini"):
            CompositeCurve([c1, c2])

    @pytest.mark.parametrize(
        ("lag", "base"), [([2, 3], [100.0, 99.0]), ([4, NoInput(0)], [100.0, NoInput(0)])]
    )
    def test_index_curves_take_first_value(self, lag, base) -> None:
        ic1 = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_lag=lag[0],
            index_base=base[0],
        )
        ic2 = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_lag=lag[1],
            index_base=base[1],
        )
        cc = CompositeCurve([ic1, ic2])
        assert cc.meta.index_base == base[0]
        assert cc.meta.index_lag == lag[0]

    def test_index_curves_attributes_warns(self):
        ic1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_lag=3, index_base=101.1)
        ic2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_lag=3, index_base=101.1)
        cc = CompositeCurve([ic1, ic2])

        with pytest.warns(UserWarning):
            result = cc.index_value(dt(1999, 1, 1), 3)
            expected = 0.0
            assert abs(result - expected) < 1e-5

    def test_index_curves_attributes(self) -> None:
        ic1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_lag=3, index_base=101.1)
        ic2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_lag=3, index_base=101.1)
        cc = CompositeCurve([ic1, ic2])
        assert cc.meta.index_lag == 3
        assert cc.meta.index_base == 101.1

        result = cc.index_value(dt(2022, 1, 31), 3, index_method="monthly")
        expected = 101.1
        assert abs(result - expected) < 1e-5

        result = cc.index_value(dt(2022, 1, 1), 3)
        expected = 101.1
        assert abs(result - expected) < 1e-5

    def test_index_curves_interp_raises(self) -> None:
        ic1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_lag=3, index_base=101.1)
        ic2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_lag=3, index_base=101.1)
        cc = CompositeCurve([ic1, ic2])
        with pytest.raises(ValueError, match="`index_method` as string: 'bad interp'"):
            cc.index_value(index_date=dt(2022, 1, 31), index_lag=3, index_method="bad interp")

    def test_composite_curve_proxies(self) -> None:
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

    def test_composite_curve_no_index_value_raises(self, curve) -> None:
        cc = CompositeCurve([curve])
        with pytest.raises(ValueError, match="Curve must be initialised with an `index_base`"):
            cc.index_value(dt(2022, 1, 1), 3)

    def test_historic_rate_is_none(self) -> None:
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

    def test_repr(self):
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
        expected = f"<rl.CompositeCurve:{curve.id} at {hex(id(curve))}>"
        assert expected == curve.__repr__()
        assert isinstance(curve.id, str)

    def test_typing_as_base_curve(self):
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
        assert isinstance(curve, _BaseCurve)

    def test_cache(self):
        curve1 = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.98,
            },
        )
        curve2 = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2022, 6, 30): 1.0,
                dt(2022, 7, 1): 0.999992,
                dt(2022, 12, 31): 0.999992,
                dt(2023, 1, 1): 0.999984,
            },
        )
        curve = CompositeCurve([curve1, curve2])
        curve[dt(2022, 3, 1)]
        assert curve._cache == {dt(2022, 3, 1): 0.9967396833121631}

        # update a curve
        curve2.update_node(dt(2022, 6, 30), 0.95)
        curve[dt(2022, 3, 1)]
        assert curve._cache == {dt(2022, 3, 1): 0.9801226964242061}

    def test_composite_curve_of_composite_curve(self):
        c1 = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.98,
            },
        )
        c2 = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 30): 0.99,
            }
        )
        cc1 = CompositeCurve([c1, c2])
        cc2 = CompositeCurve([cc1, c1])
        result = cc2.rate(dt(2022, 2, 15), "3m")
        assert abs(result - 4.933123726330553) < 1e-8

    def test_composite_curve_of_composite_line_curve(self):
        c1 = LineCurve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.98,
            },
        )
        c2 = LineCurve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 30): 0.99,
            }
        )
        cc1 = CompositeCurve([c1, c2])
        cc2 = CompositeCurve([cc1, c1])
        result = cc2.rate(dt(2022, 2, 15), "3m")
        assert abs(result - 2.993926361170989) < 1e-8

    def test_ad_order_is_max(self):
        c1 = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.99})
        c2 = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.99})
        c2._set_ad_order(2)

        assert CompositeCurve([c1, c2])._ad == 2
        assert CompositeCurve([c2, c1])._ad == 2

    def test_initial_df(self):
        curve1 = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.99}, ad=1, id="v")
        curve2 = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98}, ad=1, id="w")
        cc = CompositeCurve([curve1, curve2])
        result = cc[dt(2000, 1, 1)]
        expected = Dual(1.0, ["v0", "v1", "w0", "w1"], [1.0, 0.0, 1.0, 0.0])
        assert result == expected

    def test_update_meta_raises(self):
        ic1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_lag=3, index_base=101.1)
        ic2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_lag=3, index_base=101.1)
        cc = CompositeCurve([ic1, ic2])
        with pytest.raises(AttributeError, match="'CompositeCurve' object has no attribute 'updat"):
            cc.update_meta("h", 100.0)

    def test_update_meta(self):
        ic1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_lag=3, index_base=101.1)
        ic2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_lag=3, index_base=101.1)
        cc = CompositeCurve([ic1, ic2])
        before = cc.meta.credit_recovery_rate
        ic1.update_meta("credit_recovery_rate", 0.88)
        after = cc.meta.credit_recovery_rate
        assert before != after
        assert after == 0.88


class TestMultiCsaCurve:
    def test_historic_rate_is_none(self) -> None:
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

    def test_multi_raises(self, line_curve, curve) -> None:
        with pytest.raises(TypeError, match="MultiCsaCurve must use discount factors"):
            MultiCsaCurve([line_curve])

    def test_multi_csa_shift(self) -> None:
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
        cc_shift = cc.shift(100)
        with default_context("multi_csa_steps", [1, 1, 1, 1, 1, 1, 1]):
            r1 = cc_shift.rate(dt(2022, 1, 1), "1d")
            r2 = cc_shift.rate(dt(2022, 1, 2), "1d")
            r3 = cc_shift.rate(dt(2022, 1, 3), "1d")
            r4 = cc_shift.rate(dt(2022, 1, 4), "1d")

        assert abs(r1 - 5.0) < 1e-3
        assert abs(r2 - 4.5) < 1e-3
        assert abs(r3 - 4.5) < 1e-3
        assert abs(r4 - 5.0) < 1e-3

    @pytest.mark.parametrize("caching", [True, False])
    def test_multi_csa(self, caching) -> None:
        with default_context("curve_caching", caching):
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

    def test_multi_csa_granularity(self) -> None:
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 0.9, dt(2072, 1, 1): 0.5})
        c2 = Curve({dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 0.8, dt(2072, 1, 1): 0.7})

        with default_context("multi_csa_max_step", 182, "multi_csa_min_step", 182):
            cc = MultiCsaCurve([c1, c2])
            r1 = cc.rate(dt(2052, 5, 24), "1d")
            # r2 = cc.rate(dt(2052, 5, 25), "1d")
            # r3 = cc.rate(dt(2052, 5, 26), "1d")
            assert abs(r1 - 1.448374) < 1e-3

    def test_repr(self):
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
        curve = MultiCsaCurve([c1, c2, c3])
        expected = f"<rl.MultiCsaCurve:{curve.id} at {hex(id(curve))}>"
        assert expected == curve.__repr__()
        assert isinstance(curve.id, str)

    def test_typing_as_base_curve(self):
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
        curve = MultiCsaCurve([c1, c2, c3])
        assert isinstance(curve, _BaseCurve)

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("rate", (dt(2022, 1, 1), "1d")),
            ("roll", ("10d",)),
            ("translate", (dt(2022, 1, 10),)),
            ("shift", (10.0, "id")),
            ("__getitem__", (dt(2022, 1, 10),)),
        ],
    )
    def test_multi_csa_curve_precheck_cache(self, method, args) -> None:
        # test precache_check on shift
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.999})
        c2 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.998})
        cc = MultiCsaCurve([c1, c2])
        cc._cache[dt(1980, 1, 1)] = 100.0

        # mutate a curve to trigger cache id clear
        c1._set_node_vector([0.99], 0)
        getattr(cc, method)(*args)
        assert dt(1980, 1, 1) not in cc._cache

    def test_multi_csa_curve_add_to_cache(self):
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2052, 2, 1): 0.9})
        c2 = Curve({dt(2022, 1, 1): 1.0, dt(2052, 2, 1): 0.8})
        cc = MultiCsaCurve([c1, c2])
        cc[dt(2052, 2, 1)]
        assert len(cc._cache) == 31


class TestProxyCurve:
    def test_repr(self) -> None:
        fxr1 = FXRates({"usdeur": 0.95}, dt(2022, 1, 3))
        fxr2 = FXRates({"usdcad": 1.1}, dt(2022, 1, 2))
        fxf = FXForwards(
            [fxr1, fxr2],
            {
                "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 0.95}),
                "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 1.0}),
                "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 0.99}),
                "cadusd": Curve({dt(2022, 1, 1): 1.00, dt(2022, 10, 1): 0.97}),
                "cadcad": Curve({dt(2022, 1, 1): 1.00, dt(2022, 10, 1): 0.969}),
            },
        )
        curve = fxf.curve("cad", "eur")
        expected = f"<rl.ProxyCurve:{curve.id} at {hex(id(curve))}>"
        assert curve.__repr__() == expected
        assert isinstance(curve.id, str)

    def test_typing_as_basecurve(self):
        fxr1 = FXRates({"usdeur": 0.95}, dt(2022, 1, 3))
        fxr2 = FXRates({"usdcad": 1.1}, dt(2022, 1, 2))
        fxf = FXForwards(
            [fxr1, fxr2],
            {
                "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 0.95}),
                "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 1.0}),
                "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 0.99}),
                "cadusd": Curve({dt(2022, 1, 1): 1.00, dt(2022, 10, 1): 0.97}),
                "cadcad": Curve({dt(2022, 1, 1): 1.00, dt(2022, 10, 1): 0.969}),
            },
        )
        curve = fxf.curve("cad", "eur")
        assert isinstance(curve, _BaseCurve)

    def test_cache_is_validated_on_getitem_and_lookup(self):
        fxr1 = FXRates({"usdeur": 0.95}, dt(2022, 1, 3))
        fxr2 = FXRates({"usdcad": 1.1}, dt(2022, 1, 2))
        fxf = FXForwards(
            [fxr1, fxr2],
            {
                "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 0.95}),
                "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 1.0}),
                "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 0.99}),
                "cadusd": Curve({dt(2022, 1, 1): 1.00, dt(2022, 10, 1): 0.97}),
                "cadcad": Curve({dt(2022, 1, 1): 1.00, dt(2022, 10, 1): 0.969}),
            },
        )
        curve = fxf.curve("cad", "eur")
        assert curve._state == fxf._state

        fxr1.update({"usdeur": 100000000.0})
        fxf.curve("eur", "eur")._set_node_vector([0.5], 1)

        state1 = fxf._state
        # performing an action on the proxy curve will validate and update states
        curve[dt(2022, 1, 9)]
        state2 = fxf._state
        assert state1 != state2

        fxr1.update({"usdeur": 10.0})
        fxf.curve("eur", "eur")._set_node_vector([0.6], 1)
        state3 = curve._state
        assert state3 == state2  # becuase no method validation has yet occurred

    def test_update(self):
        fxr1 = FXRates({"usdeur": 0.95}, dt(2022, 1, 3))
        fxr2 = FXRates({"usdcad": 1.1}, dt(2022, 1, 2))
        fxf = FXForwards(
            [fxr1, fxr2],
            {
                "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 0.95}),
                "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 1.0}),
                "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2022, 10, 1): 0.99}),
                "cadusd": Curve({dt(2022, 1, 1): 1.00, dt(2022, 10, 1): 0.97}),
                "cadcad": Curve({dt(2022, 1, 1): 1.00, dt(2022, 10, 1): 0.969}),
            },
        )
        curve = fxf.curve("cad", "eur")
        with pytest.raises(AttributeError):
            curve.update_meta("h", 100.0)
        with pytest.raises(AttributeError):
            curve.update_node("h", 100.0)
        with pytest.raises(AttributeError):
            curve.update("h", 100.0)


class TestPlotCurve:
    def test_plot_curve(self, curve) -> None:
        fig, ax, lines = curve.plot("1d")
        result = lines[0].get_data()
        assert result[0][0] == dt(2022, 3, 1)
        assert abs(result[1][0].real - 12.004001333774994) < 1e-6
        plt.close("all")

    def test_plot_linecurve(self, line_curve) -> None:
        fig, ax, lines = line_curve.plot("0d")
        result = lines[0].get_data()
        assert result[0][0] == dt(2022, 3, 1)
        assert abs(result[1][0].real - 2.0) < 1e-6
        plt.close("all")

    @pytest.mark.parametrize("left", ["1d", dt(2022, 3, 2)])
    def test_plot_curve_left(self, curve, left) -> None:
        fig, ax, lines = curve.plot("1d", left=left)
        result = lines[0].get_data()
        assert result[0][0] == dt(2022, 3, 2)
        assert abs(result[1][0].real - 12.008005336896055) < 1e-6
        plt.close("all")

    def test_plot_curve_left_raise(self, curve) -> None:
        with pytest.raises(ValueError, match="`left` must be supplied as"):
            fig, ax, lines = curve.plot("1d", left=100.3)
        plt.close("all")

    @pytest.mark.parametrize("right", ["2d", dt(2022, 3, 3)])
    def test_plot_curve_right(self, curve, right) -> None:
        fig, ax, lines = curve.plot("1d", right=right)
        result = lines[0].get_data()
        assert result[0][-1] == dt(2022, 3, 3)
        assert abs(result[1][-1].real - 12.012012012015738) < 1e-6
        plt.close("all")

    def test_plot_curve_right_raise(self, curve) -> None:
        with pytest.raises(ValueError, match="`right` must be supplied as"):
            fig, ax, lines = curve.plot("1d", right=100.3)
        plt.close("all")

    def test_plot_comparators(self, curve) -> None:
        fig, ax, lines = curve.plot("1d", comparators=[curve])
        assert len(lines) == 2
        res1 = lines[0].get_data()
        res2 = lines[1].get_data()
        assert res1[0][0] == res2[0][0]
        assert res1[1][0] == res2[1][0]
        plt.close("all")

    def test_plot_diff(self, curve) -> None:
        fig, ax, lines = curve.plot("1d", comparators=[curve], difference=True)
        assert len(lines) == 1
        result = lines[0].get_data()
        assert result[0][0] == dt(2022, 3, 1)
        assert result[1][0] == 0
        plt.close("all")

    @pytest.mark.parametrize("left", [NoInput(0), dt(2022, 1, 1), "0d"])
    @pytest.mark.parametrize("right", [NoInput(0), dt(2022, 2, 1), "0d"])
    def test_plot_index(self, left, right) -> None:
        i_curve = Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 1.0}, index_base=2.0)
        fig, ax, lines = i_curve.plot_index(left=left, right=right)
        result = lines[0].get_data()
        assert result[0][0] == dt(2022, 1, 1)
        assert abs(result[1][0].real - 2.0) < 1e-6
        plt.close("all")

    def test_plot_index_comparators(self) -> None:
        i_curve = Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 1.0}, index_base=2.0)
        i_curv2 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 1.0}, index_base=2.0)
        fig, ax, lines = i_curve.plot_index(comparators=[i_curv2])
        assert len(lines) == 2
        res1 = lines[0].get_data()
        res2 = lines[1].get_data()
        assert res1[0][0] == res2[0][0]
        assert res1[1][0] == res2[1][0]
        plt.close("all")

    def test_plot_index_diff(self) -> None:
        i_curv = Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 1.0}, index_base=2.0)
        i_curv2 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 1.0}, index_base=2.0)
        fig, ax, lines = i_curv.plot_index("1d", comparators=[i_curv2], difference=True)
        assert len(lines) == 1
        result = lines[0].get_data()
        assert result[0][0] == dt(2022, 1, 1)
        assert result[1][0] == 0
        plt.close("all")

    def test_plot_index_raises(self) -> None:
        i_curve = Curve({dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 1.0}, index_base=2.0)
        with pytest.raises(ValueError, match="`left` must be supplied as"):
            i_curve.plot_index(left=2.0)
        with pytest.raises(ValueError, match="`right` must be supplied as"):
            i_curve.plot_index(right=2.0)

    def test_composite_curve_plot(self) -> None:
        curve1 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 12, 1): 0.95}, modifier="MF", calendar="bus")
        curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 12, 1): 0.97}, modifier="MF", calendar="bus")
        cc = CompositeCurve(curves=[curve1, curve2])
        cc.plot("1m")

    def test_plot_a_rolled_spline_curve(self) -> None:
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


class TestStateAndCache:
    @pytest.mark.parametrize(
        "curve",
        [
            Curve(nodes={dt(2000, 1, 1): 1.0, dt(2002, 1, 1): 0.99}),
            LineCurve(nodes={dt(2000, 1, 1): 1.0, dt(2002, 1, 1): 0.99}),
            Curve(
                nodes={
                    dt(2022, 1, 1): 1.0,
                    dt(2023, 1, 1): 0.98,
                },
                index_base=200.0,
            ),
        ],
    )
    @pytest.mark.parametrize(("method", "args"), [("_set_ad_order", (1,))])
    def test_method_does_not_change_state(self, curve, method, args):
        before = curve._state
        getattr(curve, method)(*args)
        after = curve._state
        assert before == after

    @pytest.mark.parametrize(
        "curve",
        [
            Curve(nodes={dt(2000, 1, 1): 1.0, dt(2002, 1, 1): 0.99, dt(2003, 1, 1): 0.98}),
            LineCurve(nodes={dt(2000, 1, 1): 1.0, dt(2002, 1, 1): 0.99}),
            Curve(
                nodes={
                    dt(2000, 1, 1): 1.0,
                    dt(2002, 1, 1): 0.98,
                },
                index_base=200.0,
            ),
        ],
    )
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_set_node_vector", ([0.99, 0.98], 1)),
            ("update_node", (dt(2002, 1, 1), 0.98)),
            ("update", ({dt(2000, 1, 1): 1.0, dt(2002, 1, 1): 0.99},)),
            ("csolve", tuple()),
        ],
    )
    def test_method_changes_state(self, curve, method, args):
        before = curve._state
        getattr(curve, method)(*args)
        after = curve._state
        assert before != after

    @pytest.mark.parametrize(
        "curve",
        [
            Curve(nodes={dt(2000, 1, 1): 1.0, dt(2002, 1, 1): 0.99}),
            LineCurve(nodes={dt(2000, 1, 1): 1.0, dt(2002, 1, 1): 0.99}),
            Curve(
                nodes={
                    dt(2000, 1, 1): 1.0,
                    dt(2002, 1, 1): 0.98,
                },
                index_base=200.0,
            ),
        ],
    )
    def test_populate_cache(self, curve):
        assert curve._cache == {}
        curve[dt(2000, 5, 1)]
        assert dt(2000, 5, 1) in curve._cache

    @pytest.mark.parametrize(
        "curve",
        [
            Curve(nodes={dt(2000, 1, 1): 1.0, dt(2002, 1, 1): 0.99, dt(2003, 1, 1): 0.98}),
            LineCurve(nodes={dt(2000, 1, 1): 1.0, dt(2002, 1, 1): 0.99}),
            Curve(
                nodes={
                    dt(2000, 1, 1): 1.0,
                    dt(2002, 1, 1): 0.98,
                },
                index_base=200.0,
            ),
        ],
    )
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_set_node_vector", ([0.99, 0.98], 1)),
            ("update_node", (dt(2002, 1, 1), 0.98)),
            ("update", ({dt(2000, 1, 1): 1.0, dt(2002, 1, 1): 0.99},)),
            ("csolve", tuple()),
            ("_set_ad_order", (1,)),
        ],
    )
    def test_method_clears_cache(self, curve, method, args):
        curve[dt(2000, 5, 1)]
        assert dt(2000, 5, 1) in curve._cache
        getattr(curve, method)(*args)
        assert curve._cache == {}

    @pytest.mark.parametrize("Klass", [CompositeCurve, MultiCsaCurve])
    def test_composite_curve_validation_cache_clearing_and_state(self, Klass):
        # test that a composite curve will validate and clear its cache
        # and following that update its own state to its composited state
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2024, 1, 1): 0.95})
        c2 = Curve({dt(2022, 1, 1): 1.0, dt(2024, 1, 1): 0.90})
        cc = Klass([c1, c2])
        cc_state_pre = cc._state
        # get a value and check the cache
        cc_result_pre = cc[dt(2022, 6, 1)]
        _ = cc[dt(2022, 6, 30)]
        assert dt(2022, 6, 1) in cc._cache
        assert dt(2022, 6, 30) in cc._cache

        # update an underlying curve
        c2.update_node(dt(2024, 1, 1), 0.85)
        # check the cache is cleared when using a get using
        cc_result_post = cc[dt(2022, 6, 1)]
        assert cc_result_post < cc_result_pre
        # check that the state of the composite curve has changed
        cc_state_post = cc._state
        assert cc_state_pre != cc_state_post
        assert cc_state_post == cc._get_composited_state()
        # check that the cache is correct
        assert dt(2022, 6, 1) in cc._cache
        assert dt(2022, 6, 30) not in cc._cache

    def test_max_cache_size(self):
        with default_context("curve_caching_max", 3):
            curve = Curve({dt(2022, 1, 1): 1.0, dt(2024, 1, 1): 0.95})
            assert curve._cache_len == 0
            curve[dt(2022, 2, 1)]
            assert curve._cache_len == 1
            curve[dt(2022, 3, 1)]
            assert curve._cache_len == 2
            curve[dt(2022, 4, 1)]
            assert curve._cache_len == 3
            curve[dt(2022, 5, 1)]
            assert curve._cache_len == 3

            assert dt(2022, 2, 1) not in curve._cache
            assert dt(2022, 3, 1) in curve._cache
            assert dt(2022, 4, 1) in curve._cache
            assert dt(2022, 5, 1) in curve._cache


class TestIndexValue:
    def test_dict_raise(self):
        with pytest.raises(
            NotImplementedError, match="`index_curve` cannot currently be supplied as dict"
        ):
            index_value(0, "curve", NoInput(0), 0, {"a": 0, "b": 0})

    def test_return_index_fixings_directly(self):
        assert index_value(0, "curve", 2.5, NoInput(0), NoInput(0)) == 2.5
        assert index_value(0, "curve", Dual(2, ["a"], []), NoInput(0), NoInput(0)) == Dual(
            2, ["a"], []
        )

    @pytest.mark.parametrize("method", ["curve", "daily"])
    def test_forecast_from_curve_no_fixings(self, method):
        # these methods should be identical when using "linear_index" interpolation directly on the
        # curve and parametrising the curve nodes with the start of month dates. See next test.
        curve = Curve(
            {dt(2000, 1, 1): 1.0, dt(2000, 2, 1): 0.99},
            index_base=100.0,
            index_lag=0,
            interpolation="linear_index",
        )
        result = index_value(0, method, NoInput(0), dt(2000, 1, 15), curve)
        expected = 100.0 / curve[dt(2000, 1, 15)]
        assert abs(result - expected) < 1e-9

    def test_forecast_from_curve_no_fixings_methods_identical(self):
        curve = Curve(
            {dt(2000, 1, 1): 1.0, dt(2000, 2, 1): 0.99},
            index_base=100.0,
            index_lag=0,
            interpolation="linear_index",
        )
        result1 = index_value(0, "curve", NoInput(0), dt(2000, 1, 15), curve)
        result2 = index_value(0, "daily", NoInput(0), dt(2000, 1, 15), curve)
        assert abs(result1 - result2) < 1e-9

    @pytest.mark.parametrize("date", [dt(2000, 2, 1), dt(2000, 2, 27)])
    def test_forecast_from_curve_no_fixings_monthly(self, date):
        # monthly interpolation should only require the date of 1st Feb from the curve
        curve = Curve(
            {dt(2000, 1, 1): 1.0, dt(2000, 2, 1): 0.99},
            index_base=100.0,
            index_lag=0,
            interpolation="linear_index",
        )
        result = index_value(0, "monthly", NoInput(0), date, curve)
        expected = 100.0 / curve[dt(2000, 2, 1)]
        assert abs(result - expected) < 1e-9

    @pytest.mark.parametrize("method", ["curve", "daily", "monthly"])
    def test_no_input_return_result_err(self, method):
        assert _try_index_value(0, method, NoInput(0), dt(2000, 1, 1), NoInput(0)).is_err

    @pytest.mark.parametrize("method", ["curve", "daily", "monthly"])
    def test_fixings_type_raises(self, method):
        with pytest.raises(TypeError, match="`index_fixings` must be of type: Str, Series, DualTy"):
            index_value(0, method, [1, 2], dt(2000, 1, 1), NoInput(0))

    def test_no_index_date_raises(self):
        with pytest.raises(ValueError, match="Must supply an `index_date` from whic"):
            index_value(0, "curve", NoInput(0), NoInput(0), NoInput(0))

    def test_non_zero_index_lag_with_curve_method_raises(self):
        ser = Series([1.0], index=[dt(2000, 1, 1)])
        fixings.add("1234FGFS6", ser)
        with pytest.raises(ValueError, match="`index_lag` must be zero when using a 'Curve' `inde"):
            index_value(
                index_lag=4,
                index_method="curve",
                index_fixings="1234FGFS6",
                index_date=dt(2000, 1, 1),
                index_curve=NoInput(0),
            )
        fixings.pop("1234FGFS6")

    def test_documentation_uk_dmo_replication(self):
        # this is an example in the index value documentation
        rpi_series = Series(
            [172.2, 173.1, 174.2, 174.4],
            index=[dt(2001, 3, 1), dt(2001, 4, 1), dt(2001, 5, 1), dt(2001, 6, 1)],
        )
        result = index_value(
            index_lag=3, index_method="daily", index_fixings=rpi_series, index_date=dt(2001, 7, 20)
        )
        expected = 173.77419
        assert abs(result - expected) < 5e-6

    def test_no_input_return_if_future_based(self):
        # the requested date is beyond the ability of the fixings series and no curve is provided
        rpi_series = Series([172.2, 173.1], index=[dt(2001, 3, 1), dt(2001, 4, 1)])

        res1 = _try_index_value(0, "curve", rpi_series, dt(2001, 4, 2))
        assert res1.is_err
        res2 = _try_index_value(0, "curve", rpi_series, dt(2001, 4, 1))
        assert res2.is_ok

    def test_mixed_forecast_value_fixings_with_curve(self):
        rpi = Series([100.0], index=[dt(2000, 1, 1)])
        curve = Curve({dt(2000, 1, 1): 1.0, dt(2000, 4, 1): 0.99}, index_base=110.0, index_lag=0)
        date = dt(2000, 5, 15)
        rpi_2 = 110 * 1.0 / curve[dt(2000, 2, 1)]
        expected = 100.0 + (14 / 31) * (rpi_2 - 100.0)
        result = index_value(4, "daily", rpi, date, curve)
        assert abs(result - expected) < 1e-9

    def test_mixed_forecast_value_fixings_with_curve2(self):
        rpi = Series([100.0], index=[dt(2000, 1, 1)])
        curve = Curve(
            nodes={dt(2000, 2, 1): 1.0, dt(2000, 5, 1): 0.99}, index_base=110.0, index_lag=1
        )

        date = dt(2000, 5, 15)
        rpi_2 = 110 * 1.0 / curve[dt(2000, 3, 1)]
        expected = 100.0 + (14 / 31) * (rpi_2 - 100.0)
        result = index_value(4, "daily", rpi, date, curve)
        assert abs(result - expected) < 1e-9

    def test_keyerror_for_series_using_curve_method(self):
        rpi = Series([9.0, 8.0], index=[dt(1999, 1, 1), dt(2000, 1, 1)])
        with pytest.raises(FixingMissingDataError, match="Fixing lookup for date "):
            index_value(0, "curve", rpi, dt(1999, 12, 31), NoInput(0))

    def test_daily_method_returns_directly_if_date_som(self):
        rpi = Series([100.0], index=[dt(2000, 1, 1)])
        assert index_value(0, "daily", rpi, dt(2000, 1, 1), NoInput(0)) == 100.0

    def test_daily_method_returns_err_if_data_unavailable(self):
        rpi = Series([100.0], index=[dt(2000, 1, 1)])
        res = _try_index_value(0, "daily", rpi, dt(2000, 1, 2), NoInput(0))
        assert res.is_err

    def test_curve_method_from_curve_with_non_zero_index_lag(self):
        curve = Curve(
            nodes={dt(2000, 1, 1): 1.0, dt(2000, 2, 1): 0.99},
            index_base=100.0,
            index_lag=1,
        )
        result = index_value(1, "curve", NoInput(0), dt(2000, 1, 15), curve)
        expected = 100.0 / curve[dt(2000, 1, 15)]
        assert abs(result - expected) < 1e-9

    @pytest.mark.parametrize(
        ("curve", "exp"),
        [
            (NoInput(0), Err),
            (
                Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.99}, index_base=100.0, index_lag=0),
                Ok,
            ),
        ],
    )
    def test_series_len_zero(self, curve, exp):
        s = Series(data=[], index=[], dtype=float)
        result = _try_index_value(0, "curve", s, dt(2000, 1, 1), curve)
        assert isinstance(result, exp)

    def test_series_and_curve_aligns_with_som_date(self):
        # the relevant value can be directly matched on the Series
        s = Series(data=[100.0], index=[dt(2000, 1, 1)])
        c = Curve({dt(2001, 1, 1): 1.0, dt(2002, 1, 1): 0.99}, index_base=100.0, index_lag=2)
        result = index_value(1, "daily", s, dt(2000, 2, 1), c)
        assert result == 100.0

    def test_mixed_series_and_curve(self):
        # the relevant value can be directly matched on the Series
        s = Series(
            data=[100.0, 200.0, 300.0], index=[dt(2000, 1, 1), dt(2000, 2, 1), dt(2000, 3, 1)]
        )
        c = Curve({dt(2001, 1, 1): 1.0, dt(2002, 1, 1): 0.99}, index_base=100.0, index_lag=2)
        result = index_value(0, "curve", s, dt(2000, 2, 1), c)
        assert result == 200.0

    def test_mixed_series_and_curve_inside_range_raises(self):
        s = Series(
            data=[100.0, 200.0, 300.0], index=[dt(2000, 1, 1), dt(2000, 2, 1), dt(2000, 3, 1)]
        )
        c = Curve({dt(2001, 1, 1): 1.0, dt(2002, 1, 1): 0.99}, index_base=100.0, index_lag=2)
        with pytest.raises(ValueError, match="The Series given for `index_fixings` requires, but"):
            index_value(0, "curve", s, dt(2000, 2, 15), c)

    def test_mixed_series_and_curve_inside_range_reverts_to_curve_due_to_lag(self):
        s = Series(
            data=[100.0, 200.0, 300.0], index=[dt(2000, 1, 1), dt(2000, 2, 1), dt(2000, 3, 1)]
        )
        c = Curve({dt(2001, 1, 1): 1.0, dt(2002, 1, 1): 0.99}, index_base=100.0, index_lag=1)
        with pytest.warns(UserWarning):
            # this warning exists when a curve returns 0.0 and the date is prior to curve start
            index_value(1, "curve", s, dt(2000, 2, 15), c)

    def test_mixed_series_and_curve_outside_range(self):
        s = Series(
            data=[100.0, 200.0, 300.0], index=[dt(2000, 1, 1), dt(2000, 2, 1), dt(2000, 3, 1)]
        )
        c = Curve({dt(2001, 1, 1): 1.0, dt(2002, 1, 1): 0.99}, index_base=100.0, index_lag=2)
        with pytest.raises(ValueError, match="The Series given for `index_fixings` requires, but"):
            index_value(0, "curve", s, dt(2000, 2, 15), c)

    def test_mixed_series_and_curve_raises_on_lag(self):
        s = Series(
            data=[100.0, 200.0, 300.0], index=[dt(2000, 1, 1), dt(2000, 2, 1), dt(2000, 3, 1)]
        )
        c = Curve({dt(2001, 1, 1): 1.0, dt(2002, 1, 1): 0.99}, index_base=100.0, index_lag=2)
        with pytest.raises(
            ValueError, match="`index_lag` must be zero when using a 'curve' `index"
        ):
            index_value(1, "curve", s, dt(2000, 2, 1), c)


class TestCurveSpline:
    @pytest.mark.parametrize("endpoints", [("natural", "natural"), ("not-a-knot", "natural")])
    @pytest.mark.parametrize("c", [NoInput(0), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    def test_equality(self, endpoints, c):
        t = [
            dt(2000, 1, 1),
            dt(2000, 1, 1),
            dt(2000, 1, 1),
            dt(2000, 1, 1),
            dt(2001, 1, 1),
            dt(2001, 6, 1),
            dt(2002, 1, 1),
            dt(2002, 1, 1),
            dt(2002, 1, 1),
            dt(2002, 1, 1),
        ]
        a = _CurveSpline(t=t, endpoints=endpoints)
        b = _CurveSpline(t=t, endpoints=endpoints)

        assert a == b

    @pytest.mark.parametrize("differ", ["t", "end"])
    def test_inequality(self, differ):
        t = [
            dt(2000, 1, 1),
            dt(2000, 1, 1),
            dt(2000, 1, 1),
            dt(2000, 1, 1),
            dt(2001, 1, 1),
            dt(2001, 6, 1),
            dt(2002, 1, 1),
            dt(2002, 1, 1),
            dt(2002, 1, 1),
            dt(2002, 1, 1),
        ]
        t_diff = [
            dt(2000, 1, 1),
            dt(2000, 1, 1),
            dt(2000, 1, 1),
            dt(2000, 1, 1),
            dt(2001, 1, 1),
            dt(2001, 7, 1),
            dt(2002, 1, 1),
            dt(2002, 1, 1),
            dt(2002, 1, 1),
            dt(2002, 1, 1),
        ]
        end = ("natural", "natural")
        end_diff = ("natural", "not-a-knot")

        a = _CurveSpline(t=t, endpoints=end)
        if differ == "t":
            b = _CurveSpline(t=t_diff, endpoints=end)
        else:
            b = _CurveSpline(t=t, endpoints=end_diff)

        assert a != b
        assert a != 10.0


class Test_CreditImpliedCurve:
    def test_credit_implied_rates(self):
        risk_free = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98})
        hazard = Curve(
            nodes={dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.95},
            credit_recovery_rate=Variable(0.4, ["RR"]),
        )
        implied = CreditImpliedCurve(risk_free=risk_free, hazard=hazard, id="my-id")
        assert implied.id == "my-id"

        rate1 = risk_free.rate(dt(2000, 2, 1), "1b")
        rate2 = hazard.rate(dt(2000, 2, 1), "1b")

        result = implied.rate(dt(2000, 2, 1), "1b")
        approximate = rate1 + rate2 * (1 - 0.4)
        assert abs(result - approximate) < 1e-9

    def test_risk_free_rates(self):
        credit = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98})
        hazard = Curve(
            nodes={dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.95},
            credit_recovery_rate=Variable(0.4, ["RR"]),
        )
        implied = CreditImpliedCurve(credit=credit, hazard=hazard)

        rate1 = credit.rate(dt(2000, 2, 1), "1b")
        rate2 = hazard.rate(dt(2000, 2, 1), "1b")

        result = implied.rate(dt(2000, 2, 1), "1b")
        approximate = rate1 - rate2 * (1 - 0.4)
        assert abs(result - approximate) < 1e-9

    def test_hazard_rates(self):
        risk_free = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98})
        credit = Curve(
            nodes={dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.95},
            credit_recovery_rate=Variable(0.4, ["RR"]),
        )
        implied = CreditImpliedCurve(credit=credit, risk_free=risk_free)

        rate1 = credit.rate(dt(2000, 2, 1), "1b")
        rate2 = risk_free.rate(dt(2000, 2, 1), "1b")

        result = implied.rate(dt(2000, 2, 1), "1b")
        approximate = (rate1 - rate2) / (1 - 0.4)
        assert abs(result - approximate) < 1e-9

    def test_round_trip_hazard(self):
        risk_free = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98})
        credit = Curve(
            nodes={dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.95},
            credit_recovery_rate=Variable(0.4, ["RR"]),
        )
        implied = CreditImpliedCurve(credit=credit, risk_free=risk_free)
        credit_implied = CreditImpliedCurve(hazard=implied, risk_free=risk_free)

        rate1 = credit.rate(dt(2000, 2, 1), "1b")
        rate2 = credit_implied.rate(dt(2000, 2, 1), "1b")

        assert abs(rate1 - rate2) < 1e-9

    def test_round_trip_credit(self):
        risk_free = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98})
        hazard = Curve(
            nodes={dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.95},
            credit_recovery_rate=Variable(0.4, ["RR"]),
        )
        implied = CreditImpliedCurve(hazard=hazard, risk_free=risk_free)
        hazard_implied = CreditImpliedCurve(credit=implied, risk_free=risk_free)

        rate1 = hazard.rate(dt(2000, 2, 1), "1b")
        rate2 = hazard_implied.rate(dt(2000, 2, 1), "1b")

        assert abs(rate1 - rate2) < 1e-9

    def test_meta_dynacism(self):
        risk_free = Curve(
            {dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98},
        )
        hazard = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98}, credit_recovery_rate=0.25)
        credit = CreditImpliedCurve(risk_free=risk_free, hazard=hazard)
        result = credit.rate(dt(2000, 1, 10), "10b")
        expected = 2.0 + 2.0 * 0.75
        assert abs(result - expected) < 3e-2

        hazard.update_meta("credit_recovery_rate", 0.90)
        result = credit.rate(dt(2000, 1, 10), "10b")
        expected = 2.0 + 2.0 * 0.1
        assert abs(result - expected) < 2e-2

    def test_meta_dynacism2(self):
        risk_free = Curve(
            {dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98},
        )
        hazard = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98}, credit_recovery_rate=0.25)
        credit = CreditImpliedCurve(risk_free=risk_free, hazard=hazard)
        hazard.update_meta("credit_recovery_rate", 0.90)
        result = credit.meta.credit_recovery_rate
        expected = 0.90
        assert abs(result - expected) < 1e-12


class TestMeta:
    def test_meta_mutation(self, curve, line_curve):
        # test all the rateslib curve types metas can be mutated

        curves = [curve, line_curve]
        dependent_curves = []

        dependent_curves.append(CompositeCurve([curve, curve]))
        dependent_curves.append(curve.shift(10))
        dependent_curves.append(curve.roll("10d"))
        dependent_curves.append(curve.translate(dt(2022, 3, 14)))
        dependent_curves.append(MultiCsaCurve([curve, curve]))
        fxf = FXForwards(
            FXRates({"eurusd": 1.10}, dt(2022, 3, 1)),
            {"eureur": curve, "eurusd": curve, "usdusd": curve},
        )
        dependent_curves.append(fxf.curve("usd", "eur"))
        dependent_curves.append(CreditImpliedCurve(risk_free=curve, hazard=curve))

        for c in dependent_curves + curves:
            from random import random

            x = int(random() * 100.0)
            c.meta._credit_discretization = x
            assert c.meta.credit_discretization == x

        curve.update_meta("credit_recovery_rate", 500.0)
        for c in dependent_curves:
            print(c)
            assert c.meta.credit_recovery_rate == 500.0
