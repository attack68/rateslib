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

import warnings
from datetime import datetime as dt
from math import exp

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pandas import DataFrame, MultiIndex
from pandas.errors import PerformanceWarning
from pandas.testing import assert_frame_equal, assert_series_equal
from rateslib import default_context
from rateslib.curves import CompositeCurve, Curve, LineCurve, MultiCsaCurve, index_left
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, gradient, ift_1dim, newton_1dim, newton_ndim
from rateslib.fx import FXForwards, FXRates
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface, FXSabrSmile, FXSabrSurface
from rateslib.instruments import (
    IRS,
    XCS,
    FloatRateNote,
    FXBrokerFly,
    FXCall,
    FXRiskReversal,
    FXStraddle,
    FXStrangle,
    FXSwap,
    Portfolio,
    Value,
)
from rateslib.solver import Gradients, Solver


class TestIFTSolver:
    @pytest.mark.parametrize("args", [(2.0, 3.0), (-2.0, -1.0)])
    def test_failed_state(self, args):
        def s(x):
            return x

        result = ift_1dim(s, 1.0, "bisection", args, raise_on_fail=False)
        assert result["state"] == -2

    def test_failed_state_raises(self):
        def s(x):
            return x

        with pytest.raises(ValueError, match="The internal iterative function `h` has reported"):
            ift_1dim(s, 1.0, "bisection", (2.0, 3.0), raise_on_fail=True)

    def test_solution_func_tol_state(self):
        def s(x):
            return x**2

        result = ift_1dim(s, 9.0, "bisection", (1.0, 5.0), func_tol=1e-10)
        # function should perform 2 iterations and arrive at 3.0
        assert result["state"] == 2
        assert result["g"] == 3.0

    def test_solution_conv_tol_state(self):
        def s(x):
            return x**2

        result = ift_1dim(s, 9.0, "bisection", (1.15, 5.0), conv_tol=1e-5)
        # function should perform many bisections iterations and arrive close to 3.0 with conv_tol
        assert result["state"] == 1
        assert result["iterations"] > 16
        assert abs(result["g"] - 3.0) < 1e-5

    def test_solution_max_iter_state(self):
        def s(x):
            return x**2

        result = ift_1dim(
            s, 9.0, "bisection", (1.15, 5.0), conv_tol=1e-5, max_iter=5, raise_on_fail=False
        )
        # function should perform many bisections iterations and arrive close to 3.0 with conv_tol
        assert result["state"] == -1

    def test_dual_returns(self):
        def s(x):
            return 3.0 / (1 + x / 100.0) + (100.0 + 3.0) / (1 + x / 100.0) ** 2

        result = ift_1dim(s, Dual(101.0, ["s"], []), "bisection", (2.0, 4.0), conv_tol=1e-5)

        # ds_dx = -3 / (1+g)**2 - 2*(103) / (1+g)**3
        g = result["g"]
        ds_dx = -3.0 / (1.0 + g / 100.0) ** 2 - 2.0 * (103.0) / (1.0 + g / 100.0) ** 3
        dg_ds_analytic = 1 / ds_dx * 100.0
        dg_ds_ad = gradient(g, ["s"])[0]

        assert abs(dg_ds_ad - dg_ds_analytic) < 1e-10

    def test_dual2_returns(self):
        # second part of dual returns
        def s(x):
            return 3.0 / (1 + x / 100.0) + (100.0 + 3.0) / (1 + x / 100.0) ** 2

        result = ift_1dim(s, Dual2(101.0, ["s"], [], []), "bisection", (2.0, 4.0), conv_tol=1e-5)

        # d2s_dx2 = 2.3 / (1+g)**3 + 6*(103) / (1+g)**4
        g = result["g"]
        ds_dx = -3.0 / (1.0 + g / 100.0) ** 2 - 2.0 * (103.0) / (1.0 + g / 100.0) ** 3
        d2s_dx2 = 6.0 / (1.0 + g / 100.0) ** 3 + 6.0 * (103.0) / (1.0 + g / 100.0) ** 4

        d2g_ds2_analytic = -100 * d2s_dx2 / ds_dx**3
        d2g_ds2_ad = gradient(g, ["s"], order=2)[0][0]

        assert abs(d2g_ds2_ad - d2g_ds2_analytic) < 1e-10

    def test_dekker(self):
        def s(x):
            return exp(x) + x**2

        s_tgt = s(2.0)
        result = ift_1dim(s, s_tgt, "modified_dekker", (1.15, 5.0), conv_tol=1e-12)
        assert result["g"] == 2.0
        assert result["iterations"] < 12

        result2 = ift_1dim(s, s_tgt, "bisection", (1.15, 5.0), conv_tol=1e-12)
        assert 30 < result2["iterations"] < 50

    def test_dekker_conv_tol(self):
        def s(x):
            return exp(x) + x**2

        s_tgt = s(2.0)
        result = ift_1dim(s, s_tgt, "modified_dekker", (1.15, 5.0), conv_tol=1e-3)
        assert result["state"] == 1

    def test_brent(self):
        def s(x):
            return exp(x) + x**2

        s_tgt = s(2.0)
        result = ift_1dim(s, s_tgt, "modified_brent", (1.15, 5.0), conv_tol=1e-12)
        assert result["g"] == 2.0
        assert result["iterations"] < 12

        # result2 = ift_1dim(s, s_tgt, "bisection", (1.15, 5.0), conv_tol=1e-12)
        # assert result["time"] <= result2["time"]

    def test_brent_conv_tol(self):
        def s(x):
            return exp(x) + x**2

        s_tgt = s(2.0)
        result = ift_1dim(s, s_tgt, "modified_brent", (1.15, 5.0), conv_tol=1e-3)
        assert result["state"] == 1

    def test_another_func(self):
        def s(g):
            from math import cos

            return cos(g) + g**3 + 2 * g**2 - 1.2

        s_tgt = s(-1.5)  # close to zero, 3 roots in [-4.0, 2.0]
        r_bi = ift_1dim(s, s_tgt, "bisection", (-4.0, 2.0))
        r_dk = ift_1dim(s, s_tgt, "modified_dekker", (-4.0, 2.0))
        r_br = ift_1dim(s, s_tgt, "modified_brent", (-4.0, 2.0))

        assert r_bi["status"] == "SUCCESS"
        assert r_dk["status"] == "SUCCESS"
        assert r_br["status"] == "SUCCESS"


class TestGradients:
    @classmethod
    def setup_class(cls):
        class Inst:
            def __init__(self, rate):
                self._rate = rate

            def rate(self, *args, **kwargs):
                return self._rate

        class SolverProxy(Gradients):
            variables = ["v1", "v2", "v3"]
            r = [Dual(1.0, ["v1"], []), Dual(3.0, ["v1", "v2", "v3"], [2.0, 1.0, -2.0])]
            _J = None
            instruments = [
                [Inst(Dual2(1.0, ["v1"], [1.0], [4.0])), {}],
                [
                    Inst(
                        Dual2(
                            3.0,
                            ["v1", "v2", "v3"],
                            [2.0, 1.0, -2.0],
                            [-2.0, 1.0, 1.0, 1.0, -3.0, 2.0, 1.0, 2.0, -4.0],
                        ),
                    ),
                    {},
                ],
            ]
            _J2 = None
            _ad = 2
            _grad_s_vT = np.array(
                [
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                ],
            )

        setattr(cls, "solver", SolverProxy())

    def test_J(self) -> None:
        expected = np.array(
            [
                [1.0, 2.0],
                [0.0, 1.0],
                [0.0, -2.0],
            ],
        )
        result = self.solver.J
        assert_allclose(result, expected)

    def test_grad_v_rT(self) -> None:
        assert_allclose(self.solver.J, self.solver.grad_v_rT)

    def test_J2(self) -> None:
        expected = np.array(
            [
                [
                    [8.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [-4.0, 2.0, 2.0],
                    [2.0, -6.0, 4.0],
                    [2.0, 4.0, -8.0],
                ],
            ],
        )
        expected = np.transpose(expected, (1, 2, 0))
        result = self.solver.J2
        assert_allclose(expected, result)

    def test_grad_v_v_rT(self) -> None:
        assert_allclose(self.solver.J2, self.solver.grad_v_v_rT)

    def test_grad_s_vT(self) -> None:
        expected = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
            ],
        )
        result = self.solver.grad_s_vT
        assert_allclose(expected, result)


@pytest.mark.parametrize("algo", ["gauss_newton", "levenberg_marquardt", "gradient_descent"])
def test_basic_solver(algo) -> None:
    curve = Curve(
        {
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
            dt(2025, 1, 1): 1.0,
        },
        id="v",
    )
    instruments = [
        (IRS(dt(2022, 1, 1), "1Y", "Q"), {"curves": curve}),
        (IRS(dt(2022, 1, 1), "2Y", "Q"), {"curves": curve}),
        (IRS(dt(2022, 1, 1), "3Y", "Q"), {"curves": curve}),
    ]
    s = np.array([1.0, 1.6, 2.0])
    solver = Solver(
        curves=[curve],
        instruments=instruments,
        s=s,
        algorithm=algo,
    )
    assert float(solver.g) < 1e-9
    assert curve.nodes.nodes[dt(2022, 1, 1)] == Dual(1.0, ["v0"], [1])
    expected = [1, 0.9899250357528555, 0.9680433953206192, 0.9407188354823821]
    for i, key in enumerate(curve.nodes.nodes.keys()):
        assert abs(float(curve.nodes.nodes[key]) - expected[i]) < 1e-6


def test_solver_repr():
    curve = Curve(
        {
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
            dt(2025, 1, 1): 1.0,
        },
        id="v",
    )
    instruments = [
        (IRS(dt(2022, 1, 1), "1Y", "Q"), {"curves": curve}),
        (IRS(dt(2022, 1, 1), "2Y", "Q"), {"curves": curve}),
        (IRS(dt(2022, 1, 1), "3Y", "Q"), {"curves": curve}),
    ]
    s = np.array([1.0, 1.6, 2.0])
    solver = Solver(curves=[curve], instruments=instruments, s=s, id="S_ID")
    result = solver.__repr__()
    expected = f"<rl.Solver:S_ID at {hex(id(solver))}>"
    assert result == expected


@pytest.mark.parametrize("algo", ["gauss_newton", "levenberg_marquardt", "gradient_descent"])
def test_solver_reiterate(algo) -> None:
    # test that curves are properly updated by a reiterate
    curve = Curve(
        {
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
            dt(2025, 1, 1): 1.0,
        },
        id="v",
    )
    instruments = [
        IRS(dt(2022, 1, 1), "1Y", "Q", curves="v"),
        IRS(dt(2022, 1, 1), "2Y", "Q", curves="v"),
        IRS(dt(2022, 1, 1), "3Y", "Q", curves="v"),
    ]
    s = np.array([1.0, 1.5, 2.0])
    solver = Solver(
        curves=[curve],
        instruments=instruments,
        s=s,
        algorithm=algo,
    )
    assert float(solver.g) < 1e-9

    solver.s[1] = 1.6
    solver.iterate()

    # now check that a reiteration has resolved the curve
    assert curve.nodes.nodes[dt(2022, 1, 1)] == Dual(1.0, ["v0"], [1])
    expected = [1, 0.9899250357528555, 0.9680433953206192, 0.9407188354823821]
    for i, key in enumerate(curve.nodes.nodes.keys()):
        assert abs(float(curve.nodes.nodes[key]) - expected[i]) < 1e-6


@pytest.mark.parametrize("algo", ["gauss_newton", "levenberg_marquardt", "gradient_descent"])
def test_basic_solver_line_curve(algo) -> None:
    curve = LineCurve(
        {
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
        },
        id="v",
    )
    instruments = [
        (Value(dt(2022, 1, 1)), {"curves": curve}),
        (Value(dt(2023, 1, 1)), {"curves": curve}),
        (Value(dt(2024, 1, 1)), {"curves": curve}),
    ]
    s = np.array([3.0, 3.6, 4.0])
    solver = Solver(
        curves=[curve],
        instruments=instruments,
        s=s,
        algorithm=algo,
    )
    assert float(solver.g) < 1e-9
    for i, key in enumerate(curve.nodes.nodes.keys()):
        assert abs(float(curve.nodes.nodes[key]) - s[i]) < 1e-5


def test_basic_spline_solver() -> None:
    spline_curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.99,
            dt(2024, 1, 1): 0.965,
            dt(2025, 1, 1): 0.93,
        },
        interpolation="log_linear",
        t=[
            dt(2023, 1, 1),
            dt(2023, 1, 1),
            dt(2023, 1, 1),
            dt(2023, 1, 1),
            dt(2024, 1, 1),
            dt(2025, 1, 3),
            dt(2025, 1, 3),
            dt(2025, 1, 3),
            dt(2025, 1, 3),
        ],
        id="v",
    )
    instruments = [
        (IRS(dt(2022, 1, 1), "1Y", "Q"), {"curves": spline_curve}),
        (IRS(dt(2022, 1, 1), "2Y", "Q"), {"curves": spline_curve}),
        (IRS(dt(2022, 1, 1), "3Y", "Q"), {"curves": spline_curve}),
    ]
    s = np.array([1.0, 1.6, 2.0])
    solver = Solver(
        curves=[spline_curve],
        instruments=instruments,
        s=s,
        algorithm="gauss_newton",
    )
    assert float(solver.g) < 1e-12
    assert spline_curve.nodes.nodes[dt(2022, 1, 1)] == Dual(1.0, ["v0"], [1])
    expected = [1, 0.98992503575307, 0.9680377261843034, 0.9407048036486593]
    for i, key in enumerate(spline_curve.nodes.nodes.keys()):
        assert abs(float(spline_curve.nodes.nodes[key]) - expected[i]) < 1e-11


def test_large_spline_solver() -> None:
    dates = [
        dt(2000, 1, 3),
        dt(2001, 1, 3),
        dt(2002, 1, 3),
        dt(2003, 1, 3),
        dt(2004, 1, 3),
        dt(2005, 1, 3),
        dt(2006, 1, 3),
        dt(2007, 1, 3),
        dt(2008, 1, 3),
        dt(2009, 1, 3),
        dt(2010, 1, 3),
        dt(2012, 1, 3),
        dt(2015, 1, 3),
        dt(2020, 1, 3),
        dt(2025, 1, 3),
        dt(2030, 1, 3),
        dt(2035, 1, 3),
        dt(2040, 1, 3),
        dt(2050, 1, 3),
    ]
    curve = Curve(
        nodes=dict.fromkeys(dates, 1.0),
        t=[dt(2000, 1, 3)] * 3 + dates[:-1] + [dt(2050, 1, 5)] * 4,
        calendar="nyc",
    )
    solver = Solver(
        curves=[curve],
        instruments=[IRS(dt(2000, 1, 3), _, spec="usd_irs", curves=curve) for _ in dates[1:]],
        s=[1.0 + _ / 25 for _ in range(18)],
    )
    assert solver.result["status"] == "SUCCESS"


def test_solver_raises_len() -> None:
    with pytest.raises(ValueError, match=r"`s: 2` \(rates\)  must be same length as"):
        Solver(
            instruments=[1],
            s=[1, 2],
        )

    with pytest.raises(ValueError, match=r"`instrument_labels: 2` must be same length as"):
        Solver(
            instruments=[1],
            s=[1],
            instrument_labels=[1, 2],
        )

    with pytest.raises(ValueError, match=r"`weights: 1` must be same length as"):
        Solver(
            instruments=[1, 2],
            s=[1, 2],
            instrument_labels=[1, 2],
            weights=[1],
        )


def test_basic_solver_weights() -> None:
    # This test replicates test_basic_solver with the 3Y rate at two different rates.
    # We vary the weights argument to selectively decide which one to use.
    curve = Curve(
        {
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
            dt(2025, 1, 1): 1.0,
        },
        id="v",
    )
    instruments = [
        (IRS(dt(2022, 1, 1), "1Y", "Q"), {"curves": curve}),
        (IRS(dt(2022, 1, 1), "2Y", "Q"), {"curves": curve}),
        (IRS(dt(2022, 1, 1), "3Y", "Q"), {"curves": curve}),
        (IRS(dt(2022, 1, 1), "3Y", "Q"), {"curves": curve}),
    ]
    s = np.array([1.0, 1.6, 2.02, 1.98])  # average 3Y at approximately 2.0%
    with default_context("algorithm", "gauss_newton"):
        solver = Solver(
            curves=[curve],
            instruments=instruments,
            s=s,
            func_tol=0.00085,
        )
    assert float(solver.g) < 0.00085
    assert curve.nodes.nodes[dt(2022, 1, 1)] == Dual(1.0, ["v0"], [1])
    expected = [1, 0.9899250357528555, 0.9680433953206192, 0.9407188354823821]
    for i, key in enumerate(curve.nodes.nodes.keys()):
        assert abs(float(curve.nodes.nodes[key]) - expected[i]) < 1e-6

    solver = Solver(
        curves=[curve],
        instruments=instruments,
        s=s,
        weights=[1, 1, 1, 1e-6],
        func_tol=1e-7,
        algorithm="gauss_newton",
    )
    assert abs(float(instruments[2][0].rate(curves=curve)) - 2.02) < 1e-4

    solver = Solver(
        curves=[curve],
        instruments=instruments,
        s=s,
        weights=[1, 1, 1e-6, 1],
        func_tol=1e-7,
        algorithm="gauss_newton",
    )
    assert abs(float(instruments[2][0].rate(curves=curve)) - 1.98) < 1e-4


def test_solver_independent_curve() -> None:
    # Test that a solver can use an independent curve as a static object and solve
    # without mutating that un-referenced object.
    independent_curve = Curve(
        {
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.98,
            dt(2024, 1, 1): 0.96,
            dt(2025, 1, 1): 0.94,
        },
    )
    expected = independent_curve.copy()
    var_curve = Curve(
        {
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.99,
            dt(2024, 1, 1): 0.98,
            dt(2025, 1, 1): 0.97,
        },
    )
    instruments = [
        (IRS(dt(2022, 1, 1), "1Y", "Q"), {"curves": [var_curve, independent_curve]}),
        (IRS(dt(2022, 1, 1), "2Y", "Q"), {"curves": [var_curve, independent_curve]}),
        (IRS(dt(2022, 1, 1), "3Y", "Q"), {"curves": [var_curve, independent_curve]}),
    ]
    s = np.array([2.00, 2.00, 2.00])
    with default_context("curve_not_in_solver", "ignore"):
        Solver(
            curves=[var_curve],
            instruments=instruments,
            s=s,
            func_tol=1e-13,
            conv_tol=1e-13,
        )
    for i, instrument in enumerate(instruments):
        assert abs(float(instrument[0].rate(**instrument[1]) - s[i])) < 1e-7
    assert independent_curve == expected


class TestSolverCompositeCurve:
    def test_solver_composite_curve(self) -> None:
        # this test creates a solver with a composite curve
        # for the purpose of adding a turn
        c_base = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0, dt(2024, 1, 1): 1.0, dt(2025, 1, 1): 1.0},
            id="sek_base",
        )
        c_turns = Curve(
            {
                dt(2022, 1, 1): 1.0,
                dt(2022, 12, 30): 1.0,
                dt(2023, 1, 1): 1.0,
                dt(2025, 1, 1): 1.0,
            },
            id="sek_turns",
        )
        composite_curve = CompositeCurve([c_base, c_turns], id="sek")

        instruments_turns = [
            IRS(dt(2022, 1, 1), "1d", "A", curves="sek_turns"),
            IRS(dt(2022, 12, 30), "1d", "A", curves="sek_turns"),
            IRS(dt(2023, 1, 1), "1d", "A", curves="sek_turns"),
        ]
        s_turns = [0.0, -0.50, 0.0]
        labels_turns = ["NA1", "Turn1", "NA2"]

        instruments_base = [
            IRS(dt(2022, 1, 1), "1Y", "A", curves="sek"),
            IRS(dt(2022, 1, 1), "2Y", "A", curves="sek"),
            IRS(dt(2022, 1, 1), "3Y", "A", curves="sek"),
        ]
        s_base = [2.0, 2.3, 2.4]
        labels_base = ["1Y", "2Y", "3Y"]

        solver = Solver(
            curves=[c_base, c_turns, composite_curve],
            instruments=instruments_turns + instruments_base,
            s=s_turns + s_base,
            instrument_labels=labels_turns + labels_base,
            id="solv",
        )

        test_irs = IRS(dt(2022, 6, 1), "15M", "A", notional=1e6, curves="sek")

        expected = 2.31735564
        result = test_irs.rate(solver=solver)
        assert (result - expected) < 1e-8

        delta = test_irs.delta(solver=solver)
        expected = DataFrame(
            data=[
                -0.22582768057036448,
                0.22571855114358436,
                0.00010912854804701055,
                -9.15902876400274,
                131.75543312,
                0.0033383280,
            ],
            columns=MultiIndex.from_tuples([("usd", "usd")], names=["local_ccy", "display_ccy"]),
            index=MultiIndex.from_tuples(
                [
                    ("instruments", "solv", "NA1"),
                    ("instruments", "solv", "Turn1"),
                    ("instruments", "solv", "NA2"),
                    ("instruments", "solv", "1Y"),
                    ("instruments", "solv", "2Y"),
                    ("instruments", "solv", "3Y"),
                ],
                names=["type", "solver", "label"],
            ),
        )
        assert_frame_equal(delta, expected)


def test_non_unique_curves() -> None:
    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="A")
    curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="A")
    solver = Solver(
        curves=[curve],
        instruments=[(IRS(dt(2022, 1, 1), "1Y", "Q"), {"curves": curve})],
        s=[1],
    )

    with pytest.raises(ValueError, match="`curves` must each have their own unique"):
        Solver(
            curves=[curve2],
            instruments=[(IRS(dt(2022, 1, 1), "1Y", "Q"), {"curves": curve})],
            s=[2],
            pre_solvers=[solver],
        )

    with pytest.raises(ValueError, match="`curves` must each have their own unique"):
        Solver(
            curves=[curve, curve2],
            instruments=[(IRS(dt(2022, 1, 1), "1Y", "Q"), {"curves": curve})],
            s=[2],
        )


def test_max_iterations() -> None:
    # This test replicates has an oscillatory solution between the different 3y rates.
    curve = Curve(
        {
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
            dt(2025, 1, 1): 1.0,
        },
        id="v",
    )
    instruments = [
        (IRS(dt(2022, 1, 1), "1Y", "Q"), {"curves": curve}),
        (IRS(dt(2022, 1, 1), "2Y", "Q"), {"curves": curve}),
        (IRS(dt(2022, 1, 1), "3Y", "Q"), {"curves": curve}),
        (IRS(dt(2022, 1, 1), "3Y", "Q"), {"curves": curve}),
    ]
    s = np.array([1.0, 1.6, 2.02, 1.98])  # average 3Y at approximately 2.0%
    with default_context("algorithm", "gauss_newton"):
        solver = Solver(
            curves=[curve],
            instruments=instruments,
            s=s,
            func_tol=1e-10,
            max_iter=30,
        )
    assert len(solver.g_list) == 31


def test_solver_pre_solver_dependency_generates_same_delta() -> None:
    """
    Build an ESTR curve with solver1.
    Build an IBOR curve with solver2 dependent upon solver1.

    Build an ESTR and IBOR curve simultaneously inside the same solver3.

    Test the delta and the instrument calibration error
    """
    eur_disc_curve = Curve(
        nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0, dt(2024, 1, 1): 1.0},
        id="eur",
    )
    eur_instruments = [
        (IRS(dt(2022, 1, 1), "8M", "A"), {"curves": eur_disc_curve}),
        (IRS(dt(2022, 1, 1), "16M", "A"), {"curves": eur_disc_curve}),
        (IRS(dt(2022, 1, 1), "2Y", "A"), {"curves": eur_disc_curve}),
    ]
    eur_disc_s = [2.01, 2.22, 2.55]
    eur_disc_solver = Solver([eur_disc_curve], [], eur_instruments, eur_disc_s, id="estr")

    eur_ibor_curve = Curve(
        nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0, dt(2024, 1, 1): 1.0},
        id="eur_ibor",
    )
    eur_ibor_instruments = [
        (IRS(dt(2022, 1, 1), "1Y", "A"), {"curves": [eur_ibor_curve, eur_disc_curve]}),
        (IRS(dt(2022, 1, 1), "2Y", "A"), {"curves": [eur_ibor_curve, eur_disc_curve]}),
    ]
    eur_ibor_s = [2.25, 2.65]
    eur_solver2 = Solver(
        [eur_ibor_curve],
        [],
        eur_ibor_instruments,
        eur_ibor_s,
        pre_solvers=[eur_disc_solver],
        id="ibor",
    )

    eur_disc_curve2 = Curve(
        {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0, dt(2024, 1, 1): 1.0},
        id="eur",
    )
    eur_ibor_curve2 = Curve(
        {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0, dt(2024, 1, 1): 1.0},
        id="eur_ibor",
    )
    eur_instruments2 = [
        (IRS(dt(2022, 1, 1), "8M", "A"), {"curves": eur_disc_curve2}),
        (IRS(dt(2022, 1, 1), "16M", "A"), {"curves": eur_disc_curve2}),
        (IRS(dt(2022, 1, 1), "2Y", "A"), {"curves": eur_disc_curve2}),
        (IRS(dt(2022, 1, 1), "1Y", "A"), {"curves": [eur_ibor_curve2, eur_disc_curve2]}),
        (IRS(dt(2022, 1, 1), "2Y", "A"), {"curves": [eur_ibor_curve2, eur_disc_curve2]}),
    ]
    eur_disc_s2 = [2.01, 2.22, 2.55, 2.25, 2.65]
    eur_solver_sim = Solver(
        [eur_disc_curve2, eur_ibor_curve2],
        [],
        eur_instruments2,
        eur_disc_s2,
        id="eur_sol_sim",
        instrument_labels=["estr0", "estr1", "estr2", "ibor0", "ibor1"],
    )

    eur_swap = IRS(
        dt(2022, 3, 1),
        "16M",
        "M",
        fixed_rate=3.0,
    )

    delta_sim = eur_swap.delta(curves=[eur_ibor_curve2, eur_disc_curve2], solver=eur_solver_sim)
    delta_pre = eur_swap.delta(curves=[eur_ibor_curve, eur_disc_curve], solver=eur_solver2)
    delta_pre.index = delta_sim.index
    assert_frame_equal(delta_sim, delta_pre)

    error_sim = eur_solver_sim.error
    error_pre = eur_solver2.error
    assert_series_equal(error_pre, error_sim, check_index=False, rtol=1e-5, atol=1e-3)


def test_delta_gamma_calculation() -> None:
    estr_curve = Curve(
        {dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0},
        id="estr_curve",
    )
    estr_instruments = [
        (IRS(dt(2022, 1, 1), "10Y", "A"), {"curves": estr_curve}),
        (IRS(dt(2022, 1, 1), "20Y", "A"), {"curves": estr_curve}),
    ]
    estr_solver = Solver(
        [estr_curve],
        [],
        estr_instruments,
        [2.0, 1.5],
        id="estr",
        instrument_labels=["10Y", "20Y"],
    )

    # Mechanism 1: dynamic
    eur_swap = IRS(dt(2032, 1, 1), "10Y", "A", notional=100e6)
    assert (
        74430 < float(eur_swap.delta(curves=estr_curve, solver=estr_solver).sum().iloc[0]) < 74432
    )
    assert -229 < float(eur_swap.gamma(curves=estr_curve, solver=estr_solver).sum().sum()) < -228

    # Mechanism 1: dynamic names
    assert (
        74430 < float(eur_swap.delta(curves="estr_curve", solver=estr_solver).sum().iloc[0]) < 74432
    )
    assert -229 < float(eur_swap.gamma(curves="estr_curve", solver=estr_solver).sum().sum()) < -228

    # Mechanism 1: fails on None curve specification
    with pytest.raises(TypeError, match="`curves` have not been supplied correctly"):
        assert eur_swap.delta(solver=estr_solver)
    with pytest.raises(TypeError, match="`curves` have not been supplied correctly"):
        assert eur_swap.gamma(solver=estr_solver)

    # Mechanism 2: static specific
    eur_swap = IRS(dt(2032, 1, 1), "10Y", "A", notional=100e6, curves=estr_curve)
    assert 74430 < float(eur_swap.delta(solver=estr_solver).sum().iloc[0]) < 74432
    assert -229 < float(eur_swap.gamma(solver=estr_solver).sum().sum()) < -228

    # Mechanism 2: static named
    eur_swap = IRS(dt(2032, 1, 1), "10Y", "A", notional=100e6, curves="estr_curve")
    assert 74430 < float(eur_swap.delta(solver=estr_solver).sum().iloc[0]) < 74432
    assert -229 < float(eur_swap.gamma(solver=estr_solver).sum().sum()) < -228


def test_solver_delta_fx_noinput() -> None:
    estr_curve = Curve(
        {dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0},
        id="estr_curve",
    )
    estr_instruments = [
        (IRS(dt(2022, 1, 1), "10Y", "A"), {"curves": estr_curve}),
        (IRS(dt(2022, 1, 1), "20Y", "A"), {"curves": estr_curve}),
    ]
    estr_solver = Solver(
        [estr_curve],
        [],
        estr_instruments,
        [2.0, 1.5],
        id="estr",
        instrument_labels=["10Y", "20Y"],
    )
    eur_swap = IRS(dt(2032, 1, 1), "10Y", "A", notional=100e6, fixed_rate=2)
    npv = eur_swap.npv(curves=estr_curve, solver=estr_solver, local=True)
    result = estr_solver.delta(npv)
    assert type(result) is DataFrame


def test_solver_pre_solver_dependency_generates_same_gamma() -> None:
    estr_curve = Curve({dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0})
    estr_instruments = [
        (IRS(dt(2022, 1, 1), "7Y", "A"), {"curves": estr_curve}),
        (IRS(dt(2022, 1, 1), "15Y", "A"), {"curves": estr_curve}),
        (IRS(dt(2022, 1, 1), "20Y", "A"), {"curves": estr_curve}),
    ]
    estr_s = [2.0, 1.75, 1.5]
    estr_labels = ["7ye", "15ye", "20ye"]
    estr_solver = Solver(
        [estr_curve],
        [],
        estr_instruments,
        estr_s,
        id="estr",
        instrument_labels=estr_labels,
        algorithm="gauss_newton",
    )

    ibor_curve = Curve({dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0})
    ibor_instruments = [
        (IRS(dt(2022, 1, 1), "10Y", "A"), {"curves": [ibor_curve, estr_curve]}),
        (IRS(dt(2022, 1, 1), "20Y", "A"), {"curves": [ibor_curve, estr_curve]}),
    ]
    ibor_s = [2.1, 1.65]
    ibor_labels = ["10Yi", "20Yi"]
    ibor_solver = Solver(
        [ibor_curve],
        [],
        ibor_instruments,
        ibor_s,
        id="ibor",
        instrument_labels=ibor_labels,
        pre_solvers=[estr_solver],
        algorithm="gauss_newton",
    )

    eur_swap = IRS(dt(2032, 1, 1), "10Y", "A", notional=100e6)
    gamma_pre = eur_swap.gamma(curves=[ibor_curve, estr_curve], solver=ibor_solver)
    delta_pre = eur_swap.delta(curves=[ibor_curve, estr_curve], solver=ibor_solver)

    estr_curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0})
    ibor_curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0})
    sim_instruments = [
        (IRS(dt(2022, 1, 1), "7Y", "A"), {"curves": estr_curve2}),
        (IRS(dt(2022, 1, 1), "15Y", "A"), {"curves": estr_curve2}),
        (IRS(dt(2022, 1, 1), "20Y", "A"), {"curves": estr_curve2}),
        (IRS(dt(2022, 1, 1), "10Y", "A"), {"curves": [ibor_curve2, estr_curve2]}),
        (IRS(dt(2022, 1, 1), "20Y", "A"), {"curves": [ibor_curve2, estr_curve2]}),
    ]
    simultaneous_solver = Solver(
        [estr_curve2, ibor_curve2],
        [],
        sim_instruments,
        estr_s + ibor_s,
        id="simul",
        instrument_labels=estr_labels + ibor_labels,
        algorithm="gauss_newton",
    )
    gamma_sim = eur_swap.gamma(curves=[ibor_curve2, estr_curve2], solver=simultaneous_solver)
    delta_sim = eur_swap.delta(curves=[ibor_curve2, estr_curve2], solver=simultaneous_solver)

    # check arrays in construction of gamma
    grad_s_vT_sim = simultaneous_solver.grad_s_vT_pre
    grad_s_vT_pre = ibor_solver.grad_s_vT_pre
    assert_allclose(grad_s_vT_pre, grad_s_vT_sim, atol=1e-14, rtol=1e-10)

    simultaneous_solver._set_ad_order(2)
    J2_sim = simultaneous_solver.J2_pre
    ibor_solver._set_ad_order(2)
    J2_pre = ibor_solver.J2_pre
    assert_allclose(J2_pre, J2_sim, atol=1e-14, rtol=1e-10)

    grad_s_s_vT_sim = simultaneous_solver.grad_s_s_vT_pre
    grad_s_s_vT_pre = ibor_solver.grad_s_s_vT_pre
    assert_allclose(grad_s_s_vT_pre, grad_s_s_vT_sim, atol=1e-14, rtol=1e-10)

    gamma_pre.index = gamma_sim.index
    gamma_pre.columns = gamma_sim.columns
    delta_pre.index = delta_sim.index
    assert_frame_equal(delta_sim, delta_pre)
    assert_frame_equal(gamma_sim, gamma_pre)


def test_nonmutable_presolver_defaults() -> None:
    estr_curve = Curve({dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0})
    estr_instruments = [
        (IRS(dt(2022, 1, 1), "10Y", "A"), {"curves": estr_curve}),
    ]
    estr_s = [2.0]
    estr_labels = ["10ye"]
    estr_solver = Solver(
        [estr_curve],
        [],
        estr_instruments,
        estr_s,
        id="estr",
        instrument_labels=estr_labels,
    )
    with pytest.raises(AttributeError, match="'tuple' object has no attribute"):
        estr_solver.pre_solvers.extend([1, 2, 3])


def test_solver_grad_s_vT_methods_equivalent() -> None:
    curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
            dt(2025, 1, 1): 1.0,
            dt(2026, 1, 1): 1.0,
            dt(2027, 1, 1): 1.0,
        },
    )
    instruments = [
        (IRS(dt(2022, 1, 1), "2Y", "A"), {"curves": curve}),
        (IRS(dt(2023, 1, 1), "1Y", "A"), {"curves": curve}),
        (IRS(dt(2023, 1, 1), "2Y", "A"), {"curves": curve}),
        (IRS(dt(2022, 5, 1), "4Y", "A"), {"curves": curve}),
        (IRS(dt(2023, 1, 1), "4Y", "A"), {"curves": curve}),
    ]
    s = [1.2, 1.4, 1.6, 1.7, 1.9]
    solver = Solver([curve], [], instruments, s, algorithm="gauss_newton")

    solver._grad_s_vT_method = "_grad_s_vT_final_iteration_analytical"
    grad_s_vT_final_iter_anal = solver.grad_s_vT

    solver._grad_s_vT_method = "_grad_s_vT_final_iteration_dual"
    solver._grad_s_vT_final_iteration_algo = "gauss_newton_final"
    solver._reset_properties_()
    grad_s_vT_final_iter_dual = solver.grad_s_vT

    solver._grad_s_vT_method = "_grad_s_vT_fixed_point_iteration"
    solver._reset_properties_()
    grad_s_vT_fixed_point_iter = solver.grad_s_vT

    assert_allclose(grad_s_vT_final_iter_dual, grad_s_vT_final_iter_anal, atol=1e-12)
    assert_allclose(grad_s_vT_fixed_point_iter, grad_s_vT_final_iter_anal, atol=1e-12)
    assert_allclose(grad_s_vT_final_iter_dual, grad_s_vT_fixed_point_iter, atol=1e-12)


def test_solver_grad_s_vT_methods_equivalent_overspecified_curve() -> None:
    curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
            dt(2025, 1, 1): 1.0,
            # dt(2026, 1, 1): 1.0,
            dt(2027, 1, 1): 1.0,
        },
    )
    instruments = [
        (IRS(dt(2022, 1, 1), "2Y", "A"), {"curves": curve}),
        (IRS(dt(2023, 1, 1), "1Y", "A"), {"curves": curve}),
        (IRS(dt(2023, 1, 1), "2Y", "A"), {"curves": curve}),
        (IRS(dt(2022, 5, 1), "4Y", "A"), {"curves": curve}),
        (IRS(dt(2023, 1, 1), "4Y", "A"), {"curves": curve}),
    ]
    s = [1.2, 1.4, 1.6, 1.7, 1.9]
    solver = Solver([curve], [], instruments, s, algorithm="gauss_newton")

    solver._grad_s_vT_method = "_grad_s_vT_final_iteration_analytical"
    grad_s_vT_final_iter_anal = solver.grad_s_vT

    solver._grad_s_vT_method = "_grad_s_vT_final_iteration_dual"
    solver._grad_s_vT_final_iteration_algo = "gauss_newton_final"
    solver._reset_properties_()
    grad_s_vT_final_iter_dual = solver.grad_s_vT

    solver._grad_s_vT_method = "_grad_s_vT_fixed_point_iteration"
    solver._reset_properties_()
    grad_s_vT_fixed_point_iter = solver.grad_s_vT

    assert_allclose(grad_s_vT_final_iter_dual, grad_s_vT_final_iter_anal, atol=1e-6)
    assert_allclose(grad_s_vT_fixed_point_iter, grad_s_vT_final_iter_anal, atol=1e-6)
    assert_allclose(grad_s_vT_final_iter_dual, grad_s_vT_fixed_point_iter, atol=1e-6)


def test_solver_second_order_vars_raise_on_first_order() -> None:
    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="A")
    solver = Solver(
        curves=[curve],
        instruments=[(IRS(dt(2022, 1, 1), "1Y", "Q"), {"curves": curve})],
        s=[1],
    )

    with pytest.raises(ValueError, match="Cannot perform second derivative calc"):
        solver.J2

    with pytest.raises(ValueError, match="Cannot perform second derivative calc"):
        solver.grad_s_s_vT


def test_solver_second_order_vars_raise_on_first_order_pre_solvers() -> None:
    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="A")
    solver = Solver(
        curves=[curve],
        instruments=[IRS(dt(2022, 1, 1), "1Y", "Q", curves=curve)],
        s=[1],
    )
    curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="B")
    solver2 = Solver(
        curves=[curve2],
        instruments=[IRS(dt(2022, 1, 1), "1Y", "Q", curves=curve2)],
        s=[1],
        pre_solvers=[solver],
    )

    with pytest.raises(ValueError, match="Cannot perform second derivative calc"):
        solver2.J2_pre

    with pytest.raises(ValueError, match="Cannot perform second derivative calc"):
        solver.grad_s_s_vT_pre


def test_bad_algo_raises() -> None:
    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="A")
    with pytest.raises(NotImplementedError, match="`algorithm`: bad_algo"):
        Solver(
            curves=[curve],
            instruments=[IRS(dt(2022, 1, 1), "1Y", "Q", curves=curve)],
            s=[1],
            algorithm="bad_algo",
        )


def test_solver_float_rate_bond() -> None:
    """
    This test checks the rate method of FloatRateNote when using complex rate spread
    calculations (which artificially introduces Dual2 and then removes it)
    """
    d_c = Curve(
        {
            dt(2022, 1, 1): 1.0,
            dt(2022, 7, 1): 0.94,
            dt(2023, 1, 1): 0.92,
            dt(2024, 1, 1): 0.9,
        },
        id="credit",
    )
    f_c = d_c.copy()
    f_c._id = "rfr"
    instruments = [
        (
            FloatRateNote(
                dt(2022, 1, 1),
                "6M",
                "Q",
                spread_compound_method="isda_compounding",
                settle=2,
            ),
            {"metric": "spread", "curves": [f_c, d_c]},
        ),
        (
            FloatRateNote(
                dt(2022, 1, 1),
                "1y",
                "Q",
                spread_compound_method="isda_compounding",
                settle=2,
                curves=[f_c, d_c],
            ),
            {"metric": "spread"},
        ),
        (
            FloatRateNote(
                dt(2022, 1, 1),
                "18m",
                "Q",
                spread_compound_method="isda_compounding",
                settle=2,
                curves=[f_c, d_c],
            ),
            {"metric": "spread"},
        ),
    ]
    Solver([d_c], [], instruments, [25, 25, 25])
    result = d_c.rate(dt(2022, 7, 1), "1D")
    expected = f_c.rate(dt(2022, 7, 1), "1D") + 0.25
    assert abs(result - expected) < 3e-4


def test_solver_grad_s_s_vt_methods_equivalent() -> None:
    curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
            dt(2025, 1, 1): 1.0,
            dt(2026, 1, 1): 1.0,
            dt(2027, 1, 1): 1.0,
            dt(2028, 1, 1): 1.0,
            dt(2029, 1, 1): 1.0,
        },
        id="curve",
    )
    instruments = [
        IRS(dt(2022, 1, 1), "1y", "A", curves="curve"),
        IRS(dt(2022, 1, 1), "2y", "A", curves="curve"),
        IRS(dt(2022, 1, 1), "3y", "A", curves="curve"),
        IRS(dt(2022, 1, 1), "4y", "A", curves="curve"),
        IRS(dt(2022, 1, 1), "5y", "A", curves="curve"),
        IRS(dt(2022, 1, 1), "6y", "A", curves="curve"),
        IRS(dt(2022, 1, 1), "7y", "A", curves="curve"),
    ]
    with default_context("algorithm", "gauss_newton"):
        solver = Solver(
            curves=[curve],
            instruments=instruments,
            s=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        )
    grad_s_s_vt_fwddiff = solver._grad_s_s_vT_fwd_difference_method()
    solver._set_ad_order(order=2)
    grad_s_s_vt_final = solver._grad_s_s_vT_final_iteration_analytical()
    solver._set_ad_order(order=1)
    assert_allclose(grad_s_s_vt_final, grad_s_s_vt_fwddiff, atol=5e-7)


def test_gamma_raises() -> None:
    curve = Curve(
        {
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
            dt(2025, 1, 1): 1.0,
        },
        id="v",
    )
    instruments = [
        IRS(dt(2022, 1, 1), "1Y", "Q", curves=curve),
        IRS(dt(2022, 1, 1), "2Y", "Q", curves=curve),
        IRS(dt(2022, 1, 1), "3Y", "Q", curves=curve),
    ]
    s = np.array([1.0, 1.6, 2.0])
    solver = Solver(
        curves=[curve],
        instruments=instruments,
        s=s,
    )
    with pytest.raises(ValueError, match="`Solver` must be in ad order 2"):
        solver.gamma(100)


def test_delta_irs_guide() -> None:
    # this mirrors the delta user guide page
    usd_curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2022, 2, 1): 1.0,
            dt(2022, 4, 1): 1.0,
            dt(2023, 1, 1): 1.0,
        },
        id="sofr",
    )
    instruments = [
        IRS(dt(2022, 1, 1), "1m", "A", curves="sofr"),
        IRS(dt(2022, 1, 1), "3m", "A", curves="sofr"),
        IRS(dt(2022, 1, 1), "1y", "A", curves="sofr"),
    ]
    usd_solver = Solver(
        curves=[usd_curve],
        id="usd_sofr",
        instruments=instruments,
        s=[2.5, 3.25, 4.0],
        instrument_labels=["1m", "3m", "1y"],
    )
    irs = IRS(
        effective=dt(2022, 1, 1),
        termination="6m",
        frequency="A",
        currency="usd",
        fixed_rate=6.0,
        curves="sofr",
    )
    result = irs.delta(solver=usd_solver)  # local overrides base to USD
    # result = irs.delta(solver=usd_solver, base="eur", local=True)  # local overrides base to USD
    expected = DataFrame(
        [[0], [16.77263], [32.60487]],
        index=MultiIndex.from_product(
            [["instruments"], ["usd_sofr"], ["1m", "3m", "1y"]],
            names=["type", "solver", "label"],
        ),
        columns=MultiIndex.from_tuples([("usd", "usd")], names=["local_ccy", "display_ccy"]),
    )
    assert_frame_equal(result, expected)


def test_delta_irs_guide_fx_base() -> None:
    # this mirrors the delta user guide page
    usd_curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2022, 2, 1): 1.0,
            dt(2022, 4, 1): 1.0,
            dt(2023, 1, 1): 1.0,
        },
        id="sofr",
    )
    instruments = [
        IRS(dt(2022, 1, 1), "1m", "A", curves="sofr"),
        IRS(dt(2022, 1, 1), "3m", "A", curves="sofr"),
        IRS(dt(2022, 1, 1), "1y", "A", curves="sofr"),
    ]
    usd_solver = Solver(
        curves=[usd_curve],
        id="usd_sofr",
        instruments=instruments,
        s=[2.5, 3.25, 4.0],
        instrument_labels=["1m", "3m", "1y"],
    )
    irs = IRS(
        effective=dt(2022, 1, 1),
        termination="6m",
        frequency="A",
        currency="usd",
        fixed_rate=6.0,
        curves="sofr",
    )
    fxr = FXRates({"eurusd": 1.1})
    result = irs.delta(solver=usd_solver, base="eur", fx=fxr)
    expected = DataFrame(
        [
            [0, 0, 0],
            [15.247847, 15.247847, 16.772632],
            [29.640788, 29.640788, 32.60487],
            [0.926514, 0.926514, 0.0],
        ],
        index=MultiIndex.from_tuples(
            [
                ("instruments", "usd_sofr", "1m"),
                ("instruments", "usd_sofr", "3m"),
                ("instruments", "usd_sofr", "1y"),
                ("fx", "fx", "eurusd"),
            ],
            names=["type", "solver", "label"],
        ),
        columns=MultiIndex.from_tuples(
            [
                ("all", "eur"),
                ("usd", "eur"),
                ("usd", "usd"),
            ],
            names=["local_ccy", "display_ccy"],
        ),
    )
    assert_frame_equal(result, expected)


# def test_irs_delta_curves_undefined():
#     # the IRS is not constructed under best practice.
#     # The delta solver does not know how to price the irs
#     curve = Curve({dt(2022, 1, 1): 1.0, dt(2027, 1, 1): 0.99, dt(2032, 1, 1): 0.98},
#                   id="sonia")
#     instruments = [
#         IRS(dt(2022, 1, 1), "5y", "A", curves="sonia"),
#         IRS(dt(2027, 1, 1), "5y", "A", curves="sonia"),
#     ]
#     solver = Solver(
#         curves=[curve],
#         instruments=instruments,
#         s=[2.0, 2.5],
#     )
#     irs = IRS(dt(2022, 1, 1), "10y", "S", fixed_rate=2.38)
#     with pytest.raises(TypeError, match="`curves` have not been supplied"):
#         irs.delta(solver=solver)


def test_mechanisms_guide_gamma() -> None:
    instruments = [
        IRS(dt(2022, 1, 1), "4m", "Q", curves="sofr"),
        IRS(dt(2022, 1, 1), "8m", "Q", curves="sofr"),
    ]
    s = [1.85, 2.10]
    ll_curve = Curve(
        nodes={dt(2022, 1, 1): 1.0, dt(2022, 5, 1): 1.0, dt(2022, 9, 1): 1.0},
        interpolation="log_linear",
        id="sofr",
    )
    ll_solver = Solver(
        curves=[ll_curve],
        instruments=instruments,
        s=s,
        instrument_labels=["4m", "8m"],
        id="sofr",
    )

    instruments = [
        IRS(dt(2022, 1, 1), "3m", "Q", curves="estr"),
        IRS(dt(2022, 1, 1), "9m", "Q", curves="estr"),
    ]
    s = [0.75, 1.65]
    ll_curve = Curve(
        nodes={dt(2022, 1, 1): 1.0, dt(2022, 4, 1): 1.0, dt(2022, 10, 1): 1.0},
        interpolation="log_linear",
        id="estr",
    )
    combined_solver = Solver(
        curves=[ll_curve],
        instruments=instruments,
        s=s,
        instrument_labels=["3m", "9m"],
        pre_solvers=[ll_solver],
        id="estr",
    )

    irs = IRS(
        effective=dt(2022, 1, 1),
        termination="6m",
        frequency="Q",
        currency="usd",
        notional=500e6,
        fixed_rate=2.0,
        curves="sofr",
    )
    irs2 = IRS(
        effective=dt(2022, 1, 1),
        termination="6m",
        frequency="Q",
        currency="eur",
        notional=-300e6,
        fixed_rate=1.0,
        curves="estr",
    )
    pf = Portfolio([irs, irs2])
    pf.npv(solver=combined_solver, local=True)
    pf.delta(solver=combined_solver)
    fxr = FXRates({"eurusd": 1.10})
    fxr._set_ad_order(2)
    result = pf.gamma(solver=combined_solver, fx=fxr, base="eur")
    expected = DataFrame(
        data=[
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.13769, 0.28088, 0.0],
            [0.0, 0.0, 0.28088, 0.44493, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.28930, -0.45081, 0.0, 0.0, -0.68937],
            [-0.45081, -0.47449, 0.0, 0.0, -1.37372],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.68937, -1.37372, 0.0, 0.0, 0.00064],
            [-0.31823, -0.49590, 0.0, 0.0, 0.0],
            [-0.49590, -0.52194, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.28930, -0.45081, 0.0, 0.0, -0.68937],
            [-0.45081, -0.47449, 0.0, 0.0, -1.37372],
            [0.0, 0.0, 0.13770, 0.28088, 0.0],
            [0.0, 0.0, 0.28088, 0.44493, 0.0],
            [-0.68937, -1.37372, 0.0, 0.0, 0.00064],
        ],
        index=MultiIndex.from_tuples(
            [
                ("eur", "eur", "instruments", "sofr", "4m"),
                ("eur", "eur", "instruments", "sofr", "8m"),
                ("eur", "eur", "instruments", "estr", "3m"),
                ("eur", "eur", "instruments", "estr", "9m"),
                ("eur", "eur", "fx", "fx", "eurusd"),
                ("usd", "eur", "instruments", "sofr", "4m"),
                ("usd", "eur", "instruments", "sofr", "8m"),
                ("usd", "eur", "instruments", "estr", "3m"),
                ("usd", "eur", "instruments", "estr", "9m"),
                ("usd", "eur", "fx", "fx", "eurusd"),
                ("usd", "usd", "instruments", "sofr", "4m"),
                ("usd", "usd", "instruments", "sofr", "8m"),
                ("usd", "usd", "instruments", "estr", "3m"),
                ("usd", "usd", "instruments", "estr", "9m"),
                ("usd", "usd", "fx", "fx", "eurusd"),
                ("all", "eur", "instruments", "sofr", "4m"),
                ("all", "eur", "instruments", "sofr", "8m"),
                ("all", "eur", "instruments", "estr", "3m"),
                ("all", "eur", "instruments", "estr", "9m"),
                ("all", "eur", "fx", "fx", "eurusd"),
            ],
            names=["local_ccy", "display_ccy", "type", "solver", "label"],
        ),
        columns=MultiIndex.from_tuples(
            [
                ("instruments", "sofr", "4m"),
                ("instruments", "sofr", "8m"),
                ("instruments", "estr", "3m"),
                ("instruments", "estr", "9m"),
                ("fx", "fx", "eurusd"),
            ],
            names=["type", "solver", "label"],
        ),
    )
    assert_frame_equal(result, expected, atol=1e-2, rtol=1e-4)


def test_solver_gamma_pnl_explain() -> None:
    instruments = [
        IRS(dt(2022, 1, 1), "10y", "A", currency="usd", curves="sofr"),
        IRS(dt(2032, 1, 1), "10y", "A", currency="usd", curves="sofr"),
        IRS(dt(2022, 1, 1), "10y", "A", currency="eur", curves="estr"),
        IRS(dt(2032, 1, 1), "10y", "A", currency="eur", curves="estr"),
        XCS(
            dt(2022, 1, 1),
            "10y",
            "A",
            currency="eur",
            pair="eurusd",
            curves=["estr", "eurusd", "sofr", "sofr"],
        ),
        XCS(
            dt(2032, 1, 1),
            "10y",
            "A",
            currency="usd",
            pair="usdeur",
            curves=["estr", "eurusd", "sofr", "sofr"],
        ),
    ]
    # s_base = np.array([3.45, 2.85, 2.25, 0.9, -15, -10])
    sofr = Curve(nodes={dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0}, id="sofr")
    estr = Curve(nodes={dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0}, id="estr")
    eurusd = Curve(
        nodes={dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0},
        id="eurusd",
    )
    fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
    fxf = FXForwards(fxr, {"eureur": estr, "eurusd": eurusd, "usdusd": sofr})
    sofr_solver = Solver(
        curves=[sofr],
        instruments=instruments[:2],
        s=[3.45, 2.85],
        instrument_labels=["10y", "10y10y"],
        id="sofr",
        fx=fxf,
    )
    estr_solver = Solver(
        curves=[estr],
        instruments=instruments[2:4],
        s=[2.25, 0.90],
        instrument_labels=["10y", "10y10y"],
        id="estr",
        fx=fxf,
    )
    solver = Solver(
        curves=[eurusd],
        instruments=instruments[4:],
        s=[-10, -15],
        instrument_labels=["10y", "10y10y"],
        id="xccy",
        fx=fxf,
        pre_solvers=[sofr_solver, estr_solver],
    )

    pf = Portfolio(
        [
            IRS(
                dt(2022, 1, 1),
                "20Y",
                "A",
                currency="eur",
                fixed_rate=2.0,
                notional=1e8,
                curves="estr",
            ),
        ],
    )

    npv_base = pf.npv(solver=solver, base="eur")
    expected_npv = -6230451.035973
    assert (npv_base - expected_npv) < 1e-5

    delta_base = pf.delta(solver=solver, base="usd")
    # this expectation is directly input from reviewed output.
    expected_delta = DataFrame(
        data=[
            [3.51021, 0.0, 3.51021],
            [-0.00005, 0.0, -0.00005],
            [101841.37433, 97001.98184, 101841.37433],
            [85750.45235, 81672.83139, 85750.45235],
            [-3.55593, 0.0, -3.55593],
            [0.00004, 0.0, 0.00004],
            [-623.00136, 0.0, -623.00136],
        ],
        index=MultiIndex.from_tuples(
            [
                ("instruments", "sofr", "10y"),
                ("instruments", "sofr", "10y10y"),
                ("instruments", "estr", "10y"),
                ("instruments", "estr", "10y10y"),
                ("instruments", "xccy", "10y"),
                ("instruments", "xccy", "10y10y"),
                ("fx", "fx", "eurusd"),
            ],
            names=["type", "solver", "label"],
        ),
        columns=MultiIndex.from_tuples(
            [("all", "usd"), ("eur", "eur"), ("eur", "usd")],
            names=["local_ccy", "display_ccy"],
        ),
    )
    assert_frame_equal(delta_base, expected_delta, atol=1e-2, rtol=1e-4)

    gamma_base = pf.gamma(solver=solver, base="eur")
    expected_gamma = DataFrame(
        data=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -102.972447, -81.00807888, 0.0, 0.0, 0.0],
            [0.0, 0.0, -81.00807888, -87.84105303, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        index=MultiIndex.from_tuples(
            [
                ("eur", "eur", "instruments", "sofr", "10y"),
                ("eur", "eur", "instruments", "sofr", "10y10y"),
                ("eur", "eur", "instruments", "estr", "10y"),
                ("eur", "eur", "instruments", "estr", "10y10y"),
                ("eur", "eur", "instruments", "xccy", "10y"),
                ("eur", "eur", "instruments", "xccy", "10y10y"),
                ("eur", "eur", "fx", "fx", "eurusd"),
            ],
            names=["local_ccy", "display_ccy", "type", "solver", "label"],
        ),
        columns=MultiIndex.from_tuples(
            [
                ("instruments", "sofr", "10y"),
                ("instruments", "sofr", "10y10y"),
                ("instruments", "estr", "10y"),
                ("instruments", "estr", "10y10y"),
                ("instruments", "xccy", "10y"),
                ("instruments", "xccy", "10y10y"),
                ("fx", "fx", "eurusd"),
            ],
            names=["type", "solver", "label"],
        ),
    )
    with warnings.catch_warnings():
        # TODO: pandas 3.0.0 can optionally turn off these PerformanceWarnings
        warnings.simplefilter(action="ignore", category=PerformanceWarning)
        assert_frame_equal(
            gamma_base.loc[("all", "eur")], expected_gamma.loc[("eur", "eur")], atol=1e-2, rtol=1e-4
        )


def test_gamma_with_fxrates_ad_order_1_raises() -> None:
    # when calculating gamma, AD order 2 is needed, the fx rates object passed
    # must also be converted. TODO
    pass


def test_error_labels() -> None:
    solver_with_error = Solver(
        curves=[
            Curve(
                nodes={dt(2022, 1, 1): 1.0, dt(2022, 7, 1): 1.0, dt(2023, 1, 1): 1.0},
                id="curve1",
            ),
        ],
        instruments=[
            IRS(dt(2022, 1, 1), "1M", "A", curves="curve1"),
            IRS(dt(2022, 1, 1), "2M", "A", curves="curve1"),
            IRS(dt(2022, 1, 1), "3M", "A", curves="curve1"),
            IRS(dt(2022, 1, 1), "4M", "A", curves="curve1"),
            IRS(dt(2022, 1, 1), "8M", "A", curves="curve1"),
            IRS(dt(2022, 1, 1), "12M", "A", curves="curve1"),
        ],
        s=[2.0, 2.2, 2.3, 2.4, 2.45, 2.55],
        id="rates",
    )
    result = solver_with_error.error
    assert abs(result.loc[("rates", "rates0")] - 22.798) < 1e-2


def test_solver_non_unique_id_raises() -> None:
    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="A")
    solver = Solver(
        curves=[curve],
        instruments=[IRS(dt(2022, 1, 1), "1Y", "Q", curves=curve)],
        s=[1],
        id="bad",
    )
    curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="B")
    with pytest.raises(ValueError, match="Solver `id`s must be unique"):
        Solver(
            curves=[curve2],
            instruments=[IRS(dt(2022, 1, 1), "1Y", "Q", curves=curve2)],
            s=[1],
            id="bad",
            pre_solvers=[solver],
        )


def test_solving_indirect_parameters_from_proxy_composite() -> None:
    eureur = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="eureur")
    eurspd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.999}, id="eurspd")
    eur3m = CompositeCurve([eureur, eurspd], id="eur3m")
    usdusd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="usdusd")
    eurusd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="eurusd")
    fxr = FXRates({"eurusd": 1.1}, settlement=dt(2022, 1, 3))
    fxf = FXForwards(
        fx_rates=fxr,
        fx_curves={
            "eureur": eureur,
            "usdusd": usdusd,
            "eurusd": eurusd,
        },
    )
    usdeur = fxf.curve("usd", "eur", id="usdeur")
    instruments = [
        IRS(dt(2022, 1, 1), "1Y", "A", currency="eur", curves=["eur3m", "eureur"]),
        IRS(dt(2022, 1, 1), "1Y", "A", currency="usd", curves="usdusd"),
        XCS(
            dt(2022, 1, 1),
            "1Y",
            "A",
            currency="eur",
            pair="eurusd",
            curves=["eureur", "eureur", "usdusd", "usdeur"],
        ),
    ]
    Solver(
        curves=[eureur, eur3m, usdusd, eurusd, usdeur],
        instruments=instruments,
        s=[2.0, 2.7, -15],
        fx=fxf,
    )


def test_solver_dimensions_of_matmul() -> None:
    swaps = [
        IRS(dt(2023, 7, 21), "9m", "A", fixed_rate=2.0, curves="chf", currency="chf"),
        IRS(dt(2023, 7, 21), "9m", "A", fixed_rate=2.0, curves="gbp", currency="gbp"),
        IRS(dt(2023, 7, 21), "9m", "A", fixed_rate=2.0, curves="usd", currency="usd"),
    ]
    chf_inst = [
        IRS(dt(2023, 7, 21), "6m", "A", curves="chf", currency="chf"),
        IRS(dt(2023, 7, 21), "1y", "A", curves="chf", currency="chf"),
    ]
    gbp_inst = [
        IRS(dt(2023, 7, 21), "6m", "A", curves="gbp", currency="gbp"),
        IRS(dt(2023, 7, 21), "1y", "A", curves="gbp", currency="gbp"),
    ]
    usd_inst = [
        IRS(dt(2023, 7, 21), "6m", "A", curves="usd", currency="usd"),
        IRS(dt(2023, 7, 21), "1y", "A", curves="usd", currency="usd"),
    ]
    usd = Curve(
        {dt(2023, 7, 21): 1.0, dt(2024, 1, 21): 1.0, dt(2024, 7, 21): 1.0},
        id="usd",
    )
    gbp = Curve(
        {dt(2023, 7, 21): 1.0, dt(2024, 1, 21): 1.0, dt(2024, 7, 21): 1.0},
        id="gbp",
    )
    chf = Curve(
        {dt(2023, 7, 21): 1.0, dt(2024, 1, 21): 1.0, dt(2024, 7, 21): 1.0},
        id="chf",
    )
    fxr = FXRates({"gbpusd": 1.25, "chfgbp": 1.1})
    solver1 = Solver(curves=[chf], instruments=chf_inst, s=[1.5, 1.8], id="CHF")
    solver2 = Solver(
        curves=[gbp],
        instruments=gbp_inst,
        s=[1.6, 1.7],
        id="GBP",
        pre_solvers=[solver1],
    )
    solver3 = Solver(
        curves=[usd],
        instruments=usd_inst,
        s=[1.7, 1.9],
        id="USD",
        pre_solvers=[solver2],
    )
    pf = Portfolio(swaps)
    pf.delta(solver=solver3, base="gbp", fx=fxr)
    pf.gamma(solver=solver3, base="gbp", fx=fxr)


def test_pre_solver_single_fx_object() -> None:
    # this test considers building up FXForwards using chined solvers.
    uu = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="uu")
    ee = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="ee")
    gg = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="gg")
    eu = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="eu")
    gu = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="gu")

    fxf1 = FXForwards(
        fx_rates=FXRates({"eurusd": 1.0}, settlement=dt(2022, 1, 1)),
        fx_curves={
            "usdusd": uu,
            "eureur": ee,
            "eurusd": eu,
        },
    )

    fxf2 = FXForwards(
        fx_rates=FXRates({"eurusd": 1.0, "gbpusd": 1.5}, settlement=dt(2022, 1, 1)),
        fx_curves={
            "usdusd": uu,
            "eureur": ee,
            "gbpgbp": gg,
            "eurusd": eu,
            "gbpusd": gu,
        },
    )

    s1 = Solver(
        curves=[uu, ee, gg],
        instruments=[
            IRS(dt(2022, 1, 1), "1y", "A", curves="uu"),
            IRS(dt(2022, 1, 1), "1y", "A", curves="ee"),
            IRS(dt(2022, 1, 1), "1y", "A", curves="gg"),
        ],
        s=[1.5, 1.5, 1.0],
        id="local",
    )
    s2 = Solver(
        curves=[eu],
        instruments=[
            XCS(
                dt(2022, 1, 1),
                "1Y",
                "Q",
                currency="eur",
                pair="eurusd",
                curves=["ee", "eu", "uu", "uu"],
            ),
        ],
        s=[10.0],
        id="x1",
        fx=fxf1,
        pre_solvers=[s1],
    )
    Solver(
        curves=[gu],
        instruments=[
            XCS(
                dt(2022, 1, 1),
                "1Y",
                "Q",
                currency="gbp",
                pair="gbpusd",
                curves=["gg", "gu", "uu", "uu"],
            ),
        ],
        s=[20.0],
        id="x2",
        fx=fxf2,
        pre_solvers=[s2],
    )
    result = gu[dt(2023, 1, 1)]
    expected = 0.988
    assert (result - expected) < 1e-4


def test_pre_solver_set_ad_order() -> None:
    curve1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99})
    curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99})
    curve3 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99})
    cc = CompositeCurve([curve2, curve3])
    s1 = Solver(curves=[curve1], instruments=[Value(dt(2022, 5, 1), curves=curve1)], s=[0.99])
    s2 = Solver(curves=[curve2], instruments=[Value(dt(2022, 5, 1), curves=curve1)], s=[0.99])
    s3 = Solver(
        pre_solvers=[s1, s2],
        curves=[cc, curve3],
        instruments=[Value(dt(2022, 5, 1), curves=curve1)],
        s=[0.99],
    )
    s3._set_ad_order(2)
    for c in [curve1, curve2, curve3, cc]:
        assert c._ad == 2
    assert s2._ad == 2
    assert s1._ad == 2

    s3._set_ad_order(1)
    for c in [curve1, curve2, curve3, cc]:
        assert c._ad == 1
    assert s2._ad == 1
    assert s1._ad == 1


def test_solver_jacobians_in_text() -> None:
    par_curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
            dt(2027, 1, 1): 1.0,
            dt(2032, 1, 1): 1.0,
        },
        id="curve",
    )
    par_instruments = [
        IRS(dt(2022, 1, 1), "1Y", "A", curves="curve"),
        IRS(dt(2022, 1, 1), "2Y", "A", curves="curve"),
        IRS(dt(2022, 1, 1), "5Y", "A", curves="curve"),
        IRS(dt(2022, 1, 1), "10Y", "A", curves="curve"),
    ]
    par_solver = Solver(
        curves=[par_curve],
        instruments=par_instruments,
        s=[1.21, 1.635, 1.885, 1.93],
        id="par_solver",
        instrument_labels=["1Y", "2Y", "5Y", "10Y"],
    )
    fwd_curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
            dt(2027, 1, 1): 1.0,
            dt(2032, 1, 1): 1.0,
        },
        id="curve",
    )
    fwd_instruments = [
        IRS(dt(2022, 1, 1), "1Y", "A", curves="curve"),
        IRS(dt(2023, 1, 1), "1Y", "A", curves="curve"),
        IRS(dt(2024, 1, 1), "3Y", "A", curves="curve"),
        IRS(dt(2027, 1, 1), "5Y", "A", curves="curve"),
    ]
    s_fwd = [float(_.rate(solver=par_solver)) for _ in fwd_instruments]
    fwd_solver = Solver(
        curves=[fwd_curve],
        instruments=fwd_instruments,
        s=s_fwd,
        id="fwd_solver",
        instrument_labels=["1Y", "1Y1Y", "2Y3Y", "5Y5Y"],
    )
    S_BA = par_solver.jacobian(fwd_solver).to_numpy()
    S_AB = fwd_solver.jacobian(par_solver).to_numpy()
    assert np.all(np.isclose(np.eye(4), np.matmul(S_AB, S_BA)))


def test_solver_jacobians_pre() -> None:
    par_curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
        },
        id="curve",
    )
    par_instruments = [
        IRS(dt(2022, 1, 1), "1Y", "A", curves="curve"),
        IRS(dt(2022, 1, 1), "2Y", "A", curves="curve"),
    ]
    par_solver = Solver(
        curves=[par_curve],
        instruments=par_instruments,
        s=[1.21, 1.635],
        id="par_solver",
        instrument_labels=["1Y", "2Y"],
    )
    par_curve2 = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
        },
        id="curve2",
    )
    par_instruments2 = [
        IRS(dt(2022, 1, 1), "1Y", "A", curves="curve2"),
        IRS(dt(2022, 1, 1), "2Y", "A", curves="curve2"),
    ]
    par_solver2 = Solver(
        curves=[par_curve2],
        instruments=par_instruments2,
        s=[1.21, 1.635],
        id="par_solver2",
        instrument_labels=["1Y", "2Y"],
        pre_solvers=[par_solver],
    )

    fwd_curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
        },
        id="curve",
    )
    fwd_instruments = [
        IRS(dt(2022, 1, 1), "1Y", "A", curves="curve"),
        IRS(dt(2023, 1, 1), "1Y", "A", curves="curve"),
    ]
    s_fwd = [float(_.rate(solver=par_solver2)) for _ in fwd_instruments]
    fwd_solver = Solver(
        curves=[fwd_curve],
        instruments=fwd_instruments,
        s=s_fwd,
        id="fwd_solver",
        instrument_labels=["1Y", "1Y1Y"],
    )
    fwd_curve2 = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
        },
        id="curve2",
    )
    fwd_instruments2 = [
        IRS(dt(2022, 1, 1), "1Y", "A", curves="curve2"),
        IRS(dt(2023, 1, 1), "1Y", "A", curves="curve2"),
    ]
    s_fwd2 = [float(_.rate(solver=par_solver2)) for _ in fwd_instruments2]
    fwd_solver2 = Solver(
        curves=[fwd_curve2],
        instruments=fwd_instruments2,
        s=s_fwd2,
        id="fwd_solver2",
        instrument_labels=["1Y", "1Y1Y"],
        pre_solvers=[fwd_solver],
    )

    S_BA = par_solver2.jacobian(fwd_solver2)
    S_AB = fwd_solver2.jacobian(par_solver2)
    assert np.all(np.isclose(np.eye(4), np.matmul(S_AB.to_numpy(), S_BA.to_numpy())))


def test_newton_solver_1dim_dual() -> None:
    def root(x, s):
        return x**2 - s, 2 * x

    x0 = Dual(1.0, ["x"], [])
    s = Dual(2.0, ["s"], [])
    result = newton_1dim(root, x0, args=(s,))

    expected = 0.5 / 2.0**0.5
    sensitivity = gradient(result["g"], ["s"])[0]
    assert abs(expected - sensitivity) < 1e-9


def test_newton_solver_1dim_dual2() -> None:
    def root(x, s):
        return x**2 - s, 2 * x

    x0 = Dual2(1.0, ["x"], [], [])
    s = Dual2(2.0, ["s"], [], [])
    result = newton_1dim(root, x0, args=(s,))

    expected = 0.5 / 2.0**0.5
    sensitivity = gradient(result["g"], ["s"])[0]
    assert abs(expected - sensitivity) < 1e-9

    expected = -0.25 * (1 / 2.0**1.5)
    sensitivity = gradient(result["g"], ["s"], order=2)[0, 0]
    assert abs(expected - sensitivity) < 1e-9


def test_newton_solver_2dim_dual() -> None:
    def root(g, s):
        f0 = g[0] ** 2 + g[1] ** 2 + s
        f1 = g[0] ** 2 - 2 * g[1] ** 2 - s

        f00 = 2 * g[0]
        f01 = 2 * g[1]
        f10 = 2 * g[0]
        f11 = -4 * g[1]

        return [f0, f1], [[f00, f01], [f10, f11]]

    g0 = [Dual(1.0, ["x"], []), Dual(2.0, ["y"], [])]
    s = Dual(-2.0, ["s"], [])
    result = newton_ndim(root, g0, args=(s,))

    expected_x = (2 / 3) ** 0.5
    assert abs(result["g"][0] - expected_x) < 1e-9

    expected_y = (4 / 3) ** 0.5
    assert abs(result["g"][1] - expected_y) < 1e-9

    expected_y = -0.5 * (2 / 3) ** 0.5 * (2.0) ** -0.5
    expected_x = -0.5 * (1 / 3.0) ** 0.5 * (2.0) ** -0.5

    sensitivity_x = gradient(result["g"][0], ["s"])[0]
    sensitivity_y = gradient(result["g"][1], ["s"])[0]
    assert abs(expected_x - sensitivity_x) < 1e-9
    assert abs(expected_y - sensitivity_y) < 1e-9


def test_newton_solver_2dim_dual2() -> None:
    def root(g, s):
        f0 = g[0] ** 2 + g[1] ** 2 + s
        f1 = g[0] ** 2 - 2 * g[1] ** 2 - s

        f00 = 2 * g[0]
        f01 = 2 * g[1]
        f10 = 2 * g[0]
        f11 = -4 * g[1]

        return [f0, f1], [[f00, f01], [f10, f11]]

    g0 = [Dual2(1.0, ["x"], [], []), Dual2(2.0, ["y"], [], [])]
    s = Dual2(-2.0, ["s"], [], [])
    result = newton_ndim(root, g0, args=(s,))

    expected_x = (2 / 3) ** 0.5
    assert abs(result["g"][0] - expected_x) < 1e-9
    expected_y = (4 / 3) ** 0.5
    assert abs(result["g"][1] - expected_y) < 1e-9

    expected_y = -0.5 * (2 / 3) ** 0.5 * (2.0) ** -0.5
    expected_x = -0.5 * (1 / 3.0) ** 0.5 * (2.0) ** -0.5
    sensitivity_x = gradient(result["g"][0], ["s"])[0]
    sensitivity_y = gradient(result["g"][1], ["s"])[0]
    assert abs(expected_x - sensitivity_x) < 1e-9
    assert abs(expected_y - sensitivity_y) < 1e-9

    expected_y2 = -0.25 * (2 / 3) ** 0.5 * (2.0) ** -1.5
    expected_x2 = -0.25 * (1 / 3) ** 0.5 * (2.0) ** -1.5
    sensitivity_x2 = gradient(result["g"][0], ["s"], order=2)[0, 0]
    sensitivity_y2 = gradient(result["g"][1], ["s"], order=2)[0, 0]
    assert abs(expected_x2 - sensitivity_x2) < 1e-9
    assert abs(expected_y2 - sensitivity_y2) < 1e-9


def test_newton_1d_failed_state() -> None:
    def root(g):
        f0 = g**2 + 10.0
        f1 = 2 * g
        return f0, f1

    result = newton_1dim(root, 1.5, max_iter=5, raise_on_fail=False)
    assert result["state"] == -1


def test_newton_ndim_raises() -> None:
    def root(g):
        f0_0 = g[0] ** 2 + 10.0
        f0_1 = g[0] + g[1] ** 2 - 2.0
        return [f0_0, f0_1], [[2 * g[0], 0.0], [1.0, 2 * g[1]]]

    with pytest.raises(ValueError, match="`max_iter`: 5 exceeded in 'newton_ndim'"):
        newton_ndim(root, [0.5, 1.0], max_iter=5)


def test_newton_solver_object_args():
    def root(x, s):
        return x**2 - s["some_obj"], 2 * x

    x0 = Dual(1.0, ["x"], [])
    s = {"some_obj": Dual(2.0, ["s"], [])}
    result = newton_1dim(root, x0, args=(s,))

    expected = 0.5 / 2.0**0.5
    sensitivity = gradient(result["g"], ["s"])[0]
    assert abs(expected - sensitivity) < 1e-9


def test_solver_with_vol_smile() -> None:
    eureur = Curve(
        {dt(2023, 3, 16): 1.0, dt(2023, 9, 16): 0.9851909811629752},
        calendar="tgt",
        id="eureur",
    )
    usdusd = Curve(
        {dt(2023, 3, 16): 1.0, dt(2023, 9, 16): 0.976009366603271},
        calendar="nyc",
        id="usdusd",
    )
    # eurusd = Curve({dt(2023, 3, 16): 1.0, dt(2023, 9, 16): 0.987092591908283}, id="eurusd")
    fxr = FXRates({"eurusd": 1.3088}, settlement=dt(2023, 3, 20))
    fxf = FXForwards(fx_curves={"eureur": eureur, "eurusd": eureur, "usdusd": usdusd}, fx_rates=fxr)
    fxf._set_ad_order(1)
    solver = Solver(
        curves=[eureur, usdusd],
        instruments=[
            IRS(dt(2023, 3, 20), "1m", curves=[eureur], spec="eur_irs"),
            IRS(dt(2023, 3, 20), "1m", curves=[usdusd], spec="usd_irs"),
        ],
        s=[2.0113, 0.3525],
        fx=fxf,
    )
    eurusd_1m_smile = FXDeltaVolSmile(
        nodes={
            0.25: 10.0,
            0.50: 10.0,
            0.75: 10.0,
        },
        eval_date=dt(2023, 3, 16),
        expiry=dt(2023, 4, 18),
        delta_type="spot",
        id="smile",
    )
    args = {
        "pair": "eurusd",
        "expiry": dt(2023, 4, 18),
        "curves": ["eureur", "usdusd"],
        "delta_type": "spot",
        "vol": "smile",
    }
    Solver(
        pre_solvers=[solver],
        curves=[eurusd_1m_smile],
        instruments=[
            FXStraddle(strike="atm_delta", **args),
            FXRiskReversal(strike=["-25d", "25d"], **args),
            FXStrangle(strike=["-25d", "25d"], **args),
        ],
        s=[21.6215, -0.5, 22.359],
        fx=fxf,
    )


def test_solver_with_surface() -> None:
    eureur = Curve({dt(2024, 5, 7): 1.0, dt(2025, 5, 30): 1.0}, calendar="tgt", id="eureur")
    eurusd = Curve({dt(2024, 5, 7): 1.0, dt(2025, 5, 30): 1.0}, id="eurusd")
    usdusd = Curve({dt(2024, 5, 7): 1.0, dt(2025, 5, 30): 1.0}, calendar="nyc", id="usdusd")
    # Create an FX Forward market with spot FX rate data
    fxf = FXForwards(
        fx_rates=FXRates({"eurusd": 1.0760}, settlement=dt(2024, 5, 9)),
        fx_curves={"eureur": eureur, "usdusd": usdusd, "eurusd": eurusd},
    )
    solver = Solver(
        curves=[eureur, eurusd, usdusd],
        instruments=[
            IRS(dt(2024, 5, 9), "3W", spec="eur_irs", curves="eureur"),
            IRS(dt(2024, 5, 9), "3W", spec="usd_irs", curves="usdusd"),
            FXSwap(
                dt(2024, 5, 9),
                "3W",
                pair="eurusd",
                curves=["eurusd", "usdusd"],
            ),
        ],
        s=[3.90, 5.32, 8.85],
        instrument_labels=["3w EU", "3w US", "3w FXSw"],
        fx=fxf,
    )
    surface = FXDeltaVolSurface(
        eval_date=dt(2024, 5, 7),
        expiries=[dt(2024, 5, 28), dt(2024, 6, 7)],
        delta_indexes=[0.1, 0.25, 0.5, 0.75, 0.9],
        delta_type="forward",
        node_values=np.ones(shape=(2, 5)) * 5.0,
        id="eurusd_vol",
    )
    data = DataFrame(
        data=[
            [5.493, -0.157, 0.071, -0.289, 0.238],
            [5.525, -0.213, 0.075, -0.400, 0.250],
        ],
        columns=["ATM", "25dRR", "25dBF", "10dRR", "25dBF"],
        index=[dt(2024, 5, 28), dt(2024, 6, 7)],
    )
    fx_args = dict(
        pair="eurusd",
        delta_type="spot",
        calendar="tgt",
        curves=["eurusd", "usdusd"],
        vol="eurusd_vol",
    )
    instruments, s, labels = [], [], []
    for e, row in enumerate(data.itertuples()):
        instruments.extend(
            [
                FXStraddle(strike="atm_delta", expiry=row[0], **fx_args),
                FXRiskReversal(strike=("-25d", "25d"), expiry=row[0], **fx_args),
                FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), expiry=row[0], **fx_args),
                FXRiskReversal(strike=("-10d", "10d"), expiry=row[0], **fx_args),
                FXBrokerFly(strike=(("-10d", "10d"), "atm_delta"), expiry=row[0], **fx_args),
            ],
        )
        s.extend([row[1], row[2], row[3], row[4], row[5]])
        labels.extend([f"atm{e}", f"25rr{e}", f"25bf{e}", f"10rr{e}", f"10bf{e}"])
    surf_solver = Solver(
        surfaces=[surface],
        instruments=instruments,
        s=s,
        pre_solvers=[solver],
        instrument_labels=labels,
        fx=fxf,
    )
    fxc = FXCall(expiry=dt(2024, 6, 7), strike=1.08, **fx_args)
    fxc.analytic_greeks(solver=surf_solver)
    fxc.delta(solver=surf_solver)
    fxc.gamma(solver=surf_solver)


class TestStateManagement:
    def test_solver_state_storage(self):
        # test the solver stores hashes of its objects: FXForwards, Curves and presolvers
        uu = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="uu")
        ee = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="ee")
        eu = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="eu")

        fxf1 = FXForwards(
            fx_rates=FXRates({"eurusd": 1.0}, settlement=dt(2022, 1, 1)),
            fx_curves={
                "usdusd": uu,
                "eureur": ee,
                "eurusd": eu,
            },
        )

        s1 = Solver(
            curves=[uu, ee],
            instruments=[
                IRS(dt(2022, 1, 1), "1y", "A", curves="uu"),
                IRS(dt(2022, 1, 1), "1y", "A", curves="ee"),
            ],
            s=[1.5, 1.5],
            id="local",
        )
        s2 = Solver(
            curves=[eu],
            instruments=[
                XCS(
                    dt(2022, 1, 1),
                    "1Y",
                    "Q",
                    currency="eur",
                    pair="eurusd",
                    curves=["ee", "eu", "uu", "uu"],
                ),
            ],
            s=[10.0],
            id="x1",
            fx=fxf1,
            pre_solvers=[s1],
        )
        hashes = {"fx": s2.fx._state, **{k: curve._state for k, curve in s2.pre_curves.items()}}
        assert s2._states == hashes

    @pytest.mark.parametrize(
        "method",
        [
            "delta",
            "gamma",
            "npv",
            "rate",
        ],
    )
    @pytest.mark.parametrize(
        ("obj", "args"), [("fxr", ({"eurusd": 1.0},)), ("fxf", ([{"eurusd": 1.10}],))]
    )
    def test_warning_on_fx_mutation(self, method, obj, args):
        # test the solver stores hashes of its objects: FXForwards, Curves and presolvers
        uu = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="uu")
        ee = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="ee")
        eu = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="eu")

        fxr = FXRates({"eurusd": 1.0}, settlement=dt(2022, 1, 1))
        fxf = FXForwards(fx_rates=fxr, fx_curves={"usdusd": uu, "eureur": ee, "eurusd": eu})

        s1 = Solver(
            curves=[uu, ee],
            instruments=[
                IRS(dt(2022, 1, 1), "1y", "A", curves="uu"),
                IRS(dt(2022, 1, 1), "1y", "A", curves="ee"),
            ],
            s=[1.5, 1.5],
            id="local",
        )
        s2 = Solver(
            curves=[eu],
            instruments=[
                XCS(
                    dt(2022, 1, 1),
                    "1Y",
                    "Q",
                    currency="eur",
                    pair="eurusd",
                    curves=["ee", "eu", "uu", "uu"],
                ),
            ],
            s=[10.0],
            id="x1",
            fx=fxf,
            pre_solvers=[s1],
        )

        vars()[obj].update(*args)
        irs = IRS(dt(2022, 1, 1), "3y", "A", curves="uu")
        with pytest.warns(UserWarning, match="The `fx` object associated with `solver`"):
            getattr(irs, method)(solver=s2)

    @pytest.mark.parametrize(
        "method",
        [
            "delta",
            "gamma",
            "npv",
            "rate",
        ],
    )
    def test_raise_on_pre_curve_mutation(self, method):
        # test the solver stores hashes of its objects: FXForwards, Curves and presolvers
        uu = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="uu")
        ee = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="ee")
        eu = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="eu")

        fxf1 = FXForwards(
            fx_rates=FXRates({"eurusd": 1.0}, settlement=dt(2022, 1, 1)),
            fx_curves={
                "usdusd": uu,
                "eureur": ee,
                "eurusd": eu,
            },
        )

        s1 = Solver(
            curves=[uu, ee],
            instruments=[
                IRS(dt(2022, 1, 1), "1y", "A", curves="uu"),
                IRS(dt(2022, 1, 1), "1y", "A", curves="ee"),
            ],
            s=[1.5, 1.5],
            id="local",
        )
        s2 = Solver(
            curves=[eu],
            instruments=[
                XCS(
                    dt(2022, 1, 1),
                    "1Y",
                    "Q",
                    currency="eur",
                    pair="eurusd",
                    curves=["ee", "eu", "uu", "uu"],
                ),
            ],
            s=[10.0],
            id="x1",
            fx=fxf1,
            pre_solvers=[s1],
        )

        uu._set_node_vector([0.995], 1)
        irs = IRS(dt(2022, 1, 1), "3y", "A", curves="uu")
        with pytest.raises(ValueError, match="The `curves` associated with `solver` have been upd"):
            getattr(irs, method)(solver=s2)

    @pytest.mark.parametrize(
        "method",
        [
            "delta",
            "gamma",
            "npv",
            "rate",
        ],
    )
    def test_raise_on_curve_mutation(self, method):
        # test the solver stores hashes of its objects: FXForwards, Curves and presolvers
        uu = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="uu")
        ee = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="ee")
        eu = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="eu")

        fxf1 = FXForwards(
            fx_rates=FXRates({"eurusd": 1.0}, settlement=dt(2022, 1, 1)),
            fx_curves={
                "usdusd": uu,
                "eureur": ee,
                "eurusd": eu,
            },
        )

        s1 = Solver(
            curves=[uu, ee],
            instruments=[
                IRS(dt(2022, 1, 1), "1y", "A", curves="uu"),
                IRS(dt(2022, 1, 1), "1y", "A", curves="ee"),
            ],
            s=[1.5, 1.5],
            id="local",
        )
        s2 = Solver(
            curves=[eu],
            instruments=[
                XCS(
                    dt(2022, 1, 1),
                    "1Y",
                    "Q",
                    currency="eur",
                    pair="eurusd",
                    curves=["ee", "eu", "uu", "uu"],
                ),
            ],
            s=[10.0],
            id="x1",
            fx=fxf1,
            pre_solvers=[s1],
        )

        eu._set_node_vector([0.995], 1)
        irs = IRS(dt(2022, 1, 1), "3y", "A", curves="uu")
        with pytest.raises(ValueError, match="The `curves` associated with `solver` have been up"):
            getattr(irs, method)(solver=s2)

    @pytest.mark.parametrize(
        "method",
        [
            "delta",
            "gamma",
            "npv",
            "rate",
        ],
    )
    def test_raise_on_composite_curve_mutation(self, method):
        # test the solver stores hashes of its objects: FXForwards, Curves and presolvers
        uu = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="uu")
        ee = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="ee")
        cc = CompositeCurve([uu, ee], id="cc")

        s1 = Solver(
            curves=[ee, cc],
            instruments=[
                IRS(dt(2022, 1, 1), "1y", "A", curves="cc"),
            ],
            s=[1.5],
            id="local",
        )

        uu.update_node(dt(2023, 1, 1), 0.98)
        irs = IRS(dt(2022, 1, 1), "3y", "A", curves="cc")
        with pytest.raises(ValueError, match="The `curves` associated with `solver` have been up"):
            getattr(irs, method)(solver=s1)

    def test_solver_auto_updates_fx_before_state_setting(self):
        # added `self.fx._set_ad_order(1)` to Solver.__init__
        with warnings.catch_warnings():
            warnings.simplefilter(action="error", category=UserWarning)
            smile = FXDeltaVolSmile(
                nodes={
                    0.10: 10.0,
                    0.25: 10.0,
                    0.50: 10.0,
                    0.75: 10.0,
                    0.90: 10.0,
                },
                eval_date=dt(2024, 5, 7),
                expiry=dt(2024, 5, 28),
                delta_type="spot",
                id="eurusd_3w_smile",
            )
            # Define the interest rate curves for EUR, USD and X-Ccy basis
            eureur = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="tgt", id="eureur")
            eurusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, id="eurusd")
            usdusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="nyc", id="usdusd")
            # Create an FX Forward market with spot FX rate data
            fxf = FXForwards(
                fx_rates=FXRates({"eurusd": 1.0760}, settlement=dt(2024, 5, 9)),
                fx_curves={"eureur": eureur, "usdusd": usdusd, "eurusd": eurusd},
            )
            # Setup the Solver instrument calibration for rates Curves and vol Smiles
            option_args = dict(
                pair="eurusd",
                expiry=dt(2024, 5, 28),
                calendar="tgt",
                delta_type="spot",
                curves=["eurusd", "usdusd"],
                vol="eurusd_3w_smile",
            )
            Solver(
                curves=[eureur, eurusd, usdusd, smile],
                instruments=[
                    IRS(dt(2024, 5, 9), "3W", spec="eur_irs", curves="eureur"),
                    IRS(dt(2024, 5, 9), "3W", spec="usd_irs", curves="usdusd"),
                    FXSwap(
                        dt(2024, 5, 9), "3W", pair="eurusd", curves=[None, "eurusd", None, "usdusd"]
                    ),
                    FXStraddle(strike="atm_delta", **option_args),
                    FXRiskReversal(strike=("-25d", "25d"), **option_args),
                    FXRiskReversal(strike=("-10d", "10d"), **option_args),
                    FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), **option_args),
                    FXBrokerFly(strike=(("-10d", "10d"), "atm_delta"), **option_args),
                ],
                s=[3.90, 5.32, 8.85, 5.493, -0.157, -0.289, 0.071, 0.238],
                fx=fxf,
            )

    def test_solver_dual2_auto_updates_fx_before_state_setting(self):
        with warnings.catch_warnings():
            warnings.simplefilter(action="error", category=UserWarning)
            # tests the doc page j_gamma.rst
            sofr = Curve(
                nodes={dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0}, id="sofr"
            )
            estr = Curve(
                nodes={dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0}, id="estr"
            )
            eurusd = Curve(
                nodes={dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0}, id="eurusd"
            )
            fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
            fxf = FXForwards(fxr, {"eureur": estr, "eurusd": eurusd, "usdusd": sofr})
            instruments = [
                IRS(dt(2022, 1, 1), "10y", "A", currency="usd", curves="sofr"),
                IRS(dt(2032, 1, 1), "10y", "A", currency="usd", curves="sofr"),
                IRS(dt(2022, 1, 1), "10y", "A", currency="eur", curves="estr"),
                IRS(dt(2032, 1, 1), "10y", "A", currency="eur", curves="estr"),
                XCS(
                    dt(2022, 1, 1),
                    "10y",
                    "A",
                    currency="usd",
                    pair="eurusd",
                    curves=["estr", "eurusd", "sofr", "sofr"],
                ),
                XCS(
                    dt(2032, 1, 1),
                    "10y",
                    "A",
                    currency="usd",
                    pair="eurusd",
                    curves=["estr", "eurusd", "sofr", "sofr"],
                ),
            ]
            sofr_solver = Solver(
                curves=[sofr],
                instruments=instruments[:2],
                s=[3.45, 2.85],
                instrument_labels=["10y", "10y10y"],
                id="sofr",
                fx=fxf,
            )
            estr_solver = Solver(
                curves=[estr],
                instruments=instruments[2:4],
                s=[2.25, 0.90],
                instrument_labels=["10y", "10y10y"],
                id="estr",
                fx=fxf,
            )
            solver = Solver(
                curves=[eurusd],
                instruments=instruments[4:],
                s=[-10, -15],
                instrument_labels=["10y", "10y10y"],
                id="eurusd",
                fx=fxf,
                pre_solvers=[sofr_solver, estr_solver],
            )
            pf = Portfolio(
                [
                    IRS(
                        dt(2022, 1, 1),
                        "20Y",
                        "A",
                        currency="eur",
                        fixed_rate=2.0,
                        notional=1e8,
                        curves="estr",
                    ),
                    IRS(
                        dt(2022, 1, 1),
                        "20Y",
                        "A",
                        currency="usd",
                        fixed_rate=1.5,
                        notional=-1.1e8,
                        curves="sofr",
                    ),
                ]
            )
            pf.gamma(solver=solver, base="eur")

    def test_pre_solvers_fx_is_updated_and_does_not_cause_validation_issue(self):
        with warnings.catch_warnings():
            warnings.simplefilter(action="error", category=UserWarning)
            # tests the doc page j_gamma.rst
            sofr = Curve(
                nodes={dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0}, id="sofr"
            )
            estr = Curve(
                nodes={dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0}, id="estr"
            )
            eurusd = Curve(
                nodes={dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0}, id="eurusd"
            )
            fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
            fxf = FXForwards(fxr, {"eureur": estr, "eurusd": eurusd, "usdusd": sofr})
            instruments = [
                IRS(dt(2022, 1, 1), "10y", "A", currency="usd", curves="sofr"),
                IRS(dt(2032, 1, 1), "10y", "A", currency="usd", curves="sofr"),
                IRS(dt(2022, 1, 1), "10y", "A", currency="eur", curves="estr"),
                IRS(dt(2032, 1, 1), "10y", "A", currency="eur", curves="estr"),
                XCS(
                    dt(2022, 1, 1),
                    "10y",
                    "A",
                    currency="eur",
                    pair="eurusd",
                    curves=["estr", "eurusd", "sofr", "sofr"],
                ),
                XCS(
                    dt(2032, 1, 1),
                    "10y",
                    "A",
                    currency="usd",
                    pair="eurusd",
                    curves=["estr", "eurusd", "sofr", "sofr"],
                ),
            ]
            solver1 = Solver(
                curves=[sofr, estr],
                instruments=instruments[:4],
                s=[3.45, 2.85, 2.4, 1.7],
                instrument_labels=["10y", "10y10y", "10ye", "10y10ye"],
                id="sofr/estr",
                fx=fxf,
            )
            # solver 2 will solve the FX basis and update the FXForwards object which is also
            # associated with solver1. If solver1 is state validated it will then fail.
            # except when the _update_fx method of solver2 also nests calls to pre_solvers
            _solver2 = Solver(
                curves=[eurusd],
                instruments=instruments[4:],
                s=[-10, -15],
                instrument_labels=["10y", "10y10y"],
                id="eurusd",
                fx=fxf,
                pre_solvers=[solver1],
            )
            irs = IRS(
                dt(2022, 1, 1),
                "20Y",
                "A",
                currency="eur",
                fixed_rate=2.0,
                notional=1e8,
                curves="estr",
            )
            irs.gamma(solver=solver1, base="eur")

    @pytest.mark.parametrize(
        "obj",
        [
            Curve({dt(2000, 1, 1): 1.0, dt(2000, 3, 2): 0.99}),
            LineCurve({dt(2000, 1, 1): 1.0, dt(2000, 3, 2): 0.99}),
            FXDeltaVolSmile(
                nodes={0.5: 10.0},
                expiry=dt(2000, 1, 1),
                eval_date=dt(1999, 1, 1),
                delta_type="forward",
            ),
            FXRates({"eurusd": 1.0}),
            FXForwards(
                FXRates({"eurusd": 1.0}, settlement=dt(2000, 1, 3)),
                {
                    "eurusd": Curve({dt(2000, 1, 1): 1.0, dt(2000, 3, 2): 0.99}),
                    "eureur": Curve({dt(2000, 1, 1): 1.0, dt(2000, 3, 2): 0.99}),
                    "usdusd": Curve({dt(2000, 1, 1): 1.0, dt(2000, 3, 2): 0.99}),
                },
            ),
            CompositeCurve(
                [
                    Curve({dt(2000, 1, 1): 1.0, dt(2000, 3, 2): 0.99}),
                    Curve({dt(2000, 1, 1): 1.0, dt(2000, 3, 2): 0.99}),
                ]
            ),
            MultiCsaCurve(
                [
                    Curve({dt(2000, 1, 1): 1.0, dt(2000, 3, 2): 0.99}),
                    Curve({dt(2000, 1, 1): 1.0, dt(2000, 3, 2): 0.99}),
                ]
            ),
            FXDeltaVolSurface(
                delta_type="forward",
                delta_indexes=[0.5],
                expiries=[dt(2000, 1, 8), dt(2001, 1, 1)],
                eval_date=dt(1999, 1, 1),
                node_values=[[10], [11]],
            ),
            Solver(
                curves=[Curve({dt(2000, 1, 1): 1.0, dt(2000, 3, 2): 0.99}, id="abc")],
                instruments=[IRS(dt(2000, 1, 1), "1m", spec="usd_irs", curves="abc")],
                s=[2.0],
                fx=FXForwards(
                    FXRates({"eurusd": 1.0}, settlement=dt(2000, 1, 3)),
                    {
                        "eurusd": Curve({dt(2000, 1, 1): 1.0, dt(2000, 3, 2): 0.99}),
                        "eureur": Curve({dt(2000, 1, 1): 1.0, dt(2000, 3, 2): 0.99}),
                        "usdusd": Curve({dt(2000, 1, 1): 1.0, dt(2000, 3, 2): 0.99}),
                    },
                ),
            ),
        ],
    )
    def test_set_ad_order_does_not_change_object_state(self, obj):
        pre_state = obj._state
        obj._set_ad_order(2)
        post_state = obj._state
        assert pre_state == post_state

    def test_solver_validation_control(self):
        curve = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 1.0})
        solver = Solver(
            curves=[curve],
            instruments=[IRS(dt(2000, 1, 1), "1m", spec="usd_irs", curves=curve)],
            s=[2.0],
        )
        curve.update_node(dt(2001, 1, 1), 0.99)
        irs = IRS(dt(2000, 1, 1), "2m", spec="usd_irs", curves=curve)
        with pytest.raises(ValueError, match="The `curves` associated with `solver` have"):
            irs.rate(solver=solver)

        solver._do_not_validate = True
        result = irs.rate(solver=solver)
        assert abs(result - 0.989345) < 1e-5


@pytest.mark.parametrize(
    "obj",
    [
        Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 1.0}),
        LineCurve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 2.0}),
        FXRates({"eurusd": 1.0}),
        FXForwards(
            fx_curves={
                "eureur": Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 1.0}),
                "eurusd": Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 1.0}),
                "usdusd": Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 1.0}),
            },
            fx_rates=FXRates({"eurusd": 1.0}, settlement=dt(2000, 1, 3)),
        ),
        FXSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            id="vol",
        ),
        FXSabrSurface(
            eval_date=dt(2024, 5, 28),
            expiries=[dt(2025, 2, 2), dt(2025, 3, 3)],
            node_values=[[0.05, 1.0, 0.01, 0.15]] * 2,
            pair="eurusd",
            delivery_lag=2,
            calendar="tgt|fed",
            id="eurusd_vol",
        ),
        FXDeltaVolSurface(
            delta_indexes=[0.25, 0.5, 0.75],
            expiries=[dt(2024, 1, 1), dt(2025, 1, 1)],
            node_values=[[11, 10, 12], [8, 7, 9]],
            eval_date=dt(2023, 1, 1),
            delta_type="forward",
            id="vol",
        ),
        FXDeltaVolSmile(
            nodes={
                0.25: 10.15,
                0.5: 7.8,
                0.75: 8.9,
            },
            delta_type="forward",
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            id="vol",
            ad=1,
        ),
        MultiCsaCurve(
            [
                Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 1.0}),
                Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 1.0}),
            ]
        ),
        CompositeCurve(
            [
                Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 1.0}),
                Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 1.0}),
            ]
        ),
    ],
)
def test_objects_ad_attribute(obj):
    result = getattr(obj, "_ad", None)
    assert result is not None


@pytest.mark.parametrize("label", ["shift", "rolled", "translated"])
def test_curves_without_their_own_params(label):
    curve = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 1.0}, id="curve")
    _map = {
        "shift": curve.shift(5, id="shift"),
        "rolled": curve.roll(5, id="rolled"),
        "translated": curve.translate(dt(2000, 1, 1), id="translated"),
    }

    sv = Solver(
        curves=[curve, _map[label]],
        instruments=[IRS(dt(2000, 2, 1), dt(2000, 3, 1), spec="usd_irs", curves=["curve", label])],
        s=[2.0],
    )
    assert sv.result["status"] == "SUCCESS"
