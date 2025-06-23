from datetime import datetime as dt
from math import exp, log

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pandas import DataFrame, MultiIndex
from pandas.testing import assert_frame_equal, assert_series_equal
from rateslib import default_context
from rateslib.curves import CompositeCurve, Curve, LineCurve, index_left
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, gradient
from rateslib.fx import FXForwards, FXRates
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface
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
from rateslib.solver import Gradients, Solver, newton_1dim, newton_ndim


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
                [Inst(Dual2(1.0, ["v1"], [1.0], [4.0])), tuple(), {}],
                [
                    Inst(
                        Dual2(
                            3.0,
                            ["v1", "v2", "v3"],
                            [2.0, 1.0, -2.0],
                            [-2.0, 1.0, 1.0, 1.0, -3.0, 2.0, 1.0, 2.0, -4.0],
                        )
                    ),
                    tuple(),
                    {},
                ],
            ]
            _J2 = None
            _ad = 2
            _grad_s_vT = np.array(
                [
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                ]
            )

        setattr(cls, "solver", SolverProxy())

    def test_J(self):
        expected = np.array(
            [
                [1.0, 2.0],
                [0.0, 1.0],
                [0.0, -2.0],
            ]
        )
        result = self.solver.J
        assert_allclose(result, expected)

    def test_grad_v_rT(self):
        assert_allclose(self.solver.J, self.solver.grad_v_rT)

    def test_J2(self):
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
            ]
        )
        expected = np.transpose(expected, (1, 2, 0))
        result = self.solver.J2
        assert_allclose(expected, result)

    def test_grad_v_v_rT(self):
        assert_allclose(self.solver.J2, self.solver.grad_v_v_rT)

    def test_grad_s_vT(self):
        expected = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
            ]
        )
        result = self.solver.grad_s_vT
        assert_allclose(expected, result)


@pytest.mark.parametrize("algo", ["gauss_newton", "levenberg_marquardt", "gradient_descent"])
def test_basic_solver(algo):
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
        (IRS(dt(2022, 1, 1), "1Y", "Q"), (curve,), {}),
        (IRS(dt(2022, 1, 1), "2Y", "Q"), (curve,), {}),
        (IRS(dt(2022, 1, 1), "3Y", "Q"), (curve,), {}),
    ]
    s = np.array([1.0, 1.6, 2.0])
    solver = Solver(
        curves=[curve],
        instruments=instruments,
        s=s,
        algorithm=algo,
    )
    assert float(solver.g) < 1e-9
    assert curve.nodes[dt(2022, 1, 1)] == Dual(1.0, ["v0"], [1])
    expected = [1, 0.9899250357528555, 0.9680433953206192, 0.9407188354823821]
    for i, key in enumerate(curve.nodes.keys()):
        assert abs(float(curve.nodes[key]) - expected[i]) < 1e-6


@pytest.mark.parametrize("algo", ["gauss_newton", "levenberg_marquardt", "gradient_descent"])
def test_solver_reiterate(algo):
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
    assert curve.nodes[dt(2022, 1, 1)] == Dual(1.0, ["v0"], [1])
    expected = [1, 0.9899250357528555, 0.9680433953206192, 0.9407188354823821]
    for i, key in enumerate(curve.nodes.keys()):
        assert abs(float(curve.nodes[key]) - expected[i]) < 1e-6


@pytest.mark.parametrize("algo", ["gauss_newton", "levenberg_marquardt", "gradient_descent"])
def test_basic_solver_line_curve(algo):
    curve = LineCurve(
        {
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
        },
        id="v",
    )
    instruments = [
        (Value(dt(2022, 1, 1)), (curve,), {}),
        (Value(dt(2023, 1, 1)), (curve,), {}),
        (Value(dt(2024, 1, 1)), (curve,), {}),
    ]
    s = np.array([3.0, 3.6, 4.0])
    solver = Solver(
        curves=[curve],
        instruments=instruments,
        s=s,
        algorithm=algo,
    )
    assert float(solver.g) < 1e-9
    for i, key in enumerate(curve.nodes.keys()):
        assert abs(float(curve.nodes[key]) - s[i]) < 1e-5


def test_basic_spline_solver():
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
        (IRS(dt(2022, 1, 1), "1Y", "Q"), (spline_curve,), {}),
        (IRS(dt(2022, 1, 1), "2Y", "Q"), (spline_curve,), {}),
        (IRS(dt(2022, 1, 1), "3Y", "Q"), (spline_curve,), {}),
    ]
    s = np.array([1.0, 1.6, 2.0])
    solver = Solver(
        curves=[spline_curve],
        instruments=instruments,
        s=s,
        algorithm="gauss_newton",
    )
    assert float(solver.g) < 1e-12
    assert spline_curve.nodes[dt(2022, 1, 1)] == Dual(1.0, ["v0"], [1])
    expected = [1, 0.98992503575307, 0.9680377261843034, 0.9407048036486593]
    for i, key in enumerate(spline_curve.nodes.keys()):
        assert abs(float(spline_curve.nodes[key]) - expected[i]) < 1e-11


def test_large_spline_solver():
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
        nodes={_: 1.0 for _ in dates},
        t=[dt(2000, 1, 3)] * 3 + dates[:-1] + [dt(2050, 1, 5)] * 4,
        calendar="nyc",
    )
    solver = Solver(
        curves=[curve],
        instruments=[IRS(dt(2000, 1, 3), _, spec="usd_irs", curves=curve) for _ in dates[1:]],
        s=[1.0 + _ / 25 for _ in range(18)],
    )
    assert solver.result["status"] == "SUCCESS"


def test_solver_raises_len():
    with pytest.raises(ValueError, match="`instrument_rates` must be same length"):
        Solver(
            instruments=[1],
            s=[1, 2],
        )

    with pytest.raises(ValueError, match="`instrument_labels` must have length"):
        Solver(
            instruments=[1],
            s=[1],
            instrument_labels=[1, 2],
        )

    with pytest.raises(ValueError, match="`weights` must be same length"):
        Solver(
            instruments=[1, 2],
            s=[1, 2],
            instrument_labels=[1, 2],
            weights=[1],
        )


def test_basic_solver_weights():
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
        (IRS(dt(2022, 1, 1), "1Y", "Q"), (curve,), {}),
        (IRS(dt(2022, 1, 1), "2Y", "Q"), (curve,), {}),
        (IRS(dt(2022, 1, 1), "3Y", "Q"), (curve,), {}),
        (IRS(dt(2022, 1, 1), "3Y", "Q"), (curve,), {}),
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
    assert curve.nodes[dt(2022, 1, 1)] == Dual(1.0, ["v0"], [1])
    expected = [1, 0.9899250357528555, 0.9680433953206192, 0.9407188354823821]
    for i, key in enumerate(curve.nodes.keys()):
        assert abs(float(curve.nodes[key]) - expected[i]) < 1e-6

    solver = Solver(
        curves=[curve],
        instruments=instruments,
        s=s,
        weights=[1, 1, 1, 1e-6],
        func_tol=1e-7,
        algorithm="gauss_newton",
    )
    assert abs(float(instruments[2][0].rate(curve)) - 2.02) < 1e-4

    solver = Solver(
        curves=[curve],
        instruments=instruments,
        s=s,
        weights=[1, 1, 1e-6, 1],
        func_tol=1e-7,
        algorithm="gauss_newton",
    )
    assert abs(float(instruments[2][0].rate(curve)) - 1.98) < 1e-4


def test_solver_independent_curve():
    # Test that a solver can use an independent curve as a static object and solve
    # without mutating that un-referenced object.
    independent_curve = Curve(
        {
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.98,
            dt(2024, 1, 1): 0.96,
            dt(2025, 1, 1): 0.94,
        }
    )
    expected = independent_curve.copy()
    var_curve = Curve(
        {
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 0.99,
            dt(2024, 1, 1): 0.98,
            dt(2025, 1, 1): 0.97,
        }
    )
    instruments = [
        (IRS(dt(2022, 1, 1), "1Y", "Q"), ([var_curve, independent_curve],), {}),
        (IRS(dt(2022, 1, 1), "2Y", "Q"), ([var_curve, independent_curve],), {}),
        (IRS(dt(2022, 1, 1), "3Y", "Q"), ([var_curve, independent_curve],), {}),
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
        assert abs(float(instrument[0].rate(*instrument[1], **instrument[2]) - s[i])) < 1e-7
    assert independent_curve == expected


class TestSolverCompositeCurve:
    def test_solver_composite_curve(self):
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
                -0.226074787,
                0.2257131776,
                0.0003616069,
                -9.159037835,
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


def test_non_unique_curves():
    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="A")
    curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="A")
    solver = Solver(
        curves=[curve], instruments=[(IRS(dt(2022, 1, 1), "1Y", "Q"), (curve,), {})], s=[1]
    )

    with pytest.raises(ValueError, match="`curves` must each have their own unique"):
        Solver(
            curves=[curve2],
            instruments=[(IRS(dt(2022, 1, 1), "1Y", "Q"), (curve,), {})],
            s=[2],
            pre_solvers=[solver],
        )

    with pytest.raises(ValueError, match="`curves` must each have their own unique"):
        Solver(
            curves=[curve, curve2],
            instruments=[(IRS(dt(2022, 1, 1), "1Y", "Q"), (curve,), {})],
            s=[2],
        )


def test_max_iterations():
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
        (IRS(dt(2022, 1, 1), "1Y", "Q"), (curve,), {}),
        (IRS(dt(2022, 1, 1), "2Y", "Q"), (curve,), {}),
        (IRS(dt(2022, 1, 1), "3Y", "Q"), (curve,), {}),
        (IRS(dt(2022, 1, 1), "3Y", "Q"), (curve,), {}),
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


def test_solver_pre_solver_dependency_generates_same_delta():
    """
    Build an ESTR curve with solver1.
    Build an IBOR curve with solver2 dependent upon solver1.

    Build an ESTR and IBOR curve simultaneously inside the same solver3.

    Test the delta and the instrument calibration error
    """
    eur_disc_curve = Curve(
        nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0, dt(2024, 1, 1): 1.0}, id="eur"
    )
    eur_instruments = [
        (IRS(dt(2022, 1, 1), "8M", "A"), (eur_disc_curve,), {}),
        (IRS(dt(2022, 1, 1), "16M", "A"), (eur_disc_curve,), {}),
        (IRS(dt(2022, 1, 1), "2Y", "A"), (eur_disc_curve,), {}),
    ]
    eur_disc_s = [2.01, 2.22, 2.55]
    eur_disc_solver = Solver([eur_disc_curve], [], eur_instruments, eur_disc_s, id="estr")

    eur_ibor_curve = Curve(
        nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0, dt(2024, 1, 1): 1.0}, id="eur_ibor"
    )
    eur_ibor_instruments = [
        (IRS(dt(2022, 1, 1), "1Y", "A"), ([eur_ibor_curve, eur_disc_curve],), {}),
        (IRS(dt(2022, 1, 1), "2Y", "A"), ([eur_ibor_curve, eur_disc_curve],), {}),
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
        {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0, dt(2024, 1, 1): 1.0}, id="eur"
    )
    eur_ibor_curve2 = Curve(
        {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0, dt(2024, 1, 1): 1.0}, id="eur_ibor"
    )
    eur_instruments2 = [
        (IRS(dt(2022, 1, 1), "8M", "A"), (eur_disc_curve2,), {}),
        (IRS(dt(2022, 1, 1), "16M", "A"), (eur_disc_curve2,), {}),
        (IRS(dt(2022, 1, 1), "2Y", "A"), (eur_disc_curve2,), {}),
        (IRS(dt(2022, 1, 1), "1Y", "A"), ([eur_ibor_curve2, eur_disc_curve2],), {}),
        (IRS(dt(2022, 1, 1), "2Y", "A"), ([eur_ibor_curve2, eur_disc_curve2],), {}),
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

    delta_sim = eur_swap.delta([eur_ibor_curve2, eur_disc_curve2], eur_solver_sim)
    delta_pre = eur_swap.delta([eur_ibor_curve, eur_disc_curve], eur_solver2)
    delta_pre.index = delta_sim.index
    assert_frame_equal(delta_sim, delta_pre)

    error_sim = eur_solver_sim.error
    error_pre = eur_solver2.error
    assert_series_equal(error_pre, error_sim, check_index=False, rtol=1e-5, atol=1e-3)


def test_delta_gamma_calculation():
    estr_curve = Curve(
        {dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0}, id="estr_curve"
    )
    estr_instruments = [
        (IRS(dt(2022, 1, 1), "10Y", "A"), (estr_curve,), {}),
        (IRS(dt(2022, 1, 1), "20Y", "A"), (estr_curve,), {}),
    ]
    estr_solver = Solver(
        [estr_curve], [], estr_instruments, [2.0, 1.5], id="estr", instrument_labels=["10Y", "20Y"]
    )

    # Mechanism 1: dynamic
    eur_swap = IRS(dt(2032, 1, 1), "10Y", "A", notional=100e6)
    assert 74430 < float(eur_swap.delta(estr_curve, estr_solver).sum().iloc[0]) < 74432
    assert -229 < float(eur_swap.gamma(estr_curve, estr_solver).sum().sum()) < -228

    # Mechanism 1: dynamic names
    assert 74430 < float(eur_swap.delta("estr_curve", estr_solver).sum().iloc[0]) < 74432
    assert -229 < float(eur_swap.gamma("estr_curve", estr_solver).sum().sum()) < -228

    # Mechanism 1: fails on None curve specification
    with pytest.raises(TypeError, match="`curves` have not been supplied correctly"):
        assert eur_swap.delta(NoInput(0), estr_solver)
    with pytest.raises(TypeError, match="`curves` have not been supplied correctly"):
        assert eur_swap.gamma(NoInput(0), estr_solver)

    # Mechanism 2: static specific
    eur_swap = IRS(dt(2032, 1, 1), "10Y", "A", notional=100e6, curves=estr_curve)
    assert 74430 < float(eur_swap.delta(NoInput(0), estr_solver).sum().iloc[0]) < 74432
    assert -229 < float(eur_swap.gamma(NoInput(0), estr_solver).sum().sum()) < -228

    # Mechanism 2: static named
    eur_swap = IRS(dt(2032, 1, 1), "10Y", "A", notional=100e6, curves="estr_curve")
    assert 74430 < float(eur_swap.delta(NoInput(0), estr_solver).sum().iloc[0]) < 74432
    assert -229 < float(eur_swap.gamma(NoInput(0), estr_solver).sum().sum()) < -228


def test_solver_delta_fx_noinput():
    estr_curve = Curve(
        {dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0}, id="estr_curve"
    )
    estr_instruments = [
        (IRS(dt(2022, 1, 1), "10Y", "A"), (estr_curve,), {}),
        (IRS(dt(2022, 1, 1), "20Y", "A"), (estr_curve,), {}),
    ]
    estr_solver = Solver(
        [estr_curve], [], estr_instruments, [2.0, 1.5], id="estr", instrument_labels=["10Y", "20Y"]
    )
    eur_swap = IRS(dt(2032, 1, 1), "10Y", "A", notional=100e6, fixed_rate=2)
    npv = eur_swap.npv(curves=estr_curve, solver=estr_solver, local=True)
    result = estr_solver.delta(npv)
    assert type(result) is DataFrame


def test_solver_pre_solver_dependency_generates_same_gamma():
    estr_curve = Curve({dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0})
    estr_instruments = [
        (IRS(dt(2022, 1, 1), "7Y", "A"), (estr_curve,), {}),
        (IRS(dt(2022, 1, 1), "15Y", "A"), (estr_curve,), {}),
        (IRS(dt(2022, 1, 1), "20Y", "A"), (estr_curve,), {}),
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
        (IRS(dt(2022, 1, 1), "10Y", "A"), ([ibor_curve, estr_curve],), {}),
        (IRS(dt(2022, 1, 1), "20Y", "A"), ([ibor_curve, estr_curve],), {}),
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
    gamma_pre = eur_swap.gamma([ibor_curve, estr_curve], ibor_solver)
    delta_pre = eur_swap.delta([ibor_curve, estr_curve], ibor_solver)

    estr_curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0})
    ibor_curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0})
    sim_instruments = [
        (IRS(dt(2022, 1, 1), "7Y", "A"), (estr_curve2,), {}),
        (IRS(dt(2022, 1, 1), "15Y", "A"), (estr_curve2,), {}),
        (IRS(dt(2022, 1, 1), "20Y", "A"), (estr_curve2,), {}),
        (IRS(dt(2022, 1, 1), "10Y", "A"), ([ibor_curve2, estr_curve2],), {}),
        (IRS(dt(2022, 1, 1), "20Y", "A"), ([ibor_curve2, estr_curve2],), {}),
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
    gamma_sim = eur_swap.gamma([ibor_curve2, estr_curve2], simultaneous_solver)
    delta_sim = eur_swap.delta([ibor_curve2, estr_curve2], simultaneous_solver)

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


def test_nonmutable_presolver_defaults():
    estr_curve = Curve({dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0})
    estr_instruments = [
        (IRS(dt(2022, 1, 1), "10Y", "A"), (estr_curve,), {}),
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


def test_solver_grad_s_vT_methods_equivalent():
    curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
            dt(2025, 1, 1): 1.0,
            dt(2026, 1, 1): 1.0,
            dt(2027, 1, 1): 1.0,
        }
    )
    instruments = [
        (IRS(dt(2022, 1, 1), "2Y", "A"), (curve,), {}),
        (IRS(dt(2023, 1, 1), "1Y", "A"), (curve,), {}),
        (IRS(dt(2023, 1, 1), "2Y", "A"), (curve,), {}),
        (IRS(dt(2022, 5, 1), "4Y", "A"), (curve,), {}),
        (IRS(dt(2023, 1, 1), "4Y", "A"), (curve,), {}),
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


def test_solver_grad_s_vT_methods_equivalent_overspecified_curve():
    curve = Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2023, 1, 1): 1.0,
            dt(2024, 1, 1): 1.0,
            dt(2025, 1, 1): 1.0,
            # dt(2026, 1, 1): 1.0,
            dt(2027, 1, 1): 1.0,
        }
    )
    instruments = [
        (IRS(dt(2022, 1, 1), "2Y", "A"), (curve,), {}),
        (IRS(dt(2023, 1, 1), "1Y", "A"), (curve,), {}),
        (IRS(dt(2023, 1, 1), "2Y", "A"), (curve,), {}),
        (IRS(dt(2022, 5, 1), "4Y", "A"), (curve,), {}),
        (IRS(dt(2023, 1, 1), "4Y", "A"), (curve,), {}),
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


def test_solver_second_order_vars_raise_on_first_order():
    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="A")
    solver = Solver(
        curves=[curve], instruments=[(IRS(dt(2022, 1, 1), "1Y", "Q"), (curve,), {})], s=[1]
    )

    with pytest.raises(ValueError, match="Cannot perform second derivative calc"):
        solver.J2

    with pytest.raises(ValueError, match="Cannot perform second derivative calc"):
        solver.grad_s_s_vT


def test_solver_second_order_vars_raise_on_first_order_pre_solvers():
    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="A")
    solver = Solver(
        curves=[curve], instruments=[(IRS(dt(2022, 1, 1), "1Y", "Q"), (curve,), {})], s=[1]
    )
    curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="B")
    solver2 = Solver(
        curves=[curve2],
        instruments=[(IRS(dt(2022, 1, 1), "1Y", "Q"), (curve2,), {})],
        s=[1],
        pre_solvers=[solver],
    )

    with pytest.raises(ValueError, match="Cannot perform second derivative calc"):
        solver2.J2_pre

    with pytest.raises(ValueError, match="Cannot perform second derivative calc"):
        solver.grad_s_s_vT_pre


def test_bad_algo_raises():
    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="A")
    with pytest.raises(NotImplementedError, match="`algorithm`: bad_algo"):
        Solver(
            curves=[curve],
            instruments=[(IRS(dt(2022, 1, 1), "1Y", "Q"), (curve,), {})],
            s=[1],
            algorithm="bad_algo",
        )


def test_solver_float_rate_bond():
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
    f_c.id = "rfr"
    instruments = [
        (
            FloatRateNote(
                dt(2022, 1, 1), "6M", "Q", spread_compound_method="isda_compounding", settle=2
            ),
            ([f_c, d_c],),
            {"metric": "spread"},
        ),
        (
            FloatRateNote(
                dt(2022, 1, 1), "1y", "Q", spread_compound_method="isda_compounding", settle=2
            ),
            ([f_c, d_c],),
            {"metric": "spread"},
        ),
        (
            FloatRateNote(
                dt(2022, 1, 1), "18m", "Q", spread_compound_method="isda_compounding", settle=2
            ),
            ([f_c, d_c],),
            {"metric": "spread"},
        ),
    ]
    Solver([d_c], [], instruments, [25, 25, 25])
    result = d_c.rate(dt(2022, 7, 1), "1D")
    expected = f_c.rate(dt(2022, 7, 1), "1D") + 0.25
    assert abs(result - expected) < 3e-4


def test_solver_grad_s_s_vt_methods_equivalent():
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


def test_gamma_raises():
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
        (IRS(dt(2022, 1, 1), "1Y", "Q"), (curve,), {}),
        (IRS(dt(2022, 1, 1), "2Y", "Q"), (curve,), {}),
        (IRS(dt(2022, 1, 1), "3Y", "Q"), (curve,), {}),
    ]
    s = np.array([1.0, 1.6, 2.0])
    solver = Solver(
        curves=[curve],
        instruments=instruments,
        s=s,
    )
    with pytest.raises(ValueError, match="`Solver` must be in ad order 2"):
        solver.gamma(100)


def test_delta_irs_guide():
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
    result = irs.delta(solver=usd_solver, base="eur", local=True)  # local overrides base to USD
    expected = DataFrame(
        [[0], [16.77263], [32.60487]],
        index=MultiIndex.from_product(
            [["instruments"], ["usd_sofr"], ["1m", "3m", "1y"]], names=["type", "solver", "label"]
        ),
        columns=MultiIndex.from_tuples([("usd", "usd")], names=["local_ccy", "display_ccy"]),
    )
    assert_frame_equal(result, expected)


def test_delta_irs_guide_fx_base():
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


def test_mechanisms_guide_gamma():
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
        curves=[ll_curve], instruments=instruments, s=s, instrument_labels=["4m", "8m"], id="sofr"
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


def test_solver_gamma_pnl_explain():
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
            leg2_currency="usd",
            curves=["estr", "eurusd", "sofr", "sofr"],
        ),
        XCS(
            dt(2032, 1, 1),
            "10y",
            "A",
            currency="usd",
            leg2_currency="eur",
            curves=["estr", "eurusd", "sofr", "sofr"],
        ),
    ]
    # s_base = np.array([3.45, 2.85, 2.25, 0.9, -15, -10])
    sofr = Curve(nodes={dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0}, id="sofr")
    estr = Curve(nodes={dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0}, id="estr")
    eurusd = Curve(
        nodes={dt(2022, 1, 1): 1.0, dt(2032, 1, 1): 1.0, dt(2042, 1, 1): 1.0}, id="eurusd"
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
        ]
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
            [("all", "usd"), ("eur", "eur"), ("eur", "usd")], names=["local_ccy", "display_ccy"]
        ),
    )
    assert_frame_equal(delta_base, expected_delta, atol=1e-2, rtol=1e-4)

    gamma_base = pf.gamma(solver=solver, base="usd", local=True)  # local overrrides base to EUR
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
    assert_frame_equal(gamma_base, expected_gamma, atol=1e-2, rtol=1e-4)


def test_gamma_with_fxrates_ad_order_1_raises():
    # when calculating gamma, AD order 2 is needed, the fx rates object passed
    # must also be converted. TODO
    pass


def test_error_labels():
    solver_with_error = Solver(
        curves=[
            Curve(
                nodes={dt(2022, 1, 1): 1.0, dt(2022, 7, 1): 1.0, dt(2023, 1, 1): 1.0}, id="curve1"
            )
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


def test_solver_non_unique_id_raises():
    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="A")
    solver = Solver(
        curves=[curve],
        instruments=[(IRS(dt(2022, 1, 1), "1Y", "Q"), (curve,), {})],
        s=[1],
        id="bad",
    )
    curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="B")
    with pytest.raises(ValueError, match="Solver `id`s must be unique"):
        Solver(
            curves=[curve2],
            instruments=[(IRS(dt(2022, 1, 1), "1Y", "Q"), (curve2,), {})],
            s=[1],
            id="bad",
            pre_solvers=[solver],
        )


def test_solving_indirect_parameters_from_proxy_composite():
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
            leg2_currency="usd",
            curves=["eureur", "eureur", "usdusd", "usdeur"],
        ),
    ]
    Solver(
        curves=[eureur, eur3m, usdusd, eurusd, usdeur],
        instruments=instruments,
        s=[2.0, 2.7, -15],
        fx=fxf,
    )


def test_solver_dimensions_of_matmul():
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
        curves=[gbp], instruments=gbp_inst, s=[1.6, 1.7], id="GBP", pre_solvers=[solver1]
    )
    solver3 = Solver(
        curves=[usd], instruments=usd_inst, s=[1.7, 1.9], id="USD", pre_solvers=[solver2]
    )
    pf = Portfolio(swaps)
    pf.delta(solver=solver3, base="gbp", fx=fxr)
    pf.gamma(solver=solver3, base="gbp", fx=fxr)


def test_pre_solver_single_fx_object():
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
                leg2_currency="usd",
                curves=["ee", "eu", "uu", "uu"],
            )
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
                leg2_currency="usd",
                curves=["gg", "gu", "uu", "uu"],
            )
        ],
        s=[20.0],
        id="x2",
        fx=fxf2,
        pre_solvers=[s2],
    )
    result = gu[dt(2023, 1, 1)]
    expected = 0.988
    assert (result - expected) < 1e-4


def test_solver_jacobians_in_text():
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


def test_solver_jacobians_pre():
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


def test_newton_solver_1dim_dual():
    def root(x, s):
        return x**2 - s, 2 * x

    x0 = Dual(1.0, ["x"], [])
    s = Dual(2.0, ["s"], [])
    result = newton_1dim(root, x0, args=(s,))

    expected = 0.5 / 2.0**0.5
    sensitivity = gradient(result["g"], ["s"])[0]
    assert abs(expected - sensitivity) < 1e-9


def test_newton_solver_1dim_dual2():
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


def test_newton_solver_2dim_dual():
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


def test_newton_solver_2dim_dual2():
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


def test_newton_1d_failed_state():
    def root(g):
        f0 = g**2 + 10.0
        f1 = 2 * g
        return f0, f1

    result = newton_1dim(root, 1.5, max_iter=5, raise_on_fail=False)
    assert result["state"] == -1


def test_newton_ndim_raises():
    def root(g):
        f0_0 = g[0] ** 2 + 10.0
        f0_1 = g[0] + g[1] ** 2 - 2.0
        return [f0_0, f0_1], [[2 * g[0], 0.0], [1.0, 2 * g[1]]]

    with pytest.raises(ValueError, match="`max_iter`: 5 exceeded in 'newton_ndim'"):
        newton_ndim(root, [0.5, 1.0], max_iter=5)


def test_solver_with_vol_smile():
    eureur = Curve(
        {dt(2023, 3, 16): 1.0, dt(2023, 9, 16): 0.9851909811629752}, calendar="tgt", id="eureur"
    )
    usdusd = Curve(
        {dt(2023, 3, 16): 1.0, dt(2023, 9, 16): 0.976009366603271}, calendar="nyc", id="usdusd"
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
        "curves": [None, "eureur", None, "usdusd"],
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


def test_solver_with_surface():
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
                currency="eur",
                leg2_currency="usd",
                curves=[None, "eurusd", None, "usdusd"],
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
        curves=[None, "eurusd", None, "usdusd"],
        vol="eurusd_vol",
    )
    instruments, s, labels = [], [], []
    for e, row in enumerate(data.itertuples()):
        instruments.extend(
            [
                FXStraddle(strike="atm_delta", expiry=row[0], **fx_args),
                FXRiskReversal(strike=["-25d", "25d"], expiry=row[0], **fx_args),
                FXBrokerFly(strike=["-25d", "atm_delta", "25d"], expiry=row[0], **fx_args),
                FXRiskReversal(strike=["-10d", "10d"], expiry=row[0], **fx_args),
                FXBrokerFly(strike=["-10d", "atm_delta", "10d"], expiry=row[0], **fx_args),
            ]
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
