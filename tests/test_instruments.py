import pytest
from datetime import datetime as dt
from pandas import DataFrame, date_range, Series, Index, MultiIndex
from pandas.testing import assert_frame_equal
import numpy as np

import context
from rateslib import defaults, default_context
from rateslib.instruments import (
    IRS,
    IIRS,
    forward_fx,
    SBS,
    FXSwap,
    NonMtmXCS,
    FixedRateBond,
    Bill,
    Value,
    ZCS,
    ZCIS,
    _get_curve_from_solver,
    BaseMixin,
    FloatRateBond,
    FRA,
    BondFuture,
    IndexFixedRateBond,
    NonMtmFixedFloatXCS,
    NonMtmFixedFixedXCS,
    XCS,
    FixedFloatXCS,
    FixedFixedXCS,
    FloatFixedXCS,
    Portfolio,
    Spread,
    Fly,
    _get_curves_and_fx_maybe_from_solver,
)
from rateslib.dual import Dual, Dual2
from rateslib.calendars import dcf
from rateslib.curves import Curve, IndexCurve, LineCurve
from rateslib.fx import FXRates, FXForwards
from rateslib.solver import Solver


@pytest.fixture()
def curve():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.99,
        dt(2022, 7, 1): 0.98,
        dt(2022, 10, 1): 0.97,
    }
    convention = "Act360"
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture()
def curve2():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.98,
        dt(2022, 7, 1): 0.97,
        dt(2022, 10, 1): 0.95,
    }
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture()
def usdusd():
    nodes = {dt(2022, 1, 1): 1.00, dt(2022, 4, 1): 0.99}
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture()
def eureur():
    nodes = {dt(2022, 1, 1): 1.00, dt(2022, 4, 1): 0.997}
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture()
def usdeur():
    nodes = {dt(2022, 1, 1): 1.00, dt(2022, 4, 1): 0.996}
    return Curve(nodes=nodes, interpolation="log_linear")


def test_get_curve_from_solver():
    from rateslib.solver import Solver

    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="tagged")
    inst = [(Value(dt(2023, 1, 1)), ("tagged",), {})]
    solver = Solver([curve], inst, [0.975])

    result = _get_curve_from_solver("tagged", solver)
    assert result == curve

    result = _get_curve_from_solver(curve, solver)
    assert result == curve

    no_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="not in solver")

    with default_context("curve_not_in_solver", "ignore"):
        result = _get_curve_from_solver(no_curve, solver)
        assert result == no_curve

    with pytest.warns():
        with default_context("curve_not_in_solver", "warn"):
            result = _get_curve_from_solver(no_curve, solver)
            assert result == no_curve

    with pytest.raises(ValueError, match="`curve` must be in `solver`"):
        with default_context("curve_not_in_solver", "raise"):
            result = _get_curve_from_solver(no_curve, solver)


@pytest.mark.parametrize("solver", [True, False])
@pytest.mark.parametrize("fxf", [True, False])
@pytest.mark.parametrize("fx", [None, 2.0])
@pytest.mark.parametrize("crv", [True, False])
def test_get_curves_and_fx_from_solver(usdusd, usdeur, eureur, solver, fxf, fx, crv):
    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="tagged")
    inst = [(Value(dt(2023, 1, 1)), ("tagged",), {})]
    fxfs = FXForwards(
        FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3)),
        {"usdusd": usdusd, "usdeur": usdeur, "eureur": eureur},
    )
    solver = (
        Solver([curve], inst, [0.975], fx=fxfs if fxf else None) if solver else None
    )
    curve = curve if crv else None

    crv_result, fx_result = _get_curves_and_fx_maybe_from_solver(
        None, solver, curve, fx
    )

    # check the fx results. If fx is specified directly it is returned
    # otherwsie it is returned from a solver object if it is available.
    if fx is not None:
        assert fx_result == 2.0
    elif solver is None:
        assert fx_result is None
    else:
        if fxf:
            assert fx_result == fxfs
        else:
            assert fx_result is None

    assert crv_result == (curve, curve, curve, curve)


def test_get_curves_and_fx_from_solver_raises():
    from rateslib.solver import Solver

    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="tagged")
    inst = [(Value(dt(2023, 1, 1)), ("tagged",), {})]
    solver = Solver([curve], inst, [0.975])

    with pytest.raises(ValueError, match="`curves` must contain Curve, not str, if"):
        _get_curves_and_fx_maybe_from_solver(None, None, "tagged", None)

    with pytest.raises(ValueError, match="`curves` must contain str curve `id` s"):
        _get_curves_and_fx_maybe_from_solver(None, solver, "bad_id", None)

    with pytest.raises(ValueError, match="Can only supply a maximum of 4 `curves`"):
        _get_curves_and_fx_maybe_from_solver(None, solver, ["tagged"] * 5, None)


@pytest.mark.parametrize("num", [1, 2, 3, 4])
def test_get_curves_from_solver_multiply(num):
    from rateslib.solver import Solver

    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="tagged")
    inst = [(Value(dt(2023, 1, 1)), ("tagged",), {})]
    solver = Solver([curve], inst, [0.975])
    result, _ = _get_curves_and_fx_maybe_from_solver(
        None, solver, ["tagged"] * num, None
    )
    assert result == (curve, curve, curve, curve)


def test_get_proxy_curve_from_solver(usdusd, usdeur, eureur):
    # TODO: check whether curves in fxf but not is solver should be allowed???
    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="tagged")
    inst = [(Value(dt(2023, 1, 1)), ("tagged",), {})]
    fxf = FXForwards(
        FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3)),
        {"usdusd": usdusd, "usdeur": usdeur, "eureur": eureur},
    )
    solver = Solver([curve], inst, [0.975], fx=fxf)
    curve = fxf.curve("eur", "usd")
    irs = IRS(dt(2022, 1, 1), "3m", "Q")

    # test the curve will return even though it is not included within the solver
    # because it is a proxy curve.
    irs.npv(curves=curve, solver=solver)


class TestNullPricing:
    # test instruments can be priced without defining a pricing parameter.

    @pytest.mark.parametrize(
        "inst",
        [
            IRS(dt(2022, 7, 1), "3M", "A", curves="eureur", notional=1e6),
            FRA(dt(2022, 7, 1), "3M", "A", curves="eureur", notional=1e6),
            SBS(
                dt(2022, 7, 1),
                "3M",
                "A",
                curves=["eureur", "eureur", "eurusd", "eureur"],
                notional=-1e6,
            ),
            ZCS(dt(2022, 7, 1), "3M", "A", curves="eureur", notional=1e6),
            IIRS(dt(2022, 7, 1), "3M", "A", curves=["eu_cpi", "eureur", "eureur", "eureur"], notional=1e6),
            IIRS(dt(2022, 7, 1), "3M", "A",
                 curves=["eu_cpi", "eureur", "eureur", "eureur"], notional=1e6, notional_exchange=True),
            # TODO add a null price test for ZCIS
            XCS(
                dt(2022, 7, 1),
                "3M",
                "A",
                currency="usd",
                leg2_currency="eur",
                curves=["usdusd", "usdusd", "eureur", "eurusd"],
                notional=1e6,
            ),
            NonMtmXCS(
                dt(2022, 7, 1),
                "3M",
                "A",
                currency="usd",
                leg2_currency="eur",
                curves=["usdusd", "usdusd", "eureur", "eurusd"],
                notional=1e6,
            ),
            NonMtmFixedFloatXCS(
                dt(2022, 7, 1),
                "3M",
                "A",
                currency="eur",
                leg2_currency="usd",
                curves=["eureur", "eureur", "usdusd", "usdusd"],
                notional=1e6,
            ),
            NonMtmFixedFixedXCS(
                dt(2022, 7, 1),
                "3M",
                "A",
                currency="eur",
                leg2_currency="usd",
                fixed_rate=1.2,
                curves=["eureur", "eureur", "usdusd", "usdusd"],
                notional=1e6,
            ),
            FixedFloatXCS(
                dt(2022, 7, 1),
                "3M",
                "A",
                currency="eur",
                leg2_currency="usd",
                curves=["eureur", "eureur", "usdusd", "usdusd"],
                notional=1e6,
            ),
            FixedFixedXCS(
                dt(2022, 7, 1),
                "3M",
                "A",
                currency="eur",
                leg2_currency="usd",
                leg2_fixed_rate=1.3,
                curves=["eureur", "eureur", "usdusd", "usdusd"],
                notional=1e6,
            ),
            FloatFixedXCS(
                dt(2022, 7, 1),
                "3M",
                "A",
                currency="usd",
                leg2_currency="eur",
                curves=["usdusd", "usdusd", "eureur", "eureur"],
                notional=-1e6,
            ),
            FXSwap(
                dt(2022, 7, 1),
                "3M",
                "A",
                currency="eur",
                leg2_currency="usd",
                curves=["eureur", "eureur", "usdusd", "usdusd"],
                notional=1e6,
            ),
        ],
    )
    def test_null_priced_delta(self, inst):
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="usdusd")
        c2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="eureur")
        c3 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.982}, id="eurusd")
        c4 = IndexCurve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.995}, id="eu_cpi", index_base=100.0)
        fxf = FXForwards(
            FXRates({"eurusd": 1.0}, settlement=dt(2022, 1, 1)),
            {"usdusd": c1, "eureur": c2, "eurusd": c3},
        )
        ins = [
            IRS(dt(2022, 1, 1), "1y", "A", curves="eureur"),
            IRS(dt(2022, 1, 1), "1y", "A", curves="usdusd"),
            IRS(dt(2022, 1, 1), "1y", "A", curves="eurusd"),
            ZCIS(dt(2022, 1, 1), "1y", "A", curves=["eureur", "eureur", "eu_cpi", "eureur"])
        ]
        solver = Solver(
            curves=[c1, c2, c3, c4],
            instruments=ins,
            s=[1.2, 1.3, 1.33, 0.5],
            id="solver",
            instrument_labels=["eur 1y", "usd 1y", "eur 1y xcs adj.", "1y cpi"],
            fx=fxf,
        )
        result = inst.delta(solver=solver)
        assert abs((result.iloc[0, 0] - 25.0)) < 1.0
        result2 = inst.npv(solver=solver)
        assert abs(result2) < 1e-3

        # test that instruments have not been set by the previous pricing action
        solver.s = [1.3, 1.4, 1.36, 0.55]
        solver.iterate()
        result3 = inst.npv(solver=solver)
        assert abs(result3) < 1e-3


class TestIRS:
    @pytest.mark.parametrize(
        "float_spread, fixed_rate, expected",
        [
            (0, 4.03, 4.03637780),
            (3, 4.03, 4.06637780),
            (0, 5.10, 4.03637780),
        ],
    )
    def test_irs_rate(self, curve, float_spread, fixed_rate, expected):
        # test the mid-market rate ignores the given fixed_rate and reacts to float_spread
        irs = IRS(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            frequency="Q",
            fixed_rate=4.03,
            stub="ShortFront",
            leg2_float_spread=float_spread,
        )
        result = irs.rate(curve)
        assert abs(result - expected) < 1e-7

    @pytest.mark.parametrize(
        "float_spread, fixed_rate, expected",
        [
            (0, 4.03, -0.63777963),
            (200, 4.03, -0.63777963),
            (500, 4.03, -0.63777963),
            (0, 4.01, -2.63777963),
        ],
    )
    def test_irs_spread_none_simple(self, curve, float_spread, fixed_rate, expected):
        # test the mid-market float spread ignores the given float_spread and react to fixed
        irs = IRS(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            frequency="Q",
            fixed_rate=fixed_rate,
            leg2_float_spread=float_spread,
            leg2_fixing_method="rfr_payment_delay",
            leg2_spread_compound_method="none_simple",
            stub="ShortFront",
        )
        result = irs.spread(curve)
        assert abs(result - expected) < 1e-7

        irs.leg2_float_spread = result
        validate = irs.npv(curve)
        assert abs(validate) < 1e-8

    @pytest.mark.parametrize(
        "float_spread, fixed_rate, expected",
        [
            (0, 4.03, -0.6322524949759807),  # note this is the closest solution
            (
                200,
                4.03,
                -0.632906212667,
            ),  # note this is 0.0006bp inaccurate due to approx
            (500, 4.03, -0.64246185),  # note this is 0.0102bp inaccurate due to approx
            (0, 4.01, -2.61497625534),
        ],
    )
    def test_irs_spread_isda_compound(self, curve, float_spread, fixed_rate, expected):
        # test the mid-market float spread ignores the given float_spread and react to fixed
        irs = IRS(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            frequency="Q",
            fixed_rate=fixed_rate,
            leg2_float_spread=float_spread,
            leg2_fixing_method="rfr_payment_delay",
            leg2_spread_compound_method="isda_compounding",
            stub="ShortFront",
        )
        result = irs.spread(curve)
        assert abs(result - expected) < 1e-2

        irs.leg2_float_spread = result
        validate = irs.npv(curve)
        assert abs(validate) < 5e2

    @pytest.mark.parametrize(
        "float_spread, fixed_rate, expected",
        [
            (0, 4.03, -0.63500600),  # note this is the closest solution
            (
                200,
                4.03,
                -0.6348797243,
            ),  # note this is 0.0001bp inaccurate due to approx
            (
                500,
                4.03,
                -0.6346903026,
            ),  # note this is 0.0003bp inaccurate due to approx
            (0, 4.01, -2.626308241),
        ],
    )
    def test_irs_spread_isda_flat_compound(
        self, curve, float_spread, fixed_rate, expected
    ):
        # test the mid-market float spread ignores the given float_spread and react to fixed
        irs = IRS(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            frequency="Q",
            fixed_rate=fixed_rate,
            leg2_float_spread=float_spread,
            leg2_fixing_method="rfr_payment_delay",
            leg2_spread_compound_method="isda_flat_compounding",
            stub="ShortFront",
        )
        result = irs.spread(curve)
        assert abs(result - expected) < 1e-2

        irs.leg2_float_spread = result
        validate = irs.npv(curve)
        assert abs(validate) < 20

    def test_irs_npv(self, curve):
        irs = IRS(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            frequency="Q",
            fixed_rate=4.035,
            stub="ShortFront",
            leg2_float_spread=0,
        )
        result = irs.npv(curve)
        expected = irs.analytic_delta(curve) * (4.035 - irs.rate(curve)) * -100
        assert abs(result - expected) < 1e-9
        assert abs(result - 5704.13604352) < 1e-7

    def test_irs_cashflows(self, curve):
        irs = IRS(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            frequency="Q",
            fixed_rate=4.035,
            leg2_float_spread=None,
            stub="ShortFront",
        )
        result = irs.cashflows(curve)
        assert isinstance(result, DataFrame)
        assert result.index.nlevels == 2

    def test_irs_npv_mid_mkt_zero(self, curve):
        irs = IRS(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            frequency="Q",
            stub="ShortFront",
        )
        result = irs.npv(curve)
        assert abs(result) < 1e-8

        irs.fixed_rate = 1.0  # pay fixed low rate implies positive NPV
        assert irs.npv(curve) > 1

        irs.fixed_rate = None  # fixed rate set back to initial
        assert abs(irs.npv(curve)) < 1e-8

        irs.fixed_rate = float(irs.rate(curve))
        irs.leg2_float_spread = 100
        assert irs.npv(curve) > 1

        irs.leg2_float_spread = None
        assert abs(irs.npv(curve)) < 1e-8

    def test_sbs_float_spread_raises(self, curve):
        irs = IRS(dt(2022, 1, 1), "9M", "Q")
        with pytest.raises(AttributeError, match="Cannot set `float_spread`"):
            irs.float_spread = 1.0

    def test_index_base_raises(self):
        irs = IRS(dt(2022, 1, 1), "9M", "Q")
        with pytest.raises(AttributeError, match="Cannot set `index_base`"):
            irs.index_base = 1.0

        with pytest.raises(AttributeError, match="Cannot set `leg2_index_base`"):
            irs.leg2_index_base = 1.0


class TestIIRS:

    def test_index_base_none_populated(self, curve):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.5, dt(2034, 1, 1): 0.4},
            index_lag=3,
            index_base=100.0
        )
        iirs = IIRS(
            effective=dt(2022, 2, 1),
            termination="1y",
            frequency="Q",
            index_lag=3,
            notional_exchange=False,
        )
        for period in iirs.leg1.periods:
            assert period.index_base is None
        iirs.rate(curves=[i_curve, curve])
        for period in iirs.leg1.periods:
            assert period.index_base == 200.0

    def test_iirs_npv_mid_mkt_zero(self, curve):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.5, dt(2034, 1, 1): 0.4},
            index_lag=3,
            index_base=100.0
        )
        iirs = IIRS(
            effective=dt(2022, 2, 1),
            termination=dt(2022, 7, 1),
            payment_lag=0,
            notional=1e9,
            convention="Act360",
            frequency="Q",
            stub="ShortFront",
        )
        result = iirs.npv([i_curve, curve])
        assert abs(result) < 1e-8

        iirs.fixed_rate = iirs.rate([i_curve, curve])
        iirs.index_base = 1000.0  # high index base implies positive NPV
        assert iirs.npv([i_curve, curve]) > 1

        iirs.index_base = None  # index_base set back to initial
        iirs.fixed_rate = None
        assert abs(iirs.npv([i_curve, curve])) < 1e-8

        mid_fixed = float(iirs.rate([i_curve, curve]))
        iirs.base_index = 200.0  # this is index_base from i_curve
        new_mid = float(iirs.rate([i_curve, curve]))
        assert abs(mid_fixed - new_mid) < 1e-6

    def test_cashflows(self, curve):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2023, 2, 1): 0.99},
            index_lag=3,
            index_base=100.0
        )
        iirs = IIRS(
            effective=dt(2022, 2, 1),
            termination="9M",
            frequency="Q",
            index_base=Series([90, 110], index=[dt(2022, 1, 31), dt(2022, 2, 2)]),
            index_fixings=[110, 115],
            index_lag=3,
            index_method="daily",
            fixed_rate=1.0
        )
        result = iirs.cashflows([i_curve, curve, curve, curve])
        expected = DataFrame({
            "Index Val": [110.0, 115.0, 100.7754, np.nan, np.nan, np.nan],
            "Index Ratio": [1.10, 1.15, 1.00775, np.nan, np.nan, np.nan],
            "NPV": [
                -2682.655, -2869.534, -2488.937, 9849.93, 10070.85, 9963.277
            ],
            "Type": ["IndexFixedPeriod"] * 3 + ["FloatPeriod"] * 3
        }, index= MultiIndex.from_tuples([
            ("leg1", 0), ("leg1", 1), ("leg1", 2), ("leg2", 0), ("leg2", 1), ("leg2", 2)
        ]))
        assert_frame_equal(
            expected, result[["Index Val", "Index Ratio", "NPV", "Type"]], rtol=1e-3,
        )

    def test_npv_no_index_base(self, curve):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.5, dt(2034, 1, 1): 0.4},
            index_lag=3,
            index_base=100.0
        )
        iirs = IIRS(
            effective=dt(2022, 2, 1),
            termination="1y",
            frequency="Q",
            fixed_rate=2.0,
            index_lag=3,
            notional_exchange=False,
        )
        result = iirs.npv([i_curve, curve, curve, curve])
        expected = 19792.08369745
        assert abs(result - expected) < 1e-6

    def test_cashflows_no_index_base(self, curve):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.5, dt(2034, 1, 1): 0.4},
            index_lag=3,
            index_base=100.0
        )
        iirs = IIRS(
            effective=dt(2022, 2, 1),
            termination="1y",
            frequency="Q",
            fixed_rate=2.0,
            index_lag=3,
            notional_exchange=False,
        )
        result = iirs.cashflows([i_curve, curve, curve, curve])
        for i in range(4):
            assert result.iloc[i]["Index Base"] == 200.0


class TestSBS:
    def test_sbs_npv(self, curve):
        sbs = SBS(dt(2022, 1, 1), "9M", "Q", float_spread=3.0)
        a_delta = sbs.analytic_delta(curve, curve, leg=1)
        npv = sbs.npv(curve)
        assert abs(npv + 3.0 * a_delta) < 1e-9

        sbs.leg2_float_spread = 4.5
        npv = sbs.npv(curve)
        assert abs(npv - 1.5 * a_delta) < 1e-9

    def test_sbs_rate(self, curve):
        sbs = SBS(dt(2022, 1, 1), "9M", "Q", float_spread=3.0)
        result = sbs.rate([curve], leg=1)
        alias = sbs.spread([curve], leg=1)
        assert abs(result - 0) < 1e-8
        assert abs(alias - 0) < 1e-8

        result = sbs.rate([curve], leg=2)
        alias = sbs.rate([curve], leg=2)
        assert abs(result - 3.0) < 1e-8
        assert abs(alias - 3.0) < 1e-8

    def test_sbs_cashflows(self, curve):
        sbs = SBS(dt(2022, 1, 1), "9M", "Q", float_spread=3.0)
        result = sbs.cashflows(curve)
        expected = DataFrame(
            {
                "Type": ["FloatPeriod", "FloatPeriod"],
                "Period": ["Regular", "Regular"],
                "Spread": [3.0, 0.0],
            },
            index=MultiIndex.from_tuples([("leg1", 0), ("leg2", 2)]),
        )
        assert_frame_equal(
            result.loc[[("leg1", 0), ("leg2", 2)], ["Type", "Period", "Spread"]],
            expected,
        )

    def test_sbs_fixed_rate_raises(self, curve):
        sbs = SBS(dt(2022, 1, 1), "9M", "Q", float_spread=3.0)
        with pytest.raises(AttributeError, match="Cannot set `fixed_rate`"):
            sbs.fixed_rate = 1.0

        with pytest.raises(AttributeError, match="Cannot set `leg2_fixed_rate`"):
            sbs.leg2_fixed_rate = 1.0


class TestFRA:
    def test_fra_rate(self, curve):
        # test the mid-market rate ignores the given fixed_rate and reacts to float_spread
        fra = FRA(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 7, 1),
            notional=1e9,
            convention="Act360",
            frequency="S",
            fixed_rate=4.00,
        )
        result = fra.rate(curve)
        expected = 4.0590821964144
        assert abs(result - expected) < 1e-7

    def test_fra_npv(self, curve):
        fra = FRA(
            effective=dt(2022, 1, 1),
            termination="6m",
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            modifier="mf",
            frequency="S",
            fixed_rate=4.035,
        )
        result = fra.npv(curve)
        expected = fra.analytic_delta(curve) * (4.035 - fra.rate(curve)) * -100
        assert abs(result - expected) < 1e-8
        assert abs(result - 118631.8350458332) < 1e-7

    def test_fra_cashflows(self, curve):
        fra = FRA(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 7, 1),
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            frequency="Q",
            fixed_rate=4.035,
        )
        result = fra.cashflows(curve)
        assert isinstance(result, DataFrame)
        assert result.index.nlevels == 1

    def test_irs_npv_mid_mkt_zero(self, curve):
        fra = FRA(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 7, 1),
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            frequency="S",
        )
        result = fra.npv(curve)
        assert abs(result) < 1e-9

        fra.fixed_rate = 1.0  # pay fixed low rate implies positive NPV
        assert fra.npv(curve) > 1

        fra.fixed_rate = None  # fixed rate set back to initial
        assert abs(fra.npv(curve)) < 1e-9


class TestZCS:
    @pytest.mark.parametrize("freq, exp", [("Q", 3.52986327830), ("S", 3.54543819675)])
    def test_zcs_rate(self, freq, exp):
        usd = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2027, 1, 1): 0.85, dt(2032, 1, 1): 0.70},
            id="usd",
        )
        zcs = ZCS(
            effective=dt(2022, 1, 1),
            termination="10Y",
            frequency=freq,
            leg2_frequency="Q",
            calendar="nyc",
            currency="usd",
            fixed_rate=4.0,
            convention="Act360",
            notional=100e6,
            curves=["usd"],
        )
        result = zcs.rate(usd)
        assert abs(result - exp) < 1e-7

    def test_zcs_analytic_delta(self):
        usd = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2027, 1, 1): 0.85, dt(2032, 1, 1): 0.70},
            id="usd",
        )
        zcs = ZCS(
            effective=dt(2022, 1, 1),
            termination="10Y",
            frequency="Q",
            leg2_frequency="Q",
            calendar="nyc",
            currency="usd",
            fixed_rate=4.0,
            convention="Act360",
            notional=100e6,
            curves=["usd"],
        )
        result = zcs.analytic_delta(usd, usd)
        expected = 105226.66099084
        assert abs(result - expected) < 1e-7


class TestZCIS:

    def test_leg2_index_base(self, curve):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_base=200.0
        )
        zcis = ZCIS(
            effective=dt(2022, 1, 1),
            termination="9m",
            frequency="Q",
        )
        prior = zcis.rate(curves=[curve, curve, i_curve, curve])

        zcis.leg2_index_base = 100.0  # index base is lower
        result = zcis.rate(curves=[curve, curve, i_curve, curve])
        assert result > (prior + 100)


def test_forward_fx_immediate():
    d_curve = Curve(
        nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, interpolation="log_linear"
    )
    f_curve = Curve(nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.95})
    result = forward_fx(dt(2022, 4, 1), d_curve, f_curve, 10.0)
    assert abs(result - 10.102214) < 1e-6

    result = forward_fx(dt(2022, 1, 1), d_curve, f_curve, 10.0, dt(2022, 1, 1))
    assert abs(result - 10.0) < 1e-6


def test_forward_fx_spot_equivalent():
    d_curve = Curve(
        nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, interpolation="log_linear"
    )
    f_curve = Curve(nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.95})
    result = forward_fx(dt(2022, 7, 1), d_curve, f_curve, 10.102214, dt(2022, 4, 1))
    assert abs(result - 10.206626) < 1e-6


# test the commented out FXSwap variant
# def test_fx_swap(curve, curve2):
#     fxs = FXSwap(dt(2022, 1, 15), "3M", notional=1000, fx_fixing_points=(10.1, 105),
#                  currency="eur", leg2_currency="sek")
#     assert len(fxs.leg1.periods) == 2
#     assert len(fxs.leg2.periods) == 2
#
#     assert fxs.leg1.periods[0].notional == 1000
#     assert fxs.leg1.periods[0].payment == dt(2022, 1, 15)
#     assert fxs.leg1.periods[1].notional == -1000
#     assert fxs.leg1.periods[1].payment == dt(2022, 4, 15)
#
#     assert fxs.leg2.periods[0].notional == -10100
#     assert fxs.leg2.periods[0].payment == dt(2022, 1, 15)
#     assert fxs.leg2.periods[1].notional == 10110.5
#     assert fxs.leg2.periods[1].payment == dt(2022, 4, 15)
#
#     fxs.fx_fixing_points = None
#     points = fxs._rate_alt(curve, curve2, 10.0)
#     npv = fxs._npv_alt(curve, curve2, 10.0)
#     assert abs(npv) < 1e-9
#
#     fxf = FXForwards(
#         FXRates({"eursek": 10.0}, dt(2022, 1, 1)),
#         {"eureur": curve, "seksek": curve2, "sekeur": curve2}
#     )
#     points2 = fxs.rate(fxf)
#     npv2 = fxs.npv(fxf, None, "eur")
#     assert abs(npv2) < 1e-9


class TestNonMtmXCS:
    def test_nonmtmxcs_npv(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"eurusd": 1.1}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "eurusd": curve2, "eureur": curve2},
        )

        xcs = NonMtmXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="eur",
            leg2_currency="usd",
            payment_lag_exchange=0,
        )
        npv2 = xcs._npv2(curve2, curve2, curve, curve, 1.10)
        npv = xcs.npv([curve2, curve2, curve, curve], None, fxf)
        assert abs(npv) < 1e-9

        xcs = NonMtmXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            amortization=100e3,
            currency="eur",
            leg2_currency="usd",
            payment_lag_exchange=0,
        )
        npv2 = xcs._npv2(curve2, curve2, curve, curve, 1.10)
        npv = xcs.npv([curve2, curve2, curve, curve], None, fxf)
        assert abs(npv) < 1e-9

    def test_nonmtmxcs_fx_notional(self):
        xcs = NonMtmXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="eur",
            leg2_currency="usd",
            payment_lag_exchange=0,
            fx_fixing=2.0,
            notional=1e6,
        )
        assert xcs.leg2_notional == -2e6

    @pytest.mark.parametrize(
        "float_spd, compound, expected",
        [
            (10, "none_simple", 10.160794),
            (100, "none_simple", 101.60794),
            (100, "isda_compounding", 101.023590),
            (100, "isda_flat_compounding", 101.336040),
        ],
    )
    def test_nonmtmxcs_spread(self, curve, curve2, float_spd, compound, expected):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        xcs = NonMtmXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            float_spread=float_spd,
            leg2_spread_compound_method=compound,
        )

        result = xcs.rate([curve, curve, curve2, curve2], None, fxf, 2)
        assert abs(result - expected) < 1e-4
        alias = xcs.spread([curve, curve, curve2, curve2], None, fxf, 2)
        assert alias == result

        xcs.leg2_float_spread = result
        validate = xcs.npv([curve, curve, curve2, curve2], None, fxf)
        assert abs(validate) < 1e-2
        result2 = xcs.rate([curve, curve, curve2, curve2], None, fxf, 2)
        assert abs(result - result2) < 1e-3

        # reverse legs
        xcs_reverse = NonMtmXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="usd",
            leg2_currency="nok",
            payment_lag_exchange=0,
            notional=1e6,
            leg2_float_spread=float_spd,
            spread_compound_method=compound,
        )
        result = xcs_reverse.rate([curve2, curve2, curve, curve], None, fxf, 1)
        assert abs(result - expected) < 1e-4

    def test_no_fx_raises(self, curve, curve2):
        xcs = NonMtmXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )

        with pytest.raises(ValueError, match="`fx` is required when `fx_fixing` is"):
            with default_context("no_fx_fixings_for_xcs", "raise"):
                xcs.npv([curve, curve, curve2, curve2])

        with pytest.raises(ValueError, match="`fx` is required when `fx_fixing` is"):
            with default_context("no_fx_fixings_for_xcs", "raise"):
                xcs.cashflows([curve, curve, curve2, curve2])

        with pytest.warns():
            with default_context("no_fx_fixings_for_xcs", "warn"):
                xcs.npv([curve, curve, curve2, curve2])

    def test_nonmtmxcs_cashflows(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        xcs = NonMtmXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )

        result = xcs.cashflows([curve, curve, curve2, curve2], None, fxf)
        expected = DataFrame(
            {
                "Type": ["Cashflow", "FloatPeriod"],
                "Period": ["Exchange", "Regular"],
                "Ccy": ["NOK", "USD"],
                "Notional": [-10000000, -996734.0252423884],
                "FX Rate": [0.10002256337062124, 1.0],
            },
            index=MultiIndex.from_tuples([("leg1", 0), ("leg2", 8)]),
        )
        assert_frame_equal(
            result.loc[
                [("leg1", 0), ("leg2", 8)],
                ["Type", "Period", "Ccy", "Notional", "FX Rate"],
            ],
            expected,
        )

    @pytest.mark.parametrize("fix", ["fxr", "fxf", "float", "dual", "dual2"])
    def test_nonmtm_fx_fixing(self, curve, curve2, fix):
        fxr = FXRates({"usdnok": 10}, settlement=dt(2022, 1, 1))
        fxf = FXForwards(fxr, {"usdusd": curve, "nokusd": curve2, "noknok": curve2})
        mapping = {
            "fxr": fxr,
            "fxf": fxf,
            "float": 10.0,
            "dual": Dual(10.0, "x"),
            "dual2": Dual2(10.0, "x"),
        }
        xcs = NonMtmXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            fx_fixing=mapping[fix],
        )
        assert abs(xcs.npv([curve, curve, curve2, curve2])) < 1e-7


class TestNonMtmFixedFloatXCS:
    @pytest.mark.parametrize(
        "float_spd, compound, expected",
        [
            (10, "none_simple", 6.70955968),
            (100, "isda_compounding", 7.62137047),
        ],
    )
    def test_nonmtmfixxcs_rate_npv(self, curve, curve2, float_spd, compound, expected):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )
        xcs = NonMtmFixedFloatXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            leg2_spread_compound_method=compound,
            leg2_float_spread=float_spd,
        )

        result = xcs.rate([curve2, curve2, curve, curve], None, fxf, 1)
        assert abs(result - expected) < 1e-4
        assert abs(xcs.npv([curve2, curve2, curve, curve], None, fxf)) < 1e-6

        xcs.fixed_rate = result  # set the fixed rate and check revalues to zero
        assert abs(xcs.npv([curve2, curve2, curve, curve], None, fxf)) < 1e-6

        irs = IRS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_spread_compound_method=compound,
            leg2_float_spread=float_spd,
        )
        validate = irs.rate(curve2)
        assert abs(result - validate) < 1e-2

    def test_nonmtmfixxcs_fx_notional(self):
        xcs = NonMtmFixedFloatXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="eur",
            leg2_currency="usd",
            payment_lag_exchange=0,
            fx_fixing=2.0,
            notional=1e6,
        )
        assert xcs.leg2_notional == -2e6

    def test_nonmtmfixxcs_no_fx_raises(self, curve, curve2):
        xcs = NonMtmFixedFloatXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )

        with pytest.raises(ValueError, match="`fx` is required when `fx_fixing` is"):
            with default_context("no_fx_fixings_for_xcs", "raise"):
                xcs.npv([curve, curve, curve2, curve2])

        with pytest.raises(ValueError, match="`fx` is required when `fx_fixing` is"):
            with default_context("no_fx_fixings_for_xcs", "raise"):
                xcs.cashflows([curve, curve, curve2, curve2])

    def test_nonmtmfixxcs_cashflows(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        xcs = NonMtmFixedFloatXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )

        result = xcs.cashflows([curve, curve, curve2, curve2], None, fxf)
        expected = DataFrame(
            {
                "Type": ["Cashflow", "FloatPeriod"],
                "Period": ["Exchange", "Regular"],
                "Ccy": ["NOK", "USD"],
                "Notional": [-10000000, -996734.0252423884],
                "FX Rate": [0.10002256337062124, 1.0],
            },
            index=MultiIndex.from_tuples([("leg1", 0), ("leg2", 8)]),
        )
        assert_frame_equal(
            result.loc[
                [("leg1", 0), ("leg2", 8)],
                ["Type", "Period", "Ccy", "Notional", "FX Rate"],
            ],
            expected,
        )

    @pytest.mark.parametrize("fix", ["fxr", "fxf", "float", "dual", "dual2"])
    def test_nonmtmfixxcs_fx_fixing(self, curve, curve2, fix):
        fxr = FXRates({"usdnok": 10}, settlement=dt(2022, 1, 1))
        fxf = FXForwards(fxr, {"usdusd": curve, "nokusd": curve2, "noknok": curve2})
        mapping = {
            "fxr": fxr,
            "fxf": fxf,
            "float": 10.0,
            "dual": Dual(10.0, "x"),
            "dual2": Dual2(10.0, "x"),
        }
        xcs = NonMtmFixedFloatXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            fx_fixing=mapping[fix],
            leg2_float_spread=10.0,
        )
        assert abs(xcs.npv([curve2, curve2, curve, curve])) < 1e-7

    def test_nonmtmfixxcs_raises(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        xcs = NonMtmFixedFloatXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )

        with pytest.raises(ValueError, match="Cannot solve for a"):
            xcs.rate([curve, curve, curve2, curve2], None, fxf, leg=2)


class TestNonMtmFixedFixedXCS:
    # @pytest.mark.parametrize("float_spd, compound, expected",[
    #     (10, "none_simple", 6.70955968),
    #     (100, "isda_compounding", 7.62137047),
    # ])
    # def test_nonmtmfixxcs_rate_npv(self, curve, curve2, float_spd, compound, expected):
    #     fxf = FXForwards(
    #         FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
    #         {"usdusd": curve, "nokusd": curve2, "noknok": curve2}
    #     )
    #     xcs = NonMtmFixedFloatXCS(dt(2022, 2, 1), "8M", "M",
    #         payment_lag=0, currency="nok", leg2_currency="usd",
    #         payment_lag_exchange=0, notional=10e6,
    #         leg2_spread_compound_method=compound, leg2_float_spread=float_spd
    #      )
    #
    #     result = xcs.rate([curve2, curve2, curve, curve], None, fxf, 1)
    #     assert abs(result - expected) < 1e-4
    #     assert abs(xcs.npv([curve2, curve2, curve, curve], None, fxf)) < 1e-6
    #
    #     xcs.fixed_rate = result  # set the fixed rate and check revalues to zero
    #     assert abs(xcs.npv([curve2, curve2, curve, curve], None, fxf)) < 1e-6
    #
    #     irs = IRS(dt(2022, 2, 1), "8M", "M",
    #         payment_lag=0, currency="nok",
    #         leg2_spread_compound_method=compound, leg2_float_spread=float_spd)
    #     validate = irs.rate(curve2)
    #     assert abs(result - validate) < 1e-2
    #
    # def test_nonmtmfixxcs_fx_notional(self):
    #     xcs = NonMtmFixedFloatXCS(dt(2022, 2, 1), "8M", "M",
    #                     payment_lag=0, currency="eur", leg2_currency="usd",
    #                     payment_lag_exchange=0, fx_fixing=2.0, notional=1e6)
    #     assert xcs.leg2_notional == -2e6
    #
    # def test_nonmtmfixxcs_no_fx_raises(self, curve, curve2):
    #     xcs = NonMtmFixedFloatXCS(dt(2022, 2, 1), "8M", "M",
    #                     payment_lag=0, currency="nok", leg2_currency="usd",
    #                     payment_lag_exchange=0, notional=10e6)
    #
    #     with pytest.raises(ValueError, match="`fx` is required when `fx_fixing` is"):
    #         xcs.npv([curve, curve, curve2, curve2])
    #
    #     with pytest.raises(ValueError, match="`fx` is required when `fx_fixing` is"):
    #         xcs.cashflows([curve, curve, curve2, curve2])
    #
    # def test_nonmtmfixxcs_cashflows(self, curve, curve2):
    #     fxf = FXForwards(
    #         FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
    #         {"usdusd": curve, "nokusd": curve2, "noknok": curve2}
    #     )
    #
    #     xcs = NonMtmFixedFloatXCS(dt(2022, 2, 1), "8M", "M",
    #                     payment_lag=0, currency="nok", leg2_currency="usd",
    #                     payment_lag_exchange=0, notional=10e6)
    #
    #     result = xcs.cashflows([curve, curve, curve2, curve2], None, fxf)
    #     expected = DataFrame({
    #         "Type": ["Cashflow", "FloatPeriod"],
    #         "Period": ["Exchange", "Regular"],
    #         "Ccy": ["NOK", "USD"],
    #         "Notional": [-10000000, -996734.0252423884],
    #         "FX Rate": [0.10002256337062124, 1.0],
    #     }, index=MultiIndex.from_tuples([("leg1", 0), ("leg2", 8)]))
    #     assert_frame_equal(
    #         result.loc[[("leg1", 0), ("leg2", 8)], ["Type", "Period", "Ccy", "Notional", "FX Rate"]],
    #         expected,
    #     )

    @pytest.mark.parametrize("fix", ["fxr", "fxf", "float", "dual", "dual2"])
    def test_nonmtmfixxcs_fx_fixing(self, curve, curve2, fix):
        fxr = FXRates({"usdnok": 10}, settlement=dt(2022, 1, 1))
        fxf = FXForwards(fxr, {"usdusd": curve, "nokusd": curve2, "noknok": curve2})
        mapping = {
            "fxr": fxr,
            "fxf": fxf,
            "float": 10.0,
            "dual": Dual(10.0, "x"),
            "dual2": Dual2(10.0, "x"),
        }
        xcs = NonMtmFixedFixedXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            fx_fixing=mapping[fix],
            leg2_fixed_rate=2.0,
        )
        assert abs(xcs.npv([curve2, curve2, curve, curve])) < 1e-7

        xcs = NonMtmFixedFixedXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            fx_fixing=mapping[fix],
            fixed_rate=2.0,
        )
        assert abs(xcs.npv([curve2, curve2, curve, curve])) < 1e-7

    def test_nonmtmfixfixxcs_raises(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        xcs = NonMtmFixedFixedXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )

        with pytest.raises(ValueError, match="Cannot solve for a"):
            xcs.rate([curve, curve, curve2, curve2], None, fxf, leg=2)

        with pytest.raises(AttributeError, match="Cannot set `leg2_float_spread` for"):
            xcs.leg2_float_spread = 2.0


class TestXCS:
    def test_mtmxcs_npv(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"eurusd": 1.1}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "eurusd": curve2, "eureur": curve2},
        )

        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="eur",
            leg2_currency="usd",
            payment_lag_exchange=0,
        )

        npv = xcs.npv([curve2, curve2, curve, curve], None, fxf)
        assert abs(npv) < 1e-9

    def test_mtmxcs_cashflows(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )

        result = xcs.cashflows([curve, curve, curve2, curve2], None, fxf)
        expected = DataFrame(
            {
                "Type": ["Cashflow", "FloatPeriod", "Cashflow"],
                "Period": ["Exchange", "Regular", "Mtm"],
                "Ccy": ["NOK", "USD", "USD"],
                "Notional": [-10000000, -990019.24969, -3509.80082],
                "Rate": [np.nan, 8.181151773810475, 0.09829871161519926],
                "FX Rate": [0.10002256337062124, 1.0, 1.0],
            },
            index=MultiIndex.from_tuples([("leg1", 0), ("leg2", 11), ("leg2", 14)]),
        )
        assert_frame_equal(
            result.loc[
                [("leg1", 0), ("leg2", 11), ("leg2", 14)],
                ["Type", "Period", "Ccy", "Notional", "Rate", "FX Rate"],
            ],
            expected,
        )

    def test_mtmxcs_fx_fixings_raises(self):
        with pytest.raises(ValueError, match="`fx_fixings` for MTM XCS should"):
            _ = XCS(dt(2022, 2, 1), "8M", "M", fx_fixings=None)

        with pytest.raises(ValueError, match="`fx_fixings` for MTM XCS should"):
            _ = FixedFloatXCS(dt(2022, 2, 1), "8M", "M", fx_fixings=None)

        with pytest.raises(ValueError, match="`fx_fixings` for MTM XCS should"):
            _ = FixedFixedXCS(dt(2022, 2, 1), "8M", "M", fx_fixings=None)

        with pytest.raises(ValueError, match="`fx_fixings` for MTM XCS should"):
            _ = FloatFixedXCS(dt(2022, 2, 1), "8M", "M", fx_fixings=None)

    @pytest.mark.parametrize(
        "float_spd, compound, expected",
        [
            (10, "none_simple", 9.97839804),
            (100, "none_simple", 99.78398037),
            (100, "isda_compounding", 99.418428),
            (100, "isda_flat_compounding", 99.621117),
        ],
    )
    def test_mtmxcs_rate(self, float_spd, compound, expected, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            float_spread=float_spd,
            leg2_spread_compound_method=compound,
        )

        result = xcs.rate([curve2, curve2, curve, curve], None, fxf, 2)
        assert abs(result - expected) < 1e-4
        alias = xcs.spread([curve2, curve2, curve, curve], None, fxf, 2)
        assert alias == result

        xcs.leg2_float_spread = result
        validate = xcs.npv([curve2, curve2, curve, curve], None, fxf)
        assert abs(validate) < 1e-2
        result2 = xcs.rate([curve2, curve2, curve, curve], None, fxf, 2)
        assert abs(result - result2) < 1e-3


class TestFixedFloatXCS:
    def test_mtmfixxcs_rate(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        xcs = FixedFloatXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )

        result = xcs.rate([curve2, curve2, curve, curve], None, fxf, 1)

        irs = IRS(dt(2022, 2, 1), "8M", "M", currency="nok", payment_lag=0)
        validate = irs.rate(curve2)
        assert abs(result - validate) < 1e-4
        # alias = xcs.spread([curve2, curve2, curve, curve], None, fxf, 2)

    def test_mtmfixxcs_rate_reversed(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        xcs = FloatFixedXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="usd",
            leg2_currency="nok",
            payment_lag_exchange=0,
            notional=10e6,
        )

        result = xcs.rate([curve, curve, curve2, curve2], None, fxf, 2)

        irs = IRS(dt(2022, 2, 1), "8M", "M", currency="nok", payment_lag=0)
        validate = irs.rate(curve2)
        assert abs(result - validate) < 1e-2
        alias = xcs.spread([curve, curve, curve2, curve2], None, fxf, 2)
        assert abs(result - alias) < 1e-4


class TestFixedFixedXCS:
    def test_mtmfixfixxcs_rate(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        irs = IRS(dt(2022, 2, 1), "8M", "M", payment_lag=0)
        nok_rate = float(irs.rate(curve2))
        xcs = FixedFixedXCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            fixed_rate=nok_rate,
        )
        result = xcs.rate([curve2, curve2, curve, curve], None, fxf, 2)
        validate = irs.rate(curve)
        assert abs(result - validate) < 1e-4
        alias = xcs.spread([curve2, curve2, curve, curve], None, fxf, 2)
        assert abs(result - alias) < 1e-8

        ## test reverse
        usd_rate = float(irs.rate(curve))
        xcs.fixed_rate = None
        xcs.leg2_fixed_rate = usd_rate
        result = xcs.rate([curve2, curve2, curve, curve], None, fxf, 1)
        validate = irs.rate(curve2)
        assert abs(result - validate) < 1e-4
        alias = xcs.spread([curve2, curve2, curve, curve], None, fxf, 1)
        assert abs(result - alias) < 1e-8


class TestFXSwap:
    def test_fxswap_rate(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )
        fxs = FXSwap(
            dt(2022, 2, 1),
            "8M",
            "M",
            currency="usd",
            leg2_currency="nok",
            payment_lag_exchange=0,
            notional=1e6,
        )
        expected = fxf.swap("usdnok", [dt(2022, 2, 1), dt(2022, 10, 1)])
        result = fxs.rate([None, curve, None, curve2], None, fxf)
        assert result == expected

    def test_fxswap_npv(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )
        fxs = FXSwap(
            dt(2022, 2, 1),
            "8M",
            "M",
            currency="usd",
            leg2_currency="nok",
            payment_lag_exchange=0,
            notional=1e6,
        )

        assert abs(fxs.npv([None, curve, None, curve2], None, fxf)) < 1e-7

        result = fxs.rate([None, curve, None, curve2], None, fxf, fixed_rate=True)
        fxs.leg2_fixed_rate = result
        assert abs(fxs.npv([None, curve, None, curve2], None, fxf)) < 1e-7

    def test_fxswap_points_raises(self):
        with pytest.raises(ValueError, match="Cannot set `points` on FXSwap without"):
            fxs = FXSwap(
                dt(2022, 2, 1),
                "8M",
                "M",
                points=1000.0,
                currency="usd",
                leg2_currency="nok",
                payment_lag_exchange=0,
                notional=1e6,
            )

    def test_fxswap_fixing_and_points(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )
        fxs = FXSwap(
            dt(2022, 2, 1),
            "8M",
            "M",
            fx_fixing=11.0,
            points=1754.56233604,
            currency="usd",
            leg2_currency="nok",
            payment_lag_exchange=0,
            notional=1e6,
        )
        npv = fxs.npv([None, curve, None, curve2], None, fxf)
        assert abs(npv + 4166.37288388) < 1e-4

    # def test_proxy_curve_from_fxf(self, curve, curve2):
    #     # TODO this needs a solver from which to test the proxy curve (line 92)
    #     fxf = FXForwards(
    #         FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
    #         {"usdusd": curve, "nokusd": curve2, "noknok": curve2}
    #     )
    #     fxs = FXSwap(dt(2022, 2, 1), "8M", "M",
    #                         currency="usd", leg2_currency="nok",
    #                         payment_lag_exchange=0, notional=1e6,
    #                         leg2_fixed_rate=-1.0)
    #     npv_nok = fxs.npv([None, fxf.curve("usd", "nok"), None, curve2], None, fxf)
    #     npv_usd = fxs.npv([None, curve, None, fxf.curve("nok", "usd")], None, fxf)
    #     assert abs(npv_nok-npv_usd) < 1e-7  # npvs are equivalent becasue xcs basis =0


class TestFixedRateBond:
    def test_fixed_rate_bond_price(self):
        # test pricing functions against Gilt Example prices from UK DMO
        bond = FixedRateBond(
            dt(1995, 1, 1),
            dt(2015, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
        )
        assert bond.price(4.445, dt(1999, 5, 24), True) - 145.012268 < 1e-6
        assert bond.price(4.445, dt(1999, 5, 26), True) - 145.047301 < 1e-6
        assert bond.price(4.445, dt(1999, 5, 27), True) - 141.070132 < 1e-6
        assert bond.price(4.445, dt(1999, 6, 7), True) - 141.257676 < 1e-6

        bond = FixedRateBond(
            dt(1997, 1, 1),
            dt(2004, 11, 26),
            "S",
            convention="ActActICMA",
            fixed_rate=6.75,
            ex_div=7,
            calendar="ldn",
        )
        assert bond.price(4.634, dt(1999, 5, 10), True) - 113.315543 < 1e-6
        assert bond.price(4.634, dt(1999, 5, 17), True) - 113.415969 < 1e-6
        assert bond.price(4.634, dt(1999, 5, 18), True) - 110.058738 < 1e-6
        assert bond.price(4.634, dt(1999, 5, 26), True) - 110.170218 < 1e-6

    def test_fixed_rate_bond_yield(self):
        # test pricing functions against Gilt Example prices from UK DMO
        bond = FixedRateBond(
            dt(1995, 1, 1),
            dt(2015, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
        )
        assert bond.ytm(135.0, dt(1999, 5, 24), True) - 5.1620635 < 1e-6
        assert bond.ytm(135.0, dt(1999, 5, 26), True) - 5.1649111 < 1e-6
        assert bond.ytm(135.0, dt(1999, 5, 27), True) - 4.871425 < 1e-6
        assert bond.ytm(135.0, dt(1999, 6, 7), True) - 4.8856785 < 1e-6

        bond = FixedRateBond(
            dt(1997, 1, 1),
            dt(2004, 11, 26),
            "S",
            convention="ActActICMA",
            fixed_rate=6.75,
            ex_div=7,
            calendar="ldn",
        )
        assert bond.ytm(108.0, dt(1999, 5, 10), True) - 5.7009527 < 1e-6
        assert bond.ytm(108.0, dt(1999, 5, 17), True) - 5.7253361 < 1e-6
        assert bond.ytm(108.0, dt(1999, 5, 18), True) - 5.0413308 < 1e-6
        assert bond.ytm(108.0, dt(1999, 5, 26), True) - 5.0652248 < 1e-6

    def test_fixed_rate_bond_yield_domains(self):
        bond = FixedRateBond(
            dt(1995, 1, 1),
            dt(2015, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
        )
        assert bond.ytm(500.0, dt(1999, 5, 24), True) + 5.86484231333 < 1e-8
        assert bond.ytm(200, dt(1999, 5, 24), True) - 1.4366895440550 < 1e-8
        assert bond.ytm(100, dt(1999, 5, 24), True) - 8.416909601459 < 1e-8
        assert bond.ytm(50, dt(1999, 5, 24), True) - 18.486840866431 < 1e-6
        assert bond.ytm(1, dt(1999, 5, 24), True) - 13421775210.82037 < 1e-3

    def test_fixed_rate_bond_ytm_duals(self):
        bond = FixedRateBond(
            dt(1995, 1, 1),
            dt(2015, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
        )

        dPdy = bond.duration(4, dt(1995, 1, 1))
        P = bond.price(4, dt(1995, 1, 1))
        result = bond.ytm(Dual(P, ["a", "b"], [1, -0.5]), dt(1995, 1, 1))
        assert result == Dual(4.00, ["a", "b"], [-1 / dPdy, 0.5 / dPdy])

        d2ydP2 = -bond.convexity(4, dt(1995, 1, 1)) * -(dPdy**-3)
        result = bond.ytm(Dual2(P, ["a", "b"], [1, -0.5]), dt(1995, 1, 1))
        expected = Dual2(
            4.00,
            ["a", "b"],
            [-1 / dPdy, 0.5 / dPdy],
            0.5 * np.array([[d2ydP2, d2ydP2 * -0.5], [d2ydP2 * -0.5, d2ydP2 * 0.25]]),
        )
        assert result == expected

    def test_fixed_rate_bond_accrual(self):
        # test pricing functions against Gilt Example prices from UK DMO, with stub
        bond = FixedRateBond(
            dt(1999, 5, 7),
            dt(2002, 12, 7),
            "S",
            convention="ActActICMA",
            front_stub=dt(1999, 12, 7),
            fixed_rate=6,
            ex_div=7,
            calendar="ldn",
        )
        bond.accrued(dt(1999, 5, 8)) == 0.016484
        bond.accrued(dt(1999, 6, 8)) == 0.527382
        bond.accrued(dt(1999, 7, 8)) == 1.019186
        bond.accrued(dt(1999, 11, 8)) == 3.035579
        bond.accrued(dt(1999, 11, 26)) == 3.330661
        bond.accrued(dt(1999, 11, 27)) == -0.16393
        bond.accrued(dt(1999, 12, 6)) == -0.01639
        bond.accrued(dt(1999, 12, 7)) == 0.0

    def test_fixed_rate_bond_stub_ytm(self):
        # if a regular bond is set to stub similar output should be gotten
        bond = FixedRateBond(
            dt(1999, 6, 7),
            dt(2002, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=6,
            ex_div=7,
            calendar="ldn",
        )
        regular_ytm = bond.ytm(101, dt(1999, 11, 8), dirty=True)
        bond.leg1.periods[0].stub = True
        stubbed_ytm = bond.ytm(101, dt(1999, 11, 8), dirty=True)
        assert regular_ytm == stubbed_ytm

    def test_fixed_rate_bond_zero_frequency_raises(self):
        with pytest.raises(ValueError, match="FixedRateBond `frequency`"):
            FixedRateBond(dt(1999, 5, 7), dt(2002, 12, 7), "Z", convention="ActActICMA")

    @pytest.mark.parametrize("metric", ["risk", "duration", "modified"])
    def test_fixed_rate_bond_duration(self, metric):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
        )
        price0 = gilt.price(4.445, dt(1999, 5, 27))
        price1 = gilt.price(4.446, dt(1999, 5, 27))
        if metric == "risk":
            numeric = price0 - price1
        elif metric == "modified":
            numeric = (price0 - price1) / price0 * 100
        elif metric == "duration":
            numeric = (price0 - price1) / price0 * (1 + 4.445 / (100 * 2)) * 100

        result = gilt.duration(4.445, dt(1999, 5, 27), metric=metric)
        assert (result - numeric * 1000) < 1e-1

    def test_fixed_rate_bond_convexity(self):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
        )
        numeric = gilt.duration(4.445, dt(1999, 5, 27)) - gilt.duration(
            4.446, dt(1999, 5, 27)
        )
        result = gilt.convexity(4.445, dt(1999, 5, 27))
        assert (result - numeric * 1000) < 1e-3

    def test_fixed_rate_bond_rate(self):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            settle=0,
        )
        curve = Curve({dt(1998, 12, 9): 1.0, dt(2015, 12, 7): 0.50})
        clean_price = gilt.rate(curve, metric="clean_price")
        result = gilt.rate(
            curve, metric="fwd_clean_price", forward_settlement=dt(1998, 12, 9)
        )
        assert abs(result - clean_price) < 1e-8

        result = gilt.rate(curve, metric="dirty_price")
        expected = clean_price + gilt.accrued(dt(1998, 12, 9))
        assert result == expected
        result = gilt.rate(
            curve, metric="fwd_dirty_price", forward_settlement=dt(1998, 12, 9)
        )
        assert abs(result - clean_price - gilt.accrued(dt(1998, 12, 9))) < 1e-8

        result = gilt.rate(curve, metric="ytm")
        expected = gilt.ytm(clean_price, dt(1998, 12, 9), False)
        assert abs(result - expected) < 1e-8

    def test_fixed_rate_bond_npv(self):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            settle=0,
        )
        curve = Curve({dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 0.75})
        result = gilt.npv(curve)
        expected = 113.22198344812742
        assert abs(result - expected) < 1e-6

        gilt.settle = 1
        result = gilt.npv(curve)  # bond is ex div on settlement 26th Nov 2010
        expected = 109.229489312983  # bond has dropped a coupon payment of 4.
        assert abs(result - expected) < 1e-6

        result = gilt.npv(curve, local=True)
        assert abs(result["gbp"] - expected) < 1e-6

    def test_fixed_rate_bond_npv_private(self):
        # this test shadows 'fixed_rate_bond_npv' but extends it for projection
        curve = Curve(
            {dt(2004, 11, 25): 1.0, dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 0.75}
        )
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            settle=0,
        )
        result = gilt._npv_local(
            None, curve, None, None, dt(2010, 11, 26), dt(2010, 11, 25)
        )
        expected = 109.229489312983  # npv should match associated test
        assert abs(result - expected) < 1e-6

    def test_fixed_rate_bond_analytic_delta(self):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-1000000,
            settle=0,
        )
        curve = Curve({dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 1.0})
        result = gilt.analytic_delta(curve)
        expected = -550.0
        assert abs(result - expected) < 1e-6

        gilt.settle = 1
        result = gilt.analytic_delta(curve)  # bond is ex div on settle 26th Nov 2010
        expected = -500.0  # bond has dropped a 6m coupon payment
        assert abs(result - expected) < 1e-6

    def test_fixed_rate_bond_cashflows(self):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            settle=1,
        )
        curve = Curve({dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 0.75})

        flows = gilt.cashflows(curve)  # bond is ex div on 26th nov 2010
        result = flows[defaults.headers["npv"]].sum()
        expected = gilt.npv(curve)
        assert abs(result - expected) < 1e-6

        gilt.settle = 0
        flows = gilt.cashflows(curve)  # settlement from curve initial node
        result = flows[defaults.headers["npv"]].sum()
        expected = gilt.npv(curve)
        assert abs(result - expected) < 1e-6

    def test_fixed_rate_bond_rate_raises(self):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
        )
        curve = Curve({dt(1998, 12, 7): 1.0, dt(2015, 12, 7): 0.50})
        with pytest.raises(ValueError, match="`metric` must be in"):
            gilt.rate(curve, metric="bad_metric")

        with pytest.raises(ValueError, match="`forward_settlement` needed to"):
            gilt.rate(curve, metric="fwd_clean_price")

    def test_fixed_rate_bond_no_amortization(self):
        with pytest.raises(NotImplementedError, match="`amortization` for"):
            gilt = FixedRateBond(
                effective=dt(1998, 12, 7),
                termination=dt(2015, 12, 7),
                frequency="S",
                calendar="ldn",
                currency="gbp",
                convention="ActActICMA",
                ex_div=7,
                fixed_rate=8.0,
                notional=-100,
                amortization=100,
            )

    @pytest.mark.parametrize(
        "f_s, exp",
        [
            (dt(2001, 12, 31), 99.997513754),  # compounding of mid year coupon
            (dt(2002, 1, 1), 99.9975001688),  # this is now ex div on last coupon
        ],
    )
    def test_fixed_rate_bond_forward_price_analogue(self, f_s, exp):
        gilt = FixedRateBond(
            effective=dt(2001, 1, 1),
            termination=dt(2002, 1, 1),
            frequency="S",
            calendar=None,
            currency="gbp",
            convention="Act365f",
            ex_div=0,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        result = gilt.fwd_from_repo(100.0, dt(2001, 1, 1), f_s, 1.0, "act365f")
        assert abs(result - exp) < 1e-6

    @pytest.mark.parametrize(
        "f_s, exp",
        [
            (dt(2001, 12, 31), 100.49888361793),  # compounding of mid year coupon
            (dt(2002, 1, 1), 99.9975001688),  # this is now ex div on last coupon
        ],
    )
    def test_fixed_rate_bond_forward_price_analogue_dirty(self, f_s, exp):
        gilt = FixedRateBond(
            effective=dt(2001, 1, 1),
            termination=dt(2002, 1, 1),
            frequency="S",
            calendar=None,
            currency="gbp",
            convention="Act365f",
            ex_div=0,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        result = gilt.fwd_from_repo(
            100.0, dt(2001, 1, 1), f_s, 1.0, "act365f", dirty=True
        )
        assert abs(result - exp) < 1e-6

    @pytest.mark.parametrize(
        "s, f_s, exp",
        [
            (dt(2010, 11, 25), dt(2011, 11, 25), 99.9975000187),
            (dt(2010, 11, 28), dt(2011, 11, 28), 99.9975000187),
            (dt(2010, 11, 28), dt(2011, 11, 25), 99.997419419),
            (dt(2010, 11, 25), dt(2011, 11, 28), 99.997579958),
        ],
    )
    def test_fixed_rate_bond_forward_price_analogue_ex_div(self, s, f_s, exp):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="act365f",
            ex_div=7,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        result = gilt.fwd_from_repo(100.0, s, f_s, 1.0, "act365f")
        assert abs(result - exp) < 1e-6

    @pytest.mark.parametrize(
        "f_s, f_p",
        [
            (dt(2001, 12, 31), 99.997513754),  # compounding of mid year coupon
            (dt(2002, 1, 1), 99.9975001688),  # this is now ex div on last coupon
        ],
    )
    def test_fixed_rate_bond_implied_repo(self, f_s, f_p):
        gilt = FixedRateBond(
            effective=dt(2001, 1, 1),
            termination=dt(2002, 1, 1),
            frequency="S",
            calendar=None,
            currency="gbp",
            convention="Act365f",
            ex_div=0,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        result = gilt.repo_from_fwd(100.0, dt(2001, 1, 1), f_s, f_p, "act365f")
        assert abs(result - 1.00) < 1e-8

    @pytest.mark.parametrize(
        "f_s, f_p",
        [
            (dt(2001, 12, 31), 100.49888361793),  # compounding of mid year coupon
            (dt(2002, 1, 1), 99.9975001688),  # this is now ex div on last coupon
        ],
    )
    def test_fixed_rate_bond_implied_repo_analogue_dirty(self, f_s, f_p):
        gilt = FixedRateBond(
            effective=dt(2001, 1, 1),
            termination=dt(2002, 1, 1),
            frequency="S",
            calendar=None,
            currency="gbp",
            convention="Act365f",
            ex_div=0,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        result = gilt.repo_from_fwd(
            100.0, dt(2001, 1, 1), f_s, f_p, "act365f", dirty=True
        )
        assert abs(result - 1.0) < 1e-8


class TestIndexFixedRateBond:

    def test_fixed_rate_bond_price(self):
        # test pricing functions against Nominal Gilt Example prices from UK DMO
        # these prices should be equivalent for the REAL component of Index Bonds
        bond = IndexFixedRateBond(
            dt(1995, 1, 1),
            dt(2015, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
            index_base=100.0,
        )
        assert bond.price(4.445, dt(1999, 5, 24), True) - 145.012268 < 1e-6
        assert bond.price(4.445, dt(1999, 5, 26), True) - 145.047301 < 1e-6
        assert bond.price(4.445, dt(1999, 5, 27), True) - 141.070132 < 1e-6
        assert bond.price(4.445, dt(1999, 6, 7), True) - 141.257676 < 1e-6

        bond = IndexFixedRateBond(
            dt(1997, 1, 1),
            dt(2004, 11, 26),
            "S",
            convention="ActActICMA",
            fixed_rate=6.75,
            ex_div=7,
            calendar="ldn",
            index_base=100.0
        )
        assert bond.price(4.634, dt(1999, 5, 10), True) - 113.315543 < 1e-6
        assert bond.price(4.634, dt(1999, 5, 17), True) - 113.415969 < 1e-6
        assert bond.price(4.634, dt(1999, 5, 18), True) - 110.058738 < 1e-6
        assert bond.price(4.634, dt(1999, 5, 26), True) - 110.170218 < 1e-6

    def test_fixed_rate_bond_zero_frequency_raises(self):
        with pytest.raises(ValueError, match="FixedRateBond `frequency`"):
            IndexFixedRateBond(
                dt(1999, 5, 7), dt(2002, 12, 7), "Z", convention="ActActICMA"
            )

    def test_fixed_rate_bond_no_amortization(self):
        with pytest.raises(NotImplementedError, match="`amortization` for"):
            gilt = IndexFixedRateBond(
                effective=dt(1998, 12, 7),
                termination=dt(2015, 12, 7),
                frequency="S",
                calendar="ldn",
                currency="gbp",
                convention="ActActICMA",
                ex_div=7,
                fixed_rate=8.0,
                notional=-100,
                amortization=100,
                index_base=100.0
            )

    def test_fixed_rate_bond_rate_raises(self):
        gilt = IndexFixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            index_base=100.0,
        )
        curve = Curve({dt(1998, 12, 7): 1.0, dt(2015, 12, 7): 0.50})
        with pytest.raises(ValueError, match="`metric` must be in"):
            gilt.rate(curve, metric="bad_metric")

        with pytest.raises(ValueError, match="`forward_settlement` needed to"):
            gilt.rate(curve, metric="fwd_clean_price")

    @pytest.mark.parametrize("i_fixings, expected", [
        (None, 1.161227269),
        (Series([90, 290], index=[dt(2022, 4, 1), dt(2022, 4, 29)]), 2.00)
    ])
    def test_index_ratio(self, i_fixings, expected):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_lag=3,
            index_base=110.0,
        )
        bond = IndexFixedRateBond(
            dt(2022, 1, 1),
            "9m",
            "Q",
            convention="ActActICMA",
            fixed_rate=4,
            ex_div=0,
            calendar="ldn",
            index_base=95.0,
            index_fixings=i_fixings,
            index_method="daily",
        )
        result = bond.index_ratio(settlement=dt(2022, 4, 15), curve=i_curve)
        assert abs(result-expected) < 1e-5

    def test_index_ratio_raises_float_index_fixings(self):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_lag=3,
            index_base=110.0,
        )
        bond = IndexFixedRateBond(
            dt(2022, 1, 1),
            "9m",
            "Q",
            convention="ActActICMA",
            fixed_rate=4,
            ex_div=0,
            calendar="ldn",
            index_base=95.0,
            index_fixings=[100.0, 200.0],
            index_method="daily",
        )
        with pytest.raises(ValueError, match="Must provide `index_fixings` as a Seri"):
            bond.index_ratio(settlement=dt(2022, 4, 15), curve=i_curve)

    def test_fixed_rate_bond_npv_private(self):
        # this test shadows 'fixed_rate_bond_npv' but extends it for projection
        curve = Curve(
            {dt(2004, 11, 25): 1.0, dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 0.75}
        )
        index_curve = IndexCurve(
            {dt(2004, 11, 25): 1.0, dt(2034, 1, 1): 1.0}, index_base=100.0
        )
        gilt = IndexFixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            settle=0,
            index_base=50.0,
            index_lag=3,
            index_method="daily",
        )
        result = gilt._npv_local(
            index_curve, curve, None, None, dt(2010, 11, 26), dt(2010, 11, 25)
        )
        expected = 109.229489312983 * 2.0 # npv should match associated test
        assert abs(result - expected) < 1e-6

    def test_index_base_forecast(self, curve):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_lag=3,
            index_base=95.0,
        )
        bond = IndexFixedRateBond(
            dt(2022, 1, 1),
            "9m",
            "Q",
            convention="ActActICMA",
            fixed_rate=4,
            ex_div=0,
            calendar=None,
            index_method="daily",
            settle=0,
        )
        cashflows = bond.cashflows([i_curve, curve])
        for i in range(4):
            assert cashflows.iloc[i]["Index Base"] == 95.0

        result = bond.npv([i_curve, curve])
        expected = -1006875.3812
        assert abs(result - expected) < 1e-4

        result = bond.rate([i_curve, curve], metric="index_dirty_price")
        assert abs(result * -1e4 - expected) < 1e-4

    def test_fixed_rate_bond_fwd_rate(self):
        gilt = IndexFixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            settle=0,
            index_base=50.0
        )
        curve = Curve({dt(1998, 12, 9): 1.0, dt(2015, 12, 7): 0.50})
        i_curve = IndexCurve(
            {dt(1998, 12, 9): 1.0, dt(2015, 12, 7): 1.0},
            index_base=100.0
        )
        clean_price = gilt.rate([i_curve, curve], metric="clean_price")
        index_clean_price = gilt.rate([i_curve, curve], metric="index_clean_price")
        assert abs(index_clean_price * 0.5 - clean_price) < 1e-3

        result = gilt.rate(
            [i_curve, curve],
            metric="fwd_clean_price",
            forward_settlement=dt(1998, 12, 9)
        )
        assert abs(result - clean_price) < 1e-8
        result = gilt.rate(
            [i_curve, curve],
            metric="fwd_index_clean_price",
            forward_settlement=dt(1998, 12, 9)
        )
        assert abs(result * 0.5 - clean_price) < 1e-8

        result = gilt.rate([i_curve, curve], metric="dirty_price")
        expected = clean_price + gilt.accrued(dt(1998, 12, 9))
        assert result == expected
        result = gilt.rate(
            [i_curve, curve],
            metric="fwd_dirty_price",
            forward_settlement=dt(1998, 12, 9)
        )
        assert abs(result - clean_price - gilt.accrued(dt(1998, 12, 9))) < 1e-8
        result = gilt.rate(
            [i_curve, curve],
            metric="fwd_index_dirty_price",
            forward_settlement=dt(1998, 12, 9)
        )
        assert abs(result * 0.5 - clean_price - gilt.accrued(dt(1998, 12, 9))) < 1e-8

        result = gilt.rate([i_curve, curve], metric="ytm")
        expected = gilt.ytm(clean_price, dt(1998, 12, 9), False)
        assert abs(result - expected) < 1e-8

    def test_fwd_from_repo(self):
        assert False

    def test_repo_from_fwd(self):
        assert False

    def test_duration(self):
        assert False

    def test_convexity(self):
        assert False


class TestBill:
    def test_bill_discount_rate(self):
        # test pricing functions against Treasury Bill Example from US Treasury
        bill = Bill(
            effective=dt(2004, 1, 22),
            termination=dt(2004, 2, 19),
            frequency="A",
            calendar="nyc",
            currency="usd",
            convention="Act360",
        )

        assert bill.discount_rate(99.93777, dt(2004, 1, 22)) == 0.8000999999999543
        assert bill.price(0.800, dt(2004, 1, 22)) == 99.93777777777778
        # this YTM is equivalent to the FixedRateBond ytm with coupon of 0.0
        assert abs(bill.ytm(99.937778, dt(2004, 1, 22)) - 0.8034566609543146) < 1e-9

        d = dcf(dt(2004, 1, 22), dt(2004, 2, 19), "Act360")
        expected = (
            100 * (1 / (1 - 0.0080009999999 * d) - 1) / d
        )  # floating point truncation
        expected = 100 * (100 / 99.93777777777778 - 1) / d
        result = bill.simple_rate(99.93777777777778, dt(2004, 1, 22))
        assert abs(result - expected) < 1e-6

    def test_bill_rate(self):
        curve = Curve({dt(2004, 1, 22): 1.00, dt(2005, 1, 22): 0.992})

        bill = Bill(
            effective=dt(2004, 1, 22),
            termination=dt(2004, 2, 19),
            frequency="A",
            calendar="nyc",
            currency="usd",
            convention="Act360",
            settle=0,
        )

        result = bill.rate(curve, metric="price")
        expected = 99.9385705675
        assert abs(result - expected) < 1e-6

        result = bill.rate(curve, metric="discount_rate")
        expected = bill.discount_rate(99.9385705675, dt(2004, 1, 22))
        assert abs(result - expected) < 1e-6

        result = bill.rate(curve, metric="simple_rate")
        expected = bill.simple_rate(99.9385705675, dt(2004, 1, 22))
        assert abs(result - expected) < 1e-6

        result = bill.rate(curve, metric="ytm")
        expected = bill.ytm(99.9385705675, dt(2004, 1, 22))
        assert abs(result - expected) < 1e-6

        bill.settle = 2  # set the bill to T+2 settlement and re-run the calculations

        result = bill.rate(curve, metric="price")
        expected = 99.94734388985547
        assert abs(result - expected) < 1e-6

        result = bill.rate(curve, metric="discount_rate")
        expected = bill.discount_rate(99.94734388985547, dt(2004, 1, 26))
        assert abs(result - expected) < 1e-6

        result = bill.rate(curve, metric="simple_rate")
        expected = bill.simple_rate(99.94734388985547, dt(2004, 1, 26))
        assert abs(result - expected) < 1e-6

        result = bill.rate(curve, metric="ytm")
        expected = bill.ytm(99.94734388985547, dt(2004, 1, 26))
        assert abs(result - expected) < 1e-6

    def test_bill_rate_raises(self):
        curve = Curve({dt(2004, 1, 22): 1.00, dt(2005, 1, 22): 0.992})

        bill = Bill(
            effective=dt(2004, 1, 22),
            termination=dt(2004, 2, 19),
            frequency="A",
            calendar="nyc",
            currency="usd",
            convention="Act360",
        )

        with pytest.raises(ValueError, match="`metric` must be in"):
            bill.rate(curve, metric="bad vibes")


class TestFloatRateBond:
    @pytest.mark.parametrize(
        "curve_spd, method, float_spd, expected",
        [
            (10, None, 0, 10.055032859883),
            (500, None, 0, 508.93107035125325),
            (-200, None, 0, -200.053341848676),
            (10, "isda_compounding", 0, 10.00000120),
            (500, "isda_compounding", 0, 500.050371345),
            (-200, "isda_compounding", 0, -200.003309580533),
            (10, None, 25, 10.055032859883),
            (500, None, 250, 508.93107035125325),
            (10, "isda_compounding", 25, 10.00000120),
            (500, "isda_compounding", 250, 500.00635330533544),
            (10, None, -25, 10.055032859883),
            (500, None, -250, 508.93107035125325),
            (10, "isda_compounding", -25, 10.00000120),
            (500, "isda_compounding", -250, 500.16850637415),
        ],
    )
    def test_float_rate_bond_rate_spread(self, curve_spd, method, float_spd, expected):
        """
        When a DF curve is shifted it bumps daily rates.
        But under the "none_simple" compounding method this does not compound daily
        therefore the `float_spread` should be slightly higher than the bumped curve.
        When the method is "isda_compounding" this closely matches the bumping method
        of the curve.
        """

        bond = FloatRateBond(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=0,
            settle=0,
            float_spread=float_spd,
            spread_compound_method=method,
        )
        curve = Curve({dt(2007, 1, 1): 1.0, dt(2017, 1, 1): 0.9}, convention="Act365f")
        disc_curve = curve.shift(curve_spd)
        result = bond.rate([curve, disc_curve], metric="spread")
        assert abs(result - expected) < 1e-4

        bond.float_spread = result
        validate = bond.npv([curve, disc_curve])
        assert abs(validate + bond.leg1.notional) < 0.30 * abs(curve_spd)

    @pytest.mark.parametrize(
        "curve_spd, method, float_spd, expected",
        [
            (10, "isda_compounding", 0, 10.00000120),
        ],
    )
    def test_float_rate_bond_rate_spread_fx(
        self, curve_spd, method, float_spd, expected
    ):
        bond = FloatRateBond(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=0,
            settle=0,
            float_spread=float_spd,
            spread_compound_method=method,
        )
        curve = Curve({dt(2007, 1, 1): 1.0, dt(2017, 1, 1): 0.9}, convention="Act365f")
        disc_curve = curve.shift(curve_spd)
        fxr = FXRates({"usdnok": 10.0}, settlement=dt(2007, 1, 1))
        result = bond.rate(
            [curve, disc_curve],
            metric="spread",
            fx=fxr,
        )
        assert abs(result - expected) < 1e-4

        bond.float_spread = result
        validate = bond.npv([curve, disc_curve], fx=fxr)
        assert abs(validate + bond.leg1.notional) < 0.30 * abs(curve_spd)

    def test_float_rate_bond_accrued(self):
        fixings = Series(2.0, index=date_range(dt(2009, 12, 1), dt(2010, 3, 1)))
        bond = FloatRateBond(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=100,
            fixing_method="rfr_observation_shift",
            fixings=fixings,
            method_param=5,
            spread_compound_method="none_simple",
        )
        result = bond.accrued(dt(2010, 3, 3))
        expected = 0.5019275497883  # approx 2 / 12 * 3%
        assert abs(result - expected) < 1e-8

    @pytest.mark.parametrize(
        "metric, spd, exp",
        [
            ("clean_price", 0.0, 100.0),
            ("dirty_price", 0.0, 100.0),
            ("clean_price", 10.0, 99.99982764447981),  # compounding diff between shift
            ("dirty_price", 10.0, 100.0165399732469),
        ],
    )
    def test_float_rate_bond_rate_metric(self, metric, spd, exp):
        fixings = Series(0.0, index=date_range(dt(2009, 12, 1), dt(2010, 3, 1)))
        bond = FloatRateBond(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=spd,
            fixing_method="rfr_observation_shift",
            fixings=fixings,
            method_param=5,
            spread_compound_method="none_simple",
            settle=2,
        )
        curve = Curve({dt(2010, 3, 1): 1.0, dt(2017, 1, 1): 1.0}, convention="act365f")
        disc_curve = curve.shift(spd)

        result = bond.rate(curves=[curve, disc_curve], metric=metric)
        assert abs(result - exp) < 1e-8

    @pytest.mark.parametrize(
        "settlement, expected",
        [(dt(2010, 3, 3), 0.501369863013698), (dt(2010, 12, 30), -0.005479452054)],
    )
    def test_float_rate_bond_accrued_ibor(self, settlement, expected):
        fixings = Series(2.0, index=date_range(dt(2009, 12, 1), dt(2010, 3, 1)))
        bond = FloatRateBond(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=100,
            fixing_method="ibor",
            fixings=fixings,
            method_param=2,
            spread_compound_method="none_simple",
        )
        result = bond.accrued(settlement)
        assert abs(result - expected) < 1e-8

    def test_float_rate_bond_raise_frequency(self):
        with pytest.raises(ValueError, match="FloatRateBond `frequency`"):
            bond = FloatRateBond(
                effective=dt(2007, 1, 1),
                termination=dt(2017, 1, 1),
                frequency="Z",
                convention="Act365f",
                ex_div=3,
                float_spread=100,
                fixing_method="rfr_observation_shift",
                fixings=None,
                method_param=5,
                spread_compound_method="none_simple",
            )

    @pytest.mark.parametrize(
        "fixings",
        [
            Series(2.0, index=date_range(dt(2009, 12, 1), dt(2010, 3, 8))),
            [2.0, [2.0, 2.0]],
        ],
    )
    def test_negative_accrued_needs_forecasting(self, fixings):
        bond = FloatRateBond(
            effective=dt(2009, 9, 16),
            termination=dt(2017, 3, 16),
            frequency="Q",
            convention="Act365f",
            ex_div=5,
            float_spread=0,
            fixing_method="rfr_observation_shift",
            fixings=fixings,
            method_param=5,
            spread_compound_method="none_simple",
            calendar=None,
        )
        result = bond.accrued(dt(2010, 3, 11))

        # approximate calculation 5 days of negative accrued at 2% = -0.027397
        assert abs(result + 0.027397) < 1e-3

    @pytest.mark.parametrize(
        "fixings",
        [
            None,
            [2.0, 2.0],
        ],
    )
    def test_negative_accrued_raises(self, fixings):
        bond = FloatRateBond(
            effective=dt(2009, 9, 16),
            termination=dt(2017, 3, 16),
            frequency="Q",
            convention="Act365f",
            ex_div=5,
            float_spread=0,
            fixing_method="rfr_observation_shift",
            fixings=fixings,
            method_param=5,
            spread_compound_method="none_simple",
            calendar=None,
        )
        with pytest.raises(TypeError, match="`fixings` are not available for RFR"):
            result = bond.accrued(dt(2010, 3, 11))

        with pytest.raises(ValueError, match="For RFR FRNs `ex_div` must be less than"):
            bond = FloatRateBond(
                effective=dt(2009, 9, 16),
                termination=dt(2017, 3, 16),
                frequency="Q",
                ex_div=4,
                fixing_method="rfr_observation_shift",
                method_param=3,
            )

    def test_accrued_no_fixings_in_period(self):
        bond = FloatRateBond(
            effective=dt(2010, 3, 16),
            termination=dt(2017, 3, 16),
            frequency="Q",
            convention="Act365f",
            ex_div=0,
            float_spread=0,
            fixing_method="rfr_observation_shift",
            fixings=None,
            method_param=0,
            spread_compound_method="none_simple",
            calendar=None,
        )
        result = bond.accrued(dt(2010, 3, 16))
        assert result == 0.0

    def test_float_rate_bond_analytic_delta(self):
        frn = FloatRateBond(
            effective=dt(2010, 6, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            float_spread=100,
            notional=-1000000,
            settle=0,
            fixing_method="ibor",
            fixings=2.0,
        )
        curve = Curve({dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 1.0})
        result = frn.analytic_delta(curve)
        expected = -550.0
        assert abs(result - expected) < 1e-6

        frn.settle = 1
        result = frn.analytic_delta(curve)  # bond is ex div on settle 26th Nov 2010
        expected = -500.0  # bond has dropped a 6m coupon payment
        assert abs(result - expected) < 1e-6

    @pytest.mark.parametrize(
        "metric, spd, exp",
        [
            ("fwd_clean_price", 0.0, 100),
            ("fwd_dirty_price", 0.0, 100),
            ("fwd_clean_price", 50.0, 99.99602155150806),
            ("fwd_dirty_price", 50.0, 100.03848730493272),
        ],
    )
    def test_float_rate_bond_forward_prices(self, metric, spd, exp):
        bond = FloatRateBond(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=spd,
            fixing_method="rfr_observation_shift",
            method_param=5,
            spread_compound_method="none_simple",
            settle=2,
        )
        curve = Curve({dt(2010, 3, 1): 1.0, dt(2017, 1, 1): 1.0}, convention="act365f")
        disc_curve = curve.shift(spd)

        result = bond.rate(
            curves=[curve, disc_curve], metric=metric, forward_settlement=dt(2010, 8, 1)
        )
        assert abs(result - exp) < 1e-8

    def test_float_rate_bond_forward_accrued(self):
        bond = FloatRateBond(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=0,
            fixing_method="rfr_observation_shift",
            method_param=5,
            spread_compound_method="none_simple",
            settle=2,
        )
        curve = Curve({dt(2010, 3, 1): 1.0, dt(2017, 1, 1): 0.9}, convention="act365f")
        disc_curve = curve.shift(0)
        result = bond.accrued(dt(2010, 8, 1), forecast=True, curve=curve)
        expected = 0.13083715795372267
        assert abs(result - expected) < 1e-8

    def test_rate_raises(self):
        bond = FloatRateBond(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=0.0,
            fixing_method="rfr_observation_shift",
            method_param=5,
            spread_compound_method="none_simple",
            settle=2,
        )
        with pytest.raises(ValueError, match="`forward_settlement` needed to "):
            bond.rate(None, metric="fwd_clean_price", forward_settlement=None)

        with pytest.raises(ValueError, match="`metric` must be in"):
            bond.rate(None, metric="BAD")

    def test_forecast_ibor(self, curve):
        f_curve = LineCurve({
            dt(2022, 1, 1): 3.0,
            dt(2022, 2, 1): 4.0
        })
        frn = FloatRateBond(
            effective=dt(2022, 2, 1),
            termination="3m",
            frequency="Q",
            fixing_method="ibor",
            method_param=0,
        )
        result = frn.accrued(dt(2022, 2, 5), forecast=True, curve=f_curve)
        expected = 0.044444444
        assert abs(result - expected) < 1e-4


class TestBondFuture:
    @pytest.mark.parametrize(
        "delivery, mat, coupon, exp",
        [
            (dt(2023, 6, 12), dt(2032, 2, 15), 0.0, 0.603058),
            (dt(2023, 6, 12), dt(2032, 8, 15), 1.7, 0.703125),
            (dt(2023, 6, 12), dt(2033, 2, 15), 2.3, 0.733943),
            (dt(2023, 9, 11), dt(2032, 8, 15), 1.7, 0.709321),
            (dt(2023, 9, 11), dt(2033, 2, 15), 2.3, 0.739087),
            (dt(2023, 12, 11), dt(2032, 8, 15), 1.7, 0.715464),
            (dt(2023, 12, 11), dt(2033, 2, 15), 2.3, 0.744390),
        ],
    )
    def test_conversion_factors_eurex_bund(self, delivery, mat, coupon, exp):
        # The expected results are downloaded from the EUREX website
        # regarding precalculated conversion factors.
        # this test allows for an error in the cf < 1e-4.
        kwargs = dict(
            effective=dt(2020, 1, 1),
            stub="ShortFront",
            frequency="A",
            calendar="tgt",
            currency="eur",
            convention="ActActICMA",
        )
        bond1 = FixedRateBond(termination=mat, fixed_rate=coupon, **kwargs)

        fut = BondFuture(delivery=delivery, coupon=6.0, basket=[bond1])
        result = fut.cfs
        assert abs(result[0] - exp) < 1e-4

    @pytest.mark.parametrize(
        "mat, coupon, exp",
        [
            (dt(2032, 6, 7), 4.25, 1.0187757),
            (dt(2033, 7, 31), 0.875, 0.7410593),
            (dt(2034, 9, 7), 4.5, 1.0449380),
            (dt(2035, 7, 31), 0.625, 0.6773884),
            (dt(2036, 3, 7), 4.25, 1.0247516),
        ],
    )
    def test_conversion_factors_ice_gilt(self, mat, coupon, exp):
        # The expected results are downloaded from the ICE LIFFE website
        # regarding precalculated conversion factors.
        # this test allows for an error in the cf < 1e-6.
        kwargs = dict(
            effective=dt(2020, 1, 1),
            stub="ShortFront",
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
        )
        bond1 = FixedRateBond(termination=mat, fixed_rate=coupon, **kwargs)

        fut = BondFuture(
            delivery=(dt(2023, 6, 1), dt(2023, 6, 30)), coupon=4.0, basket=[bond1]
        )
        result = fut.cfs
        assert abs(result[0] - exp) < 1e-6

    def test_dlv_screen_print(self):
        kws = dict(ex_div=7, frequency="S", convention="ActActICMA", calendar=None)
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(
            delivery=(dt(2000, 6, 1), dt(2000, 6, 30)), coupon=7.0, basket=bonds
        )
        result = future.dlv(
            future_price=112.98,
            prices=[102.732, 131.461, 107.877, 134.455],
            repo_rate=6.24,
            settlement=dt(2000, 3, 16),
            convention="Act365f",
        )
        expected = DataFrame(
            {
                "Bond": [
                    "5.750% 07-12-2009",
                    "9.000% 12-07-2011",
                    "6.250% 25-11-2010",
                    "9.000% 06-08-2012",
                ],
                "Price": [102.732, 131.461, 107.877, 134.455],
                "YTM": [5.384243, 5.273217, 5.275481, 5.193851],
                "C.Factor": [0.914225, 1.152571, 0.944931, 1.161956],
                "Gross Basis": [-0.557192, 1.243582, 1.118677, 3.177230],
                "Implied Repo": [7.381345, 3.564685, 2.199755, -1.414670],
                "Actual Repo": [6.24, 6.24, 6.24, 6.24],
                "Net Basis": [-0.343654, 1.033668, 1.275866, 3.010371],
            }
        )
        assert_frame_equal(result, expected)

        result2 = future.dlv(
            future_price=112.98,
            prices=[102.732, 131.461, 107.877, 134.455],
            repo_rate=[6.24, 6.24, 6.24, 6.24],  # test individual repo input
            settlement=dt(2000, 3, 16),
            convention="Act365f",
        )
        assert_frame_equal(result2, expected)

    def test_notional(self):
        future = BondFuture(
            coupon=0, delivery=dt(2000, 6, 1), basket=[], nominal=100000, contracts=10
        )
        assert future.notional == -1e6

    def test_dirty_in_methods(self):
        kws = dict(ex_div=7, frequency="S", convention="ActActICMA", calendar=None)
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(
            delivery=(dt(2000, 6, 1), dt(2000, 6, 30)), coupon=7.0, basket=bonds
        )
        prices = [102.732, 131.461, 107.877, 134.455]
        dirty_prices = [
            price + future.basket[i].accrued(dt(2000, 3, 16))
            for i, price in enumerate(prices)
        ]
        result = future.gross_basis(112.98, dirty_prices, dt(2000, 3, 16), True)
        expected = future.gross_basis(112.98, prices, dt(2000, 3, 16), False)
        assert result == expected

    def test_delivery_in_methods(self):
        kws = dict(ex_div=7, frequency="S", convention="ActActICMA", calendar=None)
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(
            delivery=(dt(2000, 6, 1), dt(2000, 6, 30)), coupon=7.0, basket=bonds
        )
        prices = [102.732, 131.461, 107.877, 134.455]
        expected = future.net_basis(112.98, prices, 6.24, dt(2000, 3, 16))
        result = future.net_basis(
            112.98, prices, 6.24, dt(2000, 3, 16), delivery=dt(2000, 6, 30)
        )
        assert result == expected

        expected = future.implied_repo(112.98, prices, dt(2000, 3, 16))
        result = future.implied_repo(
            112.98, prices, dt(2000, 3, 16), delivery=dt(2000, 6, 30)
        )
        assert result == expected

        expected = future.ytm(112.98)
        result = future.ytm(112.98, delivery=dt(2000, 6, 30))
        assert result == expected

        expected = future.duration(112.98)
        result = future.duration(112.98, delivery=dt(2000, 6, 30))
        assert result == expected

    def test_ctd_index(self):
        kws = dict(ex_div=7, frequency="S", convention="ActActICMA", calendar=None)
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(
            delivery=(dt(2000, 6, 1), dt(2000, 6, 30)), coupon=7.0, basket=bonds
        )
        prices = [102.732, 131.461, 107.877, 134.455]
        assert future.ctd_index(112.98, prices, dt(2000, 3, 16)) == 0

    @pytest.mark.parametrize(
        "metric, expected", [("future_price", 112.98), ("ytm", 5.301975)]
    )
    @pytest.mark.parametrize("delivery", [None, dt(2000, 6, 30)])
    def test_futures_rates(self, metric, expected, delivery):
        curve = Curve(
            nodes={
                dt(2000, 3, 15): 1.0,
                dt(2000, 6, 30): 1.0,
                dt(2009, 12, 7): 1.0,
                dt(2010, 11, 25): 1.0,
                dt(2011, 7, 12): 1.0,
                dt(2012, 8, 6): 1.0,
            },
            id="gilt_curve",
            convention="act365f",
        )
        kws = dict(
            ex_div=7,
            frequency="S",
            convention="ActActICMA",
            calendar=None,
            settle=1,
            curves="gilt_curve",
        )
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        solver = Solver(
            curves=[curve],
            instruments=[
                IRS(
                    dt(2000, 3, 15),
                    dt(2000, 6, 30),
                    "A",
                    convention="act365f",
                    curves="gilt_curve",
                )
            ]
            + bonds,
            s=[7.381345, 102.732, 131.461, 107.877, 134.455],
        )  # note the repo rate as defined by 'gilt_curve' is set to analogue implied
        future = BondFuture(
            coupon=7.0,
            delivery=(dt(2000, 6, 1), dt(2000, 6, 30)),
            basket=bonds,
        )
        result = future.rate(None, solver, metric=metric, delivery=delivery)
        assert abs(result - expected) < 1e-3

    def test_future_rate_raises(self):
        kws = dict(
            ex_div=7,
            frequency="S",
            convention="ActActICMA",
            calendar=None,
            settle=1,
            curves="gilt_curve",
        )
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(
            coupon=7.0,
            delivery=(dt(2000, 6, 1), dt(2000, 6, 30)),
            basket=bonds,
        )
        with pytest.raises(ValueError, match="`metric`"):
            result = future.rate(metric="badstr")

    def test_futures_npv(self):
        curve = Curve(
            nodes={
                dt(2000, 3, 15): 1.0,
                dt(2000, 6, 30): 1.0,
                dt(2009, 12, 7): 1.0,
                dt(2010, 11, 25): 1.0,
                dt(2011, 7, 12): 1.0,
                dt(2012, 8, 6): 1.0,
            },
            id="gilt_curve",
            convention="act365f",
        )
        kws = dict(
            ex_div=7,
            frequency="S",
            convention="ActActICMA",
            calendar=None,
            settle=1,
            curves="gilt_curve",
            currency="gbp",
        )
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        solver = Solver(
            curves=[curve],
            instruments=[
                IRS(
                    dt(2000, 3, 15),
                    dt(2000, 6, 30),
                    "A",
                    convention="act365f",
                    curves="gilt_curve",
                )
            ]
            + bonds,
            s=[7.381345, 102.732, 131.461, 107.877, 134.455],
        )  # note the repo rate as defined by 'gilt_curve' is set to analogue implied
        future = BondFuture(
            coupon=7.0,
            delivery=(dt(2000, 6, 1), dt(2000, 6, 30)),
            basket=bonds,
            nominal=100000,
            contracts=10,
            currency="gbp",
        )
        result = future.npv(None, solver, local=False)
        expected = 1129798.770872
        assert abs(result - expected) < 1e-5

        result2 = future.npv(None, solver, local=True)
        assert abs(result2["gbp"] - expected) < 1e-5

    @pytest.mark.parametrize("delivery", [None, dt(2000, 6, 30)])
    def test_futures_duration_and_convexity(self, delivery):
        kws = dict(
            ex_div=7,
            frequency="S",
            convention="ActActICMA",
            calendar=None,
            settle=1,
            curves="gilt_curve",
        )
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(
            coupon=7.0,
            delivery=(dt(2000, 6, 1), dt(2000, 6, 30)),
            basket=bonds,
        )
        result = future.duration(112.98, delivery=delivery)[0]
        expected = 8.20178546111
        assert abs(result - expected) < 1e-3

        expected = (
            future.duration(112.98, delivery=delivery)[0]
            - future.duration(112.98 - result / 100, delivery=delivery)[0]
        )
        result2 = future.convexity(112.98, delivery=delivery)[0]
        assert abs(result2 - expected * 100) < 1e-3

        # Bond future duration which is not risk is not adjusted by CFs
        result = future.duration(112.98, delivery=delivery, metric="modified")[0]
        expected = 7.23419455163
        assert abs(result - expected) < 1e-3


class TestPricingMechanism:
    def test_value(self, curve):
        ob = Value(dt(2022, 1, 28), curves=curve)
        ob.rate()

    def test_irs(self, curve):
        ob = IRS(dt(2022, 1, 28), "6m", "Q", curves=curve)
        ob.rate()
        ob.npv()
        ob.cashflows()
        ob.spread()

    def test_iirs(self, curve):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_base=100.0,
        )
        ob = IIRS(dt(2022, 1, 28), "6m", "Q", curves=[i_curve, curve, curve, curve])
        ob.rate()
        ob.npv()
        ob.cashflows()
        ob.spread()

    def test_sbs(self, curve):
        ob = SBS(dt(2022, 1, 28), "6m", "Q", curves=curve)
        ob.rate()
        ob.npv()
        ob.cashflows()
        ob.spread()

    def test_fra(self, curve):
        ob = FRA(dt(2022, 1, 28), "6m", "S", curves=curve)
        ob.rate()
        ob.npv()
        ob.cashflows()

    @pytest.mark.parametrize(
        "klass, kwargs",
        [
            (NonMtmXCS, {}),
            (NonMtmFixedFloatXCS, {"fixed_rate": 2.0}),
            (NonMtmFixedFixedXCS, {"fixed_rate": 2.0}),
            (XCS, {}),
            (FixedFloatXCS, {"fixed_rate": 2.0}),
            (FloatFixedXCS, {}),
            (FixedFixedXCS, {"fixed_rate": 2.0}),
        ],
    )
    def test_allxcs(self, klass, kwargs, curve, curve2):
        ob = klass(
            dt(2022, 1, 28),
            "6m",
            "S",
            currency="usd",
            leg2_currency="eur",
            curves=[curve, None, curve2, None],
            **kwargs,
        )
        fxf = FXForwards(
            FXRates({"eurusd": 1.1}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "eurusd": curve2, "eureur": curve2},
        )
        ob.rate(leg=2, fx=fxf)
        ob.npv(fx=fxf)
        ob.cashflows(fx=fxf)

    def test_zcs(self, curve):
        ob = ZCS(dt(2022, 1, 28), "6m", "S", curves=curve)
        ob.rate()
        ob.npv()
        ob.cashflows()

    def test_zcis(self, curve):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_base=100.0,
        )
        ob = ZCIS(dt(2022, 1, 28), "6m", "S", curves=[curve, curve, i_curve, curve])
        ob.rate()
        ob.npv()
        ob.cashflows()


class TestPortfolio:
    def test_portfolio_npv(self, curve):
        irs1 = IRS(dt(2022, 1, 1), "6m", "Q", fixed_rate=1.0, curves=curve)
        irs2 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=2.0, curves=curve)
        pf = Portfolio([irs1, irs2])
        assert pf.npv() == irs1.npv() + irs2.npv()

        pf = Portfolio([irs1] * 5)
        assert pf.npv() == irs1.npv() * 5

        with default_context("pool", 2):  # also test parallel processing
            result = pf.npv()
            assert result == irs1.npv() * 5

    def test_portfolio_npv_local(self, curve):
        irs1 = IRS(dt(2022, 1, 1), "6m", "Q", fixed_rate=1.0, curves=curve, currency="usd")
        irs2 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=2.0, curves=curve, currency="eur")
        irs3 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=2.0, curves=curve, currency="usd")
        pf = Portfolio([irs1, irs2, irs3])

        result = pf.npv(local=True)
        expected = {
            "usd": 20093.295095887483,
            "eur": 5048.87332403382,
        }
        assert result == expected

        with default_context("pool", 2):  # also test parallel processing
            result = pf.npv(local=True)
            assert result == expected


class TestFly:
    @pytest.mark.parametrize("mechanism", [False, True])
    def test_fly_npv(self, curve, mechanism):
        mechanism = curve if mechanism else None
        inverse = curve if mechanism is None else None
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=mechanism)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=mechanism)
        irs3 = IRS(dt(2022, 1, 1), "5m", "Q", fixed_rate=1.0, curves=mechanism)
        fly = Fly(irs1, irs2, irs3)
        assert fly.npv(inverse) == irs1.npv(inverse) + irs2.npv(inverse) + irs3.npv(
            inverse
        )

    @pytest.mark.parametrize("mechanism", [False, True])
    def test_fly_rate(self, curve, mechanism):
        mechanism = curve if mechanism else None
        inv = curve if mechanism is None else None
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=mechanism)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=mechanism)
        irs3 = IRS(dt(2022, 1, 1), "5m", "Q", fixed_rate=1.0, curves=mechanism)
        fly = Fly(irs1, irs2, irs3)
        assert fly.rate(inv) == -irs1.rate(inv) + 2 * irs2.rate(inv) - irs3.rate(inv)

    def test_fly_cashflows_executes(self, curve):
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=curve)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=curve)
        irs3 = IRS(dt(2022, 1, 1), "5m", "Q", fixed_rate=1.0, curves=curve)
        fly = Fly(irs1, irs2, irs3)
        fly.cashflows()

    def test_local_npv(self, curve):
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=curve, currency="eur")
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=curve, currency="usd")
        irs3 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=curve, currency="gbp")
        fly = Fly(irs1, irs2, irs3)
        result = fly.npv(local=True)
        expected = {
            "eur": 7523.321141258284,
            "usd": 6711.514715925333,
            "gbp": 7523.321141258284,
        }
        assert result == expected


class TestSpread:
    @pytest.mark.parametrize("mechanism", [False, True])
    def test_spread_npv(self, curve, mechanism):
        mechanism = curve if mechanism else None
        inverse = curve if mechanism is None else None
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=mechanism)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=mechanism)
        spd = Spread(irs1, irs2)
        assert spd.npv(inverse) == irs1.npv(inverse) + irs2.npv(inverse)

    @pytest.mark.parametrize("mechanism", [False, True])
    def test_spread_rate(self, curve, mechanism):
        mechanism = curve if mechanism else None
        inverse = curve if mechanism is None else None
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=mechanism)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=mechanism)
        spd = Spread(irs1, irs2)
        assert spd.rate(inverse) == -irs1.rate(inverse) + irs2.rate(inverse)

    def test_spread_cashflows_executes(self, curve):
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=curve)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=curve)
        spd = Spread(irs1, irs2)
        spd.cashflows()

    def test_local_npv(self, curve):
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=curve, currency="eur")
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=curve, currency="usd")
        spd = Spread(irs1, irs2)
        result = spd.npv(local=True)
        expected = {
            "eur": 7523.321141258284,
            "usd": 6711.514715925333,
        }
        assert result == expected


class TestSensitivities:
    def test_sensitivity_raises(self):
        irs = IRS(dt(2022, 1, 1), "6m", "Q")
        with pytest.raises(ValueError, match="`solver` is required"):
            irs.delta()

        with pytest.raises(ValueError, match="`solver` is required"):
            irs.gamma()
