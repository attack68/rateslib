import pytest
from datetime import datetime as dt
from pandas import DataFrame, date_range, Series, Index, MultiIndex
from pandas.testing import assert_frame_equal
import numpy as np

import context
from rateslib import defaults, default_context
from rateslib.instruments import (
    IRS, forward_fx, SBS, FXSwap, NonMtmXCS, FixedRateBond, Bill, Value,
    _get_curve_from_solver, BaseMixin, FloatRateBond, FRA,
    NonMtmFixedFloatXCS, NonMtmFixedFixedXCS, XCS, FixedFloatXCS, FixedFixedXCS, FloatFixedXCS,
    Portfolio, Spread, Fly
)
from rateslib.dual import Dual, Dual2
from rateslib.calendars import dcf
from rateslib.curves import Curve
from rateslib.fx import FXRates, FXForwards
from rateslib.solver import Solver


@pytest.fixture()
def curve():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.99,
        dt(2022, 7, 1): 0.98,
        dt(2022, 10, 1): 0.97
    }
    convention = "Act360"
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture()
def curve2():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.98,
        dt(2022, 7, 1): 0.97,
        dt(2022, 10, 1): 0.95
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
    solver = Solver(
        [curve], inst, [0.975]
    )

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
        {
            "usdusd": usdusd,
            "usdeur": usdeur,
            "eureur": eureur
        }
    )
    solver = Solver(
        [curve], inst, [0.975], fx=fxfs if fxf else None
    ) if solver else None
    curve = curve if crv else None

    a = BaseMixin()
    crv_result, fx_result = a._get_curves_and_fx_maybe_from_solver(
        solver, curve, fx
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

    a = BaseMixin()
    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="tagged")
    inst = [(Value(dt(2023, 1, 1)), ("tagged",), {})]
    solver = Solver([curve], inst, [0.975])

    with pytest.raises(ValueError, match="`curves` must contain Curve, not str, if"):
        a._get_curves_and_fx_maybe_from_solver(None, "tagged", None)

    with pytest.raises(ValueError, match="`curves` must contain str curve `id` s"):
        a._get_curves_and_fx_maybe_from_solver(solver, "bad_id", None)

    with pytest.raises(ValueError, match="Can only supply a maximum of 4 `curves`"):
        a._get_curves_and_fx_maybe_from_solver(solver, ["tagged"] * 5, None)


@pytest.mark.parametrize("num", [1, 2, 3, 4])
def test_get_curves_from_solver_multiply(num):
    from rateslib.solver import Solver

    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="tagged")
    inst = [(Value(dt(2023, 1, 1)), ("tagged",), {})]
    solver = Solver([curve], inst, [0.975])
    a = BaseMixin()
    result, _ = a._get_curves_and_fx_maybe_from_solver(solver, ["tagged"] * num, None)
    assert result == (curve, curve, curve, curve)


class TestIRS:

    @pytest.mark.parametrize("float_spread, fixed_rate, expected", [
        (0, 4.03, 4.03637780),
        (3, 4.03, 4.06637780),
        (0, 5.10, 4.03637780),
    ])
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

    @pytest.mark.parametrize("float_spread, fixed_rate, expected", [
        (0, 4.03, -0.63777963),
        (10, 4.03, -0.63777963),
        (0, 4.01, -2.63777963),
    ])
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

    @pytest.mark.parametrize("float_spread, fixed_rate, expected", [
        # (0, 4.03, -0.6322524949759807),  # note this is the closest solution
        # (200, 4.03, -0.632906212667),  # note this is 0.0007bp inaccurate
        # (500, 4.03, -0.64246185393653),  # note this is 0.0102bp inaccurate
        (0, 4.01, -2.61497625534),
    ])
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
        assert abs(result - expected) < 1e-7

        # irs.float_spread = result
        # validate = irs.npv(curve)
        # assert abs(validate) < 1e-4

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
        assert abs(result) < 1e-9

        irs.fixed_rate = 1.0  # pay fixed low rate implies positive NPV
        assert irs.npv(curve) > 1

        irs.fixed_rate = None  # fixed rate set back to initial
        assert abs(irs.npv(curve)) < 1e-9

        irs.fixed_rate = float(irs.rate(curve))
        irs.leg2_float_spread = 100
        assert irs.npv(curve) > 1

        irs.leg2_float_spread = None
        assert abs(irs.npv(curve)) < 1e-9

    def test_sbs_float_spread_raises(self, curve):
        irs = IRS(dt(2022, 1, 1), "9M", "Q")
        with pytest.raises(AttributeError, match="Cannot set `float_spread`"):
            irs.float_spread = 1.0


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
        expected = DataFrame({
            "Type": ["FloatPeriod", "FloatPeriod"],
            "Period": ["Regular", "Regular"],
            "Spread": [3.0, 0.0],
        }, index=MultiIndex.from_tuples([("leg1", 0), ("leg2", 2)]))
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
        assert abs(result - expected) < 1e-9
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


def test_forward_fx_immediate():
    d_curve = Curve(nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
                    interpolation="log_linear")
    f_curve = Curve(nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.95})
    result = forward_fx(dt(2022, 4, 1), d_curve, f_curve, 10.0)
    assert abs(result - 10.102214) < 1e-6

    result = forward_fx(dt(2022, 1, 1), d_curve, f_curve, 10.0, dt(2022, 1, 1))
    assert abs(result - 10.0) < 1e-6


def test_forward_fx_spot_equivalent():
    d_curve = Curve(nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
                    interpolation="log_linear")
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
            {"usdusd": curve, "eurusd": curve2, "eureur": curve2}
        )

        xcs = NonMtmXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="eur", leg2_currency="usd",
                        payment_lag_exchange=0)
        npv2 = xcs._npv2(curve2, curve2, curve, curve, 1.10)
        npv = xcs.npv([curve2, curve2, curve, curve], None, fxf)
        assert abs(npv) < 1e-9

        xcs = NonMtmXCS(dt(2022, 2, 1), "8M", "M", payment_lag=0, amortization=100e3,
                        currency="eur", leg2_currency="usd",
                        payment_lag_exchange=0)
        npv2 = xcs._npv2(curve2, curve2, curve, curve, 1.10)
        npv = xcs.npv([curve2, curve2, curve, curve], None, fxf)
        assert abs(npv) < 1e-9

    def test_nonmtmxcs_fx_notional(self):
        xcs = NonMtmXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="eur", leg2_currency="usd",
                        payment_lag_exchange=0, fx_fixing=2.0, notional=1e6)
        assert xcs.leg2_notional == -2e6

    @pytest.mark.parametrize("float_spd, compound, expected",[
        (10, "none_simple", 10.160794),
        (100, "none_simple", 101.60794),
        (100, "isda_compounding", 101.00),
    ])
    def test_nonmtmxcs_spread(self, curve, curve2, float_spd, compound, expected):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2}
        )

        xcs = NonMtmXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="nok", leg2_currency="usd",
                        payment_lag_exchange=0, notional=10e6,
                        float_spread=float_spd, leg2_spread_compound_method=compound)

        result = xcs.rate([curve, curve, curve2, curve2], None, fxf, 2)
        assert abs(result - expected) < 1e-4
        alias = xcs.spread([curve, curve, curve2, curve2], None, fxf, 2)
        assert alias == result

        xcs.leg2_float_spread = result
        validate = xcs.npv([curve, curve, curve2, curve2], None, fxf)
        assert abs(validate) < 1e-2
        result2 = xcs.rate([curve, curve, curve2, curve2], None, fxf, 2)
        assert abs(result - result2) < 1e-9

        # reverse legs
        xcs_reverse = NonMtmXCS(dt(2022, 2, 1), "8M", "M",
                                payment_lag=0, currency="usd", leg2_currency="nok",
                                payment_lag_exchange=0, notional=1e6,
                                leg2_float_spread=float_spd,
                                spread_compound_method=compound)
        result = xcs_reverse.rate([curve2, curve2, curve, curve], None, fxf, 1)
        assert abs(result - expected) < 1e-4

    def test_no_fx_raises(self, curve, curve2):
        xcs = NonMtmXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="nok", leg2_currency="usd",
                        payment_lag_exchange=0, notional=10e6)

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
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2}
        )

        xcs = NonMtmXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="nok", leg2_currency="usd",
                        payment_lag_exchange=0, notional=10e6)

        result = xcs.cashflows([curve, curve, curve2, curve2], None, fxf)
        expected = DataFrame({
            "Type": ["Cashflow", "FloatPeriod"],
            "Period": ["Exchange", "Regular"],
            "Ccy": ["NOK", "USD"],
            "Notional": [-10000000, -996734.0252423884],
            "FX Rate": [0.10002256337062124, 1.0],
        }, index=MultiIndex.from_tuples([("leg1", 0), ("leg2", 8)]))
        assert_frame_equal(
            result.loc[[("leg1", 0), ("leg2", 8)], ["Type", "Period", "Ccy", "Notional", "FX Rate"]],
            expected,
        )

    @pytest.mark.parametrize("fix", ["fxr", "fxf", "float", "dual", "dual2"])
    def test_nonmtm_fx_fixing(self, curve, curve2, fix):
        fxr = FXRates({"usdnok": 10}, settlement=dt(2022, 1, 1))
        fxf = FXForwards(fxr, {"usdusd": curve, "nokusd": curve2, "noknok": curve2})
        mapping = {
            "fxr": fxr, "fxf": fxf, "float": 10.0,
            "dual": Dual(10.0, "x"), "dual2": Dual2(10.0, "x")
        }
        xcs = NonMtmXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="nok", leg2_currency="usd",
                        payment_lag_exchange=0, notional=10e6, fx_fixing=mapping[fix])
        assert abs(xcs.npv([curve, curve, curve2, curve2])) < 1e-7


class TestNonMtmFixedFloatXCS:

    @pytest.mark.parametrize("float_spd, compound, expected",[
        (10, "none_simple", 6.70955968),
        (100, "isda_compounding", 7.62137047),
    ])
    def test_nonmtmfixxcs_rate_npv(self, curve, curve2, float_spd, compound, expected):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2}
        )
        xcs = NonMtmFixedFloatXCS(dt(2022, 2, 1), "8M", "M",
            payment_lag=0, currency="nok", leg2_currency="usd",
            payment_lag_exchange=0, notional=10e6,
            leg2_spread_compound_method=compound, leg2_float_spread=float_spd
         )

        result = xcs.rate([curve2, curve2, curve, curve], None, fxf, 1)
        assert abs(result - expected) < 1e-4
        assert abs(xcs.npv([curve2, curve2, curve, curve], None, fxf)) < 1e-6

        xcs.fixed_rate = result  # set the fixed rate and check revalues to zero
        assert abs(xcs.npv([curve2, curve2, curve, curve], None, fxf)) < 1e-6

        irs = IRS(dt(2022, 2, 1), "8M", "M",
            payment_lag=0, currency="nok",
            leg2_spread_compound_method=compound, leg2_float_spread=float_spd)
        validate = irs.rate(curve2)
        assert abs(result - validate) < 1e-2

    def test_nonmtmfixxcs_fx_notional(self):
        xcs = NonMtmFixedFloatXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="eur", leg2_currency="usd",
                        payment_lag_exchange=0, fx_fixing=2.0, notional=1e6)
        assert xcs.leg2_notional == -2e6

    def test_nonmtmfixxcs_no_fx_raises(self, curve, curve2):
        xcs = NonMtmFixedFloatXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="nok", leg2_currency="usd",
                        payment_lag_exchange=0, notional=10e6)

        with pytest.raises(ValueError, match="`fx` is required when `fx_fixing` is"):
            with default_context("no_fx_fixings_for_xcs", "raise"):
                xcs.npv([curve, curve, curve2, curve2])

        with pytest.raises(ValueError, match="`fx` is required when `fx_fixing` is"):
            with default_context("no_fx_fixings_for_xcs", "raise"):
                xcs.cashflows([curve, curve, curve2, curve2])

    def test_nonmtmfixxcs_cashflows(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2}
        )

        xcs = NonMtmFixedFloatXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="nok", leg2_currency="usd",
                        payment_lag_exchange=0, notional=10e6)

        result = xcs.cashflows([curve, curve, curve2, curve2], None, fxf)
        expected = DataFrame({
            "Type": ["Cashflow", "FloatPeriod"],
            "Period": ["Exchange", "Regular"],
            "Ccy": ["NOK", "USD"],
            "Notional": [-10000000, -996734.0252423884],
            "FX Rate": [0.10002256337062124, 1.0],
        }, index=MultiIndex.from_tuples([("leg1", 0), ("leg2", 8)]))
        assert_frame_equal(
            result.loc[[("leg1", 0), ("leg2", 8)], ["Type", "Period", "Ccy", "Notional", "FX Rate"]],
            expected,
        )

    @pytest.mark.parametrize("fix", ["fxr", "fxf", "float", "dual", "dual2"])
    def test_nonmtmfixxcs_fx_fixing(self, curve, curve2, fix):
        fxr = FXRates({"usdnok": 10}, settlement=dt(2022, 1, 1))
        fxf = FXForwards(fxr, {"usdusd": curve, "nokusd": curve2, "noknok": curve2})
        mapping = {
            "fxr": fxr, "fxf": fxf, "float": 10.0,
            "dual": Dual(10.0, "x"), "dual2": Dual2(10.0, "x")
        }
        xcs = NonMtmFixedFloatXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="nok", leg2_currency="usd",
                        payment_lag_exchange=0, notional=10e6, fx_fixing=mapping[fix],
                        leg2_float_spread=10.0)
        assert abs(xcs.npv([curve2, curve2, curve, curve])) < 1e-7

    def test_nonmtmfixxcs_raises(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2}
        )

        xcs = NonMtmFixedFloatXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="nok", leg2_currency="usd",
                        payment_lag_exchange=0, notional=10e6)

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
            "fxr": fxr, "fxf": fxf, "float": 10.0,
            "dual": Dual(10.0, "x"), "dual2": Dual2(10.0, "x")
        }
        xcs = NonMtmFixedFixedXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="nok", leg2_currency="usd",
                        payment_lag_exchange=0, notional=10e6, fx_fixing=mapping[fix],
                        leg2_fixed_rate=2.0,
                        )
        assert abs(xcs.npv([curve2, curve2, curve, curve])) < 1e-7

        xcs = NonMtmFixedFixedXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="nok", leg2_currency="usd",
                        payment_lag_exchange=0, notional=10e6, fx_fixing=mapping[fix],
                        fixed_rate=2.0,
                        )
        assert abs(xcs.npv([curve2, curve2, curve, curve])) < 1e-7

    def test_nonmtmfixfixxcs_raises(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2}
        )

        xcs = NonMtmFixedFixedXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="nok", leg2_currency="usd",
                        payment_lag_exchange=0, notional=10e6)

        with pytest.raises(ValueError, match="Cannot solve for a"):
            xcs.rate([curve, curve, curve2, curve2], None, fxf, leg=2)

        with pytest.raises(AttributeError, match="Cannot set `leg2_float_spread` for"):
            xcs.leg2_float_spread = 2.0


class TestXCS:

    def test_mtmxcs_npv(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"eurusd": 1.1}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "eurusd": curve2, "eureur": curve2}
        )

        xcs = XCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="eur", leg2_currency="usd",
                        payment_lag_exchange=0)

        npv = xcs.npv([curve2, curve2, curve, curve], None, fxf)
        assert abs(npv) < 1e-9

    def test_mtmxcs_cashflows(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2}
        )

        xcs = XCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="nok", leg2_currency="usd",
                        payment_lag_exchange=0, notional=10e6)

        result = xcs.cashflows([curve, curve, curve2, curve2], None, fxf)
        expected = DataFrame({
            "Type": ["Cashflow", "FloatPeriod", "Cashflow"],
            "Period": ["Exchange", "Regular", "Mtm"],
            "Ccy": ["NOK", "USD", "USD"],
            "Notional": [-10000000, -990019.24969, -3509.80082],
            "Rate": [np.nan,  8.181151773810475, 0.09829871161519926],
            "FX Rate": [0.10002256337062124, 1.0, 1.0],
        }, index=MultiIndex.from_tuples([("leg1", 0), ("leg2", 11), ("leg2", 14)]))
        assert_frame_equal(
            result.loc[[("leg1", 0), ("leg2", 11), ("leg2", 14)],
                ["Type", "Period", "Ccy", "Notional", "Rate", "FX Rate"]],
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

    @pytest.mark.parametrize("float_spd, compound, expected", [
        (10, "none_simple", 9.97839804),
        (100, "none_simple", 99.78398037),
        (100, "isda_compounding", 101.00),
    ])
    def test_mtmxcs_rate(self, float_spd, compound, expected, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2}
        )

        xcs = XCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="nok", leg2_currency="usd",
                        payment_lag_exchange=0, notional=10e6,
                        float_spread=float_spd, leg2_spread_compound_method=compound)

        result = xcs.rate([curve2, curve2, curve, curve], None, fxf, 2)
        assert abs(result - expected) < 1e-4
        alias = xcs.spread([curve2, curve2, curve, curve], None, fxf, 2)
        assert alias == result

        xcs.leg2_float_spread = result
        validate = xcs.npv([curve2, curve2, curve, curve], None, fxf)
        assert abs(validate) < 1e-2
        result2 = xcs.rate([curve2, curve2, curve, curve], None, fxf, 2)
        assert abs(result - result2) < 1e-9


class TestFixedFloatXCS:

    def test_mtmfixxcs_rate(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2}
        )

        xcs = FixedFloatXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="nok", leg2_currency="usd",
                        payment_lag_exchange=0, notional=10e6)

        result = xcs.rate([curve2, curve2, curve, curve], None, fxf, 1)

        irs = IRS(dt(2022, 2, 1), "8M", "M", currency="nok", payment_lag=0)
        validate = irs.rate(curve2)
        assert abs(result - validate) < 1e-4
        # alias = xcs.spread([curve2, curve2, curve, curve], None, fxf, 2)

    def test_mtmfixxcs_rate_reversed(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2}
        )

        xcs = FloatFixedXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="usd", leg2_currency="nok",
                        payment_lag_exchange=0, notional=10e6)

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
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2}
        )

        irs = IRS(dt(2022, 2, 1), "8M", "M", payment_lag=0)
        nok_rate = float(irs.rate(curve2))
        xcs = FixedFixedXCS(dt(2022, 2, 1), "8M", "M",
                        payment_lag=0, currency="nok", leg2_currency="usd",
                        payment_lag_exchange=0, notional=10e6,
                        fixed_rate=nok_rate)
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
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2}
        )
        fxs = FXSwap(dt(2022, 2, 1), "8M", "M",
                            currency="usd", leg2_currency="nok",
                            payment_lag_exchange=0, notional=1e6,
                            )
        expected = fxf.swap("usdnok", [dt(2022, 2, 1), dt(2022, 10, 1)])
        result = fxs.rate([None, curve, None, curve2], None, fxf)
        assert result == expected

    def test_fxswap_npv(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2}
        )
        fxs = FXSwap(dt(2022, 2, 1), "8M", "M",
                            currency="usd", leg2_currency="nok",
                            payment_lag_exchange=0, notional=1e6,
                            )

        assert abs(fxs.npv([None, curve, None, curve2], None, fxf)) < 1e-7

        result = fxs.rate([None, curve, None, curve2], None, fxf, fixed_rate=True)
        fxs.leg2_fixed_rate = result
        assert abs(fxs.npv([None, curve, None, curve2], None, fxf)) < 1e-7

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
        bond = FixedRateBond(dt(1995, 1, 1), dt(2015, 12, 7), "S", convention="ActActICMA",
                             fixed_rate=8, ex_div=7, calendar="ldn")
        assert bond.price(4.445, dt(1999, 5, 24), True) - 145.012268 < 1e-6
        assert bond.price(4.445, dt(1999, 5, 26), True) - 145.047301 < 1e-6
        assert bond.price(4.445, dt(1999, 5, 27), True) - 141.070132 < 1e-6
        assert bond.price(4.445, dt(1999, 6, 7), True) - 141.257676 < 1e-6

        bond = FixedRateBond(dt(1997, 1, 1), dt(2004, 11, 26), "S", convention="ActActICMA",
                             fixed_rate=6.75, ex_div=7, calendar="ldn")
        assert bond.price(4.634, dt(1999, 5, 10), True) - 113.315543 < 1e-6
        assert bond.price(4.634, dt(1999, 5, 17), True) - 113.415969 < 1e-6
        assert bond.price(4.634, dt(1999, 5, 18), True) - 110.058738 < 1e-6
        assert bond.price(4.634, dt(1999, 5, 26), True) - 110.170218 < 1e-6

    def test_fixed_rate_bond_yield(self):
        # test pricing functions against Gilt Example prices from UK DMO
        bond = FixedRateBond(dt(1995, 1, 1), dt(2015, 12, 7), "S", convention="ActActICMA",
                             fixed_rate=8, ex_div=7, calendar="ldn")
        assert bond.ytm(135., dt(1999, 5, 24), True) - 5.1620635 < 1e-6
        assert bond.ytm(135., dt(1999, 5, 26), True) - 5.1649111 < 1e-6
        assert bond.ytm(135., dt(1999, 5, 27), True) - 4.871425 < 1e-6
        assert bond.ytm(135., dt(1999, 6, 7), True) - 4.8856785 < 1e-6

        bond = FixedRateBond(dt(1997, 1, 1), dt(2004, 11, 26), "S", convention="ActActICMA",
                             fixed_rate=6.75, ex_div=7, calendar="ldn")
        assert bond.ytm(108., dt(1999, 5, 10), True) - 5.7009527 < 1e-6
        assert bond.ytm(108., dt(1999, 5, 17), True) - 5.7253361 < 1e-6
        assert bond.ytm(108., dt(1999, 5, 18), True) - 5.0413308 < 1e-6
        assert bond.ytm(108., dt(1999, 5, 26), True) - 5.0652248 < 1e-6

    def test_fixed_rate_bond_ytm_duals(self):
        bond = FixedRateBond(dt(1995, 1, 1), dt(2015, 12, 7), "S", convention="ActActICMA",
                             fixed_rate=8, ex_div=7, calendar="ldn")

        dPdy = bond.duration(4, dt(1995, 1, 1))
        P = bond.price(4, dt(1995, 1, 1))
        result = bond.ytm(Dual(P, ["a", "b"], [1, -0.5]), dt(1995, 1, 1))
        assert result == Dual(4.00, ["a", "b"], [-1 / dPdy, 0.5 / dPdy])

        d2ydP2 = - bond.convexity(4, dt(1995, 1, 1)) * -dPdy ** -3
        result = bond.ytm(Dual2(P, ["a", "b"], [1, -0.5]), dt(1995, 1, 1))
        expected = Dual2(
            4.00,
            ["a", "b"],
            [-1 / dPdy, 0.5 / dPdy],
            0.5 * np.array([[d2ydP2, d2ydP2 * -0.5], [d2ydP2 * -0.5, d2ydP2 * 0.25]])
        )
        assert result == expected

    def test_fixed_rate_bond_accrual(self):
        # test pricing functions against Gilt Example prices from UK DMO, with stub
        bond = FixedRateBond(dt(1999, 5, 7), dt(2002, 12, 7), "S",  convention="ActActICMA",
                             front_stub=dt(1999, 12, 7),
                             fixed_rate=6, ex_div=7, calendar="ldn")
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
        bond = FixedRateBond(dt(1999, 6, 7), dt(2002, 12, 7), "S",
                             convention="ActActICMA",
                             fixed_rate=6, ex_div=7, calendar="ldn")
        regular_ytm = bond.ytm(101, dt(1999, 11, 8), dirty=True)
        bond.leg1.periods[0].stub = True
        stubbed_ytm = bond.ytm(101, dt(1999, 11, 8), dirty=True)
        assert regular_ytm == stubbed_ytm

    def test_fixed_rate_bond_zero_frequency_raises(self):
        with pytest.raises(ValueError, match="FixedRateBond `frequency`"):
            FixedRateBond(dt(1999, 5, 7), dt(2002, 12, 7), "Z",  convention="ActActICMA")

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
            fixed_rate=8.0
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
            fixed_rate=8.0
        )
        numeric = gilt.duration(4.445, dt(1999, 5, 27)) - \
                  gilt.duration(4.446, dt(1999, 5, 27))
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
        curve = Curve({dt(1998, 12, 7): 1.0, dt(2015, 12, 7): 0.50})
        dirty_price = gilt.rate(curve)

        assert (
            gilt.rate(curve, metric="clean_price")
            ==
            dirty_price - gilt.accrued(dt(1998, 12, 7))
        )

        assert (
            gilt.rate(curve, metric="ytm")
            ==
            gilt.ytm(dirty_price, dt(1998, 12, 7), True)
        )

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

    def test_fixed_rate_bond_npv_private(self):
        # this test shadows 'fixed_rate_bond_npv' but extends it for projection
        curve = Curve({
            dt(2004, 11, 25): 1.0,
            dt(2010, 11, 25): 1.0,
            dt(2015, 12, 7): 0.75}
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
                amortization=100
            )

    @pytest.mark.parametrize("f_s, exp", [
        (dt(2001, 12, 31), 99.997513754),  # compounding of mid year coupon
        (dt(2002, 1, 1), 99.9975001688)  # this is now ex div on last coupon
    ])
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
        result = gilt.fwd_from_repo(
            100.0, dt(2001, 1, 1), f_s, 1.0, "act365f"
        )
        assert abs(result - exp) < 1e-6

    @pytest.mark.parametrize("s, f_s, exp", [
        (dt(2010, 11, 25), dt(2011, 11, 25), 99.9975000187),
        (dt(2010, 11, 28), dt(2011, 11, 28), 99.9975000187),
        (dt(2010, 11, 28), dt(2011, 11, 25), 99.997419419),
        (dt(2010, 11, 25), dt(2011, 11, 28), 99.997579958),
    ])
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
        expected = 100 * (1 / (1-0.0080009999999*d) - 1) / d  # floating point truncation
        expected = 100 * (100 / 99.93777777777778 - 1) / d
        result = bill.simple_rate(99.93777777777778, dt(2004, 1, 22))
        assert abs(result - expected) < 1e-6

    def test_bill_rate(self):
        curve = Curve({
            dt(2004, 1, 22): 1.00,
            dt(2005, 1, 22): 0.992
        })

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
        curve = Curve({
            dt(2004, 1, 22): 1.00,
            dt(2005, 1, 22): 0.992
        })

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

    @pytest.mark.parametrize("curve_spd, method, float_spd, expected", [
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
    ])
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
        curve = Curve({
            dt(2007, 1, 1): 1.0,
            dt(2017, 1, 1): 0.9
        }, convention="Act365f"
        )
        disc_curve = curve.shift(curve_spd)
        result = bond.rate(
            [curve, disc_curve],
            metric="spread"
        )
        assert abs(result - expected) < 1e-4

        bond.float_spread = result
        validate = bond.npv([curve, disc_curve])
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

    @pytest.mark.parametrize("fixings", [
        Series(2.0, index=date_range(dt(2009, 12, 1), dt(2010, 3, 8))),
        [2.0, [2.0, 2.0]],
    ])
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
        assert abs(result+0.027397) < 1e-3

    @pytest.mark.parametrize("fixings", [
        None,
        [2.0, 2.0],
    ])
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
        assert result == 0.

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
            fixings=2.0
        )
        curve = Curve({dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 1.0})
        result = frn.analytic_delta(curve)
        expected = -550.0
        assert abs(result - expected) < 1e-6

        frn.settle = 1
        result = frn.analytic_delta(curve)  # bond is ex div on settle 26th Nov 2010
        expected = -500.0  # bond has dropped a 6m coupon payment
        assert abs(result - expected) < 1e-6


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

    @pytest.mark.parametrize("klass, kwargs", [
        (NonMtmXCS, {}),
        (NonMtmFixedFloatXCS, {"fixed_rate": 2.0}),
        (NonMtmFixedFixedXCS, {"fixed_rate": 2.0}),
        (XCS, {}),
        (FixedFloatXCS, {"fixed_rate": 2.0}),
        (FloatFixedXCS, {}),
        (FixedFixedXCS, {"fixed_rate": 2.0}),
    ])
    def test_allxcs(self, klass, kwargs, curve, curve2):
        ob = klass(dt(2022, 1, 28), "6m", "S", currency="usd", leg2_currency="eur",
                   curves=[curve, None, curve2, None], **kwargs)
        fxf = FXForwards(
            FXRates({"eurusd": 1.1}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "eurusd": curve2, "eureur": curve2}
        )
        ob.rate(leg=2, fx=fxf)
        ob.npv(fx=fxf)
        ob.cashflows(fx=fxf)


class TestPortfolio:

    def test_portfolio_npv(self, curve):

        irs1 = IRS(dt(2022, 1, 1), "6m", "Q", fixed_rate=1.0, curves=curve)
        irs2 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=2.0, curves=curve)
        pf = Portfolio([irs1, irs2])
        assert pf.npv() == irs1.npv() + irs2.npv()

        pf = Portfolio([irs1]*5)
        assert pf.npv() == irs1.npv() * 5


class TestFly:

    @pytest.mark.parametrize("mechanism", [False, True])
    def test_fly_npv(self, curve, mechanism):
        mechanism = curve if mechanism else None
        inverse = curve if mechanism is None else None
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=mechanism)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=mechanism)
        irs3 = IRS(dt(2022, 1, 1), "5m", "Q", fixed_rate=1.0, curves=mechanism)
        fly = Fly(irs1, irs2, irs3)
        assert fly.npv(inverse) == irs1.npv(inverse)+irs2.npv(inverse)+irs3.npv(inverse)

    @pytest.mark.parametrize("mechanism", [False, True])
    def test_fly_rate(self, curve, mechanism):
        mechanism = curve if mechanism else None
        inv = curve if mechanism is None else None
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=mechanism)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=mechanism)
        irs3 = IRS(dt(2022, 1, 1), "5m", "Q", fixed_rate=1.0, curves=mechanism)
        fly = Fly(irs1, irs2, irs3)
        assert fly.rate(inv) == -irs1.rate(inv) + 2*irs2.rate(inv) - irs3.rate(inv)

    def test_fly_cashflows_executes(self, curve):
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=curve)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=curve)
        irs3 = IRS(dt(2022, 1, 1), "5m", "Q", fixed_rate=1.0, curves=curve)
        fly = Fly(irs1, irs2, irs3)
        fly.cashflows()


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


class TestSensitivities:

    def test_sensitivity_raises(self):

        irs = IRS(dt(2022, 1, 1), "6m", "Q")
        with pytest.raises(ValueError, match="`solver` is required"):
            irs.delta()

        with pytest.raises(ValueError, match="`solver` is required"):
            irs.gamma()