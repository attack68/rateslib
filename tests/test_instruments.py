import pytest
from datetime import datetime as dt
from pandas import DataFrame, date_range, Series, Index, MultiIndex
from pandas.testing import assert_frame_equal
import numpy as np

import context
from rateslib import defaults, default_context
from rateslib.default import NoInput
from rateslib.instruments import (
    FixedRateBond,
    IndexFixedRateBond,
    FloatRateNote,
    Bill,
    IRS,
    STIRFuture,
    IIRS,
    SBS,
    FXSwap,
    FXExchange,
    Value,
    ZCS,
    ZCIS,
    _get_curve_from_solver,
    FRA,
    XCS,
    Portfolio,
    Spread,
    Fly,
    _get_curves_fx_and_base_maybe_from_solver,
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
    # convention = "Act360"
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


@pytest.fixture()
def simple_solver():
    curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0, dt(2024, 1, 1): 1.0}, id="curve")
    solver = Solver(
        curves=[curve],
        instruments=[
            IRS(dt(2022, 1, 1), "1Y", "A", curves="curve"),
            IRS(dt(2022, 1, 1), "2Y", "A", curves="curve"),
        ],
        s=[2.5, 3.0],
        id="solver",
        instrument_labels=["1Y", "2Y"],
    )
    return solver


class TestCurvesandSolver:

    def test_get_curve_from_solver(self):
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
                _get_curve_from_solver(no_curve, solver)

        with pytest.raises(AttributeError, match="`curve` has no attribute `id`, likely it not"):
            _get_curve_from_solver(100.0, solver)

    @pytest.mark.parametrize("solver", [True, False])
    @pytest.mark.parametrize("fxf", [True, False])
    @pytest.mark.parametrize("fx", [NoInput(0), 2.0])
    @pytest.mark.parametrize("crv", [True, False])
    def test_get_curves_and_fx_from_solver(self, usdusd, usdeur, eureur, solver, fxf, fx, crv):
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="tagged")
        inst = [(Value(dt(2023, 1, 1)), ("tagged",), {})]
        fxfs = FXForwards(
            FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3)),
            {"usdusd": usdusd, "usdeur": usdeur, "eureur": eureur},
        )
        solver = Solver([curve], inst, [0.975], fx=fxfs if fxf else NoInput(0)) if solver else NoInput(0)
        curve = curve if crv else NoInput(0)

        if solver is not NoInput(0) and fxf and fx is not NoInput(0):
            with pytest.warns(UserWarning):
                #  Solver contains an `fx` attribute but an `fx` argument has been supplied
                crv_result, fx_result, _ = _get_curves_fx_and_base_maybe_from_solver(
                    NoInput(0), solver, curve, fx, NoInput(0), "usd"
                )
        else:
            crv_result, fx_result, _ = _get_curves_fx_and_base_maybe_from_solver(
                NoInput(0), solver, curve, fx, NoInput(0), "usd"
            )

        # check the fx results. If fx is specified directly it is returned
        # otherwsie it is returned from a solver object if it is available.
        if fx is not NoInput(0):
            assert fx_result == 2.0
        elif solver is NoInput(0):
            assert fx_result is NoInput(0)
        else:
            if fxf:
                assert fx_result == fxfs
            else:
                assert fx_result is NoInput(0)

        assert crv_result == (curve, curve, curve, curve)

    def test_get_curves_and_fx_from_solver_raises(self):
        from rateslib.solver import Solver

        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="tagged")
        inst = [(Value(dt(2023, 1, 1)), ("tagged",), {})]
        solver = Solver([curve], inst, [0.975])

        with pytest.raises(ValueError, match="`curves` must contain Curve, not str, if"):
            _get_curves_fx_and_base_maybe_from_solver(NoInput(0), NoInput(0), "tagged", NoInput(0), NoInput(0), "")

        with pytest.raises(ValueError, match="`curves` must contain str curve `id` s"):
            _get_curves_fx_and_base_maybe_from_solver(NoInput(0), solver, "bad_id", NoInput(0), NoInput(0), "")

        with pytest.raises(ValueError, match="Can only supply a maximum of 4 `curves`"):
            _get_curves_fx_and_base_maybe_from_solver(NoInput(0), solver, ["tagged"] * 5, NoInput(0), NoInput(0), "")

    @pytest.mark.parametrize("num", [1, 2, 3, 4])
    def test_get_curves_from_solver_multiply(self, num):
        from rateslib.solver import Solver

        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="tagged")
        inst = [(Value(dt(2023, 1, 1)), ("tagged",), {})]
        solver = Solver([curve], inst, [0.975])
        result, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            NoInput(0), solver, ["tagged"] * num, NoInput(0), NoInput(0), ""
        )
        assert result == (curve, curve, curve, curve)

    def test_get_proxy_curve_from_solver(self, usdusd, usdeur, eureur):
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

    def test_ambiguous_curve_in_out_id_solver_raises(self):
        curve = Curve({dt(2022, 1, 1): 1.0}, id="cloned-id")
        curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="cloned-id")
        solver = Solver(
            curves=[curve2],
            instruments=[IRS(dt(2022, 1 ,1), "1y", "A", curves="cloned-id")],
            s=[5.0],
        )
        irs = IRS(dt(2022, 1, 1), "1y", "A", fixed_rate=2.0)
        with pytest.raises(ValueError, match="A curve has been supplied, as part of ``curves``,"):
            irs.npv(curves=curve, solver=solver)


class TestSolverFXandBase:
    """
    Test the npv method with combinations of solver fx and base args.
    """

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        cls.curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.96}, id="curve")
        cls.fxr = FXRates({"eurusd": 1.1, "gbpusd": 1.25}, base="gbp")
        cls.irs = IRS(dt(2022, 2, 1), "6M", "A", curves=cls.curve, fixed_rate=4.0)
        cls.solver = Solver(
            curves=[cls.curve],
            instruments=[IRS(dt(2022, 1, 1), "1y", "A", curves=cls.curve)],
            s=[4.109589041095898],
            id="Solver",
        )
        cls.nxcs = XCS(
            dt(2022, 2, 1),
            "6M",
            "A",
            fixed=False,
            leg2_fixed=False,
            leg2_mtm=False,
            curves=[cls.curve] * 4,
            currency="eur",
            leg2_currency="usd",
            float_spread=2.0,
        )

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to
        setup_class.
        """
        pass

    # ``base`` is explcit

    def test_base_and_fx(self):
        # calculable since base aligns with local currency
        result = self.irs.npv(fx=self.fxr, base="eur")
        expected = 330.4051154763001 / 1.1
        assert abs(result - expected) < 1e-4

        with pytest.warns(UserWarning):
            # warn about numeric
            result = self.irs.npv(fx=1 / 1.1, base="eur")

        # raises because no FX data to calculate a conversion
        with pytest.raises(KeyError, match="'usd'"):
            result = self.irs.npv(fx=FXRates({"eurgbp": 1.1}), base="eur")

    def test_base_and_solverfx(self):
        # should take fx from solver and calculated
        self.solver.fx = FXRates({"eurusd": 1.1})
        result = self.irs.npv(solver=self.solver, base="eur")
        expected = 330.4051154763001 / 1.1
        assert abs(result - expected) < 1e-4
        self.solver.fx = NoInput(0)

    def test_base_and_fx_and_solverfx(self):
        # should take fx and ignore solver.fx
        fxr = FXRates({"eurusd": 1.2})
        self.solver.fx = fxr

        # no warning becuase objects are the same
        result = self.irs.npv(solver=self.solver, base="eur", fx=fxr)
        expected = 330.4051154763001 / 1.2
        assert abs(result - expected) < 1e-4

        # should give warning because obj id are different
        with pytest.warns(UserWarning):
            result = self.irs.npv(solver=self.solver, base="eur", fx=self.fxr)
            expected = 330.4051154763001 / 1.1
            assert abs(result - expected) < 1e-4

        self.solver.fx = NoInput(0)

    def test_base_only(self):
        # calculable since base aligns with local currency
        result = self.irs.npv(base="usd")
        expected = 330.4051154763001
        assert abs(result - expected) < 1e-4

        # raises becuase no FX data to calculate a conversion
        with pytest.raises(ValueError, match="`base` "):
            result = self.irs.npv(base="eur")

    def test_base_solvernofx(self):
        # calculable since base aligns with local currency
        result = self.irs.npv(base="usd", solver=self.solver)
        expected = 330.4051154763001
        assert abs(result - expected) < 1e-4

        # raises becuase no FX data to calculate a conversion
        with pytest.raises(ValueError, match="`base` "):
            result = self.irs.npv(base="eur", solver=self.solver)

    # ``base`` is inferred

    def test_no_args(self):
        # should result in a local NPV calculation
        result = self.irs.npv()
        expected = 330.4051154763001
        assert abs(result - expected) < 1e-4

    def test_fx(self):
        # should repeat the "_just_base" case.
        result = self.irs.npv(fx=self.fxr)
        expected = 330.4051154763001 / 1.25
        assert abs(result - expected) < 1e-4

    def test_fx_solverfx(self):
        fxr = FXRates({"eurusd": 1.2}, base="eur")
        self.solver.fx = fxr

        # no warning becuase objects are the same
        result = self.irs.npv(solver=self.solver, fx=fxr)
        expected = 330.4051154763001 / 1.2
        assert abs(result - expected) < 1e-4

        # should give warning because obj id are different
        with pytest.warns(UserWarning):
            result = self.irs.npv(solver=self.solver, fx=self.fxr)
            expected = 330.4051154763001 / 1.25  # base in this case inferred as GBP
            assert abs(result - expected) < 1e-4

        self.solver.fx = NoInput(0)

    def test_solverfx(self):
        fxr = FXRates({"eurusd": 1.2}, base="eur")
        self.solver.fx = fxr

        # no warning becuase objects are the same
        result = self.irs.npv(solver=self.solver)
        expected = 330.4051154763001  # base in this should be local currency not eur.
        assert abs(result - expected) < 1e-4

        self.solver.fx = NoInput(0)


class TestNullPricing:
    # test instruments can be priced without defining a pricing parameter.

    @pytest.mark.parametrize(
        "inst",
        [
            IRS(dt(2022, 7, 1), "3M", "A", curves="eureur", notional=1e6),
            STIRFuture(dt(2022, 3, 16), dt(2022, 6, 15), "Q", curves="eureur", bp_value=25.0, contracts=-1),
            FRA(dt(2022, 7, 1), "3M", "A", curves="eureur", notional=1e6),
            SBS(
                dt(2022, 7, 1),
                "3M",
                "A",
                curves=["eureur", "eureur", "eurusd", "eureur"],
                notional=-1e6,
            ),
            ZCS(dt(2022, 7, 1), "3M", "A", curves="eureur", notional=1e6),
            IIRS(
                dt(2022, 7, 1),
                "3M",
                "A",
                curves=["eu_cpi", "eureur", "eureur", "eureur"],
                notional=1e6,
            ),
            IIRS(
                dt(2022, 7, 1),
                "3M",
                "A",
                curves=["eu_cpi", "eureur", "eureur", "eureur"],
                notional=1e6,
                notional_exchange=True,
            ),
            # TODO add a null price test for ZCIS
            XCS(  # XCS - FloatFloat
                dt(2022, 7, 1),
                "3M",
                "A",
                currency="usd",
                leg2_currency="eur",
                curves=["usdusd", "usdusd", "eureur", "eurusd"],
                notional=1e6,
            ),
            XCS(  # XCS-FloatFloatNonMtm
                dt(2022, 7, 1),
                "3M",
                "A",
                fixed=False,
                leg2_fixed=False,
                leg2_mtm=False,
                currency="usd",
                leg2_currency="eur",
                curves=["usdusd", "usdusd", "eureur", "eurusd"],
                notional=1e6,
            ),
            XCS( # XCS-FixedFloatNonMtm
                dt(2022, 7, 1),
                "3M",
                "A",
                fixed=True,
                leg2_fixed=False,
                leg2_mtm=False,
                currency="eur",
                leg2_currency="usd",
                curves=["eureur", "eureur", "usdusd", "usdusd"],
                notional=1e6,
            ),
            XCS(  # XCS-FixedFixedNonMtm
                dt(2022, 7, 1),
                "3M",
                "A",
                fixed=True,
                leg2_fixed=True,
                leg2_mtm=False,
                currency="eur",
                leg2_currency="usd",
                fixed_rate=1.2,
                curves=["eureur", "eureur", "usdusd", "usdusd"],
                notional=1e6,
            ),
            XCS(  # XCS - FixedFloat
                dt(2022, 7, 1),
                "3M",
                "A",
                fixed=True,
                leg2_fixed=False,
                leg2_mtm=True,
                currency="eur",
                leg2_currency="usd",
                curves=["eureur", "eureur", "usdusd", "usdusd"],
                notional=1e6,
            ),
            XCS(  # XCS-FixedFixed
                dt(2022, 7, 1),
                "3M",
                "A",
                fixed=True,
                leg2_fixed=True,
                leg2_mtm=True,
                currency="eur",
                leg2_currency="usd",
                leg2_fixed_rate=1.3,
                curves=["eureur", "eureur", "usdusd", "usdusd"],
                notional=1e6,
            ),
            XCS(  # XCS - FloatFixed
                dt(2022, 7, 1),
                "3M",
                "A",
                fixed=False,
                leg2_fixed=True,
                leg2_mtm=True,
                currency="usd",
                leg2_currency="eur",
                curves=["usdusd", "usdusd", "eureur", "eureur"],
                notional=-1e6,
            ),
            # FXSwap(
            #     dt(2022, 7, 1),
            #     "3M",
            #     "A",
            #     currency="eur",
            #     leg2_currency="usd",
            #     curves=["eureur", "eureur", "usdusd", "usdusd"],
            #     notional=1e6,
            #     fx_fixing=0.999851,
            #     split_notional=1003052.812,
            #     points=2.523505,
            # ),
            FXSwap(
                dt(2022, 7, 1),
                "3M",
                currency="usd",
                leg2_currency="eur",
                curves=["usdusd", "usdusd", "eureur", "eureur"],
                notional=-1e6,
                # fx_fixing=0.999851,
                # split_notional=1003052.812,
                # points=2.523505,
            ),
            FXExchange(
                settlement=dt(2022, 10, 1),
                currency="eur",
                leg2_currency="usd",
                curves=["eureur", "eureur", "usdusd", "usdusd"],
                notional=1e6 * 25 / 74.27,
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
            ZCIS(dt(2022, 1, 1), "1y", "A", curves=["eureur", "eureur", "eu_cpi", "eureur"]),
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

    @pytest.mark.parametrize("inst, param", [
        (IRS(dt(2022, 7, 1), "3M", "A", curves="usdusd"), "fixed_rate"),
        (FRA(dt(2022, 7, 1), "3M", "Q", curves="usdusd"), "fixed_rate"),
        (SBS(dt(2022, 7, 1), "3M", "Q", curves=["usdusd", "usdusd", "eureur", "usdusd"]), "float_spread"),
        (ZCS(dt(2022, 1, 1), "1Y", "Q", curves=["usdusd"]), "fixed_rate"),
        (ZCIS(dt(2022, 1, 1), "1Y", "A", curves=["usdusd", "usdusd", "eu_cpi", "usdusd"]), "fixed_rate"),
        (IIRS(dt(2022, 1, 1), "1Y", "Q", curves=["eu_cpi", "usdusd", "usdusd", "usdusd"]), "fixed_rate"),
        (FXExchange(dt(2022, 3, 1), currency="usd", leg2_currency="eur", curves=[NoInput(0), "usdusd", NoInput(0), "eurusd"]), "fx_rate")
    ])
    def test_null_priced_delta_round_trip_one_pricing_param(self, inst, param):
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
            ZCIS(dt(2022, 1, 1), "1y", "A", curves=["eureur", "eureur", "eu_cpi", "eureur"]),
        ]
        solver = Solver(
            curves=[c1, c2, c3, c4],
            instruments=ins,
            s=[1.2, 1.3, 1.33, 0.5],
            id="solver",
            instrument_labels=["eur 1y", "usd 1y", "eur 1y xcs adj.", "1y cpi"],
            fx=fxf,
        )

        unpriced_delta = inst.delta(solver=solver)
        mid_market_price = inst.rate(solver=solver)
        setattr(inst, param, float(mid_market_price))
        priced_delta = inst.delta(solver=solver)

        assert_frame_equal(unpriced_delta, priced_delta)

    @pytest.mark.parametrize("inst, param", [
        (FXSwap(dt(2022, 2, 1), "3M", currency="eur", leg2_currency="usd", curves=[NoInput(0), "eurusd", NoInput(0), "usdusd"]), "points"),
    ])
    def test_null_priced_delta_round_trip_one_pricing_param_fx_fix(self, inst, param):
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
            ZCIS(dt(2022, 1, 1), "1y", "A", curves=["eureur", "eureur", "eu_cpi", "eureur"]),
        ]
        solver = Solver(
            curves=[c1, c2, c3, c4],
            instruments=ins,
            s=[1.2, 1.3, 1.33, 0.5],
            id="solver",
            instrument_labels=["eur 1y", "usd 1y", "eur 1y xcs adj.", "1y cpi"],
            fx=fxf,
        )

        unpriced_delta = inst.delta(solver=solver, fx=fxf)
        mid_market_price = inst.rate(solver=solver, fx=fxf)
        setattr(inst, param, float(mid_market_price))
        priced_delta = inst.delta(solver=solver, fx=fxf)

        assert_frame_equal(unpriced_delta, priced_delta)


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
    def test_irs_spread_isda_flat_compound(self, curve, float_spread, fixed_rate, expected):
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
            leg2_float_spread=NoInput(0),
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

        irs.fixed_rate = NoInput(0)  # fixed rate set back to initial
        assert abs(irs.npv(curve)) < 1e-8

        irs.fixed_rate = float(irs.rate(curve))
        irs.leg2_float_spread = 100
        assert irs.npv(curve) > 1

        irs.leg2_float_spread = NoInput(0)
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

    def test_irs_interpolated_stubs(self, curve):
        curve6 = LineCurve({dt(2022, 1, 1): 4.0, dt(2023, 2, 1): 4.0})
        curve3 = LineCurve({dt(2022, 1, 1): 3.0, dt(2023, 2, 1): 3.0})
        curve1 = LineCurve({dt(2022, 1, 1): 1.0, dt(2023, 2, 1): 1.0})
        irs = IRS(
            effective=dt(2022, 1, 3),
            termination=dt(2023, 1, 3),
            front_stub=dt(2022, 2, 10),
            back_stub=dt(2022, 8, 10),
            frequency="Q",
            convention="act360",
            curves=[{"3m": curve3, "1m": curve1, "6M": curve6}, curve],
            leg2_fixing_method="ibor",
        )
        cashflows = irs.cashflows()
        assert (cashflows.loc[("leg2", 0), "Rate"] - 1.23729) < 1e-4
        assert (cashflows.loc[("leg2", 3), "Rate"] - 3.58696) < 1e-4

    def test_irs_interpolated_stubs_solver(self):
        curve6 = Curve({dt(2022, 1, 1): 4.0, dt(2023, 2, 1): 4.0}, id="6m")
        curve3 = Curve({dt(2022, 1, 1): 3.0, dt(2023, 2, 1): 3.0}, id="3m")
        solver = Solver(
            curves=[curve6, curve3],
            instruments=[
                IRS(dt(2022, 1, 1), "1Y", "A", curves=curve6),
                IRS(dt(2022, 1, 1), "1Y", "A", curves=curve3),
            ],
            s=[6.0, 3.0],
        )
        irs = IRS(
            effective=dt(2022, 1, 3),
            termination=dt(2022, 11, 3),
            front_stub=dt(2022, 5, 3),
            stub="Front",
            frequency="Q",
            convention="act360",
            curves=[{"3m": "3m", "6m": "6m"}, "3m"],
            leg2_fixing_method="ibor",
        )
        cashflows = irs.cashflows(solver=solver)
        assert (cashflows.loc[("leg2", 0), "Rate"] - 3.93693) < 1e-4


class TestIIRS:
    def test_index_base_none_populated(self, curve):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.5, dt(2034, 1, 1): 0.4},
            index_lag=3,
            index_base=100.0,
        )
        iirs = IIRS(
            effective=dt(2022, 2, 1),
            termination="1y",
            frequency="Q",
            index_lag=3,
            notional_exchange=False,
        )
        for period in iirs.leg1.periods:
            assert period.index_base is NoInput(0)
        iirs.rate(curves=[i_curve, curve])
        for period in iirs.leg1.periods:
            assert period.index_base == 200.0

    def test_iirs_npv_mid_mkt_zero(self, curve):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.5, dt(2034, 1, 1): 0.4},
            index_lag=3,
            index_base=100.0,
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

        iirs.index_base = NoInput(0)  # index_base set back to initial
        iirs.fixed_rate = NoInput(0)
        assert abs(iirs.npv([i_curve, curve])) < 1e-8

        mid_fixed = float(iirs.rate([i_curve, curve]))
        iirs.base_index = 200.0  # this is index_base from i_curve
        new_mid = float(iirs.rate([i_curve, curve]))
        assert abs(mid_fixed - new_mid) < 1e-6

    def test_cashflows(self, curve):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2023, 2, 1): 0.99}, index_lag=3, index_base=100.0
        )
        iirs = IIRS(
            effective=dt(2022, 2, 1),
            termination="9M",
            frequency="Q",
            index_base=Series([90, 110], index=[dt(2022, 1, 31), dt(2022, 2, 2)]),
            index_fixings=[110, 115],
            index_lag=3,
            index_method="daily",
            fixed_rate=1.0,
        )
        result = iirs.cashflows([i_curve, curve, curve, curve])
        expected = DataFrame(
            {
                "Index Val": [110.0, 115.0, 100.7754, np.nan, np.nan, np.nan],
                "Index Ratio": [1.10, 1.15, 1.00775, np.nan, np.nan, np.nan],
                "NPV": [-2682.655, -2869.534, -2488.937, 9849.93, 10070.85, 9963.277],
                "Type": ["IndexFixedPeriod"] * 3 + ["FloatPeriod"] * 3,
            },
            index=MultiIndex.from_tuples(
                [("leg1", 0), ("leg1", 1), ("leg1", 2), ("leg2", 0), ("leg2", 1), ("leg2", 2)]
            ),
        )
        assert_frame_equal(
            expected,
            result[["Index Val", "Index Ratio", "NPV", "Type"]],
            rtol=1e-3,
        )

    def test_npv_no_index_base(self, curve):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.5, dt(2034, 1, 1): 0.4},
            index_lag=3,
            index_base=100.0,
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
            index_base=100.0,
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
            frequency="s",
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

        fra.fixed_rate = NoInput(0)  # fixed rate set back to initial
        assert abs(fra.npv(curve)) < 1e-9

    @pytest.mark.parametrize("eom, exp", [(True, dt(2021, 5, 31)), (False, dt(2021, 5, 26))])
    def test_fra_roll_inferral(self, eom, exp):
        fra = FRA(
            effective=dt(2021, 2, 26),
            termination="3m",
            frequency="Q",
            eom=eom,
            calendar="bus",
        )
        assert fra.leg1.schedule.termination == exp


class TestZCS:
    @pytest.mark.parametrize("freq, exp", [("Q", 3.529690979), ("S", 3.54526437721296)])
    def test_zcs_rate(self, freq, exp):
        usd = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2027, 1, 1): 0.85, dt(2032, 1, 1): 0.70},
            id="usd",
            calendar="bus",
        )
        zcs = ZCS(
            effective=dt(2022, 1, 1),
            termination="10Y",
            frequency=freq,
            leg2_frequency="Q",
            calendar="bus",
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
        i_curve = IndexCurve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, index_base=200.0)
        zcis = ZCIS(
            effective=dt(2022, 1, 1),
            termination="9m",
            frequency="Q",
        )
        prior = zcis.rate(curves=[curve, curve, i_curve, curve])

        zcis.leg2_index_base = 100.0  # index base is lower
        result = zcis.rate(curves=[curve, curve, i_curve, curve])
        assert result > (prior + 100)


class TestValue:
    def test_npv_adelta_cashflows_raises(self):
        value = Value(dt(2022, 1, 1))
        with pytest.raises(NotImplementedError):
            value.npv()

        with pytest.raises(NotImplementedError):
            value.cashflows()

        with pytest.raises(NotImplementedError):
            value.analytic_delta()

    def test_cc_zero_rate(self, curve):
        v = Value(effective=dt(2022, 7, 1), convention="act365f", metric="cc_zero_rate")
        result = v.rate(curve)
        expected = 4.074026613753926
        assert result == expected

    def test_index_value(self):
        curve = IndexCurve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.995}, id="eu_cpi", index_base=100.0)
        v =  Value(effective=dt(2022, 7, 1), metric="index_value")
        result = v.rate(curve)
        expected = 100.24919116128588
        assert result == expected

    def test_value_raise(self, curve):
        with pytest.raises(ValueError):
            Value(effective=dt(2022, 7, 1), metric="bad").rate(curve)


class TestFXExchange:
    def test_cashflows(self):
        fxe = FXExchange(
            settlement=dt(2022, 10, 1),
            currency="eur",
            leg2_currency="usd",
            notional=1e6,
            fx_rate=2.05,
        )
        result = fxe.cashflows()
        expected = DataFrame(
            {
                "Type": ["Cashflow", "Cashflow"],
                "Period": ["Exchange", "Exchange"],
                "Ccy": ["EUR", "USD"],
                "Payment": [dt(2022, 10, 1), dt(2022, 10, 1)],
                "Notional": [1e6, -2050000.0],
                "Rate": [None, 2.05],
                "Cashflow": [-1e6, 2050000.0],
            },
            index=MultiIndex.from_tuples([("leg1", 0), ("leg2", 0)])
        )
        result = result[["Type", "Period", "Ccy", "Payment", "Notional", "Rate", "Cashflow"]]
        assert_frame_equal(result, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "base, fx",
        [
            ("eur", 1.20),
            ("usd", 1.20),
            ("eur", FXRates({"eurusd": 1.20})),
        ],
    )
    def test_npv_rate(self, curve, curve2, base, fx):
        fxe = FXExchange(
            settlement=dt(2022, 3, 1),
            currency="eur",
            leg2_currency="usd",
            fx_rate=1.2080131682341035,
        )
        if not isinstance(fx, FXRates):
            with pytest.warns(UserWarning):
                result = fxe.npv([NoInput(0), curve, NoInput(0), curve2], NoInput(0), fx, base, local=False)
        else:
            result = fxe.npv([NoInput(0), curve, NoInput(0), curve2], NoInput(0), fx, base, local=False)
        assert abs(result - 0.0) < 1e-8

    def test_rate(self, curve, curve2):
        fxe = FXExchange(
            settlement=dt(2022, 3, 1),
            currency="eur",
            leg2_currency="usd",
            fx_rate=1.2080131682341035,
        )
        result = fxe.rate([NoInput(0), curve, NoInput(0), curve2], NoInput(0), 1.20)
        expected = 1.2080131682341035
        assert abs(result - expected) < 1e-7

    def test_npv_fx_numeric(self, curve):
        # This demonstrates the ambiguity and poor practice of
        # using numeric fx as pricing input, although it will return.
        fxe = FXExchange(
            settlement=dt(2022, 3, 1),
            currency="eur",
            leg2_currency="usd",
            fx_rate=1.2080131682341035,
        )
        # result_ = fxe.npv([curve] * 4, fx=2.0, local=True)
        with pytest.warns(UserWarning):
            result = fxe.npv([curve] * 4, fx=2.0)
            expected = -993433.103425 * 2.0 + 1200080.27069
            assert abs(result - expected) < 1e-5

        # with pytest.raises(ValueError, match="Cannot calculate `npv`"):
        #     fxe.npv([curve] * 4, fx=2.0, base="bad")

    def test_npv_no_fx_raises(self, curve):
        fxe = FXExchange(
            settlement=dt(2022, 3, 1),
            currency="eur",
            leg2_currency="usd",
            fx_rate=1.2080131682341035,
        )
        with pytest.raises(ValueError, match="Must have some FX info"):
            fxe.npv(curve)


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
#     fxs.fx_fixing_points = NoInput(0)
#     points = fxs._rate_alt(curve, curve2, 10.0)
#     npv = fxs._npv_alt(curve, curve2, 10.0)
#     assert abs(npv) < 1e-9
#
#     fxf = FXForwards(
#         FXRates({"eursek": 10.0}, dt(2022, 1, 1)),
#         {"eureur": curve, "seksek": curve2, "sekeur": curve2}
#     )
#     points2 = fxs.rate(fxf)
#     npv2 = fxs.npv(fxf, NoInput(0), "eur")
#     assert abs(npv2) < 1e-9


class TestNonMtmXCS:
    def test_nonmtmxcs_npv(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"eurusd": 1.1}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "eurusd": curve2, "eureur": curve2},
        )

        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=False,
            leg2_fixed=False,
            leg2_mtm=False,
            payment_lag=0,
            currency="eur",
            leg2_currency="usd",
            payment_lag_exchange=0,
        )
        # npv2 = xcs._npv2(curve2, curve2, curve, curve, 1.10)
        npv = xcs.npv([curve2, curve2, curve, curve], NoInput(0), fxf)
        assert abs(npv) < 1e-9

        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=False,
            leg2_fixed=False,
            leg2_mtm=False,
            payment_lag=0,
            amortization=100e3,
            currency="eur",
            leg2_currency="usd",
            payment_lag_exchange=0,
        )
        # npv2 = xcs._npv2(curve2, curve2, curve, curve, 1.10)
        npv = xcs.npv([curve2, curve2, curve, curve], NoInput(0), fxf)
        assert abs(npv) < 1e-9

    def test_nonmtmxcs_fx_notional(self):
        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=False,
            leg2_fixed=False,
            leg2_mtm=False,
            payment_lag=0,
            currency="eur",
            leg2_currency="usd",
            payment_lag_exchange=0,
            fx_fixings=2.0,
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

        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=False,
            leg2_fixed=False,
            leg2_mtm=False,
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            float_spread=float_spd,
            leg2_spread_compound_method=compound,
        )

        result = xcs.rate([curve, curve, curve2, curve2], NoInput(0), fxf, 2)
        assert abs(result - expected) < 1e-4
        alias = xcs.spread([curve, curve, curve2, curve2], NoInput(0), fxf, 2)
        assert alias == result

        xcs.leg2_float_spread = result
        validate = xcs.npv([curve, curve, curve2, curve2], NoInput(0), fxf)
        assert abs(validate) < 1e-2
        result2 = xcs.rate([curve, curve, curve2, curve2], NoInput(0), fxf, 2)
        assert abs(result - result2) < 1e-3

        # reverse legs
        xcs_reverse = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=False,
            leg2_fixed=False,
            leg2_mtm=False,
            payment_lag=0,
            currency="usd",
            leg2_currency="nok",
            payment_lag_exchange=0,
            notional=1e6,
            leg2_float_spread=float_spd,
            spread_compound_method=compound,
        )
        result = xcs_reverse.rate([curve2, curve2, curve, curve], NoInput(0), fxf, 1)
        assert abs(result - expected) < 1e-4

    def test_no_fx_raises(self, curve, curve2):
        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=False,
            leg2_fixed=False,
            leg2_mtm=False,
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            float_spread=0.0,
        )

        with pytest.raises(ValueError, match="`fx` is required when `fx_fixings` is"):
            with default_context("no_fx_fixings_for_xcs", "raise"):
                xcs.npv([curve, curve, curve2, curve2])

        with pytest.raises(ValueError, match="`fx` is required when `fx_fixings` is"):
            with default_context("no_fx_fixings_for_xcs", "raise"):
                xcs.cashflows([curve, curve, curve2, curve2])

        # with pytest.warns():
        #     with default_context("no_fx_fixings_for_xcs", "warn"):
        #         xcs.npv([curve, curve, curve2, curve2])

    def test_nonmtmxcs_cashflows(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=False,
            leg2_fixed=False,
            leg2_mtm=False,
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )

        result = xcs.cashflows([curve, curve, curve2, curve2], NoInput(0), fxf)
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
            "dual": Dual(10.0, ["x"], []),
            "dual2": Dual2(10.0, ["x"], [], []),
        }
        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=False,
            leg2_fixed=False,
            leg2_mtm=False,
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            fx_fixings=mapping[fix],
        )
        assert abs(xcs.npv([curve, curve, curve2, curve2], fx=fxr)) < 1e-7

    def test_is_priced(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )
        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=False,
            leg2_fixed=False,
            leg2_mtm=False,
            leg2_float_spread=1.0,
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )
        result = xcs.npv(curves=[curve2, curve2, curve, curve], fx=fxf)
        assert abs(result - 65.766356) < 1e-5

    def test_no_fx_warns(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )
        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=False,
            leg2_fixed=False,
            leg2_mtm=False,
            leg2_float_spread=1.0,
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )
        with default_context("no_fx_fixings_for_xcs", "warn"):
            with pytest.warns(UserWarning):
                xcs.npv(curves=[curve2, curve2, curve, curve], local=True)


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
        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            fixed=True,
            leg2_fixed=False,
            leg2_mtm=False,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            leg2_spread_compound_method=compound,
            leg2_float_spread=float_spd,
        )

        result = xcs.rate([curve2, curve2, curve, curve], NoInput(0), fxf, 1)
        assert abs(result - expected) < 1e-4
        assert abs(xcs.npv([curve2, curve2, curve, curve], NoInput(0), fxf)) < 1e-6

        xcs.fixed_rate = result  # set the fixed rate and check revalues to zero
        assert abs(xcs.npv([curve2, curve2, curve, curve], NoInput(0), fxf)) < 1e-6

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
        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=True,
            leg2_fixed=False,
            leg2_mtm=False,
            payment_lag=0,
            currency="eur",
            leg2_currency="usd",
            payment_lag_exchange=0,
            fx_fixings=2.0,
            notional=1e6,
        )
        assert xcs.leg2_notional == -2e6

    def test_nonmtmfixxcs_no_fx_raises(self, curve, curve2):
        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=True,
            leg2_fixed=False,
            leg2_mtm=False,
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )

        with pytest.raises(ValueError, match="`fx` is required when `fx_fixings` is"):
            with default_context("no_fx_fixings_for_xcs", "raise"):
                xcs.npv([curve, curve, curve2, curve2])

        with pytest.raises(ValueError, match="`fx` is required when `fx_fixings` is"):
            with default_context("no_fx_fixings_for_xcs", "raise"):
                xcs.cashflows([curve, curve, curve2, curve2])

    def test_nonmtmfixxcs_cashflows(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=True,
            leg2_fixed=False,
            leg2_mtm=False,
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )

        result = xcs.cashflows([curve, curve, curve2, curve2], NoInput(0), fxf)
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
            "dual": Dual(10.0, ["x"], []),
            "dual2": Dual2(10.0, ["x"], [], []),
        }
        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=True,
            leg2_fixed=False,
            leg2_mtm=False,
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            fx_fixings=mapping[fix],
            leg2_float_spread=10.0,
        )
        assert abs(xcs.npv([curve2, curve2, curve, curve], fx=fxf)) < 1e-7

    def test_nonmtmfixxcs_raises(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=True,
            leg2_fixed=False,
            leg2_mtm=False,
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )

        with pytest.raises(ValueError, match="Cannot solve for a"):
            xcs.rate([curve, curve, curve2, curve2], NoInput(0), fxf, leg=2)


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
    #     result = xcs.rate([curve2, curve2, curve, curve], NoInput(0), fxf, 1)
    #     assert abs(result - expected) < 1e-4
    #     assert abs(xcs.npv([curve2, curve2, curve, curve], NoInput(0), fxf)) < 1e-6
    #
    #     xcs.fixed_rate = result  # set the fixed rate and check revalues to zero
    #     assert abs(xcs.npv([curve2, curve2, curve, curve], NoInput(0), fxf)) < 1e-6
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
    #     result = xcs.cashflows([curve, curve, curve2, curve2], NoInput(0), fxf)
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
            "dual": Dual(10.0, ["x"], []),
            "dual2": Dual2(10.0, ["x"], [], []),
        }
        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=True,
            leg2_fixed=True,
            leg2_mtm = False,
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            fx_fixings=mapping[fix],
            leg2_fixed_rate=2.0,
        )
        assert abs(xcs.npv([curve2, curve2, curve, curve], fx=fxr)) < 1e-7

    def test_nonmtmfixfixxcs_raises(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            fixed=True,
            leg2_fixed=True,
            leg2_mtm=False,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )

        with pytest.raises(ValueError, match="Cannot solve for a"):
            xcs.rate([curve, curve, curve2, curve2], NoInput(0), fxf, leg=2)

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

        npv = xcs.npv([curve2, curve2, curve, curve], NoInput(0), fxf)
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

        result = xcs.cashflows([curve, curve, curve2, curve2], NoInput(0), fxf)
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

    @pytest.mark.skip(reason="After merging all XCS to one class inputting `fx_fixings` as list was changed.")
    def test_mtmxcs_fx_fixings_raises(self):
        with pytest.raises(ValueError, match="`fx_fixings` for MTM XCS should"):
            _ = XCS(dt(2022, 2, 1), "8M", "M", fx_fixings=NoInput(0), currency="usd", leg2_currency="eur")

        with pytest.raises(ValueError, match="`fx_fixings` for MTM XCS should"):
            _ = XCS(dt(2022, 2, 1), "8M", "M", fx_fixings=NoInput(0), fixed=True, leg2_fixed=False, leg2_mtm=True, currency="usd", leg2_currency="eur")

        with pytest.raises(ValueError, match="`fx_fixings` for MTM XCS should"):
            _ = XCS(dt(2022, 2, 1), "8M", "M", fx_fixings=NoInput(0), fixed=True, leg2_fixed=True, leg2_mtm=True, currency="usd", leg2_currency="eur")

        with pytest.raises(ValueError, match="`fx_fixings` for MTM XCS should"):
            _ = XCS(dt(2022, 2, 1), "8M", "M", fx_fixings=NoInput(0), fixed=False, leg2_fixed=True, leg2_mtm=True, currency="usd", leg2_currency="eur")

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

        result = xcs.rate([curve2, curve2, curve, curve], NoInput(0), fxf, 2)
        assert abs(result - expected) < 1e-4
        alias = xcs.spread([curve2, curve2, curve, curve], NoInput(0), fxf, 2)
        assert alias == result

        xcs.leg2_float_spread = result
        validate = xcs.npv([curve2, curve2, curve, curve], NoInput(0), fxf)
        assert abs(validate) < 1e-2
        result2 = xcs.rate([curve2, curve2, curve, curve], NoInput(0), fxf, 2)
        assert abs(result - result2) < 1e-3

    def test_fx_fixings_2_tuple(self):
        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            fx_fixings=(1.25, Series([1.5, 1.75], index=[dt(2022, 3, 1), dt(2022, 4, 1)]))
        )
        assert xcs.leg2.fx_fixings == [1.25, 1.5, 1.75]


class TestFixedFloatXCS:
    def test_mtmfixxcs_rate(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=True,
            leg2_fixed=False,
            leg2_mtm=True,
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )

        result = xcs.rate([curve2, curve2, curve, curve], NoInput(0), fxf, 1)

        irs = IRS(dt(2022, 2, 1), "8M", "M", currency="nok", payment_lag=0)
        validate = irs.rate(curve2)
        assert abs(result - validate) < 1e-4
        # alias = xcs.spread([curve2, curve2, curve, curve], NoInput(0), fxf, 2)

    def test_mtmfixxcs_rate_reversed(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=False,
            leg2_fixed=True,
            leg2_mtm=True,
            payment_lag=0,
            currency="usd",
            leg2_currency="nok",
            payment_lag_exchange=0,
            notional=10e6,
        )

        result = xcs.rate([curve, curve, curve2, curve2], NoInput(0), fxf, 2)

        irs = IRS(dt(2022, 2, 1), "8M", "M", currency="nok", payment_lag=0)
        validate = irs.rate(curve2)
        assert abs(result - validate) < 1e-2
        alias = xcs.spread([curve, curve, curve2, curve2], NoInput(0), fxf, 2)
        assert abs(result - alias) < 1e-4


class TestFixedFixedXCS:
    def test_mtmfixfixxcs_rate(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )

        irs = IRS(dt(2022, 2, 1), "8M", "M", payment_lag=0)
        nok_rate = float(irs.rate(curve2))
        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=True,
            leg2_fixed=True,
            leg2_mtm=True,
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            fixed_rate=nok_rate,
        )
        result = xcs.rate([curve2, curve2, curve, curve], NoInput(0), fxf, 2)
        validate = irs.rate(curve)
        assert abs(result - validate) < 1e-4
        alias = xcs.spread([curve2, curve2, curve, curve], NoInput(0), fxf, 2)
        assert abs(result - alias) < 1e-8

        # test reverse
        usd_rate = float(irs.rate(curve))
        xcs.fixed_rate = NoInput(0)
        xcs.leg2_fixed_rate = usd_rate
        result = xcs.rate([curve2, curve2, curve, curve], NoInput(0), fxf, 1)
        validate = irs.rate(curve2)
        assert abs(result - validate) < 1e-4
        alias = xcs.spread([curve2, curve2, curve, curve], NoInput(0), fxf, 1)
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
            currency="usd",
            leg2_currency="nok",
            payment_lag=0,
            notional=1e6,
        )
        expected = fxf.swap("usdnok", [dt(2022, 2, 1), dt(2022, 10, 1)])
        result = fxs.rate([NoInput(0), curve, NoInput(0), curve2], NoInput(0), fxf)
        assert abs(result-expected) < 1e-10
        assert np.isclose(result.dual, expected.dual)

    def test_fxswap_npv(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )
        fxs = FXSwap(
            dt(2022, 2, 1),
            "8M",
            currency="usd",
            leg2_currency="nok",
            payment_lag=0,
            notional=1e6,
        )

        assert abs(fxs.npv([NoInput(0), curve, NoInput(0), curve2], NoInput(0), fxf)) < 1e-7

        result = fxs.rate([NoInput(0), curve, NoInput(0), curve2], NoInput(0), fxf, fixed_rate=True)
        fxs.leg2_fixed_rate = result
        assert abs(fxs.npv([NoInput(0), curve, NoInput(0), curve2], NoInput(0), fxf)) < 1e-7

    @pytest.mark.parametrize("points, split_notional", [
        (100, 1e6), (NoInput(0), 1e6), (100, NoInput(0))
    ])
    def test_fxswap_points_raises(self, points, split_notional):
        if points is not NoInput(0):
            msg = "Cannot initialise FXSwap with `points` but without `fx_fixings`."
            with pytest.raises(ValueError, match=msg):
                FXSwap(
                    dt(2022, 2, 1),
                    "8M",
                    currency="usd",
                    leg2_currency="nok",
                    payment_lag=0,
                    notional=1e6,
                    split_notional=split_notional,
                    points=points,
                )
        else:
            msg = "Cannot initialise FXSwap with `split_notional` but without `fx_fixings`"
            with pytest.raises(ValueError, match=msg):
                FXSwap(
                    dt(2022, 2, 1),
                    "8M",
                    currency="usd",
                    leg2_currency="nok",
                    payment_lag=0,
                    notional=1e6,
                    split_notional=split_notional,
                    points=points,
                )

    def test_fxswap_points_warns(self):
        with pytest.warns(UserWarning):
            fxs = FXSwap(
                dt(2022, 2, 1),
                "8M",
                fx_fixings=11.0,
                currency="usd",
                leg2_currency="nok",
                payment_lag=0,
                notional=1e6,
            )
            assert fxs._is_split is False

        with pytest.warns(UserWarning):
            fxs = FXSwap(
                dt(2022, 2, 1),
                "8M",
                fx_fixings=11.0,
                currency="usd",
                leg2_currency="nok",
                payment_lag=0,
                notional=1e6,
                split_notional=1e6,
            )
            assert fxs._is_split is True

    @pytest.mark.parametrize("fx_fixings, points, split_notional, expected", [
        (NoInput(0), NoInput(0), NoInput(0), Dual(0, ["fx_usdnok"], [-1712.833785])),
        (11.0, 1800.0, NoInput(0), Dual(-3734.617680, ["fx_usdnok"], [3027.88203904])),
        (11.0, 1754.5623360395632, NoInput(0), Dual(-4166.37288388, ["fx_usdnok"], [3071.05755945])),
        (10.032766762996951, 1754.5623360395632, NoInput(0), Dual(0, ["fx_usdnok"], [2654.42027107])),
        (10.032766762996951, 1754.5623360395632, 1027365.1574336714, Dual(0, ["fx_usdnok"], [0.0]))
    ])
    def test_fxswap_parameter_combinations_off_mids_given(
            self, curve, curve2, fx_fixings, points, split_notional, expected
    ):
        # curve._set_ad_order(1)
        # curve2._set_ad_order(1)
        # risk sensitivity to curve is checked in:
        # test_null_priced_delta_round_trip_one_pricing_param_fx_fix

        # the exact values of relevance here are:
        # usdnok: 10.032766762996951,
        # points:  1754.5623360395632
        # split_notional: 1027365.1574336714

        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )
        fxs = FXSwap(
            dt(2022, 2, 1),
            "8M",
            fx_fixings=fx_fixings,
            points=points,
            split_notional=split_notional,
            currency="usd",
            leg2_currency="nok",
            payment_lag=0,
            notional=1e6,
        )
        assert fxs.points == points
        result = fxs.npv(curves=[NoInput(0), curve, NoInput(0), curve2], fx=fxf, base="usd")

        assert abs(result-expected) < 1e-6
        assert np.isclose(result.dual, expected.dual)

    def test_rate_with_fixed_parameters(self, curve, curve2):
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )
        fxs = FXSwap(
            dt(2022, 2, 1),
            "8M",
            fx_fixings=10.01,
            points=1765,
            split_notional=1.01e6,
            currency="usd",
            leg2_currency="nok",
            payment_lag=0,
            notional=1e6,
        )
        result = fxs.rate([NoInput(0), curve, NoInput(0), curve2], fx=fxf)
        expected = 1746.59802
        assert abs(result - expected) < 1e-4

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
    #     npv_nok = fxs.npv([NoInput(0), fxf.curve("usd", "nok"), NoInput(0), curve2], NoInput(0), fxf)
    #     npv_usd = fxs.npv([NoInput(0), curve, NoInput(0), fxf.curve("nok", "usd")], NoInput(0), fxf)
    #     assert abs(npv_nok-npv_usd) < 1e-7  # npvs are equivalent becasue xcs basis =0


class TestSTIRFuture:
    def test_stir_rate(self, curve, curve2):
        stir = STIRFuture(
            effective=dt(2022, 3, 16),
            termination=dt(2022, 6, 15),
            spec="usd_stir",
        )
        expected = 95.96254344884888
        result = stir.rate(curve, metric="price")
        assert abs(100 - result -stir.rate(curve)) < 1e-8
        assert abs(result-expected) < 1e-8

    def test_stir_no_gamma(self, curve):
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="usdusd")
        ins = [
            IRS(dt(2022, 3, 16), dt(2022, 6, 15), "Q", curves="usdusd"),
        ]
        solver = Solver(
            curves=[c1],
            instruments=ins,
            s=[1.2],
            id="solver",
            instrument_labels=["usd fut"],
        )
        stir = STIRFuture(
            effective=dt(2022, 3, 16),
            termination=dt(2022, 6, 15),
            spec="usd_stir",
            curves="usdusd",
        )
        result = stir.delta(solver=solver).sum().sum()
        assert abs(result + 25.0) < 1e-7

        result = stir.gamma(solver=solver).sum().sum()
        assert abs(result) < 1e-7

    def test_stir_npv(self):
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="usdusd")
        # irs = IRS(dt(2022, 3, 16), dt(2022, 6, 15), "Q", curves="usdusd")
        stir = STIRFuture(
            effective=dt(2022, 3, 16),
            termination=dt(2022, 6, 15),
            spec="usd_stir",
            curves="usdusd",
            price=99.50,
        )
        result = stir.npv(curves=c1)
        expected = (99.5 - (100 - 0.99250894761)) * 2500 * -1.0
        assert abs(result - expected) < 1e-7

    def test_stir_raises(self):
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="usdusd")
        # irs = IRS(dt(2022, 3, 16), dt(2022, 6, 15), "Q", curves="usdusd")
        stir = STIRFuture(
            effective=dt(2022, 3, 16),
            termination=dt(2022, 6, 15),
            spec="usd_stir",
            curves="usdusd",
            price=99.50,
        )
        with pytest.raises(ValueError, match="`metric` must be in"):
            stir.rate(curves=c1, metric="bad")

    def test_analytic_delta(self):
        stir = STIRFuture(
            effective=dt(2022, 3, 16),
            termination=dt(2022, 6, 15),
            spec="usd_stir",
            curves="usdusd",
            price=99.50,
            contracts=100,
        )
        expected = -2500.0
        result = stir.analytic_delta()
        assert abs(result-expected) < 1e-10


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
            (XCS, {"fixed": False, "leg2_fixed": False, "leg2_mtm": False}),
            (XCS, {"fixed": True, "leg2_fixed": False, "leg2_mtm": False, "fixed_rate": 2.0}),
            (XCS, {"fixed": True, "leg2_fixed": True, "leg2_mtm": False, "fixed_rate": 2.0}),
            (XCS, {}),  # defaults to fixed:False, leg2_fixed: False, leg2_mtm: True
            (XCS, {"fixed": True, "leg2_fixed": False, "leg2_mtm": True, "fixed_rate": 2.0}),
            (XCS, {"fixed": False, "leg2_fixed": True, "leg2_mtm": True}),
            (XCS, {"fixed": True, "leg2_fixed": True, "leg2_mtm": True, "fixed_rate": 2.0}),
        ],
    )
    def test_allxcs(self, klass, kwargs, curve, curve2):
        ob = klass(
            dt(2022, 1, 28),
            "6m",
            "S",
            currency="usd",
            leg2_currency="eur",
            curves=[curve, NoInput(0), curve2, NoInput(0)],
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

    # TODO FXEXchange and FXSwap


class TestPortfolio:
    def test_portfolio_npv(self, curve):
        irs1 = IRS(dt(2022, 1, 1), "6m", "Q", fixed_rate=1.0, curves=curve)
        irs2 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=2.0, curves=curve)
        pf = Portfolio([irs1, irs2])
        assert pf.npv(base="usd") == irs1.npv() + irs2.npv()

        pf = Portfolio([irs1] * 5)
        assert pf.npv(base="usd") == irs1.npv() * 5

        with default_context("pool", 2):  # also test parallel processing
            result = pf.npv(base="usd")
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

    def test_portfolio_local_parallel(self, curve):
        irs1 = IRS(dt(2022, 1, 1), "6m", "Q", fixed_rate=1.0, curves=curve, currency="usd")
        irs2 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=2.0, curves=curve, currency="eur")
        irs3 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=2.0, curves=curve, currency="usd")
        pf = Portfolio([irs1, irs2, irs3])

        expected = {
            "usd": 20093.295095887483,
            "eur": 5048.87332403382,
        }
        with default_context("pool", 2):  # also test parallel processing
            result = pf.npv(local=True)
            assert result == expected

    def test_portfolio_mixed_currencies(self):
        ll_curve = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2022, 5, 1): 1.0,
                dt(2022, 9, 3): 1.0
            },
            interpolation="log_linear",
            id="sofr",
        )
        ll_solver = Solver(
            curves=[ll_curve],
            instruments=[
                IRS(dt(2022, 1, 1), "4m", "Q", curves="sofr"),
                IRS(dt(2022, 1, 1), "8m", "Q", curves="sofr"),
            ],
            s=[1.85, 2.10],
            instrument_labels=["4m", "8m"],
            id="sofr"
        )

        ll_curve = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2022, 4, 1): 1.0,
                dt(2022, 10, 1): 1.0
            },
            interpolation="log_linear",
            id="estr",
        )
        combined_solver = Solver(
            curves=[ll_curve],
            instruments=[
                IRS(dt(2022, 1, 1), "3m", "Q", curves="estr"),
                IRS(dt(2022, 1, 1), "9m", "Q", curves="estr"),
            ],
            s=[0.75, 1.65],
            instrument_labels=["3m", "9m"],
            pre_solvers=[ll_solver],
            id="estr"
        )

        irs = IRS(
            effective=dt(2022, 1, 1),
            termination="6m",
            frequency="Q",
            currency="usd",
            notional=500e6,
            fixed_rate=2.0,
            curves="sofr",  # or ["sofr", "sofr"] for forecasting and discounting
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
        result = pf.npv(solver=combined_solver, local=True)
        assert "eur" in result and "usd" in result

        # the following should execute without warnings
        pf.delta(solver=combined_solver)
        pf.gamma(solver=combined_solver)


class TestFly:
    @pytest.mark.parametrize("mechanism", [False, True])
    def test_fly_npv(self, curve, mechanism):
        mechanism = curve if mechanism else NoInput(0)
        inverse = curve if mechanism is NoInput(0) else NoInput(0)
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=mechanism)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=mechanism)
        irs3 = IRS(dt(2022, 1, 1), "5m", "Q", fixed_rate=1.0, curves=mechanism)
        fly = Fly(irs1, irs2, irs3)
        assert fly.npv(inverse) == irs1.npv(inverse) + irs2.npv(inverse) + irs3.npv(inverse)

    @pytest.mark.parametrize("mechanism", [False, True])
    def test_fly_rate(self, curve, mechanism):
        mechanism = curve if mechanism else NoInput(0)
        inv = curve if mechanism is NoInput(0) else NoInput(0)
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=mechanism)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=mechanism)
        irs3 = IRS(dt(2022, 1, 1), "5m", "Q", fixed_rate=1.0, curves=mechanism)
        fly = Fly(irs1, irs2, irs3)
        assert fly.rate(inv) == (-irs1.rate(inv) + 2 * irs2.rate(inv) - irs3.rate(inv)) * 100.0

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

    def test_delta(self, simple_solver):
        irs1 = IRS(dt(2022, 1, 1), "6m", "A", fixed_rate=1.0, notional=-3e6, curves="curve")
        irs2 = IRS(dt(2022, 1, 1), "1Y", "A", fixed_rate=2.0, notional=3e6, curves="curve")
        irs3 = IRS(dt(2022, 1, 1), "18m", "A", fixed_rate=1.0, notional=-1e6, curves="curve")
        fly = Fly(irs1, irs2, irs3)
        result = fly.delta(solver=simple_solver).to_numpy()
        expected = np.array([[102.08919479], [-96.14488074]])
        assert np.all(np.isclose(result, expected))

    def test_gamma(self, simple_solver):
        irs1 = IRS(dt(2022, 1, 1), "6m", "A", fixed_rate=1.0, notional=-3e6, curves="curve")
        irs2 = IRS(dt(2022, 1, 1), "1Y", "A", fixed_rate=2.0, notional=3e6, curves="curve")
        irs3 = IRS(dt(2022, 1, 1), "18m", "A", fixed_rate=1.0, notional=-1e6, curves="curve")
        fly = Fly(irs1, irs2, irs3)
        result = fly.gamma(solver=simple_solver).to_numpy()
        expected = np.array([[-0.02944899, 0.009254014565], [0.009254014565, 0.0094239781314]])
        assert np.all(np.isclose(result, expected))


class TestSpread:
    @pytest.mark.parametrize("mechanism", [False, True])
    def test_spread_npv(self, curve, mechanism):
        mechanism = curve if mechanism else NoInput(0)
        inverse = curve if mechanism is NoInput(0) else NoInput(0)
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=mechanism)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=mechanism)
        spd = Spread(irs1, irs2)
        assert spd.npv(inverse) == irs1.npv(inverse) + irs2.npv(inverse)

    @pytest.mark.parametrize("mechanism", [False, True])
    def test_spread_rate(self, curve, mechanism):
        mechanism = curve if mechanism else NoInput(0)
        inverse = curve if mechanism is NoInput(0) else NoInput(0)
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=mechanism)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=mechanism)
        spd = Spread(irs1, irs2)
        assert spd.rate(inverse) == (-irs1.rate(inverse) + irs2.rate(inverse)) * 100.0

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


class TestSpec:

    def test_spec_overwrites(self):
         irs = IRS(
             effective=dt(2022, 1, 1),
             termination=dt(2024, 2, 26),
             calendar="tgt",
             frequency="Q",
             leg2_method_param=0,
             notional=250.0,
             spec="test",
         )
         expected = dict(
             effective=dt(2022, 1, 1),
             termination=dt(2024, 2, 26),
             frequency="Q",
             stub="longfront",
             front_stub=NoInput(0),
             back_stub=NoInput(0),
             roll=NoInput(0),
             eom=False,
             modifier="p",
             calendar="tgt",
             payment_lag=4,
             notional=250.0,
             currency="tes",
             amortization=NoInput(0),
             convention="test",
             leg2_effective=dt(2022, 1, 1),
             leg2_termination=dt(2024, 2, 26),
             leg2_frequency="m",
             leg2_stub="longback",
             leg2_front_stub=NoInput(0),
             leg2_back_stub=NoInput(0),
             leg2_roll=1,
             leg2_eom=False,
             leg2_modifier="mp",
             leg2_calendar="nyc,tgt,ldn",
             leg2_payment_lag=3,
             leg2_notional=-250.0,
             leg2_currency="tes",
             leg2_convention="test2",
             leg2_amortization=NoInput(0),
             fixed_rate=NoInput(0),
             leg2_fixing_method=NoInput(0),
             leg2_method_param=0,
             leg2_spread_compound_method=NoInput(0),
             leg2_fixings=NoInput(0),
             leg2_float_spread=NoInput(0),
         )
         assert irs.kwargs == expected

    def test_irs(self):
        irs = IRS(
            effective=dt(2022, 1, 1),
            termination="1Y",
            spec="usd_irs",
            convention="30e360",
            fixed_rate=2.0
        )
        assert irs.kwargs["convention"] == "30e360"
        assert irs.kwargs["leg2_convention"] == "act360"
        assert irs.kwargs["currency"] == "usd"
        assert irs.kwargs["fixed_rate"] == 2.0

    def test_stir(self):
        irs = STIRFuture(
            effective=dt(2022, 3, 16),
            termination=dt(2022, 6, 15),
            spec="usd_stir",
            convention="30e360",
        )
        assert irs.kwargs["convention"] == "30e360"
        assert irs.kwargs["leg2_convention"] == "act360"
        assert irs.kwargs["currency"] == "usd"
        assert irs.kwargs["roll"] == "imm"

    def test_sbs(self):
        inst = SBS(
            effective=dt(2022, 1, 1),
            termination="1Y",
            spec="eur_sbs36",
            convention="30e360",
        )
        assert inst.kwargs["convention"] == "30e360"
        assert inst.kwargs["leg2_convention"] == "act360"
        assert inst.kwargs["currency"] == "eur"
        assert inst.kwargs["fixing_method"] == "ibor"

    def test_zcis(self):
        inst = ZCIS(
            effective=dt(2022, 1, 1),
            termination="1Y",
            spec="eur_zcis",
            leg2_calendar="nyc,tgt",
            calendar="nyc,tgt",
        )
        assert inst.kwargs["convention"] == "1+"
        assert inst.kwargs["leg2_convention"] == "1+"
        assert inst.kwargs["currency"] == "eur"
        assert inst.kwargs["leg2_index_method"] == "monthly"
        assert inst.kwargs["leg2_calendar"] == "nyc,tgt"

    def test_zcs(self):
        inst = ZCS(
            effective=dt(2022, 1, 1),
            termination="5Y",
            spec="gbp_zcs",
            leg2_calendar="nyc,tgt",
            calendar="nyc,tgt",
            fixed_rate=3.0,
        )
        assert inst.kwargs["convention"] == "act365f"
        assert inst.kwargs["leg2_frequency"] == "a"
        assert inst.kwargs["currency"] == "gbp"
        assert inst.kwargs["leg2_calendar"] == "nyc,tgt"
        assert inst.kwargs["fixed_rate"] == 3.0
        assert inst.kwargs["leg2_spread_compound_method"] == "none_simple"

    def test_iirs(self):
        inst = IIRS(
            effective=dt(2022, 1, 1),
            termination="1Y",
            spec="sek_iirs",
            leg2_calendar="nyc,tgt",
            calendar="nyc,tgt",
            fixed_rate=3.0,
        )
        assert inst.kwargs["convention"] == "actacticma"
        assert inst.kwargs["leg2_frequency"] == "q"
        assert inst.kwargs["currency"] == "sek"
        assert inst.kwargs["leg2_calendar"] == "nyc,tgt"
        assert inst.kwargs["fixed_rate"] == 3.0
        assert inst.kwargs["leg2_spread_compound_method"] == "none_simple"

    def test_fixedratebond(self):
        bond = FixedRateBond(
            effective=dt(2022, 1, 1),
            termination="1Y",
            spec="ust",
            calc_mode="ust_31bii",
            fixed_rate=2.0
        )
        assert bond.calc_mode == "ust_31bii"
        assert bond.kwargs["convention"] == "actacticma"
        assert bond.kwargs["currency"] == "usd"
        assert bond.kwargs["fixed_rate"] == 2.0
        assert bond.kwargs["ex_div"] == 1

    def test_indexfixedratebond(self):
        bond = IndexFixedRateBond(
            effective=dt(2022, 1, 1),
            termination="1Y",
            spec="ukti",
            calc_mode="ust",
            fixed_rate=2.0
        )
        assert bond.calc_mode == "ust"
        assert bond.kwargs["convention"] == "actacticma"
        assert bond.kwargs["currency"] == "gbp"
        assert bond.kwargs["fixed_rate"] == 2.0
        assert bond.kwargs["ex_div"] == 7

    def test_bill(self):
        bill = Bill(
            effective=dt(2022, 1, 1),
            termination="3m",
            spec="ustb",
            convention="act365f",
        )
        assert bill.calc_mode == "ustb"
        assert bill.kwargs["convention"] == "act365f"
        assert bill.kwargs["currency"] == "usd"
        assert bill.kwargs["fixed_rate"] == 0.0

    def test_fra(self):
        fra = FRA(
            effective=dt(2022, 1, 1),
            termination="3m",
            spec="eur_fra3",
            payment_lag=5,
            modifier="F",
            fixed_rate=2.0
        )
        assert fra.kwargs["leg2_fixing_method"] == "ibor"
        assert fra.kwargs["convention"] == "act360"
        assert fra.kwargs["currency"] == "eur"
        assert fra.kwargs["fixed_rate"] == 2.0
        assert fra.kwargs["leg2_payment_lag"] == 5
        assert fra.kwargs["leg2_modifier"] == "F"

    def test_frn(self):
        frn = FloatRateNote(
            effective=dt(2022, 1, 1),
            termination="3y",
            spec="usd_frn5",
            payment_lag=5,
        )
        assert frn.kwargs["fixing_method"] == "rfr_observation_shift"
        assert frn.kwargs["method_param"] == 5
        assert frn.kwargs["convention"] == "act360"
        assert frn.kwargs["currency"] == "usd"
        assert frn.kwargs["payment_lag"] == 5
        assert frn.kwargs["modifier"] == "mf"

    def test_xcs(self):
        xcs = XCS(
            effective=dt(2022, 1, 1),
            termination="3y",
            spec="eurusd_xcs",
            payment_lag=5,
            calendar="ldn,tgt,nyc",
        )
        assert xcs.kwargs["fixing_method"] == "rfr_payment_delay"
        assert xcs.kwargs["convention"] == "act360"
        assert xcs.kwargs["currency"] == "eur"
        assert xcs.kwargs["calendar"] == "ldn,tgt,nyc"
        assert xcs.kwargs["payment_lag"] == 5
        assert xcs.kwargs["leg2_payment_lag"] == 2
        assert xcs.kwargs["leg2_calendar"] == "tgt,nyc"


@pytest.mark.parametrize("inst, expected", [
    (IRS(dt(2022, 1, 1), "9M", "Q", currency="eur", curves=["eureur", "eur_eurusd"]),
     DataFrame([-0.21319, -0.00068, 0.21656],
               index=Index([dt(2022, 4, 3), dt(2022, 7, 3), dt(2022, 10, 3)], name="payment"),
               columns=MultiIndex.from_tuples([("EUR", "usd,eur")], names=["local_ccy", "collateral_ccy"])
               )
     ),
    (SBS(dt(2022, 1, 1), "9M", "Q", leg2_frequency="S", currency="eur", curves=["eureur", "eurusd"]),
     DataFrame([-0.51899, -6260.7208, 6299.28759],
               index=Index([dt(2022, 4, 3), dt(2022, 7, 3), dt(2022, 10, 3)], name="payment"),
               columns=MultiIndex.from_tuples([("EUR", "usd")], names=["local_ccy", "collateral_ccy"])
               )
     ),
    (
    FRA(dt(2022, 1, 15), "3M", "Q", currency="eur", curves=["eureur", "eureur"]),
    DataFrame([0.0],
              index=Index([dt(2022, 1, 15)], name="payment"),
              columns=MultiIndex.from_tuples([("EUR", "eur")],
                                             names=["local_ccy", "collateral_ccy"])
              )
    ),
    (
    FXExchange(dt(2022, 1, 15), currency="eur", leg2_currency="usd", curves=["eureur", "eureur", "usdusd", "usdeur"]),
    DataFrame([[-1000000.0, 1101072.93429]],
              index=Index([dt(2022, 1, 15)], name="payment"),
              columns=MultiIndex.from_tuples([("EUR", "eur"), ("USD", "eur")],
                                             names=["local_ccy", "collateral_ccy"])
              )
    ),
    (
    XCS(dt(2022, 1, 5), "3M", "M", currency="eur", leg2_currency="usd", curves=["eureur", "eurusd", "usdusd", "usdusd"]),
    DataFrame([[1000000.0, -1100306.44592],
               [0.0, -2377.85237],
               [-2042.44624, 4630.97800],
               [0.0, -2152.15417],
               [-1844.59236, 4191.00589],
               [-1000000, 1104836.45246],
               [-2042.44624, 4650.04393]],
              index=Index([dt(2022, 1, 5), dt(2022, 2, 5), dt(2022, 2, 7),
                            dt(2022, 3, 5), dt(2022, 3, 7), dt(2022, 4, 5),
                            dt(2022, 4, 7)], name="payment"),
              columns=MultiIndex.from_tuples([("EUR", "usd"), ("USD", "usd")],
                                             names=["local_ccy", "collateral_ccy"])
              )
    ),
    (
    FXSwap(dt(2022, 1, 5), "3M", currency="eur", leg2_currency="usd", curves=["eureur", "eurusd", "usdusd", "usdusd"]),
    DataFrame([[1000000.0, -1100306.44592],
               [-1005943.73163, 1113805.13741]],
              index=Index([dt(2022, 1, 5), dt(2022, 4, 5)], name="payment"),
              columns=MultiIndex.from_tuples([("EUR", "usd"), ("USD", "usd")],
                                             names=["local_ccy", "collateral_ccy"])
              )
    ),
])
def test_fx_settlements_table(inst, expected):
    usdusd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.95}, id="usdusd")
    eureur = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.975}, id="eureur")
    eurusd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.976}, id="eurusd")
    fxr = FXRates({"eurusd": 1.1}, settlement=dt(2022, 1, 1))
    fxf = FXForwards(
        fx_rates=fxr,
        fx_curves={
            "usdusd": usdusd,
            "eureur": eureur,
            "eurusd": eurusd,
        }
    )
    usdeur = fxf.curve("usd", "eur", id="usdeur")
    eur_eurusd = fxf.curve("eur", ["usd", "eur"], id="eur_eurusd")

    solver = Solver(
        curves=[usdusd, eureur, eurusd, usdeur, eur_eurusd],
        instruments=[
            IRS(dt(2022, 1, 1), "1y", "A", curves=usdusd),
            IRS(dt(2022, 1, 1), "1y", "A", curves=eureur),
            XCS(dt(2022, 1, 1), "1y", "Q", currency="eur", leg2_currency="usd", curves=[eureur, eurusd, usdusd, usdusd]),
        ],
        s=[5.0, 2.5, -10],
        fx=fxf,
    )
    assert eureur.collateral == "eur"  # collateral tags populated by FXForwards

    pf = Portfolio([inst])
    result = pf.cashflows_table(solver=solver)
    assert_frame_equal(expected, result, atol=1e-4)

    result = inst.cashflows_table(solver=solver)
    assert_frame_equal(expected, result, atol=1e-4)


def test_fx_settlements_table_no_fxf():
    solver = Solver(
        curves=[Curve({dt(2023, 8, 1): 1.0, dt(2024, 8, 1): 1.0}, id="usd")],
        instruments=[IRS(dt(2023, 8, 1), "1Y", "Q", curves="usd")],
        s=[2.0],
        instrument_labels=["1Y"],
        id="us_rates",
        algorithm="gauss_newton",
    )
    irs_mkt = IRS(
        dt(2023, 8, 1), "1Y", "Q", curves="usd", fixed_rate=2.0, notional=999556779.81,
    )
    result = irs_mkt.cashflows_table(solver=solver)
    assert abs(result.iloc[0, 0] - 69.49810) < 1e-5
    assert abs(result.iloc[3, 0] - 69.49810) < 1e-5
