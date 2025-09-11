import os
from datetime import datetime as dt

import numpy as np
import pytest
from pandas import DataFrame, Index, MultiIndex, Series, isna
from pandas.testing import assert_frame_equal
from rateslib import default_context, defaults
from rateslib.curves import CompositeCurve, Curve, LineCurve, MultiCsaCurve
from rateslib.curves._parsers import _map_curve_from_solver
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, Variable, dual_exp, gradient
from rateslib.fx import FXForwards, FXRates
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface, FXSabrSmile, FXSabrSurface
from rateslib.instruments import (
    CDS,
    FRA,
    IIRS,
    IRS,
    NDF,
    SBS,
    XCS,
    ZCIS,
    ZCS,
    Bill,
    FixedRateBond,
    FloatRateNote,
    Fly,
    FXBrokerFly,
    FXCall,
    FXExchange,
    FXPut,
    FXRiskReversal,
    FXStraddle,
    FXStrangle,
    FXSwap,
    IndexFixedRateBond,
    Portfolio,
    Spread,
    STIRFuture,
    Value,
    VolValue,
)
from rateslib.instruments.utils import (
    _get_curves_fx_and_base_maybe_from_solver,
)
from rateslib.legs.base import Amortization
from rateslib.scheduling import Adjuster, NamedCal, Schedule, add_tenor
from rateslib.solver import Solver


@pytest.fixture
def curve():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.99,
        dt(2022, 7, 1): 0.98,
        dt(2022, 10, 1): 0.97,
    }
    # convention = "Act360"
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture
def curve2():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.98,
        dt(2022, 7, 1): 0.97,
        dt(2022, 10, 1): 0.95,
    }
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture
def usdusd():
    nodes = {dt(2022, 1, 1): 1.00, dt(2022, 4, 1): 0.99}
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture
def eureur():
    nodes = {dt(2022, 1, 1): 1.00, dt(2022, 4, 1): 0.997}
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture
def usdeur():
    nodes = {dt(2022, 1, 1): 1.00, dt(2022, 4, 1): 0.996}
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture
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


@pytest.mark.parametrize(
    "inst",
    [
        IRS(dt(2022, 7, 1), "3M", "A", curves="eureur", notional=1e6),
        STIRFuture(
            dt(2022, 3, 16),
            dt(2022, 6, 15),
            "Q",
            curves="eureur",
            bp_value=25.0,
            contracts=-1,
        ),
        FRA(dt(2022, 7, 1), "3M", "A", curves="eureur", notional=1e6),
        SBS(
            dt(2022, 7, 1),
            "3M",
            "A",
            curves=["eureur", "eureur", "eurusd", "eureur"],
            notional=-1e6,
        ),
        ZCS(dt(2022, 7, 1), "3M", "A", curves="eureur", notional=1e6),
        ZCIS(dt(2022, 1, 1), "1Y", "A", curves=["usdusd", "usdusd", "eu_cpi", "usdusd"]),
        IIRS(
            dt(2022, 7, 1),
            "3M",
            "A",
            curves=["eu_cpi", "eureur", "eureur", "eureur"],
            notional=1e6,
        ),
        XCS(  # XCS - FloatFloat
            dt(2022, 7, 1),
            "3M",
            "A",
            currency="usd",
            leg2_currency="eur",
            curves=["usdusd", "usdusd", "eureur", "eurusd"],
            notional=1e6,
        ),
        FXSwap(
            dt(2022, 7, 1),
            "3M",
            currency="usd",
            leg2_currency="eur",
            curves=["usdusd", "usdusd", "eureur", "eureur"],
            notional=-1e6,
        ),
        FXExchange(
            settlement=dt(2022, 10, 1),
            pair="eurusd",
            curves=[None, "eureur", None, "usdusd"],
            notional=-1e6 * 25 / 74.27,
        ),
    ],
)
def test_instrument_repr(inst):
    result = inst.__repr__()
    expected = f"<rl.{type(inst).__name__} at {hex(id(inst))}>"
    assert result == expected


class TestCurvesandSolver:
    def test_get_curve_from_solver(self) -> None:
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="tagged")
        inst = [(Value(dt(2023, 1, 1)), ("tagged",), {})]
        solver = Solver([curve], [], inst, [0.975])

        result = _map_curve_from_solver("tagged", solver)
        assert result == curve

        result = _map_curve_from_solver(curve, solver)
        assert result == curve

        no_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="not in solver")

        with default_context("curve_not_in_solver", "ignore"):
            result = _map_curve_from_solver(no_curve, solver)
            assert result == no_curve

        with pytest.warns(), default_context("curve_not_in_solver", "warn"):
            result = _map_curve_from_solver(no_curve, solver)
            assert result == no_curve

        with (
            pytest.raises(ValueError, match="`curve` must be in `solver`"),
            default_context("curve_not_in_solver", "raise"),
        ):
            _map_curve_from_solver(no_curve, solver)

        with pytest.raises(AttributeError, match="`curve` has no attribute `id`, likely it not"):
            _map_curve_from_solver(100.0, solver)

    @pytest.mark.parametrize("solver", [True, False])
    @pytest.mark.parametrize("fxf", [True, False])
    @pytest.mark.parametrize("fx", [NoInput(0), 2.0])
    @pytest.mark.parametrize("crv", [True, False])
    def test_get_curves_and_fx_from_solver(
        self,
        usdusd,
        usdeur,
        eureur,
        solver,
        fxf,
        fx,
        crv,
    ) -> None:
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="tagged")
        inst = [(Value(dt(2023, 1, 1)), ("tagged",), {})]
        fxfs = FXForwards(
            FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3)),
            {"usdusd": usdusd, "usdeur": usdeur, "eureur": eureur},
        )
        solver = (
            Solver([curve], [], inst, [0.975], fx=fxfs if fxf else NoInput(0))
            if solver
            else NoInput(0)
        )
        curve = curve if crv else NoInput(0)

        if solver is not NoInput(0) and fxf and fx is not NoInput(0):
            with pytest.warns(UserWarning):
                #  Solver contains an `fx` attribute but an `fx` argument has been supplied
                crv_result, fx_result, _ = _get_curves_fx_and_base_maybe_from_solver(
                    NoInput(0),
                    solver,
                    curve,
                    fx,
                    NoInput(0),
                    "usd",
                )
        else:
            crv_result, fx_result, _ = _get_curves_fx_and_base_maybe_from_solver(
                NoInput(0),
                solver,
                curve,
                fx,
                NoInput(0),
                "usd",
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

    @pytest.mark.parametrize(
        "obj",
        [
            (Curve({dt(2000, 1, 1): 1.0})),
            (LineCurve({dt(2000, 1, 1): 1.0})),
            (Curve({dt(2000, 1, 1): 1.0}, index_base=100.0)),
            (CompositeCurve([Curve({dt(2000, 1, 1): 1.0})])),
            (MultiCsaCurve([Curve({dt(2000, 1, 1): 1.0})])),
            (
                FXDeltaVolSmile(
                    {0.1: 1.0, 0.2: 2.0, 0.5: 3.0, 0.7: 4.0, 0.9: 5.0},
                    dt(2023, 3, 16),
                    dt(2023, 6, 16),
                    "forward",
                )
            ),
        ],
    )
    def test_get_curves_fx_and_base_maybe_from_solver_object_types(self, obj) -> None:
        crv_result, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            obj,
            NoInput(0),
            NoInput(0),
            NoInput(0),
            NoInput(0),
            NoInput(0),
        )
        assert crv_result == (obj,) * 4

    def test_get_curves_and_fx_from_solver_raises(self) -> None:
        from rateslib.solver import Solver

        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="tagged")
        inst = [(Value(dt(2023, 1, 1)), ("tagged",), {})]
        solver = Solver([curve], [], inst, [0.975])

        with pytest.raises(ValueError, match="`curves` must contain Curve, not str, if"):
            _get_curves_fx_and_base_maybe_from_solver(
                NoInput(0),
                NoInput(0),
                "tagged",
                NoInput(0),
                NoInput(0),
                "",
            )

        with pytest.raises(ValueError, match="`curves` must contain str curve `id` s"):
            _get_curves_fx_and_base_maybe_from_solver(
                NoInput(0),
                solver,
                "bad_id",
                NoInput(0),
                NoInput(0),
                "",
            )

        with pytest.raises(ValueError, match="Can only supply a maximum of 4 `curves`"):
            _get_curves_fx_and_base_maybe_from_solver(
                NoInput(0),
                solver,
                ["tagged"] * 5,
                NoInput(0),
                NoInput(0),
                "",
            )

    @pytest.mark.parametrize("num", [1, 2, 3, 4])
    def test_get_curves_from_solver_multiply(self, num) -> None:
        from rateslib.solver import Solver

        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="tagged")
        inst = [(Value(dt(2023, 1, 1)), ("tagged",), {})]
        solver = Solver([curve], [], inst, [0.975])
        result, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            NoInput(0),
            solver,
            ["tagged"] * num,
            NoInput(0),
            NoInput(0),
            "",
        )
        assert result == (curve, curve, curve, curve)

    def test_get_proxy_curve_from_solver(self, usdusd, usdeur, eureur) -> None:
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="tagged")
        inst = [(Value(dt(2023, 1, 1)), ("tagged",), {})]
        fxf = FXForwards(
            FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3)),
            {"usdusd": usdusd, "usdeur": usdeur, "eureur": eureur},
        )
        solver = Solver([curve], [], inst, [0.975], fx=fxf)
        curve = fxf.curve("eur", "usd")
        irs = IRS(dt(2022, 1, 1), "3m", "Q")

        # test the curve will return even though it is not included within the solver
        # because it is a proxy curve.
        irs.npv(curves=curve, solver=solver)

    def test_ambiguous_curve_in_out_id_solver_raises(self) -> None:
        curve = Curve({dt(2022, 1, 1): 1.0}, id="cloned-id")
        curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="cloned-id")
        solver = Solver(
            curves=[curve2],
            instruments=[IRS(dt(2022, 1, 1), "1y", "A", curves="cloned-id")],
            s=[5.0],
        )
        irs = IRS(dt(2022, 1, 1), "1y", "A", fixed_rate=2.0)
        with pytest.raises(ValueError, match="A curve has been supplied, as part of ``curves``,"):
            irs.npv(curves=curve, solver=solver)

    def test_get_multicsa_curve_from_solver(self, usdusd, usdeur, eureur) -> None:
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="tagged")
        inst = [(Value(dt(2023, 1, 1)), ("tagged",), {})]
        fxf = FXForwards(
            FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3)),
            {"usdusd": usdusd, "usdeur": usdeur, "eureur": eureur},
        )
        solver = Solver([curve], [], inst, [0.975], fx=fxf)
        curve = fxf.curve("eur", ("usd", "eur"))
        irs = IRS(dt(2022, 1, 1), "3m", "Q")

        # test the curve will return even though it is not included within the solver
        # because it is a proxy curve.
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

    def test_base_and_fx(self) -> None:
        # calculable since base aligns with local currency
        result = self.irs.npv(fx=self.fxr, base="eur")
        expected = 330.4051154763001 / 1.1
        assert abs(result - expected) < 1e-4

        with pytest.warns(UserWarning):
            # warn about numeric
            self.irs.npv(fx=1 / 1.1, base="eur")

        # raises because no FX data to calculate a conversion
        with pytest.raises(KeyError, match="'usd'"):
            self.irs.npv(fx=FXRates({"eurgbp": 1.1}), base="eur")

    def test_base_and_solverfx(self) -> None:
        # should take fx from solver and calculated
        self.solver.fx = FXRates({"eurusd": 1.1})
        self.solver._set_new_state()
        result = self.irs.npv(solver=self.solver, base="eur")
        expected = 330.4051154763001 / 1.1
        assert abs(result - expected) < 1e-4
        self.solver.fx = NoInput(0)

    def test_base_and_fx_and_solverfx(self) -> None:
        # should take fx and ignore solver.fx
        fxr = FXRates({"eurusd": 1.2})
        self.solver.fx = fxr
        self.solver._set_new_state()

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

    def test_base_only(self) -> None:
        # calculable since base aligns with local currency
        result = self.irs.npv(base="usd")
        expected = 330.4051154763001
        assert abs(result - expected) < 1e-4

        # raises becuase no FX data to calculate a conversion
        with pytest.raises(ValueError, match="`base` "):
            result = self.irs.npv(base="eur")

    def test_base_solvernofx(self) -> None:
        # calculable since base aligns with local currency
        result = self.irs.npv(base="usd", solver=self.solver)
        expected = 330.4051154763001
        assert abs(result - expected) < 1e-4

        # raises becuase no FX data to calculate a conversion
        with pytest.raises(ValueError, match="`base` "):
            result = self.irs.npv(base="eur", solver=self.solver)

    # ``base`` is inferred

    def test_no_args(self) -> None:
        # should result in a local NPV calculation
        result = self.irs.npv()
        expected = 330.4051154763001
        assert abs(result - expected) < 1e-4

    def test_fx(self) -> None:
        # should repeat the "_just_base" case.
        result = self.irs.npv(fx=self.fxr)
        expected = 330.4051154763001 / 1.25
        assert abs(result - expected) < 1e-4

    def test_fx_solverfx(self) -> None:
        fxr = FXRates({"eurusd": 1.2}, base="eur")
        self.solver.fx = fxr
        self.solver._set_new_state()

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

    def test_solverfx(self) -> None:
        fxr = FXRates({"eurusd": 1.2}, base="eur")
        self.solver.fx = fxr
        self.solver._set_new_state()

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
            CDS(
                dt(2022, 7, 1), "3M", "Q", curves=["eureur", "usdusd"], notional=1e6 * 25 / 14.91357
            ),
            IRS(dt(2022, 7, 1), "3M", "A", curves="eureur", notional=1e6),
            STIRFuture(
                dt(2022, 3, 16),
                dt(2022, 6, 15),
                "Q",
                curves="eureur",
                bp_value=25.0,
                contracts=-1,
            ),
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
            XCS(  # XCS-FixedFloatNonMtm
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
                pair="eurusd",
                curves=[None, "eureur", None, "usdusd"],
                notional=-1e6 * 25 / 74.27,
            ),
        ],
    )
    def test_null_priced_delta(self, inst) -> None:
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="usdusd")
        c2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="eureur")
        c3 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.982}, id="eurusd")
        c4 = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.995},
            id="eu_cpi",
            index_base=100.0,
            interpolation="linear_index",
            index_lag=3,
        )
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
        assert abs(result.iloc[0, 0] - 25.0) < 1.0
        result2 = inst.npv(solver=solver)
        assert abs(result2) < 1e-3

        # test that instruments have not been set by the previous pricing action
        solver.s = [1.3, 1.4, 1.36, 0.55]
        solver.iterate()
        result3 = inst.npv(solver=solver)
        assert abs(result3) < 1e-3

    @pytest.mark.parametrize(
        "inst",
        [
            NDF(
                pair="eurusd",
                notional=1e6 * 0.333,
                settlement=dt(2022, 10, 1),
                curves="usdusd",
            )
        ],
    )
    def test_null_priced_delta2(self, inst) -> None:
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="usdusd")
        c2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="eureur")
        c3 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.982}, id="eurusd")
        c4 = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.995},
            id="eu_cpi",
            index_base=100.0,
            interpolation="linear_index",
            index_lag=3,
        )
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
        assert abs(result.iloc[1, 0] - 25.0) < 1.0
        result2 = inst.npv(solver=solver)
        assert abs(result2) < 1e-3

        # test that instruments have not been set by the previous pricing action
        solver.s = [1.3, 1.4, 1.36, 0.55]
        solver.iterate()
        result3 = inst.npv(solver=solver)
        assert abs(result3) < 1e-3

    @pytest.mark.parametrize(
        "inst",
        [
            NDF(
                pair="eurusd",
                notional=1e6 * 0.333,
                settlement=dt(2022, 10, 1),
                curves="usdusd",
            )
        ],
    )
    def test_null_priced_gamma2(self, inst) -> None:
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="usdusd")
        c2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="eureur")
        c3 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.982}, id="eurusd")
        c4 = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.995},
            id="eu_cpi",
            index_base=100.0,
            interpolation="linear_index",
            index_lag=3,
        )
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
        result = inst.gamma(solver=solver)
        assert isinstance(result, DataFrame)

    @pytest.mark.parametrize(
        ("inst", "param"),
        [
            (IRS(dt(2022, 7, 1), "3M", "A", curves="usdusd"), "fixed_rate"),
            (FRA(dt(2022, 7, 1), "3M", "Q", curves="usdusd"), "fixed_rate"),
            (
                SBS(dt(2022, 7, 1), "3M", "Q", curves=["usdusd", "usdusd", "eureur", "usdusd"]),
                "float_spread",
            ),
            (ZCS(dt(2022, 1, 1), "1Y", "Q", curves=["usdusd"]), "fixed_rate"),
            (
                ZCIS(dt(2022, 1, 1), "1Y", "A", curves=["usdusd", "usdusd", "eu_cpi", "usdusd"]),
                "fixed_rate",
            ),
            (
                IIRS(dt(2022, 1, 1), "1Y", "Q", curves=["eu_cpi", "usdusd", "usdusd", "usdusd"]),
                "fixed_rate",
            ),
            (
                FXExchange(
                    dt(2022, 3, 1),
                    pair="usdeur",
                    curves=[NoInput(0), "usdusd", NoInput(0), "eurusd"],
                ),
                "fx_rate",
            ),
        ],
    )
    def test_null_priced_delta_round_trip_one_pricing_param(self, inst, param) -> None:
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="usdusd")
        c2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="eureur")
        c3 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.982}, id="eurusd")
        c4 = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.995},
            id="eu_cpi",
            index_base=100.0,
            interpolation="linear_index",
            index_lag=3,
        )
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

    @pytest.mark.parametrize(
        ("inst", "param"),
        [
            (
                FXSwap(
                    dt(2022, 2, 1),
                    "3M",
                    currency="eur",
                    leg2_currency="usd",
                    curves=[NoInput(0), "eurusd", NoInput(0), "usdusd"],
                ),
                "points",
            ),
        ],
    )
    def test_null_priced_delta_round_trip_one_pricing_param_fx_fix(self, inst, param) -> None:
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="usdusd")
        c2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="eureur")
        c3 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.982}, id="eurusd")
        c4 = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.995},
            id="eu_cpi",
            index_base=100.0,
            interpolation="linear_index",
            index_lag=3,
        )
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

    @pytest.mark.parametrize(
        "inst",
        [
            CDS(dt(2022, 7, 1), "3M", "Q", notional=1e6 * 25 / 14.91357),
            IRS(dt(2022, 7, 1), "3M", "A", notional=1e6),
            STIRFuture(
                dt(2022, 3, 16),
                dt(2022, 6, 15),
                "Q",
                bp_value=25.0,
                contracts=-1,
            ),
            FRA(dt(2022, 7, 1), "3M", "A", notional=1e6),
            SBS(
                dt(2022, 7, 1),
                "3M",
                "A",
                notional=-1e6,
            ),
            ZCS(dt(2022, 7, 1), "3M", "A", notional=1e6),
            IIRS(
                dt(2022, 7, 1),
                "3M",
                "A",
                notional=1e6,
            ),
            IIRS(
                dt(2022, 7, 1),
                "3M",
                "A",
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
                notional=1e6,
            ),
            XCS(  # XCS-FixedFloatNonMtm
                dt(2022, 7, 1),
                "3M",
                "A",
                fixed=True,
                leg2_fixed=False,
                leg2_mtm=False,
                currency="eur",
                leg2_currency="usd",
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
                leg2_fixed_rate=1.2,
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
                notional=1e6,
            ),
            FXSwap(
                dt(2022, 7, 1),
                "3M",
                currency="usd",
                leg2_currency="eur",
                notional=-1e6,
                # fx_fixing=0.999851,
                # split_notional=1003052.812,
                # points=2.523505,
            ),
            FXExchange(
                settlement=dt(2022, 10, 1),
                pair="eurusd",
                notional=-1e6 * 25 / 74.27,
            ),
            NDF(
                pair="eurusd",  # settlement currency defaults to right hand side: usd
                settlement=dt(2022, 10, 1),
            ),
        ],
    )
    def test_set_pricing_does_not_overwrite_unpriced_status(self, inst):
        # unpriced instruments run a `set_pricing_mid` method
        # this test ensures that after that run the price is not permanently set and
        # will reset when priced from an alternative set of curves.
        # test is slightly different to null_priced_delta: uses fx and includes rate call
        curve1 = Curve({dt(2022, 1, 1): 1.0, dt(2024, 1, 1): 0.99}, index_base=66)
        curve2 = Curve({dt(2022, 1, 1): 1.0, dt(2024, 1, 1): 0.98}, index_base=66)
        curve3 = Curve({dt(2022, 1, 1): 1.0, dt(2024, 1, 1): 0.97})
        curve4 = Curve({dt(2022, 1, 1): 1.0, dt(2024, 1, 1): 0.96}, index_base=80)
        curve5 = Curve({dt(2022, 1, 1): 1.0, dt(2024, 1, 1): 0.95}, index_base=80)
        curve6 = Curve({dt(2022, 1, 1): 1.0, dt(2024, 1, 1): 0.94})
        fxr1 = FXRates({"eurusd": 1.0}, settlement=dt(2022, 1, 1))
        fxr2 = FXRates({"eurusd": 1.5}, settlement=dt(2022, 1, 1))
        fxf1 = FXForwards(fxr1, {"usdusd": curve1, "eureur": curve2, "eurusd": curve3})
        fxf2 = FXForwards(fxr2, {"usdusd": curve4, "eureur": curve5, "eurusd": curve6})

        rate1 = inst.rate(curves=[curve1, curve1, curve2, curve3], fx=fxf1)
        npv1 = inst.npv(curves=[curve1, curve1, curve2, curve3], fx=fxf1)
        assert abs(npv1) < 1e-8

        rate2 = inst.rate(curves=[curve4, curve4, curve5, curve6], fx=fxf2)
        npv2 = inst.npv(curves=[curve4, curve4, curve5, curve6], fx=fxf2)
        assert rate1 != rate2
        assert abs(npv2) < 1e-8


class TestIRS:
    @pytest.mark.parametrize(
        ("float_spread", "fixed_rate", "expected"),
        [
            (0, 4.03, 4.03637780),
            (3, 4.03, 4.06637780),
            (0, 5.10, 4.03637780),
        ],
    )
    def test_irs_rate(self, curve, float_spread, fixed_rate, expected) -> None:
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
        ("float_spread", "fixed_rate", "expected"),
        [
            (0, 4.03, -0.63777963),
            (200, 4.03, -0.63777963),
            (500, 4.03, -0.63777963),
            (0, 4.01, -2.63777963),
        ],
    )
    def test_irs_spread_none_simple(self, curve, float_spread, fixed_rate, expected) -> None:
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
        ("float_spread", "fixed_rate", "expected"),
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
    def test_irs_spread_isda_compound(self, curve, float_spread, fixed_rate, expected) -> None:
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
        ("float_spread", "fixed_rate", "expected"),
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
    def test_irs_spread_isda_flat_compound(self, curve, float_spread, fixed_rate, expected) -> None:
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

    def test_irs_npv(self, curve) -> None:
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

    def test_irs_cashflows(self, curve) -> None:
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

    def test_irs_npv_mid_mkt_zero(self, curve) -> None:
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

    def test_sbs_float_spread_raises(self, curve) -> None:
        irs = IRS(dt(2022, 1, 1), "9M", "Q")
        with pytest.raises(AttributeError, match="Cannot set `float_spread`"):
            irs.float_spread = 1.0

    def test_index_base_raises(self) -> None:
        irs = IRS(dt(2022, 1, 1), "9M", "Q")
        with pytest.raises(AttributeError, match="Cannot set `index_base`"):
            irs.index_base = 1.0

        with pytest.raises(AttributeError, match="Cannot set `leg2_index_base`"):
            irs.leg2_index_base = 1.0

    def test_irs_interpolated_stubs(self, curve) -> None:
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

    def test_irs_interpolated_stubs_solver(self) -> None:
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

    def test_no_rfr_fixings_raises(self) -> None:
        # GH 170
        T_irs = IRS(
            effective=dt(2020, 12, 15),
            termination=dt(2037, 12, 15),
            notional=-600e6,
            frequency="A",
            leg2_frequency="A",
            fixed_rate=4.5,
            curves="curve",
        )
        par_curve = Curve(
            nodes={
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 1.0,
                dt(2024, 1, 1): 1.0,
                dt(2025, 1, 1): 1.0,
            },
            id="curve",
        )
        with pytest.raises(ValueError, match="RFRs could not be calculated, have you missed "):
            T_irs.cashflows(curves=par_curve)
        with pytest.raises(ValueError, match="RFRs could not be calculated, have you missed "):
            T_irs.npv(curves=par_curve)

    def test_no_rfr_fixings_raises2(self) -> None:
        # GH 357
        sofr = Curve(
            id="sofr",
            convention="Act360",
            calendar="nyc",
            modifier="MF",
            interpolation="log_linear",
            nodes={
                dt(2023, 8, 21): 1.0,
                dt(2026, 8, 25): 0.97,
            },
        )
        irs = IRS(
            effective=dt(2023, 8, 18),
            termination=dt(2025, 8, 18),
            notional=1e6,
            curves=sofr,
            fixed_rate=4.86,
            spec="usd_irs",
        )
        with pytest.raises(ValueError, match="RFRs could not be calculated, have you missed "):
            irs.npv()

    def test_1b_tenor_swaps(self):
        irs = IRS(dt(2024, 12, 30), "1b", spec="sek_irs")  # 31st is a holiday.
        assert irs.leg1.schedule.uschedule == [dt(2024, 12, 30), dt(2025, 1, 2)]

    def test_1d_tenor_swaps(self):
        irs = IRS(dt(2024, 12, 30), "1d", spec="sek_irs")  # 31st is a holiday.
        assert irs.leg1.schedule.uschedule == [dt(2024, 12, 30), dt(2025, 1, 2)]

    def test_fixings_table(self, curve):
        irs = IRS(dt(2022, 1, 15), "6m", spec="usd_irs", curves=curve)
        result = irs.fixings_table()
        assert isinstance(result, DataFrame)

    def test_1d_instruments(self):
        # GH484
        with pytest.raises(ValueError, match="A Schedule could not be generated from the pa"):
            IRS(dt(2025, 1, 1), "1d", spec="sek_irs")

    def test_custom_amortization_raises(self):
        with pytest.raises(ValueError, match="Custom amortisation schedules must have `n-1` amort"):
            IRS(dt(2000, 1, 1), dt(2000, 4, 1), "M", notional=1000, amortization=[100, 400, 50])

    def test_custom_amortization(self):
        irs = IRS(dt(2000, 1, 1), dt(2000, 5, 1), "M", notional=1000, amortization=[100, 400, 50])
        assert irs.leg1.amortization.outstanding == (1000.0, 900.0, 500.0, 450.0)
        assert irs.leg1.amortization.amortization == (100.0, 400.0, 50.0)
        assert irs.leg2.amortization.outstanding == (-1000.0, -900.0, -500.0, -450.0)
        assert irs.leg2.amortization.amortization == (-100.0, -400.0, -50.0)

    def test_custom_amortization_as_object(self):
        # test an Amortization object can be passed and is negated correctly
        amort = Amortization(4, 1000.0, [100.0, 400.0, 50.0])
        irs = IRS(dt(2000, 1, 1), dt(2000, 5, 1), "M", notional=1000, amortization=amort)
        assert irs.leg1.amortization.outstanding == (1000.0, 900.0, 500.0, 450.0)
        assert irs.leg1.amortization.amortization == (100.0, 400.0, 50.0)
        assert irs.leg2.amortization.outstanding == (-1000.0, -900.0, -500.0, -450.0)
        assert irs.leg2.amortization.amortization == (-100.0, -400.0, -50.0)


class TestIIRS:
    def test_index_base_none_populated(self, curve) -> None:
        i_curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.5, dt(2034, 1, 1): 0.4},
            index_lag=3,
            index_base=100.0,
            interpolation_method="linear_index",
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

    def test_iirs_npv_mid_mkt_zero(self, curve) -> None:
        i_curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.5, dt(2034, 1, 1): 0.4},
            index_lag=3,
            index_base=100.0,
            interpolation="linear_index",
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

    def test_cashflows(self, curve) -> None:
        i_curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 2, 1): 0.99},
            index_lag=3,
            index_base=100.0,
            interpolation="linear_index",
        )
        iirs = IIRS(
            effective=dt(2022, 2, 1),
            termination="9M",
            frequency="Q",
            index_base=Series([100.0], index=[dt(2021, 11, 1)]),
            index_fixings=Series([110.0, 115], index=[dt(2022, 2, 1), dt(2022, 5, 1)]),
            index_lag=3,
            index_method="monthly",
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
                [("leg1", 0), ("leg1", 1), ("leg1", 2), ("leg2", 0), ("leg2", 1), ("leg2", 2)],
            ),
        )
        assert_frame_equal(
            expected,
            result[["Index Val", "Index Ratio", "NPV", "Type"]],
            rtol=1e-3,
        )

    def test_npv_no_index_base(self, curve) -> None:
        i_curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.5, dt(2034, 1, 1): 0.4},
            index_lag=3,
            index_base=100.0,
            interpolation="linear_index",
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

    def test_cashflows_no_index_base(self, curve) -> None:
        i_curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2022, 2, 1): 0.5, dt(2034, 1, 1): 0.4},
            index_lag=3,
            index_base=100.0,
            interpolation="linear_index",
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

    def test_fixings_table(self, curve):
        iirs = IIRS(dt(2022, 1, 15), "6m", "Q", curves=curve)
        result = iirs.fixings_table()
        assert isinstance(result, DataFrame)

    def test_fixing_in_the_past(self):
        # this test will also initialise `index_base` from the provided `index_fixings`
        discount = Curve({dt(2025, 5, 15): 1.0, dt(2027, 5, 15): 0.96})
        inflation = Curve(
            {dt(2025, 4, 1): 1.0, dt(2027, 5, 1): 0.98}, index_base=100.0, index_lag=0
        )
        fixings = Series(
            [97, 98, 99, 100.0],
            index=[dt(2025, 1, 1), dt(2025, 2, 1), dt(2025, 3, 1), dt(2025, 4, 1)],
        )
        iirs = IIRS(dt(2025, 5, 15), "1y", "Q", index_fixings=fixings)
        result = iirs.rate(curves=[inflation, discount])
        assert abs(result - 0.938782232) < 1e-8


class TestSBS:
    def test_sbs_npv(self, curve) -> None:
        sbs = SBS(dt(2022, 1, 1), "9M", "Q", float_spread=3.0)
        a_delta = sbs.analytic_delta(curve, curve, leg=1)
        npv = sbs.npv(curve)
        assert abs(npv + 3.0 * a_delta) < 1e-9

        sbs.leg2_float_spread = 4.5
        npv = sbs.npv(curve)
        assert abs(npv - 1.5 * a_delta) < 1e-9

    def test_sbs_rate(self, curve) -> None:
        sbs = SBS(dt(2022, 1, 1), "9M", "Q", float_spread=3.0)
        result = sbs.rate([curve], leg=1)
        alias = sbs.spread([curve], leg=1)
        assert abs(result - 0) < 1e-8
        assert abs(alias - 0) < 1e-8

        result = sbs.rate([curve], leg=2)
        alias = sbs.rate([curve], leg=2)
        assert abs(result - 3.0) < 1e-8
        assert abs(alias - 3.0) < 1e-8

    def test_sbs_cashflows(self, curve) -> None:
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

    def test_sbs_fixed_rate_raises(self, curve) -> None:
        sbs = SBS(dt(2022, 1, 1), "9M", "Q", float_spread=3.0)
        with pytest.raises(AttributeError, match="Cannot set `fixed_rate`"):
            sbs.fixed_rate = 1.0

        with pytest.raises(AttributeError, match="Cannot set `leg2_fixed_rate`"):
            sbs.leg2_fixed_rate = 1.0

    def test_fixings_table(self, curve):
        inst = SBS(dt(2022, 1, 15), "6m", spec="usd_irs", curves=curve)
        result = inst.fixings_table()
        assert isinstance(result, DataFrame)

    def test_fixings_table_3s1s(self, curve, curve2):
        inst = SBS(
            dt(2022, 1, 15),
            "6m",
            fixing_method="ibor",
            method_param=0,
            leg2_fixing_method="ibor",
            leg2_method_param=1,
            frequency="Q",
            leg2_frequency="m",
            curves=[curve, curve, curve2, curve],
        )
        result = inst.fixings_table()
        assert isinstance(result, DataFrame)
        assert len(result.columns) == 8
        assert len(result.index) == 8


class TestFRA:
    def test_fra_rate(self, curve) -> None:
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

    def test_fra_npv(self, curve) -> None:
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

    def test_fra_cashflows(self, curve) -> None:
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

    def test_irs_npv_mid_mkt_zero(self, curve) -> None:
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

    @pytest.mark.parametrize(("eom", "exp"), [(True, dt(2021, 5, 31)), (False, dt(2021, 5, 26))])
    def test_fra_roll_inferral(self, eom, exp) -> None:
        fra = FRA(
            effective=dt(2021, 2, 26),
            termination="3m",
            frequency="Q",
            eom=eom,
            calendar="bus",
        )
        assert fra.leg1.schedule.termination == exp

    def test_imm_dated(self):
        FRA(effective=dt(2024, 12, 18), termination=dt(2025, 3, 19), spec="sek_fra3", roll="imm")

    def test_fra_fixings_table(self, curve) -> None:
        fra = FRA(
            effective=dt(2022, 1, 1),
            termination="6m",
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            modifier="mf",
            frequency="S",
            fixed_rate=4.035,
            curves=curve,
        )
        result = fra.fixings_table()
        assert isinstance(result, DataFrame)

    def test_imm_dated_fixings_table(self, curve):
        # This is an IMM FRA: the DCF is different to standard tenor.
        fra = FRA(
            effective=dt(2024, 12, 18),
            termination=dt(2025, 3, 19),
            spec="sek_fra3",
            roll="imm",
            curves=curve,
            notional=1e9,
        )
        result = fra.fixings_table()
        assert isinstance(result, DataFrame)
        assert abs(result.iloc[0, 0] - 1010998964) < 1


class TestZCS:
    @pytest.mark.parametrize(("freq", "exp"), [("Q", 3.53163356950), ("S", 3.54722411409218)])
    def test_zcs_rate(self, freq, exp) -> None:
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
            modifier="MF",
            currency="usd",
            fixed_rate=4.0,
            convention="Act360",
            notional=100e6,
            curves=["usd"],
        )
        result = zcs.rate(usd)
        assert abs(result - exp) < 1e-7

    def test_zcs_analytic_delta(self) -> None:
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
        expected = 105186.21760654295
        assert abs(result - expected) < 1e-7

    def test_zcs_raise_frequency(self) -> None:
        with pytest.raises(ValueError, match="`frequency` for a ZeroFixedLeg should not be 'Z'."):
            ZCS(
                effective=dt(2022, 1, 5),
                termination="10Y",
                modifier="mf",
                frequency="Z",
                fixed_rate=4.22566695954813,
            )

    def test_fixings_table(self, curve):
        zcs = ZCS(
            effective=dt(2022, 1, 15),
            termination="2y",
            frequency="Q",
            leg2_fixing_method="ibor",
            leg2_method_param=0,
            calendar="all",
            convention="30e360",
            curves=curve,
        )
        result = zcs.fixings_table()
        assert isinstance(result, DataFrame)
        for i in range(8):
            abs(result.iloc[i, 2] - 24.678) < 1e-3


class TestZCIS:
    def test_leg2_index_base(self, curve) -> None:
        i_curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_base=200.0,
            interpolation="linear_index",
            index_lag=3,
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

    def test_solver_failure_unspecified_index_base(self, curve) -> None:
        # GH 349
        curve = Curve({dt(2022, 1, 15): 1.0, dt(2023, 1, 1): 0.98})
        i_curve = Curve(
            {dt(2022, 1, 15): 1.0, dt(2023, 1, 1): 0.99},
            index_base=200.0,
            interpolation="linear_index",
        )
        zcis = ZCIS(
            effective=dt(2022, 1, 15),
            termination="9m",
            frequency="A",
            convention="1+",
            calendar="nyc",
            leg2_index_method="monthly",
            currency="usd",
            curves=[curve, curve, i_curve, curve],
            leg2_index_lag=3,
        )
        with pytest.raises(ValueError, match="Forecasting the `index_base`"):  # noqa: SIM117
            with pytest.warns(UserWarning):
                zcis.rate()

    def test_fixing_in_the_past(self):
        # this test will also initialise `index_base` from the provided `index_fixings`
        discount = Curve({dt(2025, 5, 15): 1.0, dt(2027, 5, 15): 0.96})
        inflation = Curve(
            {dt(2025, 4, 1): 1.0, dt(2027, 5, 1): 0.98}, index_base=100.0, index_lag=0
        )
        fixings = Series(
            [97, 98, 99, 100.0],
            index=[dt(2025, 1, 1), dt(2025, 2, 1), dt(2025, 3, 1), dt(2025, 4, 1)],
        )
        zcis = ZCIS(dt(2025, 5, 15), "1y", spec="eur_zcis", leg2_index_fixings=fixings)
        result = zcis.rate(curves=[inflation, discount])
        assert abs(result - 2.8742266148532813) < 1e-8


class TestValue:
    def test_npv_adelta_cashflows_raises(self) -> None:
        value = Value(dt(2022, 1, 1))
        with pytest.raises(NotImplementedError):
            value.npv()

        with pytest.raises(NotImplementedError):
            value.cashflows()

        with pytest.raises(NotImplementedError):
            value.analytic_delta()

    def test_cc_zero_rate(self, curve) -> None:
        v = Value(effective=dt(2022, 7, 1), convention="act365f", metric="cc_zero_rate")
        result = v.rate(curve)
        expected = 4.074026613753926
        assert result == expected

    def test_on_rate(self, curve) -> None:
        c = Curve({dt(2000, 1, 1): 1.0, dt(2000, 7, 1): 1.0})
        v = Value(effective=dt(2000, 2, 1), metric="o/n_rate")
        result = v.rate(c)
        expected = 0.0
        assert abs(result - expected) < 1e-8

    def test_index_value(self) -> None:
        curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.995},
            id="eu_cpi",
            index_base=100.0,
            interpolation="linear_index",
        )
        v = Value(effective=dt(2022, 7, 1), metric="index_value")
        result = v.rate(curve)
        expected = 100.24919116128588
        assert result == expected

    def test_value_raise(self, curve) -> None:
        with pytest.raises(ValueError):
            Value(effective=dt(2022, 7, 1), metric="bad").rate(curve)


class TestFXExchange:
    def test_cashflows(self) -> None:
        fxe = FXExchange(
            settlement=dt(2022, 10, 1),
            pair="eurusd",
            notional=-1e6,
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
            index=MultiIndex.from_tuples([("leg1", 0), ("leg2", 0)]),
        )
        result = result[["Type", "Period", "Ccy", "Payment", "Notional", "Rate", "Cashflow"]]
        assert_frame_equal(result, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        ("base", "fx"),
        [
            ("eur", 1.20),
            ("usd", 1.20),
            ("eur", FXRates({"eurusd": 1.20})),
        ],
    )
    def test_npv_rate(self, curve, curve2, base, fx) -> None:
        fxe = FXExchange(
            settlement=dt(2022, 3, 1),
            pair="eurusd",
            fx_rate=1.2080131682341035,
        )
        if not isinstance(fx, FXRates):
            with pytest.warns(UserWarning):
                result = fxe.npv(
                    [NoInput(0), curve, NoInput(0), curve2],
                    NoInput(0),
                    fx,
                    base,
                    local=False,
                )
        else:
            result = fxe.npv(
                [NoInput(0), curve, NoInput(0), curve2],
                NoInput(0),
                fx,
                base,
                local=False,
            )
        assert abs(result - 0.0) < 1e-8

    def test_rate(self, curve, curve2) -> None:
        fxe = FXExchange(
            settlement=dt(2022, 3, 1),
            pair="eurusd",
            fx_rate=1.2080131682341035,
        )
        result = fxe.rate([NoInput(0), curve, NoInput(0), curve2], NoInput(0), 1.20)
        expected = 1.2080131682341035
        assert abs(result - expected) < 1e-7

    def test_npv_fx_numeric(self, curve) -> None:
        # This demonstrates the ambiguity and poor practice of
        # using numeric fx as pricing input, although it will return.
        fxe = FXExchange(
            settlement=dt(2022, 3, 1),
            pair="eurusd",
            fx_rate=1.2080131682341035,
            notional=-1e6,
        )
        # result_ = fxe.npv([curve] * 4, fx=2.0, local=True)
        with pytest.warns(UserWarning):
            result = fxe.npv([curve] * 4, fx=2.0)
            expected = -993433.103425 * 2.0 + 1200080.27069
            assert abs(result - expected) < 1e-5

        # with pytest.raises(ValueError, match="Cannot calculate `npv`"):
        #     fxe.npv([curve] * 4, fx=2.0, base="bad")

    def test_npv_no_fx_raises(self, curve) -> None:
        fxe = FXExchange(
            settlement=dt(2022, 3, 1),
            pair="eurusd",
            fx_rate=1.2080131682341035,
        )
        with pytest.raises(ValueError, match="Must have some FX info"):
            fxe.npv(curve)

    def test_notional_direction(self, curve, curve2) -> None:
        fx1 = FXExchange(notional=1e6, pair="eurusd", settlement=dt(2022, 1, 1), fx_rate=1.20)
        fx2 = FXExchange(notional=-1e6, pair="eurusd", settlement=dt(2022, 1, 1), fx_rate=1.30)
        pf = Portfolio([fx1, fx2])
        fx = FXRates({"eurusd": 1.30}, base="usd")
        result = pf.npv(curves=[None, curve, None, curve2], fx=fx)
        expected = 100000.0
        assert abs(result - expected) < 1e-8
        result = pf.npv(curves=[None, curve, None, curve2], fx=fx, base="eur")
        expected = 100000.0 / 1.30
        assert abs(result - expected) < 1e-8

    def test_no_defined_analytic_delta(self) -> None:
        with pytest.raises(NotImplementedError):
            FXExchange(
                settlement=dt(2022, 3, 1),
                pair="eurusd",
                fx_rate=1.2080131682341035,
            ).analytic_delta()

    def test_error_msg_for_no_fx(self) -> None:
        eur = Curve({dt(2024, 6, 20): 1.0, dt(2024, 9, 30): 1.0}, calendar="tgt")
        usd = Curve({dt(2024, 6, 20): 1.0, dt(2024, 9, 30): 1.0}, calendar="nyc")
        eurusd = Curve({dt(2024, 6, 20): 1.0, dt(2024, 9, 30): 1.0})
        with pytest.raises(ValueError, match="`fx` must be supplied to price FXExchange"):
            Solver(
                curves=[eur, usd, eurusd],
                instruments=[
                    IRS(dt(2024, 6, 24), "3m", spec="eur_irs", curves=eur),
                    IRS(dt(2024, 6, 24), "3m", spec="usd_irs", curves=usd),
                    FXExchange(
                        pair="eurusd",
                        settlement=dt(2024, 9, 24),
                        curves=[None, eurusd, None, usd],
                    ),
                ],
                s=[3.77, 5.51, 1.0775],
            )


class TestNDF:
    def test_construction(self) -> None:
        ndf = NDF(
            pair="brlusd",
            settlement=dt(2022, 1, 1),
        )
        assert ndf.periods[0].currency == "usd"
        assert ndf.periods[0].reference_currency == "brl"
        assert ndf.periods[0].fx_reversed is False

    def test_construction_reversed(self) -> None:
        ndf = NDF(pair="usdbrl", settlement=dt(2022, 1, 1), currency="usd")
        assert ndf.periods[0].currency == "usd"
        assert ndf.periods[0].reference_currency == "brl"
        assert ndf.periods[0].fx_reversed is True

    @pytest.mark.parametrize(
        ("lag", "eval1", "exp2"),
        [
            (2, dt(2009, 8, 11), dt(2009, 11, 13)),
            (3, dt(2009, 8, 10), dt(2009, 11, 13)),
        ],
    )
    def test_dates(self, lag, eval1, exp2):
        ndf = NDF(
            pair="eurusd",
            settlement="3m",
            eval_date=eval1,
            currency="usd",
            calendar="tgt|fed",
            payment_lag=lag,
        )
        assert ndf.periods[0].payment == exp2

    @pytest.mark.parametrize(
        ("eom", "exp"),
        [
            (True, dt(2025, 5, 30)),
            (False, dt(2025, 5, 28)),
        ],
    )
    def test_roll(self, eom, exp):
        ndf = NDF(
            pair="eurusd",
            settlement="3m",
            eval_date=dt(2025, 2, 26),
            currency="usd",
            calendar="tgt|fed",
            payment_lag=2,
            eom=eom,
        )
        assert ndf.periods[0].payment == exp

    def test_zero_analytic_delta(self):
        ndf = NDF(
            pair="eurusd",
            settlement="3m",
            eval_date=dt(2009, 8, 13),
            currency="usd",
            calendar="tgt|fed",
            payment_lag=2,
        )
        assert ndf.analytic_delta() == 0.0

    def test_bad_currency_raises(self):
        with pytest.raises(ValueError, match="`currency` must be one of the currencies in `pair`."):
            NDF(
                pair="eurusd",
                currency="jpy",
                settlement="3m",
                eval_date=dt(2009, 8, 13),
                calendar="tgt|fed",
                payment_lag=2,
            )

    def test_cashflows(self, usdusd, usdeur, eureur):
        fxf = FXForwards(
            FXRates({"eurusd": 1.02}, settlement=dt(2022, 1, 3)),
            {"eureur": eureur, "usdeur": usdeur, "usdusd": usdusd},
        )
        ndf = NDF(
            pair="eurusd",
            settlement="3m",
            eval_date=dt(2022, 1, 1),
            currency="usd",
            calendar="tgt|fed",
            payment_lag=2,
            fx_rate=1.05,
        )
        result = ndf.cashflows(curves=usdusd, fx=fxf)
        assert result.loc[("leg1", 0), "Type"] == "NonDeliverableCashflow"
        assert result.loc[("leg1", 0), "Period"] == "EURUSD"
        assert result.loc[("leg1", 0), "Ccy"] == "USD"
        assert result.loc[("leg1", 0), "Payment"] == dt(2022, 4, 4)
        assert result.loc[("leg1", 0), "Rate"] == 1.0210354810081033
        assert result.loc[("leg1", 1), "Rate"] == 1.05
        assert result.loc[("leg1", 1), "Notional"] == 1050000.0

    @pytest.mark.parametrize(("base", "expected"), [("eur", -28103.831), ("usd", -28665.269)])
    def test_npv(self, usdusd, usdeur, eureur, base, expected):
        fxf = FXForwards(
            FXRates({"eurusd": 1.02}, settlement=dt(2022, 1, 3)),
            {"eureur": eureur, "usdeur": usdeur, "usdusd": usdusd},
        )
        ndf = NDF(
            pair="eurusd",
            settlement="3m",
            eval_date=dt(2022, 1, 1),
            currency="usd",
            calendar="tgt|fed",
            payment_lag=2,
            fx_rate=1.05,
            notional=1e6,
        )
        result = ndf.npv(curves=usdusd, fx=fxf, base=base)
        assert abs(result - expected) < 1e-3

        expected = {"usd": -28665.269}
        local_result = ndf.npv(curves=usdusd, fx=fxf, base=base, local=True)
        assert len(local_result.keys()) == 1
        assert abs(local_result["usd"] - expected["usd"]) < 1e-3

    @pytest.mark.parametrize(("pair", "rate"), [("eurusd", 1.05), ("usdeur", 0.952380952)])
    def test_npv_direction(self, usdusd, usdeur, eureur, pair, rate):
        fxf = FXForwards(
            FXRates({"eurusd": 1.02}, settlement=dt(2022, 1, 3)),
            {"eureur": eureur, "usdeur": usdeur, "usdusd": usdusd},
        )
        ndf = NDF(
            pair=pair,
            settlement="3m",
            eval_date=dt(2022, 1, 1),
            currency="usd",
            calendar="tgt|fed",
            payment_lag=2,
            fx_rate=rate,
            notional=1e6,
        )
        result = ndf.npv(curves=usdusd, fx=fxf)
        expected = -28665.26900
        assert abs(result - expected) < 1e-3

    @pytest.mark.parametrize(("base", "expected"), [("eur", 0.0), ("usd", 0.0)])
    def test_npv_unpriced(self, usdusd, usdeur, eureur, base, expected):
        fxf = FXForwards(
            FXRates({"eurusd": 1.02}, settlement=dt(2022, 1, 3)),
            {"eureur": eureur, "usdeur": usdeur, "usdusd": usdusd},
        )
        ndf = NDF(
            pair="eurusd",
            settlement="3m",
            eval_date=dt(2022, 1, 1),
            currency="usd",
            calendar="tgt|fed",
            payment_lag=2,
        )
        result = ndf.npv(curves=usdusd, fx=fxf, base=base)
        assert abs(result - expected) < 1e-3

        local_result = ndf.npv(curves=usdusd, fx=fxf, base=base, local=True)
        expected = {"usd": 0.0}
        assert len(local_result.keys()) == 1
        assert abs(local_result["usd"] - expected["usd"]) < 1e-3

    def test_rate(self, usdusd, usdeur, eureur):
        fxf = FXForwards(
            FXRates({"eurusd": 1.02}, settlement=dt(2022, 1, 3)),
            {"eureur": eureur, "usdeur": usdeur, "usdusd": usdusd},
        )
        ndf = NDF(
            pair="eurusd",
            settlement="3m",
            eval_date=dt(2022, 1, 1),
            currency="usd",
            calendar="tgt|fed",
            payment_lag=2,
        )
        result = ndf.rate(curves=usdusd, fx=fxf)
        expected = 1.021035
        assert abs(result - expected) < 1e-6


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
    def test_nonmtmxcs_npv(self, curve, curve2) -> None:
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

    def test_nonmtmxcs_fx_notional(self) -> None:
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
        ("float_spd", "compound", "expected"),
        [
            (10, "none_simple", 10.160794),
            (100, "none_simple", 101.60794),
            (100, "isda_compounding", 101.023590),
            (100, "isda_flat_compounding", 101.336040),
        ],
    )
    def test_nonmtmxcs_spread(self, curve, curve2, float_spd, compound, expected) -> None:
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

    def test_no_fx_raises(self, curve, curve2) -> None:
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

        with (
            pytest.raises(ValueError, match="`fx` is required when `fx_fixings` is"),
            default_context("no_fx_fixings_for_xcs", "raise"),
        ):
            xcs.npv([curve, curve, curve2, curve2])

        with (
            pytest.raises(ValueError, match="`fx` is required when `fx_fixings` is"),
            default_context("no_fx_fixings_for_xcs", "raise"),
        ):
            xcs.cashflows([curve, curve, curve2, curve2])

        # with pytest.warns():
        #     with default_context("no_fx_fixings_for_xcs", "warn"):
        #         xcs.npv([curve, curve, curve2, curve2])

    def test_nonmtmxcs_cashflows(self, curve, curve2) -> None:
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

    @pytest.mark.parametrize("fix", ["fxr", "fxf", "float", "dual", "variable"])
    def test_nonmtm_fx_fixing(self, curve, curve2, fix) -> None:
        fxr = FXRates({"usdnok": 10}, settlement=dt(2022, 1, 1))
        fxf = FXForwards(fxr, {"usdusd": curve, "nokusd": curve2, "noknok": curve2})
        mapping = {
            "fxr": fxr,
            "fxf": fxf,
            "float": 10.0,
            "dual": Dual(10.0, ["x"], []),
            "variable": Variable(10.0, ["x"], []),
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

    def test_nonmtm_fx_fixing_raises_type_crossing(self, curve, curve2):
        fxr = FXRates({"usdnok": 10}, settlement=dt(2022, 1, 1))
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
            fx_fixings=Dual2(10.0, ["x"], [], []),
        )
        # the given fixing is not downcast to Float because it is a specific user provided value.
        # Users should technically use a Variable.
        with pytest.raises(TypeError, match=r"Dual2 operation with incompatible type \(Dual\)"):
            xcs.npv([curve, curve, curve2, curve2], fx=fxr) < 1e-7

    def test_is_priced(self, curve, curve2) -> None:
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

    def test_no_fx_warns(self, curve, curve2) -> None:
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
        with default_context("no_fx_fixings_for_xcs", "warn"), pytest.warns(UserWarning):
            xcs.npv(curves=[curve2, curve2, curve, curve], local=True)

    def test_npv_fx_as_float_valid(self) -> None:
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
        curve = Curve({dt(2022, 2, 1): 1.0, dt(2024, 2, 1): 0.9})
        # TODO(low) this returns a warning with "noknok" for one variety. Should be corrected.
        with pytest.warns(UserWarning):
            result = xcs.npv(curves=curve, fx=10.0)
        assert abs(result) < 1e-6

    def test_npv_fx_as_rates_valid(self) -> None:
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
        curve = Curve({dt(2022, 2, 1): 1.0, dt(2024, 2, 1): 0.9})
        result = xcs.npv(curves=curve, fx=FXRates({"usdnok": 10.0}))
        assert abs(result) < 1e-6

    def test_setting_fx_fixing_no_input(self):
        # Define the interest rate curves for EUR, USD and X-Ccy basis
        usdusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 11, 7): 0.98}, calendar="nyc", id="usdusd")
        eureur = Curve({dt(2024, 5, 7): 1.0, dt(2024, 11, 7): 0.99}, calendar="tgt", id="eureur")
        eurusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 11, 7): 0.992}, id="eurusd")

        # Create an FX Forward market with spot FX rate data
        fxr = FXRates({"eurusd": 1.0760}, settlement=dt(2024, 5, 9))
        fxf = FXForwards(
            fx_rates=fxr,
            fx_curves={"eureur": eureur, "usdusd": usdusd, "eurusd": eurusd},
        )

        xcs = XCS(
            dt(2024, 5, 9),
            "6M",
            "Q",
            fixed=False,
            leg2_fixed=False,
            leg2_mtm=False,
            payment_lag=0,
            currency="eur",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
        )
        xcs.npv(curves=[eureur, eurusd, usdusd, usdusd], fx=fxf)
        assert xcs.leg2.notional == -Dual(1.0760, ["fx_eurusd"], []) * 10e6


class TestNonMtmFixedFloatXCS:
    @pytest.mark.parametrize(
        ("float_spd", "compound", "expected"),
        [
            (10, "none_simple", 6.70955968),
            (100, "isda_compounding", 7.62137047),
        ],
    )
    def test_nonmtmfixxcs_rate_npv(self, curve, curve2, float_spd, compound, expected) -> None:
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

    def test_nonmtmfixxcs_fx_notional(self) -> None:
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

    def test_nonmtmfixxcs_no_fx_raises(self, curve, curve2) -> None:
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

        with (
            pytest.raises(ValueError, match="`fx` is required when `fx_fixings` is"),
            default_context("no_fx_fixings_for_xcs", "raise"),
        ):
            xcs.npv([curve, curve, curve2, curve2])

        with (
            pytest.raises(ValueError, match="`fx` is required when `fx_fixings` is"),
            default_context("no_fx_fixings_for_xcs", "raise"),
        ):
            xcs.cashflows([curve, curve, curve2, curve2])

    def test_nonmtmfixxcs_cashflows(self, curve, curve2) -> None:
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

    @pytest.mark.parametrize("fix", ["fxr", "fxf", "float", "dual", "variable"])
    def test_nonmtmfixxcs_fx_fixing(self, curve, curve2, fix) -> None:
        fxr = FXRates({"usdnok": 10}, settlement=dt(2022, 1, 1))
        fxf = FXForwards(fxr, {"usdusd": curve, "nokusd": curve2, "noknok": curve2})
        mapping = {
            "fxr": fxr,
            "fxf": fxf,
            "float": 10.0,
            "dual": Dual(10.0, ["x"], []),
            "variable": Variable(10.0, ["x"], []),
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

    def test_nonmtmfixxcs_fx_fixing_raises_type_crossing(self, curve, curve2) -> None:
        fxr = FXRates({"usdnok": 10}, settlement=dt(2022, 1, 1))
        fxf = FXForwards(fxr, {"usdusd": curve, "nokusd": curve2, "noknok": curve2})
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
            fx_fixings=Dual2(2.0, ["c"], [], []),
            leg2_float_spread=10.0,
        )
        with pytest.raises(TypeError, match=r"Dual2 operation with incompatible type \(Dual\)."):
            xcs.npv([curve2, curve2, curve, curve], fx=fxf)

    def test_nonmtmfixxcs_raises(self, curve, curve2) -> None:
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
    #         result.loc[[("leg1", 0), ("leg2", 8)],
    #                    ["Type", "Period", "Ccy", "Notional", "FX Rate"]],
    #         expected,
    #     )

    @pytest.mark.parametrize("fix", ["fxr", "fxf", "float", "dual", "variable"])
    def test_nonmtmfixxcs_fx_fixing(self, curve, curve2, fix) -> None:
        fxr = FXRates({"usdnok": 10}, settlement=dt(2022, 1, 1))
        fxf = FXForwards(fxr, {"usdusd": curve, "nokusd": curve2, "noknok": curve2})
        mapping = {
            "fxr": fxr,
            "fxf": fxf,
            "float": 10.0,
            "dual": Dual(10.0, ["x"], []),
            "variable": Variable(10.0, ["x"], []),
        }
        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=True,
            leg2_fixed=True,
            leg2_mtm=False,
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            fx_fixings=mapping[fix],
            leg2_fixed_rate=2.0,
        )
        assert abs(xcs.npv([curve2, curve2, curve, curve], fx=fxr)) < 1e-7

    def test_nonmtmfixxcs_fx_fixing_type_crossing_raises(self, curve, curve2) -> None:
        fxr = FXRates({"usdnok": 10}, settlement=dt(2022, 1, 1))
        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            fixed=True,
            leg2_fixed=True,
            leg2_mtm=False,
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            fx_fixings=Dual2(10.0, ["s"], [], []),
            leg2_fixed_rate=2.0,
        )
        with pytest.raises(TypeError, match=r"Dual2 operation with incompatible type \(Dual\)."):
            xcs.npv([curve2, curve2, curve, curve], fx=fxr)

    def test_nonmtmfixfixxcs_raises(self, curve, curve2) -> None:
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


@pytest.fixture
def isda_credit_curves_40rr_20quote():
    # https://www.cdsmodel.com/rfr-test-grids.html?
    # USD 22 June 2022

    # from rateslib.scheduling import get_calendar
    # trade = dt(2022, 6, 22)
    # spot = get_calendar("nyc").add_bus_days(trade, 2, False)
    # tenors = ["1m", "2m", "3m", "6m", "1y", "2y", "3y", "4y", "5y", "6y", "7y", "8y", "9y"]
    # tenors += ["10y", "12y", "15y", "20y", "25y", "30y"]
    # curve = Curve(
    #     nodes={
    #         trade: 1.0,
    #         **{add_tenor(spot, _, "f", "nyc"): 1.0 for _ in tenors},
    #     },
    #     interpolation="log_linear",
    # )
    # solver = Solver(
    #     curves=[curve],
    #     instruments=[IRS(spot, _, spec="usd_irs", curves=curve) for _ in tenors],
    #     s=[1.5088, 1.8228, 1.9729, 2.5640, 3.1620, 3.3169, 3.2441, 3.1771, 3.1371, 3.1131, 3.0951,
    #        3.0841, 3.0811, 3.0871, 3.1061, 3.1201, 3.0601, 2.9381, 2.8221]
    # )
    #
    # credit_curve = Curve(
    #     nodes={trade: 1.0, dt(2055, 1, 1): 1.0}, credit_recovery_rate=0.4
    # )
    # solver2 = Solver(
    #     curves=[credit_curve],
    #     pre_solvers=[solver],
    #     instruments=[
    #         CDS(dt(2022, 6, 20), dt(2023, 6, 20), spec="us_ig_cds", curves=[credit_curve, curve])], #noqa: E501
    #     s=[0.20]
    # )

    curve = Curve(
        {
            dt(2022, 6, 22, 0, 0): 1.0,
            dt(2022, 7, 25, 0, 0): 0.9986187857823194,
            dt(2022, 8, 24, 0, 0): 0.9968373705612348,
            dt(2022, 9, 26, 0, 0): 0.994791605422867,
            dt(2022, 12, 27, 0, 0): 0.9868431949407511,
            dt(2023, 6, 26, 0, 0): 0.9686906539113461,
            dt(2024, 6, 24, 0, 0): 0.9357773336285784,
            dt(2025, 6, 24, 0, 0): 0.9073411683282268,
            dt(2026, 6, 24, 0, 0): 0.8808780124060293,
            dt(2027, 6, 24, 0, 0): 0.8551765951547667,
            dt(2028, 6, 26, 0, 0): 0.8298749243478529,
            dt(2029, 6, 25, 0, 0): 0.8056454824131845,
            dt(2030, 6, 24, 0, 0): 0.7819517736960135,
            dt(2031, 6, 24, 0, 0): 0.7584699996495646,
            dt(2032, 6, 24, 0, 0): 0.7349334728363958,
            dt(2034, 6, 26, 0, 0): 0.6890701260967745,
            dt(2037, 6, 24, 0, 0): 0.62634116393611,
            dt(2042, 6, 24, 0, 0): 0.5441094046550682,
            dt(2047, 6, 24, 0, 0): 0.4864281755586489,
            dt(2052, 6, 24, 0, 0): 0.4409891618081753,
        }
    )

    return (None, curve)


class TestCDS:
    def okane_curve(self):
        today = dt(2019, 8, 12)
        spot = dt(2019, 8, 14)
        tenors = [
            "1b",
            "1m",
            "2m",
            "3m",
            "6m",
            "12M",
            "2y",
            "3y",
            "4y",
            "5y",
            "6y",
            "7y",
            "8y",
            "9y",
            "10y",
        ]
        ibor = Curve(
            nodes={today: 1.0, **{add_tenor(spot, _, "mf", "nyc"): 1.0 for _ in tenors}},
            convention="act360",
            calendar="nyc",
            id="ibor",
        )
        rates = [
            2.2,
            2.2009,
            2.2138,
            2.1810,
            2.0503,
            1.9930,
            1.591,
            1.499,
            1.4725,
            1.4664,
            1.48,
            1.4995,
            1.5118,
            1.5610,
            1.6430,
        ]
        ib_sv = Solver(
            curves=[ibor],
            instruments=[
                IRS(
                    spot,
                    _,
                    leg2_fixing_method="ibor",
                    leg2_method_param=2,
                    calendar="nyc",
                    payment_lag=0,
                    convention="30e360",
                    leg2_convention="act360",
                    frequency="s",
                    curves=ibor,
                )
                for _ in tenors
            ],
            s=rates,
        )
        cds_tenor = ["6m", "12m", "2y", "3y", "4y", "5y", "7y", "10y"]
        credit_curve = Curve(
            nodes={today: 1.0, **{add_tenor(today, _, "mf", "nyc"): 1.0 for _ in cds_tenor}},
            convention="act365f",
            calendar="all",
            id="credit",
            credit_discretization=5,
        )
        cc_sv = Solver(
            curves=[credit_curve],
            pre_solvers=[ib_sv],
            instruments=[
                CDS(
                    today,
                    add_tenor(dt(2019, 9, 20), _, "mf", "nyc"),
                    front_stub=dt(2019, 9, 20),
                    frequency="q",
                    convention="act360",
                    payment_lag=0,
                    curves=["credit", "ibor"],
                    fixed_rate=4.00,
                    premium_accrued=True,
                    calendar="nyc",
                )
                for _ in cds_tenor
            ],
            s=[4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00],
        )
        return credit_curve, ibor, cc_sv

    def test_okane_values(self):
        # These values are validated against finance Py. Not identical but within tolerance.
        cds = CDS(
            dt(2019, 8, 12),
            dt(2029, 6, 20),
            front_stub=dt(2019, 9, 20),
            frequency="q",
            fixed_rate=1.50,
            curves=["credit", "ibor"],
            calendar="nyc",
        )
        c1, c2, solver = self.okane_curve()
        result1 = cds.rate(solver=solver)
        assert abs(result1 - 3.9999960) < 5e-5

        result2 = cds.npv(solver=solver)
        assert abs(result2 - 170739.5956) < 175

        result3 = cds.leg1.npv(c1, c2)
        assert abs(result3 + 104508.9265 - 2125) < 50

        result4 = cds.leg2.npv(c1, c2)
        assert abs(result4 - 273023.5221) < 110

    def test_unpriced_npv(self, curve, curve2) -> None:
        cds = CDS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="eur",
        )

        npv = cds.npv([curve2, curve], NoInput(0))
        assert abs(npv) < 1e-9

    def test_rate(self, curve, curve2) -> None:
        hazard_curve = curve
        disc_curve = curve2

        cds = CDS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="eur",
        )

        rate = cds.rate([hazard_curve, disc_curve])
        expected = 2.4164004881061285
        assert abs(rate - expected) < 1e-7

    def test_npv(self, curve, curve2) -> None:
        hazard_curve = curve
        disc_curve = curve2

        cds = CDS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="eur",
            fixed_rate=1.00,
        )

        npv = cds.npv([hazard_curve, disc_curve])
        expected = 9075.835204292109  # uses cds_discretization = 23 as default
        assert abs(npv - expected) < 1e-7

    def test_analytic_delta(self, curve, curve2) -> None:
        hazard_curve = curve
        disc_curve = curve2

        cds = CDS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="eur",
        )

        result = cds.analytic_delta(hazard_curve, disc_curve, leg=1)
        expected = 64.07675851924779
        assert abs(result - expected) < 1e-7

        result = cds.analytic_delta(hazard_curve, disc_curve, leg=2)
        expected = 0.0
        assert abs(result - expected) < 1e-7

    def test_cds_cashflows(self, curve, curve2) -> None:
        hazard_curve = curve
        disc_curve = curve2

        cds = CDS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="eur",
        )
        result = cds.cashflows(curves=[hazard_curve, disc_curve])
        assert isinstance(result, DataFrame)
        assert result.index.nlevels == 2

    def test_solver(self, curve2):
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="disc")
        c2 = Curve({dt(2022, 1, 1): 1.0, dt(2022, 7, 1): 0.99, dt(2023, 1, 1): 0.98}, id="haz")

        solver = Solver(
            curves=[c2],
            instruments=[
                CDS(dt(2022, 1, 1), "6m", frequency="Q", curves=["haz", c1]),
                CDS(dt(2022, 1, 1), "12m", frequency="Q", curves=["haz", c1]),
            ],
            s=[0.30, 0.40],
            instrument_labels=["6m", "12m"],
        )
        inst = CDS(dt(2022, 7, 1), "3M", "Q", curves=["haz", c1], notional=1e6)
        result = inst.delta(solver=solver)
        assert abs(result.sum().iloc[0] - 25.294894375736) < 1e-6

    def test_okane_paper(self):
        # Figure 12 of Turnbull and O'Kane 2003 Valuation of CDS
        usd_libor = Curve(
            nodes={
                dt(2003, 6, 19): 1.0,
                dt(2003, 12, 23): 1.0,
                dt(2004, 6, 23): 1.0,
                dt(2005, 6, 23): 1.0,
                dt(2006, 6, 23): 1.0,
                dt(2007, 6, 23): 1.0,
                dt(2008, 6, 23): 1.0,
            },
            convention="act360",
            calendar="nyc",
            id="libor",
        )
        args = dict(spec="eur_irs6", frequency="s", calendar="nyc", curves="libor", currency="usd")
        solver = Solver(
            curves=[usd_libor],
            instruments=[
                IRS(dt(2003, 6, 23), "6m", **args),
                IRS(dt(2003, 6, 23), "1y", **args),
                IRS(dt(2003, 6, 23), "2y", **args),
                IRS(dt(2003, 6, 23), "3y", **args),
                IRS(dt(2003, 6, 23), "4y", **args),
                IRS(dt(2003, 6, 23), "5y", **args),
            ],
            s=[1.35, 1.43, 1.90, 2.47, 2.936, 3.311],
        )
        haz_curve = Curve(
            nodes={
                dt(2003, 6, 19): 1.0,
                dt(2004, 6, 20): 1.0,
                dt(2005, 6, 20): 1.0,
                dt(2006, 6, 20): 1.0,
                dt(2007, 6, 20): 1.0,
                dt(2008, 6, 20): 1.0,
            },
            convention="act365f",
            calendar="all",
            id="hazard",
        )
        args = dict(
            calendar="nyc", frequency="q", roll=20, curves=["hazard", "libor"], convention="act360"
        )
        solver = Solver(
            curves=[haz_curve],
            pre_solvers=[solver],
            instruments=[
                CDS(dt(2003, 6, 20), "1y", **args),
                CDS(dt(2003, 6, 20), "2y", **args),
                CDS(dt(2003, 6, 20), "3y", **args),
                CDS(dt(2003, 6, 20), "4y", **args),
                CDS(dt(2003, 6, 20), "5y", **args),
            ],
            s=[1.10, 1.20, 1.30, 1.40, 1.50],
        )
        cds = CDS(dt(2003, 6, 20), dt(2007, 9, 20), fixed_rate=2.00, notional=10e6, **args)
        result = cds.rate(solver=solver)
        assert abs(result - 1.427) < 0.0030

        _table = cds.cashflows(solver=solver)
        leg1_npv = cds.leg1.npv(haz_curve, usd_libor)
        leg2_npv = cds.leg2.npv(haz_curve, usd_libor)
        assert abs(leg1_npv + 781388) < 250
        assert abs(leg2_npv - 557872) < 900

        a_delta = cds.analytic_delta(haz_curve, usd_libor)
        assert abs(a_delta - 3899) < 10

        npv = cds.npv(solver=solver)
        assert abs(npv + 223516) < 670

    def test_accrued(self):
        cds = CDS(
            dt(2022, 1, 1), "6M", "Q", payment_lag=0, currency="eur", notional=1e9, fixed_rate=2.0
        )
        result = cds.accrued(dt(2022, 2, 1))
        assert abs(result + 0.25 * 1e9 * 0.02 * 31 / 90) < 1e-6

    @pytest.mark.parametrize(
        ("cash", "tenor", "quote"),
        [
            (-79690.03, "1y", 0.20),
            (-156453.96, "2y", 0.20),
            (-230320.76, "3y", 0.20),
            (-370875.32, "5y", 0.20),
            (-502612.64, "7y", 0.20),
            (-684299.75, "10y", 0.20),
            (116199.85, "1y", 2.20),
            (225715.34, "2y", 2.20),
            (327602.22, "3y", 2.20),
            (512001.20, "5y", 2.20),
            (673570.58, "7y", 2.20),
            (878545.53, "10y", 2.20),
        ],
    )
    def test_standard_model_test_grid(self, cash, tenor, quote, isda_credit_curves_40rr_20quote):
        # https://www.cdsmodel.com/rfr-test-grids.html?
        # USD 22 June 2022
        credit_curve, curve = isda_credit_curves_40rr_20quote

        credit_curve = Curve({dt(2022, 6, 22): 1.0, dt(2052, 6, 30): 1.0}, credit_recovery_rate=0.4)
        Solver(
            curves=[credit_curve],
            instruments=[
                CDS(dt(2022, 6, 20), tenor, spec="us_ig_cds", curves=[credit_curve, curve])
            ],
            s=[quote],
        )

        cds = CDS(
            dt(2022, 6, 20), tenor, spec="us_ig_cds", curves=[credit_curve, curve], notional=10e6
        )
        result = cds.npv()
        assert abs(result - cash) < 875
        print(abs(result - cash))


class TestXCS:
    def test_mtmxcs_npv(self, curve, curve2) -> None:
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

    def test_mtmxcs_cashflows(self, curve, curve2) -> None:
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

    @pytest.mark.skip(
        reason="After merging all XCS to one class inputting `fx_fixings` as list was changed.",
    )
    def test_mtmxcs_fx_fixings_raises(self) -> None:
        with pytest.raises(ValueError, match="`fx_fixings` for MTM XCS should"):
            _ = XCS(
                dt(2022, 2, 1),
                "8M",
                "M",
                fx_fixings=NoInput(0),
                currency="usd",
                leg2_currency="eur",
            )

        with pytest.raises(ValueError, match="`fx_fixings` for MTM XCS should"):
            _ = XCS(
                dt(2022, 2, 1),
                "8M",
                "M",
                fx_fixings=NoInput(0),
                fixed=True,
                leg2_fixed=False,
                leg2_mtm=True,
                currency="usd",
                leg2_currency="eur",
            )

        with pytest.raises(ValueError, match="`fx_fixings` for MTM XCS should"):
            _ = XCS(
                dt(2022, 2, 1),
                "8M",
                "M",
                fx_fixings=NoInput(0),
                fixed=True,
                leg2_fixed=True,
                leg2_mtm=True,
                currency="usd",
                leg2_currency="eur",
            )

        with pytest.raises(ValueError, match="`fx_fixings` for MTM XCS should"):
            _ = XCS(
                dt(2022, 2, 1),
                "8M",
                "M",
                fx_fixings=NoInput(0),
                fixed=False,
                leg2_fixed=True,
                leg2_mtm=True,
                currency="usd",
                leg2_currency="eur",
            )

    @pytest.mark.parametrize(
        ("float_spd", "compound", "expected"),
        [
            (10, "none_simple", 9.97839804),
            (100, "none_simple", 99.78398037),
            (100, "isda_compounding", 99.418428),
            (100, "isda_flat_compounding", 99.621117),
        ],
    )
    def test_mtmxcs_rate(self, float_spd, compound, expected, curve, curve2) -> None:
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

    def test_fx_fixings_2_tuple(self) -> None:
        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            "M",
            payment_lag=0,
            currency="nok",
            leg2_currency="usd",
            payment_lag_exchange=0,
            notional=10e6,
            fx_fixings=(1.25, Series([1.5, 1.75], index=[dt(2022, 3, 1), dt(2022, 4, 1)])),
        )
        assert xcs.leg2.fx_fixings == [1.25, 1.5, 1.75]

    def test_initialisation_nonmtm_xcs_leg_notional_unused(self) -> None:
        xcs = XCS(
            effective=dt(2000, 1, 1),
            termination="1y",
            frequency="q",
            notional=135e6,
            fx_fixings=0.7407407407407407,
            leg2_notional=20e6,
            currency="cad",
            leg2_currency="usd",
            leg2_mtm=False,
        )
        assert abs(xcs.leg2.notional + 100e6) < 1e-8  # not 20e6

    @pytest.mark.parametrize("fixed1", [True, False])
    @pytest.mark.parametrize("fixed2", [True, False])
    @pytest.mark.parametrize("mtm", [True, False])
    def test_fixings_table(self, curve, curve2, fixed1, fixed2, mtm):
        curve._id = "c1"
        curve2._id = "c2"
        fxf = FXForwards(
            FXRates({"eurusd": 1.1}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "eurusd": curve2, "eureur": curve2},
        )

        xcs = XCS(
            dt(2022, 2, 1),
            "8M",
            frequency="M",
            payment_lag=0,
            currency="eur",
            leg2_currency="usd",
            payment_lag_exchange=0,
            fixed=fixed1,
            leg2_fixed=fixed2,
            leg2_mtm=mtm,
            fixing_method="ibor",
            leg2_fixing_method="ibor",
        )
        result = xcs.fixings_table(curves=[curve, curve, curve2, curve2], fx=fxf)
        assert isinstance(result, DataFrame)

    def test_initialisation_bug(self):
        XCS(
            dt(2000, 1, 7),
            "9m",
            spec="eurusd_xcs",
            leg2_fixed=True,
            leg2_mtm=False,
            fixing_method="ibor",
            method_param=2,
            leg2_fixed_rate=2.4,
        )

        XCS(dt(2000, 1, 7), "9m", spec="eurusd_xcs", fixed=True, fixed_rate=3.0)

    def test_fixing_doc(self):
        # tests a series as sting can be provided to XCS in tuple
        curve = Curve({dt(2023, 1, 15): 1.0, dt(2028, 1, 1): 0.96})
        name = str(hash(os.urandom(8)))
        defaults.fixings.add(
            name,
            Series(
                index=[dt(2023, 1, 17), dt(2023, 4, 17), dt(2023, 7, 17)],
                data=[1.19, 1.21, 1.24],
            ),
        )
        xcs = XCS(
            effective=dt(2023, 1, 15),
            termination="9M",
            spec="gbpusd_xcs",
            fx_fixings=(1.20, name),
        )
        result = xcs.cashflows(
            curves=curve, fx=1.25
        )  # arguments here used as a placeholder to display values.
        assert isinstance(result, DataFrame)


class TestFixedFloatXCS:
    def test_mtmfixxcs_rate(self, curve, curve2) -> None:
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

    def test_mtmfixxcs_rate_reversed(self, curve, curve2) -> None:
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
    def test_mtmfixfixxcs_rate(self, curve, curve2) -> None:
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
    def test_fxswap_rate(self, curve, curve2) -> None:
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
        assert abs(result - expected) < 1e-10
        assert np.isclose(result.dual, expected.dual)

    def test_fxswap_pair_arg(self, curve, curve2) -> None:
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )
        fxs = FXSwap(
            dt(2022, 2, 1),
            "8M",
            pair="usdnok",
            payment_lag=0,
            notional=1e6,
        )
        expected = fxf.swap("usdnok", [dt(2022, 2, 1), dt(2022, 10, 1)])
        result = fxs.rate([NoInput(0), curve, NoInput(0), curve2], NoInput(0), fxf)
        assert abs(result - expected) < 1e-10
        assert np.isclose(result.dual, expected.dual)

    def test_currency_arg_pair_overlap(self) -> None:
        fxs = FXSwap(
            dt(2022, 2, 1),
            "8M",
            pair="usdnok",
            currency="jpy",
        )
        assert fxs.leg1.currency == "usd"

    def test_fxswap_npv(self, curve, curve2) -> None:
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

    @pytest.mark.parametrize(
        ("points", "split_notional"),
        [(100, 1e6), (NoInput(0), 1e6), (100, NoInput(0))],
    )
    def test_fxswap_points_raises(self, points, split_notional) -> None:
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

    def test_fxswap_points_warns(self) -> None:
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

    @pytest.mark.parametrize(
        ("fx_fixings", "points", "split_notional", "expected"),
        [
            (NoInput(0), NoInput(0), NoInput(0), Dual(0, ["fx_usdnok"], [-1712.833785])),
            (11.0, 1800.0, NoInput(0), Dual(-3734.617680, ["fx_usdnok"], [3027.88203904])),
            (
                11.0,
                1754.5623360395632,
                NoInput(0),
                Dual(-4166.37288388, ["fx_usdnok"], [3071.05755945]),
            ),
            (
                10.032766762996951,
                1754.5623360395632,
                NoInput(0),
                Dual(0, ["fx_usdnok"], [2654.42027107]),
            ),
            (
                10.032766762996951,
                1754.5623360395632,
                1027365.1574336714,
                Dual(0, ["fx_usdnok"], [0.0]),
            ),
        ],
    )
    def test_fxswap_parameter_combinations_off_mids_given(
        self,
        curve,
        curve2,
        fx_fixings,
        points,
        split_notional,
        expected,
    ) -> None:
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

        assert abs(result - expected) < 1e-6
        assert np.isclose(result.dual, expected.dual)

    def test_rate_with_fixed_parameters(self, curve, curve2) -> None:
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
    #     npv_nok =
    #         fxs.npv([NoInput(0), fxf.curve("usd", "nok"), NoInput(0), curve2], NoInput(0), fxf)
    #     npv_usd =
    #         fxs.npv([NoInput(0), curve, NoInput(0), fxf.curve("nok", "usd")], NoInput(0), fxf)
    #     assert abs(npv_nok-npv_usd) < 1e-7  # npvs are equivalent becasue xcs basis =0

    def test_transition_from_dual_to_dual2(self, curve, curve2) -> None:
        # Test added for BUG, see PR: XXX
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )
        fxf._set_ad_order(1)
        fxs = FXSwap(
            dt(2022, 2, 1),
            "8M",
            currency="usd",
            leg2_currency="nok",
            payment_lag=0,
            notional=1e6,
        )
        fxs.npv(curves=[None, fxf.curve("usd", "usd"), None, fxf.curve("nok", "usd")], fx=fxf)
        fxf._set_ad_order(2)
        fxs.npv(curves=[None, fxf.curve("usd", "usd"), None, fxf.curve("nok", "usd")], fx=fxf)

    def test_transition_from_dual_to_dual2_rate(self, curve, curve2) -> None:
        # Test added for BUG, see PR: XXX
        fxf = FXForwards(
            FXRates({"usdnok": 10}, settlement=dt(2022, 1, 3)),
            {"usdusd": curve, "nokusd": curve2, "noknok": curve2},
        )
        fxf._set_ad_order(1)
        fxs = FXSwap(
            dt(2022, 2, 1),
            "8M",
            currency="usd",
            leg2_currency="nok",
            payment_lag=0,
            notional=1e6,
        )
        fxs.rate(curves=[None, fxf.curve("usd", "usd"), None, fxf.curve("nok", "usd")], fx=fxf)
        fxf._set_ad_order(2)
        fxs.rate(curves=[None, fxf.curve("usd", "usd"), None, fxf.curve("nok", "usd")], fx=fxf)

    def test_split_notional_raises(self):
        # this is an unpriced FXswap with split notional
        fxs = FXSwap(effective=dt(2022, 2, 1), termination="3m", pair="eurusd")
        with pytest.raises(ValueError, match="A `curve` is required to determine a `split_notion"):
            fxs.rate()


class TestSTIRFuture:
    def test_stir_rate(self, curve, curve2) -> None:
        stir = STIRFuture(
            effective=dt(2022, 3, 16),
            termination=dt(2022, 6, 15),
            spec="usd_stir",
        )
        expected = 95.96254344884888
        result = stir.rate(curve, metric="price")
        assert abs(100 - result - stir.rate(curve)) < 1e-8
        assert abs(result - expected) < 1e-8

    def test_stir_no_gamma(self, curve) -> None:
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

    def test_stir_npv(self) -> None:
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

    def test_stir_npv_currency_bug(self) -> None:
        # GH653: instantiation without a currency failed to NPV when an fx object provided.
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99})
        c2 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98})
        c3 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.97})
        fxf = FXForwards(
            FXRates({"eurusd": 1.1}, dt(2022, 1, 1)), {"eureur": c1, "eurusd": c2, "usdusd": c3}
        )
        stir = STIRFuture(
            effective=dt(2022, 3, 16),
            termination=dt(2022, 6, 15),
            frequency="Q",
            bp_value=25.0,
            contracts=-1,
        )
        result = stir.npv(curves=[c1, c1, c2, c3], fx=fxf)
        assert abs(result) < 1e-7

    def test_stir_npv_fx(self) -> None:
        c1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, id="usdusd")
        # irs = IRS(dt(2022, 3, 16), dt(2022, 6, 15), "Q", curves="usdusd")
        stir = STIRFuture(
            effective=dt(2022, 3, 16),
            termination=dt(2022, 6, 15),
            spec="usd_stir",
            curves="usdusd",
            price=99.50,
        )

        fxr = FXRates({"usdeur": 0.85})
        result = stir.npv(curves=c1, fx=fxr, base="eur")
        expected = ((99.5 - (100 - 0.99250894761)) * 2500 * -1.0) * 0.85

        assert abs(result - expected) < 1e-7

    def test_stir_raises(self) -> None:
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

    def test_analytic_delta(self) -> None:
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
        assert abs(result - expected) < 1e-10

    def test_analytic_delta_fx(self) -> None:
        stir = STIRFuture(
            effective=dt(2022, 3, 16),
            termination=dt(2022, 6, 15),
            spec="usd_stir",
            curves="usdusd",
            price=99.50,
            contracts=100,
        )
        expected = -2500.0 * 0.85
        fxr = FXRates({"usdeur": 0.85})
        result = stir.analytic_delta(fx=fxr, base="eur")
        assert abs(result - expected) < 1e-10

    def test_fixings_table(self, curve):
        stir = STIRFuture(
            effective=dt(2022, 3, 16),
            termination="3m",
            spec="eur_stir3",
            contracts=100,
            curves=curve,
        )
        result = stir.fixings_table()
        assert isinstance(result, DataFrame)
        assert result[f"{curve.id}", "risk"][dt(2022, 3, 14)] == -2500.0


class TestPricingMechanism:
    def test_value(self, curve) -> None:
        ob = Value(dt(2022, 1, 28), curves=curve)
        ob.rate()

    def test_irs(self, curve) -> None:
        ob = IRS(dt(2022, 1, 28), "6m", "Q", curves=curve)
        ob.rate()
        ob.npv()
        ob.cashflows()
        ob.spread()

    def test_iirs(self, curve) -> None:
        i_curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_base=100.0,
            interpolation="linear_index",
            index_lag=3,
        )
        ob = IIRS(dt(2022, 1, 28), "6m", "Q", curves=[i_curve, curve, curve, curve])
        ob.rate()
        ob.npv()
        ob.cashflows()
        ob.spread()

    def test_sbs(self, curve) -> None:
        ob = SBS(dt(2022, 1, 28), "6m", "Q", curves=curve)
        ob.rate()
        ob.npv()
        ob.cashflows()
        ob.spread()

    def test_fra(self, curve) -> None:
        ob = FRA(dt(2022, 1, 28), "6m", "S", curves=curve)
        ob.rate()
        ob.npv()
        ob.cashflows()

    @pytest.mark.parametrize(
        ("klass", "kwargs"),
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
    def test_allxcs(self, klass, kwargs, curve, curve2) -> None:
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

    def test_zcs(self, curve) -> None:
        ob = ZCS(dt(2022, 1, 28), "6m", "S", curves=curve)
        ob.rate()
        ob.npv()
        ob.cashflows()

    def test_zcis(self, curve) -> None:
        i_curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_base=100.0,
            interpolation="linear_index",
            index_lag=3,
        )
        ob = ZCIS(dt(2022, 1, 28), "6m", "S", curves=[curve, curve, i_curve, curve])
        ob.rate()
        ob.npv()
        ob.cashflows()

    # TODO FXEXchange and FXSwap


class TestPortfolio:
    def test_portfolio_npv(self, curve) -> None:
        irs1 = IRS(dt(2022, 1, 1), "6m", "Q", fixed_rate=1.0, curves=curve)
        irs2 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=2.0, curves=curve)
        pf = Portfolio([irs1, irs2])
        assert pf.npv(base="usd") == irs1.npv() + irs2.npv()

        pf = Portfolio([irs1] * 5)
        assert pf.npv(base="usd") == irs1.npv() * 5

    def test_portoflio_npv_pool(self, curve) -> None:
        irs1 = IRS(dt(2022, 1, 1), "6m", "Q", fixed_rate=1.0, curves=curve)
        pf = Portfolio([irs1] * 5)
        with default_context("pool", 2):  # also test parallel processing
            result = pf.npv(base="usd")
            assert result == irs1.npv() * 5

    def test_portfolio_npv_local(self, curve) -> None:
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

    def test_portfolio_local_parallel(self, curve) -> None:
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

    def test_portfolio_mixed_currencies(self) -> None:
        ll_curve = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 5, 1): 1.0, dt(2022, 9, 3): 1.0},
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
            id="sofr",
        )

        ll_curve = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 4, 1): 1.0, dt(2022, 10, 1): 1.0},
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
            id="estr",
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
        assert "eur" in result
        assert "usd" in result

        # the following should execute without warnings
        pf.delta(solver=combined_solver)
        pf.gamma(solver=combined_solver)

    def test_repr(self, curve) -> None:
        irs1 = IRS(dt(2022, 1, 1), "6m", "Q", fixed_rate=1.0, curves=curve)
        irs2 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=2.0, curves=curve)
        pf = Portfolio([irs1, irs2])
        expected = f"<rl.Portfolio at {hex(id(pf))}>"
        assert pf.__repr__() == expected

    def test_fixings_table(self, curve, curve2):
        curve._id = "c1"
        curve2._id = "c2"
        irs1 = IRS(dt(2022, 1, 17), "6m", spec="eur_irs3", curves=curve, notional=3e6)
        irs2 = IRS(dt(2022, 1, 23), "6m", spec="eur_irs6", curves=curve2, notional=1e6)
        irs3 = IRS(dt(2022, 1, 17), "6m", spec="eur_irs3", curves=curve, notional=-2e6)
        pf = Portfolio([irs1, irs2, irs3])
        result = pf.fixings_table()

        # irs1 and irs3 are summed over curve c1 notional
        assert abs(result["c1", "notional"][dt(2022, 1, 15)] - 1021994.16) < 1e-2
        # irs1 and irs3 are summed over curve c1 risk
        assert abs(result["c1", "risk"][dt(2022, 1, 15)] - 25.249) < 1e-2
        # c1 has no exposure to 22nd Jan
        assert isna(result["c1", "risk"][dt(2022, 1, 22)])
        # c1 dcf is not summed
        assert abs(result["c1", "dcf"][dt(2022, 1, 15)] - 0.25) < 1e-3

        # irs2 is included
        assert abs(result["c2", "notional"][dt(2022, 1, 22)] - 1005297.17) < 1e-2
        # irs1 and irs3 are summed over curve c1 risk
        assert abs(result["c2", "risk"][dt(2022, 1, 22)] - 48.773) < 1e-3
        # c2 has no exposure to 15 Jan
        assert isna(result["c2", "risk"][dt(2022, 1, 15)])
        # c2 has DCF
        assert abs(result["c2", "dcf"][dt(2022, 1, 22)] - 0.50277) < 1e-3

    def test_fixings_table_null_inst(self, curve):
        irs = IRS(dt(2022, 1, 15), "6m", spec="eur_irs3", curves=curve)
        frb = FixedRateBond(dt(2022, 1, 1), "5y", "A", fixed_rate=2.0, curves=curve)
        pf = Portfolio([irs, frb])
        assert isinstance(pf.fixings_table(), DataFrame)


class TestFly:
    @pytest.mark.parametrize("mechanism", [False, True])
    def test_fly_npv(self, curve, mechanism) -> None:
        mechanism = curve if mechanism else NoInput(0)
        inverse = curve if mechanism is NoInput(0) else NoInput(0)
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=mechanism)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=mechanism)
        irs3 = IRS(dt(2022, 1, 1), "5m", "Q", fixed_rate=1.0, curves=mechanism)
        fly = Fly(irs1, irs2, irs3)
        assert fly.npv(inverse) == irs1.npv(inverse) + irs2.npv(inverse) + irs3.npv(inverse)

    @pytest.mark.parametrize("mechanism", [False, True])
    def test_fly_rate(self, curve, mechanism) -> None:
        mechanism = curve if mechanism else NoInput(0)
        inv = curve if mechanism is NoInput(0) else NoInput(0)
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=mechanism)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=mechanism)
        irs3 = IRS(dt(2022, 1, 1), "5m", "Q", fixed_rate=1.0, curves=mechanism)
        fly = Fly(irs1, irs2, irs3)
        assert fly.rate(inv) == (-irs1.rate(inv) + 2 * irs2.rate(inv) - irs3.rate(inv)) * 100.0

    def test_fly_cashflows_executes(self, curve) -> None:
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=curve)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=curve)
        irs3 = IRS(dt(2022, 1, 1), "5m", "Q", fixed_rate=1.0, curves=curve)
        fly = Fly(irs1, irs2, irs3)
        fly.cashflows()

    def test_local_npv(self, curve) -> None:
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

    def test_delta(self, simple_solver) -> None:
        irs1 = IRS(dt(2022, 1, 1), "6m", "A", fixed_rate=1.0, notional=-3e6, curves="curve")
        irs2 = IRS(dt(2022, 1, 1), "1Y", "A", fixed_rate=2.0, notional=3e6, curves="curve")
        irs3 = IRS(dt(2022, 1, 1), "18m", "A", fixed_rate=1.0, notional=-1e6, curves="curve")
        fly = Fly(irs1, irs2, irs3)
        result = fly.delta(solver=simple_solver).to_numpy()
        expected = np.array([[102.08919479], [-96.14488074]])
        assert np.all(np.isclose(result, expected))

    def test_gamma(self, simple_solver) -> None:
        irs1 = IRS(dt(2022, 1, 1), "6m", "A", fixed_rate=1.0, notional=-3e6, curves="curve")
        irs2 = IRS(dt(2022, 1, 1), "1Y", "A", fixed_rate=2.0, notional=3e6, curves="curve")
        irs3 = IRS(dt(2022, 1, 1), "18m", "A", fixed_rate=1.0, notional=-1e6, curves="curve")
        fly = Fly(irs1, irs2, irs3)
        result = fly.gamma(solver=simple_solver).to_numpy()
        expected = np.array([[-0.02944899, 0.009254014565], [0.009254014565, 0.0094239781314]])
        assert np.all(np.isclose(result, expected))

    def test_repr(self):
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0)
        spd = Spread(irs1, irs2)
        expected = f"<rl.Spread at {hex(id(spd))}>"
        assert expected == spd.__repr__()

    def test_fixings_table(self, curve, curve2):
        curve._id = "c1"
        curve2._id = "c2"
        irs1 = IRS(dt(2022, 1, 17), "6m", spec="eur_irs3", curves=curve, notional=3e6)
        irs2 = IRS(dt(2022, 1, 23), "6m", spec="eur_irs6", curves=curve2, notional=1e6)
        irs3 = IRS(dt(2022, 1, 17), "6m", spec="eur_irs3", curves=curve, notional=-2e6)
        fly = Fly(irs1, irs2, irs3)
        result = fly.fixings_table()

        # irs1 and irs3 are summed over curve c1 notional
        assert abs(result["c1", "notional"][dt(2022, 1, 15)] - 1021994.16) < 1e-2
        # irs1 and irs3 are summed over curve c1 risk
        assert abs(result["c1", "risk"][dt(2022, 1, 15)] - 25.249) < 1e-2
        # c1 has no exposure to 22nd Jan
        assert isna(result["c1", "risk"][dt(2022, 1, 22)])
        # c1 dcf is not summed
        assert abs(result["c1", "dcf"][dt(2022, 1, 15)] - 0.25) < 1e-3

        # irs2 is included
        assert abs(result["c2", "notional"][dt(2022, 1, 22)] - 1005297.17) < 1e-2
        # irs1 and irs3 are summed over curve c1 risk
        assert abs(result["c2", "risk"][dt(2022, 1, 22)] - 48.773) < 1e-3
        # c2 has no exposure to 15 Jan
        assert isna(result["c2", "risk"][dt(2022, 1, 15)])
        # c2 has DCF
        assert abs(result["c2", "dcf"][dt(2022, 1, 22)] - 0.50277) < 1e-3

    def test_fixings_table_null_inst(self, curve):
        irs = IRS(dt(2022, 1, 15), "6m", spec="eur_irs3", curves=curve)
        frb = FixedRateBond(dt(2022, 1, 1), "5y", "A", fixed_rate=2.0, curves=curve)
        fly = Fly(irs, frb, irs)
        assert isinstance(fly.fixings_table(), DataFrame)


class TestSpread:
    @pytest.mark.parametrize("mechanism", [False, True])
    def test_spread_npv(self, curve, mechanism) -> None:
        mechanism = curve if mechanism else NoInput(0)
        inverse = curve if mechanism is NoInput(0) else NoInput(0)
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=mechanism)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=mechanism)
        spd = Spread(irs1, irs2)
        assert spd.npv(inverse) == irs1.npv(inverse) + irs2.npv(inverse)

    @pytest.mark.parametrize("mechanism", [False, True])
    def test_spread_rate(self, curve, mechanism) -> None:
        mechanism = curve if mechanism else NoInput(0)
        inverse = curve if mechanism is NoInput(0) else NoInput(0)
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=mechanism)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=mechanism)
        spd = Spread(irs1, irs2)
        assert spd.rate(inverse) == (-irs1.rate(inverse) + irs2.rate(inverse)) * 100.0

    def test_spread_cashflows_executes(self, curve) -> None:
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=curve)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=curve)
        spd = Spread(irs1, irs2)
        spd.cashflows()

    def test_local_npv(self, curve) -> None:
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0, curves=curve, currency="eur")
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0, curves=curve, currency="usd")
        spd = Spread(irs1, irs2)
        result = spd.npv(local=True)
        expected = {
            "eur": 7523.321141258284,
            "usd": 6711.514715925333,
        }
        assert result == expected

    def test_repr(self):
        irs1 = IRS(dt(2022, 1, 1), "3m", "Q", fixed_rate=1.0)
        irs2 = IRS(dt(2022, 1, 1), "4m", "Q", fixed_rate=2.0)
        irs3 = IRS(dt(2022, 1, 1), "5m", "Q", fixed_rate=1.0)
        fly = Fly(irs1, irs2, irs3)
        expected = f"<rl.Fly at {hex(id(fly))}>"
        assert expected == fly.__repr__()

    def test_fixings_table(self, curve, curve2):
        curve._id = "c1"
        curve2._id = "c2"
        irs1 = IRS(dt(2022, 1, 17), "6m", spec="eur_irs3", curves=curve, notional=3e6)
        irs2 = IRS(dt(2022, 1, 23), "6m", spec="eur_irs6", curves=curve2, notional=1e6)
        irs3 = IRS(dt(2022, 1, 17), "6m", spec="eur_irs3", curves=curve, notional=-2e6)
        spd = Spread(irs1, Spread(irs2, irs3))
        result = spd.fixings_table()

        # irs1 and irs3 are summed over curve c1 notional
        assert abs(result["c1", "notional"][dt(2022, 1, 15)] - 1021994.16) < 1e-2
        # irs1 and irs3 are summed over curve c1 risk
        assert abs(result["c1", "risk"][dt(2022, 1, 15)] - 25.249) < 1e-2
        # c1 has no exposure to 22nd Jan
        assert isna(result["c1", "risk"][dt(2022, 1, 22)])
        # c1 dcf is not summed
        assert abs(result["c1", "dcf"][dt(2022, 1, 15)] - 0.25) < 1e-3

        # irs2 is included
        assert abs(result["c2", "notional"][dt(2022, 1, 22)] - 1005297.17) < 1e-2
        # irs1 and irs3 are summed over curve c1 risk
        assert abs(result["c2", "risk"][dt(2022, 1, 22)] - 48.773) < 1e-3
        # c2 has no exposure to 15 Jan
        assert isna(result["c2", "risk"][dt(2022, 1, 15)])
        # c2 has DCF
        assert abs(result["c2", "dcf"][dt(2022, 1, 22)] - 0.50277) < 1e-3

    def test_fixings_table_null_inst(self, curve):
        irs = IRS(dt(2022, 1, 15), "6m", spec="eur_irs3", curves=curve)
        frb = FixedRateBond(dt(2022, 1, 1), "5y", "A", fixed_rate=2.0, curves=curve)
        spd = Spread(irs, frb)
        assert isinstance(spd.fixings_table(), DataFrame)


class TestSensitivities:
    def test_sensitivity_raises(self) -> None:
        irs = IRS(dt(2022, 1, 1), "6m", "Q")
        with pytest.raises(ValueError, match="`solver` is required"):
            irs.delta()

        with pytest.raises(ValueError, match="`solver` is required"):
            irs.gamma()


class TestSpec:
    def test_spec_overwrites(self) -> None:
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
            schedule=Schedule(
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
                payment_lag_exchange=0,
            ),
            leg2_schedule=Schedule(
                effective=dt(2022, 1, 1),
                termination=dt(2024, 2, 26),
                frequency="m",
                stub="longback",
                front_stub=NoInput(0),
                back_stub=NoInput(0),
                roll=1,
                eom=False,
                modifier="mp",
                calendar="nyc,tgt,ldn",
                payment_lag=3,
                payment_lag_exchange=0,
            ),
            notional=250.0,
            currency="tes",
            amortization=NoInput(0),
            convention="yearsmonths",
            leg2_notional=-250.0,
            leg2_currency="tes",
            leg2_convention="one",
            leg2_amortization=NoInput(0),
            fixed_rate=NoInput(0),
            leg2_fixing_method=NoInput(0),
            leg2_method_param=0,
            leg2_spread_compound_method=NoInput(0),
            leg2_fixings=NoInput(0),
            leg2_float_spread=NoInput(0),
        )
        assert irs.kwargs == expected

    def test_irs(self) -> None:
        irs = IRS(
            effective=dt(2022, 1, 1),
            termination="1Y",
            spec="usd_irs",
            convention="30e360",
            fixed_rate=2.0,
        )
        assert irs.kwargs["convention"] == "30e360"
        assert irs.kwargs["leg2_convention"] == "30e360"
        assert irs.kwargs["currency"] == "usd"
        assert irs.kwargs["fixed_rate"] == 2.0

    def test_stir(self) -> None:
        irs = STIRFuture(
            effective=dt(2022, 3, 16),
            termination=dt(2022, 6, 15),
            spec="usd_stir",
            convention="30e360",
        )
        assert irs.kwargs["convention"] == "30e360"
        assert irs.kwargs["leg2_convention"] == "30e360"
        assert irs.kwargs["currency"] == "usd"
        assert irs.kwargs["schedule"].roll == "IMM"

    def test_sbs(self) -> None:
        inst = SBS(
            effective=dt(2022, 1, 1),
            termination="1Y",
            spec="eur_sbs36",
            convention="30e360",
            frequency="A",
        )
        assert inst.kwargs["convention"] == "30e360"
        assert inst.kwargs["leg2_convention"] == "30e360"
        assert inst.kwargs["currency"] == "eur"
        assert inst.kwargs["fixing_method"] == "ibor"
        assert inst.kwargs["schedule"].frequency == "A"
        assert inst.kwargs["leg2_schedule"].frequency == "S"

    def test_zcis(self) -> None:
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
        assert inst.kwargs["leg2_schedule"].calendar == NamedCal("nyc,tgt")

    def test_zcs(self) -> None:
        inst = ZCS(
            effective=dt(2022, 1, 1),
            termination="5Y",
            spec="gbp_zcs",
            leg2_calendar="nyc,tgt",
            calendar="nyc,tgt",
            fixed_rate=3.0,
        )
        assert inst.kwargs["convention"] == "act365f"
        assert inst.kwargs["currency"] == "gbp"
        assert inst.kwargs["leg2_schedule"].calendar == NamedCal("nyc,tgt")
        assert inst.kwargs["leg2_schedule"].frequency == "A"
        assert inst.kwargs["fixed_rate"] == 3.0
        assert inst.kwargs["leg2_spread_compound_method"] == "none_simple"

    def test_iirs(self) -> None:
        inst = IIRS(
            effective=dt(2022, 1, 1),
            termination="1Y",
            spec="sek_iirs",
            leg2_calendar="nyc,tgt",
            calendar="nyc,tgt",
            fixed_rate=3.0,
        )
        assert inst.kwargs["convention"] == "actacticma"
        assert inst.kwargs["leg2_schedule"].frequency == "Q"
        assert inst.kwargs["currency"] == "sek"
        assert inst.kwargs["leg2_schedule"].calendar == NamedCal("nyc,tgt")
        assert inst.kwargs["fixed_rate"] == 3.0
        assert inst.kwargs["leg2_spread_compound_method"] == "none_simple"

    def test_fixedratebond(self) -> None:
        bond = FixedRateBond(
            effective=dt(2022, 1, 1),
            termination="1Y",
            spec="us_gb",
            calc_mode="ust_31bii",
            fixed_rate=2.0,
        )
        from rateslib.instruments.bonds.conventions import US_GB_TSY

        assert bond.calc_mode.kwargs == US_GB_TSY.kwargs
        assert bond.kwargs["convention"] == "actacticma"
        assert bond.kwargs["currency"] == "usd"
        assert bond.kwargs["fixed_rate"] == 2.0
        assert bond.kwargs["ex_div"] == 1

    def test_indexfixedratebond(self) -> None:
        bond = IndexFixedRateBond(
            effective=dt(2022, 1, 1),
            termination="1Y",
            spec="uk_gbi",
            calc_mode="ust",
            fixed_rate=2.0,
        )
        from rateslib.instruments.bonds.conventions import US_GB

        assert bond.calc_mode.kwargs == US_GB.kwargs
        assert bond.kwargs["convention"] == "actacticma"
        assert bond.kwargs["currency"] == "gbp"
        assert bond.kwargs["fixed_rate"] == 2.0
        assert bond.kwargs["ex_div"] == 7

    def test_bill(self) -> None:
        bill = Bill(
            effective=dt(2022, 1, 1),
            termination="3m",
            spec="us_gbb",
            convention="act365f",
        )
        from rateslib.instruments.bonds.conventions import US_GBB

        assert bill.calc_mode.kwargs == US_GBB.kwargs
        assert bill.kwargs["convention"] == "act365f"
        assert bill.kwargs["currency"] == "usd"
        assert bill.kwargs["fixed_rate"] == 0.0

    def test_fra(self) -> None:
        fra = FRA(
            effective=dt(2022, 1, 1),
            termination="3m",
            spec="eur_fra3",
            payment_lag=5,
            modifier="F",
            fixed_rate=2.0,
        )
        assert fra.kwargs["leg2_fixing_method"] == "ibor"
        assert fra.kwargs["convention"] == "act360"
        assert fra.kwargs["currency"] == "eur"
        assert fra.kwargs["fixed_rate"] == 2.0
        assert fra.kwargs["leg2_schedule"].payment_adjuster == Adjuster.BusDaysLagSettle(5)
        assert fra.kwargs["leg2_schedule"].modifier == Adjuster.Following()

    def test_frn(self) -> None:
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
        assert frn.kwargs["schedule"].payment_adjuster == Adjuster.BusDaysLagSettle(5)
        assert frn.kwargs["schedule"].modifier == Adjuster.ModifiedFollowing()

    def test_xcs(self) -> None:
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
        assert xcs.kwargs["schedule"].calendar == NamedCal("ldn,tgt,nyc")
        assert xcs.kwargs["schedule"].payment_adjuster == Adjuster.BusDaysLagSettle(5)
        assert xcs.kwargs["leg2_schedule"].payment_adjuster == Adjuster.BusDaysLagSettle(5)
        assert xcs.kwargs["leg2_schedule"].calendar == NamedCal("ldn,tgt,nyc")


@pytest.mark.parametrize(
    ("inst", "expected"),
    [
        (
            IRS(dt(2022, 1, 1), "9M", "Q", currency="eur", curves=["eureur", "eur_eurusd"]),
            DataFrame(
                [-0.21319, -0.00068, 0.21656],
                index=Index([dt(2022, 4, 3), dt(2022, 7, 3), dt(2022, 10, 3)], name="payment"),
                columns=MultiIndex.from_tuples(
                    [("EUR", "usd,eur")],
                    names=["local_ccy", "collateral_ccy"],
                ),
            ),
        ),
        (
            SBS(
                dt(2022, 1, 1),
                "9M",
                "Q",
                leg2_frequency="S",
                currency="eur",
                curves=["eureur", "eurusd"],
            ),
            DataFrame(
                [-0.51899, -6260.7208, 6299.28759],
                index=Index([dt(2022, 4, 3), dt(2022, 7, 3), dt(2022, 10, 3)], name="payment"),
                columns=MultiIndex.from_tuples(
                    [("EUR", "usd")],
                    names=["local_ccy", "collateral_ccy"],
                ),
            ),
        ),
        (
            FRA(dt(2022, 1, 15), "3M", "Q", currency="eur", curves=["eureur", "eureur"]),
            DataFrame(
                [0],
                index=Index([dt(2022, 1, 15)], name="payment"),
                columns=MultiIndex.from_tuples(
                    [("EUR", "eur")],
                    names=["local_ccy", "collateral_ccy"],
                ),
            ),
        ),
        (
            FXExchange(
                dt(2022, 1, 15),
                pair="eurusd",
                curves=["eureur", "eureur", "usdusd", "usdeur"],
            ),
            DataFrame(
                [[1000000.0, -1101072.93429]],
                index=Index([dt(2022, 1, 15)], name="payment"),
                columns=MultiIndex.from_tuples(
                    [("EUR", "eur"), ("USD", "eur")],
                    names=["local_ccy", "collateral_ccy"],
                ),
            ),
        ),
        (
            XCS(
                dt(2022, 1, 5),
                "3M",
                "M",
                currency="eur",
                leg2_currency="usd",
                curves=["eureur", "eurusd", "usdusd", "usdusd"],
            ),
            DataFrame(
                [
                    [1000000.0, -1100306.44592],
                    [0.0, -2377.85237],
                    [-2042.44624, 4630.97800],
                    [0.0, -2152.15417],
                    [-1844.59236, 4191.00589],
                    [-1000000, 1104836.45246],
                    [-2042.44624, 4650.04393],
                ],
                index=Index(
                    [
                        dt(2022, 1, 5),
                        dt(2022, 2, 5),
                        dt(2022, 2, 7),
                        dt(2022, 3, 5),
                        dt(2022, 3, 7),
                        dt(2022, 4, 5),
                        dt(2022, 4, 7),
                    ],
                    name="payment",
                ),
                columns=MultiIndex.from_tuples(
                    [("EUR", "usd"), ("USD", "usd")],
                    names=["local_ccy", "collateral_ccy"],
                ),
            ),
        ),
        (
            FXSwap(
                dt(2022, 1, 5),
                "3M",
                currency="eur",
                leg2_currency="usd",
                curves=["eureur", "eurusd", "usdusd", "usdusd"],
            ),
            DataFrame(
                [[1000000.0, -1100306.44592], [-1005943.73163, 1113805.13741]],
                index=Index([dt(2022, 1, 5), dt(2022, 4, 5)], name="payment"),
                columns=MultiIndex.from_tuples(
                    [("EUR", "usd"), ("USD", "usd")],
                    names=["local_ccy", "collateral_ccy"],
                ),
            ),
        ),
    ],
)
def test_fx_settlements_table(inst, expected) -> None:
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
        },
    )
    usdeur = fxf.curve("usd", "eur", id="usdeur")
    eur_eurusd = fxf.curve("eur", ["usd", "eur"], id="eur_eurusd")

    solver = Solver(
        curves=[usdusd, eureur, eurusd, usdeur, eur_eurusd],
        instruments=[
            IRS(dt(2022, 1, 1), "1y", "A", curves=usdusd),
            IRS(dt(2022, 1, 1), "1y", "A", curves=eureur),
            XCS(
                dt(2022, 1, 1),
                "1y",
                "Q",
                currency="eur",
                leg2_currency="usd",
                curves=[eureur, eurusd, usdusd, usdusd],
            ),
        ],
        s=[5.0, 2.5, -10],
        fx=fxf,
    )
    assert eureur.meta.collateral == "eur"  # collateral tags populated by FXForwards

    pf = Portfolio([inst])
    result = pf.cashflows_table(solver=solver)
    assert_frame_equal(expected, result, atol=1e-4)

    result = inst.cashflows_table(solver=solver)
    assert_frame_equal(expected, result, atol=1e-4)


def test_fx_settlements_table_no_fxf() -> None:
    solver = Solver(
        curves=[Curve({dt(2023, 8, 1): 1.0, dt(2024, 8, 1): 1.0}, id="usd")],
        instruments=[IRS(dt(2023, 8, 1), "1Y", "Q", curves="usd")],
        s=[2.0],
        instrument_labels=["1Y"],
        id="us_rates",
        algorithm="gauss_newton",
    )
    irs_mkt = IRS(
        dt(2023, 8, 1),
        "1Y",
        "Q",
        curves="usd",
        fixed_rate=2.0,
        notional=999556779.81,
    )
    result = irs_mkt.cashflows_table(solver=solver)
    assert abs(result.iloc[0, 0] - 69.49810) < 1e-5
    assert abs(result.iloc[3, 0] - 69.49810) < 1e-5


@pytest.fixture
def fxfo():
    # FXForwards for FX Options tests
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
    eurusd = Curve({dt(2023, 3, 16): 1.0, dt(2023, 9, 16): 0.987092591908283}, id="eurusd")
    fxr = FXRates({"eurusd": 1.0615}, settlement=dt(2023, 3, 20))
    fxf = FXForwards(fx_curves={"eureur": eureur, "eurusd": eurusd, "usdusd": usdusd}, fx_rates=fxr)
    # fxf.swap("eurusd", [dt(2023, 3, 20), dt(2023, 6, 20)]) = 60.10
    return fxf


class TestFXOptions:
    # replicate https://quant.stackexchange.com/a/77802/29443
    @pytest.mark.parametrize(
        ("pay", "k", "exp_pts", "exp_prem", "dlty", "exp_dl"),
        [
            (dt(2023, 3, 20), 1.101, 69.378, 138756.54, "spot", 0.250124),
            (dt(2023, 3, 20), 1.101, 69.378, 138756.54, "forward", 0.251754),
            (dt(2023, 6, 20), 1.101, 70.226, 140451.53, "spot", 0.250124),
            (dt(2023, 6, 20), 1.101, 70.226, 140451.53, "forward", 0.251754),
            (dt(2023, 6, 20), 1.10101922, 70.180, 140360.17, "spot", 0.250000),
        ],
    )
    @pytest.mark.parametrize("smile", [True, False])
    def test_big_usd_pips(self, fxfo, pay, k, exp_pts, exp_prem, dlty, exp_dl, smile) -> None:
        vol = FXDeltaVolSmile(
            {
                0.75: 8.9,
            },
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            delta_type="spot",
            id="vol",
            ad=1,
        )
        vol = vol if smile else 8.90
        fxc = FXCall(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            strike=k,
            payment_lag=pay,
            delivery_lag=2,
            calendar="tgt",
            modifier="mf",
            premium_ccy="usd",
            delta_type=dlty,
        )
        result = fxc.rate(
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            fx=fxfo,
            vol=vol,
        )
        assert abs(result - exp_pts) < 1e-3

        result = fxc.rate(
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            fx=fxfo,
            vol=vol,
            metric="premium",
        )
        assert abs(result - exp_prem) < 1e-2

    @pytest.mark.parametrize(
        ("pay", "k", "exp_pts", "exp_prem", "exp_dl"),
        [
            (dt(2023, 3, 20), 1.101, 0.6536, 130717.44, 0.245175),
            (dt(2023, 6, 20), 1.101, 0.6578, 131569.29, 0.245178),
        ],
    )
    @pytest.mark.parametrize(
        "vol",
        [
            8.9,
            FXDeltaVolSmile(
                nodes={0.5: 8.9},
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type="forward",
            ),
            FXSabrSmile(
                nodes={"alpha": 0.089, "beta": 1.0, "rho": 0.0, "nu": 0.0},
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
            ),
        ],
    )
    def test_premium_big_eur_pc(self, fxfo, pay, k, exp_pts, exp_prem, exp_dl, vol) -> None:
        fxo = FXCall(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery_lag=dt(2023, 6, 20),
            payment_lag=pay,
            strike=k,
            notional=20e6,
            delta_type="forward",
            premium_ccy="eur",
        )
        result = fxo.rate(
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            fx=fxfo,
            vol=vol,
        )
        expected = exp_pts
        assert abs(result - expected) < 1e-3

        result = 20e6 * result / 100
        expected = exp_prem
        assert abs(result - expected) < 1e-1

    @pytest.mark.parametrize(
        ("pay", "k", "exp_pts", "exp_prem", "exp_dl"),
        [
            (dt(2023, 3, 20), 1.101, 0.6536, 130717.44, 0.243588),
            (dt(2023, 6, 20), 1.101, 0.6578, 131569.29, 0.243548),
        ],
    )
    @pytest.mark.parametrize(
        "vol",
        [
            8.9,
            FXDeltaVolSmile(
                nodes={0.5: 8.9},
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type="spot",
            ),
            FXSabrSmile(
                nodes={"alpha": 0.089, "beta": 1.0, "rho": 0.0, "nu": 0.0},
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
            ),
        ],
    )
    def test_premium_big_eur_pc_spot(self, fxfo, pay, k, exp_pts, exp_prem, exp_dl, vol) -> None:
        fxo = FXCall(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery_lag=dt(2023, 6, 20),
            payment_lag=pay,
            strike=k,
            notional=20e6,
            delta_type="spot",
            premium_ccy="eur",
        )
        result = fxo.rate(
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            fx=fxfo,
            vol=vol,
        )
        expected = exp_pts
        assert abs(result - expected) < 1e-3

        result = 20e6 * result / 100
        expected = exp_prem
        assert abs(result - expected) < 1e-1

    def test_fx_call_npv_unpriced(self, fxfo) -> None:
        fxo = FXCall(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike=1.101,
        )
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.npv(curves, fx=fxfo, vol=8.9)
        expected = 0.0
        assert abs(result - expected) < 1e-6

    def test_fx_call_cashflows(self, fxfo) -> None:
        fxo = FXCall(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike=1.101,
        )
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.cashflows(curves, fx=fxfo, vol=8.9)
        assert isinstance(result, DataFrame)
        assert result.loc[0, "Type"] == "FXCallPeriod"
        assert result.loc[1, "Type"] == "Cashflow"

    def test_fx_call_cashflows_table(self, fxfo) -> None:
        fxo = FXCall(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike=1.101,
        )
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.cashflows_table(curves, fx=fxfo, vol=8.9)
        expected = DataFrame(
            data=[[0.0]],
            index=Index([dt(2023, 6, 20)], name="payment"),
            columns=MultiIndex.from_tuples([("USD", "usd")], names=["local_ccy", "collateral_ccy"]),
        )
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        ("ccy", "exp_rate", "exp_strike"),
        [
            ("usd", 70.180131, 1.10101920113408469),
            ("eur", 0.680949, 1.099976),
        ],
    )
    @pytest.mark.parametrize(
        "vol",
        [
            8.90,
            FXDeltaVolSmile(
                {
                    0.75: 8.9,
                },
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type="spot",
                id="vol",
                ad=1,
            ),
            FXSabrSmile(
                nodes={"alpha": 0.089, "beta": 1.0, "rho": 0.0, "nu": 0.0},
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                id="vol",
                ad=1,
            ),
        ],
    )
    def test_fx_call_rate_delta_strike(self, fxfo, ccy, exp_rate, exp_strike, vol) -> None:
        fxo = FXCall(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike="25d",
            delta_type="spot",
            premium_ccy=ccy,
        )
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.rate(curves, fx=fxfo, vol=vol)
        expected = exp_rate
        assert abs(result - expected) < 1e-6
        assert abs(fxo.periods[0].strike - exp_strike) < 1e-4

    def test_fx_call_rate_expiry_tenor(self, fxfo) -> None:
        fxo = FXCall(
            pair="eurusd",
            expiry="3m",
            eval_date=dt(2023, 3, 16),
            modifier="mf",
            notional=20e6,
            delivery_lag=2,
            payment_lag=dt(2023, 6, 20),
            calendar="tgt",
            strike="25d",
            delta_type="spot",
        )
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.rate(curves, fx=fxfo, vol=8.9)
        expected = 70.180131
        assert abs(result - expected) < 1e-6

    def test_fx_call_plot_payoff(self, fxfo) -> None:
        fxc = FXCall(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            strike=1.101,
            premium=0.0,
        )
        result = fxc.plot_payoff(
            [1.03, 1.12],
            fx=fxfo,
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
        )
        x, y = result[2][0]._x, result[2][0]._y
        assert x[0] == 1.03
        assert x[1000] == 1.12
        assert y[0] == 0.0
        assert y[1000] == (1.12 - 1.101) * 20e6

    def test_fx_put_rate(self, fxfo) -> None:
        fxo = FXPut(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike="-25d",
            delta_type="spot",
        )
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.rate(curves, fx=fxfo, vol=10.15)
        expected = 83.975596
        assert abs(result - expected) < 1e-6

    def test_str_tenor_raises(self) -> None:
        with pytest.raises(ValueError, match="`expiry` as string tenor requires `eval_date`"):
            FXCall(
                pair="eurusd",
                expiry="3m",
            )

    def test_premium_ccy_raises(self) -> None:
        with pytest.raises(
            ValueError,
            match="`premium_ccy`: 'chf' must be one of option currency pair",
        ):
            FXCall(
                pair="eurusd",
                expiry="3m",
                eval_date=dt(2023, 3, 16),
                premium_ccy="chf",
            )

    @pytest.mark.parametrize("dlty", [("forward")])
    def test_call_put_parity_50d(self, fxfo, dlty) -> None:
        fxp = FXPut(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike="-50d",
            premium_ccy="usd",
            delta_type=dlty,
        )
        fxc = FXCall(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike="50d",
            premium_ccy="usd",
            delta_type=dlty,
        )
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        assert abs(fxc.analytic_greeks(curves, fx=fxfo, vol=10.0)["delta"] - 0.5) < 1e-14
        assert abs(fxc.periods[0].strike - 1.068856) < 1e-6
        assert abs(fxp.analytic_greeks(curves, fx=fxfo, vol=10.0)["delta"] + 0.5) < 1e-14
        assert abs(fxp.periods[0].strike - 1.068856) < 1e-6

    def test_analytic_vega(self, fxfo) -> None:
        fxo = FXCall(
            pair="eurusd",
            expiry="3m",
            eval_date=dt(2023, 3, 16),
            modifier="mf",
            notional=20e6,
            delivery_lag=2,
            payment_lag=dt(2023, 3, 16),
            calendar="tgt",
            strike=1.101,
            delta_type="spot",
        )
        result = fxo.analytic_greeks(
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            fx=fxfo,
            vol=8.9,
        )["vega"]
        # see test_periods/test_analytic_vega
        assert abs(result * 20e6 / 100 - 33757.945) < 1e-2

    def test_rate_vol_raises(self, fxfo) -> None:
        args = {
            "expiry": dt(2009, 6, 16),
            "pair": "eurusd",
            "curves": [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            "delta_type": "spot",
        }
        vol = FXDeltaVolSmile(
            {0.75: 8.9},
            eval_date=dt(2009, 3, 16),
            expiry=dt(2023, 6, 16),
            delta_type="spot",
            id="vol",
            ad=1,
        )
        fxc = FXCall(strike=1.10, **args, notional=100e6, vol=vol)
        with pytest.raises(ValueError, match="The `eval_date` on the FXDeltaVolSmile and the"):
            fxc.rate(fx=fxfo)

    @pytest.mark.parametrize("phi", [-1.0, 1.0])
    @pytest.mark.parametrize("prem_ccy", ["usd", "eur"])
    @pytest.mark.parametrize("dt_0", ["spot", "forward"])
    @pytest.mark.parametrize("dt_1", ["spot", "forward", "spot_pa", "forward_pa"])
    @pytest.mark.parametrize("smile", [True, False])
    def test_atm_rates(self, fxfo, phi, prem_ccy, smile, dt_0, dt_1) -> None:
        FXOp = FXCall if phi > 0 else FXPut
        fxvs = FXDeltaVolSmile(
            {0.25: 10.15, 0.5: 7.8, 0.75: 8.9},
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            delta_type=dt_1,
            id="vol",
        )
        vol = fxvs if smile else 9.50
        fxo = FXOp(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery_lag=dt(2023, 6, 20),
            payment_lag=dt(2023, 6, 20),
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            delta_type=dt_0,
            vol=vol,
            premium_ccy=prem_ccy,
            strike="atm_delta",
        )
        result = fxo.analytic_greeks(fx=fxfo)

        f_d = fxfo.rate("eurusd", dt(2023, 6, 20))
        eta = 0.5 if prem_ccy == "usd" else -0.5
        expected = f_d * dual_exp(result["__vol"] ** 2 * fxvs.meta.t_expiry * eta)
        assert abs(result["__strike"] - expected) < 1e-8

    @pytest.mark.parametrize("phi", [-1.0, 1.0])
    @pytest.mark.parametrize("prem_ccy", ["usd", "eur"])
    @pytest.mark.parametrize("dt_0", ["spot", "forward"])
    def test_atm_rates_sabr(self, fxfo, phi, prem_ccy, dt_0) -> None:
        FXOp = FXCall if phi > 0 else FXPut
        vol = FXSabrSmile(
            {"alpha": 0.072, "beta": 1.0, "rho": -0.1, "nu": 0.80},
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            id="vol",
        )
        fxo = FXOp(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery_lag=dt(2023, 6, 20),
            payment_lag=dt(2023, 6, 20),
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            delta_type=dt_0,
            vol=vol,
            premium_ccy=prem_ccy,
            strike="atm_delta",
        )
        result = fxo.analytic_greeks(fx=fxfo)

        f_d = fxfo.rate("eurusd", dt(2023, 6, 20))
        eta = 0.5 if prem_ccy == "usd" else -0.5
        expected = f_d * dual_exp(result["__vol"] ** 2 * vol.meta.t_expiry * eta)
        assert abs(result["__strike"] - expected) < 1e-8

    @pytest.mark.parametrize("phi", [1.0, -1.0])
    @pytest.mark.parametrize(
        ("vol_", "expected"),
        [
            (
                FXDeltaVolSmile(
                    {0.25: 10.15, 0.5: 7.8, 0.75: 8.9},
                    eval_date=dt(2023, 3, 16),
                    expiry=dt(2023, 6, 16),
                    delta_type="spot",
                ),
                8.899854,
            ),
            (
                FXSabrSmile(
                    nodes={"alpha": 0.078, "beta": 1.0, "rho": 0.03, "nu": 0.04},
                    eval_date=dt(2023, 3, 16),
                    expiry=dt(2023, 6, 16),
                ),
                7.799409,
            ),
            (
                FXSabrSurface(
                    expiries=[dt(2023, 5, 16), dt(2023, 7, 16)],
                    node_values=[
                        [0.078, 1.0, 0.03, 0.04],
                        [0.08, 1.0, 0.04, 0.05],
                    ],
                    eval_date=dt(2023, 3, 16),
                    pair="eurusd",
                    calendar="tgt|fed",
                ),
                7.934473,
            ),
        ],
    )
    def test_traded_option_rate_vol(self, fxfo, phi, vol_, expected) -> None:
        FXOp = FXCall if phi > 0 else FXPut
        fxo = FXOp(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery_lag=dt(2023, 6, 20),
            payment_lag=dt(2023, 6, 20),
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            delta_type="spot",
            premium_ccy="usd",
            strike=1.05,
            premium=100000.0,
        )
        result = fxo.rate(
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            vol=vol_,
            fx=fxfo,
            metric="vol",
        )
        assert abs(result - expected) < 1e-6

    def test_option_strike_premium_validation(self) -> None:
        with pytest.raises(ValueError, match="`strike` for FXOption must be set"):
            FXCall(
                pair="eurusd",
                expiry=dt(2023, 6, 16),
            )

        with pytest.raises(ValueError, match="FXOption with string delta as `strike` cannot be"):
            FXCall(pair="eurusd", expiry=dt(2023, 6, 16), strike="25d", premium=0.0)

    @pytest.mark.parametrize(
        ("notn", "expected", "phi"),
        [
            (1e6, [0.5, 500000], 1.0),
            (2e6, [0.5, 1000000], 1.0),
            (-2e6, [0.5, 1000000], 1.0),
            (1e6, [-0.5, -500000], -1.0),
            (2e6, [-0.5, -1000000], -1.0),
            (-2e6, [-0.5, -1000000], -1.0),
        ],
    )
    def test_greeks_delta_direction(self, fxfo, notn, expected, phi) -> None:
        # test the delta and delta_eur are not impacted by a Buy or Sell. Delta is expressed
        # relative to a Buy.
        FXOp = FXCall if phi > 0 else FXPut
        delta = f"{'-' if phi < 0 else ''}50d"
        fxo = fxo = FXOp(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery_lag=dt(2023, 6, 20),
            payment_lag=dt(2023, 6, 20),
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            delta_type="forward",
            premium_ccy="usd",
            strike=delta,
            notional=notn,
        )
        fxvs = FXDeltaVolSmile(
            {0.25: 10.15, 0.5: 7.8, 0.75: 8.9},
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            delta_type="forward",
        )
        result = fxo.analytic_greeks(
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            vol=fxvs,
            fx=fxfo,
        )
        assert abs(result["delta"] - expected[0]) < 1e-6
        assert abs(result["delta_eur"] - expected[1]) < 1e-6

    def test_metric_and_period_metric_compatible(self) -> None:
        # ensure that vol and pips_or_% can be interchanged

        eur = Curve({dt(2024, 6, 20): 1.0, dt(2024, 9, 30): 1.0}, calendar="tgt")
        usd = Curve({dt(2024, 6, 20): 1.0, dt(2024, 9, 30): 1.0}, calendar="nyc")
        eurusd = Curve({dt(2024, 6, 20): 1.0, dt(2024, 9, 30): 1.0})
        fxr = FXRates({"eurusd": 1.0727}, settlement=dt(2024, 6, 24))
        fxf = FXForwards(fx_rates=fxr, fx_curves={"eureur": eur, "eurusd": eurusd, "usdusd": usd})
        pre_solver = Solver(
            curves=[eur, usd, eurusd],
            instruments=[
                IRS(dt(2024, 6, 24), "3m", spec="eur_irs", curves=eur),
                IRS(dt(2024, 6, 24), "3m", spec="usd_irs", curves=usd),
                FXExchange(
                    pair="eurusd",
                    settlement=dt(2024, 9, 24),
                    curves=[None, eurusd, None, usd],
                ),
            ],
            s=[3.77, 5.51, 1.0775],
            fx=fxf,
        )

        smile = FXDeltaVolSmile(
            nodes={0.25: 5.0, 0.50: 5.0, 0.75: 5.0},
            eval_date=dt(2024, 6, 20),
            expiry=dt(2024, 9, 20),
            delta_type="spot",
        )
        fx_args = dict(
            expiry=dt(2024, 9, 20),
            pair="eurusd",
            delta_type="spot",
            metric="vol",  # note how the option is pre-configured with a metric as "vol"
            curves=[None, eurusd, None, usd],
            vol=smile,
            premium_ccy="eur",
            delivery_lag=2,
            payment_lag=2,
        )
        solver = Solver(
            pre_solvers=[pre_solver],
            curves=[smile],
            instruments=[
                FXPut(strike=1.0504, **fx_args),
                FXCall(strike=1.0728, **fx_args),
                FXCall(strike=1.0998, **fx_args),
            ],
            s=[7.621, 6.60, 6.12],
            fx=fxf,
        )

        result = FXCall(strike=1.0728, **fx_args).rate(metric="pips_or_%", solver=solver)
        expected = 1.543289  # % of EUR notional
        assert abs(result - expected) < 1e-6

        result = FXCall(strike=1.0728, **fx_args).rate(solver=solver)  # should default to "vol"
        expected = 6.60  # vol points
        assert abs(result - expected) < 1e-6

    @pytest.mark.parametrize(
        ("evald", "eom", "expected"),
        [
            (
                dt(2024, 4, 26),
                True,
                dt(2024, 5, 29),
            ),  # 2bd before 31st May (rolled from End of April)
            (
                dt(2024, 4, 26),
                False,
                dt(2024, 5, 28),
            ),  # 2bd before 30th May (rolled from 30th April)
        ],
    )
    def test_expiry_delivery_tenor_eom(self, evald, eom, expected) -> None:
        fxo = FXCall(
            pair="eurusd",
            expiry="1m",
            eval_date=evald,
            eom=eom,
            calendar="tgt|fed",
            modifier="mf",
            strike=1.0,
        )
        assert fxo.kwargs["expiry"] == expected

    def test_single_vol_not_no_input(self, fxfo):
        fxo = FXCall(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery_lag=dt(2023, 6, 20),
            payment_lag=dt(2023, 6, 20),
            curves=[None, fxfo.curve("eur", "eur"), None, fxfo.curve("usd", "eur")],
            delta_type="forward",
            premium_ccy="usd",
            strike=1.1,
            notional=1e6,
        )
        with pytest.raises(ValueError, match="`vol` must be supplied. Got"):
            fxo.rate(metric="vol", fx=fxfo)

    def test_hyper_parameter_setting_and_solver_interaction(self):
        # Define the interest rate curves for EUR, USD and X-Ccy basis
        usdusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="nyc", id="usdusd")
        eureur = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="tgt", id="eureur")
        eurusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, id="eurusd")

        # Create an FX Forward market with spot FX rate data
        fxr = FXRates({"eurusd": 1.0760}, settlement=dt(2024, 5, 9))
        fxf = FXForwards(
            fx_rates=fxr,
            fx_curves={"eureur": eureur, "usdusd": usdusd, "eurusd": eurusd},
        )

        pre_solver = Solver(
            curves=[eureur, eurusd, usdusd],
            instruments=[
                IRS(dt(2024, 5, 9), "3W", spec="eur_irs", curves="eureur"),
                IRS(dt(2024, 5, 9), "3W", spec="usd_irs", curves="usdusd"),
                FXSwap(
                    dt(2024, 5, 9), "3W", pair="eurusd", curves=[None, "eurusd", None, "usdusd"]
                ),
            ],
            s=[3.90, 5.32, 8.85],
            fx=fxf,
            id="rates_sv",
        )
        dv_smile = FXDeltaVolSmile(
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
        option_args = dict(
            pair="eurusd",
            expiry=dt(2024, 5, 28),
            calendar="tgt|fed",
            delta_type="spot",
            curves=[None, "eurusd", None, "usdusd"],
            vol="eurusd_3w_smile",
        )

        dv_solver = Solver(
            pre_solvers=[pre_solver],
            curves=[dv_smile],
            instruments=[
                FXStraddle(strike="atm_delta", **option_args),
                FXRiskReversal(strike=("-25d", "25d"), **option_args),
                FXRiskReversal(strike=("-10d", "10d"), **option_args),
                FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), **option_args),
                FXBrokerFly(strike=(("-10d", "10d"), "atm_delta"), **option_args),
            ],
            s=[5.493, -0.157, -0.289, 0.071, 0.238],
            fx=fxf,
            id="dv_solver",
        )
        fc = FXCall(
            expiry=dt(2024, 5, 28),
            pair="eurusd",
            strike=1.07,
            notional=100e6,
            curves=[None, "eurusd", None, "usdusd"],
            vol="eurusd_3w_smile",
            premium=98.216647 * 1e8 / 1e4,
            premium_ccy="usd",
            delta_type="spot",
        )
        assert abs(fc.npv(solver=dv_solver, base="usd")) < 1e-2
        delta = fc.delta(solver=dv_solver, base="usd").loc[("fx", "fx", "eurusd"), ("all", "usd")]
        gamma = fc.gamma(solver=dv_solver, base="usd").loc[
            ("all", "usd", "fx", "fx", "eurusd"), ("fx", "fx", "eurusd")
        ]

        fxr.update({"eurusd": 1.0761})
        pre_solver.iterate()
        dv_solver.iterate()

        result = fc.npv(solver=dv_solver, base="usd")
        expected = delta + 0.5 * gamma
        assert abs(result - expected) < 5e-2

        fxr.update({"eurusd": 1.0759})
        pre_solver.iterate()
        dv_solver.iterate()

        result = fc.npv(solver=dv_solver, base="usd")
        expected = -delta + 0.5 * gamma
        assert abs(result - expected) < 5e-2

    @pytest.mark.parametrize("k", [1.07, "25d", "atm_delta"])
    def test_pricing_with_interpolated_sabr_surface(self, k, fxfo):
        surf = FXSabrSurface(
            eval_date=dt(2023, 3, 16),
            expiries=[dt(2023, 6, 16), dt(2023, 10, 17)],
            node_values=[[0.05, 1.0, 0.03, 0.04], [0.055, 1.0, 0.04, 0.05]],
            pair="eurusd",
            calendar="tgt|fed",
            ad=1,
            id="v",
        )
        fxc = FXCall(
            expiry=dt(2023, 7, 21),
            pair="eurusd",
            calendar="tgt|fed",
            delta_type="spot",
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            vol=surf,
            strike=k,
        )
        fxc.rate(fx=fxfo)
        result = fxc._pricing
        assert abs(result.vol - 5.25) < 1e-2
        assert np.all(gradient(result.vol, vars=["v_0_0", "v_1_0"]) > 49.2)
        assert np.all(gradient(result.vol, vars=["v_0_0", "v_1_0"]) < 50.6)


class TestRiskReversal:
    @pytest.mark.parametrize(
        ("metric", "expected"),
        [
            ("pips_or_%", -13.795465),
            ("vol", -1.25),
            ("premium", -27590.930533),
        ],
    )
    def test_risk_reversal_rate_metrics(self, fxfo, metric, expected) -> None:
        fxo = FXRiskReversal(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike=["-25d", "25d"],
            delta_type="spot",
        )
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.rate(curves, fx=fxfo, vol=[10.15, 8.9], metric=metric)
        assert abs(result - expected) < 1e-6

    @pytest.mark.parametrize(
        ("prem", "prem_ccy", "local", "exp"),
        [
            ((NoInput(0), NoInput(0)), NoInput(0), False, 0.0),
            ((NoInput(0), NoInput(0)), "eur", False, 0.0),
            ((-167500.0, 140500.0), "usd", False, -219.590678),
            ((-167500 / 1.06751, 140500 / 1.06751), "eur", False, -219.590678),
            (
                (-167500 / 1.06751, 140500 / 1.06751),
                "eur",
                True,
                {"eur": 25121.646, "usd": -26879.673},
            ),
        ],
    )
    def test_risk_reversal_npv(self, fxfo, prem, prem_ccy, local, exp) -> None:
        fxo = FXRiskReversal(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike=[1.033, 1.101],
            premium=prem,
            premium_ccy=prem_ccy,
        )
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.npv(curves, fx=fxfo, vol=[10.15, 8.9], local=local)
        expected = exp
        if not local:
            assert abs(result - expected) < 1e-6
        else:
            for k in expected:
                assert abs(result[k] - expected[k]) < 1e-3

    def test_risk_reversal_plot(self, fxfo) -> None:
        fxo = FXRiskReversal(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike=[1.033, 1.101],
        )
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.plot_payoff([1.03, 1.12], curves, fx=fxfo, vol=[10.15, 8.9])
        x, y = result[2][0]._x, result[2][0]._y
        assert x[0] == 1.03
        assert x[1000] == 1.12
        assert abs(y[0] + (1.033 - 1.03) * 20e6) < 1e-5
        assert abs(y[1000] - (1.12 - 1.101) * 20e6) < 1e-5

    def test_rr_strike_premium_validation(self) -> None:
        with pytest.raises(ValueError, match="`strike` for FXRiskReversal must be set"):
            FXRiskReversal(
                pair="eurusd",
                expiry=dt(2023, 6, 16),
            )

        with pytest.raises(ValueError, match="FXRiskReversal with string delta as `strike` cannot"):
            FXRiskReversal(
                pair="eurusd",
                expiry=dt(2023, 6, 16),
                strike=["25d", "35d"],
                premium=[NoInput(0), 1.0],
            )

    @pytest.mark.parametrize(
        ("notn", "expected_grks", "expected_ccy"),
        [
            (1e6, [0.5, -1.329654, -0.035843], [500000, -14194.192533, -358.428628]),
            (2e6, [0.5, -1.329654, -0.035843], [1000000, -28388.384, -716.8572]),
            (-2e6, [0.5, -1.329654, -0.035843], [1000000, -28388.384, -716.8572]),
        ],
    )
    def test_greeks_delta_direction(self, fxfo, notn, expected_grks, expected_ccy) -> None:
        # test the delta and delta_eur are not impacted by a Buy or Sell. Delta is expressed
        # relative to a Buy.
        fxo = FXRiskReversal(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery_lag=dt(2023, 6, 20),
            payment_lag=dt(2023, 6, 20),
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            delta_type="forward",
            premium_ccy="usd",
            strike=["-30d", "20d"],
            notional=notn,
        )
        fxvs = FXDeltaVolSmile(
            {0.25: 10.15, 0.5: 7.8, 0.75: 8.9},
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            delta_type="forward",
        )
        result = fxo.analytic_greeks(
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            vol=fxvs,
            fx=fxfo,
        )
        assert abs(result["delta"] - expected_grks[0]) < 1e-6
        assert abs(result["gamma"] - expected_grks[1]) < 1e-6
        assert abs(result["vega"] - expected_grks[2]) < 1e-6

        assert abs(result["delta_eur"] - expected_ccy[0]) < 1e-2
        assert abs(result["gamma_eur_1%"] - expected_ccy[1]) < 1e-2
        assert abs(result["vega_usd"] - expected_ccy[2]) < 1e-2

    def test_repr(self):
        fxo = FXRiskReversal(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike=[1.033, 1.101],
        )
        expected = f"<rl.FXRiskReversal at {hex(id(fxo))}>"
        assert fxo.__repr__() == expected


class TestFXStraddle:
    @pytest.mark.parametrize(
        ("dlty", "strike", "ccy", "exp"),
        [
            # ("forward", ["50d", "-50d"], "usd", [1.068856203, 1.068856203]),
            # ("spot", ["50d", "-50d"], "usd", [1.06841799, 1.069294591]),
            ("spot", "atm_forward", "usd", [1.06750999, 1.06750999]),
            ("spot", "atm_spot", "usd", [1.061500, 1.061500]),
            ("forward", "atm_delta", "usd", [1.068856203, 1.068856203]),
            ("spot", "atm_delta", "usd", [1.068856203, 1.068856203]),
            ("spot", "atm_forward", "eur", [1.06750999, 1.06750999]),
            ("spot", "atm_spot", "eur", [1.061500, 1.061500]),
            ("forward", "atm_delta", "eur", [1.06616549, 1.06616549]),
            ("spot", "atm_delta", "eur", [1.06616549, 1.06616549]),
            # ("forward", ["50d", "-50d"], "eur", [1.0660752074, 1.06624508149]),  # pa strikes
            # ("spot", ["50d", "-50d"], "eur", [1.0656079102, 1.066656812]),  # pa strikes
        ],
    )
    @pytest.mark.parametrize("smile", [True, False])
    def test_straddle_strikes(self, fxfo, dlty, strike, ccy, exp, smile) -> None:
        fxvs = FXDeltaVolSmile(
            nodes={0.5: 10.0},
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            delta_type="forward",
        )
        vol_ = fxvs if smile else 10.0
        fxo = FXStraddle(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike=strike,
            premium_ccy=ccy,
            delta_type=dlty,
        )
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        fxo.npv(curves, fx=fxfo, vol=vol_)
        call_k = fxo.periods[0].periods[0].strike
        put_k = fxo.periods[1].periods[0].strike
        assert abs(call_k - exp[0]) < 1e-7
        assert abs(put_k - exp[1]) < 1e-7

    @pytest.mark.parametrize(
        ("metric", "expected"),
        [
            ("pips_or_%", 337.998151),
            ("vol", 7.9),
            ("premium", 675996.301147),
        ],
    )
    def test_straddle_rate_metrics(self, fxfo, metric, expected) -> None:
        fxo = FXStraddle(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike="atm_delta",
            delta_type="spot",
        )
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.rate(curves, fx=fxfo, vol=7.9, metric=metric)
        assert abs(result - expected) < 1e-6

    def test_strad_strike_premium_validation(self) -> None:
        with pytest.raises(ValueError, match="`strike` for FXStraddle must be set"):
            FXStraddle(
                pair="eurusd",
                expiry=dt(2023, 6, 16),
            )

        with pytest.raises(ValueError, match="FXStraddle with string delta as `strike` cannot"):
            FXStraddle(
                pair="eurusd",
                expiry=dt(2023, 6, 16),
                strike="25d",
                premium=[NoInput(0), 1.0],
            )

    @pytest.mark.parametrize(
        ("notn", "expected_grks", "expected_ccy"),
        [
            (1e6, [0.0, 19.086488, 0.422238], [0, 203750.1688, 4222.379]),
            (2e6, [0.0, 19.086488, 0.422238], [0, 407500.336, 8444.758]),
            (-2e6, [0.0, 19.086488, 0.422238], [0, 407500.336, 8444.758]),
        ],
    )
    def test_greeks_delta_direction(self, fxfo, notn, expected_grks, expected_ccy) -> None:
        # test the delta and delta_eur are not impacted by a Buy or Sell. Delta is expressed
        # relative to a Buy.
        fxo = FXStraddle(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery_lag=dt(2023, 6, 20),
            payment_lag=dt(2023, 6, 20),
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            delta_type="forward",
            premium_ccy="usd",
            strike="atm_delta",
            notional=notn,
        )
        fxvs = FXDeltaVolSmile(
            {0.25: 10.15, 0.5: 7.8, 0.75: 8.9},
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            delta_type="forward",
        )
        result = fxo.analytic_greeks(
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            vol=fxvs,
            fx=fxfo,
        )
        assert abs(result["delta"] - expected_grks[0]) < 1e-6
        assert abs(result["gamma"] - expected_grks[1]) < 1e-6
        assert abs(result["vega"] - expected_grks[2]) < 1e-6

        assert abs(result["delta_eur"] - expected_ccy[0]) < 1e-2
        assert abs(result["gamma_eur_1%"] - expected_ccy[1]) < 1e-2
        assert abs(result["vega_usd"] - expected_ccy[2]) < 1e-2

    def test_repr(self):
        fxo = FXStraddle(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike=1.0,
        )
        expected = f"<rl.FXStraddle at {hex(id(fxo))}>"
        assert expected == fxo.__repr__()


class TestFXStrangle:
    @pytest.mark.parametrize(
        ("strike", "ccy"),
        [
            ([1.02, 1.10], "usd"),
            (["-20d", "20d"], "usd"),
            ([1.02, 1.10], "eur"),
            (["-20d", "20d"], "eur"),
        ],
    )
    @pytest.mark.parametrize(
        "vol",
        [
            FXDeltaVolSmile(
                nodes={
                    0.25: 10.15,
                    0.50: 7.9,
                    0.75: 8.9,
                },
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type="forward",
            ),
            FXDeltaVolSmile(
                nodes={
                    0.25: 10.15,
                    0.50: 7.9,
                    0.75: 8.9,
                },
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type="spot_pa",
            ),
            10.0,
            FXSabrSmile(
                nodes={
                    "alpha": 0.10,
                    "beta": 1.0,
                    "rho": 0.00,
                    "nu": 0.50,
                },
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
            ),
        ],
    )
    def test_strangle_rate_forward(self, fxfo, strike, ccy, vol) -> None:
        # test pricing a straddle with vol 10.0 returns 10.0
        fxo = FXStrangle(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike=strike,
            premium_ccy=ccy,
            delta_type="forward",
        )

        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.rate(curves, fx=fxfo, vol=vol)

        premium = fxo.rate(curves, fx=fxfo, vol=result, metric="pips_or_%")
        metric = "pips" if ccy == "usd" else "percent"
        premium_vol = (
            fxo.periods[0]
            .periods[0]
            .rate(
                fxfo.curve("eur", "usd"),
                fxfo.curve("usd", "usd"),
                fx=fxfo,
                vol=vol,
                metric=metric,
            )
        )
        premium_vol += (
            fxo.periods[1]
            .periods[0]
            .rate(
                fxfo.curve("eur", "usd"),
                fxfo.curve("usd", "usd"),
                fx=fxfo,
                vol=vol,
                metric=metric,
            )
        )
        assert abs(premium - premium_vol) < 5e-2

    @pytest.mark.parametrize(
        "vol",
        [
            FXDeltaVolSmile(
                nodes={
                    0.25: 10.15,
                    0.50: 7.9,
                    0.75: 8.9,
                },
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type="spot",
                ad=1,
            ),
            FXSabrSmile(
                nodes={
                    "alpha": 0.079,
                    "beta": 1.0,
                    "rho": 0.00,
                    "nu": 0.50,
                },
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
            ),
        ],
    )
    def test_strangle_rate_strike_str(self, fxfo, vol) -> None:
        # test pricing a strangle with delta as string that is not a delta percent should fail?
        fxo = FXStrangle(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike=["atm_spot", "atm_forward"],
            premium_ccy="eur",
            delta_type="forward",
        )
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.rate(curves, fx=fxfo, vol=vol)

        premium = fxo.rate(curves, fx=fxfo, vol=result, metric="pips_or_%")
        metric = "percent"
        premium_vol = (
            fxo.periods[0]
            .periods[0]
            .rate(
                fxfo.curve("eur", "usd"),
                fxfo.curve("usd", "usd"),
                fx=fxfo,
                vol=vol,
                metric=metric,
            )
        )
        premium_vol += (
            fxo.periods[1]
            .periods[0]
            .rate(
                fxfo.curve("eur", "usd"),
                fxfo.curve("usd", "usd"),
                fx=fxfo,
                vol=vol,
                metric=metric,
            )
        )
        assert abs(premium - premium_vol) < 5e-2

    @pytest.mark.parametrize(
        "vol",
        [
            FXDeltaVolSmile(
                nodes={
                    0.25: 10.15,
                    0.50: 7.9,
                    0.75: 8.9,
                },
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type="spot",
                ad=1,
                id="vol",
            ),
            FXSabrSmile(
                nodes={
                    "alpha": 0.079,
                    "beta": 1.0,
                    "rho": 0.00,
                    "nu": 0.50,
                },
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                ad=1,
                id="vol",
            ),
        ],
    )
    def test_strangle_rate_ad(self, fxfo, vol) -> None:
        # test pricing a strangle with delta as string that is not a delta percent should fail?
        fxo = FXStrangle(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike=["atm_spot", "atm_forward"],
            premium_ccy="eur",
            delta_type="forward",
        )
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.rate(curves, fx=fxfo, vol=vol)

        # test fwd diff
        v = vol._get_node_vector()
        m_ = {
            0: [0.001, 0.0, 0.0],
            1: [0.0, 0.001, 0.0],
            2: [0.0, 0.0, 0.001],
        }
        for i in range(3):
            vol._set_node_vector(v + np.array(m_[i]), ad=1)
            result2 = fxo.rate(curves, fx=fxfo, vol=vol)
            fwd_diff = (result2 - result) * 1000.0
            assert abs(fwd_diff - gradient(result, [f"vol{i}"])[0]) < 2e-4

    @pytest.mark.parametrize(
        "vol",
        [
            FXDeltaVolSmile(
                nodes={
                    0.25: 10.15,
                    0.50: 7.9,
                    0.75: 8.9,
                },
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type="spot",
                ad=2,
                id="vol",
            ),
            FXSabrSmile(
                nodes={
                    "alpha": 0.079,
                    "beta": 1.0,
                    "rho": 0.00,
                    "nu": 0.50,
                },
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                ad=2,
                id="vol",
            ),
        ],
    )
    def test_strangle_rate_ad2(self, fxfo, vol) -> None:
        # test pricing a strangle with delta as string that is not a delta percent should fail?
        fxo = FXStrangle(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike=["atm_spot", "atm_forward"],
            premium_ccy="eur",
            delta_type="forward",
        )
        fxfo._set_ad_order(2)
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.rate(curves, fx=fxfo, vol=vol)

        # test fwd diff
        m_ = {
            0: [0.001, 0.0, 0.0],
            1: [0.0, 0.001, 0.0],
            2: [0.0, 0.0, 0.001],
        }
        n_ = {
            0: [-0.001, 0.0, 0.0],
            1: [0.0, -0.001, 0.0],
            2: [0.0, 0.0, -0.001],
        }
        v = vol._get_node_vector()
        for i in range(3):
            vol._set_node_vector(v + np.array(m_[i]), ad=2)
            result_plus = fxo.rate(curves, fx=fxfo, vol=vol)
            vol._set_node_vector(v + np.array(n_[i]), ad=2)
            result_min = fxo.rate(curves, fx=fxfo, vol=vol)

            fwd_diff = (result_plus + result_min - 2 * result) * 1000000.0
            assert abs(fwd_diff - gradient(result, [f"vol{i}"], order=2)[0][0]) < 1e-4

    def test_strangle_rate_2vols(self, fxfo) -> None:
        # test pricing a straddle with vol [8.0, 10.0] returns a valid value close to 9.0
        fxo = FXStrangle(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=20e6,
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike=["-25d", "25d"],
            premium_ccy="usd",
            delta_type="forward",
        )
        vol = [8.0, 10.0]
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.rate(curves, fx=fxfo, vol=vol)

        premium = fxo.rate(curves, fx=fxfo, vol=result, metric="pips_or_%")
        premium_vol = (
            fxo.periods[0]
            .periods[0]
            .rate(
                fxfo.curve("eur", "usd"),
                fxfo.curve("usd", "usd"),
                fx=fxfo,
                vol=vol[0],
            )
        )
        premium_vol += (
            fxo.periods[1]
            .periods[0]
            .rate(
                fxfo.curve("eur", "usd"),
                fxfo.curve("usd", "usd"),
                fx=fxfo,
                vol=vol[1],
            )
        )

        assert abs(premium - premium_vol) < 5e-2

    @pytest.mark.parametrize(
        ("notn", "expected_grks", "expected_ccy"),
        [
            (1e6, [-0.026421, 10.217368, 0.294605], [-26421.408, 109071.429, 2946.046]),
            (2e6, [-0.026421, 10.217368, 0.294605], [-52842.816, 218142.858, 5892.092]),
            (-2e6, [-0.026421, 10.217368, 0.294605], [-52842.816, 218142.858, 5892.092]),
        ],
    )
    @pytest.mark.parametrize("strikes", [("-20d", "20d"), (1.0238746345527665, 1.1159199351325004)])
    def test_greeks_delta_direction(self, fxfo, notn, expected_grks, expected_ccy, strikes) -> None:
        # test the delta and delta_eur are not impacted by a Buy or Sell. Delta is expressed
        # relative to a Buy.
        fxo = FXStrangle(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery_lag=dt(2023, 6, 20),
            payment_lag=dt(2023, 6, 20),
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            delta_type="forward",
            premium_ccy="usd",
            strike=strikes,
            notional=notn,
        )
        fxvs = FXDeltaVolSmile(
            {0.25: 10.15, 0.5: 7.8, 0.75: 8.9},
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            delta_type="forward",
        )
        result = fxo.analytic_greeks(
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            vol=fxvs,
            fx=fxfo,
        )
        assert abs(result["delta"] - expected_grks[0]) < 1e-6
        assert abs(result["gamma"] - expected_grks[1]) < 1e-6
        assert abs(result["vega"] - expected_grks[2]) < 1e-6

        assert abs(result["delta_eur"] - expected_ccy[0]) < 1e-1
        assert abs(result["gamma_eur_1%"] - expected_ccy[1]) < 1e-1
        assert abs(result["vega_usd"] - expected_ccy[2]) < 1e-1

    def test_strang_strike_premium_validation(self) -> None:
        with pytest.raises(ValueError, match="`strike` for FXStrangle must be set"):
            FXStrangle(
                pair="eurusd",
                expiry=dt(2023, 6, 16),
                strike=["25d", NoInput(0)],
            )

        with pytest.raises(ValueError, match="FXStrangle with string delta as `strike` cannot"):
            FXStrangle(
                pair="eurusd",
                expiry=dt(2023, 6, 16),
                strike=["25d", "35d"],
                premium=[NoInput(0), 1.0],
            )

    def test_repr(self):
        fxo = FXStrangle(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery_lag=dt(2023, 6, 20),
            payment_lag=dt(2023, 6, 20),
            delta_type="forward",
            premium_ccy="usd",
            strike=[1.0, 1.1],
        )
        expected = f"<rl.FXStrangle at {hex(id(fxo))}>"
        assert expected == fxo.__repr__()


class TestFXBrokerFly:
    @pytest.mark.parametrize(
        ("strike", "ccy"),
        [
            ([[1.024, 1.116], 1.0683], "usd"),
            ([["-20d", "20d"], "atm_delta"], "usd"),
            ([[1.024, 1.116], 1.0683], "eur"),
            ([["-20d", "20d"], "atm_delta"], "eur"),
        ],
    )
    @pytest.mark.parametrize(
        ("vol", "expected"),
        [
            (
                FXDeltaVolSmile(
                    nodes={
                        0.25: 10.15,
                        0.50: 7.9,
                        0.75: 8.9,
                    },
                    eval_date=dt(2023, 3, 16),
                    expiry=dt(2023, 6, 16),
                    delta_type="forward",
                ),
                2.225,
            ),
            (
                FXDeltaVolSmile(
                    nodes={
                        0.25: 10.15,
                        0.50: 7.9,
                        0.75: 8.9,
                    },
                    eval_date=dt(2023, 3, 16),
                    expiry=dt(2023, 6, 16),
                    delta_type="spot_pa",
                ),
                2.39,
            ),
            (9.5, 0.0),
            (
                FXSabrSmile(
                    nodes={
                        "alpha": 0.071,
                        "beta": 1.0,
                        "rho": 0.00,
                        "nu": 2.5,
                    },
                    eval_date=dt(2023, 3, 16),
                    expiry=dt(2023, 6, 16),
                ),
                2.065,
            ),
        ],
    )
    def test_fxbf_rate(self, fxfo, strike, ccy, vol, expected) -> None:
        # test pricing a straddle with vol 10.0 returns 10.0
        fxo = FXBrokerFly(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=[20e6, NoInput(0)],
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike=strike,
            premium_ccy=ccy,
            delta_type="forward",
        )
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.rate(curves, fx=fxfo, vol=vol)

        assert abs(result - expected) < 3e-2

    @pytest.mark.parametrize(
        ("strike", "ccy"),
        [
            ([[1.024, 1.116], 1.0683], "usd"),
            ([["-20d", "20d"], "atm_delta"], "usd"),
            ([[1.0228, 1.1147], 1.0683], "eur"),
            ([["-20d", "20d"], "atm_delta"], "eur"),
        ],
    )
    @pytest.mark.parametrize("smile", [True])
    def test_fxbf_rate_pips(self, fxfo, strike, ccy, smile) -> None:
        fxo = FXBrokerFly(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=[20e6, NoInput(0)],
            delivery_lag=2,
            payment_lag=2,
            calendar="tgt",
            strike=strike,
            premium_ccy=ccy,
            delta_type="forward",
            metric="pips_or_%",
        )
        fxvs = FXDeltaVolSmile(
            nodes={
                0.25: 10.15,
                0.50: 7.8,
                0.75: 8.9,
            },
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            delta_type="spot",
        )
        vol = fxvs if smile else 9.5
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.rate(curves, fx=fxfo, vol=vol)
        expected = (-111.2, 0.1) if ccy == "usd" else (-1.041, 0.02)
        assert abs(result - expected[0]) < expected[1]

    @pytest.mark.parametrize(
        ("strike", "ccy"),
        [
            ([[1.024, 1.116], 1.0683], "usd"),
            ([["-20d", "20d"], "atm_delta"], "usd"),
            ([[1.024, 1.116], 1.06668], "eur"),
            ([["-20d", "20d"], "atm_delta"], "eur"),
        ],
    )
    @pytest.mark.parametrize(
        ("vol", "expected"),
        [
            (
                FXDeltaVolSmile(
                    nodes={
                        0.25: 10.15,
                        0.50: 7.8,
                        0.75: 8.9,
                    },
                    eval_date=dt(2023, 3, 16),
                    expiry=dt(2023, 6, 16),
                    delta_type="forward",
                ),
                (-221743, -210350),
            ),
            (
                FXSabrSmile(
                    nodes={
                        "alpha": 0.071,
                        "beta": 1.0,
                        "rho": 0.00,
                        "nu": 2.5,
                    },
                    eval_date=dt(2023, 3, 16),
                    expiry=dt(2023, 6, 16),
                ),
                (-240740, -225500),
            ),
        ],
    )
    def test_fxbf_rate_premium(self, fxfo, strike, ccy, vol, expected) -> None:
        fxo = FXBrokerFly(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=[20e6, NoInput(0)],
            delivery_lag=dt(2023, 6, 20),
            payment_lag=dt(2023, 6, 20),
            strike=strike,
            premium_ccy=ccy,
            delta_type="forward",
            metric="premium",
        )
        curves = [None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")]
        result = fxo.rate(curves, fx=fxfo, vol=vol)

        tolerance = 300 if ccy == "usd" else 800
        expected = expected[0] if ccy == "usd" else expected[1]
        assert abs(result - expected) < tolerance

    def test_bf_rate_vols_list(self, fxfo) -> None:
        fxbf = FXBrokerFly(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            notional=[20e6, -13.5e6],
            strike=(("-20d", "20d"), "atm_delta"),
            payment_lag=2,
            delivery_lag=2,
            calendar="tgt",
            premium_ccy="usd",
            delta_type="spot",
        )
        result = fxbf.rate(
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            fx=fxfo,
            vol=[[10.15, 8.9], 1.0],
        )
        expected = 8.539499
        assert abs(result - expected) < 1e-6

        result = fxbf.rate(
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            fx=fxfo,
            vol=[[10.15, 8.9], 7.8],
            metric="pips_or_%",
        )
        expected = -110.098920
        assert abs(result - expected) < 1e-6

    @pytest.mark.parametrize(
        ("notn", "expected_grks", "expected_ccy"),
        [
            ([1e6, NoInput(0)], [-0.026421, -3.099693, 0.000000], [-26421.408, -33089.534, 0.000]),
            ([2e6, NoInput(0)], [-0.026421, -3.099693, 0.000000], [-52842.816, -66179.068, 0.000]),
            ([-2e6, NoInput(0)], [-0.026421, -3.099693, 0.000000], [-52842.816, -66179.068, 0.000]),
            ([1e6, -600e3], [-0.026421, -1.234524, 0.041262], [-26421.408, -13178.672, 412.619]),
        ],
    )
    @pytest.mark.parametrize(
        "strikes",
        [
            (("-20d", "20d"), "atm_delta"),
            ((1.0238746345527665, 1.1159199351325004), 1.0683288279019205),
        ],
    )
    def test_greeks_delta_direction(self, fxfo, notn, expected_grks, expected_ccy, strikes) -> None:
        # test the delta and delta_eur are not impacted by a Buy or Sell. Delta is expressed
        # relative to a Buy.
        fxo = FXBrokerFly(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery_lag=dt(2023, 6, 20),
            payment_lag=dt(2023, 6, 20),
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            delta_type="forward",
            premium_ccy="usd",
            strike=strikes,
            notional=notn,
        )
        fxvs = FXDeltaVolSmile(
            {0.25: 10.15, 0.5: 7.8, 0.75: 8.9},
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            delta_type="forward",
        )
        result = fxo.analytic_greeks(
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            vol=fxvs,
            fx=fxfo,
        )
        assert abs(result["delta"] - expected_grks[0]) < 1e-6
        assert abs(result["gamma"] - expected_grks[1]) < 1e-4
        assert abs(result["vega"] - expected_grks[2]) < 1e-5

        assert abs(result["delta_eur"] - expected_ccy[0]) < 1e-1
        assert abs(result["gamma_eur_1%"] - expected_ccy[1]) < 1.5
        assert abs(result["vega_usd"] - expected_ccy[2]) < 1e-1

    def test_single_vol_definition(self, fxfo) -> None:
        # test the metric of the rate can be input as "single_vol" and a result returned.
        fxvs = FXDeltaVolSmile(
            nodes={
                0.25: 10.15,
                0.50: 7.9,
                0.75: 8.9,
            },
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            delta_type="forward",
        )
        fxo = FXBrokerFly(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery_lag=dt(2023, 6, 20),
            payment_lag=dt(2023, 6, 20),
            curves=[None, fxfo.curve("eur", "usd"), None, fxfo.curve("usd", "usd")],
            delta_type="forward",
            premium_ccy="usd",
            strike=[["-20d", "20d"], "atm_delta"],
            vol=fxvs,
        )
        result = fxo.rate(metric="single_vol", fx=fxfo)
        expected = 10.147423 - 7.90
        assert (result - expected) < 1e-6

    def test_repr(self):
        fxo = FXBrokerFly(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery_lag=dt(2023, 6, 20),
            payment_lag=dt(2023, 6, 20),
            delta_type="forward",
            premium_ccy="usd",
            strike=[["-20d", "20d"], "atm_delta"],
        )
        expected = f"<rl.FXBrokerFly at {hex(id(fxo))}>"
        assert expected == fxo.__repr__()


class TestVolValue:
    def test_solver_passthrough(self) -> None:
        smile = FXDeltaVolSmile(
            nodes={0.25: 10.0, 0.5: 10.0, 0.75: 10.0},
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            delta_type="forward",
            id="VolSmile",
        )
        instruments = [
            VolValue(0.25, vol=smile),
            VolValue(0.5, vol="VolSmile"),
            VolValue(0.75, vol="VolSmile"),
        ]
        Solver(curves=[smile], instruments=instruments, s=[8.9, 8.2, 9.1])
        assert abs(smile[0.25] - 8.9) < 5e-7
        assert abs(smile[0.5] - 8.2) < 5e-7
        assert abs(smile[0.75] - 9.1) < 5e-7

    def test_solver_surface_passthrough(self) -> None:
        surface = FXDeltaVolSurface(
            delta_indexes=[0.5],
            expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
            node_values=[[1.0], [1.0]],
            eval_date=dt(1999, 12, 1),
            delta_type="forward",
            id="VolSurf",
        )
        instruments = [
            VolValue(0.25, dt(2000, 1, 1), vol=surface),
            VolValue(0.5, dt(2001, 1, 1), vol="VolSurf"),
        ]
        Solver(surfaces=[surface], instruments=instruments, s=[8.9, 8.2], func_tol=1e-14)
        assert abs(surface._get_index(0.5, dt(2000, 1, 1)) - 8.9) < 5e-7
        assert abs(surface._get_index(0.5, dt(2001, 1, 1)) - 8.2) < 5e-7

    def test_no_solver_vol_value(self) -> None:
        vv = VolValue(0.25, vol="string_id")
        with pytest.raises(ValueError, match="String `vol` ids require a `solver`"):
            vv.rate()

    def test_repr(self):
        v = VolValue(0.25)
        expected = f"<rl.VolValue at {hex(id(v))}>"
        assert v.__repr__() == expected
