from datetime import datetime as dt

import numpy as np
import pytest
from pandas import DataFrame, Index, MultiIndex, Series, date_range, isna
from pandas.testing import assert_frame_equal, assert_series_equal
from rateslib import default_context, defaults
from rateslib.curves import Curve
from rateslib.default import NoInput
from rateslib.dual import Dual
from rateslib.fx import FXForwards, FXRates
from rateslib.legs import (
    CreditPremiumLeg,
    CreditProtectionLeg,
    CustomLeg,
    FixedLeg,
    FixedLegMtm,
    FloatLeg,
    FloatLegMtm,
    IndexFixedLeg,
    ZeroFixedLeg,
    ZeroFloatLeg,
    ZeroIndexLeg,
)
from rateslib.periods import (
    Cashflow,
    CreditPremiumPeriod,
    CreditProtectionPeriod,
    FixedPeriod,
    FloatPeriod,
    IndexCashflow,
    IndexFixedPeriod,
)


@pytest.fixture
def curve():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.99,
        dt(2022, 7, 1): 0.98,
        dt(2022, 10, 1): 0.97,
    }
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture
def hazard_curve():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.999,
        dt(2022, 7, 1): 0.997,
        dt(2022, 10, 1): 0.991,
    }
    return Curve(nodes=nodes, interpolation="log_linear", id="hazard_fixture")


@pytest.mark.parametrize(
    "Leg",
    [
        FloatLeg,
        FixedLeg,
        ZeroFixedLeg,
        ZeroFloatLeg,
    ],
)
def test_repr(Leg):
    leg = Leg(
        effective=dt(2022, 1, 1),
        termination="1y",
        frequency="Q",
    )
    result = leg.__repr__()
    expected = f"<rl.{type(leg).__name__} at {hex(id(leg))}>"
    assert result == expected


@pytest.mark.parametrize(
    "Leg",
    [
        IndexFixedLeg,
        ZeroIndexLeg,
    ],
)
def test_repr_index(Leg):
    leg = Leg(effective=dt(2022, 1, 1), termination="1y", frequency="Q", index_base=100.0)
    result = leg.__repr__()
    expected = f"<rl.{type(leg).__name__} at {hex(id(leg))}>"
    assert result == expected


@pytest.mark.parametrize("Leg", [FixedLegMtm, FloatLegMtm])
def test_repr_mtm(Leg):
    leg = Leg(
        effective=dt(2022, 1, 1),
        termination="1y",
        frequency="Q",
        alt_currency="usd",
        currency="eur",
    )
    result = leg.__repr__()
    expected = f"<rl.{type(leg).__name__} at {hex(id(leg))}>"
    assert result == expected


def test_repr_custom():
    period = FixedPeriod(
        start=dt(2000, 1, 1), end=dt(2000, 2, 1), payment=dt(2000, 2, 1), frequency="m"
    )
    leg = CustomLeg([period])
    assert leg.__repr__() == f"<rl.CustomLeg at {hex(id(leg))}>"


class TestFloatLeg:
    @pytest.mark.parametrize(
        "obj",
        [
            (
                FloatLeg(
                    effective=dt(2022, 1, 1),
                    termination=dt(2022, 6, 1),
                    payment_lag=0,
                    notional=1e9,
                    convention="Act360",
                    frequency="Q",
                    fixing_method="rfr_payment_delay",
                    spread_compound_method="none_simple",
                    currency="nok",
                )
            ),
            (
                FloatLeg(
                    effective=dt(2022, 1, 1),
                    termination=dt(2022, 6, 1),
                    payment_lag=0,
                    payment_lag_exchange=0,
                    notional=1e9,
                    convention="Act360",
                    frequency="Q",
                    fixing_method="rfr_payment_delay",
                    spread_compound_method="none_simple",
                    currency="nok",
                    initial_exchange=True,
                    final_exchange=True,
                )
            ),
            (
                FloatLegMtm(
                    effective=dt(2022, 1, 1),
                    termination=dt(2022, 6, 1),
                    payment_lag=0,
                    payment_lag_exchange=0,
                    convention="Act360",
                    frequency="Q",
                    fixing_method="rfr_payment_delay",
                    spread_compound_method="none_simple",
                    currency="nok",
                    alt_currency="usd",
                    alt_notional=1e8,
                )
            ),
        ],
    )
    def test_float_leg_analytic_delta_with_npv(self, curve, obj) -> None:
        if type(obj) is FloatLegMtm:
            with pytest.warns(UserWarning):
                # Using 1.0 for FX, no `fx` or `fx_fixing` given to object
                result = 5 * obj.analytic_delta(curve, curve)
                before_npv = -obj.npv(curve, curve)
                obj.float_spread = 5
                after_npv = -obj.npv(curve, curve)
                expected = after_npv - before_npv
                assert abs(result - expected) < 1e-7
        else:
            result = 5 * obj.analytic_delta(curve, curve)
            before_npv = -obj.npv(curve, curve)
            obj.float_spread = 5
            after_npv = -obj.npv(curve, curve)
            expected = after_npv - before_npv
            assert abs(result - expected) < 1e-7

    def test_float_leg_analytic_delta(self, curve) -> None:
        float_leg = FloatLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            frequency="Q",
        )
        result = float_leg.analytic_delta(curve)
        assert abs(result - 41400.42965267) < 1e-7

    def test_float_leg_cashflows(self, curve) -> None:
        float_leg = FloatLeg(
            float_spread=NoInput(0),
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            frequency="Q",
        )
        result = float_leg.cashflows(curve)
        # test a couple of return elements
        assert abs(result.loc[0, defaults.headers["cashflow"]] + 6610305.76834) < 1e-4
        assert abs(result.loc[1, defaults.headers["df"]] - 0.98307) < 1e-4
        assert abs(result.loc[1, defaults.headers["notional"]] - 1e9) < 1e-7

    def test_float_leg_npv(self, curve) -> None:
        float_leg = FloatLeg(
            float_spread=NoInput(0),
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            frequency="Q",
        )
        result = float_leg.npv(curve)
        assert abs(result + 16710777.50089434) < 1e-7

    def test_float_leg_fixings(self, curve) -> None:
        float_leg = FloatLeg(dt(2022, 2, 1), "9M", "Q", payment_lag=0, fixings=[10, 20])
        assert float_leg.periods[0].fixings == 10
        assert float_leg.periods[1].fixings == 20
        assert float_leg.periods[2].fixings is NoInput(0)

    def test_float_leg_fixings_series(self, curve) -> None:
        fixings = Series(0.5, index=date_range(dt(2021, 11, 1), dt(2022, 2, 15)))
        float_leg = FloatLeg(dt(2021, 12, 1), "9M", "M", payment_lag=0, fixings=fixings)
        assert_series_equal(float_leg.periods[0].fixings, fixings)  # december fixings
        assert_series_equal(float_leg.periods[1].fixings, fixings)  # january fixings
        assert_series_equal(float_leg.periods[2].fixings, fixings)  # february fixings
        assert float_leg.periods[4].fixings is NoInput(0)  # no march fixings

    def test_float_leg_fixings_scalar(self, curve) -> None:
        float_leg = FloatLeg(dt(2022, 2, 1), "9M", "Q", payment_lag=0, fixings=5.0)
        assert float_leg.periods[0].fixings == 5.0
        assert float_leg.periods[1].fixings is NoInput(0)
        assert float_leg.periods[2].fixings is NoInput(0)

    @pytest.mark.parametrize(
        ("method", "param"),
        [
            ("rfr_payment_delay", NoInput(0)),
            ("rfr_lockout", 1),
            ("rfr_observation_shift", 0),
        ],
    )
    @pytest.mark.parametrize(
        "fixings",
        [
            [[1.19, 1.19, -8.81]],
            Series(
                [1.19, 1.19, -8.81],
                index=[dt(2022, 12, 28), dt(2022, 12, 29), dt(2022, 12, 30)],
            ),
        ],
    )
    def test_float_leg_rfr_fixings_table(self, method, param, fixings, curve) -> None:
        curve._set_ad_order(order=1)
        float_leg = FloatLeg(
            effective=dt(2022, 12, 28),
            termination="2M",
            frequency="M",
            fixings=fixings,
            currency="SEK",
            fixing_method=method,
            method_param=param,
            payment_lag=0,
        )
        float_leg.cashflows(curve)
        result = float_leg.fixings_table(curve)[dt(2022, 12, 28) : dt(2023, 1, 1)]
        expected = DataFrame(
            {
                "obs_dates": [
                    dt(2022, 12, 28),
                    dt(2022, 12, 29),
                    dt(2022, 12, 30),
                    dt(2022, 12, 31),
                    dt(2023, 1, 1),
                ],
                "notional": [
                    0.0,
                    0.0,
                    0.0,
                    -999821.37380,
                    -999932.84380,
                ],
                "risk": [0.0, 0.0, 0.0, -0.26664737, -0.26664737],
                "dcf": [0.0027777777777777778] * 5,
                "rates": [1.19, 1.19, -8.81, 4.01364, 4.01364],
            },
        ).set_index("obs_dates")
        expected.columns = MultiIndex.from_tuples(
            [(curve.id, "notional"), (curve.id, "risk"), (curve.id, "dcf"), (curve.id, "rates")]
        )
        assert_frame_equal(result, expected, rtol=1e-5)

    def test_rfr_with_fixings_fixings_table_issue(self) -> None:
        from rateslib import IRS

        instruments = [
            IRS(dt(2024, 1, 15), dt(2024, 3, 20), spec="eur_irs", curves="estr"),
            IRS(dt(2024, 3, 20), dt(2024, 6, 19), spec="eur_irs", curves="estr"),
            IRS(dt(2024, 6, 19), dt(2024, 9, 18), spec="eur_irs", curves="estr"),
        ]
        curve = Curve(
            nodes={
                dt(2024, 1, 11): 1.0,
                dt(2024, 3, 20): 1.0,
                dt(2024, 6, 19): 1.0,
                dt(2024, 9, 18): 1.0,
            },
            calendar="tgt",
            convention="act360",
            id="estr",
        )
        from rateslib import Solver

        Solver(
            curves=[curve],
            instruments=instruments,
            s=[
                3.89800324,
                3.63414284,
                3.16864932,
            ],
            id="eur",
        )
        fixings = Series(
            data=[
                3.904,
                3.904,
                3.904,
                3.905,
                3.902,
                3.904,
                3.906,
                3.882,
                3.9,
                3.9,
                3.899,
                3.899,
                3.901,
                3.901,
            ],
            index=[
                dt(2024, 1, 10),
                dt(2024, 1, 9),
                dt(2024, 1, 8),
                dt(2024, 1, 5),
                dt(2024, 1, 4),
                dt(2024, 1, 3),
                dt(2024, 1, 2),
                dt(2023, 12, 29),
                dt(2023, 12, 28),
                dt(2023, 12, 27),
                dt(2023, 12, 22),
                dt(2023, 12, 21),
                dt(2023, 12, 20),
                dt(2023, 12, 19),
            ],
        )

        swap = IRS(
            dt(2023, 12, 20),
            dt(2024, 1, 31),
            spec="eur_irs",
            curves="estr",
            leg2_fixings=fixings,
            notional=3e9,
            fixed_rate=3.922,
        )
        result = swap.leg2.fixings_table(curve)
        assert result.loc[dt(2024, 1, 10), (curve.id, "notional")] == 0.0
        assert abs(result.loc[dt(2024, 1, 11), (curve.id, "notional")] - 3006829846) < 1.0
        assert abs(result.loc[dt(2023, 12, 20), (curve.id, "rates")] - 3.901) < 0.001

    def test_float_leg_set_float_spread(self, curve) -> None:
        float_leg = FloatLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
        )
        assert float_leg.float_spread is NoInput(0)
        assert float_leg.periods[0].float_spread == 0

        float_leg.float_spread = 2.0
        assert float_leg.float_spread == 2.0
        assert float_leg.periods[0].float_spread == 2.0

    @pytest.mark.parametrize(
        ("method", "spread_method", "expected"),
        [
            ("ibor", NoInput(0), True),
            ("rfr_payment_delay", "none_simple", True),
            ("rfr_payment_delay", "isda_compounding", False),
            ("rfr_payment_delay", "isda_flat_compounding", False),
        ],
    )
    def test_is_linear(self, method, spread_method, expected) -> None:
        float_leg = FloatLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
            fixing_method=method,
            spread_compound_method=spread_method,
        )
        assert float_leg._is_linear is expected

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("ISDA_compounding", 357.7055853),
            ("ISDA_flat_compounding", 360.658902),
            ("NONE_Simple", 362.2342162),
        ],
    )
    def test_float_leg_spread_calculation(self, method, expected, curve) -> None:
        leg = FloatLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=0,
            notional=1e9,
            convention="Act360",
            frequency="Q",
            fixing_method="rfr_payment_delay",
            spread_compound_method=method,
            currency="nok",
            float_spread=0,
        )
        base_npv = leg.npv(curve, curve)
        result = leg._spread(-15000000, curve, curve)
        assert abs(result - expected) < 1e-3
        leg.float_spread = result
        assert abs(leg.npv(curve, curve) - base_npv + 15000000) < 2e2

    def test_fixing_method_raises(self) -> None:
        with pytest.raises(ValueError, match="`fixing_method`"):
            FloatLeg(dt(2022, 2, 1), "9M", "Q", fixing_method="bad")

    @pytest.mark.parametrize(
        ("eff", "term", "freq", "stub", "expected"),
        [
            (
                dt(2022, 1, 1),
                dt(2022, 6, 15),
                "Q",
                "ShortFront",
                [dt(2022, 1, 1), dt(2022, 3, 15), dt(2022, 6, 15)],
            ),
            (
                dt(2022, 1, 1),
                dt(2022, 6, 15),
                "Q",
                "ShortBack",
                [dt(2022, 1, 1), dt(2022, 4, 1), dt(2022, 6, 15)],
            ),
            (
                dt(2022, 1, 1),
                dt(2022, 9, 15),
                "Q",
                "LongFront",
                [dt(2022, 1, 1), dt(2022, 6, 15), dt(2022, 9, 15)],
            ),
            (
                dt(2022, 1, 1),
                dt(2022, 9, 15),
                "Q",
                "LongBack",
                [dt(2022, 1, 1), dt(2022, 4, 1), dt(2022, 9, 15)],
            ),
        ],
    )
    def test_leg_periods_unadj_dates(self, eff, term, freq, stub, expected) -> None:
        leg = FloatLeg(effective=eff, termination=term, frequency=freq, stub=stub)
        assert leg.schedule.uschedule == expected

    @pytest.mark.parametrize(
        ("eff", "term", "freq", "stub", "expected"),
        [
            (
                dt(2022, 1, 1),
                dt(2022, 6, 15),
                "Q",
                "ShortFront",
                [dt(2022, 1, 3), dt(2022, 3, 15), dt(2022, 6, 15)],
            ),
            (
                dt(2022, 1, 1),
                dt(2022, 6, 15),
                "Q",
                "ShortBack",
                [dt(2022, 1, 3), dt(2022, 4, 1), dt(2022, 6, 15)],
            ),
            (
                dt(2022, 1, 1),
                dt(2022, 9, 15),
                "Q",
                "LongFront",
                [dt(2022, 1, 3), dt(2022, 6, 15), dt(2022, 9, 15)],
            ),
            (
                dt(2022, 1, 1),
                dt(2022, 9, 15),
                "Q",
                "LongBack",
                [dt(2022, 1, 3), dt(2022, 4, 1), dt(2022, 9, 15)],
            ),
        ],
    )
    def test_leg_periods_adj_dates(self, eff, term, freq, stub, expected) -> None:
        leg = FloatLeg(effective=eff, termination=term, frequency=freq, stub=stub, calendar="bus")
        assert leg.schedule.aschedule == expected

    @pytest.mark.parametrize(
        ("eff", "term", "freq", "stub", "expected"),
        [
            (
                dt(2022, 1, 1),
                dt(2022, 6, 15),
                "Q",
                "ShortFront",
                [
                    FloatPeriod(
                        start=dt(2022, 1, 3),
                        end=dt(2022, 3, 15),
                        payment=dt(2022, 3, 17),
                        frequency="Q",
                        notional=defaults.notional,
                        convention=defaults.convention,
                        termination=dt(2022, 6, 15),
                    ),
                    FloatPeriod(
                        start=dt(2022, 3, 15),
                        end=dt(2022, 6, 15),
                        payment=dt(2022, 6, 17),
                        frequency="Q",
                        notional=defaults.notional,
                        convention=defaults.convention,
                        termination=dt(2022, 6, 15),
                    ),
                ],
            ),
        ],
    )
    def test_leg_periods_adj_dates2(self, eff, term, freq, stub, expected) -> None:
        leg = FloatLeg(
            effective=eff,
            termination=term,
            frequency=freq,
            stub=stub,
            payment_lag=2,
            calendar="bus",
        )
        for i in range(2):
            assert leg.periods[i].__str__() == expected[i].__str__()

    def test_spread_compound_method_raises(self) -> None:
        with pytest.raises(ValueError, match="`spread_compound_method`"):
            FloatLeg(dt(2022, 2, 1), "9M", "Q", spread_compound_method="bad")

    def test_float_leg_fixings_table_with_defined_fixings(self) -> None:
        swestr_curve = Curve({dt(2023, 1, 2): 1.0, dt(2023, 7, 2): 0.99}, calendar="stk")
        float_leg = FloatLeg(
            effective=dt(2022, 12, 28),
            termination="2M",
            frequency="M",
            fixings=[[1.19, 1.19, -8.81]],
            currency="SEK",
            calendar="stk",
        )
        result = float_leg.fixings_table(swestr_curve)[dt(2022, 12, 28) : dt(2023, 1, 4)]
        assert result.iloc[0, 0] == 0.0
        assert result.iloc[1, 0] == 0.0
        assert result.iloc[2, 0] == 0.0

    def test_float_leg_fixings_table_with_defined_fixings_approximate(self) -> None:
        swestr_curve = Curve({dt(2023, 1, 2): 1.0, dt(2023, 7, 2): 0.99}, calendar="stk")
        float_leg = FloatLeg(
            effective=dt(2022, 12, 28),
            termination="2M",
            frequency="M",
            fixings=[[1.19, 1.19, -8.81]],
            currency="SEK",
            calendar="stk",
        )
        with pytest.warns(UserWarning):
            result = float_leg.fixings_table(swestr_curve, approximate=True)
        assert result.iloc[0, 0] == 0.0
        assert result.iloc[1, 0] == 0.0
        assert result.iloc[2, 0] == 0.0

    def test_leg_fixings_as_2_tuple(self) -> None:
        ser = Series([2.0, 3.0], index=[dt(2022, 6, 2), dt(2022, 7, 4)])
        float_leg = FloatLeg(
            effective=dt(2022, 5, 2),
            termination="4M",
            frequency="M",
            fixings=(1.5, ser),
            currency="SEK",
            calendar="stk",
            fixing_method="ibor",
            method_param=0,
        )
        assert float_leg.periods[0].fixings == 1.5
        assert id(float_leg.periods[1].fixings) == id(ser)
        assert id(float_leg.periods[2].fixings) == id(ser)
        assert float_leg.periods[3].fixings == NoInput.blank


class TestZeroFloatLeg:
    def test_zero_float_leg_set_float_spread(self, curve) -> None:
        float_leg = ZeroFloatLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
        )
        assert float_leg.float_spread is NoInput(0)
        assert float_leg.periods[0].float_spread == 0

        float_leg.float_spread = 2.0
        assert float_leg.float_spread == 2.0
        assert float_leg.periods[0].float_spread == 2.0

    def test_zero_float_leg_dcf(self) -> None:
        ftl = ZeroFloatLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
        )
        result = ftl.dcf
        expected = ftl.periods[0].dcf + ftl.periods[1].dcf
        assert result == expected

    def test_zero_float_leg_rate(self, curve) -> None:
        ftl = ZeroFloatLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
            float_spread=500,
        )
        result = ftl.rate(curve)
        expected = 1 + ftl.periods[0].dcf * ftl.periods[0].rate(curve) / 100
        expected *= 1 + ftl.periods[1].dcf * ftl.periods[1].rate(curve) / 100
        expected = (expected - 1) / ftl.dcf * 100
        assert result == expected

    def test_zero_float_leg_cashflows(self, curve) -> None:
        ftl = ZeroFloatLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
            float_spread=500,
        )
        result = ftl.cashflows(curve)
        expected = DataFrame(
            {
                "Type": ["ZeroFloatLeg"],
                "Acc Start": [dt(2022, 1, 1)],
                "Acc End": [dt(2022, 6, 1)],
                "DCF": [0.419444444444444],
                "Spread": [500.0],
            },
        )
        assert_frame_equal(result[["Type", "Acc Start", "Acc End", "DCF", "Spread"]], expected)

    def test_zero_float_leg_npv(self, curve) -> None:
        ftl = ZeroFloatLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
        )
        result = ftl.npv(curve)
        expected = 16710778.891147703
        assert abs(result - expected) < 1e-2
        result2 = ftl.npv(curve, local=True)
        assert abs(result2["usd"] - expected) < 1e-2

    def test_cashflows_none(self) -> None:
        ftl = ZeroFloatLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
        )
        result = ftl.cashflows()
        assert result.iloc[0].to_dict()[defaults.headers["npv"]] is None
        assert result.iloc[0].to_dict()[defaults.headers["npv_fx"]] is None

    def test_amortization_raises(self) -> None:
        with pytest.raises(ValueError, match="`ZeroFloatLeg` cannot be defined with `amortizati."):
            ZeroFloatLeg(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                notional=-1e9,
                convention="Act360",
                frequency="Q",
                amortization=1.0,
            )

    def test_rfr_fixings_table(self, curve) -> None:
        zfl = ZeroFloatLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 10, 1),
            payment_lag=0,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
        )
        # fl = FloatLeg(
        #     effective=dt(2022, 1, 1),
        #     termination=dt(2022, 10, 1),
        #     payment_lag=0,
        #     notional=-1e9,
        #     convention="Act360",
        #     frequency="Q",
        # )
        result = zfl.fixings_table(curve)
        # compare = fl.fixings_table(curve)
        for i in range(len(result.index)):
            # consistent risk throught the compounded leg
            assert abs(result.iloc[i, 1] - 277.75) < 1e-1

    def test_ibor_fixings_table(self, curve) -> None:
        zfl = ZeroFloatLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 10, 1),
            payment_lag=0,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
            fixing_method="ibor",
            method_param=0,
        )
        result = zfl.fixings_table(curve)
        assert abs(result.iloc[0, 0] - 1e9) < 1e-3
        assert abs(result.iloc[1, 0] - 1010101010.10) < 1e-2
        assert abs(result.iloc[2, 0] - 1020408163.26) < 1e-2

    def test_ibor_stub_fixings_table(self, curve) -> None:
        curve2 = curve.copy()
        curve2._id = "3mIBOR"
        curve._id = "1mIBOR"
        zfl = ZeroFloatLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 9, 1),
            payment_lag=0,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
            fixing_method="ibor",
            method_param=0,
        )
        result = zfl.fixings_table({"1m": curve, "3m": curve2}, disc_curve=curve)
        assert abs(result.iloc[0, 0] - 996878112.103) < 1e-2
        assert abs(result.iloc[0, 4] - 312189976.385) < 1e-2
        assert isna(result.iloc[1, 0])
        assert abs(result.iloc[2, 5] - 25294.7235) < 1e-3

    @pytest.mark.parametrize(
        "fixings", [[2.0, 2.5], Series([2.0, 2.5], index=[dt(2021, 7, 1), dt(2021, 10, 1)])]
    )
    def test_ibor_fixings_table_after_known_fixings(self, curve, fixings) -> None:
        curve2 = curve.copy()
        curve2._id = "3mIBOR"
        curve._id = "1mIBOR"
        zfl = ZeroFloatLeg(
            effective=dt(2021, 7, 1),
            termination=dt(2022, 9, 1),
            payment_lag=0,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
            fixing_method="ibor",
            method_param=0,
            stub="shortBack",
            fixings=fixings,
        )
        result = zfl.fixings_table({"1m": curve, "3m": curve2}, disc_curve=curve)
        assert abs(result.iloc[0, 0] - 0) < 1e-2
        assert abs(result.iloc[1, 0] - 0) < 1e-2
        assert isna(result.iloc[0, 4])
        assert abs(result.iloc[4, 0] - 354684373.65) < 1e-2
        assert abs(result.iloc[4, 5] - 8508.6111) < 1e-3

    def test_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="`frequency` for a ZeroFloatLeg should not be 'Z'"):
            ZeroFloatLeg(
                effective=dt(2022, 1, 1),
                termination="5y",
                payment_lag=0,
                notional=-1e8,
                convention="ActAct",
                frequency="Z",
            )

    def test_zero_float_leg_analytic_delta(self, curve) -> None:
        zfl = ZeroFloatLeg(
            effective=dt(2022, 1, 1),
            termination="5y",
            payment_lag=0,
            notional=-1e8,
            convention="ActAct",
            frequency="A",
            float_spread=1.0,
        )
        result = zfl.analytic_delta(curve)
        expected = -47914.3660

        assert abs(result - expected) < 1e-3


class TestZeroFixedLeg:
    @pytest.mark.parametrize(
        ("freq", "cash", "rate"),
        [
            ("A", 13140821.29, 2.50),
            ("S", 13227083.80, 2.50),
            ("A", None, NoInput(0)),
        ],
    )
    def test_zero_fixed_leg_cashflows(self, freq, cash, rate, curve) -> None:
        zfl = ZeroFixedLeg(
            effective=dt(2022, 1, 1),
            termination="5y",
            payment_lag=0,
            notional=-1e8,
            convention="ActAct",
            frequency=freq,
            fixed_rate=rate,
        )
        result = zfl.cashflows(curve)
        expected = DataFrame(
            {
                "Type": ["ZeroFixedLeg"],
                "Acc Start": [dt(2022, 1, 1)],
                "Acc End": [dt(2027, 1, 1)],
                "DCF": [5.0],
                "Rate": [rate],
                "Cashflow": [cash],
            },
        )
        assert_frame_equal(
            result[["Type", "Acc Start", "Acc End", "DCF", "Rate", "Cashflow"]],
            expected,
            rtol=1e-3,
        )

    def test_zero_fixed_leg_cashflows_cal(self, curve) -> None:
        # assert stated cashflows accrual dates are adjusted according to calendar
        # GH561/562
        zfl = ZeroFixedLeg(
            effective=dt(2024, 12, 15),
            termination="5y",
            payment_lag=0,
            notional=-1e8,
            calendar="tgt",
            modifier="mf",
            convention="ActAct",
            frequency="A",
            fixed_rate=2.0,
        )
        result = zfl.cashflows(curve)
        expected = DataFrame(
            {
                "Type": ["ZeroFixedLeg"],
                "Acc Start": [dt(2024, 12, 16)],
                "Acc End": [dt(2029, 12, 17)],
                "DCF": [5.0],
                "Rate": [2.0],
            },
        )
        assert_frame_equal(
            result[["Type", "Acc Start", "Acc End", "DCF", "Rate"]],
            expected,
            rtol=1e-3,
        )

    def test_zero_fixed_leg_npv(self, curve) -> None:
        zfl = ZeroFixedLeg(
            effective=dt(2022, 1, 1),
            termination="5y",
            payment_lag=0,
            notional=-1e8,
            convention="ActAct",
            frequency="A",
            fixed_rate=2.5,
        )
        result = zfl.npv(curve)
        expected = 13140821.29 * curve[dt(2027, 1, 1)]
        assert abs(result - expected) < 1e-2
        result2 = zfl.npv(curve, local=True)
        assert abs(result2["usd"] - expected) < 1e-2

    def test_zero_fixed_leg_analytic_delta(self, curve) -> None:
        zfl = ZeroFixedLeg(
            effective=dt(2022, 1, 1),
            termination="5y",
            payment_lag=0,
            notional=-1e8,
            convention="ActAct",
            frequency="A",
            fixed_rate=2.5,
        )
        result1 = zfl._analytic_delta(curve)
        result2 = zfl.analytic_delta(curve)
        assert abs(result1 + 40789.7007) < 1e-3
        assert abs(result2 + 45024.1974) < 1e-3

    def test_zero_fixed_spread(self, curve) -> None:
        zfl = ZeroFixedLeg(
            effective=dt(2022, 1, 1),
            termination="5y",
            payment_lag=0,
            notional=-1e8,
            convention="ActAct",
            frequency="A",
            fixed_rate=NoInput(0),
        )
        result = zfl._spread(13140821.29 * curve[dt(2027, 1, 1)], NoInput(0), curve)
        assert (result / 100 - 2.50) < 1e-3

    def test_amortization_raises(self) -> None:
        with pytest.raises(ValueError, match="`ZeroFixedLeg` cannot be defined with `amortizatio"):
            ZeroFixedLeg(
                effective=dt(2022, 1, 1),
                termination="5y",
                payment_lag=0,
                notional=-1e8,
                convention="ActAct",
                frequency="A",
                fixed_rate=NoInput(0),
                amortization=1.0,
            )

    def test_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="`frequency` for a ZeroFixedLeg should not be 'Z'"):
            ZeroFixedLeg(
                effective=dt(2022, 1, 1),
                termination="5y",
                payment_lag=0,
                notional=-1e8,
                convention="ActAct",
                frequency="Z",
                fixed_rate=NoInput(0),
            )

    def test_analytic_delta_no_fixed_rate(self, curve) -> None:
        zfl = ZeroFixedLeg(
            effective=dt(2022, 1, 1),
            termination="5y",
            payment_lag=0,
            notional=-1e8,
            convention="ActAct",
            frequency="A",
            fixed_rate=NoInput(0),
        )
        with pytest.raises(ValueError, match="Must have `fixed_rate` on ZeroFixedLeg for analy"):
            zfl.analytic_delta(curve)


class TestZeroIndexLeg:
    @pytest.mark.parametrize(
        ("index_base", "index_fixings", "meth", "exp"),
        [
            (NoInput(0), NoInput(0), "monthly", -61855.670),
            (NoInput(0), NoInput(0), "daily", -61782.379),
            (100.0, NoInput(0), "monthly", -61855.670),
            (NoInput(0), 110.0, "monthly", -100000.0),
            (NoInput(0), 110.0, "daily", -98696.645),
            (100.0, 110.0, "monthly", -100000.0),
            (100.0, 110.0, "daily", -100000.0),
        ],
    )
    def test_zero_index_cashflow(self, index_base, index_fixings, meth, exp) -> None:
        index_curve = Curve(
            {
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.97,
            },
            index_base=100.0,
            index_lag=3,
            interpolation="linear_index",
        )
        zil = ZeroIndexLeg(
            effective=dt(2022, 1, 15),
            termination="2Y",
            frequency="A",
            convention="1+",
            index_base=index_base,
            index_fixings=index_fixings,
            index_method=meth,
        )
        result = zil.cashflow(index_curve)
        assert abs(result - exp) < 1e-3

    def test_set_index_leg_after_init(self) -> None:
        leg = ZeroIndexLeg(
            effective=dt(2022, 3, 15),
            termination="9M",
            frequency="Q",
            convention="1+",
            payment_lag=0,
            notional=40e6,
            index_base=None,
        )
        for period in leg.periods[:1]:
            assert period.index_base is None
        leg.index_base = 205.0
        for period in leg.periods[:1]:
            assert period.index_base == 205.0

    def test_zero_analytic_delta(self) -> None:
        zil = ZeroIndexLeg(
            effective=dt(2022, 1, 15),
            termination="2Y",
            frequency="A",
            convention="1+",
        )
        assert zil.analytic_delta() == 0.0

    def test_cashflows(self) -> None:
        index_curve = Curve(
            {
                dt(2022, 1, 1): 1.0,
                dt(2023, 1, 1): 0.97,
            },
            index_base=100.0,
            index_lag=3,
            interpolation="linear_index",
        )
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.97})
        zil = ZeroIndexLeg(
            effective=dt(2022, 1, 15),
            termination="2Y",
            frequency="A",
            convention="1+",
        )
        result = zil.cashflows(index_curve, curve)
        expected = DataFrame(
            {
                "Type": ["ZeroIndexLeg"],
                "Notional": [1000000.0],
                "Real Cashflow": [-1000000.0],
                "Index Base": [100.11863],
                "Index Ratio": [1.06178],
                "Cashflow": [-61782.379],
                "NPV": [-58053.47605],
            },
        )
        assert_frame_equal(
            result[
                [
                    "Type",
                    "Notional",
                    "Real Cashflow",
                    "Index Base",
                    "Index Ratio",
                    "Cashflow",
                    "NPV",
                ]
            ],
            expected,
            rtol=1e-3,
        )


class TestFloatLegExchange:
    def test_float_leg_exchange_notional_setter(self) -> None:
        float_leg_exc = FloatLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
            initial_exchange=True,
            final_exchange=True,
        )
        float_leg_exc.notional = 200
        assert float_leg_exc.notional == 200

    def test_float_leg_exchange_amortization_setter(self) -> None:
        float_leg_exc = FloatLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 10, 1),
            payment_lag=2,
            notional=-1000,
            convention="Act360",
            frequency="Q",
            initial_exchange=True,
            final_exchange=True,
        )
        float_leg_exc.amortization = -200

        cashflows = [2, 4, 6]
        cash_notionals = [None, -200, None, -200, None, -600]
        fixed_notionals = [None, -1000, None, -800, None, -600]
        for i in cashflows:
            assert isinstance(float_leg_exc.periods[i], Cashflow)
            assert float_leg_exc.periods[i].notional == cash_notionals[i - 1]

            assert isinstance(float_leg_exc.periods[i - 1], FloatPeriod)
            assert float_leg_exc.periods[i - 1].notional == fixed_notionals[i - 1]

    def test_float_leg_exchange_set_float_spread(self) -> None:
        float_leg_exc = FloatLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 10, 1),
            payment_lag=2,
            notional=-1000,
            convention="Act360",
            frequency="Q",
            initial_exchange=True,
            final_exchange=True,
        )
        assert float_leg_exc.float_spread is NoInput(0)
        float_leg_exc.float_spread = 2.0
        assert float_leg_exc.float_spread == 2.0
        for period in float_leg_exc.periods:
            if isinstance(period, FloatPeriod):
                period.float_spread == 2.0

    def test_float_leg_exchange_amortization(self, curve) -> None:
        leg = FloatLeg(
            dt(2022, 1, 1),
            dt(2023, 1, 1),
            "Q",
            notional=5e6,
            amortization=1e6,
            payment_lag=0,
            initial_exchange=True,
            final_exchange=True,
        )
        assert len(leg.periods) == 9
        for i in [0, 2, 4, 6, 8]:
            assert type(leg.periods[i]) is Cashflow
        for i in [1, 3, 5, 7]:
            assert type(leg.periods[i]) is FloatPeriod
        assert leg.periods[1].notional == 5e6
        assert leg.periods[7].notional == 2e6
        assert leg.periods[8].notional == 2e6
        assert abs(leg.npv(curve).real) < 1e-9

    def test_float_leg_exchange_npv(self, curve) -> None:
        fle = FloatLeg(
            dt(2022, 2, 1),
            "6M",
            "Q",
            payment_lag=0,
            initial_exchange=True,
            final_exchange=True,
        )
        result = fle.npv(curve)
        assert abs(result) < 1e-9

    def test_float_leg_exchange_fixings_table(self, curve) -> None:
        fle = FloatLeg(
            dt(2022, 2, 1),
            "6M",
            "Q",
            payment_lag=0,
            initial_exchange=True,
            final_exchange=True,
        )
        result = fle.fixings_table(curve)
        expected = DataFrame(
            {
                "notional": [-1009872.33778, -1000000.00000],
                "risk": [-0.2767869527597316, -0.27405055522733884],
                "dcf": [0.002777777777777778, 0.002777777777777778],
                "rates": [4.01655, 4.01655],
            },
            index=Index([dt(2022, 4, 30), dt(2022, 5, 1)], name="obs_dates"),
        )
        expected.columns = MultiIndex.from_tuples(
            [(curve.id, "notional"), (curve.id, "risk"), (curve.id, "dcf"), (curve.id, "rates")]
        )
        assert_frame_equal(result[dt(2022, 4, 30) : dt(2022, 5, 1)], expected)


class TestFixedLeg:
    def test_fixed_leg_analytic_delta(self, curve) -> None:
        fixed_leg = FixedLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            frequency="Q",
        )
        result = fixed_leg.analytic_delta(curve)
        assert abs(result - 41400.42965267) < 1e-7

    def test_fixed_leg_npv(self, curve) -> None:
        fixed_leg = FixedLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            frequency="Q",
            fixed_rate=4.00,
        )
        result = fixed_leg.npv(curve)
        assert abs(result + 400 * fixed_leg.analytic_delta(curve)) < 1e-7

    def test_fixed_leg_cashflows(self, curve) -> None:
        fixed_leg = FixedLeg(
            fixed_rate=4.00,
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
        )
        result = fixed_leg.cashflows(curve)
        # test a couple of return elements
        assert abs(result.loc[0, defaults.headers["cashflow"]] - 6555555.55555) < 1e-4
        assert abs(result.loc[1, defaults.headers["df"]] - 0.98307) < 1e-4
        assert abs(result.loc[1, defaults.headers["notional"]] + 1e9) < 1e-7

    def test_fixed_leg_set_fixed(self, curve) -> None:
        fixed_leg = FixedLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
        )
        assert fixed_leg.fixed_rate is NoInput(0)
        assert fixed_leg.periods[0].fixed_rate is NoInput(0)

        fixed_leg.fixed_rate = 2.0
        assert fixed_leg.fixed_rate == 2.0
        assert fixed_leg.periods[0].fixed_rate == 2.0


class TestCreditPremiumLeg:
    @pytest.mark.parametrize(
        ("premium_accrued", "exp"), [(True, 41357.455568685626), (False, 41330.94188109829)]
    )
    def test_premium_leg_analytic_delta(self, hazard_curve, curve, premium_accrued, exp) -> None:
        leg = CreditPremiumLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            frequency="Q",
            premium_accrued=premium_accrued,
        )
        result = leg.analytic_delta(hazard_curve, curve)
        assert abs(result - exp) < 1e-7

    @pytest.mark.parametrize(("premium_accrued"), [True, False])
    def test_premium_leg_npv(self, hazard_curve, curve, premium_accrued) -> None:
        leg = CreditPremiumLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=1e9,
            convention="Act360",
            frequency="Q",
            premium_accrued=premium_accrued,
            fixed_rate=4.00,
        )
        result = leg.npv(hazard_curve, curve)
        assert abs(result + 400 * leg.analytic_delta(hazard_curve, curve)) < 1e-7

    def test_premium_leg_cashflows(self, hazard_curve, curve) -> None:
        leg = CreditPremiumLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
            fixed_rate=4.00,
        )
        result = leg.cashflows(hazard_curve, curve)
        # test a couple of return elements
        assert abs(result.loc[0, defaults.headers["cashflow"]] - 6555555.55555) < 1e-4
        assert abs(result.loc[1, defaults.headers["df"]] - 0.98307) < 1e-4
        assert abs(result.loc[1, defaults.headers["notional"]] + 1e9) < 1e-7

    def test_premium_leg_set_fixed_rate(self, curve) -> None:
        leg = CreditPremiumLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
        )
        assert leg.fixed_rate is NoInput(0)
        assert leg.periods[0].fixed_rate is NoInput(0)

        leg.fixed_rate = 2.0
        assert leg.fixed_rate == 2.0
        assert leg.periods[0].fixed_rate == 2.0

    @pytest.mark.parametrize(
        ("date", "exp"),
        [
            (dt(2022, 2, 1), 1e9 * 0.02 * 0.25 * 31 / 90),
            (dt(2022, 3, 1), 0.0),
            (dt(2022, 6, 1), 0.0),
        ],
    )
    def test_premium_leg_accrued(self, date, exp):
        leg = CreditPremiumLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=-1e9,
            convention="ActActICMA",
            frequency="Q",
            fixed_rate=2.0,
        )
        result = leg.accrued(date)
        assert abs(result - exp) < 1e-6

    @pytest.mark.parametrize("final", [True, False])
    def test_exchanges_raises(self, final):
        with pytest.raises(ValueError, match="`initial_exchange` and `final_exchange` cannot be"):
            CreditPremiumLeg(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                notional=-1e9,
                convention="ActActICMA",
                frequency="Q",
                fixed_rate=2.0,
                initial_exchange=final,
                final_exchange=not final,
            )


class TestCreditProtectionLeg:
    def test_leg_analytic_delta(self, hazard_curve, curve) -> None:
        leg = CreditProtectionLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=1e9,
            frequency="Q",
        )
        result = leg.analytic_delta(hazard_curve, curve)
        assert abs(result) < 1e-7

    def test_leg_analytic_rec_risk(self, hazard_curve, curve) -> None:
        leg = CreditProtectionLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2027, 1, 1),
            payment_lag=2,
            notional=1e7,
            frequency="Q",
        )
        result = leg.analytic_rec_risk(hazard_curve, curve)

        pv0 = leg.npv(hazard_curve, curve)
        hazard_curve.update_meta("credit_recovery_rate", 0.41)
        pv1 = leg.npv(hazard_curve, curve)
        expected = pv1 - pv0
        assert abs(result - expected) < 1e-7

    @pytest.mark.parametrize(("premium_accrued"), [True, False])
    def test_leg_npv(self, hazard_curve, curve, premium_accrued) -> None:
        leg = CreditProtectionLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=1e9,
            frequency="Z",
        )
        result = leg.npv(hazard_curve, curve)
        expected = -1390922.0390295777  # with 1 cds_discretization this is -1390906.242843
        assert abs(result - expected) < 1e-7

    def test_leg_cashflows(self, hazard_curve, curve) -> None:
        leg = CreditProtectionLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            notional=-1e9,
            convention="Act360",
            frequency="Q",
        )
        result = leg.cashflows(hazard_curve, curve)
        # test a couple of return elements
        assert abs(result.loc[0, defaults.headers["cashflow"]] - 600e6) < 1e-4
        assert abs(result.loc[1, defaults.headers["df"]] - 0.98307) < 1e-4
        assert abs(result.loc[1, defaults.headers["notional"]] + 1e9) < 1e-7

    def test_leg_zero_sched(self):
        leg = CreditProtectionLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2024, 6, 1),
            notional=-1e9,
            convention="Act360",
            frequency="Z",
        )
        assert len(leg.periods) == 1
        assert leg.periods[0].end == dt(2024, 6, 1)


class TestIndexFixedLegExchange:
    @pytest.mark.parametrize(
        "i_fixings",
        [
            NoInput(0),
            # [210, 220, 230], # list not supported in v2.0
            210,
            Series(
                [210.0, 220.0, 230.0],
                index=[dt(2022, 6, 15), dt(2022, 9, 15), dt(2022, 12, 15)],
            ),
        ],
    )
    def test_idx_leg_cashflows(self, i_fixings) -> None:
        leg = IndexFixedLeg(
            effective=dt(2022, 3, 15),
            termination="9M",
            frequency="Q",
            convention="ActActICMA",
            payment_lag=0,
            notional=40e6,
            fixed_rate=5.0,
            index_base=200.0,
            index_lag=0,
            index_fixings=i_fixings,
            initial_exchange=False,
            final_exchange=True,
            index_method="curve",
        )
        index_curve = Curve(
            nodes={
                dt(2022, 3, 15): 1.0,
                dt(2022, 6, 15): 1.0 / 1.05,
                dt(2022, 9, 15): 1.0 / 1.10,
                dt(2022, 12, 15): 1.0 / 1.15,
            },
            index_base=200.0,
            interpolation="linear_index",
            index_lag=0,
        )
        disc_curve = Curve({dt(2022, 3, 15): 1.0, dt(2022, 12, 15): 1.0})
        flows = leg.cashflows(curve=index_curve, disc_curve=disc_curve)

        def equals_with_tol(a, b):
            if isinstance(a, str):
                return a == b
            else:
                return abs(a - b) < 1e-7

        expected = {
            "Type": "IndexFixedPeriod",
            "DCF": 0.250,
            "Notional": 40e6,
            "Rate": 5.0,
            "Real Cashflow": -500e3,
            "Index Val": 210.0,
            "Index Ratio": 1.05,
            "Cashflow": -525000,
        }
        flow = flows.iloc[0].to_dict()
        for key in set(expected.keys()) & set(flow.keys()):
            assert equals_with_tol(expected[key], flow[key])

        final_flow = flows.iloc[3].to_dict()
        expected = {
            "Type": "IndexCashflow",
            "Notional": 40e6,
            "Real Cashflow": -40e6,
            "Index Val": 230.0,
            "Index Ratio": 1.15,
            "Cashflow": -46e6,
        }
        for key in set(expected.keys()) & set(final_flow.keys()):
            assert equals_with_tol(expected[key], final_flow[key])

    def test_args_raises(self) -> None:
        with pytest.raises(ValueError, match="`index_method` must be in"):
            IndexFixedLeg(
                effective=dt(2022, 3, 15),
                termination="9M",
                frequency="Q",
                index_base=200.0,
                index_method="BAD",
                initial_exchange=True,
                final_exchange=True,
            )

    def test_set_index_leg_after_init(self) -> None:
        leg = IndexFixedLeg(
            effective=dt(2022, 3, 15),
            termination="9M",
            frequency="Q",
            convention="ActActICMA",
            payment_lag=0,
            notional=40e6,
            fixed_rate=5.0,
            index_base=None,
            initial_exchange=False,
            final_exchange=True,
        )
        for period in leg.periods:
            assert period.index_base is None
        leg.index_base = 205.0
        for period in leg.periods:
            assert period.index_base == 205.0

    def test_npv(self) -> None:
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98})
        index_curve = Curve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_base=100.0,
            interpolation="linear_index",
            index_lag=3,
        )
        index_leg_exch = IndexFixedLeg(
            dt(2022, 1, 1),
            "9M",
            "Q",
            notional=1000000,
            amortization=200000,
            index_base=100.0,
            initial_exchange=False,
            fixed_rate=1.0,
            final_exchange=True,
            index_lag=3,
        )
        result = index_leg_exch.npv(index_curve, curve)
        expected = -999971.65702
        assert abs(result - expected) < 1e-4

    def test_index_lag_on_periods(self):
        index_leg_exch = IndexFixedLeg(
            dt(2022, 1, 1),
            "6M",
            "Q",
            notional=1000000,
            amortization=200000,
            index_base=100.0,
            fixed_rate=1.0,
            final_exchange=True,
            index_lag=4,
        )
        for period in index_leg_exch.periods:
            assert period.index_lag == 4


class TestIndexFixedLeg:
    @pytest.mark.parametrize(
        ("i_fixings", "meth"),
        [
            (NoInput(0), "daily"),
            # ([210, 220, 230], "daily"),  # list unsupported in v2.0
            (210, "daily"),
            (
                Series(
                    [210.0, 210, 220, 220, 230, 230],
                    index=[
                        dt(2022, 6, 1),
                        dt(2022, 7, 1),
                        dt(2022, 9, 1),
                        dt(2022, 10, 1),
                        dt(2022, 12, 1),
                        dt(2023, 1, 1),
                    ],
                ),
                "daily",
            ),
            (
                Series(
                    [210.0, 220, 230],
                    index=[dt(2022, 6, 1), dt(2022, 9, 1), dt(2022, 12, 1)],
                ),
                "monthly",
            ),
        ],
    )
    def test_idx_leg_cashflows(self, i_fixings, meth) -> None:
        leg = IndexFixedLeg(
            effective=dt(2022, 3, 15),
            termination="9M",
            frequency="Q",
            convention="ActActICMA",
            payment_lag=0,
            notional=40e6,
            fixed_rate=5.0,
            index_base=200.0,
            index_fixings=i_fixings,
            index_method=meth,
            index_lag=0,
        )
        index_curve = Curve(
            nodes={
                dt(2022, 3, 15): 1.0,
                dt(2022, 6, 15): 1.0 / 1.05,
                dt(2022, 9, 15): 1.0 / 1.10,
                dt(2022, 12, 15): 1.0 / 1.15,
            },
            index_base=200.0,
            interpolation="linear_index",
            index_lag=0,
        )
        disc_curve = Curve({dt(2022, 3, 15): 1.0, dt(2022, 12, 15): 1.0})
        flows = leg.cashflows(curve=index_curve, disc_curve=disc_curve)

        def equals_with_tol(a, b):
            if isinstance(a, str):
                return a == b
            else:
                return abs(a - b) < 1e-7

        expected = {
            "Type": "IndexFixedPeriod",
            "DCF": 0.250,
            "Notional": 40e6,
            "Rate": 5.0,
            "Real Cashflow": -500e3,
            "Index Val": 210.0,
            "Index Ratio": 1.05,
            "Cashflow": -525000,
        }
        flow = flows.iloc[0].to_dict()
        for key in set(expected.keys()) & set(flow.keys()):
            assert equals_with_tol(expected[key], flow[key])

    @pytest.mark.parametrize(("meth", "exp"), [("daily", 230.0), ("monthly", 227.91208)])
    def test_missing_fixings(self, meth, exp) -> None:
        i_fixings = Series(
            [210.0, 210, 220, 220],
            index=[dt(2022, 6, 1), dt(2022, 7, 1), dt(2022, 9, 1), dt(2022, 10, 1)],
        )
        leg = IndexFixedLeg(
            effective=dt(2022, 3, 20),
            termination="9M",
            frequency="Q",
            convention="ActActICMA",
            payment_lag=0,
            notional=40e6,
            fixed_rate=5.0,
            index_base=200.0,
            index_fixings=i_fixings,
            index_method=meth,
            index_lag=0,
        )
        index_curve = Curve(
            nodes={
                dt(2022, 3, 20): 1.0,
                dt(2022, 6, 20): 1.0 / 1.05,
                dt(2022, 9, 20): 1.0 / 1.10,
                dt(2022, 12, 20): 1.0 / 1.15,
            },
            index_base=200.0,
            interpolation="linear_index",
            index_lag=0,
        )
        cashflows = leg.cashflows(index_curve)
        result = cashflows.iloc[2]["Index Val"]
        assert abs(result - exp) < 1e-3

    def test_set_index_leg_after_init(self) -> None:
        leg = IndexFixedLeg(
            effective=dt(2022, 3, 15),
            termination="9M",
            frequency="Q",
            convention="ActActICMA",
            payment_lag=0,
            notional=40e6,
            fixed_rate=5.0,
            index_base=None,
        )
        for period in leg.periods:
            assert period.index_base is None
        leg.index_base = 205.0
        for period in leg.periods:
            assert period.index_base == 205.0

    @pytest.mark.parametrize(
        "i_base",
        [
            200.0,
            Series([199.0, 201.0], index=[dt(2022, 4, 1), dt(2022, 5, 1)]),
        ],
    )
    def test_set_index_base(self, curve, i_base) -> None:
        leg = IndexFixedLeg(
            effective=dt(2022, 4, 16),
            termination=dt(2022, 5, 1),
            payment_lag=2,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
            index_base=None,
            index_method="daily",
            index_lag=0,
        )
        assert leg.index_base is None
        assert leg.periods[0].index_base is None

        leg.index_base = i_base
        assert leg.index_base == 200.0
        assert leg.periods[0].index_base == 200.0

    @pytest.mark.parametrize(
        ("i_base", "exp"),
        [
            (Series([199.0, 200.0], index=[dt(2021, 12, 31), dt(2022, 1, 1)]), 200.0),
            (Series([1.0, 2.0], index=[dt(2000, 1, 1), dt(2000, 12, 1)]), NoInput(0)),
            (NoInput(0), NoInput(0)),
            (110.0, 110.0),
        ],
    )
    def test_initialise_index_base(self, i_base, exp) -> None:
        leg = IndexFixedLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 6, 1),
            payment_lag=2,
            notional=-1e9,
            convention="Act360",
            frequency="Q",
            index_base=i_base,
            index_lag=0,
        )
        assert leg.index_base == exp

    # this test is for coverage. When implemented this is OK to remove.
    def test_initial_exchange_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="Cannot construct `IndexFixedL"):
            IndexFixedLeg(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 6, 1),
                payment_lag=2,
                notional=-1e9,
                convention="Act360",
                frequency="Q",
                index_base=None,
                initial_exchange=True,
            )

    @pytest.mark.skip(reason="fixings as list removed in v2.0")
    def test_index_fixings_as_list(self) -> None:
        leg = IndexFixedLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 10, 1),
            payment_lag=2,
            convention="Act360",
            frequency="Q",
            notional=1e6,
            amortization=250e3,
            index_base=NoInput(0),
            index_fixings=[100.0, 200.0],
        )
        assert leg.periods[0].index_fixings == 100.0
        assert leg.periods[1].index_fixings == 200.0
        assert leg.periods[2].index_fixings == NoInput(0)

    @pytest.mark.skip(reason="fixings as list removed in v2.0")
    def test_index_fixings_as_list_final_exchange(self) -> None:
        leg = IndexFixedLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 10, 1),
            payment_lag=2,
            convention="Act360",
            frequency="Q",
            notional=1e6,
            amortization=250e3,
            index_base=NoInput(0),
            index_fixings=[100.0, 100.0, 200.0, 199.0],
            final_exchange=True,
        )
        assert leg.periods[0].index_fixings == 100.0
        assert leg.periods[1].index_fixings == 100.0
        assert leg.periods[2].index_fixings == 200.0
        assert leg.periods[3].index_fixings == 199.0
        assert leg.periods[4].index_fixings == NoInput(0)
        assert leg.periods[5].index_fixings == NoInput(0)

    @pytest.mark.parametrize(
        "index_fixings",
        [
            Series([1, 2, 3], index=[dt(2000, 1, 1), dt(1999, 1, 1), dt(2001, 1, 1)]),
            Series([1, 2, 3], index=[dt(2000, 1, 1), dt(2000, 1, 1), dt(2001, 1, 1)]),
        ],
    )
    def test_index_as_series_invalid(self, index_fixings):
        with pytest.raises(ValueError, match="`index_fixings` as Series must be"):
            IndexFixedLeg(
                effective=dt(2022, 1, 1),
                termination=dt(2022, 10, 1),
                frequency="Q",
                index_base=NoInput(0),
                index_fixings=index_fixings,
            )

    def test_index_reverse_monotonic_decreasing_series(self):
        s = Series([1, 2, 3], index=[dt(2000, 1, 1), dt(1999, 1, 1), dt(1998, 1, 1)])
        assert s.index.is_monotonic_decreasing
        leg = IndexFixedLeg(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 10, 1),
            frequency="Q",
            index_base=NoInput(0),
            index_fixings=s,
        )
        assert leg.index_fixings.index.is_monotonic_increasing


class TestFloatLegExchangeMtm:
    @pytest.mark.parametrize(
        ("fx_fixings", "exp"),
        [
            (NoInput(0), [NoInput(0), NoInput(0), NoInput(0)]),
            ([1.5], [1.5, NoInput(0), NoInput(0)]),
            (1.25, [1.25, NoInput(0), NoInput(0)]),
            ([1.25, 1.35], [1.25, 1.35, NoInput(0)]),
            (Series([1.25, 1.3], index=[dt(2022, 1, 6), dt(2022, 4, 6)]), [1.25, 1.3, NoInput(0)]),
            (Series([1.25], index=[dt(2022, 1, 6)]), [1.25, NoInput(0), NoInput(0)]),
        ],
    )
    def test_float_leg_exchange_mtm(self, fx_fixings, exp) -> None:
        float_leg_exch = FloatLegMtm(
            effective=dt(2022, 1, 3),
            termination=dt(2022, 7, 3),
            frequency="Q",
            notional=265,
            float_spread=5.0,
            currency="usd",
            alt_currency="eur",
            alt_notional=10e6,
            payment_lag_exchange=3,
            fx_fixings=fx_fixings,
        )
        fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
        fxf = FXForwards(
            fxr,
            {
                "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
                "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
                "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
            },
        )

        d = [
            dt(2022, 1, 6),
            dt(2022, 4, 6),
            dt(2022, 7, 6),
        ]  # payment_lag_exchange is 3 days.
        rate = [_ if _ is not NoInput(0) else fxf.rate("eurusd", d[i]) for i, _ in enumerate(exp)]

        float_leg_exch.cashflows(fxf.curve("usd", "usd"), fxf.curve("usd", "usd"), fxf)
        assert float(float_leg_exch.periods[0].cashflow - 10e6 * rate[0]) < 1e-6
        assert float(float_leg_exch.periods[2].cashflow - 10e6 * (rate[1] - rate[0])) < 1e-6
        assert float(float_leg_exch.periods[4].cashflow - 10e6 * (rate[2] - rate[1])) < 1e-6
        assert float_leg_exch.periods[4].payment == d[-1]

        assert float_leg_exch.periods[1].notional == 10e6 * rate[0]
        assert type(float_leg_exch.periods[1]) is FloatPeriod
        assert float_leg_exch.periods[3].notional == 10e6 * rate[1]
        assert type(float_leg_exch.periods[3]) is FloatPeriod

        assert float_leg_exch.periods[-1].notional == 10e6 * rate[1]

    def test_float_leg_exchange_fixings_table(self) -> None:
        float_leg_exch = FloatLegMtm(
            effective=dt(2022, 1, 3),
            termination=dt(2022, 7, 3),
            frequency="Q",
            notional=265,
            float_spread=5.0,
            currency="usd",
            alt_currency="eur",
            alt_notional=10e6,
            payment_lag_exchange=3,
            fixing_method="ibor",
            method_param=0,
        )
        fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
        fxf = FXForwards(
            fxr,
            {
                "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
                "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
                "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
            },
        )

        result = float_leg_exch.fixings_table(fxf.curve("usd", "usd"), fxf.curve("usd", "usd"), fxf)
        assert isinstance(result, DataFrame)
        assert isinstance(
            result.iloc[0, 0], float
        )  # Dual converted to float for fixings table display
        assert abs(result.iloc[0, 0] + 10499895) < 1
        assert abs(result.iloc[1, 0] + 10558419) < 1

    def test_float_leg_exchange_fixings_table_rfr(self) -> None:
        float_leg_exch = FloatLegMtm(
            effective=dt(2022, 1, 3),
            termination=dt(2022, 7, 3),
            frequency="Q",
            notional=265,
            float_spread=5.0,
            currency="usd",
            alt_currency="eur",
            alt_notional=10e6,
            payment_lag_exchange=0,
        )
        fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
        fxf = FXForwards(
            fxr,
            {
                "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
                "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
                "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
            },
        )

        result = float_leg_exch.fixings_table(fxf.curve("usd", "usd"), fxf.curve("usd", "usd"), fxf)
        assert isinstance(result, DataFrame)
        assert isinstance(result.iloc[0, 0], float)  # Dual is converted to float for fixings table

    def test_mtm_leg_exchange_spread(self) -> None:
        leg = FloatLegMtm(
            effective=dt(2022, 1, 3),
            termination=dt(2022, 7, 3),
            frequency="Q",
            notional=265,
            currency="usd",
            alt_currency="eur",
            alt_notional=1e9,
            fixing_method="rfr_payment_delay",
            spread_compound_method="isda_compounding",
            payment_lag_exchange=0,
            payment_lag=0,
            float_spread=0.0,
        )
        fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
        fxf = FXForwards(
            fxr,
            {
                "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
                "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
                "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
            },
        )

        npv = leg.npv(fxf.curve("usd", "usd"), fxf.curve("usd", "usd"), fxf)
        # a_delta = leg.analytic_delta(fxf.curve("usd", "usd"), fxf.curve("usd", "usd"), fxf)
        result = leg._spread(100, fxf.curve("usd", "usd"), fxf.curve("usd", "usd"), fxf)
        leg.float_spread = result
        npv2 = leg.npv(fxf.curve("usd", "usd"), fxf.curve("usd", "usd"), fxf)
        assert abs(npv2 - npv - 100) < 0.01

    @pytest.mark.parametrize(
        ("fx_fixings", "exp"),
        [
            (NoInput(0), [NoInput(0), NoInput(0), NoInput(0)]),
            ([1.5], [1.5, NoInput(0), NoInput(0)]),
            (1.25, [1.25, NoInput(0), NoInput(0)]),
        ],
    )
    def test_mtm_leg_fx_fixings_warn_raise(self, curve, fx_fixings, exp) -> None:
        float_leg_exch = FloatLegMtm(
            effective=dt(2022, 1, 3),
            termination=dt(2022, 7, 3),
            frequency="Q",
            notional=265,
            float_spread=5.0,
            currency="usd",
            alt_currency="eur",
            alt_notional=10e6,
            payment_lag_exchange=3,
            fx_fixings=fx_fixings,
        )

        with pytest.warns(UserWarning), default_context("no_fx_fixings_for_xcs", "warn"):
            float_leg_exch.npv(curve)

        with (
            pytest.raises(ValueError, match="`fx` is required when `fx_fixings` are"),
            default_context("no_fx_fixings_for_xcs", "raise"),
        ):
            float_leg_exch.npv(curve)

    def test_mtm_leg_fx_fixings_series_raises(self) -> None:
        with pytest.raises(ValueError, match="A Series is provided for FX fixings but"):
            FloatLegMtm(
                effective=dt(2022, 1, 3),
                termination=dt(2022, 7, 3),
                frequency="Q",
                notional=265,
                float_spread=5.0,
                currency="usd",
                alt_currency="eur",
                alt_notional=10e6,
                payment_lag_exchange=3,
                fx_fixings=Series([1.25], index=[dt(2022, 2, 6)]),
            )

    def test_mtm_raises_alt(self) -> None:
        with pytest.raises(ValueError, match="`alt_currency` and `currency` must be supplied"):
            FloatLegMtm(
                effective=dt(2022, 1, 3),
                termination=dt(2022, 7, 3),
                frequency="Q",
                notional=265,
                float_spread=5.0,
                currency="usd",
                # alt_currency="eur",
                alt_notional=10e6,
                payment_lag_exchange=3,
            )


class TestCustomLeg:
    @pytest.mark.parametrize(
        "period",
        [
            FixedPeriod(
                start=dt(2022, 1, 1),
                end=dt(2023, 1, 1),
                payment=dt(2023, 1, 9),
                frequency="A",
                fixed_rate=1.0,
            ),
            FloatPeriod(
                start=dt(2022, 1, 1),
                end=dt(2022, 4, 1),
                payment=dt(2022, 4, 3),
                notional=1e9,
                convention="Act360",
                termination=dt(2022, 4, 1),
                frequency="Q",
                float_spread=10.0,
            ),
            CreditPremiumPeriod(
                start=dt(2022, 1, 1),
                end=dt(2022, 4, 1),
                payment=dt(2022, 4, 3),
                notional=1e9,
                convention="Act360",
                termination=dt(2022, 4, 1),
                frequency="Q",
                fixed_rate=4.0,
                currency="usd",
            ),
            CreditProtectionPeriod(
                start=dt(2022, 1, 1),
                end=dt(2022, 4, 1),
                payment=dt(2022, 4, 3),
                notional=1e9,
                convention="Act360",
                termination=dt(2022, 4, 1),
                frequency="Q",
                currency="usd",
            ),
            Cashflow(notional=1e9, payment=dt(2022, 4, 3)),
            IndexFixedPeriod(
                start=dt(2022, 1, 3),
                end=dt(2022, 4, 3),
                payment=dt(2022, 4, 3),
                notional=1e9,
                convention="Act360",
                termination=dt(2022, 4, 3),
                frequency="Q",
                fixed_rate=4.00,
                currency="usd",
                index_base=100.0,
            ),
            IndexCashflow(
                notional=200.0,
                payment=dt(2022, 2, 1),
                index_base=100.0,
            ),
        ],
    )
    def test_init(self, curve, period) -> None:
        CustomLeg(periods=[period, period])

    def test_npv(self, curve) -> None:
        cl = CustomLeg(
            periods=[
                FixedPeriod(
                    start=dt(2022, 1, 1),
                    end=dt(2023, 1, 1),
                    payment=dt(2023, 1, 9),
                    frequency="A",
                    fixed_rate=1.0,
                ),
                FixedPeriod(
                    start=dt(2022, 2, 1),
                    end=dt(2023, 2, 1),
                    payment=dt(2023, 2, 9),
                    frequency="A",
                    fixed_rate=2.0,
                ),
            ],
        )
        result = cl.npv(curve)
        expected = -29109.962157023772
        assert abs(result - expected) < 1e-6

    def test_cashflows(self, curve) -> None:
        cl = CustomLeg(
            periods=[
                FixedPeriod(
                    start=dt(2022, 1, 1),
                    end=dt(2023, 1, 1),
                    payment=dt(2023, 1, 9),
                    frequency="A",
                    fixed_rate=1.0,
                ),
                FixedPeriod(
                    start=dt(2022, 2, 1),
                    end=dt(2023, 2, 1),
                    payment=dt(2023, 2, 9),
                    frequency="A",
                    fixed_rate=2.0,
                ),
            ],
        )
        result = cl.cashflows(curve)
        assert isinstance(result, DataFrame)
        assert len(result.index) == 2

    def test_analytic_delta(self, curve) -> None:
        cl = CustomLeg(
            periods=[
                FixedPeriod(
                    start=dt(2022, 1, 1),
                    end=dt(2023, 1, 1),
                    payment=dt(2023, 1, 9),
                    frequency="A",
                    fixed_rate=1.0,
                ),
                FixedPeriod(
                    start=dt(2022, 2, 1),
                    end=dt(2023, 2, 1),
                    payment=dt(2023, 2, 9),
                    frequency="A",
                    fixed_rate=2.0,
                ),
            ],
        )
        result = cl.analytic_delta(curve)
        expected = 194.1782607729773
        assert abs(result - expected) < 1e-6


def test_leg_amortization() -> None:
    fixed_leg = FixedLeg(
        dt(2022, 1, 1),
        dt(2022, 10, 1),
        frequency="Q",
        notional=1e6,
        amortization=250e3,
        fixed_rate=2.0,
    )
    for i, period in enumerate(fixed_leg.periods):
        assert period.notional == 1e6 - 250e3 * i

    float_leg = FloatLeg(
        dt(2022, 1, 1),
        dt(2022, 10, 1),
        frequency="Q",
        notional=1e6,
        amortization=250e3,
        float_spread=2.0,
    )
    for i, period in enumerate(float_leg.periods):
        assert period.notional == 1e6 - 250e3 * i

    index_leg = IndexFixedLeg(
        dt(2022, 1, 1),
        dt(2022, 10, 1),
        frequency="Q",
        notional=1e6,
        amortization=250e3,
        fixed_rate=2.0,
        index_base=100.0,
    )
    for i, period in enumerate(index_leg.periods):
        assert period.notional == 1e6 - 250e3 * i

    index_leg_exchange = IndexFixedLeg(
        dt(2022, 1, 1),
        dt(2022, 10, 1),
        frequency="Q",
        notional=1e6,
        amortization=250e3,
        fixed_rate=2.0,
        index_base=100.0,
        initial_exchange=False,
        final_exchange=True,
    )
    for i, period in enumerate(index_leg_exchange.periods[0::2]):
        assert period.notional == 1e6 - 250e3 * i
    for i, period in enumerate(index_leg_exchange.periods[1:4:2]):
        assert period.notional == 250e3


def test_custom_leg_raises() -> None:
    with pytest.raises(ValueError):
        _ = CustomLeg(periods=["bad_period"])


def test_custom_leg() -> None:
    float_leg = FloatLeg(effective=dt(2022, 1, 1), termination=dt(2023, 1, 1), frequency="S")
    custom_leg = CustomLeg(periods=float_leg.periods)
    for i, period in enumerate(custom_leg.periods):
        assert period == float_leg.periods[i]


@pytest.mark.parametrize(
    ("fx_fixings", "exp"),
    [
        (NoInput(0), [NoInput(0), NoInput(0), NoInput(0)]),
        ([1.5], [1.5, NoInput(0), NoInput(0)]),
        (1.25, [1.25, NoInput(0), NoInput(0)]),
        ((1.25, Series([1.5], index=[dt(2022, 4, 6)])), [1.25, 1.5, NoInput(0)]),
    ],
)
def test_fixed_leg_exchange_mtm(fx_fixings, exp) -> None:
    fixed_leg_exch = FixedLegMtm(
        effective=dt(2022, 1, 3),
        termination=dt(2022, 7, 3),
        frequency="Q",
        notional=265,
        fixed_rate=5.0,
        currency="usd",
        alt_currency="eur",
        alt_notional=10e6,
        payment_lag_exchange=3,
        fx_fixings=fx_fixings,
    )
    fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
    fxf = FXForwards(
        fxr,
        {
            "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
            "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
            "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
        },
    )

    d = [
        dt(2022, 1, 6),
        dt(2022, 4, 6),
        dt(2022, 7, 6),
    ]  # payment_lag_exchange is 3 days.
    rate = [_ if _ is not NoInput(0) else fxf.rate("eurusd", d[i]) for i, _ in enumerate(exp)]

    fixed_leg_exch.cashflows(fxf.curve("usd", "usd"), fxf.curve("usd", "usd"), fxf)
    assert float(fixed_leg_exch.periods[0].cashflow - 10e6 * rate[0]) < 1e-6
    assert float(fixed_leg_exch.periods[2].cashflow - 10e6 * (rate[1] - rate[0])) < 1e-6
    assert float(fixed_leg_exch.periods[4].cashflow - 10e6 * (rate[2] - rate[1])) < 1e-6
    assert fixed_leg_exch.periods[4].payment == dt(2022, 7, 6)

    assert fixed_leg_exch.periods[1].notional == 10e6 * rate[0]
    assert type(fixed_leg_exch.periods[1]) is FixedPeriod
    assert fixed_leg_exch.periods[3].notional == 10e6 * rate[1]
    assert type(fixed_leg_exch.periods[3]) is FixedPeriod

    assert fixed_leg_exch.periods[-1].notional == 10e6 * rate[1]


@pytest.mark.parametrize("type_", [FloatLegMtm, FixedLegMtm])
def test_mtm_leg_raises(type_) -> None:
    with pytest.raises(ValueError, match="`amortization`"):
        type_(
            effective=dt(2022, 1, 3),
            termination=dt(2022, 7, 3),
            frequency="Q",
            notional=265,
            currency="usd",
            alt_currency="eur",
            alt_notional=10e6,
            payment_lag_exchange=3,
            amortization=1000,
        )

    with pytest.raises(TypeError, match="`fx_fixings` should be scalar"):
        type_(
            effective=dt(2022, 1, 3),
            termination=dt(2022, 7, 3),
            frequency="Q",
            notional=265,
            currency="usd",
            alt_currency="eur",
            alt_notional=10e6,
            payment_lag_exchange=3,
            fx_fixings="bad_type",
        )


@pytest.mark.parametrize(
    ("type_", "expected", "kw"),
    [
        (FloatLegMtm, [522.324262, 522.324262], {"float_spread": 1.0}),
        (FixedLegMtm, [522.324262, 53772.226595], {"fixed_rate": 2.5}),
    ],
)
def test_mtm_leg_exchange_metrics(type_, expected, kw) -> None:
    leg = type_(
        effective=dt(2022, 1, 3),
        termination=dt(2022, 7, 3),
        frequency="Q",
        notional=265,
        currency="usd",
        alt_currency="eur",
        alt_notional=10e6,
        payment_lag_exchange=0,
        payment_lag=0,
        **kw,
    )
    fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3))
    fxf = FXForwards(
        fxr,
        {
            "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
            "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
            "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
        },
    )

    # d = [
    #     dt(2022, 1, 6),
    #     dt(2022, 4, 6),
    #     dt(2022, 7, 6),
    # ]  # payment_lag_exchange is 3 days.
    # rate = [fxf.rate("eurusd", d[i]) for i in range(3)]

    result = leg.analytic_delta(fxf.curve("usd", "usd"), fxf.curve("usd", "usd"), fxf)
    assert float(result - expected[0]) < 1e-6

    result = leg.npv(fxf.curve("usd", "usd"), fxf.curve("usd", "usd"), fxf)
    assert float(result - expected[1]) < 1e-6


@pytest.mark.parametrize(
    ("klass", "kwargs", "expected"),
    [
        (IndexFixedLeg, {}, [200.0, 300.0, 400.0]),
        (
            IndexFixedLeg,
            {"initial_exchange": False, "final_exchange": True},
            [200.0, 300.0, 400.0, 400.0],
        ),
        (ZeroIndexLeg, {}, [400.0]),
    ],
)
def test_set_index_fixings_series_leg_types(klass, kwargs, expected) -> None:
    index_fixings = Series(
        [100.0, 200.0, 300, 400.0, 500.0],
        index=[dt(2022, 1, 1), dt(2022, 2, 1), dt(2022, 5, 1), dt(2022, 8, 1), dt(2022, 11, 1)],
    )
    obj = klass(
        effective=dt(2022, 2, 5),
        termination="9M",
        frequency="Q",
        index_fixings=index_fixings,
        index_base=100.0,
        index_lag=3,
        index_method="monthly",
        **kwargs,
    )
    for i, period in enumerate(obj.periods):
        if type(period) is Cashflow:
            continue
        assert period.index_fixings == expected[i]


@pytest.mark.skip(reason="fixings as a list removed in v2.0")
@pytest.mark.parametrize(
    ("klass", "kwargs", "expected"),
    [
        (IndexFixedLeg, {"index_fixings": [200.0, 300.0, 400.0]}, [200.0, 300.0, 400.0]),
        (
            IndexFixedLeg,
            {
                "initial_exchange": False,
                "final_exchange": True,
                "index_fixings": [200.0, 300.0, 400.0, 400.0],
            },
            [200.0, 300.0, 400.0, 400.0],
        ),
        (ZeroIndexLeg, {"index_fixings": [400.0]}, [400.0]),
    ],
)
def test_set_index_fixings_list_leg_types(klass, kwargs, expected) -> None:
    obj = klass(
        effective=dt(2022, 2, 5),
        termination="9M",
        frequency="Q",
        index_base=100.0,
        index_lag=3,
        index_method="monthly",
        **kwargs,
    )
    for i, period in enumerate(obj.periods):
        if type(period) is Cashflow:
            continue
        assert period.index_fixings == expected[i]


@pytest.mark.parametrize(
    ("klass", "kwargs", "expected"),
    [
        (IndexFixedLeg, {"index_fixings": 200.0}, [200.0, NoInput(0), NoInput(0)]),
        (
            IndexFixedLeg,
            {"initial_exchange": False, "final_exchange": True, "index_fixings": 200.0},
            [200.0, NoInput(0), NoInput(0), NoInput(0)],
        ),
        (ZeroIndexLeg, {"index_fixings": 400.0}, [400.0]),
    ],
)
def test_set_index_fixings_float_leg_types(klass, kwargs, expected) -> None:
    obj = klass(
        effective=dt(2022, 2, 5),
        termination="9M",
        frequency="Q",
        index_base=100.0,
        index_lag=3,
        index_method="monthly",
        **kwargs,
    )
    for i, period in enumerate(obj.periods):
        if type(period) is Cashflow:
            continue
        assert period.index_fixings == expected[i]
