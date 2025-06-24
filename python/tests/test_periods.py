import re
from dataclasses import replace
from datetime import datetime as dt
from datetime import timedelta
from sys import prefix

import pytest
from pandas import NA, DataFrame, Index, MultiIndex, Series, date_range
from pandas.testing import assert_frame_equal
from rateslib import defaults
from rateslib.calendars import Cal
from rateslib.curves import CompositeCurve, Curve, LineCurve
from rateslib.default import NoInput
from rateslib.dual import Dual, gradient
from rateslib.fx import FXForwards, FXRates
from rateslib.fx_volatility import FXDeltaVolSmile, FXSabrSmile, FXSabrSurface
from rateslib.fx_volatility.utils import _d_plus_min_u
from rateslib.periods import (
    Cashflow,
    CreditPremiumPeriod,
    CreditProtectionPeriod,
    FixedPeriod,
    FloatPeriod,
    FXCallPeriod,
    FXPutPeriod,
    IndexCashflow,
    IndexFixedPeriod,
    IndexMixin,
    NonDeliverableCashflow,
    NonDeliverableFixedPeriod,
)


@pytest.fixture
def curve():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.99,
        dt(2022, 7, 1): 0.98,
        dt(2022, 10, 1): 0.97,
    }
    return Curve(nodes=nodes, interpolation="log_linear", id="curve_fixture")


@pytest.fixture
def hazard_curve():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.999,
        dt(2022, 7, 1): 0.997,
        dt(2022, 10, 1): 0.991,
    }
    return Curve(nodes=nodes, interpolation="log_linear", id="hazard_fixture")


@pytest.fixture
def fxr():
    return FXRates({"usdnok": 10.0})


@pytest.fixture
def rfr_curve():
    v1 = 1 / (1 + 0.01 / 365)
    v2 = v1 / (1 + 0.02 / 365)
    v3 = v2 / (1 + 0.03 / 365)
    v4 = v3 / (1 + 0.04 / 365)

    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 1, 2): v1,
        dt(2022, 1, 3): v2,
        dt(2022, 1, 4): v3,
        dt(2022, 1, 5): v4,
    }
    return Curve(nodes=nodes, interpolation="log_linear", convention="act365f")


@pytest.fixture
def line_curve():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 1, 2): 2.00,
        dt(2022, 1, 3): 3.00,
        dt(2022, 1, 4): 4.00,
        dt(2022, 1, 5): 5.00,
    }
    return LineCurve(nodes=nodes, interpolation="linear", convention="act365f")


@pytest.mark.parametrize(
    "obj",
    [
        FixedPeriod(dt(2000, 1, 1), dt(2000, 2, 1), dt(2000, 2, 1), frequency="m", fixed_rate=2.0),
        Cashflow(notional=1e6, payment=dt(2022, 1, 1), currency="usd"),
        IndexCashflow(notional=1e6, payment=dt(2022, 1, 1), currency="usd", index_base=100.0),
        IndexFixedPeriod(
            dt(2000, 1, 1), dt(2000, 2, 1), dt(2000, 2, 1), frequency="m", fixed_rate=2.0
        ),
        FloatPeriod(dt(2000, 1, 1), dt(2000, 2, 1), dt(2000, 2, 1), frequency="m"),
        FXCallPeriod(
            pair="eurusd",
            expiry=dt(2000, 1, 1),
            delivery=dt(2000, 1, 1),
            payment=dt(2000, 1, 1),
        ),
        FXPutPeriod(
            pair="eurusd",
            expiry=dt(2000, 1, 1),
            delivery=dt(2000, 1, 1),
            payment=dt(2000, 1, 1),
        ),
    ],
)
def test_repr(obj):
    result = obj.__repr__()
    expected = f"<rl.{type(obj).__name__} at {hex(id(obj))}>"
    assert result == expected


class TestFXandBase:
    def test_fx_and_base_raise(self) -> None:
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.96}, id="curve")
        per = FixedPeriod(
            dt(2022, 2, 1),
            dt(2022, 3, 1),
            dt(2022, 3, 1),
            "A",
            fixed_rate=2,
            currency="usd",
        )
        with pytest.raises(ValueError, match="`base` "):
            per.npv(curve, curve, base="eur")

    def test_fx_and_base_warn1(self) -> None:
        # base and numeric fx given.
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.96}, id="curve")
        per = FixedPeriod(
            dt(2022, 2, 1),
            dt(2022, 3, 1),
            dt(2022, 3, 1),
            "A",
            fixed_rate=2,
            currency="usd",
        )
        with pytest.warns(UserWarning, match="`base` "):
            per.npv(curve, curve, fx=1.1, base="eur")

    def test_fx_and_base_warn2(self) -> None:
        # base is none and numeric fx given.
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.96}, id="curve")
        per = FixedPeriod(
            dt(2022, 2, 1),
            dt(2022, 3, 1),
            dt(2022, 3, 1),
            "A",
            fixed_rate=2,
            currency="usd",
        )
        with pytest.warns(UserWarning, match="It is not best practice to provide"):
            per.npv(curve, curve, fx=1.1)


class TestFloatPeriod:
    def test_none_cashflow(self) -> None:
        float_period = FloatPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
        )
        assert float_period.cashflow(None) is None

    @pytest.mark.parametrize(
        ("spread_method", "float_spread", "expected"),
        [
            ("none_simple", 100.0, 24744.478172244584),
            ("isda_compounding", 0.0, 24744.478172244584),
            ("isda_compounding", 100.0, 25053.484941157145),
            ("isda_flat_compounding", 100.0, 24867.852396116967),
        ],
    )
    def test_float_period_analytic_delta(
        self,
        curve,
        spread_method,
        float_spread,
        expected,
    ) -> None:
        float_period = FloatPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            float_spread=float_spread,
            spread_compound_method=spread_method,
        )
        result = float_period.analytic_delta(curve)
        assert abs(result - expected) < 1e-7

    @pytest.mark.parametrize(
        ("spread", "crv", "fx"),
        [
            (4.00, True, 2.0),
            (NoInput(0), False, 2.0),
            (4.00, True, 10.0),
            (NoInput(0), False, 10.0),
        ],
    )
    def test_float_period_cashflows(self, curve, fxr, spread, crv, fx) -> None:
        float_period = FloatPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            float_spread=spread,
        )
        curve = curve if crv else None
        rate = None if curve is None else float(float_period.rate(curve))
        cashflow = None if rate is None else rate * -1e9 * float_period.dcf / 100
        expected = {
            defaults.headers["type"]: "FloatPeriod",
            defaults.headers["stub_type"]: "Regular",
            defaults.headers["a_acc_start"]: dt(2022, 1, 1),
            defaults.headers["a_acc_end"]: dt(2022, 4, 1),
            defaults.headers["payment"]: dt(2022, 4, 3),
            defaults.headers["notional"]: 1e9,
            defaults.headers["currency"]: "USD",
            defaults.headers["convention"]: "Act360",
            defaults.headers["dcf"]: float_period.dcf,
            defaults.headers["df"]: 0.9897791268897856 if crv else None,
            defaults.headers["rate"]: rate,
            defaults.headers["spread"]: 0 if spread is NoInput.blank else spread,
            defaults.headers["npv"]: -10096746.871171726 if crv else None,
            defaults.headers["cashflow"]: cashflow,
            defaults.headers["fx"]: fx,
            defaults.headers["npv_fx"]: -10096746.871171726 * fx if crv else None,
            defaults.headers["collateral"]: None,
        }
        if fx == 2.0:
            with pytest.warns(UserWarning):
                # It is not best practice to provide `fx` as numeric
                result = float_period.cashflows(
                    curve if crv else NoInput(0),
                    fx=2.0,
                    base=NoInput(0),
                )
        else:
            result = float_period.cashflows(
                curve if crv else NoInput(0),
                fx=fxr,
                base="nok",
            )
        assert result == expected

    def test_spread_compound_raises(self) -> None:
        with pytest.raises(ValueError, match="`spread_compound_method`"):
            FloatPeriod(
                start=dt(2022, 1, 1),
                end=dt(2022, 4, 1),
                payment=dt(2022, 4, 3),
                frequency="Q",
                spread_compound_method="bad_vibes",
            )

    def test_spread_compound_calc_raises(self) -> None:
        period = FloatPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            frequency="Q",
            spread_compound_method="none_simple",
            float_spread=1,
        )
        period.spread_compound_method = "bad_vibes"
        with pytest.raises(ValueError, match="`spread_compound_method` must be in"):
            period._rate_rfr_isda_compounded_with_spread(Series([1, 2]), Series([1, 1]))

    def test_rfr_lockout_too_few_dates(self, curve) -> None:
        period = FloatPeriod(
            start=dt(2022, 1, 10),
            end=dt(2022, 1, 15),
            payment=dt(2022, 1, 15),
            frequency="M",
            fixing_method="rfr_lockout",
            method_param=6,
        )
        with pytest.raises(ValueError, match="period has too few dates"):
            period.rate(curve)

    def test_fixing_method_raises(self) -> None:
        with pytest.raises(ValueError, match="`fixing_method`"):
            FloatPeriod(
                start=dt(2022, 1, 1),
                end=dt(2022, 4, 1),
                payment=dt(2022, 4, 3),
                frequency="Q",
                fixing_method="bad_vibes",
            )

    def test_float_period_npv(self, curve) -> None:
        float_period = FloatPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
        )
        result = float_period.npv(curve)
        assert abs(result + 9997768.95848275) < 1e-7

    def test_rfr_avg_method_raises(self, curve) -> None:
        period = FloatPeriod(
            dt(2022, 1, 1),
            dt(2022, 1, 4),
            dt(2022, 1, 4),
            "Q",
            fixing_method="rfr_payment_delay_avg",
            spread_compound_method="isda_compounding",
        )
        msg = "`spread_compound` method must be 'none_simple' in an RFR averaging period."
        with pytest.raises(ValueError, match=msg):
            period.rate(curve)

    @pytest.mark.parametrize("curve_type", ["curve", "line_curve"])
    def test_rfr_payment_delay_method(self, curve_type, rfr_curve, line_curve) -> None:
        curve = rfr_curve if curve_type == "curve" else line_curve
        period = FloatPeriod(
            dt(2022, 1, 1),
            dt(2022, 1, 4),
            dt(2022, 1, 4),
            "Q",
            fixing_method="rfr_payment_delay",
        )
        result = period.rate(curve)
        expected = ((1 + 0.01 / 365) * (1 + 0.02 / 365) * (1 + 0.03 / 365) - 1) * 36500 / 3
        assert abs(result - expected) < 1e-12

    @pytest.mark.parametrize("curve_type", ["curve", "line_curve"])
    def test_rfr_payment_delay_method_with_fixings(self, curve_type, rfr_curve, line_curve) -> None:
        curve = rfr_curve if curve_type == "curve" else line_curve
        period = FloatPeriod(
            dt(2022, 1, 1),
            dt(2022, 1, 4),
            dt(2022, 1, 4),
            "Q",
            fixing_method="rfr_payment_delay",
            fixings=[10, 8],
        )
        result = period.rate(curve)
        expected = ((1 + 0.10 / 365) * (1 + 0.08 / 365) * (1 + 0.03 / 365) - 1) * 36500 / 3
        assert abs(result - expected) < 1e-12

    @pytest.mark.parametrize("curve_type", ["curve", "line_curve"])
    def test_rfr_payment_delay_avg_method(self, curve_type, rfr_curve, line_curve) -> None:
        curve = rfr_curve if curve_type == "curve" else line_curve
        period = FloatPeriod(
            dt(2022, 1, 1),
            dt(2022, 1, 4),
            dt(2022, 1, 4),
            "Q",
            fixing_method="rfr_payment_delay_avg",
        )
        result = period.rate(curve)
        expected = (1.0 + 2.0 + 3.0) / 3
        assert abs(result - expected) < 1e-11

    @pytest.mark.parametrize("curve_type", ["curve", "line_curve"])
    def test_rfr_payment_delay_avg_method_with_fixings(
        self,
        curve_type,
        rfr_curve,
        line_curve,
    ) -> None:
        curve = rfr_curve if curve_type == "curve" else line_curve
        period = FloatPeriod(
            dt(2022, 1, 1),
            dt(2022, 1, 4),
            dt(2022, 1, 4),
            "Q",
            fixing_method="rfr_payment_delay_avg",
            fixings=[10, 8],
        )
        result = period.rate(curve)
        expected = (10.0 + 8.0 + 3.0) / 3
        assert abs(result - expected) < 1e-11

    @pytest.mark.parametrize("curve_type", ["curve", "line_curve"])
    def test_rfr_lockout_avg_method(self, curve_type, rfr_curve, line_curve) -> None:
        curve = rfr_curve if curve_type == "curve" else line_curve
        period = FloatPeriod(
            dt(2022, 1, 1),
            dt(2022, 1, 4),
            dt(2022, 1, 4),
            "Q",
            fixing_method="rfr_lockout_avg",
            method_param=2,
        )
        assert period._is_inefficient is True  # lockout requires all fixings.
        result = period.rate(curve)
        expected = 1.0
        assert abs(result - expected) < 1e-11

        period = FloatPeriod(
            dt(2022, 1, 2),
            dt(2022, 1, 5),
            dt(2022, 1, 5),
            "Q",
            fixing_method="rfr_lockout_avg",
            method_param=1,
        )
        result = period.rate(rfr_curve)
        expected = (2 + 3.0 + 3.0) / 3
        assert abs(result - expected) < 1e-11

    @pytest.mark.parametrize("curve_type", ["curve", "line_curve"])
    def test_rfr_lockout_avg_method_with_fixings(self, curve_type, rfr_curve, line_curve) -> None:
        curve = rfr_curve if curve_type == "curve" else line_curve
        period = FloatPeriod(
            dt(2022, 1, 1),
            dt(2022, 1, 4),
            dt(2022, 1, 4),
            "Q",
            fixing_method="rfr_lockout_avg",
            method_param=2,
            fixings=[10, 8],
        )
        result = period.rate(curve)
        expected = 10.0
        assert abs(result - expected) < 1e-12

        period = FloatPeriod(
            dt(2022, 1, 2),
            dt(2022, 1, 5),
            dt(2022, 1, 5),
            "Q",
            fixing_method="rfr_lockout_avg",
            method_param=1,
            fixings=[10, 8],
        )
        result = period.rate(rfr_curve)
        expected = (10.0 + 8.0 + 8.0) / 3
        assert abs(result - expected) < 1e-12

    @pytest.mark.parametrize("curve_type", ["curve", "line_curve"])
    def test_rfr_lockout_method(self, curve_type, rfr_curve, line_curve) -> None:
        curve = rfr_curve if curve_type == "curve" else line_curve
        period = FloatPeriod(
            dt(2022, 1, 1),
            dt(2022, 1, 4),
            dt(2022, 1, 4),
            "Q",
            fixing_method="rfr_lockout",
            method_param=2,
        )
        assert period._is_inefficient is True  # lockout requires all fixings.
        result = period.rate(curve)
        expected = ((1 + 0.01 / 365) * (1 + 0.01 / 365) * (1 + 0.01 / 365) - 1) * 36500 / 3
        assert abs(result - expected) < 1e-12

        period = FloatPeriod(
            dt(2022, 1, 2),
            dt(2022, 1, 5),
            dt(2022, 1, 5),
            "Q",
            fixing_method="rfr_lockout",
            method_param=1,
        )
        result = period.rate(rfr_curve)
        expected = ((1 + 0.02 / 365) * (1 + 0.03 / 365) * (1 + 0.03 / 365) - 1) * 36500 / 3
        assert abs(result - expected) < 1e-12

    @pytest.mark.parametrize("curve_type", ["curve", "line_curve"])
    def test_rfr_lockout_method_with_fixings(self, curve_type, rfr_curve, line_curve) -> None:
        curve = rfr_curve if curve_type == "curve" else line_curve
        period = FloatPeriod(
            dt(2022, 1, 1),
            dt(2022, 1, 4),
            dt(2022, 1, 4),
            "Q",
            fixing_method="rfr_lockout",
            method_param=2,
            fixings=[10, 8],
        )
        result = period.rate(curve)
        expected = ((1 + 0.10 / 365) * (1 + 0.10 / 365) * (1 + 0.10 / 365) - 1) * 36500 / 3
        assert abs(result - expected) < 1e-12

        period = FloatPeriod(
            dt(2022, 1, 2),
            dt(2022, 1, 5),
            dt(2022, 1, 5),
            "Q",
            fixing_method="rfr_lockout",
            method_param=1,
            fixings=[10, 8],
        )
        result = period.rate(rfr_curve)
        expected = ((1 + 0.10 / 365) * (1 + 0.08 / 365) * (1 + 0.08 / 365) - 1) * 36500 / 3
        assert abs(result - expected) < 1e-12

    @pytest.mark.parametrize("curve_type", ["curve", "line_curve"])
    def test_rfr_observation_shift_method(self, curve_type, rfr_curve, line_curve) -> None:
        curve = rfr_curve if curve_type == "curve" else line_curve
        period = FloatPeriod(
            dt(2022, 1, 2),
            dt(2022, 1, 5),
            dt(2022, 1, 5),
            "Q",
            fixing_method="rfr_observation_shift",
            method_param=1,
        )
        result = period.rate(curve)
        expected = ((1 + 0.01 / 365) * (1 + 0.02 / 365) * (1 + 0.03 / 365) - 1) * 36500 / 3
        assert abs(result - expected) < 1e-12

        period = FloatPeriod(
            dt(2022, 1, 3),
            dt(2022, 1, 5),
            dt(2022, 1, 5),
            "Q",
            fixing_method="rfr_observation_shift",
            method_param=2,
        )
        result = period.rate(curve)
        expected = ((1 + 0.01 / 365) * (1 + 0.02 / 365) - 1) * 36500 / 2
        assert abs(result - expected) < 1e-12

    @pytest.mark.parametrize("curve_type", ["curve", "line_curve"])
    def test_rfr_observation_shift_method_with_fixings(
        self,
        curve_type,
        rfr_curve,
        line_curve,
    ) -> None:
        curve = rfr_curve if curve_type == "curve" else line_curve
        period = FloatPeriod(
            dt(2022, 1, 2),
            dt(2022, 1, 5),
            dt(2022, 1, 5),
            "Q",
            fixing_method="rfr_observation_shift",
            method_param=1,
            fixings=[10, 8],
        )
        result = period.rate(curve)
        expected = ((1 + 0.10 / 365) * (1 + 0.08 / 365) * (1 + 0.03 / 365) - 1) * 36500 / 3
        assert abs(result - expected) < 1e-12

        period = FloatPeriod(
            dt(2022, 1, 3),
            dt(2022, 1, 5),
            dt(2022, 1, 5),
            "Q",
            fixing_method="rfr_observation_shift",
            method_param=2,
            fixings=[10, 8],
        )
        result = period.rate(curve)
        expected = ((1 + 0.10 / 365) * (1 + 0.08 / 365) - 1) * 36500 / 2
        assert abs(result - expected) < 1e-12

    @pytest.mark.parametrize("curve_type", ["curve", "line_curve"])
    def test_rfr_observation_shift_avg_method(self, curve_type, rfr_curve, line_curve) -> None:
        curve = rfr_curve if curve_type == "curve" else line_curve
        period = FloatPeriod(
            dt(2022, 1, 2),
            dt(2022, 1, 5),
            dt(2022, 1, 5),
            "Q",
            fixing_method="rfr_observation_shift_avg",
            method_param=1,
        )
        result = period.rate(curve)
        expected = (1.0 + 2 + 3) / 3
        assert abs(result - expected) < 1e-11

        period = FloatPeriod(
            dt(2022, 1, 3),
            dt(2022, 1, 5),
            dt(2022, 1, 5),
            "Q",
            fixing_method="rfr_observation_shift_avg",
            method_param=2,
        )
        result = period.rate(curve)
        expected = (1.0 + 2.0) / 2
        assert abs(result - expected) < 1e-11

    @pytest.mark.parametrize("curve_type", ["curve", "line_curve"])
    def test_rfr_observation_shift_avg_method_with_fixings(
        self,
        curve_type,
        rfr_curve,
        line_curve,
    ) -> None:
        curve = rfr_curve if curve_type == "curve" else line_curve
        period = FloatPeriod(
            dt(2022, 1, 2),
            dt(2022, 1, 5),
            dt(2022, 1, 5),
            "Q",
            fixing_method="rfr_observation_shift_avg",
            method_param=1,
            fixings=[10, 8],
        )
        result = period.rate(curve)
        expected = (10.0 + 8.0 + 3.0) / 3
        assert abs(result - expected) < 1e-11

        period = FloatPeriod(
            dt(2022, 1, 3),
            dt(2022, 1, 5),
            dt(2022, 1, 5),
            "Q",
            fixing_method="rfr_observation_shift_avg",
            method_param=2,
            fixings=[10, 8],
        )
        result = period.rate(curve)
        expected = (10.0 + 8) / 2
        assert abs(result - expected) < 1e-11

    def test_dcf_obs_period_raises(self) -> None:
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, calendar="ldn")
        float_period = FloatPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 12, 31),
            payment=dt(2022, 12, 31),
            frequency="A",
            fixing_method="rfr_lookback",
            method_param=5,
        )
        # this may only raise when lookback is used ?
        with pytest.raises(ValueError, match="RFR Observation and Accrual DCF dates "):
            float_period.rate(curve)

    @pytest.mark.parametrize("curve_type", ["curve", "linecurve"])
    @pytest.mark.parametrize(
        ("method", "expected", "expected_date"),
        [
            ("rfr_payment_delay", [1000000, 1000082, 1000191, 1000561], dt(2022, 1, 6)),
            ("rfr_observation_shift", [1499240, 1499281, 1499363, 1499486], dt(2022, 1, 4)),
            ("rfr_lockout", [999931, 4999411, 0, 0], dt(2022, 1, 6)),
            ("rfr_lookback", [999657, 999685, 2998726, 999821], dt(2022, 1, 4)),
        ],
    )
    def test_rfr_fixings_array(self, curve_type, method, expected, expected_date) -> None:
        # tests the fixings array and the compounding for different types of curve
        # at different rates in the period.

        v1 = 1 / (1 + 0.01 / 365)
        v2 = v1 / (1 + 0.02 / 365)
        v3 = v2 / (1 + 0.03 / 365)
        v4 = v3 / (1 + 0.04 / 365)
        v5 = v4 / (1 + 0.045 * 3 / 365)
        v6 = v5 / (1 + 0.05 / 365)
        v7 = v6 / (1 + 0.055 / 365)

        nodes = {
            dt(2022, 1, 3): 1.00,
            dt(2022, 1, 4): v1,
            dt(2022, 1, 5): v2,
            dt(2022, 1, 6): v3,
            dt(2022, 1, 7): v4,
            dt(2022, 1, 10): v5,
            dt(2022, 1, 11): v6,
            dt(2022, 1, 12): v7,
        }
        curve = Curve(
            nodes=nodes,
            interpolation="log_linear",
            convention="act365f",
            calendar="bus",
        )

        line_curve = LineCurve(
            nodes={
                dt(2022, 1, 2): -99,
                dt(2022, 1, 3): 1.0,
                dt(2022, 1, 4): 2.0,
                dt(2022, 1, 5): 3.0,
                dt(2022, 1, 6): 4.0,
                dt(2022, 1, 7): 4.5,
                dt(2022, 1, 10): 5.0,
                dt(2022, 1, 11): 5.5,
            },
            interpolation="linear",
            convention="act365f",
            calendar="bus",
        )
        rfr_curve = curve if curve_type == "curve" else line_curve

        period = FloatPeriod(
            dt(2022, 1, 5),
            dt(2022, 1, 11),
            dt(2022, 1, 11),
            "Q",
            fixing_method=method,
            convention="act365f",
            notional=-1000000,
        )
        _d = period._rfr_get_individual_fixings_data(
            rfr_curve.meta.calendar, rfr_curve.meta.convention, rfr_curve
        )
        rate, table = period._rfr_fixings_array(
            d=_d,
            disc_curve=curve,
        )

        assert table["obs_dates"][1] == expected_date
        for i, val in table["notional"].iloc[:-1].items():
            assert abs(expected[i] - val) < 1

    def test_rfr_fixings_array_raises(self, rfr_curve) -> None:
        period = FloatPeriod(
            dt(2022, 1, 5),
            dt(2022, 1, 11),
            dt(2022, 1, 11),
            "Q",
            fixing_method="rfr_payment_delay",
            notional=-1000000,
        )
        period.fixing_method = "bad_vibes"
        with pytest.raises(NotImplementedError, match="`fixing_method`"):
            period._rfr_fixings_array(
                period._rfr_get_individual_fixings_data(
                    rfr_curve.meta.calendar, rfr_curve.meta.convention, rfr_curve
                ),
                rfr_curve,
            )

    def test_rfr_fixings_array_raises2(self, line_curve) -> None:
        period = FloatPeriod(
            dt(2022, 1, 5),
            dt(2022, 1, 11),
            dt(2022, 1, 11),
            "Q",
            fixing_method="rfr_payment_delay",
            convention="act365f",
            notional=-1000000,
        )
        with pytest.raises(ValueError, match="`disc_curve` cannot be inferred from a non-DF"):
            period.fixings_table(curve=line_curve)

        with pytest.raises(ValueError, match="`disc_curve` cannot be inferred from a dictionary"):
            period.fixings_table(curve={"1m": line_curve, "2m": line_curve})

    @pytest.mark.parametrize(
        ("method", "param", "expected"),
        [
            ("rfr_payment_delay", 0, 1000000),
            ("rfr_observation_shift", 1, 333319),
            ("rfr_lookback", 1, 333319),
        ],
    )
    def test_rfr_fixings_array_single_period(self, method, param, expected) -> None:
        rfr_curve = Curve(
            nodes={dt(2022, 1, 3): 1.0, dt(2022, 1, 15): 0.9995},
            interpolation="log_linear",
            convention="act365f",
            calendar="bus",
        )
        period = FloatPeriod(
            dt(2022, 1, 10),
            dt(2022, 1, 11),
            dt(2022, 1, 11),
            "Q",
            fixing_method=method,
            method_param=param,
            notional=-1000000,
            convention="act365f",
        )
        result = period.fixings_table(rfr_curve)
        assert abs(result[(rfr_curve.id, "notional")].iloc[0] - expected) < 1

    @pytest.mark.parametrize(
        ("method", "param", "expected"),
        [
            ("rfr_payment_delay", 0, 3.20040557),
            ("rfr_lockout", 1, 3.80063892),
            ("rfr_lookback", 1, 3.20040557),
            ("rfr_observation_shift", 1, 4.00045001),
        ],
    )
    def test_rfr_period_all_types_with_defined_fixings(self, method, param, expected):
        # This is probably a redundant test but it came later after some refactoring and
        # was double checked with manual calculation in Excel. Easy to do.
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2022, 3, 1): 1.0}, calendar="nyc")
        period = FloatPeriod(
            start=dt(2022, 1, 28),
            end=dt(2022, 2, 2),
            frequency="A",
            payment=dt(2022, 1, 1),
            fixing_method=method,
            method_param=param,
            convention="act360",
            calendar="nyc",
            fixings=[3.0, 5.0, 2.0],
        )
        result = period.rate(curve)
        assert abs(result - expected) < 1e-8

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            (
                "none_simple",
                ((1 + 0.01 / 365) * (1 + 0.02 / 365) * (1 + 0.03 / 365) - 1) * 36500 / 3
                + 100 / 100,
            ),
            (
                "isda_compounding",
                ((1 + 0.02 / 365) * (1 + 0.03 / 365) * (1 + 0.04 / 365) - 1) * 36500 / 3,
            ),
            ("isda_flat_compounding", 3.000173518986841),
        ],
    )
    def test_rfr_compounding_float_spreads(self, method, expected, rfr_curve) -> None:
        period = FloatPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 1, 4),
            payment=dt(2022, 1, 4),
            frequency="M",
            float_spread=100,
            spread_compound_method=method,
            convention="act365f",
        )
        result = period.rate(rfr_curve)
        assert abs(result - expected) < 1e-8

    def test_ibor_rate_line_curve(self, line_curve) -> None:
        period = FloatPeriod(
            start=dt(2022, 1, 5),
            end=dt(2022, 4, 5),
            payment=dt(2022, 4, 5),
            frequency="Q",
            fixing_method="ibor",
            method_param=2,
        )
        assert period._is_inefficient is False
        assert period.rate(line_curve) == 3.0

    def test_ibor_fixing_table(self, line_curve, curve) -> None:
        float_period = FloatPeriod(
            start=dt(2022, 1, 4),
            end=dt(2022, 4, 4),
            payment=dt(2022, 4, 4),
            frequency="Q",
            fixing_method="ibor",
            method_param=2,
            convention="act365f",
        )
        result = float_period.fixings_table(line_curve, disc_curve=curve)
        expected = DataFrame(
            {
                "obs_dates": [dt(2022, 1, 2)],
                "notional": [-1e6],
                "risk": [-24.402790080357686],
                "dcf": [0.2465753424657534],
                "rates": [2.0],
            },
        ).set_index("obs_dates")
        expected.columns = MultiIndex.from_tuples(
            [
                (line_curve.id, "notional"),
                (line_curve.id, "risk"),
                (line_curve.id, "dcf"),
                (line_curve.id, "rates"),
            ]
        )
        assert_frame_equal(expected, result)

    def test_ibor_fixing_table_right(self, line_curve, curve) -> None:
        float_period = FloatPeriod(
            start=dt(2022, 1, 4),
            end=dt(2022, 4, 4),
            payment=dt(2022, 4, 4),
            frequency="Q",
            fixing_method="ibor",
            method_param=2,
            convention="act365f",
        )
        result = float_period.fixings_table(line_curve, disc_curve=curve, right=dt(2022, 1, 1))
        expected = DataFrame(
            {
                "obs_dates": [],
                "notional": [],
                "risk": [],
                "dcf": [],
                "rates": [],
            },
        ).set_index("obs_dates")
        expected.columns = MultiIndex.from_tuples(
            [
                (line_curve.id, "notional"),
                (line_curve.id, "risk"),
                (line_curve.id, "dcf"),
                (line_curve.id, "rates"),
            ]
        )
        assert_frame_equal(expected, result)

    def test_ibor_fixing_table_fast(self, line_curve, curve) -> None:
        float_period = FloatPeriod(
            start=dt(2022, 1, 4),
            end=dt(2022, 4, 4),
            payment=dt(2022, 4, 4),
            frequency="Q",
            fixing_method="ibor",
            method_param=2,
            convention="act365f",
        )
        result = float_period.fixings_table(line_curve, disc_curve=curve, approximate=True)
        expected = DataFrame(
            {
                "obs_dates": [dt(2022, 1, 2)],
                "notional": [-1e6],
                "risk": [-24.402790080357686],
                "dcf": [0.2465753424657534],
                "rates": [2.0],
            },
        ).set_index("obs_dates")
        expected.columns = MultiIndex.from_tuples(
            [
                (line_curve.id, "notional"),
                (line_curve.id, "risk"),
                (line_curve.id, "dcf"),
                (line_curve.id, "rates"),
            ]
        )
        assert_frame_equal(expected, result)

    def test_ibor_fixings(self) -> None:
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2025, 1, 1): 0.90}, calendar="bus")
        fixings = Series(
            [1.00, 2.801, 1.00, 1.00],
            index=[dt(2023, 3, 1), dt(2023, 3, 2), dt(2023, 3, 3), dt(2023, 3, 6)],
        )
        float_period = FloatPeriod(
            start=dt(2023, 3, 6),
            end=dt(2023, 6, 6),
            payment=dt(2023, 6, 6),
            frequency="Q",
            fixing_method="ibor",
            method_param=2,
            fixings=fixings,
        )
        result = float_period.rate(curve)
        assert result == 2.801

    @pytest.mark.parametrize("approx", [False, True])
    def test_ibor_fixings_table_historical_before_curve(self, approx) -> None:
        # fixing table should return a DataFrame with an unknown rate and zero exposure
        # the fixing has occurred in the past but is unspecified.
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2025, 1, 1): 0.90}, calendar="bus")
        float_period = FloatPeriod(
            start=dt(2000, 2, 2),
            end=dt(2000, 5, 2),
            payment=dt(2000, 5, 2),
            frequency="Q",
            fixing_method="ibor",
            method_param=0,
        )
        result = float_period.fixings_table(curve, approximate=approx)
        expected = DataFrame(
            data=[[0.0, 0.0, 0.25, NA]],
            index=Index([dt(2000, 2, 2)], name="obs_dates"),
            columns=MultiIndex.from_tuples(
                [
                    (curve.id, "notional"),
                    (curve.id, "risk"),
                    (curve.id, "dcf"),
                    (curve.id, "rates"),
                ],
            ),
        )
        assert_frame_equal(expected, result)

    @pytest.mark.parametrize("approx", [False, True])
    def test_rfr_fixings_table_historical_before_curve(self, approx) -> None:
        # fixing table should return a DataFrame with an unknown rate and zero exposure
        # the fixing has occurred in the past but is unspecified.
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2025, 1, 1): 0.90}, calendar="bus")
        float_period = FloatPeriod(
            start=dt(2000, 2, 2),
            end=dt(2000, 3, 2),
            payment=dt(2000, 3, 2),
            frequency="Q",
            fixing_method="rfr_payment_delay",
            method_param=0,
        )
        if approx:
            with pytest.warns(UserWarning, match="Errored approximating a fixings table"):
                result = float_period.fixings_table(curve, approximate=approx)
        else:
            result = float_period.fixings_table(curve, approximate=approx)
        assert isinstance(result, DataFrame)
        assert result.iloc[0, 0] == 0.0
        assert result[f"{curve.id}", "notional"][dt(2000, 3, 1)] == 0.0

    def test_ibor_fixing_unavailable(self) -> None:
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2025, 1, 1): 0.90}, calendar="bus")
        lcurve = LineCurve({dt(2022, 1, 1): 2.0, dt(2025, 1, 1): 4.0}, calendar="bus")
        fixings = Series([2.801], index=[dt(2023, 3, 1)])
        float_period = FloatPeriod(
            start=dt(2023, 3, 20),
            end=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            frequency="Q",
            fixing_method="ibor",
            method_param=2,
            fixings=fixings,
        )
        result = float_period.rate(curve)  # fixing occurs 18th Mar, not in `fixings`
        assert abs(result - 3.476095729528156) < 1e-5
        result = float_period.rate(lcurve)  # fixing occurs 18th Mar, not in `fixings`
        assert abs(result - 2.801094890510949) < 1e-5

    @pytest.mark.parametrize("float_spread", [0, 100])
    def test_ibor_rate_df_curve(self, float_spread, curve) -> None:
        period = FloatPeriod(
            start=dt(2022, 4, 1),
            end=dt(2022, 7, 1),
            payment=dt(2022, 7, 1),
            frequency="Q",
            fixing_method="ibor",
            method_param=2,
            float_spread=float_spread,
        )
        expected = (0.99 / 0.98 - 1) * 36000 / 91 + float_spread / 100
        assert period.rate(curve) == expected

    @pytest.mark.parametrize("float_spread", [0, 100])
    def test_ibor_rate_stub_df_curve(self, float_spread, curve) -> None:
        period = FloatPeriod(
            start=dt(2022, 4, 1),
            end=dt(2022, 5, 1),
            payment=dt(2022, 5, 1),
            frequency="Q",
            fixing_method="ibor",
            method_param=2,
            stub=True,
            float_spread=float_spread,
        )
        expected = (0.99 / curve[dt(2022, 5, 1)] - 1) * 36000 / 30 + float_spread / 100
        assert period.rate(curve) == expected

    def test_single_fixing_override(self, curve) -> None:
        period = FloatPeriod(
            start=dt(2022, 4, 1),
            end=dt(2022, 5, 1),
            payment=dt(2022, 5, 1),
            frequency="Q",
            fixing_method="ibor",
            method_param=2,
            stub=True,
            float_spread=100,
            fixings=7.5,
        )
        expected = 7.5 + 1
        assert period.rate(curve) == expected

    @pytest.mark.parametrize("curve_type", ["curve", "linecurve"])
    def test_period_historic_fixings(self, curve_type, line_curve, rfr_curve) -> None:
        curve = rfr_curve if curve_type == "curve" else line_curve
        period = FloatPeriod(
            start=dt(2021, 12, 30),
            end=dt(2022, 1, 3),
            payment=dt(2022, 1, 3),
            frequency="Q",
            fixing_method="rfr_payment_delay",
            float_spread=100,
            fixings=[1.5, 2.5],
            convention="act365F",
        )
        expected = (
            (1 + 0.015 / 365) * (1 + 0.025 / 365) * (1 + 0.01 / 365) * (1 + 0.02 / 365) - 1
        ) * 36500 / 4 + 1
        assert period.rate(curve) == expected

    @pytest.mark.parametrize("curve_type", ["curve", "linecurve"])
    def test_period_historic_fixings_series(self, curve_type, line_curve, rfr_curve) -> None:
        curve = rfr_curve if curve_type == "curve" else line_curve
        fixings = Series(
            [99, 99, 1.5, 2.5],
            index=[dt(1995, 1, 1), dt(2021, 12, 29), dt(2021, 12, 30), dt(2021, 12, 31)],
        )
        period = FloatPeriod(
            start=dt(2021, 12, 30),
            end=dt(2022, 1, 3),
            payment=dt(2022, 1, 3),
            frequency="Q",
            fixing_method="rfr_payment_delay",
            float_spread=100,
            fixings=fixings,
            convention="act365F",
        )
        expected = (
            (1 + 0.015 / 365) * (1 + 0.025 / 365) * (1 + 0.01 / 365) * (1 + 0.02 / 365) - 1
        ) * 36500 / 4 + 1
        result = period.rate(curve)
        assert result == expected

    @pytest.mark.parametrize("curve_type", ["linecurve", "curve"])
    def test_period_historic_fixings_series_missing_warns(
        self,
        curve_type,
        line_curve,
        rfr_curve,
    ) -> None:
        #
        # This test modified by PR 357. The warning is still produced but the code also now
        # later errors due to the missing fixing and no forecasting method.
        #

        curve = rfr_curve if curve_type == "curve" else line_curve
        fixings = Series([4.0, 3.0, 2.5], index=[dt(1995, 12, 1), dt(2021, 12, 30), dt(2022, 1, 1)])
        period = FloatPeriod(
            start=dt(2021, 12, 30),
            end=dt(2022, 1, 3),
            payment=dt(2022, 1, 3),
            frequency="Q",
            fixing_method="rfr_payment_delay",
            float_spread=100,
            fixings=fixings,
            convention="act365F",
        )
        # expected = ((1 + 0.015 / 365) * (1 + 0.025 / 365) * (1 + 0.01 / 365) * (
        #             1 + 0.02 / 365) - 1) * 36500 / 4 + 1

        # with pytest.warns(UserWarning):
        #     period.rate(curve)

        with (
            pytest.raises(ValueError, match="RFRs could not be calculated, have you missed"),
            pytest.warns(UserWarning, match="`fixings` has missed a calendar value"),
        ):
            period.rate(curve)

    def test_more_fixings_than_expected_by_calednar_raises(self):
        # Create historical fixings spanning 5 days for a FloatPeriod.
        # But set a Cal that does not expect all of these - one holdiay midweek.
        # Observe the rate calculation.
        fixings = Series(
            data=[1.0, 2.0, 3.0, 4.0, 5.0],
            index=[
                dt(2023, 1, 23),
                dt(2023, 1, 24),
                dt(2023, 1, 25),
                dt(2023, 1, 26),
                dt(2023, 1, 27),
            ],
        )
        cal = Cal(holidays=[dt(2023, 1, 25)], week_mask=[5, 6])
        period = FloatPeriod(
            start=dt(2023, 1, 23),
            end=dt(2023, 1, 30),
            payment=dt(2023, 1, 30),
            frequency="Q",
            fixing_method="rfr_payment_delay",
            fixings=fixings,
            convention="act365F",
            calendar="bus",
        )
        curve = Curve({dt(2023, 1, 26): 1.0, dt(2025, 1, 26): 1.0}, calendar=cal)
        msg = "The supplied `fixings` contain more fixings than were expected"
        with pytest.raises(ValueError, match=msg):
            period.rate(curve)

    def test_fewer_fixings_than_expected_by_calendar_warns_and_fills(self):
        # Create historical fixings spanning 4 days for a FloatPeriod, with mid-week holiday
        # But set a Cal that expects 5 (the cal does not have the holiday)
        # Observe the rate calculation.

        # this tests performs a minimal version of test_period_historic_fixings_series_missing_warns
        fixings = Series(
            data=[1.0, 2.0, 4.0, 5.0],
            index=[dt(2023, 1, 23), dt(2023, 1, 24), dt(2023, 1, 26), dt(2023, 1, 27)],
        )
        cal = Cal(holidays=[], week_mask=[5, 6])
        period = FloatPeriod(
            start=dt(2023, 1, 23),
            end=dt(2023, 1, 30),
            payment=dt(2023, 1, 30),
            frequency="Q",
            fixing_method="rfr_payment_delay",
            fixings=fixings,
            convention="act365F",
            calendar="bus",
        )
        curve = Curve({dt(2023, 1, 29): 1.0, dt(2025, 1, 22): 1.0}, calendar=cal)
        with (
            pytest.raises(ValueError, match="RFRs could not be calculated, have you missed"),
            pytest.warns(UserWarning, match="`fixings` has missed a calendar value"),
        ):
            period.rate(curve)

    def test_fixing_with_float_spread_warning(self, curve) -> None:
        float_period = FloatPeriod(
            start=dt(2022, 1, 4),
            end=dt(2022, 4, 4),
            payment=dt(2022, 4, 4),
            frequency="Q",
            fixing_method="rfr_payment_delay",
            spread_compound_method="isda_compounding",
            float_spread=100,
            fixings=1.0,
        )
        with pytest.warns(UserWarning):
            result = float_period.rate(curve)
        assert result == 2.0

    def test_float_period_fixings_list_raises_on_ibor(self, curve, line_curve) -> None:
        with pytest.raises(ValueError, match="`fixings` cannot be supplied as list,"):
            FloatPeriod(
                start=dt(2022, 1, 4),
                end=dt(2022, 4, 4),
                payment=dt(2022, 4, 4),
                frequency="Q",
                fixing_method="ibor",
                method_param=2,
                fixings=[1.00],
            )

        float_period = FloatPeriod(
            start=dt(2022, 1, 4),
            end=dt(2022, 4, 4),
            payment=dt(2022, 4, 4),
            frequency="Q",
            fixing_method="ibor",
            method_param=2,
        )
        float_period.fixings = [1.0]
        with pytest.raises(ValueError, match="`fixings` cannot be supplied as list,"):
            float_period.rate(curve)
        with pytest.raises(ValueError, match="`fixings` cannot be supplied as list,"):
            float_period.rate(line_curve)

    @pytest.mark.parametrize(
        ("meth", "exp"),
        [
            (
                "rfr_payment_delay",
                DataFrame(
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
                        "risk": [0.0, 0.0, 0.0, -0.26664737262, -0.26664737262],
                        "dcf": [0.0027777777777777778] * 5,
                        "rates": [1.19, 1.19, -8.81, 4.01364, 4.01364],
                    },
                ).set_index("obs_dates"),
            ),
            (
                "rfr_payment_delay_avg",
                DataFrame(
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
                            -999888.52252,
                            -1000000.00000,
                        ],
                        "risk": [0.0, 0.0, 0.0, -0.26666528084917104, -0.26666528084917104],
                        "dcf": [0.0027777777777777778] * 5,
                        "rates": [1.19, 1.19, -8.81, 4.01364, 4.01364],
                    },
                ).set_index("obs_dates"),
            ),
        ],
    )
    def test_rfr_fixings_table(self, curve, meth, exp) -> None:
        exp.columns = MultiIndex.from_tuples(
            [(curve.id, "notional"), (curve.id, "risk"), (curve.id, "dcf"), (curve.id, "rates")]
        )
        float_period = FloatPeriod(
            start=dt(2022, 12, 28),
            end=dt(2023, 1, 2),
            payment=dt(2023, 1, 2),
            frequency="M",
            fixings=[1.19, 1.19, -8.81],
            fixing_method=meth,
        )
        result = float_period.fixings_table(curve)
        assert_frame_equal(result, exp, rtol=1e-4)

        curve._set_ad_order(order=1)
        # assert values are unchanged even if curve can calculate derivatives
        result = float_period.fixings_table(curve)
        assert_frame_equal(result, exp)

    @pytest.mark.parametrize(
        ("right", "exp"),
        [
            (dt(2021, 1, 1), 0),
            (dt(2022, 12, 31), 4),
        ],
    )
    def test_rfr_fixings_table_right(self, curve, right, exp) -> None:
        float_period = FloatPeriod(
            start=dt(2022, 12, 28),
            end=dt(2023, 1, 2),
            payment=dt(2023, 1, 2),
            frequency="M",
            fixings=[1.19, 1.19, -8.81],
            fixing_method="rfr_payment_delay",
        )
        result = float_period.fixings_table(curve, right=right)
        assert isinstance(result, DataFrame)
        assert len(result.index) == exp

    def test_rfr_fixings_table_right_non_bus_day(self) -> None:
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2022, 11, 19): 0.98}, calendar="tgt")
        float_period = FloatPeriod(
            start=dt(2022, 2, 1),
            end=dt(2022, 2, 28),
            payment=dt(2022, 2, 28),
            frequency="M",
            fixing_method="rfr_payment_delay",
        )
        result = float_period.fixings_table(curve, right=dt(2022, 2, 13))
        assert isinstance(result, DataFrame)
        assert len(result.index) == 9

    @pytest.mark.parametrize(
        ("method", "param"),
        [
            ("rfr_payment_delay", NoInput(0)),
            ("rfr_lookback", 4),
            ("rfr_lockout", 1),
            ("rfr_observation_shift", 2),
        ],
    )
    @pytest.mark.parametrize(
        ("scm", "spd"),
        [
            ("none_simple", 1000.0),
            ("isda_compounding", 1000.0),
            ("isda_flat_compounding", 1000.0),
        ],
    )
    @pytest.mark.parametrize(
        "crv",
        [
            Curve(
                {
                    dt(2022, 1, 1): 1.00,
                    dt(2022, 4, 1): 0.99,
                    dt(2022, 7, 1): 0.98,
                    dt(2022, 10, 1): 0.97,
                    dt(2023, 6, 1): 0.96,
                },
                interpolation="log_linear",
                calendar="bus",
            ),
        ],
    )
    def test_rfr_fixings_table_fast(self, method, param, scm, spd, crv) -> None:
        float_period = FloatPeriod(
            start=dt(2022, 12, 28),
            end=dt(2023, 1, 3),
            payment=dt(2023, 1, 3),
            frequency="M",
            fixing_method=method,
            method_param=param,
            spread_compound_method=scm,
            float_spread=spd,
        )
        expected = float_period.fixings_table(crv)
        result = float_period.fixings_table(crv, approximate=True)
        assert_frame_equal(result, expected, rtol=1e-2)

    @pytest.mark.parametrize(
        "right",
        [
            dt(2022, 12, 31),
            dt(2021, 1, 1),
        ],
    )
    def test_rfr_fixings_table_fast_right(self, curve, right) -> None:
        float_period = FloatPeriod(
            start=dt(2022, 12, 28),
            end=dt(2023, 1, 3),
            payment=dt(2023, 1, 3),
            frequency="M",
            fixing_method="rfr_payment_delay",
        )
        expected = float_period.fixings_table(curve, right=right)
        result = float_period.fixings_table(curve, approximate=True, right=right)
        assert_frame_equal(result, expected, rtol=1e-2, check_dtype=False)

    @pytest.mark.parametrize(
        ("method", "param"),
        [
            ("rfr_payment_delay_avg", None),
            ("rfr_lookback_avg", 4),
            ("rfr_lockout_avg", 1),
            ("rfr_observation_shift_avg", 2),
        ],
    )
    @pytest.mark.parametrize(
        "crv",
        [
            Curve(
                {
                    dt(2022, 1, 1): 1.00,
                    dt(2022, 4, 1): 0.99,
                    dt(2022, 7, 1): 0.98,
                    dt(2022, 10, 1): 0.97,
                    dt(2023, 6, 1): 0.96,
                },
                interpolation="log_linear",
                calendar="bus",
            ),
        ],
    )
    def test_rfr_fixings_table_fast_avg(self, method, param, crv) -> None:
        float_period = FloatPeriod(
            start=dt(2022, 12, 28),
            end=dt(2023, 1, 3),
            payment=dt(2023, 1, 3),
            frequency="M",
            fixing_method=method,
            method_param=param,
            spread_compound_method="none_simple",
            float_spread=100.0,
        )
        expected = float_period.fixings_table(crv)
        result = float_period.fixings_table(crv, approximate=True)
        assert_frame_equal(result, expected, rtol=1e-2)

    def test_rfr_rate_fixings_series_monotonic_error(self) -> None:
        nodes = {
            dt(2022, 1, 1): 1.00,
            dt(2022, 4, 1): 0.99,
            dt(2022, 7, 1): 0.98,
            dt(2022, 10, 1): 0.97,
        }
        curve = Curve(nodes=nodes, interpolation="log_linear")
        fixings = Series(
            [99, 2.25, 2.375, 2.5],
            index=[dt(1995, 12, 1), dt(2021, 12, 30), dt(2022, 12, 31), dt(2022, 1, 1)],
        )
        period = FloatPeriod(
            start=dt(2021, 12, 30),
            end=dt(2022, 1, 3),
            payment=dt(2022, 1, 3),
            frequency="Q",
            fixing_method="rfr_payment_delay",
            float_spread=100,
            fixings=fixings,
            convention="act365F",
        )
        with pytest.raises(ValueError, match="`fixings` as a Series"):
            period.rate(curve)

    @pytest.mark.parametrize(
        ("scm", "exp"),
        [
            ("none_simple", True),
            ("isda_compounding", False),
        ],
    )
    def test_float_spread_affects_fixing_exposure(self, scm, exp) -> None:
        nodes = {
            dt(2022, 1, 1): 1.00,
            dt(2022, 4, 1): 0.99,
            dt(2022, 7, 1): 0.98,
            dt(2022, 10, 1): 0.97,
        }
        curve = Curve(nodes=nodes, interpolation="log_linear")
        period = FloatPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 7, 1),
            payment=dt(2022, 7, 1),
            frequency="S",
            fixing_method="rfr_payment_delay",
            float_spread=0,
            convention="act365F",
            spread_compound_method=scm,
        )
        table = period.fixings_table(curve)
        period.float_spread = 200
        table2 = period.fixings_table(curve)
        assert (
            table[(curve.id, "notional")].iloc[0] == table2[(curve.id, "notional")].iloc[0]
        ) == exp

    def test_custom_interp_rate_nan(self) -> None:
        float_period = FloatPeriod(
            start=dt(2022, 12, 28),
            end=dt(2023, 1, 2),
            payment=dt(2023, 1, 2),
            frequency="M",
            fixings=[1.19, 1.19],
        )

        def interp(date, nodes):
            if date < dt(2023, 1, 1):
                return None
            return 2.0

        line_curve = LineCurve({dt(2023, 1, 1): 3.0, dt(2023, 2, 1): 2.0}, interpolation=interp)
        curve = Curve({dt(2023, 1, 1): 1.0, dt(2023, 2, 1): 0.999})
        with pytest.raises(ValueError, match="RFRs could not be calculated"):
            float_period.fixings_table(line_curve, disc_curve=curve)

    def test_method_param_raises(self) -> None:
        with pytest.raises(ValueError, match='`method_param` must be >0 for "rfr_lock'):
            FloatPeriod(
                start=dt(2022, 1, 4),
                end=dt(2022, 4, 4),
                payment=dt(2022, 4, 4),
                frequency="Q",
                fixing_method="rfr_lockout",
                method_param=0,
                fixings=[1.00],
            )

        with pytest.raises(ValueError, match="`method_param` should not be used"):
            FloatPeriod(
                start=dt(2022, 1, 4),
                end=dt(2022, 4, 4),
                payment=dt(2022, 4, 4),
                frequency="Q",
                fixing_method="rfr_payment_delay",
                method_param=2,
                fixings=[1.00],
            )

    def test_analytic_delta_no_curve_raises(self) -> None:
        float_period = FloatPeriod(
            start=dt(2022, 12, 28),
            end=dt(2023, 1, 2),
            payment=dt(2023, 1, 2),
            frequency="M",
            fixings=[1.19, 1.19, -8.81],
            spread_compound_method="isda_compounding",
            float_spread=1.0,
        )
        with pytest.raises(TypeError, match="`curve` must be supplied"):
            float_period.analytic_delta()

    def test_more_series_fixings_than_calendar_from_curve_raises(self) -> None:
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, calendar="bus")
        fixings = Series(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            index=[
                dt(2022, 1, 4),
                dt(2022, 1, 5),
                dt(2022, 1, 6),
                dt(2022, 1, 7),
                dt(2022, 1, 8),
                dt(2022, 1, 9),
                dt(2022, 1, 10),
            ],
        )
        period = FloatPeriod(
            start=dt(2022, 1, 4),
            end=dt(2022, 1, 11),
            frequency="Q",
            fixing_method="rfr_payment_delay",
            payment=dt(2022, 1, 9),
            float_spread=10.0,
            fixings=fixings,
        )
        with pytest.raises(ValueError, match="The supplied `fixings` contain more"):
            period.rate(curve)

    def test_series_fixings_not_applicable_to_period(self) -> None:
        # if a series is historic and of no relevance all fixings are forecast from crv
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99}, calendar="bus")
        fixings = Series([1.0, 2.0, 3.0], index=[dt(2021, 1, 4), dt(2021, 1, 5), dt(2021, 1, 6)])
        period = FloatPeriod(
            start=dt(2022, 1, 4),
            end=dt(2022, 1, 11),
            frequency="Q",
            fixing_method="rfr_payment_delay",
            payment=dt(2022, 1, 9),
            float_spread=10.0,
            fixings=fixings,
        )
        result = period.rate(curve)
        expected = 1.09136153  # series fixings are completely ignored
        assert abs(result - expected) < 1e-5

    @pytest.mark.parametrize(
        ("meth", "param", "exp"),
        [
            ("rfr_payment_delay", NoInput(0), 3.1183733605),
            ("rfr_observation_shift", 2, 3.085000395),
            ("rfr_lookback", 2, 3.05163645),
            ("rfr_lockout", 7, 3.00157855),
        ],
    )
    def test_norges_bank_nowa_calc_same(self, meth, param, exp) -> None:
        # https://app.norges-bank.no/nowa/#/en/
        curve = Curve({dt(2023, 8, 4): 1.0}, calendar="osl", convention="act365f")
        period = FloatPeriod(
            start=dt(2023, 4, 27),
            end=dt(2023, 5, 12),
            payment=dt(2023, 5, 16),
            frequency="A",
            fixing_method=meth,
            method_param=param,
            float_spread=0.0,
            fixings=defaults.fixings["nowa"],
        )
        result = period.rate(curve)
        assert abs(result - exp) < 1e-7

    def test_interpolated_ibor_warns(self) -> None:
        period = FloatPeriod(
            start=dt(2023, 4, 27),
            end=dt(2023, 6, 12),
            payment=dt(2023, 6, 16),
            frequency="A",
            fixing_method="ibor",
            method_param=1,
            float_spread=0.0,
            stub=True,
        )
        curve1 = LineCurve({dt(2022, 1, 1): 1.0, dt(2024, 2, 1): 1.0})
        with pytest.warns(UserWarning):
            period.rate({"1m": curve1})
        with pytest.warns(UserWarning):
            period.rate({"3m": curve1})

    def test_interpolated_ibor_rate_line(self) -> None:
        period = FloatPeriod(
            start=dt(2023, 2, 1),
            end=dt(2023, 4, 1),
            payment=dt(2023, 4, 1),
            frequency="A",
            fixing_method="ibor",
            method_param=1,
            float_spread=0.0,
            stub=True,
        )
        curve3 = LineCurve({dt(2022, 1, 1): 3.0, dt(2023, 2, 1): 3.0})
        curve1 = LineCurve({dt(2022, 1, 1): 1.0, dt(2023, 2, 1): 1.0})
        result = period.rate({"1M": curve1, "3m": curve3})
        expected = 1.0 + (3.0 - 1.0) * (dt(2023, 4, 1) - dt(2023, 3, 1)) / (
            dt(2023, 5, 1) - dt(2023, 3, 1)
        )
        assert abs(result - expected) < 1e-8

    def test_interpolated_ibor_rate_df(self) -> None:
        period = FloatPeriod(
            start=dt(2023, 2, 1),
            end=dt(2023, 4, 1),
            payment=dt(2023, 4, 1),
            frequency="A",
            fixing_method="ibor",
            method_param=1,
            float_spread=0.0,
            stub=True,
        )
        curve3 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 2, 1): 0.97})
        curve1 = Curve({dt(2022, 1, 1): 1.0, dt(2023, 2, 1): 0.99})
        result = period.rate({"1M": curve1, "3m": curve3})
        a, b = 0.91399161, 2.778518365
        expected = a + (b - a) * (dt(2023, 4, 1) - dt(2023, 3, 1)) / (
            dt(2023, 5, 1) - dt(2023, 3, 1)
        )
        assert abs(result - expected) < 1e-8

    def test_rfr_period_curve_dict_raises(self, curve) -> None:
        period = FloatPeriod(
            start=dt(2023, 2, 1),
            end=dt(2023, 4, 1),
            payment=dt(2023, 4, 1),
            frequency="A",
            fixing_method="rfr_payment_delay",
            float_spread=0.0,
            stub=True,
        )
        with pytest.raises(ValueError, match="A `curve` supplied as dict to an RFR based period m"):
            period.rate({"bad_index": curve})

    def test_rfr_period_curve_dict_allowed(self, curve) -> None:
        period = FloatPeriod(
            start=dt(2023, 2, 1),
            end=dt(2023, 4, 1),
            payment=dt(2023, 4, 1),
            frequency="A",
            fixing_method="rfr_payment_delay",
            float_spread=0.0,
            stub=True,
        )
        expected = 4.02664128485892
        result = period.rate({"rfr": curve})
        assert result == expected

    def test_ibor_stub_fixings_table(self) -> None:
        period = FloatPeriod(
            start=dt(2023, 2, 1),
            end=dt(2023, 4, 1),
            payment=dt(2023, 4, 1),
            frequency="A",
            fixing_method="ibor",
            method_param=1,
            float_spread=0.0,
            stub=True,
        )
        curve3 = LineCurve({dt(2022, 1, 1): 3.0, dt(2023, 2, 1): 3.0})
        curve1 = LineCurve({dt(2022, 1, 1): 1.0, dt(2023, 2, 1): 1.0})
        result = period.fixings_table({"1M": curve1, "3m": curve3}, disc_curve=curve1)
        assert isinstance(result, DataFrame)
        assert abs(result.iloc[0, 0] + 1036300) < 1
        assert abs(result.iloc[0, 4] + 336894) < 1
        assert abs(result.iloc[0, 1] + 8.0601) < 1e-4
        assert abs(result.iloc[0, 5] + 8.32877) < 1e-4

    def test_ibor_stub_fixings_table_right(self) -> None:
        period = FloatPeriod(
            start=dt(2023, 2, 1),
            end=dt(2023, 4, 1),
            payment=dt(2023, 4, 1),
            frequency="A",
            fixing_method="ibor",
            method_param=1,
            float_spread=0.0,
            stub=True,
        )
        curve3 = LineCurve({dt(2022, 1, 1): 3.0, dt(2023, 2, 1): 3.0})
        curve1 = LineCurve({dt(2022, 1, 1): 1.0, dt(2023, 2, 1): 1.0})
        result = period.fixings_table(
            {"1M": curve1, "3m": curve3}, disc_curve=curve1, right=dt(2022, 1, 1)
        )
        assert isinstance(result, DataFrame)
        assert len(result.index) == 0

    def test_ibor_non_stub_fixings_table(self) -> None:
        period = FloatPeriod(
            start=dt(2023, 2, 1),
            end=dt(2023, 5, 1),
            payment=dt(2023, 5, 1),
            frequency="q",
            fixing_method="ibor",
            method_param=1,
            float_spread=0.0,
        )
        curve3 = LineCurve({dt(2022, 1, 1): 3.0, dt(2023, 2, 1): 3.0})
        curve1 = LineCurve({dt(2022, 1, 1): 1.0, dt(2023, 2, 1): 1.0})
        result = period.fixings_table({"1M": curve1, "3M": curve3}, disc_curve=curve1)
        expected = DataFrame(
            data=[[-1e6, -24.722222222222, 0.24722222222222223, 3.0]],
            index=Index([dt(2023, 1, 31)], name="obs_dates"),
            columns=["notional", "risk", "dcf", "rates"],
        )
        expected.columns = MultiIndex.from_tuples(
            [(curve3.id, "notional"), (curve3.id, "risk"), (curve3.id, "dcf"), (curve3.id, "rates")]
        )
        assert_frame_equal(result, expected)

    def test_local_historical_pay_date_issue(self, curve) -> None:
        period = FloatPeriod(dt(2021, 1, 1), dt(2021, 4, 1), dt(2021, 4, 1), "Q")
        result = period.npv(curve, local=True)
        assert result == {"usd": 0.0}

    @pytest.mark.parametrize(
        "curve", [NoInput(0), LineCurve({dt(2000, 1, 1): 2.0, dt(2001, 1, 1): 2.0})]
    )
    @pytest.mark.parametrize("fixing_method", ["ibor", "rfr_payment_delay_avg"])
    @pytest.mark.parametrize("fixings", [3.0, NoInput(0)])
    def test_rate_optional_curve(self, fixings, fixing_method, curve) -> None:
        # GH530. Allow forecasting rates without necessarily providing curve if unnecessary
        period = FloatPeriod(
            start=dt(2000, 1, 12),
            end=dt(2000, 4, 12),
            fixing_method=fixing_method,
            frequency="q",
            fixings=fixings,
            payment=dt(2000, 4, 12),
        )
        if isinstance(curve, NoInput) and isinstance(fixings, NoInput):
            # then no data to price
            msg = "Must supply a `curve` to FloatPeriod.rate()"
            with pytest.raises(ValueError, match=msg):
                period.rate(curve)
        elif isinstance(fixings, NoInput):
            result = period.rate(curve)
            assert abs(result - 2.0) < 1e-8  # uses curve
        else:
            result = period.rate(curve)
            assert abs(result - 3.0) < 1e-8  # uses fixing

    @pytest.mark.parametrize(
        "fixings",
        [
            [2.0, 2.0, 2.0],  # some unknown
            [2.0] * 31,  # exhaustive
            Series(2.0, index=date_range(dt(2000, 1, 1), dt(2001, 1, 1))),
        ],
    )
    @pytest.mark.parametrize(
        "curve", [NoInput(0), LineCurve({dt(2000, 1, 1): 2.0, dt(2001, 1, 1): 2.0})]
    )
    def test_rate_optional_curve_rfr(self, curve, fixings) -> None:
        # GH530. Test RFR periods what happens when supply/not supply a Curve and fixings
        # are either exhaustive/ not exhaustive
        period = FloatPeriod(
            start=dt(2000, 1, 1),
            end=dt(2000, 2, 1),
            fixing_method="rfr_payment_delay_avg",
            frequency="m",
            calendar="all",
            fixings=fixings,
            payment=dt(2000, 2, 1),
        )

        # When a curve is not supplied for RFR period currently it will still fail
        # even if exhaustive fixings are available. There is currently no branching handling this.
        if isinstance(curve, NoInput) and len(fixings) == 3:
            with pytest.raises(ValueError, match="Must supply a `curve` to FloatPeriod.rate()"):
                period.rate(curve)
        else:
            # it will conclude without fail, the exhaustive case is captured.
            period.rate(curve)


class TestFixedPeriod:
    def test_fixed_period_analytic_delta(self, curve, fxr) -> None:
        fixed_period = FixedPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            currency="usd",
        )
        result = fixed_period.analytic_delta(curve)
        assert abs(result - 24744.478172244584) < 1e-7

        result = fixed_period.analytic_delta(curve, curve, fxr, "nok")
        assert abs(result - 247444.78172244584) < 1e-7

    def test_fixed_period_analytic_delta_fxr_base(self, curve, fxr) -> None:
        fixed_period = FixedPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            currency="usd",
        )
        fxr = FXRates({"usdnok": 10.0}, base="NOK")
        result = fixed_period.analytic_delta(curve, curve, fxr)
        assert abs(result - 247444.78172244584) < 1e-7

    @pytest.mark.parametrize(
        ("rate", "crv", "fx"),
        [
            (4.00, True, 2.0),
            (NoInput(0), False, 2.0),
            (4.00, True, 10),
            (NoInput(0), False, 10),
        ],
    )
    def test_fixed_period_cashflows(self, curve, fxr, rate, crv, fx) -> None:
        # also test the inputs to fx as float and as FXRates (10 is for
        fixed_period = FixedPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            fixed_rate=rate,
        )

        cashflow = None if rate is NoInput.blank else rate * -1e9 * fixed_period.dcf / 100
        expected = {
            defaults.headers["type"]: "FixedPeriod",
            defaults.headers["stub_type"]: "Regular",
            defaults.headers["a_acc_start"]: dt(2022, 1, 1),
            defaults.headers["a_acc_end"]: dt(2022, 4, 1),
            defaults.headers["payment"]: dt(2022, 4, 3),
            defaults.headers["notional"]: 1e9,
            defaults.headers["currency"]: "USD",
            defaults.headers["convention"]: "Act360",
            defaults.headers["dcf"]: fixed_period.dcf,
            defaults.headers["df"]: 0.9897791268897856 if crv else None,
            defaults.headers["rate"]: rate,
            defaults.headers["spread"]: None,
            defaults.headers["npv"]: -9897791.268897856 if crv else None,
            defaults.headers["cashflow"]: cashflow,
            defaults.headers["fx"]: fx,
            defaults.headers["npv_fx"]: -9897791.268897855 * fx if crv else None,
            defaults.headers["collateral"]: None,
        }
        if fx == 2.0:
            with pytest.warns(UserWarning):
                # supplying `fx` as numeric
                result = fixed_period.cashflows(
                    curve if crv else NoInput(0),
                    fx=2.0,
                    base=NoInput(0),
                )
        else:
            result = fixed_period.cashflows(curve if crv else NoInput(0), fx=fxr, base="nok")
        assert result == expected

    def test_fixed_period_npv(self, curve, fxr) -> None:
        fixed_period = FixedPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            fixed_rate=4.00,
            currency="usd",
        )
        result = fixed_period.npv(curve)
        assert abs(result + 9897791.268897833) < 1e-7

        result = fixed_period.npv(curve, curve, fxr, "nok")
        assert abs(result + 98977912.68897833) < 1e-6

    def test_fixed_period_npv_raises(self, curve) -> None:
        fixed_period = FixedPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            fixed_rate=4.00,
            currency="usd",
        )
        with pytest.raises(
            TypeError,
            match=re.escape("`curves` have not been supplied correctly."),
        ):
            fixed_period.npv()

    def test_npv_no_fixed_rate(self, curve):
        period = FixedPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            currency="usd",
        )
        with pytest.raises(TypeError, match="`fixed_rate` must be set on the Period"):
            period.npv(curve)


class TestCreditPremiumPeriod:
    @pytest.mark.parametrize(
        ("accrued", "exp"), [(True, -9892843.47762896), (False, -9887893.477628957)]
    )
    def test_period_npv(self, hazard_curve, curve, fxr, accrued, exp) -> None:
        premium_period = CreditPremiumPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            fixed_rate=4.0,
            currency="usd",
            premium_accrued=accrued,
        )
        result = premium_period.npv(hazard_curve, curve)
        assert abs(result - exp) < 1e-7

        result = premium_period.npv(hazard_curve, curve, fxr, "nok")
        assert abs(result - exp * 10.0) < 1e-6

    def test_period_npv_raises(self, curve, hazard_curve) -> None:
        premium_period = CreditPremiumPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            fixed_rate=4.00,
            currency="usd",
        )
        with pytest.raises(
            TypeError,
            match=re.escape("`curves` have not been supplied correctly."),
        ):
            premium_period.npv(hazard_curve)
        with pytest.raises(
            TypeError,
            match=re.escape("`curves` have not been supplied correctly."),
        ):
            premium_period.npv(NoInput(0), curve)

    def test_period_npv_no_spread_raises(self, curve, hazard_curve) -> None:
        premium_period = CreditPremiumPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            currency="usd",
        )
        with pytest.raises(
            ValueError,
            match=re.escape("`fixed_rate` must be set as a value"),
        ):
            premium_period.npv(hazard_curve, curve)

    @pytest.mark.parametrize(
        ("accrued", "exp"), [(True, 24732.108694072398), (False, 24719.733694072398)]
    )
    def test_period_analytic_delta(self, hazard_curve, curve, fxr, accrued, exp) -> None:
        premium_period = CreditPremiumPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            fixed_rate=4.00,
            currency="usd",
            premium_accrued=accrued,
        )
        result = premium_period.analytic_delta(hazard_curve, curve)
        assert abs(result - exp) < 1e-7

        result = premium_period.analytic_delta(hazard_curve, curve, fxr, "nok")
        assert abs(result - exp * 10.0) < 1e-7

    def test_period_analytic_delta_fxr_base(self, hazard_curve, curve, fxr) -> None:
        premium_period = CreditPremiumPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            fixed_rate=4.00,
            currency="usd",
        )
        fxr = FXRates({"usdnok": 10.0}, base="NOK")
        result = premium_period.analytic_delta(hazard_curve, curve, fxr)
        assert abs(result - 247321.086941) < 1e-6

    def test_period_cashflows(self, hazard_curve, curve, fxr) -> None:
        # also test the inputs to fx as float and as FXRates (10 is for
        premium_period = CreditPremiumPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            fixed_rate=4.00,
            currency="usd",
        )

        cashflow = 400 * -1e9 * premium_period.dcf / 10000
        expected = {
            defaults.headers["type"]: "CreditPremiumPeriod",
            defaults.headers["stub_type"]: "Regular",
            defaults.headers["a_acc_start"]: dt(2022, 1, 1),
            defaults.headers["a_acc_end"]: dt(2022, 4, 1),
            defaults.headers["payment"]: dt(2022, 4, 3),
            defaults.headers["notional"]: 1e9,
            defaults.headers["currency"]: "USD",
            defaults.headers["convention"]: "Act360",
            defaults.headers["dcf"]: premium_period.dcf,
            defaults.headers["df"]: 0.9897791268897856,
            defaults.headers["rate"]: 4.0,
            defaults.headers["survival"]: 0.999,
            defaults.headers["npv"]: -9892843.47762896,
            defaults.headers["cashflow"]: cashflow,
            defaults.headers["fx"]: 10.0,
            defaults.headers["npv_fx"]: -9892843.47762896 * 10.0,
            defaults.headers["collateral"]: None,
        }
        result = premium_period.cashflows(hazard_curve, curve, fx=fxr, base="nok")
        assert result == expected

    def test_period_cashflows_no_curves(self, fxr) -> None:
        # also test the inputs to fx as float and as FXRates (10 is for
        premium_period = CreditPremiumPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            fixed_rate=4.00,
            currency="usd",
        )

        cashflow = 400 * -1e9 * premium_period.dcf / 10000
        expected = {
            defaults.headers["type"]: "CreditPremiumPeriod",
            defaults.headers["stub_type"]: "Regular",
            defaults.headers["a_acc_start"]: dt(2022, 1, 1),
            defaults.headers["a_acc_end"]: dt(2022, 4, 1),
            defaults.headers["payment"]: dt(2022, 4, 3),
            defaults.headers["notional"]: 1e9,
            defaults.headers["currency"]: "USD",
            defaults.headers["convention"]: "Act360",
            defaults.headers["dcf"]: premium_period.dcf,
            defaults.headers["df"]: None,
            defaults.headers["rate"]: 4.0,
            defaults.headers["survival"]: None,
            defaults.headers["npv"]: None,
            defaults.headers["cashflow"]: cashflow,
            defaults.headers["fx"]: 10.0,
            defaults.headers["npv_fx"]: None,
            defaults.headers["collateral"]: None,
        }
        result = premium_period.cashflows(fx=fxr, base="nok")
        assert result == expected

    def test_mid_period_accrued(self, hazard_curve, curve):
        p1 = CreditPremiumPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="ActActICMA",
            termination=dt(2022, 4, 1),
            frequency="Q",
            fixed_rate=4.00,
            currency="usd",
        )
        p2 = CreditPremiumPeriod(
            start=dt(2021, 10, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="ActActICMA",
            termination=dt(2022, 4, 1),
            frequency="S",
            fixed_rate=2.00,
            currency="usd",
        )
        r1 = p1.npv(hazard_curve, curve)
        r2 = p2.npv(hazard_curve, curve)

        assert 2505 > r1 - r2 > 2500

    def test_null_cashflow(self):
        premium_period = CreditPremiumPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            currency="usd",
        )
        assert premium_period.cashflow is None

    def test_no_accrued(self):
        premium_period = CreditPremiumPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            currency="usd",
        )
        assert premium_period.accrued(dt(2022, 2, 1)) is None

    def test_accrued_out_of_range(self):
        premium_period = CreditPremiumPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            currency="usd",
            fixed_rate=2.0,
        )
        assert premium_period.accrued(dt(2022, 9, 1)) == 0.0
        assert premium_period.accrued(dt(2021, 9, 1)) == 0.0

    def test_accrued(self):
        premium_period = CreditPremiumPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="ActActICMA",
            termination=dt(2022, 4, 1),
            frequency="Q",
            currency="usd",
            fixed_rate=2.0,
        )
        assert abs(premium_period.accrued(dt(2022, 2, 1)) - (-1e9 * 0.25 * 31 / 90 * 0.02)) < 1e-9


class TestCreditProtectionPeriod:
    def test_period_npv(self, hazard_curve, curve, fxr) -> None:
        period = CreditProtectionPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            currency="usd",
        )
        period.discretization = 1
        result = period.npv(hazard_curve, curve)  # discounted properly this is -596962.1422873045
        assert abs(result - -596962.1422873045) < 34

        period.discretization = 23
        result = period.npv(hazard_curve, curve)
        exp = -596995.7591843301
        assert abs(result - exp) < 1e-7

        result = period.npv(hazard_curve, curve, fxr, "nok")
        assert abs(result - exp * 10.0) < 1e-6

    def test_period_npv_raises(self, curve, hazard_curve) -> None:
        period = CreditProtectionPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            currency="usd",
        )
        with pytest.raises(
            TypeError,
            match=re.escape("`curves` have not been supplied correctly."),
        ):
            period.npv(hazard_curve)
        with pytest.raises(
            TypeError,
            match=re.escape("`curves` have not been supplied correctly."),
        ):
            period.npv(NoInput(0), curve)

    def test_period_analytic_delta(self, hazard_curve, curve, fxr) -> None:
        period = CreditProtectionPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            currency="usd",
        )
        result = period.analytic_delta(hazard_curve, curve)
        assert abs(result - 0.0) < 1e-7

        result = period.analytic_delta(hazard_curve, curve, fxr, "nok")
        assert abs(result - 0.0 * 10.0) < 1e-7

    def test_period_analytic_delta_fxr_base(self, hazard_curve, curve, fxr) -> None:
        period = CreditProtectionPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            currency="usd",
        )
        fxr = FXRates({"usdnok": 10.0}, base="NOK")
        result = period.analytic_delta(hazard_curve, curve, fxr)
        assert abs(result - 0.0) < 1e-7

    def test_period_cashflows(self, hazard_curve, curve, fxr) -> None:
        # also test the inputs to fx as float and as FXRates (10 is for
        period = CreditProtectionPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            currency="usd",
        )

        cashflow = -period.notional * (1 - hazard_curve.meta.credit_recovery_rate)
        expected = {
            defaults.headers["type"]: "CreditProtectionPeriod",
            defaults.headers["stub_type"]: "Regular",
            defaults.headers["a_acc_start"]: dt(2022, 1, 1),
            defaults.headers["a_acc_end"]: dt(2022, 4, 1),
            defaults.headers["payment"]: dt(2022, 4, 3),
            defaults.headers["notional"]: 1e9,
            defaults.headers["currency"]: "USD",
            defaults.headers["convention"]: "Act360",
            defaults.headers["dcf"]: period.dcf,
            defaults.headers["df"]: 0.9897791268897856,
            defaults.headers["recovery"]: 0.4,
            defaults.headers["survival"]: 0.999,
            defaults.headers["npv"]: -596995.7591843301,
            defaults.headers["cashflow"]: cashflow,
            defaults.headers["fx"]: 10.0,
            defaults.headers["npv_fx"]: -596995.7591843301 * 10.0,
            defaults.headers["collateral"]: None,
        }
        result = period.cashflows(hazard_curve, curve, fx=fxr, base="nok")

        for key in expected:
            assert key in result
            assert result[key] == expected[key] or abs(result[key] - expected[key]) < 1e-6

    def test_period_cashflows_no_curves(self, fxr) -> None:
        # also test the inputs to fx as float and as FXRates (10 is for
        period = CreditProtectionPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            currency="usd",
        )
        cashflow = None
        expected = {
            defaults.headers["type"]: "CreditProtectionPeriod",
            defaults.headers["stub_type"]: "Regular",
            defaults.headers["a_acc_start"]: dt(2022, 1, 1),
            defaults.headers["a_acc_end"]: dt(2022, 4, 1),
            defaults.headers["payment"]: dt(2022, 4, 3),
            defaults.headers["notional"]: 1e9,
            defaults.headers["currency"]: "USD",
            defaults.headers["convention"]: "Act360",
            defaults.headers["dcf"]: period.dcf,
            defaults.headers["df"]: None,
            defaults.headers["recovery"]: None,
            defaults.headers["survival"]: None,
            defaults.headers["npv"]: None,
            defaults.headers["cashflow"]: cashflow,
            defaults.headers["fx"]: 10.0,
            defaults.headers["npv_fx"]: None,
            defaults.headers["collateral"]: None,
        }
        result = period.cashflows(fx=fxr, base="nok")
        assert result == expected

    def test_discretization_period(self, hazard_curve, curve):
        p1 = CreditProtectionPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 1),
            notional=1e9,
            frequency="Q",
        )
        h1 = hazard_curve.copy()
        h2 = hazard_curve.copy()
        h1._meta = replace(h1.meta, _credit_discretization=1)
        h2._meta = replace(h2.meta, _credit_discretization=31)
        r1 = p1.npv(h1, curve)
        r2 = p1.npv(h2, curve)
        assert 0.1 < abs(r1 - r2) < 1.0  # very similar result but not identical

    def test_mid_period(self, hazard_curve, curve):
        period = CreditProtectionPeriod(
            start=dt(2021, 10, 4),
            end=dt(2022, 1, 4),
            payment=dt(2022, 1, 4),
            notional=1e9,
            frequency="Q",
        )
        r1 = period.npv(hazard_curve, curve)
        exp = -20006.321837529074
        assert abs(r1 - exp) < 1e-7

    def test_recovery_risk(self, hazard_curve, curve):
        period = CreditProtectionPeriod(
            start=dt(2021, 10, 4),
            end=dt(2022, 1, 4),
            payment=dt(2022, 1, 4),
            notional=1e9,
            frequency="Q",
        )

        result = period.analytic_rec_risk(hazard_curve, curve)
        p1 = period.npv(hazard_curve, curve)
        hazard_curve.update_meta("credit_recovery_rate", 0.41)
        p2 = period.npv(hazard_curve, curve)
        expected = p2 - p1
        assert abs(result - expected) < 1e-9


class TestCashflow:
    def test_cashflow_analytic_delta(self, curve) -> None:
        cashflow = Cashflow(notional=1e6, payment=dt(2022, 1, 1))
        assert cashflow.analytic_delta(curve) == 0

    @pytest.mark.parametrize(
        ("crv", "fx"),
        [
            (True, 2.0),
            (False, 2.0),
            (True, 10.0),
            (False, 10.0),
        ],
    )
    def test_cashflow_cashflows(self, curve, fxr, crv, fx) -> None:
        cashflow = Cashflow(notional=1e9, payment=dt(2022, 4, 3))
        curve = curve if crv else NoInput(0)
        expected = {
            defaults.headers["type"]: "Cashflow",
            defaults.headers["stub_type"]: None,
            # defaults.headers["a_acc_start"]: None,
            # defaults.headers["a_acc_end"]: None,
            defaults.headers["payment"]: dt(2022, 4, 3),
            defaults.headers["currency"]: "USD",
            defaults.headers["notional"]: 1e9,
            # defaults.headers["convention"]: None,
            # defaults.headers["dcf"]: None,
            defaults.headers["df"]: 0.9897791268897856 if crv else None,
            defaults.headers["rate"]: None,
            # defaults.headers["spread"]: None,
            defaults.headers["npv"]: -989779126.8897856 if crv else None,
            defaults.headers["cashflow"]: -1e9,
            defaults.headers["fx"]: fx,
            defaults.headers["npv_fx"]: -989779126.8897856 * fx if crv else None,
            defaults.headers["collateral"]: None,
        }
        if fx == 2.0:
            with pytest.warns(UserWarning):
                # supplying `fx` as numeric
                result = cashflow.cashflows(
                    curve if crv else NoInput(0),
                    fx=2.0,
                    base=NoInput(0),
                )
        else:
            result = cashflow.cashflows(
                curve if crv else NoInput(0),
                fx=fxr,
                base="nok",
            )
        assert result == expected

    def test_cashflow_npv_raises(self, curve) -> None:
        with pytest.raises(TypeError, match="`curves` have not been supplied correctly."):
            Cashflow(notional=1e6, payment=dt(2022, 1, 1)).npv()
        cashflow = Cashflow(notional=1e6, payment=dt(2022, 1, 1))
        assert cashflow.analytic_delta(curve) == 0

    def test_cashflow_npv_local(self, curve) -> None:
        cashflow = Cashflow(notional=1e9, payment=dt(2022, 4, 3), currency="nok")
        result = cashflow.npv(curve, local=True)
        expected = {"nok": -989779126.8897856}
        assert result == expected


class TestIndexFixedPeriod:
    @pytest.mark.parametrize(
        ("method", "expected"),
        [("daily", 201.00502512562812), ("monthly", 200.98317675333183)],
    )
    def test_period_rate(self, method, expected) -> None:
        index_period = IndexFixedPeriod(
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
            index_method=method,
        )
        index_curve = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 4, 3): 0.995},
            index_base=200.0,
            interpolation="linear_index",
            index_lag=3,
        )
        _, result, _ = index_period.index_ratio(index_curve)
        assert abs(result - expected) < 1e-8

    def test_period_cashflow(self) -> None:
        index_period = IndexFixedPeriod(
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
            index_lag=3,
        )
        index_curve = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 4, 3): 0.995},
            index_base=200.0,
            interpolation="linear_index",
            index_lag=3,
        )
        result = index_period.real_cashflow
        expected = -1e7 * ((dt(2022, 4, 1) - dt(2022, 1, 1)) / timedelta(days=360)) * 4
        assert abs(result - expected) < 1e-8

        result = index_period.cashflow(index_curve)
        expected = expected * index_curve.index_value(dt(2022, 4, 3), 3) / 100.0
        assert abs(result - expected) < 1e-8

    @pytest.mark.parametrize("method", ["daily", "curve"])
    def test_period_curve_interp_method(self, method) -> None:
        # both these methods of interpolation should give the same result with the way
        # the curve and period are configured.
        index_period = IndexFixedPeriod(
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
            index_lag=0,
            index_method=method,
        )
        index_curve = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 4, 3): 0.995},
            index_base=200.0,
            interpolation="linear_index",
            index_lag=0,
        )
        result = index_period.real_cashflow
        expected = -1e7 * ((dt(2022, 4, 1) - dt(2022, 1, 1)) / timedelta(days=360)) * 4
        assert abs(result - expected) < 1e-8

        result = index_period.cashflow(index_curve)
        assert abs(result + 20100502.512562) < 1e-6
        expected = expected * index_curve.index_value(dt(2022, 4, 3), 0) / 100.0
        assert abs(result - expected) < 1e-8

    def test_period_analytic_delta(self, fxr, curve) -> None:
        index_curve = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 4, 3): 0.995},
            index_base=200.0,
            interpolation="linear_index",
        )
        fixed_period = IndexFixedPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            currency="usd",
            index_base=200.0,
            index_fixings=300.0,
        )
        result = fixed_period.analytic_delta(index_curve, curve)
        assert abs(result - 24744.478172244584 * 300.0 / 200.0) < 1e-7

        result = fixed_period.analytic_delta(index_curve, curve, fxr, "nok")
        assert abs(result - 247444.78172244584 * 300.0 / 200.0) < 1e-7

    @pytest.mark.parametrize(("fixings", "method"), [(300.0, "daily")])
    def test_period_fixings_float(self, fixings, method, curve) -> None:
        fixed_period = IndexFixedPeriod(
            start=dt(2022, 1, 3),
            end=dt(2022, 4, 3),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 3),
            frequency="Q",
            currency="usd",
            index_base=200.0,
            index_fixings=fixings,
            index_method=method,
        )
        result = fixed_period.analytic_delta(None, curve)
        assert abs(result - 24744.478172244584 * 300.0 / 200.0) < 1e-7

    @pytest.mark.skip(reason="`index_fixings` as Series removed for Period in 2.0")
    @pytest.mark.parametrize(
        ("fixings", "method"),
        [
            (
                Series([1.0, 300, 5], index=[dt(2022, 4, 2), dt(2022, 4, 3), dt(2022, 4, 4)]),
                "daily",
            ),
            (Series([100.0, 500], index=[dt(2022, 4, 2), dt(2022, 4, 4)]), "daily"),
            (Series([300.0, 500], index=[dt(2022, 4, 1), dt(2022, 4, 5)]), "monthly"),
        ],
    )
    def test_period_fixings_series(self, fixings, method, curve) -> None:
        fixed_period = IndexFixedPeriod(
            start=dt(2022, 1, 3),
            end=dt(2022, 4, 3),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 3),
            frequency="Q",
            currency="usd",
            index_base=200.0,
            index_fixings=fixings,
            index_method=method,
        )
        result = fixed_period.analytic_delta(None, curve)
        assert abs(result - 24744.478172244584 * 300.0 / 200.0) < 1e-7

    def test_period_raises(self) -> None:
        with pytest.raises(ValueError, match="`index_method` must be "):
            IndexFixedPeriod(
                start=dt(2022, 1, 1),
                end=dt(2022, 4, 1),
                payment=dt(2022, 4, 3),
                notional=1e9,
                convention="Act360",
                termination=dt(2022, 4, 1),
                frequency="Q",
                currency="usd",
                index_base=200.0,
                index_method="BAD",
            )

    def test_period_npv(self, curve) -> None:
        index_period = IndexFixedPeriod(
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
            index_lag=3,
        )
        index_curve = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 4, 3): 0.995},
            index_base=200.0,
            interpolation="linear_index",
            index_lag=3,
        )
        result = index_period.npv(index_curve, curve)
        expected = -19895057.826930363
        assert abs(result - expected) < 1e-8

        result = index_period.npv(index_curve, curve, local=True)
        assert abs(result["usd"] - expected) < 1e-8

    def test_period_npv_raises(self) -> None:
        index_period = IndexFixedPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            fixed_rate=4.00,
            currency="usd",
            index_base=100.0,
        )
        with pytest.raises(
            TypeError,
            match=re.escape("`curves` have not been supplied correctly."),
        ):
            index_period.npv()

    @pytest.mark.parametrize("curve_", [True, False])
    def test_period_cashflows(self, curve, curve_) -> None:
        curve = curve if curve_ else NoInput(0)
        index_period = IndexFixedPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 4, 1),
            payment=dt(2022, 4, 3),
            notional=1e9,
            convention="Act360",
            termination=dt(2022, 4, 1),
            frequency="Q",
            fixed_rate=4.00,
            currency="usd",
            index_base=100.0,
            index_fixings=200.0,
        )
        result = index_period.cashflows(curve)
        expected = {
            "Type": "IndexFixedPeriod",
            "Period": "Regular",
            "Ccy": "USD",
            "Acc Start": dt(2022, 1, 1),
            "Acc End": dt(2022, 4, 1),
            "Payment": dt(2022, 4, 3),
            "Convention": "Act360",
            "DCF": 0.25,
            "DF": 0.9897791268897856 if curve_ else None,
            "Notional": 1e9,
            "Rate": 4.0,
            "Spread": None,
            "Cashflow": -20000000.0,
            "Real Cashflow": -10e6,
            "Index Base": 100.0,
            "Index Val": 200.0,
            "Index Ratio": 2.0,
            "NPV": -19795582.53779571 if curve_ else None,
            "FX Rate": 1.0,
            "NPV Ccy": -19795582.53779571 if curve_ else None,
            defaults.headers["collateral"]: None,
        }
        assert result == expected

    def test_cashflow_returns_none(self) -> None:
        i_period = IndexFixedPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 2, 1),
            payment=dt(2022, 2, 1),
            frequency="M",
            index_base=100.0,
        )
        assert i_period.cashflow() is None
        assert i_period.real_cashflow is None

    def test_cashflow_no_index_rate(self) -> None:
        i_period = IndexFixedPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 2, 1),
            payment=dt(2022, 2, 1),
            frequency="M",
            index_base=100.0,
        )
        result = i_period.cashflows()
        assert result[defaults.headers["index_ratio"]] is None

    def test_bad_curve(self) -> None:
        i_period = IndexFixedPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 2, 1),
            payment=dt(2022, 2, 1),
            frequency="M",
            index_base=100.0,
        )
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99})
        with pytest.raises(ValueError, match="Curve must be initialised with an `index_base`"):
            i_period.index_ratio(curve)

    @pytest.mark.skip(reason="`index_fixings` as Series removed for Period in 2.0")
    def test_index_fixings_linear_interp(self) -> None:
        i_fixings = Series([173.1, 174.2], index=[dt(2001, 7, 1), dt(2001, 8, 1)])
        result = IndexMixin._index_value(
            i_fixings=i_fixings,
            i_curve=None,
            i_date=dt(2001, 7, 20),
            i_lag=3,
            i_method="daily",
        )
        expected = 173.1 + 19 / 31 * (174.2 - 173.1)
        assert abs(result - expected) < 1e-6

    def test_composite_curve(self) -> None:
        index_period = IndexFixedPeriod(
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
        )
        index_curve = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 4, 3): 0.995},
            index_base=200.0,
            interpolation="linear_index",
        )
        composite_curve = CompositeCurve([index_curve])
        _, result, _ = index_period.index_ratio(composite_curve)

    def test_composite_curve_raises(self) -> None:
        index_period = IndexFixedPeriod(
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
        )
        curve = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 4, 3): 0.995},
        )
        composite_curve = CompositeCurve([curve])
        with pytest.raises(ValueError, match="Curve must be initialised with an `index_base`"):
            _, result, _ = index_period.index_ratio(composite_curve)

    @pytest.mark.parametrize(
        ("method", "expected"),
        [("daily", 201.00573790940518), ("monthly", 200.9836416123169)],
    )
    def test_index_lag_on_period_zero_curve(self, method, expected):
        # test if a period can calculate the correct value by referencing a curve with
        # zero index lag.
        index_period = IndexFixedPeriod(
            start=dt(2022, 1, 3),
            end=dt(2022, 4, 3),
            payment=dt(2022, 4, 3),
            notional=1e6,
            convention="30360",
            termination=dt(2022, 4, 3),
            frequency="Q",
            fixed_rate=4.00,
            currency="usd",
            index_base=100.0,
            index_method=method,
            index_lag=3,
        )
        index_curve = Curve(
            nodes={dt(2021, 10, 1): 1.0, dt(2022, 1, 3): 0.995},
            index_base=200.0,
            interpolation="linear_index",
            index_lag=0,
        )
        discount_curve = Curve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 4, 3): 0.99},
        )
        _, result, _ = index_period.index_ratio(index_curve)
        npv = index_period.npv(index_curve, discount_curve)
        assert abs(result - expected) < 1e-8
        expected_npv = -1e6 * 0.04 * 0.25 * result * 0.99 / 100.0
        assert abs(npv - expected_npv) < 1e-5

    def test_cashflows_available_with_series_fixings(self):
        RPI = DataFrame(
            [
                [dt(2024, 2, 1), 381.0],
                [dt(2024, 3, 1), 383.0],
                [dt(2024, 4, 1), 385.0],
                [dt(2024, 5, 1), 386.4],
                [dt(2024, 6, 1), 387.3],
                [dt(2024, 7, 1), 387.5],
                [dt(2024, 8, 1), 389.9],
                [dt(2024, 9, 1), 388.6],
                [dt(2024, 10, 1), 390.7],
                [dt(2024, 11, 1), 390.9],
                [dt(2024, 12, 1), 392.1],
                [dt(2025, 1, 1), 391.7],
                [dt(2025, 2, 1), 394.0],
                [dt(2025, 3, 1), 395.3],
            ],
            columns=["month", "rate"],
        ).set_index("month")["rate"]
        period = IndexFixedPeriod(
            start=dt(2024, 11, 27),
            end=dt(2025, 5, 27),
            fixed_rate=2.0,
            index_lag=3,
            index_fixings=RPI,
            index_base=RPI,
            frequency="S",
            payment=dt(2025, 5, 27),
        )
        result = period.cashflows()
        assert result["Index Base"] == 389.9 + (388.6 - 389.9) * (27 - 1) / 30
        assert result["Index Val"] == 394 + (395.3 - 394) * (27 - 1) / 31


class TestIndexCashflow:
    def test_cashflow_analytic_delta(self, curve) -> None:
        cashflow = IndexCashflow(notional=1e6, payment=dt(2022, 1, 1), index_base=100)
        assert cashflow.analytic_delta(curve) == 0

    def test_index_cashflow(self) -> None:
        cf = IndexCashflow(notional=1e6, payment=dt(2022, 1, 1), index_base=100, index_fixings=200)
        assert cf.real_cashflow == -1e6

        assert cf.cashflow(None) == -2e6

    def test_index_cashflow_npv(self, curve) -> None:
        cf = IndexCashflow(notional=1e6, payment=dt(2022, 1, 1), index_base=100, index_fixings=200)
        assert abs(cf.npv(curve) + 2e6) < 1e-6

    def test_cashflow_no_index_rate(self) -> None:
        i_period = IndexCashflow(
            notional=200.0,
            payment=dt(2022, 2, 1),
            index_base=100.0,
        )
        result = i_period.cashflows()
        assert result[defaults.headers["index_ratio"]] is None

    def test_index_only(self, curve) -> None:
        cf = IndexCashflow(
            notional=1e6,
            payment=dt(2022, 1, 1),
            index_base=100,
            index_fixings=200,
            index_only=True,
        )
        assert abs(cf.npv(curve) + 1e6) < 1e-6

    def test_index_cashflow_floats(self, curve) -> None:
        icurve = Curve(
            nodes={
                dt(2022, 1, 1): 1.00,
                dt(2022, 4, 1): 0.99,
                dt(2022, 7, 1): 0.98,
                dt(2022, 10, 1): 0.97,
            },
            index_base=100.0,
            interpolation="linear_index",
        )
        icurve._set_ad_order(1)
        curve._set_ad_order(1)
        cf = IndexCashflow(notional=1e6, payment=dt(2022, 7, 1), index_base=100)
        result = cf.cashflows(icurve, curve)
        assert isinstance(result["Cashflow"], float)


class TestNonDeliverableCashflow:
    @pytest.fixture(scope="class")
    def fxf_ndf(self):
        fxr = FXRates({"brlusd": 0.200}, settlement=dt(2025, 1, 23))
        fxf = FXForwards(
            fx_rates=fxr,
            fx_curves={
                "brlbrl": Curve({dt(2025, 1, 21): 1.0, dt(2026, 1, 23): 0.98}),
                "usdusd": Curve({dt(2025, 1, 21): 1.0, dt(2026, 1, 23): 0.96}),
                "brlusd": Curve({dt(2025, 1, 21): 1.0, dt(2026, 1, 23): 0.978}),
            },
        )
        return fxf

    def test_npv(self, fxf_ndf):
        ndf = NonDeliverableCashflow(
            notional=1e6,
            currency="brl",
            settlement_currency="usd",
            payment=dt(2025, 6, 1),
            fixing_date=dt(2025, 5, 29),
        )
        result = ndf.npv(disc_curve=fxf_ndf.curve("usd", "usd"), fx=fxf_ndf)
        expected = -1e6 * 0.20131018767289705 * 0.9855343095437953
        assert abs(result - expected) < 1e-8

    def test_npv_reversed(self, fxf_ndf):
        ndf = NonDeliverableCashflow(
            notional=1e6,
            currency="brl",
            settlement_currency="usd",
            payment=dt(2025, 6, 1),
            fixing_date=dt(2025, 5, 29),
            reversed=True,
        )
        result = ndf.npv(disc_curve=fxf_ndf.curve("usd", "usd"), fx=fxf_ndf)
        expected = -1e6 * 0.20131018767289705 * 0.9855343095437953
        assert abs(result - expected) < 1e-8

    def test_npv_fixing(self, fxf_ndf):
        ndf = NonDeliverableCashflow(
            notional=1e6,
            currency="brl",
            settlement_currency="usd",
            payment=dt(2025, 6, 1),
            fixing_date=dt(2025, 5, 29),
            fx_fixing=0.25,
        )
        result = ndf.npv(disc_curve=fxf_ndf.curve("usd", "usd"), fx=fxf_ndf)
        expected = -1e6 * 0.25 * 0.9855343095437953
        assert abs(result - expected) < 1e-8

    def test_rate_as_fixing(self, fxf_ndf):
        ndf = NonDeliverableCashflow(
            notional=1e6,
            currency="brl",
            settlement_currency="usd",
            payment=dt(2025, 6, 1),
            fixing_date=dt(2025, 5, 29),
            fx_fixing=0.25,
        )
        result = ndf.rate(fx=fxf_ndf)
        expected = 0.25
        assert abs(result - expected) < 1e-8

    def test_rate(self, fxf_ndf):
        ndf = NonDeliverableCashflow(
            notional=1e6,
            currency="brl",
            settlement_currency="usd",
            payment=dt(2025, 6, 1),
            fixing_date=dt(2025, 5, 29),
        )
        result = ndf.rate(fx=fxf_ndf)
        expected = fxf_ndf.rate(ndf.pair, dt(2025, 6, 1))
        assert abs(result - expected) < 1e-8

    def test_cashflows_priced(self, fxf_ndf):
        ndf = NonDeliverableCashflow(
            notional=1e6,
            currency="brl",
            settlement_currency="usd",
            payment=dt(2025, 6, 1),
            fixing_date=dt(2025, 5, 29),
            fx_fixing=0.25,
        )
        result = ndf.cashflows(disc_curve=fxf_ndf.curve("usd", "usd"), fx=fxf_ndf)
        expected = {
            "Cashflow": -250000.0,
            "Ccy": "USD",
            "Collateral": "usd",
            "DF": 0.9855343095437953,
            "FX Rate": 1.0,
            "NPV": -246383.57738594883,
            "NPV Ccy": -246383.57738594883,
            "Notional": 1000000.0,
            "Payment": dt(2025, 6, 1, 0, 0),
            "Period": "BRLUSD",
            "Rate": 0.25,
            "Type": "NonDeliverableCashflow",
        }
        assert result == expected

    def test_cashflows_no_args(self):
        ndf = NonDeliverableCashflow(
            notional=1e6,
            currency="brl",
            settlement_currency="usd",
            payment=dt(2025, 6, 1),
            fixing_date=dt(2025, 5, 29),
        )
        result = ndf.cashflows()
        expected = {
            "Cashflow": None,
            "Ccy": "USD",
            "Collateral": None,
            "DF": None,
            "FX Rate": 1.0,
            "NPV": None,
            "NPV Ccy": None,
            "Notional": 1000000.0,
            "Payment": dt(2025, 6, 1, 0, 0),
            "Period": "BRLUSD",
            "Rate": None,
            "Type": "NonDeliverableCashflow",
        }
        assert result == expected

    def test_analytic_delta(self):
        ndf = NonDeliverableCashflow(
            notional=1e6,
            currency="brl",
            settlement_currency="usd",
            payment=dt(2025, 6, 1),
            fixing_date=dt(2025, 5, 29),
            fx_fixing=0.25,
        )
        assert ndf.analytic_delta() == 0.0


class TestNonDeliverableFixedPeriod:
    @pytest.fixture(scope="class")
    def fxf_ndf(self):
        fxr = FXRates({"brlusd": 0.200}, settlement=dt(2025, 1, 23))
        fxf = FXForwards(
            fx_rates=fxr,
            fx_curves={
                "brlbrl": Curve({dt(2025, 1, 21): 1.0, dt(2026, 1, 23): 0.98}),
                "usdusd": Curve({dt(2025, 1, 21): 1.0, dt(2026, 1, 23): 0.96}),
                "brlusd": Curve({dt(2025, 1, 21): 1.0, dt(2026, 1, 23): 0.978}),
            },
        )
        return fxf

    @pytest.mark.parametrize("fx_fixing", [NoInput(0), 5.00])
    def test_cashflow_reversed(self, fx_fixing, fxf_ndf):
        ndfp = NonDeliverableFixedPeriod(
            start=dt(2025, 2, 1),
            end=dt(2025, 5, 1),
            payment=dt(2025, 5, 1),
            convention="30e360",
            currency="brl",
            settlement_currency="usd",
            notional=1e6,
            fx_fixing=fx_fixing,
            fx_fixing_date=dt(2025, 4, 29),
            frequency="q",
            fixed_rate=3.0,
            reversed=True,
        )
        cf = ndfp.cashflow(fx=fxf_ndf)
        fx_fixing = ndfp._get_fx_fixing(fx=fxf_ndf)
        expected = -1e6 * 0.25 * 0.03 / fx_fixing  # in USD
        assert abs(cf - expected) < 1e-8

    @pytest.mark.parametrize("fx_fixing", [NoInput(0), 0.2])
    def test_cashflow(self, fx_fixing, fxf_ndf):
        ndfp = NonDeliverableFixedPeriod(
            start=dt(2025, 2, 1),
            end=dt(2025, 5, 1),
            payment=dt(2025, 5, 1),
            convention="30e360",
            currency="brl",
            settlement_currency="usd",
            notional=0.2e6,
            fx_fixing=fx_fixing,
            fx_fixing_date=dt(2025, 4, 29),
            frequency="q",
            fixed_rate=3.0,
        )
        cf = ndfp.cashflow(fx=fxf_ndf)
        fx_fixing = ndfp._get_fx_fixing(fx=fxf_ndf)
        expected = -0.2e6 * 0.25 * 0.03 * fx_fixing  # in USD
        assert abs(cf - expected) < 1e-8

    @pytest.mark.parametrize("fx_fixing", [NoInput(0), 0.20])
    def test_no_cashflow(self, fx_fixing, fxf_ndf):
        ndfp = NonDeliverableFixedPeriod(
            start=dt(2025, 2, 1),
            end=dt(2025, 5, 1),
            payment=dt(2025, 5, 1),
            convention="30e360",
            settlement_currency="usd",
            currency="brl",
            notional=1e6,
            fx_fixing=fx_fixing,
            fx_fixing_date=dt(2025, 4, 29),
            frequency="q",
        )
        assert ndfp.cashflow(fx=fxf_ndf) is None

    @pytest.mark.parametrize("fx_fixing", [NoInput(0), 5.0])
    def test_analytic_delta(self, fx_fixing, fxf_ndf):
        ndfp = NonDeliverableFixedPeriod(
            start=dt(2025, 2, 1),
            end=dt(2025, 5, 1),
            payment=dt(2025, 5, 1),
            convention="30e360",
            currency="brl",
            settlement_currency="usd",
            notional=1e9,
            fx_fixing=fx_fixing,
            fx_fixing_date=dt(2025, 4, 29),
            frequency="q",
            fixed_rate=3.0,
            reversed=True,
        )
        curve = fxf_ndf.curve("usd", "usd")
        result = ndfp.analytic_delta(curve=curve, fx=fxf_ndf)
        fx_fixing = ndfp._get_fx_fixing(fx=fxf_ndf)
        expected = 1e9 * 0.25 * 0.0001 * curve[dt(2025, 5, 1)] / fx_fixing  # in USD
        assert abs(result - expected) < 1e-8

    @pytest.mark.parametrize("fx_conv", [FXRates({"usdeur": 105.0}), 105.0])
    def test_analytic_delta_base(self, fx_conv, fxf_ndf):
        ndfp = NonDeliverableFixedPeriod(
            start=dt(2025, 2, 1),
            end=dt(2025, 5, 1),
            payment=dt(2025, 5, 1),
            convention="30e360",
            currency="brl",
            settlement_currency="usd",
            notional=1e9,
            fx_fixing=5.0,
            fx_fixing_date=dt(2025, 4, 29),
            frequency="q",
            fixed_rate=3.0,
            reversed=True,
        )
        curve = fxf_ndf.curve("usd", "usd")
        result = ndfp.analytic_delta(curve=curve, fx=fx_conv, base="eur")
        fx_fixing = 5.0
        expected = 105 * 1e9 * 0.25 * 0.0001 * curve[dt(2025, 5, 1)] / fx_fixing  # in USD
        assert abs(result - expected) < 1e-8

    @pytest.mark.parametrize("fx_fixing", [NoInput(0), 5.0])
    def test_npv_reversed(self, fx_fixing, fxf_ndf):
        ndfp = NonDeliverableFixedPeriod(
            start=dt(2025, 2, 1),
            end=dt(2025, 5, 1),
            payment=dt(2025, 5, 1),
            convention="30e360",
            currency="brl",
            settlement_currency="usd",
            notional=1e9,
            fx_fixing=fx_fixing,
            fx_fixing_date=dt(2025, 4, 29),
            frequency="q",
            fixed_rate=3.0,
            reversed=True,
        )
        curve = fxf_ndf.curve("usd", "usd")
        result = ndfp.npv(curve=curve, fx=fxf_ndf)
        fx_fixing = ndfp._get_fx_fixing(fx=fxf_ndf)
        expected = -1e9 * 0.25 * 0.03 * curve[dt(2025, 5, 1)] / fx_fixing  # in USD
        assert abs(result - expected) < 1e-8

    @pytest.mark.parametrize("fx_fixing", [NoInput(0), 0.20])
    def test_npv(self, fx_fixing, fxf_ndf):
        ndfp = NonDeliverableFixedPeriod(
            start=dt(2025, 2, 1),
            end=dt(2025, 5, 1),
            payment=dt(2025, 5, 1),
            convention="30e360",
            currency="brl",
            settlement_currency="usd",
            notional=1e9,
            fx_fixing=fx_fixing,
            fx_fixing_date=dt(2025, 4, 29),
            frequency="q",
            fixed_rate=3.0,
        )
        curve = fxf_ndf.curve("usd", "usd")
        result = ndfp.npv(curve=curve, fx=fxf_ndf)
        fx_fixing = ndfp._get_fx_fixing(fx=fxf_ndf)
        expected = -1e9 * 0.25 * 0.03 * curve[dt(2025, 5, 1)] * fx_fixing  # in USD
        assert abs(result - expected) < 1e-8

    @pytest.mark.parametrize("curve", [True, False])
    @pytest.mark.parametrize("fixed_rate", [3.0])
    def test_cashflows(self, curve, fixed_rate, fxf_ndf):
        curve_ = fxf_ndf.curve("usd", "usd") if curve else NoInput(0)
        ndfp = NonDeliverableFixedPeriod(
            start=dt(2025, 2, 1),
            end=dt(2025, 5, 1),
            payment=dt(2025, 5, 1),
            convention="30e360",
            currency="brl",
            settlement_currency="usd",
            notional=1e9,
            fx_fixing=NoInput(0),
            fx_fixing_date=dt(2025, 4, 29),
            frequency="q",
            fixed_rate=fixed_rate,
        )
        result = ndfp.cashflows(curve=curve_, fx=fxf_ndf)
        expected = {
            "Acc End": dt(2025, 5, 1, 0, 0),
            "Acc Start": dt(2025, 2, 1, 0, 0),
            "Cashflow": -1507459.1627133065,
            "Ccy": "usd",
            "Collateral": "usd" if curve else None,
            "Convention": "30e360",
            "DCF": 0.25,
            "DF": 0.9889384743344495 if curve else None,
            "FX Rate": 1.0,
            "Index Val": 0.20099455502844088,
            "NPV": -1490784.364495184 if curve else None,
            "NPV Ccy": -1490784.364495184 if curve else None,
            "Notional": 1000000000.0,
            "Pair": "brlusd",
            "Payment": dt(2025, 5, 1, 0, 0),
            "Period": "Regular",
            "Rate": 3.0,
            "Spread": None,
            "Type": "NonDeliverableFixedPeriod",
        }
        assert result == expected


def test_base_period_dates_raise() -> None:
    with pytest.raises(ValueError):
        _ = FixedPeriod(dt(2023, 1, 1), dt(2022, 1, 1), dt(2024, 1, 1), "Q")


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


@pytest.fixture
def fxvs():
    vol_ = FXDeltaVolSmile(
        nodes={
            0.25: 8.9,
            0.5: 8.7,
            0.75: 10.15,
        },
        eval_date=dt(2023, 3, 16),
        expiry=dt(2023, 6, 16),
        delta_type="forward",
    )
    return vol_


class TestFXOption:
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
    @pytest.mark.parametrize("smile", [False, True])
    def test_premium_big_usd_pips(
        self,
        fxfo,
        fxvs,
        pay,
        k,
        exp_pts,
        exp_prem,
        dlty,
        exp_dl,
        smile,
    ) -> None:
        vol_ = (
            8.9
            if not smile
            else FXDeltaVolSmile(
                nodes={0.5: 8.9},
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type=dlty,
            )
        )
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=pay,
            strike=k,
            notional=20e6,
            delta_type=dlty,
        )
        result = fxo.rate(fxfo.curve("eur", "usd"), fxfo.curve("usd", "usd"), fx=fxfo, vol=vol_)
        expected = exp_pts
        assert abs(result - expected) < 1e-3

        result = 20e6 * result / 10000
        expected = exp_prem
        assert abs(result - expected) < 1e-2

        result = fxo.analytic_greeks(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=vol_,
        )["delta"]
        expected = exp_dl
        assert abs(result - expected) < 1e-6

    @pytest.mark.parametrize(
        ("pay", "k", "exp_pts", "exp_prem", "dlty", "exp_dl"),
        [
            (dt(2023, 3, 20), 1.101, 0.6536, 130717.44, "spot_pa", 0.243588),
            (dt(2023, 3, 20), 1.101, 0.6536, 130717.44, "forward_pa", 0.245175),
            (dt(2023, 6, 20), 1.101, 0.6578, 131569.29, "spot_pa", 0.243548),
            (dt(2023, 6, 20), 1.101, 0.6578, 131569.29, "forward_pa", 0.245178),
        ],
    )
    @pytest.mark.parametrize("smile", [False, True])
    def test_premium_big_eur_pc(self, fxfo, pay, k, exp_pts, exp_prem, dlty, exp_dl, smile) -> None:
        vol_ = (
            8.9
            if not smile
            else FXDeltaVolSmile(
                nodes={0.5: 8.9},
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type=dlty,
            )
        )
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=pay,
            strike=k,
            notional=20e6,
            delta_type=dlty,
            metric="percent",
        )
        result = fxo.rate(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=vol_,
        )
        expected = exp_pts
        assert abs(result - expected) < 1e-3

        result = 20e6 * result / 100
        expected = exp_prem
        assert abs(result - expected) < 1e-1

        result = fxo.analytic_greeks(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=vol_,
            premium=exp_prem,
        )["delta"]
        expected = exp_dl
        assert abs(result - expected) < 5e-5

    @pytest.mark.parametrize("smile", [False, True])
    def test_npv(self, fxfo, smile) -> None:
        vol_ = (
            8.9
            if not smile
            else FXDeltaVolSmile(
                nodes={0.5: 8.9},
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type="forward",
            )
        )
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=1.101,
            notional=20e6,
        )
        result = fxo.npv(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=vol_,
        )
        result /= fxfo.curve("usd", "usd")[dt(2023, 6, 20)]
        expected = 140451.5273  # 140500 USD premium according to Tullets calcs (may be rounded)
        assert abs(result - expected) < 1e-3

    @pytest.mark.parametrize("smile", [False, True])
    def test_npv_in_past(self, fxfo, smile) -> None:
        vol_ = (
            8.9
            if not smile
            else FXDeltaVolSmile(
                nodes={0.5: 8.9},
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type="forward",
            )
        )
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2022, 6, 16),
            delivery=dt(2022, 6, 20),
            payment=dt(2022, 6, 20),
            strike=1.101,
            notional=20e6,
        )
        result = fxo.npv(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=vol_,
        )
        assert result == 0.0

    def test_npv_option_fixing(self, fxfo) -> None:
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 3, 15),
            delivery=dt(2023, 3, 17),
            payment=dt(2023, 3, 17),
            strike=1.101,
            notional=20e6,
            option_fixing=1.102,
        )
        result = fxo.npv(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=8.9,
        )
        expected = (1.102 - 1.101) * 20e6 * fxfo.curve("usd", "usd")[dt(2023, 3, 17)]
        assert abs(result - expected) < 1e-9

        # valuable put
        fxo = FXPutPeriod(
            pair="eurusd",
            expiry=dt(2023, 3, 15),
            delivery=dt(2023, 3, 17),
            payment=dt(2023, 3, 17),
            strike=1.101,
            notional=20e6,
            option_fixing=1.100,
        )
        result = fxo.npv(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=8.9,
        )
        expected = (1.101 - 1.100) * 20e6 * fxfo.curve("usd", "usd")[dt(2023, 3, 17)]
        assert abs(result - expected) < 1e-9

        # worthless option
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 3, 15),
            delivery=dt(2023, 3, 17),
            payment=dt(2023, 3, 17),
            strike=1.101,
            notional=20e6,
            option_fixing=1.100,
        )
        result = fxo.npv(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=8.9,
        )
        expected = 0.0
        assert abs(result - expected) < 1e-9

    def test_rate_metric_raises(self, fxfo) -> None:
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=1.101,
            notional=20e6,
        )
        with pytest.raises(ValueError, match="`metric` must be in"):
            fxo.rate(
                fxfo.curve("eur", "usd"),
                fxfo.curve("usd", "usd"),
                fx=fxfo,
                vol=8.9,
                metric="bad",
            )

    @pytest.mark.parametrize("smile", [False, True])
    def test_premium_points(self, fxfo, smile) -> None:
        vol_ = (
            8.9
            if not smile
            else FXDeltaVolSmile(
                nodes={0.5: 8.9},
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type="forward",
            )
        )
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=1.101,
            notional=20e6,
        )
        result = fxo.rate(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=vol_,
        )
        expected = 70.225764  # 70.25 premium according to Tullets calcs (may be rounded)
        assert abs(result - expected) < 1e-6

    def test_implied_vol(self, fxfo) -> None:
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=1.101,
            notional=20e6,
        )
        result = fxo.implied_vol(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fx=fxfo,
            premium=70.25,
        )
        expected = 8.90141775  # Tullets have trade confo at 8.9%
        assert abs(expected - result) < 1e-8

        premium_pc = 0.007025 / fxfo.rate("eurusd", fxo.delivery) * 100.0
        result = fxo.implied_vol(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fx=fxfo,
            premium=premium_pc,
            metric="percent",
        )
        assert abs(expected - result) < 1e-8

    @pytest.mark.parametrize("smile", [False, True])
    def test_premium_put(self, fxfo, smile) -> None:
        vol_ = (
            10.15
            if not smile
            else FXDeltaVolSmile(
                nodes={0.5: 10.15},
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type="forward",
            )
        )
        fxo = FXPutPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=1.033,
            notional=20e6,
        )
        result = fxo.rate(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fxfo,
            vol=vol_,
        )
        expected = 83.836959  # Tullets trade confo has 83.75
        assert abs(result - expected) < 1e-6

    @pytest.mark.parametrize("smile", [False, True])
    def test_npv_put(self, fxfo, smile) -> None:
        vol_ = (
            10.15
            if not smile
            else FXDeltaVolSmile(
                nodes={0.5: 10.15},
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type="forward",
            )
        )
        fxo = FXPutPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=1.033,
            notional=20e6,
        )
        result = (
            fxo.npv(
                fxfo.curve("eur", "usd"),
                fxfo.curve("usd", "usd"),
                fxfo,
                vol=vol_,
            )
            / fxfo.curve("usd", "usd")[dt(2023, 6, 20)]
        )
        expected = 167673.917818  # Tullets trade confo has 167 500
        assert abs(result - expected) < 1e-6

    @pytest.mark.parametrize(
        ("dlty", "delta", "exp_k"),
        [
            ("forward", 0.25, 1.101271021340),
            ("forward_pa", 0.25, 1.10023348001),
            ("forward", 0.251754, 1.100999951),
            ("forward_pa", 0.8929, 0.9748614298),  # close to peak of premium adjusted delta graph.
            ("spot", 0.25, 1.10101920113408),
            ("spot_pa", 0.25, 1.099976469786),
            ("spot", 0.251754, 1.10074736155),
            ("spot_pa", 0.8870, 0.97543175409),  # close to peak of premium adjusted delta graph.
        ],
    )
    @pytest.mark.parametrize("smile", [False, True])
    def test_strike_from_delta(self, fxfo, dlty, delta, exp_k, smile) -> None:
        # https://quant.stackexchange.com/a/77802/29443
        vol_ = (
            8.9
            if not smile
            else FXDeltaVolSmile(
                nodes={0.5: 8.9},
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type=dlty,
            )
        )
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=1.101,
            notional=20e6,
            delta_type=dlty,
        )
        result = fxo._index_vol_and_strike_from_delta(
            delta,
            dlty,
            vol_,
            fxfo.curve("eur", "usd")[fxo.delivery],
            fxfo.curve("eur", "usd")[dt(2023, 3, 20)],
            fxfo.rate("eurusd", dt(2023, 6, 20)),
            fxo._t_to_expiry(fxfo.curve("usd", "usd").nodes.initial),
        )[2]
        expected = exp_k
        assert abs(result - expected) < 1e-8

        ## Round trip test
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=float(result),
            notional=20e6,
            delta_type=dlty,
        )
        result2 = fxo.analytic_greeks(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fxfo,
            vol=vol_,
        )["delta"]
        assert abs(result2 - delta) < 1e-8

    def test_payoff_at_expiry(self, fxfo) -> None:
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=1.101,
            notional=20e6,
        )
        result = fxo._payoff_at_expiry(rng=[1.07, 1.13])
        assert result[0][0] == 1.07
        assert result[0][-1] == 1.13
        assert result[1][0] == 0.0
        assert result[1][-1] == (1.13 - 1.101) * 20e6

    def test_payoff_at_expiry_put(self, fxfo) -> None:
        fxo = FXPutPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=1.101,
            notional=20e6,
        )
        result = fxo._payoff_at_expiry(rng=[1.07, 1.13])
        assert result[0][0] == 1.07
        assert result[0][-1] == 1.13
        assert result[1][0] == (1.101 - 1.07) * 20e6
        assert result[1][-1] == 0.0

    @pytest.mark.parametrize("delta_type", ["spot", "spot_pa", "forward", "forward_pa"])
    @pytest.mark.parametrize("smile_type", ["spot", "spot_pa", "forward", "forward_pa"])
    @pytest.mark.parametrize("delta", [-0.1, -0.25, -0.75, -0.9, -1.5])
    @pytest.mark.parametrize("vol_smile", [True, False])
    def test_strike_and_delta_idx_multisolve_from_delta_put(
        self,
        fxfo,
        delta_type,
        smile_type,
        delta,
        vol_smile,
    ) -> None:
        if delta < -1.0 and "_pa" not in delta_type:
            pytest.skip("Put delta cannot be below -1.0 in unadjusted cases.")
        fxo = FXPutPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=1.033,
            notional=20e6,
            delta_type=delta_type,
        )
        if vol_smile:
            vol_ = FXDeltaVolSmile(
                nodes={
                    0.25: 8.9,
                    0.5: 8.7,
                    0.75: 10.15,
                },
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type=smile_type,
            )
        else:
            vol_ = 9.00

        result = fxo._index_vol_and_strike_from_delta(
            delta,
            delta_type,
            vol_,
            fxfo.curve("eur", "usd")[dt(2023, 6, 20)],
            fxfo.curve("eur", "usd")[dt(2023, 3, 20)],
            fxfo.rate("eurusd", dt(2023, 6, 20)),
            fxo._t_to_expiry(fxfo.curve("eur", "usd").nodes.initial),
        )

        fxo.strike = result[2]

        if vol_smile:
            vol_ = result[1]

        expected = fxo.analytic_greeks(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=vol_,
        )["delta"]

        assert abs(delta - expected) < 1e-8

    @pytest.mark.parametrize("delta_type", ["spot", "spot_pa", "forward", "forward_pa"])
    @pytest.mark.parametrize("smile_type", ["spot", "spot_pa", "forward", "forward_pa"])
    @pytest.mark.parametrize("delta", [0.1, 0.25, 0.65, 0.9])
    @pytest.mark.parametrize("vol_smile", [True, False])
    def test_strike_and_delta_idx_multisolve_from_delta_call(
        self,
        fxfo,
        delta_type,
        smile_type,
        delta,
        vol_smile,
    ) -> None:
        if delta > 0.65 and "_pa" in delta_type:
            pytest.skip("Premium adjusted call delta cannot be above the peak ~0.7?.")
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=1.033,
            notional=20e6,
            delta_type=delta_type,
        )
        if vol_smile:
            vol_ = FXDeltaVolSmile(
                nodes={
                    0.25: 8.9,
                    0.5: 8.7,
                    0.75: 10.15,
                },
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type=smile_type,
            )
        else:
            vol_ = 9.00
        result = fxo._index_vol_and_strike_from_delta(
            delta,
            delta_type,
            vol_,
            fxfo.curve("eur", "usd")[dt(2023, 6, 20)],
            fxfo.curve("eur", "usd")[dt(2023, 3, 20)],
            fxfo.rate("eurusd", dt(2023, 6, 20)),
            fxo._t_to_expiry(fxfo.curve("eur", "usd").nodes.initial),
        )

        fxo.strike = result[2]
        if vol_smile:
            vol_ = result[1]

        expected = fxo.analytic_greeks(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=vol_,
        )["delta"]
        assert abs(delta - expected) < 1e-8

    @pytest.mark.parametrize("delta_type", ["spot_pa", "forward_pa"])
    @pytest.mark.parametrize("smile_type", ["spot", "spot_pa", "forward", "forward_pa"])
    @pytest.mark.parametrize("delta", [0.9])
    @pytest.mark.parametrize("vol_smile", [True, False])
    def test_strike_and_delta_idx_multisolve_from_delta_call_out_of_bounds(
        self,
        fxfo,
        delta_type,
        smile_type,
        delta,
        vol_smile,
    ) -> None:
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=1.033,
            notional=20e6,
            delta_type=delta_type,
        )
        if vol_smile:
            vol_ = FXDeltaVolSmile(
                nodes={
                    0.25: 8.9,
                    0.5: 8.7,
                    0.75: 10.15,
                },
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type=smile_type,
            )
        else:
            vol_ = 9.00
        with pytest.raises(ValueError, match="Newton root solver failed"):
            fxo._index_vol_and_strike_from_delta(
                delta,
                delta_type,
                vol_,
                fxfo.curve("eur", "usd")[dt(2023, 6, 20)],
                fxfo.curve("eur", "usd")[dt(2023, 3, 20)],
                fxfo.rate("eurusd", dt(2023, 6, 20)),
                fxo._t_to_expiry(fxfo.curve("eur", "usd").nodes.initial),
            )

    @pytest.mark.parametrize("delta_type", ["forward", "spot"])
    def test_analytic_gamma_fwd_diff(self, delta_type, fxfo) -> None:
        # test not suitable for pa because of the assumption of a fixed premium amount
        fxc = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 3, 16),
            notional=20e6,
            strike=1.101,
            delta_type=delta_type,
        )
        base = fxc.analytic_greeks(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=8.9,
        )
        f_d = fxfo.rate("eurusd", dt(2023, 6, 20))
        f_t = fxfo.rate("eurusd", dt(2023, 3, 20))
        fxfo.fx_rates.update({"eurusd": 1.0615001})
        fxfo.update()
        f_d2 = fxfo.rate("eurusd", dt(2023, 6, 20))
        f_t2 = fxfo.rate("eurusd", dt(2023, 3, 20))
        base_1 = fxc.analytic_greeks(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=8.9,
        )
        denomn = (f_d2 - f_d) if "forward" in delta_type else (f_t2 - f_t)
        fwd_diff = -(base["delta"] - base_1["delta"]) / denomn
        assert abs(base["gamma"] - fwd_diff) < 1e-5

    def test_analytic_vega(self, fxfo) -> None:
        fxc = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 3, 16),
            notional=20e6,
            strike=1.101,
            delta_type="forward",
        )
        result = fxc.analytic_greeks(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=8.9,
        )["vega"]
        assert abs(result * 20e6 / 100 - 33757.945) < 1e-2

        p0 = fxc.npv(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=8.9,
        )
        p1 = fxc.npv(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=8.91,
        )
        fwd_diff = (p1 - p0) / 20e6 * 10000.0
        assert abs(result - fwd_diff) < 1e-4

    def test_analytic_vomma(self, fxfo) -> None:
        fxc = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 3, 16),
            notional=1,
            strike=1.101,
            delta_type="forward",
        )
        result = fxc.analytic_greeks(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=8.9,
        )["vomma"]

        p0 = fxc.npv(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=8.9,
        )
        p1 = fxc.npv(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=8.91,
        )
        p_1 = fxc.npv(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=8.89,
        )
        fwd_diff = (p1 - p0 - p0 + p_1) * 1e4 * 1e4
        assert abs(result - fwd_diff) < 1e-6

    @pytest.mark.parametrize("payment", [dt(2023, 3, 16), dt(2023, 6, 20)])
    def test_vega_and_vomma_example(self, fxfo, payment) -> None:
        fxc = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=payment,
            notional=10e6,
            strike=1.10,
            delta_type="forward",
        )
        npv = fxc.npv(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=10.0,
        )
        npv2 = fxc.npv(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=10.1,
        )
        greeks = fxc.analytic_greeks(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=Dual(10.0, ["vol"], [100.0]),
        )
        taylor_vega = 10e6 * greeks["vega"] * 0.1 / 100.0
        taylor_vomma = 10e6 * 0.5 * greeks["vomma"] * 0.1**2 / 10000.0
        expected = npv2 - npv
        assert abs(taylor_vega + taylor_vomma - expected) < 0.2

    @pytest.mark.parametrize("payment", [dt(2023, 3, 16), dt(2023, 6, 20)])
    @pytest.mark.parametrize("delta_type", ["spot", "forward"])
    def test_delta_and_gamma_example(self, fxfo, payment, delta_type) -> None:
        fxc = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=payment,
            notional=10e6,
            strike=1.10,
            delta_type=delta_type,
        )
        npv = fxc.npv(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=10.0,
        )
        greeks = fxc.analytic_greeks(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=10.0,
        )
        f_d = fxfo.rate("eurusd", dt(2023, 6, 20))
        fxfo.fx_rates.update({"eurusd": 1.0625})
        fxfo.update()
        f_d2 = fxfo.rate("eurusd", dt(2023, 6, 20))
        npv2 = fxc.npv(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=10.0,
        )
        if delta_type == "forward":
            fwd_diff = f_d2 - f_d
            discount_date = fxc.delivery
        else:
            fwd_diff = 0.001
            discount_date = dt(2023, 3, 20)
        taylor_delta = 10e6 * greeks["delta"] * fwd_diff
        taylor_gamma = 10e6 * 0.5 * greeks["gamma"] * fwd_diff**2
        expected = npv2 - npv
        taylor = (taylor_delta + taylor_gamma) * fxfo.curve("usd", "usd")[discount_date]
        assert abs(taylor - expected) < 0.5

    @pytest.mark.parametrize("payment", [dt(2023, 6, 20), dt(2023, 3, 16)])
    @pytest.mark.parametrize("delta_type", ["spot", "forward"])
    def test_all_5_greeks_example(self, fxfo, payment, delta_type) -> None:
        fxc = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=payment,
            notional=10e6,
            strike=1.10,
            delta_type=delta_type,
        )
        npv = fxc.npv(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=10.0,
        )
        greeks = fxc.analytic_greeks(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=Dual(10.0, ["vol"], [100.0]),
        )
        f_d = fxfo.rate("eurusd", dt(2023, 6, 20))
        fxfo.fx_rates.update({"eurusd": 1.0625})
        fxfo.update()
        f_d2 = fxfo.rate("eurusd", dt(2023, 6, 20))
        if delta_type == "forward":
            fwd_diff = f_d2 - f_d
            discount_date = fxc.delivery
        else:
            fwd_diff = 0.001
            discount_date = dt(2023, 3, 20)
        npv2 = fxc.npv(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=10.1,
        )
        fxc.analytic_greeks(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=Dual(10.1, ["vol"], [100.0]),
        )
        expected = npv2 - npv
        taylor_delta = fwd_diff * greeks["delta"] * 10e6
        taylor_gamma = 0.5 * fwd_diff**2 * greeks["gamma"] * 10e6
        taylor_vega = 0.1 / 100.0 * greeks["vega"] * 10e6
        taylor_vomma = 0.5 * 0.1**2 / 10000.0 * greeks["vomma"] * 10e6
        taylor_vanna = 0.1 / 100.0 * fwd_diff * greeks["vanna"] * 10e6
        taylor = (
            fxfo.curve("usd", "usd")[discount_date] * (taylor_delta + taylor_gamma + taylor_vanna)
            + taylor_vomma
            + taylor_vega
        )
        assert abs(taylor - expected) < 5e-1

    def test_kega(self, fxfo) -> None:
        fxc = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            notional=10e6,
            strike=1.10,
            delta_type="spot_pa",
        )

        d_eta = _d_plus_min_u(1.10 / 1.065, 0.10 * 0.5, -0.5)
        result = fxc._analytic_kega(1.10 / 1.065, 0.99, -0.5, 0.10, 0.50, 1.065, 1.0, 1.10, d_eta)
        expected = 0.355964619118249
        assert abs(result - expected) < 1e-12

    def test_bad_expiries_raises(self, fxfo) -> None:
        fxc = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            notional=10e6,
            strike=1.10,
            delta_type="forward",
        )
        vol_ = FXDeltaVolSmile(
            nodes={
                0.25: 8.9,
                0.5: 8.7,
                0.75: 10.15,
            },
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 18),
            delta_type="forward",
        )
        with pytest.raises(ValueError, match="`expiry` of VolSmile and OptionPeriod do not match"):
            fxc.npv(fxfo.curve("eur", "usd"), fxfo.curve("usd", "usd"), fx=fxfo, vol=vol_)

    @pytest.mark.parametrize("smile", [True, False])
    def test_call_cashflows(self, fxfo, smile) -> None:
        vol_ = (
            8.9
            if not smile
            else FXDeltaVolSmile(
                nodes={0.5: 8.9},
                eval_date=dt(2023, 3, 16),
                expiry=dt(2023, 6, 16),
                delta_type="forward",
            )
        )
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=1.101,
            notional=20e6,
        )
        result = fxo.cashflows(
            fxfo.curve("eur", "usd"),
            fxfo.curve("usd", "usd"),
            fx=fxfo,
            vol=vol_,
            base="eur",
        )
        assert isinstance(result, dict)
        expected = 140451.5273
        assert (result[defaults.headers["cashflow"]] - expected) < 1e-3
        assert result[defaults.headers["currency"]] == "USD"
        assert result[defaults.headers["type"]] == "FXCallPeriod"

    @pytest.mark.parametrize("delta_type", ["spot", "forward"])
    def test_sticky_delta_delta_vol_smile_against_ad(self, fxfo, delta_type) -> None:
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=1.101,
            notional=20e6,
            delta_type=delta_type,
        )
        vol_ = FXDeltaVolSmile(
            nodes={
                0.25: 8.9,
                0.5: 8.7,
                0.75: 10.15,
            },
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            delta_type="spot",
        )
        gks = fxo.analytic_greeks(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            base="usd",
            vol=vol_,
        )

        v_deli = fxfo.curve("usd", "usd")[fxo.delivery]
        v_spot = fxfo.curve("usd", "usd")[dt(2023, 3, 20)]

        # this is the actual derivative of vol with respect to either spot or forward via AD
        if "spot" in delta_type:
            z_v_0 = v_deli / v_spot
            expected = gradient(gks["__vol"], ["fx_eurusd"])[0]
        else:
            z_v_0 = 1.0
            w_deli = fxfo.curve("eur", "usd")[fxo.delivery]
            w_spot = fxfo.curve("eur", "usd")[dt(2023, 3, 20)]
            expected = (
                gradient(gks["__vol"], ["fx_eurusd"])[0] * v_deli * w_spot / (v_spot * w_deli)
            )

        # this is the reverse engineered part of the sticky delta formula to get dsigma_dfspot
        result = (gks["delta_sticky"] - gks["delta"]) * v_deli / (z_v_0 * gks["vega"])
        # delta is
        assert abs(result - expected) < 1e-3

    @pytest.mark.parametrize(
        ("smile", "expected"),
        [
            (
                FXSabrSmile(
                    nodes={"alpha": 0.05, "beta": 1.0, "rho": 0.01, "nu": 0.03},
                    eval_date=dt(2024, 5, 7),
                    expiry=dt(2024, 5, 28),
                    id="smile",
                    pair="eurusd",
                ),
                0.700594,
            ),
            (
                FXSabrSurface(
                    expiries=[dt(2024, 5, 23), dt(2024, 6, 4)],
                    node_values=[[0.05, 1.0, 0.01, 0.03], [0.052, 1.0, 0.03, 0.05]],
                    eval_date=dt(2024, 5, 7),
                    id="smile",
                    pair="eurusd",
                ),
                0.701191,
            ),
            (
                FXDeltaVolSmile(
                    nodes={0.25: 10, 0.5: 9, 0.75: 11},
                    eval_date=dt(2024, 5, 7),
                    expiry=dt(2024, 5, 28),
                    delta_type="forward",
                    id="smile",
                ),
                0.704091,
            ),
        ],
    )
    def test_sticky_delta_calculation(self, smile, expected) -> None:
        from rateslib import IRS, FXBrokerFly, FXCall, FXRiskReversal, FXStraddle, FXSwap, Solver

        usd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="nyc", id="usd")
        eur = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="tgt", id="eur")
        eurusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, id="eurusd")
        # Create an FX Forward market with spot FX rate data
        spot = dt(2024, 5, 9)
        fxr = FXRates({"eurusd": 1.0760}, settlement=spot)
        fxf = FXForwards(
            fx_rates=fxr,
            fx_curves={"eureur": eur, "usdusd": usd, "eurusd": eurusd},
        )
        # Solve the Curves to market
        pre_solver = Solver(
            curves=[eur, eurusd, usd],
            instruments=[
                IRS(spot, "3W", spec="eur_irs", curves="eur"),
                IRS(spot, "3W", spec="usd_irs", curves="usd"),
                FXSwap(spot, "3W", pair="eurusd", curves=[None, "eurusd", None, "usd"]),
            ],
            s=[3.90, 5.32, 8.85],
            fx=fxf,
            id="fxf",
        )

        option_args = dict(
            pair="eurusd",
            expiry=dt(2024, 5, 28),
            calendar="tgt|fed",
            delta_type="spot",
            curves=[None, "eurusd", None, "usd"],
            vol="smile",
        )

        # Calibrate the Smile to market option data
        solver = Solver(
            pre_solvers=[pre_solver],
            curves=[smile],
            instruments=[
                FXStraddle(strike="atm_delta", **option_args),
                FXRiskReversal(strike=("-25d", "25d"), **option_args),
                FXRiskReversal(strike=("-10d", "10d"), **option_args),
                FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), **option_args),
                FXBrokerFly(strike=(("-10d", "10d"), "atm_delta"), **option_args),
            ],
            s=[5.493, -0.157, -0.289, 0.071, 0.238],
            fx=fxf,
            id="smile",
        )

        fxc = FXCall(**option_args, notional=100e6, strike=1.07, premium=982144.59)

        result = fxc.analytic_greeks(solver=solver)["delta_sticky"]
        assert abs(result - expected) < 1e-6

    def test_sticky_delta_direct_from_ad(self, fxfo) -> None:
        # this test will use AD to directly measure dP_dfs and compare that with the
        # analytical derivation of sticky delta.
        fxo = FXCallPeriod(
            pair="eurusd",
            expiry=dt(2023, 6, 16),
            delivery=dt(2023, 6, 20),
            payment=dt(2023, 6, 20),
            strike=1.101,
            notional=20e6,
            delta_type="spot",
        )
        vol_ = FXDeltaVolSmile(
            nodes={
                0.25: 8.9,
                0.5: 8.7,
                0.75: 10.15,
            },
            eval_date=dt(2023, 3, 16),
            expiry=dt(2023, 6, 16),
            delta_type="spot",
        )
        gks = fxo.analytic_greeks(
            disc_curve=fxfo.curve("eur", "usd"),
            disc_curve_ccy2=fxfo.curve("usd", "usd"),
            fx=fxfo,
            base="usd",
            vol=vol_,
        )

        P = 20e6 * gks["__bs76"]
        dP_dfs = gradient(P, ["fx_eurusd"])[0]
        v_spot = fxfo.curve("usd", "usd")[dt(2023, 3, 20)]
        result = dP_dfs / (20e6 * v_spot)
        expected = gks["delta_sticky"]
        assert abs(result - expected) < 1e-8
