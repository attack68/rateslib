import pytest
import re
from datetime import datetime as dt
from datetime import timedelta
from pandas.testing import assert_frame_equal
from pandas import DataFrame, Series, Index
import numpy as np

import context
from rateslib.default import NoInput
from rateslib.periods import (
    Cashflow,
    FixedPeriod,
    FloatPeriod,
    IndexFixedPeriod,
    IndexCashflow,
    IndexMixin,
)
from rateslib.fx import FXRates
from rateslib.default import Defaults
from rateslib.curves import Curve, LineCurve, IndexCurve, CompositeCurve
from rateslib import defaults


@pytest.fixture()
def curve():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.99,
        dt(2022, 7, 1): 0.98,
        dt(2022, 10, 1): 0.97,
    }
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture()
def fxr():
    return FXRates({"usdnok": 10.0})


@pytest.fixture()
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


@pytest.fixture()
def line_curve():
    nodes = {
        dt(2021, 12, 31): -99,
        dt(2022, 1, 1): 1.00,
        dt(2022, 1, 2): 2.00,
        dt(2022, 1, 3): 3.00,
        dt(2022, 1, 4): 4.00,
        dt(2022, 1, 5): 5.00,
    }
    return LineCurve(nodes=nodes, interpolation="linear", convention="act365f")


class TestFXandBase:
    def test_fx_and_base_raise(self):
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

    def test_fx_and_base_warn1(self):
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

    def test_fx_and_base_warn2(self):
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
    def test_none_cashflow(self):
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
        "spread_method, float_spread, expected",
        [
            ("none_simple", 100.0, 24744.478172244584),
            ("isda_compounding", 0.0, 24744.478172244584),
            ("isda_compounding", 100.0, 25053.484941157145),
            ("isda_flat_compounding", 100.0, 24867.852396116967),
        ],
    )
    def test_float_period_analytic_delta(self, curve, spread_method, float_spread, expected):
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
        "spread, crv, fx",
        [
            (4.00, True, 2.0),
            (NoInput(0), False, 2.0),
            (4.00, True, 10.0),
            (NoInput(0), False, 10.0),
        ],
    )
    def test_float_period_cashflows(self, curve, fxr, spread, crv, fx):
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
            Defaults.headers["type"]: "FloatPeriod",
            Defaults.headers["stub_type"]: "Regular",
            Defaults.headers["a_acc_start"]: dt(2022, 1, 1),
            Defaults.headers["a_acc_end"]: dt(2022, 4, 1),
            Defaults.headers["payment"]: dt(2022, 4, 3),
            Defaults.headers["notional"]: 1e9,
            Defaults.headers["currency"]: "USD",
            Defaults.headers["convention"]: "Act360",
            Defaults.headers["dcf"]: float_period.dcf,
            Defaults.headers["df"]: 0.9897791268897856 if crv else None,
            Defaults.headers["rate"]: rate,
            Defaults.headers["spread"]: 0 if spread is NoInput.blank else spread,
            Defaults.headers["npv"]: -10096746.871171726 if crv else None,
            Defaults.headers["cashflow"]: cashflow,
            Defaults.headers["fx"]: fx,
            Defaults.headers["npv_fx"]: -10096746.871171726 * fx if crv else None,
            Defaults.headers["collateral"]: None,
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

    def test_spread_compound_raises(self):
        with pytest.raises(ValueError, match="`spread_compound_method`"):
            FloatPeriod(
                start=dt(2022, 1, 1),
                end=dt(2022, 4, 1),
                payment=dt(2022, 4, 3),
                frequency="Q",
                spread_compound_method="bad_vibes",
            )

    def test_spread_compound_calc_raises(self):
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
            period._isda_compounded_rate_with_spread(Series([1, 2]), Series([1, 1]))

    def test_rfr_lockout_too_few_dates(self, curve):
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

    def test_fixing_method_raises(self):
        with pytest.raises(ValueError, match="`fixing_method`"):
            FloatPeriod(
                start=dt(2022, 1, 1),
                end=dt(2022, 4, 1),
                payment=dt(2022, 4, 3),
                frequency="Q",
                fixing_method="bad_vibes",
            )

    def test_float_period_npv(self, curve):
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

    def test_rfr_avg_method_raises(self, curve):
        period = FloatPeriod(
            dt(2022, 1, 1), dt(2022, 1, 4), dt(2022, 1, 4), "Q",
            fixing_method="rfr_payment_delay_avg",
            spread_compound_method="isda_compounding",
        )
        msg = "`spread_compound` method must be 'none_simple' in an RFR averaging " \
              "period."
        with pytest.raises(ValueError, match=msg):
            period.rate(curve)

    @pytest.mark.parametrize("curve_type", ["curve", "line_curve"])
    def test_rfr_payment_delay_method(self, curve_type, rfr_curve, line_curve):
        curve = rfr_curve if curve_type == "curve" else line_curve
        period = FloatPeriod(
            dt(2022, 1, 1), dt(2022, 1, 4), dt(2022, 1, 4), "Q", fixing_method="rfr_payment_delay"
        )
        result = period.rate(curve)
        expected = ((1 + 0.01 / 365) * (1 + 0.02 / 365) * (1 + 0.03 / 365) - 1) * 36500 / 3
        assert abs(result - expected) < 1e-12

    @pytest.mark.parametrize("curve_type", ["curve", "line_curve"])
    def test_rfr_payment_delay_method_with_fixings(self, curve_type, rfr_curve, line_curve):
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
    def test_rfr_payment_delay_avg_method(self, curve_type, rfr_curve, line_curve):
        curve = rfr_curve if curve_type == "curve" else line_curve
        period = FloatPeriod(
            dt(2022, 1, 1), dt(2022, 1, 4), dt(2022, 1, 4), "Q", fixing_method="rfr_payment_delay_avg"
        )
        result = period.rate(curve)
        expected = (1.0 + 2.0 + 3.0) / 3
        assert abs(result - expected) < 1e-11

    @pytest.mark.parametrize("curve_type", ["curve", "line_curve"])
    def test_rfr_payment_delay_avg_method_with_fixings(self, curve_type, rfr_curve, line_curve):
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
    def test_rfr_lockout_avg_method(self, curve_type, rfr_curve, line_curve):
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
    def test_rfr_lockout_avg_method_with_fixings(self, curve_type, rfr_curve, line_curve):
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
        expected = (10.0 + 8.0 + 8.0 ) /3
        assert abs(result - expected) < 1e-12

    @pytest.mark.parametrize("curve_type", ["curve", "line_curve"])
    def test_rfr_lockout_method(self, curve_type, rfr_curve, line_curve):
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
    def test_rfr_lockout_method_with_fixings(self, curve_type, rfr_curve, line_curve):
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
    def test_rfr_observation_shift_method(self, curve_type, rfr_curve, line_curve):
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
    def test_rfr_observation_shift_method_with_fixings(self, curve_type, rfr_curve, line_curve):
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
    def test_rfr_observation_shift_avg_method(self, curve_type, rfr_curve, line_curve):
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
        expected = (1.0 + 2 + 3) /3
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
    def test_rfr_observation_shift_avg_method_with_fixings(self, curve_type, rfr_curve, line_curve):
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

    def test_dcf_obs_period_raises(self):
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
        "method, expected, expected_date",
        [
            ("rfr_payment_delay", [1000000, 1000082, 1000191, 1000561], dt(2022, 1, 6)),
            ("rfr_observation_shift", [1499240, 1499281, 1499363, 1499486], dt(2022, 1, 4)),
            ("rfr_lockout", [999931, 4999411, 0, 0], dt(2022, 1, 6)),
            ("rfr_lookback", [999657, 999685, 2998726, 999821], dt(2022, 1, 4)),
        ],
    )
    def test_rfr_fixings_array(self, curve_type, method, expected, expected_date):
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
        rate, table = period._rfr_fixings_array(
            curve=rfr_curve, fixing_exposure=True, disc_curve=curve
        )

        assert table["obs_dates"][1] == expected_date
        for i, val in table["notional"].iloc[:-1].items():
            assert abs(expected[i] - val) < 1

    def test_rfr_fixings_array_raises(self, rfr_curve):
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
            period._rfr_fixings_array(rfr_curve)

    @pytest.mark.parametrize(
        "method, param, expected",
        [
            ("rfr_payment_delay", 0, 1000000),
            ("rfr_observation_shift", 1, 333319),
            ("rfr_lookback", 1, 333319),
        ],
    )
    def test_rfr_fixings_array_single_period(self, method, param, expected):
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
        assert abs(result["notional"].iloc[0] - expected) < 1

    @pytest.mark.parametrize(
        "method, expected",
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
    def test_rfr_compounding_float_spreads(self, method, expected, rfr_curve):
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

    def test_ibor_rate_line_curve(self, line_curve):
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

    def test_ibor_fixing_table(self, line_curve):
        float_period = FloatPeriod(
            start=dt(2022, 1, 4),
            end=dt(2022, 4, 4),
            payment=dt(2022, 4, 4),
            frequency="Q",
            fixing_method="ibor",
            method_param=2,
        )
        result = float_period.fixings_table(line_curve)
        expected = DataFrame(
            {"obs_dates": [dt(2022, 1, 2)], "notional": [-1e6], "dcf": [None], "rates": [2.0]}
        ).set_index("obs_dates")
        assert_frame_equal(expected, result)

    def test_ibor_fixing_table_fast(self, line_curve):
        float_period = FloatPeriod(
            start=dt(2022, 1, 4),
            end=dt(2022, 4, 4),
            payment=dt(2022, 4, 4),
            frequency="Q",
            fixing_method="ibor",
            method_param=2,
        )
        result = float_period.fixings_table(line_curve, approximate=True)
        expected = DataFrame(
            {"obs_dates": [dt(2022, 1, 2)], "notional": [-1e6], "dcf": [None], "rates": [2.0]}
        ).set_index("obs_dates")
        assert_frame_equal(expected, result)

    def test_ibor_fixings(self):
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

    def test_ibor_fixing_unavailable(self):
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
    def test_ibor_rate_df_curve(self, float_spread, curve):
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
    def test_ibor_rate_stub_df_curve(self, float_spread, curve):
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

    def test_single_fixing_override(self, curve):
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
    def test_period_historic_fixings(self, curve_type, line_curve, rfr_curve):
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
    def test_period_historic_fixings_series(self, curve_type, line_curve, rfr_curve):
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

    @pytest.mark.parametrize("curve_type", ["curve", "linecurve"])
    def test_period_historic_fixings_series_missing_warns(self, curve_type, line_curve, rfr_curve):
        curve = rfr_curve if curve_type == "curve" else line_curve
        fixings = Series([99, 99, 2.5], index=[dt(1995, 12, 1), dt(2021, 12, 30), dt(2022, 1, 1)])
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
        with pytest.warns(UserWarning):
            period.rate(curve)
        # assert result == expected

    def test_fixing_with_float_spread_warning(self, curve):
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

    # @pytest.mark.skip(reason="str is an erroneous input to function: test redundant.")
    # def test_float_period_rate_raises(self):
    #     float_period = FloatPeriod(
    #         start=dt(2022, 1, 4),
    #         end=dt(2022, 4, 4),
    #         payment=dt(2022, 4, 4),
    #         frequency="Q",
    #     )
    #     with pytest.raises(TypeError, match="Curve must be of type"):
    #         float_period.rate("bad_curve")

    def test_float_period_fixings_list_raises_on_ibor(self, curve, line_curve):
        with pytest.raises(ValueError, match="`fixings` cannot be supplied as list,"):
            float_period = FloatPeriod(
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

    @pytest.mark.parametrize("meth, exp", [
        ("rfr_payment_delay", DataFrame(
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
                "dcf": [0.0027777777777777778] * 5,
                "rates": [1.19, 1.19, -8.81, 4.01364, 4.01364],
            }
        ).set_index("obs_dates")),
        ("rfr_payment_delay_avg", DataFrame(
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
                "dcf": [0.0027777777777777778] * 5,
                "rates": [1.19, 1.19, -8.81, 4.01364, 4.01364],
            }
        ).set_index("obs_dates"))
    ])
    def test_rfr_fixings_table(self, curve, meth, exp):
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
        "method, param",
        [
            ("rfr_payment_delay", NoInput(0)),
            ("rfr_lookback", 4),
            ("rfr_lockout", 1),
            ("rfr_observation_shift", 2),
        ],
    )
    @pytest.mark.parametrize(
        "scm, spd",
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
    def test_rfr_fixings_table_fast(self, method, param, scm, spd, crv):
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
        "method, param",
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
    def test_rfr_fixings_table_fast_avg(self, method, param, crv):
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

    def test_rfr_rate_fixings_series_monotonic_error(self):
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
        "scm, exp",
        [
            ("none_simple", True),
            ("isda_compounding", False),
        ],
    )
    def test_float_spread_affects_fixing_exposure(self, scm, exp):
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
        assert (table["notional"].iloc[0] == table2["notional"].iloc[0]) == exp

    def test_custom_interp_rate_nan(self):
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

    def test_method_param_raises(self):
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

    def test_analytic_delta_no_curve_raises(self):
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

    def test_more_series_fixings_than_calendar_from_curve_raises(self):
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

    def test_series_fixings_not_applicable_to_period(self):
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
        "meth, param, exp",
        [
            ("rfr_payment_delay", NoInput(0), 3.1183733605),
            ("rfr_observation_shift", 2, 3.085000395),
            ("rfr_lookback", 2, 3.05163645),
            ("rfr_lockout", 7, 3.00157855),
        ],
    )
    def test_norges_bank_nowa_calc_same(self, meth, param, exp):
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
            fixings=defaults.fixings.nowa,
        )
        result = period.rate(curve)
        assert abs(result - exp) < 1e-7

    def test_interpolated_ibor_warns(self):
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

    def test_interpolated_ibor_rate_line(self):
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
        expected = 1.0 + (3.0 - 1.0) * (dt(2023, 4, 1) - dt(2023, 3, 1)) / (dt(2023, 5, 1) - dt(2023, 3, 1))
        assert abs(result - expected) < 1e-8

    def test_interpolated_ibor_rate_df(self):
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
        expected = a + (b - a) * (dt(2023, 4, 1) - dt(2023, 3, 1)) / (dt(2023, 5, 1) - dt(2023, 3, 1))
        assert abs(result - expected) < 1e-8

    def test_rfr_period_curve_dict_raises(self, curve):
        period = FloatPeriod(
            start=dt(2023, 2, 1),
            end=dt(2023, 4, 1),
            payment=dt(2023, 4, 1),
            frequency="A",
            fixing_method="rfr_payment_delay",
            float_spread=0.0,
            stub=True,
        )
        with pytest.raises(ValueError, match="Must supply a valid curve for forecasting"):
            period.rate({"rfr": curve})

    def test_ibor_stub_fixings_table(self):
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
        expected = DataFrame(
            data=[[-1e6, None, 2.01639]],
            index=Index([dt(2023, 1, 31)], name="obs_dates"),
            columns=["notional", "dcf", "rates"],
        )
        assert_frame_equal(result, expected)


class TestFixedPeriod:
    def test_fixed_period_analytic_delta(self, curve, fxr):
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

    def test_fixed_period_analytic_delta_fxr_base(self, curve, fxr):
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
        "rate, crv, fx",
        [
            (4.00, True, 2.0),
            (NoInput(0), False, 2.0),
            (4.00, True, 10),
            (NoInput(0), False, 10),
        ],
    )
    def test_fixed_period_cashflows(self, curve, fxr, rate, crv, fx):
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
            Defaults.headers["type"]: "FixedPeriod",
            Defaults.headers["stub_type"]: "Regular",
            Defaults.headers["a_acc_start"]: dt(2022, 1, 1),
            Defaults.headers["a_acc_end"]: dt(2022, 4, 1),
            Defaults.headers["payment"]: dt(2022, 4, 3),
            Defaults.headers["notional"]: 1e9,
            Defaults.headers["currency"]: "USD",
            Defaults.headers["convention"]: "Act360",
            Defaults.headers["dcf"]: fixed_period.dcf,
            Defaults.headers["df"]: 0.9897791268897856 if crv else None,
            Defaults.headers["rate"]: rate,
            Defaults.headers["spread"]: None,
            Defaults.headers["npv"]: -9897791.268897856 if crv else None,
            Defaults.headers["cashflow"]: cashflow,
            Defaults.headers["fx"]: fx,
            Defaults.headers["npv_fx"]: -9897791.268897855 * fx if crv else None,
            Defaults.headers["collateral"]: None,
        }
        if fx == 2.0:
            with pytest.warns(UserWarning):
                # supplying `fx` as numeric
                result = fixed_period.cashflows(curve if crv else NoInput(0), fx=2.0, base=NoInput(0))
        else:
            result = fixed_period.cashflows(curve if crv else NoInput(0), fx=fxr, base="nok")
        assert result == expected

    def test_fixed_period_npv(self, curve, fxr):
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

    def test_fixed_period_npv_raises(self, curve):
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
            match=re.escape("`curves` have not been supplied correctly.")
        ):
            fixed_period.npv()


class TestCashflow:
    def test_cashflow_analytic_delta(self, curve):
        cashflow = Cashflow(notional=1e6, payment=dt(2022, 1, 1))
        assert cashflow.analytic_delta(curve) == 0

    @pytest.mark.parametrize(
        "crv, fx",
        [
            (True, 2.0),
            (False, 2.0),
            (True, 10.0),
            (False, 10.0),
        ],
    )
    def test_cashflow_cashflows(self, curve, fxr, crv, fx):
        cashflow = Cashflow(notional=1e9, payment=dt(2022, 4, 3))
        curve = curve if crv else NoInput(0)
        expected = {
            Defaults.headers["type"]: "Cashflow",
            Defaults.headers["stub_type"]: None,
            Defaults.headers["a_acc_start"]: None,
            Defaults.headers["a_acc_end"]: None,
            Defaults.headers["payment"]: dt(2022, 4, 3),
            Defaults.headers["currency"]: "USD",
            Defaults.headers["notional"]: 1e9,
            Defaults.headers["convention"]: None,
            Defaults.headers["dcf"]: None,
            Defaults.headers["df"]: 0.9897791268897856 if crv else None,
            Defaults.headers["rate"]: None,
            Defaults.headers["spread"]: None,
            Defaults.headers["npv"]: -989779126.8897856 if crv else None,
            Defaults.headers["cashflow"]: -1e9,
            Defaults.headers["fx"]: fx,
            Defaults.headers["npv_fx"]: -989779126.8897856 * fx if crv else None,
            Defaults.headers["collateral"]: None,
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

    def test_cashflow_npv_raises(self, curve):
        with pytest.raises(
            TypeError,
            match="`curves` have not been supplied correctly."
        ):
            cashflow = Cashflow(notional=1e6, payment=dt(2022, 1, 1))
            cashflow.npv()
        cashflow = Cashflow(notional=1e6, payment=dt(2022, 1, 1))
        assert cashflow.analytic_delta(curve) == 0

    def test_cashflow_npv_local(self, curve):
        cashflow = Cashflow(notional=1e9, payment=dt(2022, 4, 3), currency="nok")
        result = cashflow.npv(curve, local=True)
        expected = {"nok": -989779126.8897856}
        assert result == expected


class TestIndexFixedPeriod:
    @pytest.mark.parametrize(
        "method, expected", [("daily", 201.00502512562812), ("monthly", 200.98317675333183)]
    )
    def test_period_rate(self, method, expected):
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
        index_curve = IndexCurve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 4, 3): 0.995},
            index_base=200.0,
        )
        _, result, _ = index_period.index_ratio(index_curve)
        assert abs(result - expected) < 1e-8

    def test_period_cashflow(self):
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
        index_curve = IndexCurve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 4, 3): 0.995},
            index_base=200.0,
        )
        result = index_period.real_cashflow
        expected = -1e7 * ((dt(2022, 4, 1) - dt(2022, 1, 1)) / timedelta(days=360)) * 4
        assert abs(result - expected) < 1e-8

        result = index_period.cashflow(index_curve)
        expected = expected * index_curve.index_value(dt(2022, 4, 3)) / 100.0
        assert abs(result - expected) < 1e-8

    def test_period_analytic_delta(self, fxr, curve):
        index_curve = IndexCurve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 4, 3): 0.995},
            index_base=200.0,
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

    @pytest.mark.parametrize(
        "fixings, method",
        [
            (300.0, "daily"),
            (
                Series([1.0, 300, 5], index=[dt(2022, 4, 2), dt(2022, 4, 3), dt(2022, 4, 4)]),
                "daily",
            ),
            (Series([100.0, 500], index=[dt(2022, 4, 2), dt(2022, 4, 4)]), "daily"),
            (Series([300.0, 500], index=[dt(2022, 4, 1), dt(2022, 4, 5)]), "monthly"),
        ],
    )
    def test_period_fixings_series(self, fixings, method, curve):
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

    def test_period_raises(self):
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

    def test_period_npv(self, curve):
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
        index_curve = IndexCurve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 4, 3): 0.995},
            index_base=200.0,
        )
        result = index_period.npv(index_curve, curve)
        expected = -19895057.826930363
        assert abs(result - expected) < 1e-8

        result = index_period.npv(index_curve, curve, local=True)
        assert abs(result["usd"] - expected) < 1e-8

    def test_period_npv_raises(self):
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
                match=re.escape("`curves` have not been supplied correctly.")
        ):
            index_period.npv()

    @pytest.mark.parametrize("curve_", [True, False])
    def test_period_cashflows(self, curve, curve_):
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
            Defaults.headers["collateral"]: None,
        }
        assert result == expected

    def test_cashflow_returns_none(self):
        i_period = IndexFixedPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 2, 1),
            payment=dt(2022, 2, 1),
            frequency="M",
            index_base=100.0,
        )
        assert i_period.cashflow() is None
        assert i_period.real_cashflow is None

    def test_cashflow_no_index_rate(self):
        i_period = IndexFixedPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 2, 1),
            payment=dt(2022, 2, 1),
            frequency="M",
            index_base=100.0,
        )
        result = i_period.cashflows()
        assert result[Defaults.headers["index_ratio"]] is None

    def test_bad_curve(self):
        i_period = IndexFixedPeriod(
            start=dt(2022, 1, 1),
            end=dt(2022, 2, 1),
            payment=dt(2022, 2, 1),
            frequency="M",
            index_base=100.0,
        )
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99})
        with pytest.raises(TypeError, match="`index_value` must be forecast from"):
            i_period.index_ratio(curve)

    # TEST REDUNDANT: function was changed to fallback to forecast from curve
    # def test_cannot_forecast_from_fixings(self):
    #     i_fixings = Series([100], index=[dt(2021, 1, 1)])
    #     i_period = IndexFixedPeriod(
    #         start=dt(2022, 1, 1),
    #         end=dt(2022, 2, 1),
    #         payment=dt(2022, 2, 1),
    #         frequency="M",
    #         index_base=100.0,
    #         index_fixings=i_fixings,
    #     )
    #     curve = IndexCurve(
    #         {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
    #         index_lag=3,
    #         index_base=100.0
    #     )
    #     with pytest.raises(ValueError, match="`index_fixings` cannot forecast the"):
    #         i_period.index_ratio(curve)

    def test_index_fixings_linear_interp(self):
        i_fixings = Series([173.1, 174.2], index=[dt(2001, 7, 1), dt(2001, 8, 1)])
        result = IndexMixin._index_value(
            i_fixings=i_fixings, i_curve=None, i_date=dt(2001, 7, 20), i_lag=3, i_method="daily"
        )
        expected = 173.1 + 19 / 31 * (174.2 - 173.1)
        assert abs(result - expected) < 1e-6

    def test_composite_curve(self):
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
        index_curve = IndexCurve(
            nodes={dt(2022, 1, 1): 1.0, dt(2022, 4, 3): 0.995},
            index_base=200.0,
        )
        composite_curve = CompositeCurve([index_curve])
        _, result, _ = index_period.index_ratio(composite_curve)

    def test_composite_curve_raises(self):
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
        with pytest.raises(TypeError, match="`index_value` must be forecast from"):
            _, result, _ = index_period.index_ratio(composite_curve)


class TestIndexCashflow:
    def test_cashflow_analytic_delta(self, curve):
        cashflow = IndexCashflow(notional=1e6, payment=dt(2022, 1, 1), index_base=100)
        assert cashflow.analytic_delta(curve) == 0

    def test_index_cashflow(self):
        cf = IndexCashflow(notional=1e6, payment=dt(2022, 1, 1), index_base=100, index_fixings=200)
        assert cf.real_cashflow == -1e6

        assert cf.cashflow(None) == -2e6

    def test_index_cashflow_npv(self, curve):
        cf = IndexCashflow(notional=1e6, payment=dt(2022, 1, 1), index_base=100, index_fixings=200)
        assert abs(cf.npv(curve) + 2e6) < 1e-6

    def test_cashflow_no_index_rate(self):
        i_period = IndexCashflow(
            notional=200.0,
            payment=dt(2022, 2, 1),
            index_base=100.0,
        )
        result = i_period.cashflows()
        assert result[Defaults.headers["index_ratio"]] is None

    def test_index_only(self, curve):
        cf = IndexCashflow(
            notional=1e6,
            payment=dt(2022, 1, 1),
            index_base=100,
            index_fixings=200,
            index_only=True,
        )
        assert abs(cf.npv(curve) + 1e6) < 1e-6


def test_base_period_dates_raise():
    with pytest.raises(ValueError):
        _ = FixedPeriod(dt(2023, 1, 1), dt(2022, 1, 1), dt(2024, 1, 1), "Q")
