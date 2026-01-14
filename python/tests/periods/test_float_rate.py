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

import os
from datetime import datetime as dt

import pytest
import rateslib.errors as err
from pandas import NA, Series
from pandas.testing import assert_series_equal
from rateslib import fixings
from rateslib.curves import Curve, LineCurve
from rateslib.data.fixings import FloatRateIndex, FloatRateSeries, _RFRRate
from rateslib.data.loader import FixingMissingForecasterError
from rateslib.default import NoInput
from rateslib.enums.parameters import FloatFixingMethod, SpreadCompoundMethod
from rateslib.periods.float_rate import rate_value
from rateslib.scheduling import Adjuster, Convention, Frequency, NamedCal


@pytest.fixture
def curve():
    return Curve(
        nodes={
            dt(2000, 1, 3): 1.00,
            dt(2000, 4, 3): 1.00 / (1.0 + 0.02 * 91 / 360),
        },
        convention="Act360",
        calendar="bus",
    )


@pytest.fixture
def curve2():
    return Curve(
        nodes={
            dt(2000, 1, 3): 1.00,
            dt(2000, 7, 3): 1.00 / (1.0 + 0.03 * 182 / 360),
        },
        convention="Act360",
        calendar="bus",
    )


@pytest.fixture
def line_curve():
    return LineCurve(
        nodes={
            dt(1999, 12, 30): 2.00,
            dt(2000, 3, 31): 10.0,
        },
        convention="Act360",
        calendar="bus",
    )


@pytest.fixture
def line_curve2():
    return LineCurve(
        nodes={
            dt(1999, 12, 30): 3.00,
            dt(2000, 3, 31): 10.0,
        },
        convention="Act360",
        calendar="bus",
    )


D = 1 / 360.0


class TestFloatRateIndex:
    def test_init_attributes(self):
        s = FloatRateSeries(
            lag=1,
            calendar="bus",
            convention="Act360",
            modifier="mf",
            eom=False,
        )
        assert s.calendar == NamedCal("bus")
        assert isinstance(s.calendar, NamedCal)
        assert s.convention == Convention.Act360
        assert s.modifier == Adjuster.ModifiedFollowing()
        assert not s.eom
        assert s.lag == 1

    def test_init_index_attrbutes(self):
        s = FloatRateIndex(
            frequency="Q",
            series="usd_ibor",
        )
        assert s.calendar == NamedCal("nyc")
        assert isinstance(s.calendar, NamedCal)
        assert s.convention == Convention.Act360
        assert s.modifier == Adjuster.ModifiedFollowing()
        assert not s.eom
        assert s.lag == 2
        assert s.frequency == Frequency.Months(3, None)


class TestIBORRate:
    def test_tenor_rate_from_curve(self, curve, line_curve):
        # test an IBOR rate is calculated correctly from a forecast curve
        for rate_curve in [curve, line_curve]:
            result = rate_value(
                rate_curve=rate_curve,
                rate_fixings=NoInput(0),
                start=dt(2000, 1, 3),
                end=dt(2000, 4, 3),
                stub=False,
                frequency="3M",
                fixing_method="IBOR",
                method_param=2,
                float_spread=18.0,
            )
            assert abs(result - 2.18) < 1e-12

    def test_tenor_rate_from_curve_fail_from_history(self, curve, line_curve):
        # test an IBOR rate cannot be forecast in the past
        for rate_curve in [curve, line_curve]:
            with pytest.raises(ValueError, match="`effective` date for rate period is before the"):
                rate_value(
                    rate_curve=rate_curve,
                    rate_fixings=NoInput(0),
                    start=dt(1980, 1, 3),
                    end=dt(1980, 4, 3),
                    stub=False,
                    frequency="3M",
                    fixing_method="IBOR",
                    method_param=2,
                    float_spread=18.0,
                )

    def test_tenor_rate_from_dict_curve(self, curve, curve2, line_curve, line_curve2):
        # test an IBOR rate is calculated correctly from a dict of forecast curves
        for rate_curve in [{"3m": curve, "6m": curve2}, {"3m": line_curve, "6m": line_curve2}]:
            result = rate_value(
                rate_curve=rate_curve,
                rate_fixings=NoInput(0),
                start=dt(2000, 1, 3),
                end=dt(2000, 4, 3),
                stub=False,
                frequency="3M",
                fixing_method="IBOR",
                method_param=2,
                float_spread=18.0,
            )
            assert abs(result - 2.18) < 1e-12

    def test_tenor_rate_from_scalar_fixing(self, curve, curve2, line_curve, line_curve2):
        # test an IBOR rate is calculated correctly from a direct scalar fixing
        for rate_curve in [
            curve,
            line_curve,
            {"3m": curve, "6m": curve2},
            {"3m": line_curve, "6m": line_curve2},
        ]:
            result = rate_value(
                rate_curve=rate_curve,
                rate_fixings=1.5,
                start=dt(2000, 1, 3),
                end=dt(2000, 4, 3),
                stub=False,
                frequency="3M",
                fixing_method="IBOR",
                method_param=2,
                float_spread=18.0,
            )
            assert abs(result - 1.68) < 1e-12

    def test_tenor_rate_from_fixing_str(self, curve, line_curve, curve2, line_curve2):
        # test an IBOR rate is calculated correctly from a fixing series
        fixings.add("TEST_VALUES_3M", Series(index=[dt(1999, 12, 30)], data=[1.2]))
        for rate_curve in [
            curve,
            line_curve,
            {"3m": curve, "6m": curve2},
            {"3m": line_curve, "6m": line_curve2},
        ]:
            result = rate_value(
                rate_curve=rate_curve,
                rate_fixings="TEST_VALUES_3M",
                start=dt(2000, 1, 3),
                end=dt(2000, 4, 3),
                stub=False,
                frequency="3M",
                fixing_method="IBOR",
                method_param=2,
                float_spread=18.0,
            )
            assert abs(result - 1.38) < 1e-12
        fixings.pop("TEST_VALUES_3M")

    def test_tenor_rate_from_fixing_str_fallback(self, curve, line_curve, curve2, line_curve2):
        # test an IBOR rate is calculated correctly from a curve when no fixing date exists
        name = str(hash(os.urandom(8)))
        fixings.add(f"{name}_3M", Series(index=[dt(2001, 1, 1)], data=[1.2]))
        for rate_curve in [
            curve,
            line_curve,
            {"3m": curve, "6m": curve2},
            {"3m": line_curve, "6m": line_curve2},
        ]:
            with pytest.warns(UserWarning, match=f"Fixings are provided in series: '{name}_3M',"):
                result = rate_value(
                    rate_curve=rate_curve,
                    rate_fixings=f"{name}_3M",
                    start=dt(2000, 1, 3),
                    end=dt(2000, 4, 3),
                    stub=False,
                    frequency="3M",
                    fixing_method="IBOR",
                    method_param=2,
                    float_spread=18.0,
                )
            assert abs(result - 2.18) < 1e-12
        fixings.pop(f"{name}_3M")

    def test_stub_rate_from_fixing_dict(self, curve, line_curve, curve2, line_curve2):
        # test an IBOR rate is calculated correctly from a fixing series
        fixings.add("TEST_VALUES_3M", Series(index=[dt(1999, 12, 30)], data=[1.2]))
        fixings.add("TEST_VALUES_6M", Series(index=[dt(1999, 12, 30)], data=[2.2]))
        for rate_curve in [
            curve,
            line_curve,
            {"3m": curve, "6m": curve2},
            {"3m": line_curve, "6m": line_curve2},
        ]:
            result = rate_value(
                rate_curve=rate_curve,
                rate_fixings="TEST_VALUES",
                start=dt(2000, 1, 3),
                end=dt(2000, 5, 18),
                stub=True,
                frequency="3M",
                fixing_method="IBOR",
                method_param=2,
                float_spread=18.0,
            )
            expected = 1.2 + 1.0 * 45 / 91 + 0.18
            assert abs(result - expected) < 1e-12
        fixings.pop("TEST_VALUES_3M")
        fixings.pop("TEST_VALUES_6M")

    def test_stub_rate_from_fixing_dict_missing_data(self, curve, line_curve, curve2, line_curve2):
        # test an IBOR rate is calculated correctly from a fixing series
        fixings.add("TEST_VALUES_3M", Series(index=[dt(1999, 12, 1)], data=[1.2]))
        fixings.add("TEST_VALUES_6M", Series(index=[dt(1999, 12, 1)], data=[2.2]))
        for rate_curve, expected in [
            (curve, 2.18249787441),
            (line_curve, 2.180),
            ({"3m": curve, "6m": curve2}, 2.674505494505512),
            ({"3m": line_curve, "6m": line_curve2}, 2.6745054945054947),
        ]:
            result = rate_value(
                rate_curve=rate_curve,
                rate_fixings="TEST_VALUES",
                start=dt(2000, 1, 3),
                end=dt(2000, 5, 18),
                stub=True,
                frequency="3M",
                fixing_method="IBOR",
                method_param=2,
                float_spread=18.0,
            )
            # expected = 1.2 + 1.0 * 45 / 91 + 0.18
            assert abs(result - expected) < 1e-11
        fixings.pop("TEST_VALUES_3M")
        fixings.pop("TEST_VALUES_6M")

    def test_stub_rate_from_fixing_dict_1tenor(self, curve, line_curve, curve2, line_curve2):
        # test an IBOR rate is calculated correctly from a fixing series
        fixings.add("TEST_VALUES_6M", Series(index=[dt(1999, 12, 30)], data=[4.1]))
        for rate_curve in [
            curve,
            line_curve,
            {"3m": curve, "6m": curve2},
            {"3m": line_curve, "6m": line_curve2},
        ]:
            result = rate_value(
                rate_curve=rate_curve,
                rate_fixings="TEST_VALUES",
                start=dt(2000, 1, 3),
                end=dt(2000, 5, 18),
                stub=True,
                frequency="3M",
                fixing_method="IBOR",
                method_param=2,
                float_spread=18.0,
            )
            expected = 4.1 + 0.18
            assert abs(result - expected) < 1e-12
        fixings.pop("TEST_VALUES_6M")

    def test_stub_rate_from_scalar_fixing(self, curve, line_curve, curve2, line_curve2):
        # test an IBOR stub rate is calculated correctly from a fixing scalar
        for rate_curve in [
            curve,
            line_curve,
            {"3m": curve, "6m": curve2},
            {"3m": line_curve, "6m": line_curve2},
        ]:
            result = rate_value(
                rate_curve=rate_curve,
                rate_fixings=9.9,
                start=dt(2000, 1, 3),
                end=dt(2000, 5, 18),
                stub=True,
                frequency="3M",
                fixing_method="IBOR",
                method_param=2,
                float_spread=18.0,
            )
            expected = 9.9 + 0.18
            assert abs(result - expected) < 1e-12

    def test_stub_rate_from_dict_curve(self, curve, curve2, line_curve, line_curve2):
        # test an IBOR stub rate is calculated correctly from a dict of forecast curves
        for rate_curve in [{"3m": curve, "6m": curve2}, {"3m": line_curve, "6m": line_curve2}]:
            result = rate_value(
                rate_curve=rate_curve,
                rate_fixings=NoInput(0),
                start=dt(2000, 1, 3),
                end=dt(2000, 5, 18),
                stub=True,
                frequency="3M",
                fixing_method="IBOR",
                method_param=2,
                float_spread=18.0,
            )
            expected = 2.0 * 46 / 91 + 3.0 * 45 / 91 + 0.18
            assert abs(result - expected) < 1e-12

    def test_stub_rate_from_dict_curve_long_curves(self, curve, curve2, line_curve, line_curve2):
        # test an IBOR stub rate is calculated correctly from a dict of forecast curves
        for rate_curve in [{"9m": curve, "6m": curve2}, {"9m": line_curve, "6m": line_curve2}]:
            with pytest.warns(UserWarning, match="Interpolated stub period has a length shorter"):
                result = rate_value(
                    rate_curve=rate_curve,
                    rate_fixings=NoInput(0),
                    start=dt(2000, 1, 3),
                    end=dt(2000, 5, 18),
                    stub=True,
                    frequency="3M",
                    fixing_method="IBOR",
                    method_param=2,
                    float_spread=18.0,
                )
            expected = 3.0 + 0.18  # just the 6m curve
            assert abs(result - expected) < 1e-12

    def test_stub_rate_from_dict_curve_short_curves(self, curve, curve2, line_curve, line_curve2):
        # test an IBOR stub rate is calculated correctly from a dict of forecast curves
        for rate_curve in [{"3m": curve, "1m": curve2}, {"3m": line_curve, "1m": line_curve2}]:
            with pytest.warns(UserWarning, match="Interpolated stub period has a length longer"):
                result = rate_value(
                    rate_curve=rate_curve,
                    rate_fixings=NoInput(0),
                    start=dt(2000, 1, 3),
                    end=dt(2000, 5, 18),
                    stub=True,
                    frequency="3M",
                    fixing_method="IBOR",
                    method_param=2,
                    float_spread=18.0,
                )
            expected = 2.0 + 0.18  # just the 3m curve
            assert abs(result - expected) < 1e-12

    def test_stub_rate_from_single_curve(self, curve, curve2, line_curve, line_curve2):
        # test an IBOR stub rate is calculated from a single forecast curve
        for rate_curve in [curve, line_curve]:
            result = rate_value(
                rate_curve=rate_curve,
                rate_fixings=NoInput(0),
                start=dt(2000, 1, 3),
                end=dt(2000, 5, 18),
                stub=True,
                frequency="3M",
                fixing_method="IBOR",
                method_param=2,
                float_spread=18.0,
            )
            expected = 2.0 + 0.18
            assert abs(result - expected) < 3e-3

    def test_stub_rate_from_dict_curve_on_fixing_fail(self, curve, curve2, line_curve, line_curve2):
        # test an IBOR stub rate is calculated from curve when no fixing is found
        for rate_curve in [{"3m": curve, "6m": curve2}, {"3m": line_curve, "6m": line_curve2}]:
            result = rate_value(
                rate_curve=rate_curve,
                rate_fixings="NO_DATA",
                start=dt(2000, 1, 3),
                end=dt(2000, 5, 18),
                stub=True,
                frequency="3M",
                fixing_method="IBOR",
                method_param=2,
                float_spread=18.0,
            )
            expected = 2.0 * 46 / 91 + 3.0 * 45 / 91 + 0.18
            assert abs(result - expected) < 1e-12


class TestRFRRate:
    def test_pandas_series_update_mechanism(self):
        # rateslib relies on the following mechanism. Test this for compatibility.
        a = Series(index=[3, 4, 5, 6, 7], data=[NA, NA, NA, NA, NA])
        b = Series(index=[1, 2, 3, 4, 5], data=[2, 4, 6, 8, 10])
        a.update(b)
        assert a.index.to_list() == [3, 4, 5, 6, 7]
        assert a.to_list() == [6, 8, 10, NA, NA]

    def test_populate_rates_from_rate_fixings(self):
        fixing_rates = Series(
            index=[dt(2000, 1, 1), dt(2000, 1, 2), dt(2000, 1, 3), dt(2000, 1, 4)], data=NA
        )
        fixings.add(
            "USD_SOFR_1B",
            Series(index=[dt(1999, 1, 1), dt(2000, 1, 1), dt(2000, 1, 2)], data=[1.0, 2.0, 3.0]),
        )
        result, _, _ = _RFRRate._push_rate_fixings_as_series_to_fixing_rates(
            fixing_rates, "USD_SOFR_1B", FloatFixingMethod.RFRPaymentDelay, 0
        )
        assert_series_equal(
            result,
            Series(
                index=[dt(2000, 1, 1), dt(2000, 1, 2), dt(2000, 1, 3), dt(2000, 1, 4)],
                data=[2.0, 3.0, NA, NA],
            ),
        )
        fixings.pop("USD_SOFR_1B")

    def test_populate_rates_from_rate_fixings_all_filled(self):
        fixing_rates = Series(index=[dt(2000, 1, 1), dt(2000, 1, 2), dt(2000, 1, 3)], data=NA)
        fixings.add(
            "USD_SOFR_1B",
            Series(
                index=[
                    dt(1999, 1, 1),
                    dt(2000, 1, 1),
                    dt(2000, 1, 2),
                    dt(2000, 1, 3),
                    dt(2000, 1, 4),
                ],
                data=[1.0, 2.0, 3.0, 4.0, 5.0],
            ),
        )
        result, _, _ = _RFRRate._push_rate_fixings_as_series_to_fixing_rates(
            fixing_rates, "USD_SOFR_1B", FloatFixingMethod.RFRPaymentDelay, 0
        )
        assert_series_equal(
            result,
            Series(
                index=[dt(2000, 1, 1), dt(2000, 1, 2), dt(2000, 1, 3)],
                data=[2.0, 3.0, 4.0],
                dtype=object,
            ),
        )
        fixings.pop("USD_SOFR_1B")

    def test_populate_rates_from_rate_fixings_none_filled(self):
        fixing_rates = Series(index=[dt(2000, 1, 1), dt(2000, 1, 2)], data=NA)
        fixings.add(
            "USD_SOFR_1B",
            Series(index=[dt(1999, 1, 1)], data=[1.0]),
        )
        result, _, _ = _RFRRate._push_rate_fixings_as_series_to_fixing_rates(
            fixing_rates, "USD_SOFR_1B", FloatFixingMethod.RFRPaymentDelay, 0
        )
        assert_series_equal(
            result,
            Series(index=[dt(2000, 1, 1), dt(2000, 1, 2)], data=[NA, NA], dtype=object),
        )
        fixings.pop("USD_SOFR_1B")

    def test_populate_rates_from_rate_fixings_missing_fixing(self):
        fixing_rates = Series(
            index=[dt(2000, 1, 1), dt(2000, 1, 2), dt(2000, 1, 3), dt(2000, 1, 4)], data=NA
        )
        fixings.add("USD_SOFR_1B", Series(index=[dt(1999, 1, 1), dt(2000, 1, 2)], data=[1.0, 3.0]))
        with pytest.raises(ValueError, match="The fixings series 'USD_SOFR_1B' for the RFR 1B rat"):
            _RFRRate._push_rate_fixings_as_series_to_fixing_rates(
                fixing_rates, "USD_SOFR_1B", FloatFixingMethod.RFRPaymentDelay, 0
            )
        fixings.pop("USD_SOFR_1B")

    @pytest.mark.skip(reason="Not expecting the most recent fixing is an allowed oversight.")
    def test_populate_rates_from_rate_fixings_extra_fixing(self):
        # this test will fail becuase of the validation that is applied. The missing fixing
        # is right at the end of the series and is not detected at the populated/unpopulated
        # crossover point.
        fixing_rates = Series(
            index=[dt(2000, 1, 1), dt(2000, 1, 2), dt(2000, 1, 4), dt(2000, 1, 5)], data=NA
        )
        fixings.add(
            "USD_SOFR_1B",
            Series(index=[dt(2000, 1, 1), dt(2000, 1, 2), dt(2000, 1, 3)], data=[1.0, 2.0, 3.0]),
        )
        with pytest.warns(UserWarning, match="The fixings series 'USD_SOFR' for the RFR 1B rates"):
            _RFRRate._push_rate_fixings_as_series_to_fixing_rates(fixing_rates, "USD_SOFR_1B")
        fixings.pop("USD_SOFR_1B")

    def test_populate_rates_from_rate_fixings_extra_fixing2(self):
        # the lengths of the expected fixings in the return and fixing series is different and
        # detected.
        fixing_rates = Series(
            index=[dt(2000, 1, 1), dt(2000, 1, 2), dt(2000, 1, 4), dt(2000, 1, 5)], data=NA
        )
        fixings.add(
            "USD_SOFR_1B",
            Series(
                index=[dt(2000, 1, 1), dt(2000, 1, 2), dt(2000, 1, 3), dt(2000, 1, 4)],
                data=[1.0, 2.0, 3.0, 4.0],
            ),
        )
        with pytest.warns(UserWarning, match="The fixings series 'USD_SOFR_1B' for the RFR 1B rat"):
            _RFRRate._push_rate_fixings_as_series_to_fixing_rates(
                fixing_rates, "USD_SOFR_1B", FloatFixingMethod.RFRPaymentDelay, 0
            )
        fixings.pop("USD_SOFR_1B")

    @pytest.mark.parametrize(
        ("fixing_method"),
        [FloatFixingMethod.RFRPaymentDelay, FloatFixingMethod.RFRObservationShift],
    )
    @pytest.mark.parametrize(
        ("spread_compound_method", "float_spread"),
        [
            (SpreadCompoundMethod.NoneSimple, 10.0),
            (SpreadCompoundMethod.ISDACompounding, 0.0),
            (SpreadCompoundMethod.ISDAFlatCompounding, 0.0),
        ],
    )
    def test_efficient_calc(self, curve, fixing_method, spread_compound_method, float_spread):
        # rates
        r0 = curve._rate_with_raise(dt(2000, 1, 3), dt(2000, 1, 4))
        r1 = curve._rate_with_raise(dt(2000, 1, 4), dt(2000, 1, 5))
        r2 = curve._rate_with_raise(dt(2000, 1, 5), dt(2000, 1, 6))
        r3 = curve._rate_with_raise(dt(2000, 1, 6), dt(2000, 1, 7))

        result = rate_value(
            start=dt(2000, 1, 4),
            end=dt(2000, 1, 7),
            rate_curve=curve,
            spread_compound_method=spread_compound_method,
            float_spread=float_spread,
            method_param=1,
        )

        if fixing_method == FloatFixingMethod.RFRObservationShift:
            expected = (
                (1 + r0 / 36000) * (1 + r1 / 36000) * (1 + r2 / 36000) - 1
            ) * 36000 / 3.0 + float_spread / 100.0
        else:
            expected = (
                (1 + r1 / 36000) * (1 + r2 / 36000) * (1 + r3 / 36000) - 1
            ) * 36000 / 3.0 + float_spread / 100.0

        assert abs(result - expected) < 1e-10

    def test_semi_inefficient_calc_with_populated_fixings(self, curve):
        fixings.add("USD_SOFR_1B", Series(index=[dt(2000, 1, 3), dt(2000, 1, 4)], data=[1.5, 1.7]))
        r2 = curve._rate_with_raise(dt(2000, 1, 5), dt(2000, 1, 6))
        r3 = curve._rate_with_raise(dt(2000, 1, 6), dt(2000, 1, 7))

        result = rate_value(
            start=dt(2000, 1, 3),
            end=dt(2000, 1, 7),
            rate_curve=curve,
            spread_compound_method=SpreadCompoundMethod.NoneSimple,
            float_spread=10.0,
            method_param=0,
            rate_fixings="USD_SOFR_1B",
        )
        expected = (
            (1 + 0.015 / 360) * (1 + 0.017 / 360) * (1 + r2 / 36000) * (1 + r3 / 36000) - 1
        ) * 36000 / 4 + 0.1
        fixings.pop("USD_SOFR_1B")
        assert abs(result - expected) < 1e-10

    def test_inefficient_calc_with_populated_fixings_no_curve_raises(self, curve):
        fixings.add("USD_SOFR_1B", Series(index=[dt(2000, 1, 3), dt(2000, 1, 4)], data=[1.5, 1.7]))
        with pytest.raises(
            FixingMissingForecasterError, match=err.VE_NEEDS_RATE_POPULATE_FIXINGS[:25]
        ):
            rate_value(
                start=dt(2000, 1, 3),
                end=dt(2000, 1, 7),
                rate_curve=NoInput(0),
                spread_compound_method=SpreadCompoundMethod.ISDACompounding,
                float_spread=10.0,
                method_param=0,
                rate_fixings="USD_SOFR_1B",
                rate_series="usd_rfr",
            )
        fixings.pop("USD_SOFR_1B")

    def test_inefficient_calc_with_lockout_too_long_raises(self, curve):
        # the lockout param is invalid
        with pytest.raises(ValueError, match=err.VE_LOCKOUT_METHOD_PARAM[:25]):
            rate_value(
                start=dt(2000, 1, 3),
                end=dt(2000, 1, 7),
                rate_curve=curve,
                spread_compound_method=SpreadCompoundMethod.ISDACompounding,
                float_spread=10.0,
                method_param=9,
                fixing_method=FloatFixingMethod.RFRLockout,
            )

    @pytest.mark.parametrize("curve_type", ["values", "dfs"])
    def test_inefficient_calc_with_populated_fixings(self, curve_type, curve, line_curve):
        rate_curve = curve if curve_type == "dfs" else line_curve
        fixings.add("USD_SOFR_1B", Series(index=[dt(2000, 1, 3), dt(2000, 1, 4)], data=[1.5, 1.7]))
        r2 = rate_curve._rate_with_raise(dt(2000, 1, 5), dt(2000, 1, 6))
        r3 = rate_curve._rate_with_raise(dt(2000, 1, 6), dt(2000, 1, 7))

        result = rate_value(
            start=dt(2000, 1, 3),
            end=dt(2000, 1, 7),
            rate_curve=rate_curve,
            spread_compound_method=SpreadCompoundMethod.NoneSimple,
            float_spread=10.0,
            method_param=0,
            fixing_method=FloatFixingMethod.RFRLookback,
            rate_fixings="USD_SOFR_1B",
        )
        expected = (
            (1 + 0.015 / 360) * (1 + 0.017 / 360) * (1 + r2 / 36000) * (1 + r3 / 36000) - 1
        ) * 36000 / 4 + 0.1
        fixings.pop("USD_SOFR_1B")
        assert abs(result - expected) < 1e-10

    def test_inefficient_calc_with_non_overlapping_fixings(self, curve):
        fixings.add("USD_SOFR_1B", Series(index=[dt(2001, 1, 1)], data=[100.0]))

        rate_value(
            start=dt(2000, 1, 4),
            end=dt(2000, 1, 7),
            rate_curve=curve,
            spread_compound_method=SpreadCompoundMethod.NoneSimple,
            float_spread=0.0,
            method_param=0,
            rate_fixings="USD_SOFR_1B",
        )
        fixings.pop("USD_SOFR_1B")

    @pytest.mark.parametrize(
        ("fixing_method", "expected"),
        [
            (
                FloatFixingMethod.RFRPaymentDelay,
                ((1 + 0.04 * D) * (1 + 0.05 * D) * (1 + 0.06 * D) * (1 + 0.07 * D) - 1)
                * 100
                / (4 * D),
            ),
            (
                FloatFixingMethod.RFRObservationShift,
                ((1 + 0.02 * D) * (1 + 0.03 * D) * (1 + 0.04 * D) * (1 + 0.05 * D) - 1)
                * 100
                / (4 * D),
            ),
            (
                FloatFixingMethod.RFRLockout,
                ((1 + 0.04 * D) * (1 + 0.05 * D) * (1 + 0.05 * D) * (1 + 0.05 * D) - 1)
                * 100
                / (4 * D),
            ),
            (
                FloatFixingMethod.RFRLookback,
                ((1 + 0.02 * D) * (1 + 0.03 * D) * (1 + 0.04 * D) * (1 + 0.05 * D) - 1)
                * 100
                / (4 * D),
            ),
            (
                FloatFixingMethod.RFRPaymentDelayAverage,
                (4 + 5 + 6 + 7) / 4,
            ),
            (
                FloatFixingMethod.RFRObservationShiftAverage,
                (2 + 3 + 4 + 5) / 4,
            ),
            (
                FloatFixingMethod.RFRLockoutAverage,
                (4 + 5 + 5 + 5) / 4,
            ),
            (
                FloatFixingMethod.RFRLookbackAverage,
                (2 + 3 + 4 + 5) / 4,
            ),
        ],
    )
    def test_fixing_methods(self, fixing_method, expected):
        rate_curve = LineCurve(
            nodes={
                dt(2000, 1, 1): 2.0,
                dt(2000, 1, 2): 3.0,
                dt(2000, 1, 3): 4.0,
                dt(2000, 1, 4): 5.0,
                dt(2000, 1, 5): 6.0,
                dt(2000, 1, 6): 7.0,
                dt(2000, 1, 7): 8.0,
            },
            convention="act360",
            calendar="all",
        )
        result = rate_value(
            start=dt(2000, 1, 3),
            end=dt(2000, 1, 7),
            rate_curve=rate_curve,
            spread_compound_method=SpreadCompoundMethod.NoneSimple,
            float_spread=0.0,
            method_param=2,
            fixing_method=fixing_method,
        )
        assert abs(result - expected) < 1e-10

    @pytest.mark.parametrize(
        "fixing_method", [FloatFixingMethod.RFRPaymentDelay, FloatFixingMethod.RFRLockout]
    )
    def test_bus252_convention(self, fixing_method):
        rate_curve = Curve(
            nodes={
                dt(2000, 1, 3): 1.0,
                dt(2000, 1, 17): 0.999,
            },
            convention="bus252",
            calendar="bus",
        )
        result = rate_value(
            start=dt(2000, 1, 6),
            end=dt(2000, 1, 11),
            rate_curve=rate_curve,
            spread_compound_method=SpreadCompoundMethod.NoneSimple,
            float_spread=0.0,
            method_param=0,
            fixing_method=fixing_method,
        )

        r1 = rate_curve._rate_with_raise(dt(2000, 1, 6), "1b")
        r2 = rate_curve._rate_with_raise(dt(2000, 1, 7), "1b")
        r3 = rate_curve._rate_with_raise(dt(2000, 1, 10), "1b")
        expected = ((1 + r1 / 25200) * (1 + r2 / 25200) * (1 + r3 / 25200) - 1) * 25200 / 3
        assert abs(result - expected) < 1e-10
