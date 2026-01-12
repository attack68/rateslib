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
from pandas import Series
from rateslib import fixings
from rateslib.data.fixings import (
    FloatRateIndex,
    FloatRateSeries,
    FXFixing,
    FXIndex,
    IBORFixing,
    IBORStubFixing,
    RFRFixing,
)
from rateslib.data.loader import FixingMissingDataError
from rateslib.enums import FloatFixingMethod, SpreadCompoundMethod
from rateslib.enums.generics import NoInput
from rateslib.enums.parameters import IndexMethod
from rateslib.errors import VE_INDEX_BASE_NO_STR
from rateslib.periods import Cashflow, FloatPeriod
from rateslib.scheduling.frequency import Frequency


class TestIndexParams:
    def test_index_lookup_and_populate_from_str_fixings(self):
        rpi = Series(index=[dt(2000, 1, 1), dt(2000, 1, 2)], data=[101.0, 103.0])
        name = str(hash(os.urandom(8)))
        fixings.add(name, rpi)
        c = Cashflow(
            payment=dt(2000, 1, 2),
            notional=1e6,
            index_fixings=name,
            index_method=IndexMethod.Curve,
            index_base_date=dt(2000, 1, 1),
            index_lag=0,
        )
        assert c.index_params.index_fixing.value == 103.0
        assert c.index_params.index_base.value == 101.0
        fixings.pop(name)

    def test_lookup_and_populate_from_series_fixings(self):
        rpi = Series(index=[dt(2000, 1, 1), dt(2000, 1, 2)], data=[101.0, 103.0])
        with pytest.warns(FutureWarning, match=err.FW_FIXINGS_AS_SERIES[:25]):
            c = Cashflow(
                payment=dt(2000, 1, 2),
                notional=1e6,
                index_fixings=rpi,
                index_method=IndexMethod.Curve,
                index_base_date=dt(2000, 1, 1),
                index_lag=0,
            )
        assert c.index_params.index_fixing.value == 103.0
        assert c.index_params.index_base.value == 101.0

    def test_immutable_index_fixings(self):
        c = Cashflow(
            payment=dt(2000, 1, 2),
            notional=1e6,
            index_fixings=0.0,
            index_method=IndexMethod.Curve,
            index_base_date=dt(2000, 1, 1),
            index_lag=0,
        )
        with pytest.raises(ValueError, match=err.VE_ATTRIBUTE_IS_IMMUTABLE.format("index_fixing")):
            c.index_params.index_fixing = 2.0

    def test_index_fixings_determined_once(self):
        # a change in the datastore will not affect an already loaded fixing for the period
        c = Cashflow(
            payment=dt(2000, 1, 2),
            notional=1e6,
            index_fixings="rpi",
            index_method=IndexMethod.Curve,
            index_base_date=dt(2000, 1, 1),
            index_lag=0,
        )
        rpi = Series(index=[dt(2000, 1, 1), dt(2000, 1, 2)], data=[101.0, 103.0])
        fixings.add("rpi", rpi)
        before1 = c.index_params.index_fixing.value
        before2 = c.index_params.index_base.value

        fixings.pop("rpi")
        rpi2 = Series(index=[dt(2000, 1, 1), dt(2000, 1, 2)], data=[201.0, 203.0])
        fixings.add("rpi", rpi2)
        assert c.index_params.index_fixing.value == before1
        assert c.index_params.index_base.value == before2

    @pytest.mark.parametrize("int_or_float", [3, 3.0])
    def test_index_fixings_as_scalar(self, int_or_float):
        # a scalar value for `index_fixings` will only impact `index_fixing` and not `index_base`
        c = Cashflow(
            payment=dt(2000, 1, 2),
            notional=1e6,
            index_fixings=int_or_float,
            index_method=IndexMethod.Curve,
            index_base_date=dt(2000, 1, 1),
            index_lag=0,
        )
        assert c.index_params.index_fixing.value == int_or_float
        assert c.index_params.index_base.value == NoInput(0)

    def test_index_base_as_str_raises(self):
        # index base as string series identifier will not work
        with pytest.raises(ValueError, match=VE_INDEX_BASE_NO_STR):
            Cashflow(
                payment=dt(2000, 1, 2),
                notional=1e6,
                index_fixings=0.0,
                index_method=IndexMethod.Curve,
                index_base_date=dt(2000, 1, 1),
                index_base="str",
                index_lag=0,
            )

    def test_index_realtime_updates(self):
        # test that the first series contains no data and an update adds new data
        rpi = Series(index=[dt(2000, 1, 1), dt(2000, 1, 2)], data=[101.0, 103.0])
        name = str(hash(os.urandom(8)))
        fixings.add(name, rpi)
        c = Cashflow(
            payment=dt(2000, 1, 3),
            notional=1e6,
            index_fixings=name,
            index_method=IndexMethod.Curve,
            index_base_date=dt(2000, 1, 3),
            index_lag=0,
        )
        assert c.index_params.index_fixing.value == NoInput(0)
        assert c.index_params.index_base.value == NoInput(0)
        fixings.pop(name)
        rpi = Series(index=[dt(2000, 1, 1), dt(2000, 1, 3)], data=[101.0, 105.0])
        fixings.add(name, rpi)
        assert c.index_params.index_fixing.value == 105.0
        assert c.index_params.index_base.value == 105.0


class TestSettlementParams:
    def test_fx_fixings_no_input(
        self,
    ):
        c = Cashflow(currency="usd", pair="eurusd", payment=dt(2000, 1, 2), notional=2.0)
        assert isinstance(c.non_deliverable_params.fx_fixing, FXFixing)
        assert c.non_deliverable_params.fx_fixing.value is NoInput(0)

    def test_fx_fixings_scalar_input(self):
        c = Cashflow(
            currency="usd", pair="eurusd", payment=dt(2000, 1, 2), notional=2.0, fx_fixings=2.0
        )
        assert c.non_deliverable_params.fx_fixing.value == 2.0
        assert c.non_deliverable_params.fx_fixing._state == 0

    def test_fx_fixings_series_input(self):
        s = Series(index=[dt(1999, 12, 29), dt(1999, 12, 30)], data=[1.1, 2.1])
        c = Cashflow(
            currency="usd", pair="eurusd", payment=dt(2000, 1, 2), notional=2.0, fx_fixings=s
        )
        assert c.non_deliverable_params.fx_fixing._state == 0
        assert c.non_deliverable_params.fx_fixing.value == 2.1

    def test_fx_fixings_str_input(self):
        s = Series(index=[dt(1999, 12, 29), dt(1999, 12, 30)], data=[1.1, 2.1])
        name = str(hash(os.urandom(8)))
        fixings.add(name + "_eurusd", s)
        c = Cashflow(
            currency="usd", pair="eurusd", payment=dt(2000, 1, 2), notional=2.0, fx_fixings=name
        )
        assert c.non_deliverable_params.fx_fixing.value == 2.1
        assert isinstance(c.non_deliverable_params.fx_fixing.identifier, str)
        assert c.non_deliverable_params.fx_fixing._state == hash(fixings[name + "_eurusd"][0])
        fixings.pop(name + "_eurusd")

    def test_fx_fixings_str_state_cache(self):
        s = Series(index=[dt(2000, 1, 1), dt(2000, 1, 2)], data=[1.1, 2.1])
        name = str(hash(os.urandom(8)))
        fixings.add(name + "_eurusd", s)
        c = Cashflow(
            currency="usd",
            pair="eurusd",
            payment=dt(2000, 1, 3),  # <- not in Series
            notional=2.0,
            fx_fixings=name,
        )
        assert c.non_deliverable_params.fx_fixing.value is NoInput(0)
        assert isinstance(c.non_deliverable_params.fx_fixing.identifier, str)
        # states match the hash because the FXFixing uses composite FXFixingMajors
        assert c.non_deliverable_params.fx_fixing._state == hash(fixings[name + "_eurusd"][0])

        assert c.non_deliverable_params.fx_fixing.value is NoInput(0)
        assert c.non_deliverable_params.fx_fixing._state == hash(fixings[name + "_eurusd"][0])
        fixings.pop(name + "_eurusd")

    def test_fx_fixing_cashflow(self):
        s = Series(index=[dt(1999, 12, 29), dt(1999, 12, 30)], data=[1.1, 2.1])
        name = str(hash(os.urandom(8)))
        fixings.add(name + "_eurusd", s)
        c = Cashflow(
            notional=100,
            payment=dt(2000, 1, 2),
            currency="usd",
            pair="eurusd",
            fx_fixings=name,
        )
        cf = c.cashflows()
        assert cf["FX Fixing"] == 2.1
        fix = c.non_deliverable_params.fx_fixing.value
        assert fix == 2.1
        fixings.pop(name + "_eurusd")

    def test_immutable_fx_fixing(self):
        c = Cashflow(
            payment=dt(2000, 1, 2),
            notional=1e6,
            currency="usd",
            pair="eurusd",
            fx_fixings=0.0,
        )
        with pytest.raises(ValueError, match=err.VE_ATTRIBUTE_IS_IMMUTABLE.format("fx_fixing")):
            c.non_deliverable_params.fx_fixing = 2.0

    def test_fx_missing_data_raises(self):
        s = Series(index=[dt(1999, 12, 29), dt(2000, 1, 1)], data=[1.1, 2.1])
        name = str(hash(os.urandom(8)))
        fixings.add(name + "_eurusd", s)
        c = Cashflow(
            notional=100,
            payment=dt(2000, 1, 2),
            currency="usd",
            pair="eurusd",
            fx_fixings=name,
        )
        with pytest.raises(FixingMissingDataError, match="Fixing lookup for date "):
            c.non_deliverable_params.fx_fixing.value
        fixings.pop(name + "_eurusd")

    def test_fx_missing_data_raises_cross(self):
        s = Series(index=[dt(1999, 12, 29), dt(1999, 12, 30)], data=[1.1, 2.1])
        s2 = Series(index=[dt(1999, 12, 29), dt(2000, 1, 1)], data=[1.1, 2.1])
        name = str(hash(os.urandom(8)))
        fixings.add(name + "_usdinr", s)
        fixings.add(name + "_usdrub", s2)
        c = Cashflow(
            notional=100,
            payment=dt(2000, 1, 2),
            currency="inr",
            pair=FXIndex("inrrub", "mum|fed", 2, "mum", -2),
            fx_fixings=name,
        )
        with pytest.raises(FixingMissingDataError, match="Fixing lookup for date "):
            c.non_deliverable_params.fx_fixing.value
        fixings.pop(name + "_usdinr")
        fixings.pop(name + "_usdrub")


class TestRateParams:
    def test_rate_fixings_input_as_str_out_of_range(
        self,
    ):
        s = Series(index=[dt(1999, 1, 1), dt(1999, 1, 2)], data=[1.1, 2.1])
        fixings.add("IBOR123dfgs_1M", s)
        c = FloatPeriod(
            start=dt(2000, 1, 1),
            end=dt(2000, 2, 1),
            payment=dt(2000, 2, 1),
            notional=2.0,
            frequency="M",
            fixing_series="usd_ibor",
            fixing_method="IBOR",
            method_param=2,
            rate_fixings="IBOR123dfgs",
        )
        assert c.rate_params.rate_fixing.value == NoInput(0)
        assert c.rate_params.rate_fixing.value == NoInput(0)
        assert c.rate_params.rate_fixing.identifier == "IBOR123dfgs_1M".upper()
        assert c.rate_params.rate_fixing._state == fixings["IBOR123dfgs_1M"][0]
        fixings.pop("IBOR123dfgs_1M")

    def test_rate_fixings_no_input(
        self,
    ):
        c = FloatPeriod(
            start=dt(2000, 1, 1),
            end=dt(2000, 2, 1),
            payment=dt(2000, 2, 1),
            notional=2.0,
            frequency="M",
            fixing_method="IBOR",
            fixing_series="usd_ibor",
            method_param=2,
            rate_fixings=NoInput(0),
        )
        assert c.rate_params.rate_fixing.value == NoInput(0)
        assert c.rate_params.rate_fixing.value == NoInput(0)
        assert c.rate_params.rate_fixing._state == 0

    def test_rate_fixings_scalar(
        self,
    ):
        c = FloatPeriod(
            start=dt(2000, 1, 1),
            end=dt(2000, 2, 1),
            payment=dt(2000, 2, 1),
            notional=2.0,
            frequency="M",
            fixing_method="IBOR",
            fixing_series="usd_ibor",
            method_param=2,
            rate_fixings=2.5,
        )
        assert c.rate_params.rate_fixing.value == 2.5
        assert c.rate_params.rate_fixing.value == 2.5
        assert c.rate_params.rate_fixing._state == 0

    def test_ibor_fixing_load(self):
        name = str(hash(os.urandom(8)))
        fixings.add(f"{name}_3M", Series(index=[dt(2022, 1, 3)], data=[55.0]))
        f = IBORFixing(
            accrual_start=dt(2022, 1, 5),
            rate_index=FloatRateIndex(
                frequency=Frequency.Months(3, None),
                series="eur_ibor",
            ),
            identifier=f"{name}_3M",
        )
        assert f.value == 55.0
        assert f._state == fixings[f"{name}_3M"][0]

    def test_stub_ibor_fixing_load(self):
        name = str(hash(os.urandom(8)))
        fixings.add(f"{name}_3M", Series(index=[dt(2022, 1, 3)], data=[55.0]))
        fixings.add(f"{name}_6M", Series(index=[dt(2022, 1, 3)], data=[65.0]))
        index_series = FloatRateIndex(
            frequency=Frequency.Months(3, None),
            series="eur_ibor",
        ).series
        f = IBORStubFixing(
            accrual_start=dt(2022, 1, 5),
            accrual_end=dt(2022, 5, 21),
            rate_series=index_series,
            identifier=name,
        )
        assert f.value == 55 * 45 / 91 + 65 * 46 / 91
        fixings.pop(f"{name}_3M")
        fixings.pop(f"{name}_6M")

    def test_rfr_fixings_load(self):
        name = str(hash(os.urandom(8)))
        fixings.add(
            f"{name}_1B",
            Series(
                index=[dt(2023, 2, 8), dt(2023, 2, 9), dt(2023, 2, 10), dt(2023, 2, 13)],
                data=[1.0, 2.0, 3.0, 4.0],
            ),
        )
        rate_index = FloatRateIndex(
            frequency="1B",
            series="usd_rfr",
        )
        f = RFRFixing(
            accrual_start=dt(2023, 2, 8),
            accrual_end=dt(2023, 2, 13),
            rate_index=rate_index,
            fixing_method=FloatFixingMethod.RFRPaymentDelay,
            method_param=0,
            spread_compound_method=SpreadCompoundMethod.NoneSimple,
            identifier=f"{name}_1B",
            float_spread=0.0,
        )
        result = f.value
        expected = ((1 + 1 / 36000) * (1 + 2 / 36000) * (1 + 3 * 3 / 36000) - 1) * 36000 / 5
        assert abs(result - expected) < 1e-10

        f = RFRFixing(
            accrual_start=dt(2023, 2, 8),
            accrual_end=dt(2023, 2, 17),
            rate_index=rate_index,
            fixing_method=FloatFixingMethod.RFRPaymentDelay,
            method_param=0,
            spread_compound_method=SpreadCompoundMethod.NoneSimple,
            identifier=f"{name}_1B",
            float_spread=0.0,
        )
        result = f.value
        assert result == NoInput(0)

    def test_stub_ibor_warns_no_series(self):
        with pytest.warns(UserWarning, match=err.UW_NO_TENORS[:15]):
            fix = IBORStubFixing(
                accrual_start=dt(2022, 1, 5),
                accrual_end=dt(2022, 5, 21),
                rate_series=FloatRateSeries(
                    lag=0,
                    calendar="tgt",
                    convention="act360",
                    modifier="mf",
                    eom=False,
                ),
                identifier="NOT_AVAILABLE",
            )
        assert isinstance(fix.value, NoInput)

    def test_rfr_fixing_identifier(self):
        p = FloatPeriod(
            start=dt(2000, 1, 1),
            end=dt(2000, 4, 1),
            frequency=Frequency.Months(3, None),
            payment=dt(2000, 1, 4),
            fixing_method=FloatFixingMethod.RFRPaymentDelay,
            rate_fixings="TEST",
        )
        assert p.rate_params.fixing_identifier == "TEST"
        assert p.rate_params.rate_fixing.identifier == "TEST_1B"

    def test_ibor_fixing_identifier(self):
        p = FloatPeriod(
            start=dt(2000, 1, 1),
            end=dt(2000, 4, 1),
            frequency=Frequency.Months(3, None),
            payment=dt(2000, 1, 4),
            fixing_method=FloatFixingMethod.IBOR,
            rate_fixings="TEST",
        )
        assert p.rate_params.fixing_identifier == "TEST"
        assert p.rate_params.rate_fixing.identifier == "TEST_3M"

    def test_ibor12M_fixing_identifier(self):
        p = FloatPeriod(
            start=dt(2000, 1, 1),
            end=dt(2001, 1, 1),
            frequency=Frequency.Months(12, None),
            payment=dt(2000, 1, 4),
            fixing_method=FloatFixingMethod.IBOR,
            rate_fixings="TEST",
        )
        assert p.rate_params.fixing_identifier == "TEST"
        assert p.rate_params.rate_fixing.identifier == "TEST_12M"

    def test_ibor_stub_fixing_identifier(self):
        with pytest.warns(UserWarning, match=err.UW_NO_TENORS[:15]):
            p = FloatPeriod(
                start=dt(2000, 1, 1),
                end=dt(2000, 3, 1),
                frequency=Frequency.Months(3, None),
                payment=dt(2000, 1, 4),
                fixing_method=FloatFixingMethod.IBOR,
                stub=True,
                rate_fixings="TEST",
            )
            assert p.rate_params.fixing_identifier == "TEST"
            assert p.rate_params.rate_fixing.identifier == "TEST"
