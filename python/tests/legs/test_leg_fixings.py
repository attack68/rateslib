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

from datetime import datetime as dt

import pytest
from pandas import Series
from rateslib import fixings
from rateslib.curves import Curve
from rateslib.enums.generics import NoInput
from rateslib.legs import FixedLeg, FloatLeg
from rateslib.scheduling import Schedule


class TestFixedLeg:
    def test_populated_resets(self):
        fixings.add(
            name="index",
            series=Series(
                index=[dt(2000, 1, 1), dt(2000, 7, 1), dt(2001, 1, 1)], data=[1.0, 1.1, 1.2]
            ),
            state=100,
        )
        fixings.add(
            name="fx_eurusd", series=Series(index=[dt(1999, 12, 30)], data=[2.0]), state=100
        )

        fl = FixedLeg(
            schedule=Schedule(dt(2000, 1, 1), "1y", "S"),
            index_fixings="index",
            index_lag=0,
            index_method="monthly",
            pair="eurusd",
            fx_fixings="fx",
        )
        assert fl.periods[0].non_deliverable_params.fx_fixing.value == 2.0
        assert fl.periods[1].non_deliverable_params.fx_fixing.value == 2.0
        assert fl.periods[0].index_params.index_fixing.value == 1.1
        assert fl.periods[0].index_params.index_base.value == 1.0
        assert fl.periods[1].index_params.index_fixing.value == 1.2
        assert fl.periods[1].index_params.index_base.value == 1.0

        fixings.pop("index")
        fixings.pop("fx_eurusd")
        fl.reset_fixings(100)
        assert fl.periods[0].non_deliverable_params.fx_fixing._value == NoInput(0)
        assert fl.periods[1].non_deliverable_params.fx_fixing._value == NoInput(0)
        assert fl.periods[0].index_params.index_fixing._value == NoInput(0)
        assert fl.periods[0].index_params.index_base._value == NoInput(0)
        assert fl.periods[1].index_params.index_fixing._value == NoInput(0)
        assert fl.periods[1].index_params.index_base._value == NoInput(0)

    def test_populated_at_init_no_reset(self):
        fixings.add(
            name="index",
            series=Series(
                index=[dt(2000, 1, 1), dt(2000, 7, 1), dt(2001, 1, 1)], data=[1.0, 1.1, 1.2]
            ),
            state=100,
        )
        fixings.add(
            name="fx_eurusd", series=Series(index=[dt(1999, 12, 30)], data=[2.0]), state=100
        )

        fl = FixedLeg(
            schedule=Schedule(dt(2000, 1, 1), "1y", "S"),
            index_fixings="index",
            index_lag=0,
            index_method="monthly",
            pair="eurusd",
            fx_fixings="fx",
        )
        assert fl.periods[0].non_deliverable_params.fx_fixing.value == 2.0
        assert fl.periods[1].non_deliverable_params.fx_fixing.value == 2.0
        assert fl.periods[0].index_params.index_fixing.value == 1.1
        assert fl.periods[0].index_params.index_base.value == 1.0
        assert fl.periods[1].index_params.index_fixing.value == 1.2
        assert fl.periods[1].index_params.index_base.value == 1.0

        fixings.pop("index")
        fixings.pop("fx_eurusd")
        fl.reset_fixings(666)
        assert fl.periods[0].non_deliverable_params.fx_fixing.value == 2.0
        assert fl.periods[1].non_deliverable_params.fx_fixing.value == 2.0
        assert fl.periods[0].index_params.index_fixing.value == 1.1
        assert fl.periods[0].index_params.index_base.value == 1.0
        assert fl.periods[1].index_params.index_fixing.value == 1.2
        assert fl.periods[1].index_params.index_base.value == 1.0

    def test_populated_resets_notional_exchanges(self):
        fixings.add(
            name="index",
            series=Series(
                index=[dt(2000, 1, 1), dt(2000, 7, 1), dt(2001, 1, 1)], data=[1.0, 1.1, 1.2]
            ),
            state=100,
        )
        fixings.add(
            name="fx_eurusd", series=Series(index=[dt(1999, 12, 30)], data=[2.0]), state=100
        )

        fl = FixedLeg(
            schedule=Schedule(dt(2000, 1, 1), "1y", "S"),
            index_fixings="index",
            index_lag=0,
            index_method="monthly",
            pair="eurusd",
            fx_fixings="fx",
            initial_exchange=True,
        )
        assert fl.periods[0].non_deliverable_params.fx_fixing.value == 2.0
        assert fl.periods[-1].non_deliverable_params.fx_fixing.value == 2.0
        assert fl.periods[0].index_params.index_fixing.value == 1.0
        assert fl.periods[0].index_params.index_base.value == 1.0
        assert fl.periods[-1].index_params.index_fixing.value == 1.2
        assert fl.periods[-1].index_params.index_base.value == 1.0

        fixings.pop("index")
        fixings.pop("fx_eurusd")
        fl.reset_fixings(100)
        assert fl.periods[0].non_deliverable_params.fx_fixing._value == NoInput(0)
        assert fl.periods[-1].non_deliverable_params.fx_fixing._value == NoInput(0)
        assert fl.periods[0].index_params.index_fixing._value == NoInput(0)
        assert fl.periods[0].index_params.index_base._value == NoInput(0)
        assert fl.periods[-1].index_params.index_fixing._value == NoInput(0)
        assert fl.periods[-1].index_params.index_base._value == NoInput(0)


class TestFloatLeg:
    def test_populated_resets_ibor(self):
        fixings.add(
            name="index",
            series=Series(
                index=[dt(2000, 1, 1), dt(2000, 3, 1), dt(2000, 6, 1)], data=[1.0, 1.1, 1.2]
            ),
            state=100,
        )
        fixings.add(
            name="fx_eurusd", series=Series(index=[dt(1999, 12, 30)], data=[2.0]), state=100
        )
        fixings.add(
            name="ibor_1M",
            series=Series(index=[dt(2000, 1, 1), dt(2000, 3, 1)], data=[1.0, 2.0]),
            state=100,
        )
        fixings.add(
            name="ibor_3M",
            series=Series(index=[dt(2000, 1, 1), dt(2000, 3, 1)], data=[1.1, 2.1]),
            state=100,
        )

        fl = FloatLeg(
            schedule=Schedule(dt(2000, 1, 1), "5m", "Q"),
            index_fixings="index",
            index_lag=0,
            index_method="monthly",
            pair="eurusd",
            fx_fixings="fx",
            fixing_method="ibor",
            method_param=0,
            rate_fixings="ibor",
        )
        assert fl.periods[0].rate_params.rate_fixing.value == 1.0483333333333333
        assert fl.periods[1].rate_params.rate_fixing.value == 2.1
        assert fl.periods[0].non_deliverable_params.fx_fixing.value == 2.0
        assert fl.periods[1].non_deliverable_params.fx_fixing.value == 2.0
        assert fl.periods[0].index_params.index_fixing.value == 1.1
        assert fl.periods[0].index_params.index_base.value == 1.0
        assert fl.periods[1].index_params.index_fixing.value == 1.2
        assert fl.periods[1].index_params.index_base.value == 1.0
        assert fl.periods[1].index_params.index_fixing.value == 1.2
        assert fl.periods[1].index_params.index_base.value == 1.0

        fixings.pop("index")
        fixings.pop("fx_eurusd")
        fixings.pop("ibor_1M")
        fixings.pop("ibor_3M")
        fixings.add(name="ibor_1M", series=Series(index=[dt(1999, 1, 1)], data=[99.0]), state=100)
        fixings.add(
            name="ibor_3M",
            series=Series(
                index=[
                    dt(1999, 1, 1),
                ],
                data=[99.0],
            ),
            state=100,
        )

        fl.reset_fixings(100)
        assert fl.periods[0].rate_params.rate_fixing.value == NoInput(0)
        assert fl.periods[1].rate_params.rate_fixing.value == NoInput(0)
        assert fl.periods[0].non_deliverable_params.fx_fixing._value == NoInput(0)
        assert fl.periods[1].non_deliverable_params.fx_fixing._value == NoInput(0)
        assert fl.periods[0].index_params.index_fixing._value == NoInput(0)
        assert fl.periods[0].index_params.index_base._value == NoInput(0)
        assert fl.periods[1].index_params.index_fixing._value == NoInput(0)
        assert fl.periods[1].index_params.index_base._value == NoInput(0)

    def test_populated_at_init_no_reset(self):
        fixings.add(
            name="index",
            series=Series(
                index=[dt(2000, 1, 1), dt(2000, 3, 1), dt(2000, 6, 1)], data=[1.0, 1.1, 1.2]
            ),
            state=100,
        )
        fixings.add(
            name="fx_eurusd", series=Series(index=[dt(1999, 12, 30)], data=[2.0]), state=100
        )
        fixings.add(
            name="ibor_1M",
            series=Series(index=[dt(2000, 1, 1), dt(2000, 3, 1)], data=[1.0, 2.0]),
            state=100,
        )
        fixings.add(
            name="ibor_3M",
            series=Series(index=[dt(2000, 1, 1), dt(2000, 3, 1)], data=[1.1, 2.1]),
            state=100,
        )

        fl = FloatLeg(
            schedule=Schedule(dt(2000, 1, 1), "5m", "Q"),
            index_fixings="index",
            index_lag=0,
            index_method="monthly",
            pair="eurusd",
            fx_fixings="fx",
            fixing_method="ibor",
            method_param=0,
            rate_fixings="ibor",
        )
        assert fl.periods[0].rate_params.rate_fixing.value == 1.0483333333333333
        assert fl.periods[1].rate_params.rate_fixing.value == 2.1
        assert fl.periods[0].non_deliverable_params.fx_fixing.value == 2.0
        assert fl.periods[1].non_deliverable_params.fx_fixing.value == 2.0
        assert fl.periods[0].index_params.index_fixing.value == 1.1
        assert fl.periods[0].index_params.index_base.value == 1.0
        assert fl.periods[1].index_params.index_fixing.value == 1.2
        assert fl.periods[1].index_params.index_base.value == 1.0

        fixings.pop("index")
        fixings.pop("fx_eurusd")
        fixings.pop("ibor_1M")
        fixings.pop("ibor_3M")
        fixings.add(name="ibor_1M", series=Series(index=[dt(1999, 1, 1)], data=[99.0]), state=100)
        fixings.add(
            name="ibor_3M",
            series=Series(
                index=[
                    dt(1999, 1, 1),
                ],
                data=[99.0],
            ),
            state=100,
        )
        fl.reset_fixings(666)
        assert fl.periods[0].rate_params.rate_fixing.value == 1.0483333333333333
        assert fl.periods[1].rate_params.rate_fixing.value == 2.1
        assert fl.periods[0].non_deliverable_params.fx_fixing.value == 2.0
        assert fl.periods[1].non_deliverable_params.fx_fixing.value == 2.0
        assert fl.periods[0].index_params.index_fixing.value == 1.1
        assert fl.periods[0].index_params.index_base.value == 1.0
        assert fl.periods[1].index_params.index_fixing.value == 1.2
        assert fl.periods[1].index_params.index_base.value == 1.0
