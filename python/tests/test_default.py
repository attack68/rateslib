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

import pytest
from rateslib import NamedCal, __version__, default_context, defaults, dt, fixings


def test_version() -> None:
    assert __version__ == "2.5.1"


def test_context_raises() -> None:
    with pytest.raises(ValueError, match="Need to invoke as "):
        default_context("only 1 arg")


def test_reset_defaults() -> None:
    defaults.modifier = "MP"
    defaults.base_currency = "gbp"
    assert defaults.modifier == "MP"
    assert defaults.base_currency == "gbp"
    defaults.calendars["TEST"] = 10.0
    assert defaults.calendars["TEST"] == 10.0

    defaults.reset_defaults()
    assert defaults.modifier == "MF"
    assert defaults.base_currency == "usd"
    assert "TEST" not in defaults.calendars


def test_defaults_singleton() -> None:
    from rateslib.default import Defaults

    other = Defaults()
    assert id(other) == id(defaults)


def test_fixings_singleton() -> None:
    from rateslib.data.loader import Fixings

    other = Fixings()
    assert id(other) == id(fixings)


def test_fx_index_change() -> None:
    # test that default fx indexes can be overwritten and are loaded by constructed objects
    from rateslib.data.fixings import FXFixing, FXIndex
    from rateslib.scheduling import Adjuster

    eurusd = FXFixing("eurusd", dt(2000, 1, 1))
    assert eurusd.fx_index.calendar == NamedCal("tgt|fed")
    assert eurusd.fx_index.settle == Adjuster.BusDaysLagSettle(2)
    defaults.fx_index["eurusd"] = {"pair": "eurusd", "calendar": "stk", "settle": 3}
    eurusd = FXFixing("eurusd", dt(2000, 1, 1))
    assert eurusd.fx_index.calendar == NamedCal("stk")
    assert eurusd.fx_index.settle == Adjuster.BusDaysLagSettle(3)

    defaults.reset_defaults()
    assert defaults.fx_index["eurusd"]["calendar"] == NamedCal("tgt|fed")


def test_float_series_change():
    from rateslib import IRS

    with pytest.raises(ValueError, match="The FloatRateSeries: 'monkey' was not found "):
        IRS(dt(2000, 1, 1), "1y", "A", leg2_fixing_series="monkey")

    defaults.float_series["monkey"] = dict(
        lag=0, calendar="nyc", modifier="f", eom=False, convention="act360"
    )
    IRS(dt(2000, 1, 1), "1y", "A", leg2_fixing_series="monkey")

    defaults.reset_defaults()
    assert "monkey" not in defaults.float_series
