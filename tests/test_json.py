import pytest
from datetime import datetime as dt
from pandas import DataFrame, Series, date_range, Index
from pandas.testing import assert_frame_equal, assert_series_equal
import numpy as np

import context
from rateslib import default_context
from rateslib.legs import (
    FixedLeg,
    FloatLeg,
    FloatPeriod,
    ZeroFloatLeg,
    ZeroFixedLeg,
    ZeroIndexLeg,
    FixedPeriod,
    CustomLeg,
    IndexFixedLeg,
    FloatLegMtm,
    FixedLegMtm,
    Cashflow,
)
from rateslib.fx import FXRates, FXForwards
from rateslib.default import Defaults, NoInput
from rateslib.curves import Curve, IndexCurve
from rateslib.json import Serialise


def test_fxrates_to_json():
    fxr = FXRates({"usdnok": 8.0, "eurusd": 1.05})
    result = Serialise(fxr).to_json()
    expected = '{"fx_rates": {"usdnok": 8.0, "eurusd": 1.05}, "settlement": null, "base": "usd"}'
    assert result == expected

    fxr = FXRates({"usdnok": 8.0, "eurusd": 1.05}, dt(2022, 1, 3))
    result = Serialise(fxr).to_json()
    expected = (
        '{"fx_rates": {"usdnok": 8.0, "eurusd": 1.05}, "settlement": "2022-01-03", "base": "usd"}'
    )
    assert result == expected


def test_fxrates_from_json_and_equality():
    fxr1 = FXRates({"usdnok": 8.0, "eurusd": 1.05})
    fxr2 = FXRates({"usdnok": 12.0, "eurusd": 1.10})
    assert fxr1 != fxr2

    fxr2 = Serialise.from_json(FXRates,
        '{"fx_rates": {"usdnok": 8.0, "eurusd": 1.05}, "settlement": null, "base": "usd"}'
    )
    assert fxr2 == fxr1

    fxr3 = FXRates({"usdnok": 8.0, "eurusd": 1.05}, base="NOK")
    assert fxr1 != fxr3  # base is different