from datetime import datetime as dt

import pytest
from rateslib.fx import FXRates
from rateslib.json import from_json
from rateslib.rs import Ccy, Dual, Dual2, FXRate


def test_ccy_creation():
    c1 = Ccy("usd")
    c2 = Ccy("USD")
    assert c1 == c2


@pytest.mark.parametrize("val", [0.99, Dual(0.99, ["x"], []), Dual2(0.99, ["x"], [], [])])
def test_fx_rate_creation(val):
    fxr = FXRate("usd", "eur", val, dt(2001, 1, 1))
    assert fxr.rate == val
    assert fxr.pair == "usdeur"
    assert fxr.settlement == dt(2001, 1, 1)


def test_json_round_trip():
    fxr = FXRates({"eurusd": 1.08, "usdjpy": 110.0}, dt(2004, 1, 1))
    json = fxr.to_json()
    fxr2 = from_json(json)
    assert fxr == fxr2


def test_equality():
    fxr = FXRates({"eurusd": 1.08, "usdjpy": 110.0}, dt(2004, 1, 1))
    fxr2 = FXRates({"eurusd": 1.08, "usdjpy": 110.0}, dt(2004, 1, 1))
    assert fxr == fxr2
