import context
import pytest
from datetime import datetime as dt

from rateslib.rs import FXRate, Dual, Dual2


@pytest.mark.parametrize("val", [0.99, Dual(0.99, ["x"], []), Dual2(0.99, ["x"], [], [])])
def test_fx_rate_creation(val):
    fxr = FXRate("usd", "eur", val, dt(2001, 1, 1))

    assert fxr.rate == val
    assert fxr.pair == "usdeur"
    assert fxr.settlement == dt(2001, 1, 1)



