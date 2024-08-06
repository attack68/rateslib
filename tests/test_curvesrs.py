import pytest
import context
from rateslib.curves.rs import CurveRs, LinearInterpolator
from rateslib.dual import ADOrder
from datetime import datetime as dt

@pytest.fixture()
def curve():
    return CurveRs(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.99,
        },
        interpolator=LinearInterpolator(),
        id="v",
        ad=ADOrder.One,
    )


def test_get_item(curve):
    result = curve[dt(2022, 3, 16)]
    assert abs(result - 0.995) < 1e-14


