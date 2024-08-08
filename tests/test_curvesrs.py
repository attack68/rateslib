import pytest
import context
from rateslib.curves.rs import CurveObj, LinearInterpolator, LinearZeroRateInterpolator, LogLinearInterpolator, _get_interpolator
from rateslib.dual import ADOrder
from datetime import datetime as dt

@pytest.fixture()
def curve():
    return CurveObj(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.99,
        },
        interpolator=LinearInterpolator(),
        id="v",
        ad=ADOrder.One,
    )


@pytest.mark.parametrize("name, expected", [
    ("linear", LinearInterpolator),
    ("log_linear", LogLinearInterpolator),
    ("linear_zero_rate", LinearZeroRateInterpolator),
])
def test_get_interpolator(name, expected):
    result = _get_interpolator(name)
    assert type(result) is expected


def test_get_interpolator_raises():
    with pytest.raises(ValueError, match="Interpolator `name` is invalid"):
        _get_interpolator("bad")


def test_get_item(curve):
    result = curve[dt(2022, 3, 16)]
    assert abs(result - 0.995) < 1e-14


