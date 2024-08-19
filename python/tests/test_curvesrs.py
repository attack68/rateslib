from datetime import datetime as dt

import pytest
from rateslib.calendars import _get_modifier
from rateslib.curves.rs import (
    CurveObj,
    CurveRs,
    FlatBackwardInterpolator,
    FlatForwardInterpolator,
    LinearInterpolator,
    LinearZeroRateInterpolator,
    LogLinearInterpolator,
    _get_convention,
    _get_convention_str,
    _get_interpolator,
)
from rateslib.dual import ADOrder
from rateslib.json import from_json


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
        convention=_get_convention("Act360"),
        modifier=_get_modifier("MF", True),
    )


@pytest.fixture()
def curvers():
    return CurveRs(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.99,
        },
        interpolation="log_linear",
        id="v",
        ad=1,
    )


@pytest.fixture()
def indexcurvers():
    return CurveObj(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.99,
        },
        interpolator=LinearInterpolator(),
        id="v",
        ad=ADOrder.One,
        convention=_get_convention("Act360"),
        modifier=_get_modifier("MF", True),
        index_base=100.0,
    )


@pytest.mark.parametrize(
    "name, expected",
    [
        ("linear", LinearInterpolator),
        ("log_linear", LogLinearInterpolator),
        ("linear_zero_rate", LinearZeroRateInterpolator),
        ("flat_forward", FlatForwardInterpolator),
        ("flat_backward", FlatBackwardInterpolator),
    ],
)
def test_get_interpolator(name, expected):
    result = _get_interpolator(name)
    assert type(result) is expected


def test_get_interpolation(curve):
    result = curve.interpolation
    assert result == "linear"


def test_get_modifier(curvers):
    result = curvers.modifier
    assert result == "MF"


def test_get_convention(curvers):
    result = curvers.convention
    assert result == "Act360"


def test_get_ad(curvers):
    result = curvers.ad
    assert result == 1


def test_get_interpolator_raises():
    with pytest.raises(ValueError, match="Interpolator `name` is invalid"):
        _get_interpolator("bad")


def test_get_item(curve):
    result = curve[dt(2022, 3, 16)]
    assert abs(result - 0.995) < 1e-14


def test_json_round_trip(curvers):
    json = curvers.to_json()
    curve2 = from_json(json)
    assert curvers == curve2


@pytest.mark.parametrize(
    "kind",
    [
        "linear",
        "log_linear",
        "linear_zero_rate",
        "flat_forward",
        "flat_backward",
    ],
)
def test_interp_constructs(kind):
    result = CurveRs(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2022, 3, 31): 0.99,
        },
        interpolation=kind,
        id="v",
        ad=1,
    )
    assert isinstance(result, CurveRs)


def test_index_value(indexcurvers):
    result = indexcurvers.index_value(dt(2022, 3, 31))
    assert abs(result - 100.0 / 0.99) < 1e-12
