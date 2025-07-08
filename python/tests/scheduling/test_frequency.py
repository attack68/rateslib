from datetime import datetime as dt

import pytest
from rateslib.rs import Cal, Frequency, RollDay


@pytest.mark.parametrize(
    ("method", "args", "exp"),
    [
        ("unext", (dt(2000, 1, 1),), dt(2000, 1, 11)),
        ("uprevious", (dt(2000, 1, 11),), dt(2000, 1, 1)),
        (
            "uregular",
            (dt(2000, 1, 1), dt(2000, 1, 21)),
            [dt(2000, 1, 1), dt(2000, 1, 11), dt(2000, 1, 21)],
        ),
        ("infer_ufront_stub", (dt(2000, 1, 1), dt(2000, 1, 17), True), dt(2000, 1, 7)),
        ("infer_ufront_stub", (dt(2000, 1, 1), dt(2000, 1, 27), False), dt(2000, 1, 17)),
        ("infer_uback_stub", (dt(2000, 1, 1), dt(2000, 1, 17), True), dt(2000, 1, 11)),
        ("infer_uback_stub", (dt(2000, 1, 1), dt(2000, 1, 27), False), dt(2000, 1, 11)),
    ],
)
def test_frequency_cal_days(method, args, exp):
    f = Frequency.CalDays(10)
    result = getattr(f, method)(*args)
    assert result == exp


@pytest.mark.parametrize(
    ("method", "args", "exp"),
    [
        ("unext", (dt(2025, 1, 1),), dt(2025, 1, 8)),
        ("uprevious", (dt(2025, 1, 8),), dt(2025, 1, 1)),
        (
            "uregular",
            (dt(2025, 1, 1), dt(2025, 1, 15)),
            [dt(2025, 1, 1), dt(2025, 1, 8), dt(2025, 1, 15)],
        ),
        ("infer_ufront_stub", (dt(2025, 1, 1), dt(2025, 1, 23), True), dt(2025, 1, 2)),
        ("infer_ufront_stub", (dt(2025, 1, 1), dt(2025, 1, 23), False), dt(2025, 1, 9)),
        ("infer_uback_stub", (dt(2025, 1, 1), dt(2025, 1, 23), True), dt(2025, 1, 22)),
        ("infer_uback_stub", (dt(2025, 1, 1), dt(2025, 1, 23), False), dt(2025, 1, 15)),
    ],
)
def test_frequency_bus_days(method, args, exp):
    cal = Cal([], [5, 6])
    f = Frequency.BusDays(5, cal)
    result = getattr(f, method)(*args)
    assert result == exp


@pytest.mark.parametrize(
    ("method", "args", "exp"),
    [
        ("unext", (dt(2025, 1, 1),), dt(2025, 1, 8)),
        ("uprevious", (dt(2025, 1, 8),), dt(2025, 1, 1)),
        (
            "uregular",
            (dt(2025, 1, 1), dt(2025, 1, 15)),
            [dt(2025, 1, 1), dt(2025, 1, 8), dt(2025, 1, 15)],
        ),
        ("infer_ufront_stub", (dt(2025, 1, 1), dt(2025, 1, 23), True), dt(2025, 1, 2)),
        ("infer_ufront_stub", (dt(2025, 1, 1), dt(2025, 1, 23), False), dt(2025, 1, 9)),
        ("infer_uback_stub", (dt(2025, 1, 1), dt(2025, 1, 23), True), dt(2025, 1, 22)),
        ("infer_uback_stub", (dt(2025, 1, 1), dt(2025, 1, 23), False), dt(2025, 1, 15)),
    ],
)
def test_frequency_weeks(method, args, exp):
    f = Frequency.Weeks(1)
    result = getattr(f, method)(*args)
    assert result == exp


@pytest.mark.parametrize(
    ("method", "args", "exp"),
    [
        ("unext", (dt(2025, 1, 1),), dt(2025, 2, 1)),
        ("uprevious", (dt(2025, 2, 1),), dt(2025, 1, 1)),
        (
            "uregular",
            (dt(2025, 1, 1), dt(2025, 3, 1)),
            [dt(2025, 1, 1), dt(2025, 2, 1), dt(2025, 3, 1)],
        ),
        ("infer_ufront_stub", (dt(2025, 1, 1), dt(2025, 4, 15), True), dt(2025, 1, 15)),
        ("infer_ufront_stub", (dt(2025, 1, 1), dt(2025, 4, 15), False), dt(2025, 2, 15)),
        ("infer_uback_stub", (dt(2025, 1, 1), dt(2025, 4, 15), True), dt(2025, 4, 1)),
        ("infer_uback_stub", (dt(2025, 1, 1), dt(2025, 4, 15), False), dt(2025, 3, 1)),
    ],
)
def test_frequency_months(method, args, exp):
    f = Frequency.Months(1, None)
    result = getattr(f, method)(*args)
    assert result == exp


@pytest.mark.parametrize(
    ("method", "args", "exp"),
    [
        ("unext", (dt(2025, 1, 1),), dt(9999, 1, 1)),
        ("uprevious", (dt(2025, 1, 8),), dt(1500, 1, 1)),
        ("uregular", (dt(2025, 1, 1), dt(2025, 1, 15)), [dt(2025, 1, 1), dt(2025, 1, 15)]),
    ],
)
def test_frequency_zero(method, args, exp):
    f = Frequency.Zero()
    result = getattr(f, method)(*args)
    assert result == exp


@pytest.mark.parametrize("method", ["infer_ufront_stub", "infer_uback_stub"])
def test_frequency_zero_raise(method):
    f = Frequency.Zero()
    with pytest.raises(ValueError, match="Dates are too close together to infer the desired stub"):
        getattr(f, method)(dt(2000, 1, 1), dt(2001, 1, 1), True)
