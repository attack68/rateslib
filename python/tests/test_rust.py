from datetime import datetime as dt

import pytest
from rateslib.rs import Adjuster, Cal, Frequency, Modifier, RollDay, Schedule, StubInference


def test_stub_inference_equality():
    assert StubInference.LongFront != StubInference.ShortFront
    assert StubInference.ShortFront == StubInference.ShortFront


@pytest.mark.parametrize(
    ("method", "args", "exp"),
    [
        ("next", (dt(2000, 1, 1),), dt(2000, 1, 11)),
        ("previous", (dt(2000, 1, 11),), dt(2000, 1, 1)),
        (
            "uregular",
            (dt(2000, 1, 1), dt(2000, 1, 21)),
            [dt(2000, 1, 1), dt(2000, 1, 11), dt(2000, 1, 21)],
        ),
        ("infer_front_stub", (dt(2000, 1, 1), dt(2000, 1, 17), True), dt(2000, 1, 7)),
        ("infer_front_stub", (dt(2000, 1, 1), dt(2000, 1, 27), False), dt(2000, 1, 17)),
        ("infer_back_stub", (dt(2000, 1, 1), dt(2000, 1, 17), True), dt(2000, 1, 11)),
        ("infer_back_stub", (dt(2000, 1, 1), dt(2000, 1, 27), False), dt(2000, 1, 11)),
    ],
)
def test_frequency(method, args, exp):
    f = Frequency.CalDays(10)
    result = getattr(f, method)(*args)
    assert result == exp


def test_frequency_busday():
    cal = Cal([], [5, 6])
    f = Frequency.BusDays(5, cal)
    result = f.next(dt(2023, 1, 2))
    assert result == dt(2023, 1, 9)


@pytest.mark.parametrize(
    ("ueff", "uterm", "si", "exp"),
    [
        (dt(2000, 1, 1), dt(2000, 7, 1), None, [dt(2000, 1, 1), dt(2000, 4, 1), dt(2000, 7, 1)]),
        (
            dt(2000, 1, 1),
            dt(2000, 8, 1),
            StubInference.ShortFront,
            [dt(2000, 1, 1), dt(2000, 2, 1), dt(2000, 5, 1), dt(2000, 8, 1)],
        ),
        (
            dt(2000, 1, 1),
            dt(2000, 8, 1),
            StubInference.LongFront,
            [dt(2000, 1, 1), dt(2000, 5, 1), dt(2000, 8, 1)],
        ),
        (
            dt(2000, 1, 1),
            dt(2000, 8, 1),
            StubInference.ShortBack,
            [dt(2000, 1, 1), dt(2000, 4, 1), dt(2000, 7, 1), dt(2000, 8, 1)],
        ),
        (
            dt(2000, 1, 1),
            dt(2000, 8, 1),
            StubInference.LongBack,
            [dt(2000, 1, 1), dt(2000, 4, 1), dt(2000, 8, 1)],
        ),
    ],
)
def test_uschedule(ueff, uterm, si, exp):
    s = Schedule(
        ueffective=ueff,
        utermination=uterm,
        frequency=Frequency.Months(3, RollDay.SoM()),
        calendar=Cal([], [5, 6]),
        accrual_adjuster=Adjuster.ModifiedFollowing(),
        payment_adjuster=Adjuster.BusDaysLagSettle(2),
        ufront_stub=None,
        uback_stub=None,
        stub_inference=si,
    )
    assert s.uschedule == exp
