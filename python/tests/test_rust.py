from datetime import datetime as dt

import pytest
from rateslib.rs import Adjuster, Cal, Frequency, RollDay, Schedule, StubInference


def test_stub_inference_equality():
    assert StubInference.LongFront != StubInference.ShortFront
    assert StubInference.ShortFront == StubInference.ShortFront


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
        frequency=Frequency.Months(3, RollDay.Day(1)),
        calendar=Cal([], [5, 6]),
        accrual_adjuster=Adjuster.ModifiedFollowing(),
        payment_adjuster=Adjuster.BusDaysLagSettle(2),
        ufront_stub=None,
        uback_stub=None,
        stub_inference=si,
    )
    assert s.uschedule == exp
