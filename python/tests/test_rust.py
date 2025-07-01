from datetime import datetime as dt

import pytest
from rateslib.rs import Cal, Frequency, Modifier, RollDay


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
