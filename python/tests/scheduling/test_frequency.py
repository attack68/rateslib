# SPDX-License-Identifier: LicenseRef-Rateslib-Dual
#
# Copyright (c) 2026 Siffrorna Technology Limited
#
# Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
# Source-available, not open source.
#
# See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
# and/or contact info (at) rateslib (dot) com
####################################################################################################

from datetime import datetime as dt

import pytest
from rateslib.rs import Adjuster, Cal, Frequency, RollDay


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
        ("infer_ustub", (dt(2000, 1, 1), dt(2000, 1, 17), True, True), dt(2000, 1, 7)),
        ("infer_ustub", (dt(2000, 1, 1), dt(2000, 1, 27), False, True), dt(2000, 1, 17)),
        ("infer_ustub", (dt(2000, 1, 1), dt(2000, 1, 17), True, False), dt(2000, 1, 11)),
        ("infer_ustub", (dt(2000, 1, 1), dt(2000, 1, 27), False, False), dt(2000, 1, 11)),
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
        ("infer_ustub", (dt(2025, 1, 1), dt(2025, 1, 23), True, True), dt(2025, 1, 2)),
        ("infer_ustub", (dt(2025, 1, 1), dt(2025, 1, 23), False, True), dt(2025, 1, 9)),
        ("infer_ustub", (dt(2025, 1, 1), dt(2025, 1, 23), True, False), dt(2025, 1, 22)),
        ("infer_ustub", (dt(2025, 1, 1), dt(2025, 1, 23), False, False), dt(2025, 1, 15)),
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
        ("unext", (dt(2025, 1, 15),), dt(2025, 2, 15)),
        ("uprevious", (dt(2025, 2, 15),), dt(2025, 1, 15)),
        (
            "uregular",
            (dt(2025, 1, 15), dt(2025, 3, 15)),
            [dt(2025, 1, 15), dt(2025, 2, 15), dt(2025, 3, 15)],
        ),
        ("infer_ustub", (dt(2025, 1, 1), dt(2025, 4, 15), True, True), dt(2025, 1, 15)),
        ("infer_ustub", (dt(2025, 1, 1), dt(2025, 4, 15), False, True), dt(2025, 2, 15)),
        ("infer_ustub", (dt(2025, 1, 15), dt(2025, 4, 1), True, False), dt(2025, 3, 15)),
        ("infer_ustub", (dt(2025, 1, 15), dt(2025, 4, 1), False, False), dt(2025, 2, 15)),
    ],
)
def test_frequency_months(method, args, exp):
    f = Frequency.Months(1, RollDay.Day(15))
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
        ("infer_ustub", (dt(2025, 1, 1), dt(2025, 4, 15), True, True), dt(2025, 1, 15)),
        ("infer_ustub", (dt(2025, 1, 1), dt(2025, 4, 15), False, True), dt(2025, 2, 15)),
        ("infer_ustub", (dt(2025, 1, 1), dt(2025, 4, 15), True, False), dt(2025, 4, 1)),
        ("infer_ustub", (dt(2025, 1, 1), dt(2025, 4, 15), False, False), dt(2025, 3, 1)),
    ],
)
def test_frequency_months_undefined(method, args, exp):
    with pytest.raises(ValueError, match="`udate` cannot be validated since RollDay is None."):
        getattr(Frequency.Months(1, None), method)(*args)


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


@pytest.mark.parametrize("front", [True, False])
def test_frequency_zero_raise(front):
    f = Frequency.Zero()
    result = f.infer_ustub(dt(2000, 1, 1), dt(2001, 1, 1), True, front)
    assert result is None


def test_equality():
    f = Frequency.Zero()
    assert f == Frequency.Zero()

    f = Frequency.CalDays(10)
    assert isinstance(f, Frequency.CalDays)
    assert not isinstance(f, Frequency.BusDays)


def test_rollday_equality():
    assert RollDay.Day(15) == RollDay.Day(15)
    assert RollDay.Day(15) != RollDay.Day(16)
    assert RollDay.Day(15) != RollDay.IMM()
    assert RollDay.IMM() == RollDay.IMM()


def test_string():
    assert Frequency.Zero().string() == "Z"
    assert Frequency.CalDays(10).string() == "10D"
    assert Frequency.Months(3, None).string() == "Q"


def test_adjuster_reverse():
    cal = Cal([dt(2010, 1, 1)], [])
    result = Adjuster.Following().reverse(dt(2010, 1, 2), cal)
    assert result == [dt(2010, 1, 2), dt(2010, 1, 1)]
