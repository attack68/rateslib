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
from rateslib.enums.generics import NoInput
from rateslib.rs import Adjuster, Cal, Frequency, RollDay, Schedule, StubInference


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
def test_schedule(ueff, uterm, si, exp):
    s = Schedule(
        effective=ueff,
        termination=uterm,
        frequency=Frequency.Months(3, RollDay.Day(1)),
        calendar=Cal([], [5, 6]),
        accrual_adjuster=Adjuster.ModifiedFollowing(),
        payment_adjuster=Adjuster.BusDaysLagSettle(2),
        payment_adjuster2=Adjuster.Actual(),
        eom=True,
        front_stub=None,
        back_stub=None,
        stub_inference=si,
    )
    assert s.uschedule == exp


def test_imm_schedule():
    # test that IMM rolls are automatically determined.
    s = Schedule(
        effective=dt(2025, 3, 19),
        termination=dt(2025, 9, 17),
        frequency=Frequency.Months(3, None),
        calendar=Cal([], [5, 6]),
        accrual_adjuster=Adjuster.ModifiedFollowing(),
        payment_adjuster=Adjuster.BusDaysLagSettle(2),
        payment_adjuster2=Adjuster.Actual(),
        eom=True,
        front_stub=None,
        back_stub=None,
        stub_inference=None,
    )
    assert s.frequency == Frequency.Months(3, RollDay.IMM())


def test_single_period_schedule():
    s = Schedule(
        effective=dt(2025, 3, 19),
        termination=dt(2025, 9, 19),
        frequency=Frequency.Months(12, RollDay.Day(19)),
        calendar=Cal([], [5, 6]),
        accrual_adjuster=Adjuster.ModifiedFollowing(),
        payment_adjuster=Adjuster.BusDaysLagSettle(2),
        payment_adjuster2=Adjuster.Actual(),
        eom=True,
        front_stub=None,
        back_stub=None,
        stub_inference=None,
    )
    assert s.uschedule == [dt(2025, 3, 19), dt(2025, 9, 19)]


def test_single_period_schedule2():
    from rateslib import IRS

    IRS(dt(2022, 7, 1), "3M", "A", curves="eureur", notional=1e6)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (Adjuster.ModifiedFollowing(), Adjuster.ModifiedFollowing(), True),
        (Adjuster.Following(), Adjuster.ModifiedFollowing(), False),
        (Adjuster.BusDaysLagSettleInAdvance(3), Adjuster.BusDaysLagSettleInAdvance(3), True),
        (Adjuster.BusDaysLagSettleInAdvance(3), Adjuster.Following(), False),
        (Adjuster.BusDaysLagSettle(2), Adjuster.BusDaysLagSettle(1), False),
    ],
)
def test_adjuster_equality(a, b, expected):
    result = a == b
    assert result is expected
