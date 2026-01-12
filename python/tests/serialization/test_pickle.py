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

import pickle

import pytest
from rateslib import (
    ADOrder,
    Dual,
    Dual2,
    FXForwards,
    FXRates,
    Imm,
    NamedCal,
    Variable,
    dt,
)
from rateslib.curves import (
    CompositeCurve,
    CreditImpliedCurve,
    Curve,
    LineCurve,
    MultiCsaCurve,
    ProxyCurve,
)
from rateslib.rs import Schedule as ScheduleRs
from rateslib.scheduling import (
    Adjuster,
    Cal,
    Convention,
    Frequency,
    RollDay,
    Schedule,
    StubInference,
    UnionCal,
)
from rateslib.splines import PPSplineDual, PPSplineDual2, PPSplineF64


@pytest.mark.parametrize(
    "obj",
    [
        # core
        dt(2000, 1, 1),
        # ad
        Dual(1.2, ["x"], [2.3]),
        Dual2(1.3, ["y"], [1.0], [2.0]),
        Variable(2.0, ["r"]),
        # calendars
        Cal.from_name("bus"),
        UnionCal([Cal.from_name("bus")], []),
        NamedCal("bus"),
        # scheduling
        # fx
        FXRates({"eurusd": 1.0}, dt(2000, 1, 1)),
        # curves
        Curve(
            {dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 0.98, dt(2002, 1, 1): 0.96},
            interpolation="spline",
        ),
        LineCurve({dt(2000, 1, 1): 2.0, dt(2000, 1, 2): 3.0}),
        CompositeCurve(
            [
                Curve({dt(2000, 1, 1): 1.0, dt(2000, 1, 2): 0.98}),
                Curve({dt(2000, 1, 1): 1.0, dt(2000, 1, 2): 0.98}),
            ]
        ),
        MultiCsaCurve(
            [
                Curve({dt(2000, 1, 1): 1.0, dt(2000, 1, 2): 0.98}),
                Curve({dt(2000, 1, 1): 1.0, dt(2000, 1, 2): 0.98}),
            ],
        ),
        CreditImpliedCurve(
            Curve({dt(2000, 1, 1): 1.0, dt(2000, 1, 2): 0.98}),
            Curve({dt(2000, 1, 1): 1.0, dt(2000, 1, 2): 0.98}),
        ),
        ProxyCurve(
            "usd",
            "eur",
            FXForwards(
                fx_rates=FXRates({"eurusd": 1.0}, dt(2000, 1, 1)),
                fx_curves={
                    "eureur": Curve({dt(2000, 1, 1): 1.0, dt(2000, 1, 2): 0.98}),
                    "eurusd": Curve({dt(2000, 1, 1): 1.0, dt(2000, 1, 2): 0.98}),
                    "usdusd": Curve({dt(2000, 1, 1): 1.0, dt(2000, 1, 2): 0.98}),
                },
            ),
        ),
        Curve({dt(2000, 1, 1): 1.0, dt(2000, 7, 1): 0.98}).shift(10),
        Curve({dt(2000, 1, 1): 1.0, dt(2000, 7, 1): 0.98}).roll("1m"),
        Curve({dt(2000, 1, 1): 1.0, dt(2000, 7, 1): 0.98}).translate(dt(2000, 1, 15)),
        ScheduleRs(
            effective=dt(2000, 1, 1),
            termination=dt(2001, 1, 10),
            frequency=Frequency.Months(6, RollDay.Day(1)),
            calendar=NamedCal("tgt"),
            accrual_adjuster=Adjuster.ModifiedFollowing(),
            payment_adjuster=Adjuster.BusDaysLagSettle(2),
            payment_adjuster2=Adjuster.Actual(),
            front_stub=None,
            back_stub=None,
            eom=False,
            stub_inference=StubInference.ShortBack,
        ),
        Schedule(
            effective=dt(2000, 1, 1),
            termination=dt(2001, 1, 10),
            frequency=Frequency.Months(6, RollDay.Day(1)),
            calendar=NamedCal("tgt"),
            modifier=Adjuster.ModifiedFollowing(),
            payment_lag=Adjuster.BusDaysLagSettle(2),
            stub=StubInference.ShortBack,
        ),
        PPSplineF64(3, [0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.3]),
        PPSplineDual(
            3, [0, 0, 0, 1, 1, 1], [Dual(0.1, [], []), Dual(0.2, [], []), Dual(0.3, [], [])]
        ),
        PPSplineDual2(
            3,
            [0, 0, 0, 1, 1, 1],
            [Dual2(0.1, [], [], []), Dual2(0.2, [], [], []), Dual2(0.3, [], [], [])],
        ),
    ],
)
def test_pickle_round_trip_obj_via_equality(obj):
    pickled = pickle.dumps(obj)
    loaded = pickle.loads(pickled)
    assert obj == loaded


@pytest.mark.parametrize(
    ("a1", "a2", "b1"),
    [
        (Imm.Eom, Imm.Eom, Imm.Leap),
        (StubInference.LongBack, StubInference.LongBack, StubInference.ShortFront),
        (ADOrder.Zero, ADOrder.Zero, ADOrder.One),
        (RollDay.Day(21), RollDay.Day(21), RollDay.Day(16)),
        (RollDay.Day(21), RollDay.Day(21), RollDay.IMM),
        (Adjuster.Actual(), Adjuster.Actual(), Adjuster.BusDaysLagSettle(5)),
        (
            Frequency.Months(4, RollDay.Day(2)),
            Frequency.Months(4, RollDay.Day(2)),
            Frequency.CalDays(3),
        ),
        (
            Frequency.Months(4, RollDay.Day(2)),
            Frequency.Months(4, RollDay.Day(2)),
            Frequency.Months(4, None),
        ),
        (Convention.ActActICMA, Convention.ActActICMA, Convention.ActActISDA),
    ],
)
def test_enum_equality(a1, a2, b1):
    assert a1 == a2
    assert a2 != b1


@pytest.mark.parametrize(
    ("enum", "method_filter"),
    [
        (Imm, ["next", "get", "validate", "to_json"]),
        (StubInference, ["to_json"]),
        (ADOrder, []),
        (Convention, ["dcf", "to_json"]),
    ],
)
def test_simple_enum_pickle(enum, method_filter):
    variants = [v for v in enum.__dict__ if "__" not in v and v not in method_filter]
    for v in variants:
        obj = enum.__dict__[v]
        pickled = pickle.dumps(obj)
        unpickled = pickle.loads(pickled)
        assert unpickled == enum.__dict__[v]


@pytest.mark.parametrize(
    ("enum"),
    [
        RollDay.Day(31),
        RollDay.IMM(),
        Adjuster.Actual(),
        Adjuster.Following(),
        Adjuster.ModifiedFollowing(),
        Adjuster.Previous(),
        Adjuster.ModifiedPrevious(),
        Adjuster.FollowingSettle(),
        Adjuster.ModifiedFollowingSettle(),
        Adjuster.PreviousSettle(),
        Adjuster.ModifiedPreviousSettle(),
        Adjuster.BusDaysLagSettle(4),
        Adjuster.CalDaysLagSettle(2),
        Adjuster.FollowingExLast(),
        Adjuster.FollowingExLastSettle(),
        Adjuster.BusDaysLagSettleInAdvance(2),
        Frequency.Months(4, RollDay.Day(2)),
        Frequency.Months(4, None),
        Frequency.BusDays(2, NamedCal("tgt")),
        Frequency.Zero(),
        Frequency.CalDays(3),
    ],
)
def test_complex_enum_pickle(enum):
    pickled = pickle.dumps(enum)
    unpickled = pickle.loads(pickled)
    assert unpickled == enum
