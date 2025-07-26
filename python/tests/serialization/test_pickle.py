import pickle

import pytest
from rateslib import (
    ADOrder,
    Cal,
    Dual,
    Dual2,
    FXRates,
    Imm,
    NamedCal,
    StubInference,
    UnionCal,
    Variable,
    dt,
)


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
    ],
)
def test_simple_enum_equality(a1, a2, b1):
    assert a1 == a2
    assert a2 != b1


@pytest.mark.parametrize(
    ("enum", "method_filter"),
    [(Imm, ["next", "get", "validate"]), (StubInference, []), (ADOrder, [])],
)
def test_simple_enum_pickle(enum, method_filter):
    variants = [v for v in enum.__dict__ if "__" not in v and v not in method_filter]
    for v in variants:
        obj = enum.__dict__[v]
        pickled = pickle.dumps(obj)
        unpickled = pickle.loads(pickled)
        assert unpickled == enum.__dict__[v]
