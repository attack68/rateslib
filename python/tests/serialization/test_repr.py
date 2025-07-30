import pytest
from rateslib import dt
from rateslib.scheduling import Adjuster, Frequency, Imm, NamedCal, RollDay, Schedule, StubInference
from rateslib.dual import Dual, Dual2
from rateslib.splines import PPSplineF64, PPSplineDual, PPSplineDual2

@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (Imm.Wed1_Post9_HMUZ, "Imm.Wed1_Post9_HMUZ"),
        (StubInference.ShortFront, "StubInference.ShortFront"),
        (RollDay.Day(31), "RollDay.Day(31)"),
        (RollDay.IMM(), "RollDay.IMM"),
        (Frequency.Zero(), "Frequency.Zero"),
        (Frequency.CalDays(2), "Frequency.CalDays(2)"),
        (Frequency.BusDays(3, NamedCal("tgt")), "Frequency.BusDays(3, ...)"),
        (Frequency.Months(2, RollDay.Day(31)), "Frequency.Months(2, Day(31))"),
        (Frequency.Months(4, None), "Frequency.Months(4, None)"),
        (Adjuster.ModifiedFollowing(), "Adjuster.ModifiedFollowing"),
        (Adjuster.BusDaysLagSettle(4), "Adjuster.BusDaysLagSettle(4)"),
        (Schedule(dt(2000, 1, 1), dt(2001, 2, 1), "M"), "Schedule"),
        (PPSplineF64(3, [0,0,0,1,1,1], [0.1, 0.2, 0.3]), "PPSplineF64"),
        (PPSplineDual(3, [0,0,0,1,1,1], [Dual(0.1, [], []), Dual(0.2, [], []), Dual(0.3, [], [])]), "PPSplineDual"),
        (PPSplineDual2(3, [0,0,0,1,1,1], [Dual2(0.1, [], [], []), Dual2(0.2, [], [], []), Dual2(0.3, [], [], [])]), "PPSplineDual2"),
    ],
)
def test_repr_strings(obj, expected) -> None:
    repr_ = obj.__repr__()
    assert f"<rl.{expected} at" in repr_


def test_unique_repr_simple_enum():
    a = Imm.Wed1_Post9_HMUZ
    b = Imm.Wed1_Post9_HMUZ
    assert a.__repr__() == b.__repr__()
    assert hex(id(a)) == hex(id(b))
