import pytest
from rateslib import Curve, Dual, Dual2, FXForwards, FXRates, dt, from_json
from rateslib.scheduling import Adjuster, Frequency, Imm, NamedCal, RollDay, StubInference


@pytest.mark.parametrize(
    "obj",
    [
        Dual(2, vars=["v0", "v2"], dual=[0, 3]),
        Dual2(2.5, ["a", "bb"], [1.2, 3.4], []),
        FXRates({"usdnok": 8.0, "eurusd": 1.05}),
        Imm.Wed1_Post9_HMUZ,
        StubInference.LongFront,
        RollDay.Day(31),
        RollDay.IMM(),
        Frequency.Zero(),
        Frequency.CalDays(3),
        Frequency.BusDays(3, NamedCal("tgt")),
        Frequency.Months(4, None),
        Frequency.Months(3, RollDay.IMM()),
        Adjuster.ModifiedFollowing(),
        Adjuster.BusDaysLagSettle(2),
    ],
)
def test_json_round_trip(obj) -> None:
    jstring = obj.to_json()
    reconstituted = from_json(jstring)
    assert obj == reconstituted
