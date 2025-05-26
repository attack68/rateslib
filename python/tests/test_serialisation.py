import pytest
from rateslib.calendars import get_calendar
from rateslib.curves.curves import _CurveMeta
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, Variable
from rateslib.json import from_json


@pytest.mark.parametrize("calendar", [get_calendar("tgt"), get_calendar(NoInput(0))])
@pytest.mark.parametrize(
    "index_base",
    [
        100.0,
        Dual(100.0, ["v"], []),
        Dual2(100.0, ["v"], [], []),
        NoInput(0),
    ],
)
@pytest.mark.parametrize("collateral", [None, "usd"])
def test_curvemeta_json_round_trip(calendar, index_base, collateral):
    obj = _CurveMeta(
        calendar=calendar,
        convention="act365f",
        modifier="MF",
        index_base=index_base,
        index_lag=1,
        collateral=collateral,
    )
    json_text = obj._to_json()
    round_trip = from_json(json_text)
    assert round_trip == obj
