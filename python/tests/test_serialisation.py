

import pytest
from rateslib.dual import Dual, Dual2, Variable
from rateslib.curves.curves import _CurveMeta
from rateslib.calendars import get_calendar
from rateslib.default import NoInput
from rateslib.instruments import IRS

@pytest.mark.parametrize("calendar", [get_calendar("tgt"), NoInput(0)])
@pytest.mark.parametrize("index_base", [
    100.0,
    Dual(100.0, ["v"],[]),
    Dual2(100.0, ["v"], [], []),
    Variable(100.0, ["v"]),
])
@pytest.mark.parametrize("collateral", [None, "usd"])
def test_curvemeta(calendar, index_base, collateral):
    obj = _CurveMeta(
        calendar=calendar,
        convention="act365f",
        modifier="MF",
        index_base=index_base,
        index_lag=1,
        collateral=collateral,
    )
    round_trip = _CurveMeta.from_json(obj.to_json())
    assert round_trip == obj