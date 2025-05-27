import pytest
from datetime import datetime as dt

from rateslib.calendars import get_calendar
from rateslib.curves.utils import _CurveMeta, _CurveSpline
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, Variable
from rateslib.serialization import from_json
from rateslib.serialization.utils import _enum_to_json


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


def test_curvespline_json_round_trip():
    obj = _CurveSpline(t=[dt(2000, 1, 1), dt(2002, 1, 1)], endpoints=("natural", "natural"))
    json_text = obj._to_json()
    round_trip = from_json(json_text)
    assert round_trip == obj


@pytest.mark.parametrize("value", [-1, 0, 1])
def test_no_input_round_trip(value):
    obj = NoInput(value)
    json = _enum_to_json(obj)
    result = from_json(json)
    assert result == obj
