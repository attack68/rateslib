import os

import pytest
from pandas import Series
from rateslib import dt, fixings
from rateslib.curves import Curve
from rateslib.data.fixings import FloatRateIndex, FloatRateSeries, RFRFixing
from rateslib.instruments import IRS
from rateslib.scheduling import Adjuster, get_calendar


@pytest.mark.parametrize("name", ["estr", "sonia", "sofr", "swestr", "nowa"])
def test_fixings(name) -> None:
    result = fixings[name]
    assert isinstance(result[1], Series)


def test_calendar_matches_fixings_corra() -> None:
    # this should run without warnings or errors if the "tro" calendar matches the fixings.
    swap = IRS(
        effective=dt(2017, 1, 1),
        termination=dt(2023, 7, 1),
        frequency="A",
        leg2_fixings=fixings["corra"][1],
        calendar="tro",
        fixed_rate=1.0,
    )
    curve = Curve({dt(2017, 1, 1): 1.0, dt(2027, 1, 1): 1.0}, calendar="tro")
    swap.npv(curves=curve)


def test_fixings_raises_file_error() -> None:
    with pytest.raises(ValueError, match="Fixing data for the index "):
        fixings["nofile"]


def test_add_fixings_directly() -> None:
    s = Series(
        index=[dt(2000, 2, 1), dt(2000, 3, 1), dt(2000, 1, 1)],
        data=[200.0, 300.0, 100.0],
    )
    fixings.add("my_values", s)
    assert fixings["my_values"][1].is_monotonic_increasing
    assert fixings["my_values"][1].name == "rate"
    assert fixings["my_values"][1].index.name == "reference_date"
    fixings.pop("my_values")


def test_get_stub_ibor_fixings() -> None:
    s = Series(
        index=[dt(2000, 2, 1), dt(2000, 3, 1), dt(2000, 1, 1)],
        data=[200.0, 300.0, 100.0],
    )
    fixings.add("usd_IBOR_3w", s)
    fixings.add("usd_IBOR_1m", s)
    fixings.add("usd_IBOR_2m", s)
    fixings.add("USD_ibor_3M", s)
    s, _, _ = fixings.get_stub_ibor_fixings(
        value_start_date=dt(2000, 1, 1),
        value_end_date=dt(2000, 2, 15),
        fixing_calendar=get_calendar("nyc"),
        fixing_modifier=Adjuster.Following(),
        fixing_identifier="USD_IBOR",
        fixing_date=dt(1999, 12, 30),
    )
    fixings.pop("usd_IBOR_3w")
    fixings.pop("usd_IBOR_1m")
    fixings.pop("usd_IBOR_2m")
    fixings.pop("USD_ibor_3M")
    assert s == ["1M", "2M"]


@pytest.mark.parametrize(("fixing"), [True, False])
def test_get_stub_ibor_fixings_no_left(fixing) -> None:
    s = Series(
        index=[dt(2000, 2, 1), dt(2000, 3, 1), dt(2000, 1, 1)],
        data=[200.0, 300.0, 100.0],
    )
    if fixing:
        s[dt(1999, 12, 30)] = 12345.0
    fixings.add("usd_IBOR_2w", s)
    fixings.add("usd_IBOR_3w", s)
    s, _, f = fixings.get_stub_ibor_fixings(
        value_start_date=dt(2000, 1, 1),
        value_end_date=dt(2000, 1, 8),
        fixing_calendar=get_calendar("nyc"),
        fixing_modifier=Adjuster.Following(),
        fixing_identifier="USD_IBOR",
        fixing_date=dt(1999, 12, 30),
    )
    fixings.pop("usd_IBOR_2w")
    fixings.pop("usd_IBOR_3w")
    assert s == ["2W"]
    assert f == [12345.0 if fixing else None]


@pytest.mark.parametrize(("fixing"), [True, False])
def test_get_stub_ibor_fixings_no_right(fixing) -> None:
    s = Series(
        index=[dt(2000, 2, 1), dt(2000, 3, 1), dt(2000, 1, 1)],
        data=[200.0, 300.0, 100.0],
    )
    if fixing:
        s[dt(1999, 12, 30)] = 12345.0
    fixings.add("usd_IBOR_2m", s)
    fixings.add("USD_ibor_3M", s)
    s, _, f = fixings.get_stub_ibor_fixings(
        value_start_date=dt(2000, 1, 1),
        value_end_date=dt(2000, 7, 8),
        fixing_calendar=get_calendar("nyc"),
        fixing_modifier=Adjuster.Following(),
        fixing_identifier="USD_IBOR",
        fixing_date=dt(1999, 12, 30),
    )
    fixings.pop("usd_IBOR_2m")
    fixings.pop("USD_ibor_3M")
    assert s == ["3M"]
    assert f == [12345.0 if fixing else None]


def test_get_stub_ibor_fixings_no_left_no_right() -> None:
    s, _, _ = fixings.get_stub_ibor_fixings(
        value_start_date=dt(2000, 1, 1),
        value_end_date=dt(2000, 7, 8),
        fixing_calendar=get_calendar("nyc"),
        fixing_modifier=Adjuster.Following(),
        fixing_identifier="USD_NONE",
        fixing_date=dt(1999, 12, 30),
    )
    assert s == []


def test_state_id():
    s = Series(
        index=[dt(2000, 2, 1), dt(2000, 3, 1), dt(2000, 1, 1)],
        data=[200.0, 300.0, 100.0],
    )
    fixings.add("usd_IBOR_3w", s)
    before = fixings["usd_IBOR_3w"][0]
    fixings.pop("usd_IBOR_3w")
    fixings.add("usd_IBOR_3w", s)
    assert before != fixings["usd_IBOR_3w"][0]


class TestRFRFixing:
    def test_rfr_lockout(self) -> None:
        name = str(hash(os.urandom(8))) + "_1B"
        estr_1b = Series(
            index=[dt(2025, 9, 12), dt(2025, 9, 15), dt(2025, 9, 16)], data=[1.91, 1.92, 1.93]
        )
        fixings.add(name, estr_1b)
        rfr_fixing = RFRFixing(
            accrual_start=dt(2025, 9, 12),
            accrual_end=dt(2025, 9, 19),
            identifier=name,
            spread_compound_method="NoneSimple",
            fixing_method="RFRLockout",
            method_param=2,
            float_spread=100.0,
            rate_index=FloatRateIndex(frequency="1B", series="eur_rfr"),
        )
        result = rfr_fixing.value
        assert abs(result - 2.9202637862854033) < 1e-10
        assert len(rfr_fixing.populated) == 5
