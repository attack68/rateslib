import pytest
from pandas import Series
from rateslib import defaults, dt
from rateslib.curves import Curve
from rateslib.instruments import IRS
from rateslib.scheduling import Adjuster, get_calendar


@pytest.mark.parametrize("name", ["estr", "sonia", "sofr", "swestr", "nowa"])
def test_fixings(name) -> None:
    result = defaults.fixings[name]
    assert isinstance(result, Series)


def test_calendar_matches_fixings_corra() -> None:
    # this should run without warnings or errors if the "tro" calendar matches the fixings.
    swap = IRS(
        effective=dt(2017, 1, 1),
        termination=dt(2023, 7, 1),
        frequency="A",
        leg2_fixings=defaults.fixings["corra"],
        calendar="tro",
        fixed_rate=1.0,
    )
    curve = Curve({dt(2017, 1, 1): 1.0, dt(2027, 1, 1): 1.0}, calendar="tro")
    swap.npv(curves=curve)


def test_fixings_raises_file_error() -> None:
    with pytest.raises(ValueError, match="Fixing data for the index "):
        defaults.fixings["nofile"]


def test_add_fixings_directly() -> None:
    s = Series(
        index=[dt(2000, 2, 1), dt(2000, 3, 1), dt(2000, 1, 1)],
        data=[200.0, 300.0, 100.0],
    )
    defaults.fixings.add_series("my_values", s)
    assert defaults.fixings["my_values"].is_monotonic_increasing
    assert defaults.fixings["my_values"].name == "rate"
    assert defaults.fixings["my_values"].index.name == "reference_date"
    defaults.fixings.remove_series("my_values")


def test_get_stub_ibor_fixings() -> None:
    s = Series(
        index=[dt(2000, 2, 1), dt(2000, 3, 1), dt(2000, 1, 1)],
        data=[200.0, 300.0, 100.0],
    )
    defaults.fixings.add_series("usd_IBOR_3w", s)
    defaults.fixings.add_series("usd_IBOR_1m", s)
    defaults.fixings.add_series("usd_IBOR_2m", s)
    defaults.fixings.add_series("USD_ibor_3M", s)
    s, _, _ = defaults.fixings.get_stub_ibor_fixings(
        value_start_date=dt(2000, 1, 1),
        value_end_date=dt(2000, 2, 15),
        fixing_calendar=get_calendar("nyc"),
        fixing_modifier=Adjuster.Following(),
        fixing_identifier="USD_IBOR",
        fixing_date=dt(1999, 12, 30),
    )
    defaults.fixings.remove_series("usd_IBOR_3w")
    defaults.fixings.remove_series("usd_IBOR_1m")
    defaults.fixings.remove_series("usd_IBOR_2m")
    defaults.fixings.remove_series("USD_ibor_3M")
    assert s == ["1M", "2M"]


@pytest.mark.parametrize(("fixing"), [True, False])
def test_get_stub_ibor_fixings_no_left(fixing) -> None:
    s = Series(
        index=[dt(2000, 2, 1), dt(2000, 3, 1), dt(2000, 1, 1)],
        data=[200.0, 300.0, 100.0],
    )
    if fixing:
        s[dt(1999, 12, 30)] = 12345.0
    defaults.fixings.add_series("usd_IBOR_2w", s)
    defaults.fixings.add_series("usd_IBOR_3w", s)
    s, _, f = defaults.fixings.get_stub_ibor_fixings(
        value_start_date=dt(2000, 1, 1),
        value_end_date=dt(2000, 1, 8),
        fixing_calendar=get_calendar("nyc"),
        fixing_modifier=Adjuster.Following(),
        fixing_identifier="USD_IBOR",
        fixing_date=dt(1999, 12, 30),
    )
    defaults.fixings.remove_series("usd_IBOR_2w")
    defaults.fixings.remove_series("usd_IBOR_3w")
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
    defaults.fixings.add_series("usd_IBOR_2m", s)
    defaults.fixings.add_series("USD_ibor_3M", s)
    s, _, f = defaults.fixings.get_stub_ibor_fixings(
        value_start_date=dt(2000, 1, 1),
        value_end_date=dt(2000, 7, 8),
        fixing_calendar=get_calendar("nyc"),
        fixing_modifier=Adjuster.Following(),
        fixing_identifier="USD_IBOR",
        fixing_date=dt(1999, 12, 30),
    )
    defaults.fixings.remove_series("usd_IBOR_2m")
    defaults.fixings.remove_series("USD_ibor_3M")
    assert s == ["3M"]
    assert f == [12345.0 if fixing else None]


def test_get_stub_ibor_fixings_no_left_no_right() -> None:
    s, _, _ = defaults.fixings.get_stub_ibor_fixings(
        value_start_date=dt(2000, 1, 1),
        value_end_date=dt(2000, 7, 8),
        fixing_calendar=get_calendar("nyc"),
        fixing_modifier=Adjuster.Following(),
        fixing_identifier="USD_NONE",
        fixing_date=dt(1999, 12, 30),
    )
    assert s == []
