import pytest
from pandas import Series
from rateslib import __version__, default_context, defaults, dt
from rateslib.curves import Curve
from rateslib.instruments import IRS


def test_version():
    assert __version__ == "1.4.0"


@pytest.mark.parametrize("name", ["estr", "sonia", "sofr", "swestr", "nowa"])
def test_fixings(name):
    result = defaults.fixings[name]
    assert isinstance(result, Series)


def test_context_raises():
    with pytest.raises(ValueError, match="Need to invoke as "):
        default_context("only 1 arg")


def test_reset_defaults():
    defaults.modifier = "MP"
    assert defaults.modifier == "MP"
    defaults.calendars["TEST"] = 10.0
    assert defaults.calendars["TEST"] == 10.0

    defaults.reset_defaults()
    assert defaults.modifier == "MF"
    assert "TEST" not in defaults.calendars


def test_calendar_matches_fixings_corra():
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


def test_fixings_raises_file_error():
    with pytest.raises(ValueError, match="Fixing data for the index "):
        defaults.fixings["nofile"]
