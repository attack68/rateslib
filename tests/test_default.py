import pytest
from pandas import Series

import context
from rateslib import defaults, default_context, __version__
from rateslib.instruments import IRS
from rateslib.curves import Curve
from rateslib import dt


def test_version():
    assert __version__ == "1.2.3"


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

    defaults.reset_defaults()
    assert defaults.modifier == "MF"


def test_calendar_matches_fixings_corra():
    # this should run without warnings or errors if the "tro" calendar matches the fixings.
    swap = IRS(
        effective=dt(2017, 1, 1),
        termination=dt(2023, 7, 1),
        frequency="A",
        leg2_fixings=defaults.fixings["corra"],
        calendar="tro",
        fixed_rate=1.0
    )
    curve = Curve({dt(2017, 1, 1): 1.0, dt(2027, 1, 1): 1.0}, calendar="tro")
    swap.npv(curves=curve)


def test_fixings_raises_file_error():
    with pytest.raises(ValueError, match="Fixing data for the index "):
        defaults.fixings["nofile"]