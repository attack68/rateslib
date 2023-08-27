import pytest
from pandas import Series

import context
from rateslib import defaults, default_context


@pytest.mark.parametrize("name", ["estr", "sonia", "sofr", "swestr", "nowa"])
def test_fixings(name):
    result = getattr(defaults.fixings, name, None)
    assert isinstance(result, Series)


def test_fixings_raises():
    with pytest.raises(NotImplementedError, match="Swiss SIX exchange licence not available."):
        getattr(defaults.fixings, "saron", None)


def test_context_raises():
    with pytest.raises(ValueError, match="Need to invoke as "):
        default_context("only 1 arg")


def test_reset_defaults():
    defaults.modifier = "MP"
    assert defaults.modifier == "MP"

    defaults.reset_defaults()
    assert defaults.modifier == "MF"
