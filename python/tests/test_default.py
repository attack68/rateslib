import pytest
from rateslib import __version__, default_context, defaults


def test_version() -> None:
    assert __version__ == "2.5.dev0"


def test_context_raises() -> None:
    with pytest.raises(ValueError, match="Need to invoke as "):
        default_context("only 1 arg")


def test_reset_defaults() -> None:
    defaults.modifier = "MP"
    defaults.base_currency = "gbp"
    assert defaults.modifier == "MP"
    assert defaults.base_currency == "gbp"
    defaults.calendars["TEST"] = 10.0
    assert defaults.calendars["TEST"] == 10.0

    defaults.reset_defaults()
    assert defaults.modifier == "MF"
    assert defaults.base_currency == "usd"
    assert "TEST" not in defaults.calendars


def test_defaults_singleton() -> None:
    from rateslib.default import Defaults

    other = Defaults()
    assert id(other) == id(defaults)
