# SPDX-License-Identifier: LicenseRef-Rateslib-Dual
#
# Copyright (c) 2026 Siffrorna Technology Limited
#
# Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
# Source-available, not open source.
#
# See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
# and/or contact info (at) rateslib (dot) com
####################################################################################################

import os

import pytest
from rateslib import (
    Licence,
    NamedCal,
    __version__,
    default_context,
    defaults,
    dt,
    fixings,
    licence,
)
from rateslib.verify import LicenceNotice, _LicenceStatus


def test_version() -> None:
    assert __version__ == "2.7.0"


def test_context_raises() -> None:
    with pytest.raises(ValueError, match="Need to invoke as "):
        default_context("only 1 arg")


def test_reset_defaults() -> None:
    defaults.modifier = "MP"
    defaults.base_currency = "gbp"
    assert defaults.modifier == "MP"
    assert defaults.base_currency == "gbp"

    defaults.reset_defaults()
    assert defaults.modifier == "MF"
    assert defaults.base_currency == "usd"


def test_defaults_singleton() -> None:
    from rateslib.default import Defaults

    other = Defaults()
    assert id(other) == id(defaults)


def test_fixings_singleton() -> None:
    from rateslib.data.loader import Fixings

    other = Fixings()
    assert id(other) == id(fixings)


def test_fx_index_change() -> None:
    # test that default fx indexes can be overwritten and are loaded by constructed objects
    from rateslib.data.fixings import FXFixing, FXIndex
    from rateslib.scheduling import Adjuster

    eurusd = FXFixing("eurusd", dt(2000, 1, 1))
    assert eurusd.fx_index.calendar == NamedCal("tgt|fed")
    assert eurusd.fx_index.settle == Adjuster.BusDaysLagSettle(2)
    defaults.fx_index["eurusd"] = {"pair": "eurusd", "calendar": "stk", "settle": 3}
    eurusd = FXFixing("eurusd", dt(2000, 1, 1))
    assert eurusd.fx_index.calendar == NamedCal("stk")
    assert eurusd.fx_index.settle == Adjuster.BusDaysLagSettle(3)

    defaults.reset_defaults()
    assert defaults.fx_index["eurusd"]["calendar"] == NamedCal("tgt|fed")


def test_float_series_change():
    from rateslib import IRS

    with pytest.raises(ValueError, match="The FloatRateSeries: 'monkey' was not found "):
        IRS(dt(2000, 1, 1), "1y", "A", leg2_fixing_series="monkey")

    defaults.float_series["monkey"] = dict(
        lag=0, calendar="nyc", modifier="f", eom=False, convention="act360"
    )
    IRS(dt(2000, 1, 1), "1y", "A", leg2_fixing_series="monkey")

    defaults.reset_defaults()
    assert "monkey" not in defaults.float_series


def collect_and_remove_licence() -> tuple[str | None, str | None]:
    env_licence = os.getenv("RATESLIB_LICENCE")
    if env_licence is not None:
        del os.environ["RATESLIB_LICENCE"]
    try:
        file_licence = licence.print_licence()
        licence.remove_licence()
    except ValueError:
        file_licence = None
    return env_licence, file_licence


def replace_collected_licence(env_licence, file_licence) -> None:
    if env_licence is not None:
        os.environ["RATESLIB_LICENCE"] = env_licence
    if file_licence is not None:
        licence.add_licence(file_licence)


class TestLicence:
    def test_valid_licence(self):
        # test that this system has a valid licence
        assert licence.status == _LicenceStatus.VALID

    @pytest.mark.skipif(
        os.getenv("RATESLIB_LICENCE") is not None, reason="env licence already tested."
    )
    def test_env_licence(self):
        # this test relies on `test_valid_licence`
        assert licence.status == _LicenceStatus.VALID  # licence is loaded from file.
        os.environ["RATESLIB_LICENCE"] = licence.print_licence()
        licence.remove_licence()  # remove the file licence
        x = Licence()
        assert x.status == _LicenceStatus.VALID
        licence.add_licence(os.environ["RATESLIB_LICENCE"])
        del os.environ["RATESLIB_LICENCE"]

    def test_licence_no_licence_warning(self):
        # test just the
        env_licence, file_licence = collect_and_remove_licence()
        with pytest.warns(LicenceNotice, match="No commercial licence is registered"):
            Licence()
        replace_collected_licence(env_licence, file_licence)

    def test_licence_warning_for_expired_as_file(self):
        env_licence, file_licence = collect_and_remove_licence()
        licence.add_licence(
            '{"expiry": "1900-01-01", "id": "Rateslib Tests", "xkey": "0x2cec1be74d8b2d2bdfa41aec384a4a8ede06c8c7873d6130035c19fcf244b5b92e29c7087a5e51c453a1fe7da345a689ef3d0953b8841ab1b3895a69a209aa529ff3e4d6b8217ce16b37c5572d737ece0a7f381696a3f3901bced9f843b48504930b25d204d910955f52c76eccd208a975a3a0e4433d70dd090ef5adb8de83cb", "name": "System"}'  # noqa: E501
        )
        with pytest.warns(LicenceNotice, match="expired on 1900-01-01"):
            Licence()
        licence.remove_licence()
        replace_collected_licence(env_licence, file_licence)

    def test_licence_warning_for_expired_as_env_var(self):
        env_licence, file_licence = collect_and_remove_licence()
        os.environ["RATESLIB_LICENCE"] = (
            '{"expiry": "1900-01-01", "id": "Rateslib Tests", "xkey": "0x2cec1be74d8b2d2bdfa41aec384a4a8ede06c8c7873d6130035c19fcf244b5b92e29c7087a5e51c453a1fe7da345a689ef3d0953b8841ab1b3895a69a209aa529ff3e4d6b8217ce16b37c5572d737ece0a7f381696a3f3901bced9f843b48504930b25d204d910955f52c76eccd208a975a3a0e4433d70dd090ef5adb8de83cb", "name": "System"}'  # noqa: E501
        )
        with pytest.warns(LicenceNotice, match="expired on 1900-01-01"):
            Licence()
        del os.environ["RATESLIB_LICENCE"]
        replace_collected_licence(env_licence, file_licence)

    def test_invalid_signature(self):
        env_licence, file_licence = collect_and_remove_licence()
        os.environ["RATESLIB_LICENCE"] = (
            '{"expiry": "2100-01-01", "id": "Rateslib Tests", "xkey": "0x2cec1be74d8b2d2bdfa41aec384a4a8ede06c8c7873d6130035c19fcf244b5b92e29c7087a5e51c453a1fe7da345a689ef3d0953b8841ab1b3895a69a209aa529ff3e4d6b8217ce16b37c5572d737ece0a7f381696a3f3901bced9f843b48504930b25d204d910955f52c76eccd208a975a3a0e4433d70dd090ef5adb8de83cb", "name": "System"}'  # noqa: E501
        )
        with pytest.warns(LicenceNotice, match="An invalid licence file is detected"):
            Licence()
        del os.environ["RATESLIB_LICENCE"]
        replace_collected_licence(env_licence, file_licence)

    @pytest.mark.parametrize(
        "licence_text",
        [
            "garbage",
            '{"expiry": "1900-01-01", "id": "Rateslib Tests", "xkey": "0x2cec1", "name": "System"}',
        ],
    )
    def test_add_invalid_licence(self, licence_text):
        with pytest.raises(ValueError):
            licence.add_licence(licence_text)

    @pytest.mark.parametrize(
        "licence_text",
        [
            '{"name": "RL Expiry Test", "xkey": "0x68178a21511a36f8270bb4f73451bf3a6575e23e11bc9d0ebead841fa77bfef16cbae1341ad2e6d80f0b717923a48fbd3580eb6cc216a31c0d23618a32e8b2773cc52998e6bcb0315a8f46d003ce04f7ddeb8c19e66a16c73d2e925218dff044ba5f43f7d05503626e89fadbf85751807737f73c55b2048f96fd331b202abe45"}',  # noqa: E501
            '{"name": "RL xkey missing"}',
        ],
    )
    def test_licence_missing_keys(self, licence_text):
        from rateslib.verify import _verify_licence

        assert _verify_licence(licence_text) is None
