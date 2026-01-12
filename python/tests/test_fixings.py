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
from pandas import Series
from rateslib import dt, fixings
from rateslib.curves import Curve
from rateslib.data.fixings import (
    FloatRateIndex,
    FloatRateSeries,
    FXFixing,
    FXIndex,
    RFRFixing,
    _FXFixingMajor,
    _UnitFixing,
)
from rateslib.enums.generics import NoInput
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
        leg2_rate_fixings=fixings["corra"][1],
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


def test_add_fixings_directly_with_specific_state() -> None:
    s = Series(
        index=[dt(2000, 2, 1), dt(2000, 3, 1), dt(2000, 1, 1)],
        data=[200.0, 300.0, 100.0],
    )
    fixings.add("my_values", s, 10103)
    assert fixings["my_values"][0] == 10103
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


def test_series_combine():
    from rateslib.periods.protocols.fixings import _s2_before_s1

    s1 = Series(index=[2, 3], data=[100.0, 200.0])
    s2 = Series(index=[1, 2], data=[300.0, 400.0])
    result = s1.combine(s2, _s2_before_s1)
    assert all(result == Series(index=[1, 2, 3], data=[300.0, 400.0, 200.0]))


def test_reset_doc():
    fx_fixing1 = FXFixing(delivery=dt(2021, 1, 1), fx_index="eurusd", identifier="A")
    fx_fixing2 = FXFixing(delivery=dt(2021, 1, 1), fx_index="gbpusd", identifier="B")
    fixings.add("A_eurusd", Series(index=[dt(2020, 12, 30)], data=[1.1]), state=100)
    fixings.add("B_gbpusd", Series(index=[dt(2020, 12, 30)], data=[1.4]), state=200)

    # data is populated from the available Series
    assert fx_fixing1.value == 1.1
    assert fx_fixing2.value == 1.4

    # fixings are reset according to the data state
    fx_fixing1.reset(state=100)
    fx_fixing2.reset(state=100)

    # only the private data for fixing1 is removed because of its link to the data state
    assert fx_fixing1._value == NoInput.blank
    assert fx_fixing2._value == 1.4
    fixings.pop("A_eurusd")
    fixings.pop("B_gbpusd")


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


class TestFXFixing:
    def test_direct(self) -> None:
        name = str(hash(os.urandom(8)))
        fixings.add(name + "_USDRUB", Series(index=[dt(2000, 1, 1)], data=[2.0]))

        fx_fixing = FXFixing(
            publication=dt(2000, 1, 1),
            fx_index=FXIndex("usdrub", "fed", 2),
            identifier=name,
        )
        assert fx_fixing.value == 2.0
        fixings.pop(name + "_USDRUB")

    def test_inverted(self) -> None:
        name = str(hash(os.urandom(8)))
        fixings.add(name + "_USDRUB", Series(index=[dt(2000, 1, 1)], data=[2.0]))

        fx_fixing = FXFixing(
            publication=dt(2000, 1, 1),
            fx_index=FXIndex("rubusd", "fed", 2),
            identifier=name,
        )
        assert fx_fixing.value == 0.5
        fixings.pop(name + "_USDRUB")

    def test_cross1(self) -> None:
        name = str(hash(os.urandom(8)))

        fixings.add(name + "_USDRUB", Series(index=[dt(2000, 1, 1)], data=[2.0]))
        fixings.add(name + "_USDINR", Series(index=[dt(2000, 1, 1)], data=[4.0]))

        fx_fixing = FXFixing(
            publication=dt(2000, 1, 1),
            fx_index=FXIndex("rubinr", "fed", 2),
            identifier=name,
        )
        assert fx_fixing.value == 1 / 2.0 * 4.0
        fixings.pop(name + "_USDRUB")
        fixings.pop(name + "_USDINR")

    def test_cross2(self) -> None:
        name = str(hash(os.urandom(8)))

        fixings.add(name + "_RUBUSD", Series(index=[dt(2000, 1, 1)], data=[2.0]))
        fixings.add(name + "_INRUSD", Series(index=[dt(2000, 1, 1)], data=[4.0]))

        fx_fixing = FXFixing(
            publication=dt(2000, 1, 1),
            fx_index=FXIndex("rubinr", "fed", 2),
            identifier=name,
        )
        assert fx_fixing.value == 2.0 * 1 / 4.0
        fixings.pop(name + "_RUBUSD")
        fixings.pop(name + "_INRUSD")

    def test_reset(self):
        fx_fixing = FXFixing(
            publication=dt(2000, 1, 1), fx_index=FXIndex("rubusd", "fed", 1), identifier="test"
        )
        fixings.add("test_USDRUB", Series(index=[dt(2000, 1, 1)], data=[2.0]))
        assert fx_fixing.value == 0.5
        fx_fixing.reset(state=1)
        assert fx_fixing._value == 0.5
        fx_fixing.reset(state=fixings["TEST_USDRUB"][0])
        assert fx_fixing._value == NoInput(0)
        fixings.pop("test_USDRUB")

    def test_no_state_update(self):
        # test that the fixing value and state is updated at the appropriate times.
        fx_fixing = FXFixing(
            delivery=dt(2000, 1, 1),
            fx_index=FXIndex("rubusd", "fed", 1, "all", 0),
            identifier="test",
        )
        fixings.add("test_USDRUB", Series(index=[dt(2000, 1, 1)], data=[2.0]))
        assert fx_fixing.value == 0.5
        old_state = fx_fixing._state
        fixings.pop("test_USDRUB")
        fixings.add("test_USDRUB", Series(index=[dt(2000, 1, 1)], data=[5.0]))
        # value and state are unchanged
        assert fx_fixing.value == 0.5
        assert fx_fixing._state == old_state
        fx_fixing.reset()
        # value are state are now set after reset
        assert fx_fixing.value == 0.20
        assert fx_fixing._state == hash(fixings["TEST_USDRUB"][0])
        fixings.pop("test_USDRUB")

    # test all cross constructions

    def test_construct_1_major_usd(self):
        fx_fixing = FXFixing(fx_index="usdeur", publication=dt(2000, 1, 1))
        assert isinstance(fx_fixing.fx_fixing1, _FXFixingMajor)
        assert isinstance(fx_fixing.fx_fixing2, _UnitFixing)

        fx_fixing = FXFixing(fx_index="eurusd", publication=dt(2000, 1, 1))
        assert isinstance(fx_fixing.fx_fixing1, _FXFixingMajor)
        assert isinstance(fx_fixing.fx_fixing2, _UnitFixing)

        fx_fixing = FXFixing(fx_index=FXIndex("usdbrl", "fed", 2), publication=dt(2000, 1, 1))
        assert isinstance(fx_fixing.fx_fixing1, _FXFixingMajor)
        assert isinstance(fx_fixing.fx_fixing2, _UnitFixing)

        fx_fixing = FXFixing(fx_index=FXIndex("brlusd", "fed", 2), publication=dt(2000, 1, 1))
        assert isinstance(fx_fixing.fx_fixing1, _FXFixingMajor)
        assert isinstance(fx_fixing.fx_fixing2, _UnitFixing)

    def test_construct_1_major_eur(self):
        fx_fixing = FXFixing(fx_index="eursek", publication=dt(2000, 1, 1))
        assert isinstance(fx_fixing.fx_fixing1, _FXFixingMajor)
        assert isinstance(fx_fixing.fx_fixing2, _UnitFixing)

        fx_fixing = FXFixing(fx_index="sekeur", publication=dt(2000, 1, 1))
        assert isinstance(fx_fixing.fx_fixing1, _FXFixingMajor)
        assert isinstance(fx_fixing.fx_fixing2, _UnitFixing)

    def test_construct_2_major_eur(self):
        fx_fixing = FXFixing(fx_index="usdsek", publication=dt(2000, 1, 1))
        assert isinstance(fx_fixing.fx_fixing1, _FXFixingMajor)
        assert fx_fixing.fx_fixing1.pair == "usdeur"
        assert isinstance(fx_fixing.fx_fixing2, _FXFixingMajor)
        assert isinstance(fx_fixing.fx_fixing3, _UnitFixing)

        fx_fixing = FXFixing(fx_index="sekusd", publication=dt(2000, 1, 1))
        assert isinstance(fx_fixing.fx_fixing1, _FXFixingMajor)
        assert fx_fixing.fx_fixing1.pair == "sekeur"
        assert isinstance(fx_fixing.fx_fixing2, _FXFixingMajor)
        assert isinstance(fx_fixing.fx_fixing3, _UnitFixing)

        fx_fixing = FXFixing(fx_index="seknok", publication=dt(2000, 1, 1))
        assert isinstance(fx_fixing.fx_fixing1, _FXFixingMajor)
        assert fx_fixing.fx_fixing1.pair == "sekeur"
        assert isinstance(fx_fixing.fx_fixing2, _FXFixingMajor)
        assert isinstance(fx_fixing.fx_fixing3, _UnitFixing)

    def test_construct_2_major_usd(self):
        fx_fixing = FXFixing(fx_index="eurgbp", publication=dt(2000, 1, 1))
        assert isinstance(fx_fixing.fx_fixing1, _FXFixingMajor)
        assert fx_fixing.fx_fixing1.pair == "eurusd"
        assert isinstance(fx_fixing.fx_fixing2, _FXFixingMajor)
        assert isinstance(fx_fixing.fx_fixing3, _UnitFixing)

        fx_fixing = FXFixing(fx_index="gbpeur", publication=dt(2000, 1, 1))
        assert isinstance(fx_fixing.fx_fixing1, _FXFixingMajor)
        assert fx_fixing.fx_fixing1.pair == "gbpusd"
        assert isinstance(fx_fixing.fx_fixing2, _FXFixingMajor)
        assert isinstance(fx_fixing.fx_fixing3, _UnitFixing)

        fx_fixing = FXFixing(fx_index="gbpcad", publication=dt(2000, 1, 1))
        assert isinstance(fx_fixing.fx_fixing1, _FXFixingMajor)
        assert fx_fixing.fx_fixing1.pair == "gbpusd"
        assert isinstance(fx_fixing.fx_fixing2, _FXFixingMajor)
        assert isinstance(fx_fixing.fx_fixing3, _UnitFixing)

    def test_construct_3_major(self):
        fx_fixing = FXFixing(fx_index=FXIndex("nokcad", "tro", 2), publication=dt(2000, 1, 1))
        assert isinstance(fx_fixing.fx_fixing1, _FXFixingMajor)
        assert fx_fixing.fx_fixing1.pair == "nokeur"
        assert isinstance(fx_fixing.fx_fixing2, _FXFixingMajor)
        assert fx_fixing.fx_fixing2.pair == "eurusd"
        assert isinstance(fx_fixing.fx_fixing3, _FXFixingMajor)
        assert fx_fixing.fx_fixing3.pair == "usdcad"

        fx_fixing = FXFixing(fx_index=FXIndex("cadnok", "tro", 2), publication=dt(2000, 1, 1))
        assert isinstance(fx_fixing.fx_fixing1, _FXFixingMajor)
        assert fx_fixing.fx_fixing1.pair == "cadusd"
        assert isinstance(fx_fixing.fx_fixing2, _FXFixingMajor)
        assert fx_fixing.fx_fixing2.pair == "usdeur"
        assert isinstance(fx_fixing.fx_fixing3, _FXFixingMajor)
        assert fx_fixing.fx_fixing3.pair == "eurnok"


class TestFXFixingMajor:
    def test_direct(self) -> None:
        name = str(hash(os.urandom(8)))
        fixings.add(name + "_USDRUB", Series(index=[dt(2000, 1, 1)], data=[2.0]))

        fx_fixing = _FXFixingMajor(
            publication=dt(2000, 1, 1),
            fx_index=FXIndex("usdrub", "fed", 2),
            identifier=name,
        )
        assert fx_fixing.value == 2.0
        fixings.pop(name + "_USDRUB")

    def test_inverted(self) -> None:
        name = str(hash(os.urandom(8)))
        fixings.add(name + "_USDRUB", Series(index=[dt(2000, 1, 1)], data=[2.0]))

        fx_fixing = _FXFixingMajor(
            publication=dt(2000, 1, 1),
            fx_index=FXIndex("rubusd", "fed", 2),
            identifier=name,
        )
        assert fx_fixing.value == 0.5
        fixings.pop(name + "_USDRUB")

    def test_cross1(self) -> None:
        name = str(hash(os.urandom(8)))

        fixings.add(name + "_INRRUB", Series(index=[dt(2000, 1, 1)], data=[2.0]))

        fx_fixing = _FXFixingMajor(
            publication=dt(2000, 1, 1),
            fx_index=FXIndex("rubinr", "fed", 2),
            identifier=name,
        )
        assert fx_fixing.value == 1 / 2.0
        fixings.pop(name + "_INRRUB")

    def test_reset(self):
        fx_fixing = _FXFixingMajor(
            publication=dt(2000, 1, 1), fx_index=FXIndex("rubusd", "fed", 1), identifier="test"
        )
        fixings.add("test_USDRUB", Series(index=[dt(2000, 1, 1)], data=[2.0]))
        assert fx_fixing.value == 0.5
        fx_fixing.reset(state=1)
        assert fx_fixing._value == 0.5
        fx_fixing.reset(state=fixings["TEST_USDRUB"][0])
        assert fx_fixing._value == NoInput(0)
        fixings.pop("test_USDRUB")

    def test_no_state_update(self):
        # test that the fixing value and state is updated at the appropriate times.
        fx_fixing = _FXFixingMajor(
            delivery=dt(2000, 1, 1),
            fx_index=FXIndex("rubusd", "fed", 1, "all", 0),
            identifier="test",
        )
        fixings.add("test_USDRUB", Series(index=[dt(2000, 1, 1)], data=[2.0]))
        assert fx_fixing.value == 0.5
        old_state = fx_fixing._state
        fixings.pop("test_USDRUB")
        fixings.add("test_USDRUB", Series(index=[dt(2000, 1, 1)], data=[5.0]))
        # value and state are unchanged
        assert fx_fixing.value == 0.5
        assert fx_fixing._state == old_state
        fx_fixing.reset()
        # value are state are now set after reset
        assert fx_fixing.value == 0.20
        assert fx_fixing._state == fixings["TEST_USDRUB"][0]
        fixings.pop("test_USDRUB")
