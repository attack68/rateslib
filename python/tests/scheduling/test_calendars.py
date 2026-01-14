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

from datetime import datetime as dt

import pytest
from rateslib import defaults, fixings
from rateslib.curves import Curve
from rateslib.default import NoInput
from rateslib.instruments import IRS
from rateslib.scheduling import (
    Adjuster,
    Cal,
    Convention,
    Frequency,
    RollDay,
    UnionCal,
    add_tenor,
    dcf,
    get_calendar,
    get_imm,
    next_imm,
)
from rateslib.scheduling.calendars import _adjust_date, _get_years_and_months, _is_day_type_tenor
from rateslib.scheduling.frequency import _get_frequency, _get_fx_expiry_and_delivery_and_payment


@pytest.fixture
def cal_():
    return Cal([dt(_, 1, 3) for _ in range(1970, 2200)], [5, 6])


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        (dt(2022, 1, 1), True),  # sat
        (dt(2022, 1, 2), True),  # sun
        (dt(2022, 1, 3), True),  # mon new year hol
        (dt(2022, 1, 4), False),  # tues
        (dt(2022, 1, 5), False),  # wed
    ],
)
def test_is_non_bus_day(date, expected, cal_) -> None:
    result = cal_.is_non_bus_day(date)
    assert result == expected


def test_is_non_bus_day_raises() -> None:
    obj = "not a cal object"
    with pytest.raises(AttributeError):
        obj._is_non_bus_day(dt(2022, 1, 1))


@pytest.mark.parametrize(
    "date",
    [
        dt(2021, 12, 29),
        dt(2021, 12, 30),
        dt(2021, 12, 31),
        dt(2021, 1, 1),
        dt(2021, 1, 2),
        dt(2021, 1, 3),
        dt(2021, 1, 4),
        dt(2021, 1, 5),
    ],
)
def test_cal_no_hols(date) -> None:
    cal_no_hols = Cal([], [])
    assert not cal_no_hols.is_non_bus_day(date)


def test_named_cal() -> None:
    ldn_cal = get_calendar("ldn")
    assert ldn_cal.is_non_bus_day(dt(2022, 1, 1))
    assert ldn_cal.is_bus_day(dt(2022, 1, 5))


def test_multiple_named_cal() -> None:
    ldn_cal = get_calendar("ldn")
    stk_cal = get_calendar("stk")

    assert ldn_cal.is_non_bus_day(dt(2023, 1, 2))
    assert stk_cal.is_bus_day(dt(2023, 1, 2))

    assert ldn_cal.is_bus_day(dt(2023, 1, 6))
    assert stk_cal.is_non_bus_day(dt(2023, 1, 6))

    merged_cal = get_calendar("LDN,stk")
    assert merged_cal.is_non_bus_day(dt(2023, 1, 2))
    assert merged_cal.is_non_bus_day(dt(2023, 1, 6))


def test_add_tenor_raises() -> None:
    # this raise is superfluous by the design principles of private methods
    with pytest.raises(ValueError):
        add_tenor(dt(2022, 1, 1), "1X", "mf", None)


@pytest.mark.parametrize(
    ("tenor", "expected"),
    [
        ("1M", dt(2022, 1, 31)),
        ("2m", dt(2022, 2, 28)),
        ("6M", dt(2022, 6, 30)),
        ("1d", dt(2022, 1, 1)),
        ("32d", dt(2022, 2, 1)),
        ("1y", dt(2022, 12, 31)),
        ("0.5y", dt(2022, 6, 30)),
    ],
)
def test_add_tenor(tenor, expected) -> None:
    result = add_tenor(dt(2021, 12, 31), tenor, "NONE", NoInput(0))
    assert result == expected


@pytest.mark.parametrize(
    ("tenor", "expected", "roll"),
    [
        ("-1M", dt(2022, 1, 31), "eom"),
        ("-1M", dt(2022, 1, 28), NoInput(0)),
        ("-2m", dt(2021, 12, 31), 31),
        ("-2m", dt(2021, 12, 28), NoInput(0)),
        ("-1Y", dt(2021, 2, 28), NoInput(0)),
        ("-1d", dt(2022, 2, 27), NoInput(0)),
        ("-2y", dt(2020, 2, 29), "eom"),
        ("-2y", dt(2020, 2, 28), NoInput(0)),
    ],
)
def test_add_negative_tenor(tenor, expected, roll) -> None:
    result = add_tenor(dt(2022, 2, 28), tenor, "NONE", NoInput(0), roll)
    assert result == expected


@pytest.mark.parametrize(
    ("date", "tenor", "mod", "roll", "cal", "expected"),
    [
        (dt(1990, 9, 28), "-6m", "NONE", 31, NoInput(0), dt(1990, 3, 31)),
        (dt(1990, 9, 28), "-6m", "NONE", 29, NoInput(0), dt(1990, 3, 29)),
        (dt(1990, 5, 29), "3m", "NONE", NoInput(0), NoInput(0), dt(1990, 8, 29)),
        (dt(1990, 5, 29), "3m", "NONE", 31, NoInput(0), dt(1990, 8, 31)),
        (dt(1990, 3, 31), "6m", "MF", 31, "nyc", dt(1990, 9, 28)),
        (dt(2023, 4, 21), "-3m", "P", 23, "bus", dt(2023, 1, 23)),
        (dt(2023, 6, 23), "-3m", "P", 25, "bus", dt(2023, 3, 24)),
    ],
)
def test_add_tenor_special_cases(date, tenor, mod, roll, cal, expected) -> None:
    end = add_tenor(date, tenor, mod, cal, roll)
    assert end == expected


@pytest.mark.parametrize(
    ("date", "modifier", "expected"),
    [
        (dt(2022, 1, 3), "NONE", dt(2022, 1, 3)),
        (dt(2022, 1, 3), "F", dt(2022, 1, 4)),
        (dt(2022, 1, 3), "MF", dt(2022, 1, 4)),
        (dt(2022, 1, 3), "P", dt(2021, 12, 31)),
        (dt(2022, 1, 3), "MP", dt(2022, 1, 4)),
        (dt(2022, 7, 30), "NONE", dt(2022, 7, 30)),
        (dt(2022, 7, 30), "f", dt(2022, 8, 1)),
        (dt(2022, 7, 30), "mf", dt(2022, 7, 29)),
        (dt(2022, 7, 30), "p", dt(2022, 7, 29)),
        (dt(2022, 7, 30), "mp", dt(2022, 7, 29)),
    ],
)
def test_adjust_date(date, modifier, cal_, expected) -> None:
    result = _adjust_date(date, modifier, cal_)
    assert result == expected


def test_adjust_date_cal() -> None:
    result = _adjust_date(dt(2022, 10, 1), "F", NoInput(0))
    assert result == dt(2022, 10, 1)


def test_adjust_date_raises() -> None:
    with pytest.raises(KeyError):
        _adjust_date(dt(2000, 1, 1), "BAD_STRING", NoInput(0))


@pytest.mark.parametrize(
    ("modifier", "expected"),
    [
        ("None", dt(2022, 1, 3)),
        ("F", dt(2022, 1, 4)),
        ("MF", dt(2022, 1, 4)),
        ("P", dt(2021, 12, 31)),
        ("MP", dt(2022, 1, 4)),
    ],
)
def test_modifiers_som(cal_, modifier, expected) -> None:
    result = add_tenor(dt(2021, 12, 3), "1M", modifier, cal_)
    assert result == expected


@pytest.mark.parametrize(
    ("modifier", "expected"),
    [
        ("None", dt(2021, 2, 28)),
        ("F", dt(2021, 3, 1)),
        ("MF", dt(2021, 2, 26)),
        ("P", dt(2021, 2, 26)),
        ("MP", dt(2021, 2, 26)),
    ],
)
def test_modifiers_eom(cal_, modifier, expected) -> None:
    result = add_tenor(dt(2020, 12, 31), "2M", modifier, cal_)
    assert result == expected


@pytest.mark.parametrize(
    ("start", "end", "conv", "expected"),
    [
        (dt(2022, 1, 1), dt(2022, 4, 1), "ACT365F", 0.2465753424657534),
        (dt(2021, 1, 1), dt(2022, 4, 1), "ACT365F+", 1.2465753424657535),
        (dt(2022, 1, 1), dt(2022, 4, 1), "ACT365F+", 0.2465753424657534),
        (dt(2020, 6, 1), dt(2022, 4, 1), "ACT365F+", 1.832876712328767),
        (dt(2020, 1, 1), dt(2052, 1, 2), "ACT365F", 32.02465753424657),
        (dt(2020, 1, 1), dt(2052, 1, 2), "ACT365F+", 32.0027397260274),
        (dt(2022, 1, 1), dt(2022, 4, 1), "1", 1.0),
        (dt(2022, 1, 1), dt(2022, 4, 1), "ACT360", 0.2465753424657534 * 365 / 360),
        (dt(2022, 1, 1), dt(2022, 4, 1), "30360", 0.250),
        (dt(2022, 1, 1), dt(2022, 4, 1), "30E360", 0.250),
        (dt(2022, 1, 1), dt(2022, 4, 1), "ACTACT", 0.2465753424657534),
        (dt(2022, 1, 1), dt(2022, 1, 1), "ACTACT", 0.0),
        (dt(2022, 1, 1), dt(2023, 1, 31), "1+", 1.0),
        (dt(2022, 1, 1), dt(2024, 2, 28), "1+", 2 + 1 / 12),
        (dt(2022, 1, 1), dt(2022, 4, 1), "BUS252", 0.35714285714285715),
        (dt(2022, 1, 1), dt(2022, 4, 1), "30U360", 0.25),
        (dt(2022, 1, 1), dt(2022, 4, 1), "ACT365_25", 0.2464065708418891),
        (dt(2022, 1, 1), dt(2022, 4, 1), "ACT364", 0.24725274725274726),
    ],
)
def test_dcf(start, end, conv, expected) -> None:
    result = dcf(start, end, conv, calendar="all", frequency="Q")
    assert abs(result - expected) < 1e-14


@pytest.mark.parametrize(
    ("start", "end", "conv", "expected", "freq", "term", "stub"),
    [
        (dt(2022, 6, 30), dt(2022, 7, 31), "30360", 1 / 12, NoInput(0), None, None),
        (dt(2022, 6, 30), dt(2022, 7, 31), "30E360", 1 / 12, NoInput(0), None, None),
        (dt(2022, 6, 30), dt(2022, 7, 31), "30E360ISDA", 1 / 12, "A", dt(2022, 7, 31), None),
        (dt(2022, 6, 29), dt(2022, 7, 31), "30360", 1 / 12 + 2 / 360, NoInput(0), None, None),
        (dt(2022, 6, 29), dt(2022, 7, 31), "30E360", 1 / 12 + 1 / 360, NoInput(0), None, None),
        (dt(2022, 2, 28), dt(2022, 3, 31), "30E360", 1 / 12 + 2 / 360, NoInput(0), None, None),
        (dt(2022, 2, 28), dt(2022, 3, 31), "30E360ISDA", 1 / 12, "A", dt(2022, 3, 3), None),
        (
            dt(1999, 2, 1),
            dt(1999, 7, 1),
            "ACTACTICMA",
            150 / 365,
            "A",
            dt(2000, 7, 1),
            True,
        ),  # short first
        (
            dt(2002, 8, 15),
            dt(2003, 7, 15),
            "ACTACTICMA",
            0.5 + 153 / 368,
            "S",
            dt(2004, 1, 15),
            True,
        ),  # long first
        (
            dt(2000, 1, 30),
            dt(2000, 6, 30),
            "ACTACTICMA",
            152 / 364,
            "S",
            dt(2000, 6, 30),
            True,
        ),  # short back
        (
            dt(1999, 11, 30),
            dt(2000, 4, 30),
            "ACTACTICMA",
            0.25 + 61 / 368,
            Frequency.Months(3, RollDay.Day(31)),
            dt(2000, 4, 30),
            True,
        ),
        (
            dt(1999, 11, 30),
            dt(2000, 4, 30),
            "ACTACTICMA",
            0.25 + 61 / 364,
            Frequency.Months(3, RollDay.Day(30)),
            dt(2000, 4, 30),
            True,
        ),
        # long back : with and without month end roll here
        (
            dt(1999, 11, 15),
            dt(2000, 4, 15),
            "ACTACTICMA",
            0.25 + 60 / 360,
            "Q",
            dt(2000, 4, 15),
            True,
        ),  # long back
        (dt(2002, 8, 31), dt(2002, 11, 30), "ACTACTICMA", 0.25, "Q", dt(2004, 11, 30), False),
        (
            dt(1999, 2, 1),
            dt(1999, 7, 1),
            "ACTACTICMA_STUB365F",
            150 / 365,
            "A",
            dt(2000, 7, 1),
            True,
        ),  # short first
        (
            dt(2002, 8, 15),
            dt(2003, 7, 15),
            "ACTACTICMA_STUB365F",
            0.5 + 153 / 365,
            "S",
            dt(2004, 1, 15),
            True,
        ),  # long first
        (
            dt(2000, 1, 30),
            dt(2000, 6, 30),
            "ACTACTICMA_STUB365F",
            152 / 365,
            "S",
            dt(2000, 6, 30),
            True,
        ),  # short back
        (
            dt(1999, 11, 15),
            dt(2000, 4, 15),
            "ACTACTICMA_STUB365F",
            0.25 + 60 / 365,
            "Q",
            dt(2000, 4, 15),
            True,
        ),  # long back
        (
            dt(2002, 8, 31),
            dt(2002, 11, 30),
            "ACTACTICMA_STUB365F",
            0.25,
            "Q",
            dt(2004, 11, 30),
            False,
        ),
    ],
)
def test_dcf_special(start, end, conv, expected, freq, term, stub) -> None:
    # The 4 ActICMA tests match short/long first/final stubs in 1998-ISDA-memo-EMU pdf
    result = dcf(start, end, conv, term, freq, stub)
    assert abs(result - expected) < 1e-12


@pytest.mark.parametrize(
    ("conv", "freq", "term", "stub"),
    [
        ("ACTACTICMA", NoInput(0), NoInput(0), NoInput(0)),
        ("ACTACTICMA", "Q", NoInput(0), NoInput(0)),
        ("BadConv", NoInput(0), NoInput(0), NoInput(0)),
    ],
)
def test_dcf_raises(conv, freq, term, stub) -> None:
    with pytest.raises(ValueError):
        _ = dcf(
            dt(2022, 1, 1),
            dt(2022, 4, 1),
            conv,
            term,
            freq,
            stub=stub,
        )


def test_dcf_30e360_isda_raises():
    # needs a termination if end februrary
    with pytest.raises(ValueError, match="`termination` must be provided for '30e360ISDA' conv"):
        _ = dcf(
            dt(2022, 2, 28),
            dt(2023, 2, 28),
            "30e360isda",
            NoInput(0),
        )


def test_dcf_30u360_raises():
    # needs a termination if end februrary
    with pytest.raises(ValueError, match="`frequency` must be provided or has no `roll`. A roll-d"):
        _ = dcf(
            dt(2022, 2, 28),
            dt(2023, 2, 28),
            "30u360",
        )


def test_dcf_actacticma_raises():
    with pytest.raises(ValueError, match="Stub periods under ActActICMA require `termination`, `a"):
        _ = dcf(
            dt(2022, 2, 28),
            dt(2023, 2, 28),
            "actacticma",
            NoInput(0),
            "Q",
            True,
            Cal.from_name("tgt"),
            NoInput(0),
        )


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (dt(2000, 1, 1), dt(2000, 1, 4), 1.0 / 252.0),
        (dt(2000, 1, 2), dt(2000, 1, 4), 1.0 / 252.0),
        (dt(2000, 1, 2), dt(2000, 1, 5), 2.0 / 252.0),
        (dt(2000, 1, 1), dt(2000, 1, 5), 2.0 / 252.0),
        (dt(2000, 1, 3), dt(2000, 1, 5), 1.0 / 252.0),
        (dt(2000, 1, 3), dt(2000, 1, 4), 0.0 / 252.0),
        (dt(2000, 1, 4), dt(2000, 1, 5), 1.0 / 252.0),
        (dt(2000, 1, 5), dt(2000, 1, 6), 0.0 / 252.0),
        (dt(2000, 1, 5), dt(2000, 1, 5), 0.0 / 252.0),
    ],
)
def test_bus252(start, end, expected) -> None:
    cal = Cal(
        [
            dt(2000, 1, 1),
            dt(2000, 1, 3),
            dt(2000, 1, 5),
            dt(2000, 1, 6),
        ],
        [],
    )
    assert dcf(start, end, "BUS252", calendar=cal) == expected


@pytest.mark.parametrize(
    ("start", "end", "roll", "expected"),
    [
        (dt(2024, 2, 29), dt(2025, 2, 28), "eom", 1.00),
        (dt(2024, 2, 29), dt(2025, 2, 28), 29, 0.99722222222222),
        (dt(2024, 2, 28), dt(2025, 2, 28), "eom", 1.0),
        (dt(2024, 2, 28), dt(2025, 2, 28), 28, 1.0),
        (dt(2024, 2, 29), dt(2025, 2, 27), "eom", 0.99166666666666),
        (dt(2024, 2, 29), dt(2025, 2, 27), 27, 0.99444444444444),
        (dt(2024, 2, 28), dt(2025, 2, 27), "eom", 0.99722222222222),
        (dt(2024, 2, 28), dt(2025, 2, 27), 27, 0.99722222222222),
        (dt(2024, 9, 30), dt(2024, 12, 31), None, 0.25),
        (dt(2024, 3, 31), dt(2024, 6, 30), None, 0.25),
        (dt(2024, 3, 31), dt(2024, 12, 31), None, 0.75),
        (dt(2024, 12, 1), dt(2024, 12, 31), None, 30 / 360),
        (dt(2024, 11, 30), dt(2024, 12, 31), None, 30 / 360),
        (dt(2024, 2, 29), dt(2024, 3, 31), 29, 32 / 360),
        (dt(2024, 2, 29), dt(2024, 3, 31), "eom", 30 / 360),
        (dt(2024, 2, 28), dt(2024, 3, 31), "eom", 33 / 360),
        (dt(2025, 2, 28), dt(2025, 3, 31), "eom", 30 / 360),
    ],
)
def test_30u360(start, end, roll, expected):
    freq = _get_frequency("M", roll, "all")
    result = dcf(start, end, "30U360", frequency=freq)
    assert abs(result - expected) < 1e-10


@pytest.mark.parametrize(
    ("d1", "d2", "exp"),
    [
        (dt(2009, 3, 1), dt(2012, 1, 15), (2, 10)),
        (dt(2008, 12, 1), dt(2013, 10, 31), (4, 10)),
        (dt(2008, 12, 1), dt(2018, 11, 15), (9, 11)),
        (dt(2008, 12, 1), dt(2038, 5, 15), (29, 5)),
    ],
)
def test_get_years_and_months(d1, d2, exp) -> None:
    result = _get_years_and_months(d1, d2)
    assert result == exp


@pytest.mark.parametrize(
    ("s", "e", "t", "exp"),
    [
        (dt(2024, 2, 29), dt(2024, 5, 29), dt(2024, 5, 29), 0.24657534),
        (dt(2021, 2, 28), dt(2024, 5, 29), dt(2024, 5, 29), 3.24863387),
        (dt(2021, 2, 28), dt(2024, 5, 29), dt(2026, 5, 28), 3.24657534),
    ],
)
def test_act_act_icma_z_freq(s, e, t, exp) -> None:
    with pytest.warns(UserWarning, match="`frequency` cannot be 'Zero' variant in combination wit"):
        result = dcf(
            start=s,
            end=e,
            convention="ActActICMA",
            termination=t,
            frequency=Frequency.Zero(),  # Z Frequency
            stub=True,
            calendar=Cal([], []),
            adjuster=Adjuster.Actual(),
        )
    assert abs(result - exp) < 1e-6


def test_calendar_aligns_with_fixings_tyo() -> None:
    # using this test in a regular way, and with "-W error" for error on warn ensures that:
    #  - Curve cal is a business  day and fixings cal has no fixing: is a warn
    #  - Curve cal is not a business day and fixings cal has a fixing: errors
    curve = Curve(
        {dt(2015, 6, 10): 1.0, dt(2024, 6, 3): 1.0},
        calendar="tyo",
    )
    fixings_ = fixings["jpy_rfr"][1]
    irs = IRS(dt(2015, 6, 10), dt(2024, 6, 3), "A", leg2_rate_fixings=fixings_, calendar="tyo")
    irs.rate(curves=curve)


def test_calendar_aligns_with_fixings_syd() -> None:
    # using this test in a regular way, and with "-W error" for error on warn ensures that:
    #  - Curve cal is a business  day and fixings cal has no fixing: is a warn
    #  - Curve cal is not a business day and fixings cal has a fixing: errors
    curve = Curve(
        {dt(2015, 6, 10): 1.0, dt(2024, 6, 3): 1.0},
        calendar="syd",
    )
    fixings_ = fixings["aud_rfr"][1]
    irs = IRS(dt(2015, 6, 10), dt(2024, 6, 3), "A", leg2_rate_fixings=fixings_, calendar="syd")
    irs.rate(curves=curve)


def test_book_example() -> None:
    res = add_tenor(dt(2001, 9, 28), "-6M", modifier="MF", calendar="ldn")
    assert res == dt(2001, 3, 28, 0, 0)
    res = add_tenor(dt(2001, 9, 28), "-6M", modifier="MF", calendar="ldn", roll=31)
    assert res == dt(2001, 3, 30, 0, 0)
    res = add_tenor(dt(2001, 9, 28), "-6M", modifier="MF", calendar="ldn", roll=29)
    assert res == dt(2001, 3, 29, 0, 0)


def test_book_example2() -> None:
    cal = get_calendar("tgt|nyc")
    cal2 = get_calendar("tgt,nyc")
    # 11th Nov 09 is a US holiday: test that the holiday is ignored in the settlement cal
    result = cal.add_bus_days(dt(2009, 11, 10), 2, True)
    result2 = cal2.add_bus_days(dt(2009, 11, 10), 2, True)
    assert result == dt(2009, 11, 12)
    assert result2 == dt(2009, 11, 13)

    # test that the US settlement is honoured
    result = cal.add_bus_days(dt(2009, 11, 9), 2, True)
    result2 = cal.add_bus_days(dt(2009, 11, 9), 2, False)
    assert result == dt(2009, 11, 12)
    assert result2 == dt(2009, 11, 11)


def test_pipe_vectors() -> None:
    get_calendar("tgt,stk|nyc,osl")


def test_pipe_raises() -> None:
    with pytest.raises(ValueError, match="Cannot use more than one pipe"):
        get_calendar("tgt|nyc|stk")


def test_add_and_get_custom_calendar() -> None:
    cal = Cal([dt(2023, 1, 2)], [5, 6])
    defaults.calendars["custom"] = cal
    result = get_calendar("custom")
    assert result == cal


@pytest.mark.parametrize(
    ("evald", "delivery", "expiry", "expected_expiry"),
    [
        (dt(2024, 5, 2), 2, "2m", dt(2024, 7, 4)),
        (dt(2024, 4, 30), 2, "2m", dt(2024, 7, 1)),
        (dt(2024, 5, 31), 2, "1m", dt(2024, 7, 3)),
        (dt(2024, 5, 31), 2, "2w", dt(2024, 6, 14)),
    ],
)
def test_expiries_delivery(evald, delivery, expiry, expected_expiry) -> None:
    result_expiry, _, _ = _get_fx_expiry_and_delivery_and_payment(
        evald, expiry, delivery, "tgt|fed", "mf", False, 0
    )
    assert result_expiry == expected_expiry


def test_expiries_delivery_raises() -> None:
    with pytest.raises(ValueError, match="Cannot determine FXOption expiry and delivery"):
        _get_fx_expiry_and_delivery_and_payment(
            dt(2000, 1, 1),
            "3m",
            dt(2000, 3, 2),
            "tgt|fed",
            "mf",
            False,
            0,
        )


@pytest.mark.parametrize(
    ("val", "exp"),
    [
        ("Z24", dt(2024, 12, 18)),
        ("X89", dt(2089, 11, 16)),
    ],
)
def test_get_imm_api(val, exp):
    result = get_imm(month=1, year=1, code=val)
    assert result == exp


def test_get_imm_api_no_code():
    result = get_imm(month=11, year=2089)
    assert result == dt(2089, 11, 16)


@pytest.mark.parametrize("tenor", ["1B", "1b", "3D", "3d", "2W", "2w"])
def test_is_day_type_tenor(tenor):
    assert _is_day_type_tenor(tenor)


@pytest.mark.parametrize("tenor", ["1M", "1m", "4Y", "4y"])
def test_is_not_day_type_tenor(tenor):
    assert not _is_day_type_tenor(tenor)


def test_get_calendar_from_defaults() -> None:
    defaults.calendars["custom"] = "my_object"
    assert get_calendar("custom") == "my_object"
    defaults.calendars.pop("custom")


@pytest.mark.parametrize(
    ("start", "method", "expected"),
    [
        (dt(2025, 1, 1), "wed3_hmuz", dt(2025, 3, 19)),
        (dt(2025, 1, 1), "wed3", dt(2025, 1, 15)),
        (dt(2025, 1, 1), "day20_hmuz", dt(2025, 3, 20)),
        (dt(2025, 1, 1), "day20_HU", dt(2025, 3, 20)),
        (dt(2025, 1, 1), "day20_MZ", dt(2025, 6, 20)),
        (dt(2025, 1, 15), "wed3", dt(2025, 2, 19)),
        (dt(2025, 3, 19), "wed3_hmuz", dt(2025, 6, 18)),
        (dt(2025, 3, 20), "day20_hmuz", dt(2025, 6, 20)),
        (dt(2025, 3, 20), "day20_HU", dt(2025, 9, 20)),
        (dt(2025, 3, 20), "day20_MZ", dt(2025, 6, 20)),
        (dt(2025, 9, 20), "day20_HU", dt(2026, 3, 20)),
        (dt(2025, 12, 1), "wed3_hmuz", dt(2025, 12, 17)),
        (dt(2025, 12, 1), "wed3", dt(2025, 12, 17)),
        (dt(2025, 12, 1), "day20_hmuz", dt(2025, 12, 20)),
        (dt(2025, 12, 1), "day20_HU", dt(2026, 3, 20)),
        (dt(2025, 12, 1), "day20_MZ", dt(2025, 12, 20)),
        (dt(2025, 12, 17), "wed3_hmuz", dt(2026, 3, 18)),
        (dt(2025, 12, 17), "wed3", dt(2026, 1, 21)),
        (dt(2025, 12, 20), "day20_hmuz", dt(2026, 3, 20)),
        (dt(2025, 12, 20), "day20_HU", dt(2026, 3, 20)),
        (dt(2025, 12, 20), "day20_MZ", dt(2026, 6, 20)),
    ],
)
def test_next_imm(start, method, expected):
    result = next_imm(start, method)
    assert result == expected


def test_next_imm_depr():
    with pytest.warns(DeprecationWarning):
        next_imm(dt(2000, 1, 1), "imm")


def test_get_imm_depr():
    with pytest.warns(DeprecationWarning):
        get_imm(3, 2000, definition="imm")


def test_fed_nyc_good_friday():
    assert not get_calendar("nyc").is_bus_day(dt(2024, 3, 29))
    assert get_calendar("fed").is_bus_day(dt(2024, 3, 29))


def test_fed_sunday_to_monday():
    fed = get_calendar("fed")
    assert fed.is_bus_day(dt(2021, 12, 24))
    assert not fed.is_bus_day(dt(2022, 12, 26))


def test_syd_nsw_holidays():
    cal = get_calendar("nsw")
    assert not cal.is_bus_day(dt(1970, 8, 3))
    assert not cal.is_bus_day(dt(1970, 10, 5))


def test_wlg_changes():
    cal = get_calendar("wlg")
    assert not cal.is_bus_day(dt(2022, 9, 26))
    assert not cal.is_bus_day(dt(2025, 1, 20))
    assert not cal.is_bus_day(dt(2025, 1, 27))


def test_busdayslag_reverse():
    # test that reverse operates over settleable days also
    a = Adjuster.BusDaysLagSettle(2)
    cal = Cal([dt(2026, 1, 1)], [5, 6])
    union = UnionCal([Cal([], [])], [cal])
    assert a.adjust(dt(2025, 12, 30), union) == dt(2026, 1, 2)
    assert a.adjust(dt(2025, 12, 31), union) == dt(2026, 1, 2)
    assert a.reverse(dt(2026, 1, 2), union) == [dt(2025, 12, 31), dt(2025, 12, 30)]


def test_mex_loads():
    cal = get_calendar("mex")
    assert not cal.is_bus_day(dt(2026, 3, 16))
    assert cal.is_bus_day(dt(2026, 3, 17))


def test_replace_whitespace():
    cal1 = get_calendar("nyc, tgt")
    cal2 = get_calendar("nyc,tgt")
    assert cal1 == cal2
