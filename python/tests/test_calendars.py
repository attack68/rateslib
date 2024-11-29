from datetime import datetime as dt

import pytest
from rateslib import defaults
from rateslib.calendars import (
    Cal,
    _adjust_date,
    _get_eom,
    _get_fx_expiry_and_delivery,
    _get_imm,
    _get_years_and_months,
    _is_day_type_tenor,
    _is_eom,
    _is_imm,
    _is_som,
    add_tenor,
    create_calendar,
    dcf,
    get_calendar,
    get_imm,
)
from rateslib.calendars.dcfs import _dcf_actacticma
from rateslib.curves import Curve
from rateslib.default import NoInput
from rateslib.instruments import IRS


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
    cal_no_hols = create_calendar([], [])
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
    ("month", "year", "expected"),
    [
        (2, 2022, dt(2022, 2, 28)),
        (2, 2024, dt(2024, 2, 29)),
        (8, 2022, dt(2022, 8, 31)),
    ],
)
def test_get_eom(month, year, expected) -> None:
    result = _get_eom(month, year)
    assert result == expected


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
    with pytest.raises(ValueError):
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
    ("date", "expected"),
    [
        (dt(2022, 3, 16), True),
        (dt(2022, 6, 15), True),
        (dt(2022, 9, 25), False),
    ],
)
def test_is_imm(date, expected) -> None:
    result = _is_imm(date)
    assert result is expected


def test_is_imm_hmuz() -> None:
    result = _is_imm(dt(2022, 8, 17), hmuz=True)  # imm in Aug
    assert not result
    result = _is_imm(dt(2022, 8, 17), hmuz=False)  # imm in Aug
    assert result


@pytest.mark.parametrize(
    ("month", "year", "expected"),
    [
        (3, 2022, dt(2022, 3, 16)),
        (6, 2022, dt(2022, 6, 15)),
        (9, 2022, dt(2022, 9, 21)),
        (12, 2022, dt(2022, 12, 21)),
    ],
)
def test_get_imm(month, year, expected) -> None:
    result = _get_imm(month, year)
    assert result == expected


def test_get_imm_namespace():
    from rateslib import get_imm as f

    f(code="h24")


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        (dt(2022, 1, 1), (True, False)),
        (dt(2022, 2, 28), (False, True)),
        (dt(2022, 2, 27), (False, False)),
    ],
)
def test_is_eom_som(date, expected) -> None:
    result = (_is_som(date), _is_eom(date))
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
    ],
)
def test_dcf(start, end, conv, expected) -> None:
    result = dcf(start, end, conv)
    assert abs(result - expected) < 1e-14


@pytest.mark.parametrize(
    ("start", "end", "conv", "expected", "freq_m", "term", "stub"),
    [
        (dt(2022, 6, 30), dt(2022, 7, 31), "30360", 1 / 12, NoInput(0), None, None),
        (dt(2022, 6, 30), dt(2022, 7, 31), "30E360", 1 / 12, NoInput(0), None, None),
        (dt(2022, 6, 30), dt(2022, 7, 31), "30E360ISDA", 1 / 12, 12, dt(2022, 7, 31), None),
        (dt(2022, 6, 29), dt(2022, 7, 31), "30360", 1 / 12 + 2 / 360, NoInput(0), None, None),
        (dt(2022, 6, 29), dt(2022, 7, 31), "30E360", 1 / 12 + 1 / 360, NoInput(0), None, None),
        (dt(2022, 2, 28), dt(2022, 3, 31), "30E360", 1 / 12 + 2 / 360, NoInput(0), None, None),
        (dt(2022, 2, 28), dt(2022, 3, 31), "30E360ISDA", 1 / 12, 12, dt(2022, 3, 3), None),
        (
            dt(1999, 2, 1),
            dt(1999, 7, 1),
            "ACTACTICMA",
            150 / 365,
            12,
            dt(2000, 7, 1),
            True,
        ),  # short first
        (
            dt(2002, 8, 15),
            dt(2003, 7, 15),
            "ACTACTICMA",
            0.5 + 153 / 368,
            6,
            dt(2004, 1, 15),
            True,
        ),  # long first
        (
            dt(2000, 1, 30),
            dt(2000, 6, 30),
            "ACTACTICMA",
            152 / 364,
            6,
            dt(2000, 6, 30),
            True,
        ),  # short back
        # (dt(1999,11,30), dt(2000,4,30), "ACTACTICMA", 0.25 + 61 / 368, 3, dt(2000, 4, 30), True),
        # long back : SKIP the _add_tenor does not account for month end roll here
        (
            dt(1999, 11, 15),
            dt(2000, 4, 15),
            "ACTACTICMA",
            0.25 + 60 / 360,
            3,
            dt(2000, 4, 15),
            True,
        ),  # long back
        (dt(2002, 8, 31), dt(2002, 11, 30), "ACTACTICMA", 0.25, 3, dt(2004, 11, 30), False),
        (
            dt(1999, 2, 1),
            dt(1999, 7, 1),
            "ACTACTICMA_STUB365F",
            150 / 365,
            12,
            dt(2000, 7, 1),
            True,
        ),  # short first
        (
            dt(2002, 8, 15),
            dt(2003, 7, 15),
            "ACTACTICMA_STUB365F",
            0.5 + 153 / 365,
            6,
            dt(2004, 1, 15),
            True,
        ),  # long first
        (
            dt(2000, 1, 30),
            dt(2000, 6, 30),
            "ACTACTICMA_STUB365F",
            152 / 365,
            6,
            dt(2000, 6, 30),
            True,
        ),  # short back
        # (dt(1999,11,30), dt(2000,4,30), "ACTACTICMA", 0.25 + 61 / 368, 3, dt(2000, 4, 30), True),
        # long back : SKIP the _add_tenor does not account for month end roll here
        (
            dt(1999, 11, 15),
            dt(2000, 4, 15),
            "ACTACTICMA_STUB365F",
            0.25 + 60 / 365,
            3,
            dt(2000, 4, 15),
            True,
        ),  # long back
        (
            dt(2002, 8, 31),
            dt(2002, 11, 30),
            "ACTACTICMA_STUB365F",
            0.25,
            3,
            dt(2004, 11, 30),
            False,
        ),
    ],
)
def test_dcf_special(start, end, conv, expected, freq_m, term, stub) -> None:
    # The 4 ActICMA tests match short/long first/final stubs in 1998-ISDA-memo-EMU pdf
    result = dcf(start, end, conv, term, freq_m, stub)
    assert abs(result - expected) < 1e-12


@pytest.mark.parametrize(
    ("conv", "freq_m", "term", "stub"),
    [
        ("ACTACTICMA", NoInput(0), NoInput(0), NoInput(0)),
        ("ACTACTICMA", 3, NoInput(0), NoInput(0)),
        ("30E360ISDA", 3, NoInput(0), NoInput(0)),
        ("ACTACTICMA", 3, dt(2022, 4, 1), NoInput(0)),
        ("BadConv", NoInput(0), NoInput(0), NoInput(0)),
    ],
)
def test_dcf_raises(conv, freq_m, term, stub) -> None:
    with pytest.raises(ValueError):
        _ = dcf(dt(2022, 1, 1), dt(2022, 4, 1), conv, term, freq_m, stub)


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
    with pytest.warns(UserWarning, match="Using `convention` 'ActActICMA' with a Period having"):
        result = _dcf_actacticma(
            start=s,
            end=e,
            termination=t,
            frequency_months=1e8,  # Z Frequency
            stub=False,
            roll=NoInput(0),
            calendar=NoInput(0),
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
    fixings = defaults.fixings["jpy_rfr"]
    irs = IRS(dt(2015, 6, 10), dt(2024, 6, 3), "A", leg2_fixings=fixings, calendar="tyo")
    irs.rate(curve)


def test_calendar_aligns_with_fixings_syd() -> None:
    # using this test in a regular way, and with "-W error" for error on warn ensures that:
    #  - Curve cal is a business  day and fixings cal has no fixing: is a warn
    #  - Curve cal is not a business day and fixings cal has a fixing: errors
    curve = Curve(
        {dt(2015, 6, 10): 1.0, dt(2024, 6, 3): 1.0},
        calendar="syd",
    )
    fixings = defaults.fixings["aud_rfr"]
    irs = IRS(dt(2015, 6, 10), dt(2024, 6, 3), "A", leg2_fixings=fixings, calendar="syd")
    irs.rate(curve)


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
    ("eval", "delivery", "expiry", "expected_expiry"),
    [
        (dt(2024, 5, 2), 2, "2m", dt(2024, 7, 4)),
        (dt(2024, 4, 30), 2, "2m", dt(2024, 7, 1)),
        (dt(2024, 5, 31), 2, "1m", dt(2024, 7, 3)),
        (dt(2024, 5, 31), 2, "2w", dt(2024, 6, 14)),
    ],
)
def test_expiries_delivery(eval, delivery, expiry, expected_expiry) -> None:
    result_expiry, _ = _get_fx_expiry_and_delivery(
        eval,
        expiry,
        delivery,
        "tgt|fed",
        "mf",
        False,
    )
    assert result_expiry == expected_expiry


def test_expiries_delivery_raises() -> None:
    with pytest.raises(ValueError, match="Cannot determine FXOption expiry and delivery"):
        _get_fx_expiry_and_delivery(
            dt(2000, 1, 1),
            "3m",
            dt(2000, 3, 2),
            "tgt|fed",
            "mf",
            False,
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
