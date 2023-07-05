import pytest
from pandas.tseries.holiday import Holiday
from datetime import datetime as dt

import context
from rateslib.calendars import (
    create_calendar,
    _is_holiday,
    dcf,
    get_calendar,
    add_tenor,
    _get_eom,
    _adjust_date,
    _is_eom,
    _is_imm,
    _is_som,
    _get_imm,
)


@pytest.fixture
def cal_():
    rules = [Holiday("New Year", month=1, day=3)]
    return create_calendar(rules=rules)


@pytest.mark.parametrize("date, expected", [
    (dt(2022, 1, 1), True),  # sat
    (dt(2022, 1, 2), True),  # sun
    (dt(2022, 1, 3), True),  # mon new year hol
    (dt(2022, 1, 4), False),  # tues
    (dt(2022, 1, 5), False),  # wed
])
def test_is_holiday(date, expected, cal_):
    result = _is_holiday(date, cal_)
    assert result == expected


def test_is_holiday_raises():
    with pytest.raises(ValueError):
        _ = _is_holiday(dt(2022, 1, 1), None)


@pytest.mark.parametrize("date", [
    dt(2021, 12, 29),
    dt(2021, 12, 30),
    dt(2021, 12, 31),
    dt(2021, 1, 1),
    dt(2021, 1, 2),
    dt(2021, 1, 3),
    dt(2021, 1, 4),
    dt(2021, 1, 5),
])
def test_cal_no_hols(date):
    cal_no_hols = create_calendar([], "Mon Tue Wed Thu Fri Sat Sun")
    assert not _is_holiday(date, cal_no_hols)


def test_named_cal():
    ldn_cal = get_calendar("ldn")
    assert _is_holiday(dt(2022, 1, 1), ldn_cal)
    assert not _is_holiday(dt(2022, 1, 5), ldn_cal)


def test_multiple_named_cal():
    ldn_cal = get_calendar("ldn")
    stk_cal = get_calendar("stk")

    assert _is_holiday(dt(2023, 1, 2), ldn_cal)
    assert not _is_holiday(dt(2023, 1, 2), stk_cal)

    assert not _is_holiday(dt(2023, 1, 6), ldn_cal)
    assert _is_holiday(dt(2023, 1, 6), stk_cal)

    merged_cal = get_calendar("LDN,stk")
    assert _is_holiday(dt(2023, 1, 2), merged_cal)
    assert _is_holiday(dt(2023, 1, 6), merged_cal)


def test_add_tenor_raises():
    # this raise is superfluous by the design principles of private methods
    with pytest.raises(ValueError):
        add_tenor(dt(2022, 1, 1), "1X", "mf", None)


@pytest.mark.parametrize("tenor, expected", [
    ("1M", dt(2022, 1, 31)),
    ("2m", dt(2022, 2, 28)),
    ("6M", dt(2022, 6, 30)),
    ("1d", dt(2022, 1, 1)),
    ("32d", dt(2022, 2, 1)),
    ("1y", dt(2022, 12, 31)),
    ("0.5y", dt(2022, 6, 30)),
])
def test_add_tenor(tenor, expected):
    result = add_tenor(dt(2021, 12, 31), tenor, None, None)
    assert result == expected


@pytest.mark.parametrize("tenor, expected", [
    ("-1M", dt(2022, 1, 28)),
    ("-2m", dt(2021, 12, 28)),
    ("-1Y", dt(2021, 2, 28)),
    ("-1d", dt(2022, 2, 27)),
    ("-2y", dt(2020, 2, 28)),
])
def test_add_negative_tenor(tenor, expected):
    result = add_tenor(dt(2022, 2, 28), tenor, None, None)
    assert result == expected


@pytest.mark.parametrize("month, year, expected", [
    (2, 2022, dt(2022, 2, 28)),
    (2, 2024, dt(2024, 2, 29)),
    (8, 2022, dt(2022, 8, 31)),
])
def test_get_eom(month, year, expected):
    result = _get_eom(month, year)
    assert result == expected


@pytest.mark.parametrize("date, modifier, expected", [
    (dt(2022, 1, 3), None, dt(2022, 1, 3)),
    (dt(2022, 1, 3), "F", dt(2022, 1, 4)),
    (dt(2022, 1, 3), "MF", dt(2022, 1, 4)),
    (dt(2022, 1, 3), "P", dt(2021, 12, 31)),
    (dt(2022, 1, 3), "MP", dt(2022, 1, 4)),
    (dt(2022, 7, 30), None, dt(2022, 7, 30)),
    (dt(2022, 7, 30), "f", dt(2022, 8, 1)),
    (dt(2022, 7, 30), "mf", dt(2022, 7, 29)),
    (dt(2022, 7, 30), "p", dt(2022, 7, 29)),
    (dt(2022, 7, 30), "mp", dt(2022, 7, 29)),
])
def test_adjust_date(date, modifier, cal_, expected):
    result = _adjust_date(date, modifier, cal_)
    assert result == expected


def test_adjust_date_cal():
    result = _adjust_date(dt(2022, 10, 1), "F", None)
    assert result == dt(2022, 10, 1)


def test_adjust_date_raises():
    with pytest.raises(ValueError):
        _adjust_date(dt(2000, 1, 1), "BAD_STRING", None)


@pytest.mark.parametrize("modifier, expected", [
    (None, dt(2022, 1, 3)),
    ("F", dt(2022, 1, 4)),
    ("MF", dt(2022, 1, 4)),
    ("P", dt(2021, 12, 31)),
    ("MP", dt(2022, 1, 4)),
])
def test_modifiers_som(cal_, modifier, expected):
    result = add_tenor(dt(2021, 12, 3), "1M", modifier, cal_)
    assert result == expected


@pytest.mark.parametrize("modifier, expected", [
    (None, dt(2021, 2, 28)),
    ("F", dt(2021, 3, 1)),
    ("MF", dt(2021, 2, 26)),
    ("P", dt(2021, 2, 26)),
    ("MP", dt(2021, 2, 26)),
])
def test_modifiers_eom(cal_, modifier, expected):
    result = add_tenor(dt(2020, 12, 31), "2M", modifier, cal_)
    assert result == expected


@pytest.mark.parametrize("date, expected", [
    (dt(2022, 3, 16), True),
    (dt(2022, 6, 15), True),
    (dt(2022, 9, 25), False),
])
def test_is_imm(date, expected):
    result = _is_imm(date)
    assert result is expected


def test_is_imm_hmuz():
    result = _is_imm(dt(2022, 8, 17), hmuz=True)  # imm in Aug
    assert not result
    result = _is_imm(dt(2022, 8, 17), hmuz=False)  # imm in Aug
    assert result


@pytest.mark.parametrize("month, year, expected", [
    (3, 2022, dt(2022, 3, 16)),
    (6, 2022, dt(2022, 6, 15)),
    (9, 2022, dt(2022, 9, 21)),
    (12, 2022, dt(2022, 12, 21))
])
def test_get_imm(month, year, expected):
    result = _get_imm(month, year)
    assert result == expected


@pytest.mark.parametrize("date, expected", [
    (dt(2022, 1, 1), (True, False)),
    (dt(2022, 2, 28), (False, True)),
    (dt(2022, 2, 27), (False, False)),
])
def test_is_eom_som(date, expected):
    result = (_is_som(date), _is_eom(date))
    assert result == expected


@pytest.mark.parametrize("start, end, conv, expected", [
    (dt(2022, 1, 1), dt(2022, 4, 1), "ACT365F", 0.2465753424657534),
    (dt(2022, 1, 1), dt(2022, 4, 1), "1", 1.0),
    (dt(2022, 1, 1), dt(2022, 4, 1), "ACT360", 0.2465753424657534 * 365 / 360),
    (dt(2022, 1, 1), dt(2022, 4, 1), "30360", 0.250),
    (dt(2022, 1, 1), dt(2022, 4, 1), "30E360", 0.250),
    (dt(2022, 1, 1), dt(2022, 4, 1), "ACTACT", 0.2465753424657534),
    (dt(2022, 1, 1), dt(2022, 1, 1), "ACTACT", 0.0),
    (dt(2022, 1, 1), dt(2023, 1, 31), "1+", 1.0),
    (dt(2022, 1, 1), dt(2024, 2, 28), "1+", 2 + 1/12),
])
def test_dcf(start, end, conv, expected):
    result = dcf(start, end, conv)
    assert result == expected


@pytest.mark.parametrize("start, end, conv, expected, freq_m, term, stub", [
    (dt(2022, 6, 30), dt(2022, 7, 31), "30360", 1 / 12, None, None, None),
    (dt(2022, 6, 30), dt(2022, 7, 31), "30E360", 1 / 12, None, None, None),
    (dt(2022, 6, 30), dt(2022, 7, 31), "30E360ISDA", 1 / 12, 12, dt(2022, 7, 31), None),
    (dt(2022, 6, 29), dt(2022, 7, 31), "30360", 1 / 12 + 2 / 360, None, None, None),
    (dt(2022, 6, 29), dt(2022, 7, 31), "30E360", 1 / 12 + 1 / 360, None, None, None),
    (dt(2022, 2, 28), dt(2022, 3, 31), "30E360", 1 / 12 + 2 / 360, None, None, None),
    (dt(2022, 2, 28), dt(2022, 3, 31), "30E360ISDA", 1 / 12, 12, dt(2022, 3, 3), None),
    (dt(1999, 2, 1), dt(1999, 7, 1), "ACTACTICMA", 150 / 365, 12, dt(2000, 7, 1), True),  # short first
    (dt(2002, 8, 15), dt(2003, 7, 15), "ACTACTICMA", 0.5 + 153 / 368, 6, dt(2004, 1, 15), True),  # long first
    (dt(2000, 1, 30), dt(2000, 6, 30), "ACTACTICMA", 152 / 364, 6, dt(2000, 6, 30), True),  # short back
    # (dt(1999, 11, 30), dt(2000, 4, 30), "ACTACTICMA", 0.25 + 61 / 368, 3, dt(2000, 4, 30), True),  # long back : SKIP the _add_tenor does not account for month end roll here
    (dt(1999, 11, 15), dt(2000, 4, 15), "ACTACTICMA", 0.25 + 60 / 360, 3, dt(2000, 4, 15), True),  # long back
    (dt(2002, 8, 31), dt(2002, 11, 30), "ACTACTICMA", 0.25, 3, dt(2004, 11, 30), False),
])
def test_dcf_special(start, end, conv, expected, freq_m, term, stub):
    # The 4 ActICMA tests match short/long first/final stubs in 1998-ISDA-memo-EMU pdf
    result = dcf(start, end, conv, term, freq_m, stub)
    assert abs(result - expected) < 1e-12


@pytest.mark.parametrize("conv, freq_m, term, stub", [
    ("ACTACTICMA", None, None, None),
    ("ACTACTICMA", 3, None, None),
    ("30E360ISDA", 3, None, None),
    ("ACTACTICMA", 3, dt(2022, 4, 1), None),
    ("BadConv", None, None, None),
])
def test_dcf_raises(conv, freq_m, term, stub):
    with pytest.raises(ValueError):
        _ = dcf(dt(2022, 1, 1), dt(2022, 4, 1), conv, term, freq_m, stub)