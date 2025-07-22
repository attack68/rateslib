from __future__ import annotations

import calendar as calendar_mod
from datetime import datetime
from typing import TYPE_CHECKING

from rateslib.default import NoInput
from rateslib.rs import RollDay
from rateslib.scheduling.calendars import _adjust_date

if TYPE_CHECKING:
    from rateslib.typing import CalInput, Callable, int_, str_


def next_imm(start: datetime, method: str = "imm") -> datetime:
    """Return the next IMM date *after* the given start date.

    Parameters
    ----------
    start : datetime
        The date from which to determine the next IMM.
    method : str in {"imm", "serial_imm", "credit_imm", "credit_imm_HU", "credit_imm_MZ"}
        A calculation identifier. See notes

    Returns
    -------
    datetime

    Notes
    -----
    Default *'imm'* returns the third Wednesday in any month of March, June, September or December.

    *'serial_imm'* returns the third Wednesday in any month of the year.

    *'credit_imm'* returns the 20th of the month in March, June, September or December.

    *'credit_imm_HU'* returns the 20th of the month in March or September, facilitating CDSs that
    rolls on a 6-month basis.

    *'credit_imm_MZ'* returns the 20th of the month in June and December.
    """
    month, year = start.month, start.year
    candidate1 = _next_imm_from_som(month, year, method)
    if start < candidate1:  # then the first detected next_imm is valid
        return candidate1
    else:
        if month == 12:
            candidate2 = _next_imm_from_som(1, year + 1, method)
        else:
            candidate2 = _next_imm_from_som(month + 1, year, method)
        return candidate2


def get_imm(
    month: int_ = NoInput(0),
    year: int_ = NoInput(0),
    code: str_ = NoInput(0),
) -> datetime:
    """
    Return an IMM date for a specified month.

    .. note::

       This is a fixed income IMM date (i.e. third wednesday of a given month) and not a
       credit IMM date (20th of a given month).

    Parameters
    ----------
    month: int
        The month of the year in which the IMM date falls.
    year: int
        The year in which the IMM date falls.
    code: str
        Identifier in the form of a one digit month code and 21st century year, e.g. "U29".
        If code is given ``month`` and ``year`` are unused.

    Returns
    -------
    datetime
    """
    if isinstance(code, str):
        year = int(code[1:]) + 2000
        month = MONTHS[code[0].upper()]
    elif isinstance(month, NoInput) or isinstance(year, NoInput):
        raise ValueError("`month` and `year` must each be valid integers if `code`not given.")
    return _get_imm(month, year)


def _get_rollday(roll: RollDay | str | int_) -> RollDay | None:
    """Convert a user str or int into a RollDay enum object."""
    if isinstance(roll, RollDay):
        return roll
    elif isinstance(roll, str):
        return {
            "EOM": RollDay.Day(31),
            "SOM": RollDay.Day(1),
            "IMM": RollDay.IMM(),
        }[roll.upper()]
    elif isinstance(roll, int):
        return RollDay.Day(roll)
    return None


def _get_roll(month: int, year: int, roll: str | int) -> datetime:
    if isinstance(roll, str):
        if roll == "eom":
            date = _get_eom(month, year)
        elif roll == "som":
            date = datetime(year, month, 1)
        elif roll == "imm":
            date = _get_imm(month, year)
    else:
        try:
            date = datetime(year, month, roll)
        except ValueError:  # day is out of range for month, i.e. 30 or 31
            date = _get_eom(month, year)
    return date


def _is_imm(date: datetime, hmuz: bool = False) -> bool:
    """
    Test whether a given date is an IMM date, defined as third wednesday in month.

    Parameters
    ----------
    date : datetime,
        Date to test
    hmuz : bool, optional
        Flag to return True for IMMs only in Mar, Jun, Sep or Dec

    Returns
    -------
    bool
    """
    if hmuz and date.month not in [3, 6, 9, 12]:
        return False
    return date == _get_imm(date.month, date.year)


def _get_imm(month: int, year: int) -> datetime:
    """
    Get the day in the month corresponding to IMM (3rd Wednesday).

    Parameters
    ----------
    month : int
        Month
    year : int
        Year

    Returns
    -------
    int : Day
    """
    imm_map = {0: 17, 1: 16, 2: 15, 3: 21, 4: 20, 5: 19, 6: 18}
    return datetime(year, month, imm_map[datetime(year, month, 1).weekday()])


def _is_eom(date: datetime) -> bool:
    """
    Test whether a given date is end of month.

    Parameters
    ----------
    date : datetime,
        Date to test

    Returns
    -------
    bool
    """
    return date.day == calendar_mod.monthrange(date.year, date.month)[1]


def _is_eom_cal(date: datetime, cal: CalInput) -> bool:
    """Test whether a given date is end of month under a specific calendar"""
    end_day = calendar_mod.monthrange(date.year, date.month)[1]
    eom = datetime(date.year, date.month, end_day)
    adj_eom = _adjust_date(eom, "P", cal)
    return date == adj_eom


def _get_eom(month: int, year: int) -> datetime:
    """
    Get the day in the month corresponding to last day.

    Parameters
    ----------
    month : int
        Month
    year : int
        Year

    Returns
    -------
    int : Day
    """
    return datetime(year, month, calendar_mod.monthrange(year, month)[1])


def _is_som(date: datetime) -> bool:
    """
    Test whether a given date is start of month.

    Parameters
    ----------
    date : datetime,
        Date to test

    Returns
    -------
    bool
    """
    return date.day == 1


def _get_2nd_friday(month: int, year: int) -> datetime:
    """
    Get the second Friday of the gien month and year.
    This is the settlement date of ASX AUD 90D Bank Bill futures (BBSW 3M Futures).

    Parameters
    ----------
    month : int
        Month
    year : int
        Year

    Returns
    -------
    datetime
    """
    map_ = {0: 12, 1: 11, 2: 10, 3: 9, 4: 8, 5: 14, 6: 13}
    return datetime(year, month, map_[datetime(year, month, 1).weekday()])


def _get_1st_wednesday_after_9th(month: int, year: int) -> datetime:
    """
    Get the first Wednesday after the 9th of a given month and year.
    This is the settlement date of ASX NZD 90D Bank Bill futures (BKBM 3M Futures).

    Parameters
    ----------
    month : int
        Month
    year : int
        Year

    Returns
    -------
    datetime
    """
    map_ = {0: 2, 1: 1, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3}
    return datetime(year, month, 9 + map_[datetime(year, month, 9).weekday()])


_IS_ROLL: dict[str, Callable[..., bool]] = {
    "eom": _is_eom,
    "som": _is_som,
    "imm": _is_imm,
}


MONTHS = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}


NEXT_IMM_MAP = {
    1: [3, 1, 3, 3, 6],
    2: [3, 2, 3, 3, 6],
    3: [3, 3, 3, 3, 6],
    4: [6, 4, 6, 9, 6],
    5: [6, 5, 6, 9, 6],
    6: [6, 6, 6, 9, 6],
    7: [9, 7, 9, 9, 12],
    8: [9, 8, 9, 9, 12],
    9: [9, 9, 9, 9, 12],
    10: [12, 10, 12, 3, 12],
    11: [12, 11, 12, 3, 12],
    12: [12, 12, 12, 3, 12],
}


def _next_imm_from_som(month: int, year: int, method: str = "imm") -> datetime:
    """Get the next IMM date after the 1st day of the given month and year."""
    _idx = {"imm": 0, "serial_imm": 1, "credit_imm": 2, "credit_imm_HU": 3, "credit_imm_MZ": 4}
    required_month = NEXT_IMM_MAP[month][_idx[method]]
    required_year = year if required_month >= month else year + 1

    if method == "imm" or method == "serial_imm":
        return get_imm(required_month, required_year)
    else:
        return datetime(required_year, required_month, 20)
