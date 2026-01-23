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

from __future__ import annotations

from calendar import monthrange
from datetime import datetime as dt
from typing import TYPE_CHECKING

import numpy as np

from rateslib import defaults
from rateslib.enums.generics import NoInput
from rateslib.rs import Cal, NamedCal, UnionCal
from rateslib.scheduling.adjuster import _convert_to_adjuster

if TYPE_CHECKING:
    from rateslib.typing import CalInput, CalTypes, datetime


def get_calendar(
    calendar: CalInput,
    named: bool = True,
) -> CalTypes:
    """
    Returns a calendar object either from an available set or a user defined input.

    Parameters
    ----------
    calendar : str, Cal, UnionCal, NamedCal
        If `str`, then the calendar is returned from pre-calculated values.
        If a specific user defined calendar this is returned without modification.
    named : bool
        If the calendar is more complex than a pre-existing single name calendar, then
        this argument determines if a :class:`~rateslib.scheduling.NamedCal` object, which is more
        compactly serialized but slower to create, or a :class:`~rateslib.scheduling.UnionCal`
        object, which is faster to create but with more verbose serialization is returned.
        The default prioritises serialization.

    Returns
    -------
    NamedCal, Cal, UnionCal

    Notes
    -----

    Please see the :ref:`defaults <defaults-doc>` section of the documentation to discover
    which named calendars are preloaded to *rateslib*.

    Combined calendars can be created with comma separated input, e.g. *"tgt,nyc"*. This would
    be the typical calendar assigned to a cross-currency derivative such as a EUR/USD
    cross-currency swap.

    For short-dated, FX instrument date calculations a concept known as an
    **associated settlement calendars** is introduced. This uses a secondary calendar to determine
    if a calculated date is a valid settlement day, but it is not used in the determination
    of tenor dates. For a EURUSD FX instrument the appropriate calendar combination is *"tgt|nyc"*.
    For a GBPEUR FX instrument the appropriate calendar combination is *"ldn,tgt|nyc"*.

    Examples
    --------
    .. ipython:: python
       :suppress:

       from rateslib import get_calendar, dt

    .. ipython:: python

       tgt_cal = get_calendar("tgt")
       tgt_cal.holidays[300:312]
       tgt_cal.add_bus_days(dt(2023, 1, 3), 5, True)
       type(tgt_cal)

    Calendars can be combined from the pre-existing names using comma separation.

    .. ipython:: python

       tgt_and_nyc_cal = get_calendar("tgt,nyc", named=False)
       tgt_and_nyc_cal.holidays[300:312]
       type(tgt_and_nyc_cal)

    """
    if isinstance(calendar, str):
        # parse the string in Python and return Rust Cal/UnionCal objects directly
        calendar = calendar.replace(" ", "")
        if calendar in defaults.calendars:
            return defaults.calendars[calendar]
        return _parse_str_calendar(calendar, named)
    elif isinstance(calendar, NoInput):
        return defaults.calendars["all"]
    else:  # calendar is a Calendar object type
        return calendar


def print_calendar(calendar: CalInput, year: int) -> str:
    """
    A string representation of a given year in a calendar.

    Parameters
    ----------
    calendar: Cal, UnionCal, NamedCal, str
        The calendar to return the representation for.
    year: int
        The year to display.

    Returns
    -------
    str

    Examples
    --------

    .. ipython:: python
       :suppress:

       from rateslib.scheduling import print_calendar

    .. ipython:: python

       print(print_calendar("bjs|fed", 2026))

    """
    calendar_ = get_calendar(calendar)
    del calendar
    data = [_print_month(calendar_, year, i + 1).split("\n") for i in range(12)]
    if isinstance(calendar_, Cal):
        cal_name = "Cal:"
    elif isinstance(calendar_, UnionCal):
        cal_name = "UnionCal:"
    else:
        cal_name = f"NamedCal({calendar_.name})"
    output = (
        f"\n"
        f"{cal_name}\n"
        f"{data[0][0]:>20}   {data[3][0]:>20}   {data[6][0]:>20}   {data[9][0]:>20}\n"
        f"{data[0][1]}  {data[3][1]}  {data[6][1]}  {data[9][1]}\n"
        f"{data[0][2]}   {data[3][2]}   {data[6][2]}   {data[9][2]}\n"
        f"{data[0][3]}   {data[3][3]}   {data[6][3]}   {data[9][3]}\n"
        f"{data[0][4]}   {data[3][4]}   {data[6][4]}   {data[9][4]}\n"
        f"{data[0][5]}   {data[3][5]}   {data[6][5]}   {data[9][5]}\n"
        f"{data[0][6]}   {data[3][6]}   {data[6][6]}   {data[9][6]}\n"
        f"{data[0][7]}   {data[3][7]}   {data[6][7]}   {data[9][7]}\n"
        f"{data[1][0]:>20}   {data[4][0]:>20}   {data[7][0]:>20}   {data[10][0]:>20}\n"
        f"{data[1][1]}  {data[4][1]}  {data[7][1]}  {data[10][1]}\n"
        f"{data[1][2]}   {data[4][2]}   {data[7][2]}   {data[10][2]}\n"
        f"{data[1][3]}   {data[4][3]}   {data[7][3]}   {data[10][3]}\n"
        f"{data[1][4]}   {data[4][4]}   {data[7][4]}   {data[10][4]}\n"
        f"{data[1][5]}   {data[4][5]}   {data[7][5]}   {data[10][5]}\n"
        f"{data[1][6]}   {data[4][6]}   {data[7][6]}   {data[10][6]}\n"
        f"{data[1][7]}   {data[4][7]}   {data[7][7]}   {data[10][7]}\n"
        f"{data[2][0]:>20}   {data[5][0]:>20}   {data[8][0]:>20}   {data[11][0]:>20}\n"
        f"{data[2][1]}  {data[5][1]}  {data[8][1]}  {data[11][1]}\n"
        f"{data[2][2]}   {data[5][2]}   {data[8][2]}   {data[11][2]}\n"
        f"{data[2][3]}   {data[5][3]}   {data[8][3]}   {data[11][3]}\n"
        f"{data[2][4]}   {data[5][4]}   {data[8][4]}   {data[11][4]}\n"
        f"{data[2][5]}   {data[5][5]}   {data[8][5]}   {data[11][5]}\n"
        f"{data[2][6]}   {data[5][6]}   {data[8][6]}   {data[11][6]}\n"
        f"{data[2][7]}   {data[5][7]}   {data[8][7]}   {data[11][7]}\n"
        f"Legend:\n"
        f"'1-31': Settleable business day         'X': Non-settleable business day \n"
        f"   '.': Non-business weekend            '*': Non-business day\n"
    )
    return output


def _print_month(calendar: CalTypes, year: int, month: int) -> str:
    """
    Legend:

    * non-business day / specific holiday
    # non-working Saturday or Sunday
    X business day but not a settleable day.
    """
    output = f"{dt(year, month, 1).strftime('%B')} {year}\n"
    output += "Su Mo Tu We Th Fr Sa \n"

    weekday, days = monthrange(year, month)
    idx_start = (weekday + 1) % 7

    arr = np.array(["  "] * 42)
    for i in range(days):
        date = dt(year, month, i + 1)
        if calendar.is_bus_day(date) and calendar.is_settlement(date):
            _: str = f"{i + 1:>2}"
        elif calendar.is_bus_day(date) and not calendar.is_settlement(date):
            _ = " X"
        elif not calendar.is_bus_day(date) and dt.weekday(date) in [5, 6]:
            _ = " ."
        else:
            _ = " *"
        arr[i + idx_start] = _

    for row in range(6):
        output += (
            f"{arr[row * 7]} {arr[row * 7 + 1]} {arr[row * 7 + 2]} {arr[row * 7 + 3]} "
            f"{arr[row * 7 + 4]} {arr[row * 7 + 5]} {arr[row * 7 + 6]}\n"
        )

    return output

    output += "Sunday\n"


def _parse_str_calendar(calendar: str, named: bool) -> CalTypes:
    """Parse the calendar string using Python and construct calendar objects."""
    vectors = calendar.split("|")
    if len(vectors) == 1:
        return _parse_str_calendar_no_associated(vectors[0], named)
    elif len(vectors) == 2:
        return _parse_str_calendar_with_associated(vectors[0], vectors[1], named)
    else:
        raise ValueError("Cannot use more than one pipe ('|') operator in `calendar`.")


def _parse_str_calendar_no_associated(calendar: str, named: bool) -> CalTypes:
    calendars = calendar.lower().split(",")
    if len(calendars) == 1:  # only one named calendar is found
        return defaults.calendars[calendars[0]]  # lookup Hashmap
    else:
        # combined calendars are not yet predefined so this does not benefit from hashmap speed
        if named:
            return NamedCal(calendar)
        else:
            cals = [defaults.calendars[_] for _ in calendars]
            cals_: list[Cal] = []
            for cal in cals:
                if isinstance(cal, Cal):
                    cals_.append(cal)
                elif isinstance(cal, NamedCal):
                    cals_.extend(cal.union_cal.calendars)
                else:
                    cals_.extend(cal.calendars)
            return UnionCal(cals_, None)


def _parse_str_calendar_with_associated(
    calendar: str, associated_calendar: str, named: bool
) -> CalTypes:
    if named:
        return NamedCal(calendar + "|" + associated_calendar)
    else:
        calendars = calendar.lower().split(",")
        cals = [defaults.calendars[_] for _ in calendars]
        cals_ = []
        for cal in cals:
            if isinstance(cal, Cal):
                cals_.append(cal)
            elif isinstance(cal, NamedCal):
                cals_.extend(cal.union_cal.calendars)
            else:
                cals_.extend(cal.calendars)

        settlement_calendars = associated_calendar.lower().split(",")
        sets = [defaults.calendars[_] for _ in settlement_calendars]
        sets_: list[Cal] = []
        for cal in sets:
            if isinstance(cal, Cal):
                sets_.append(cal)
            elif isinstance(cal, NamedCal):
                sets_.extend(cal.union_cal.calendars)
            else:
                sets_.extend(cal.calendars)

        return UnionCal(cals_, sets_)


def _get_years_and_months(d1: datetime, d2: datetime) -> tuple[int, int]:
    """
    Get the whole number of years and months between two dates
    """
    years: int = d2.year - d1.year
    if (d2.month == d1.month and d2.day < d1.day) or (d2.month < d1.month):
        years -= 1

    months: int = (d2.month - d1.month) % 12
    return years, months


def _adjust_date(
    date: datetime,
    modifier: str,
    calendar: CalInput,
    settlement: bool = True,
) -> datetime:
    """
    Modify a date under specific rule.

    Parameters
    ----------
    date : datetime
        The date to be adjusted.
    modifier : str
        The modification rule, in {"NONE", "F", "MF", "P", "MP"}. If *'NONE'* returns date.
    calendar : calendar, optional
        The holiday calendar object to use. Required only if `modifier` is not *'NONE'*.
        If not given a calendar is created where every day including weekends is valid.
    settlement : bool
        Whether to also enforce the associated settlement calendar.

    Returns
    -------
    datetime
    """
    cal_ = get_calendar(calendar)
    return _convert_to_adjuster(modifier, settlement, True).adjust(date, cal_)


def _is_day_type_tenor(tenor: str) -> bool:
    tenor_ = tenor.upper()
    return "D" in tenor_ or "B" in tenor_ or "W" in tenor_
