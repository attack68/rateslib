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

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.enums.generics import NoInput
from rateslib.rs import Cal, NamedCal, UnionCal
from rateslib.scheduling.adjuster import _convert_to_adjuster

if TYPE_CHECKING:
    from rateslib.typing import CalInput, CalTypes, datetime


def get_calendar(
    calendar: CalInput,
    named: bool = False,
) -> CalTypes:
    """
    Returns a calendar object from base implementation or user-defined and/or combinations of such.

    .. role:: red

    .. role:: green

    Parameters
    ----------
    calendar : str, Cal, UnionCal, NamedCal, :red:`required`
        If `str`, then the calendar is returned from pre-calculated values.
        If a specific user defined calendar this is returned without modification.
    named : bool, :green:`optional (set as False)`
        If *True*, will pass any string input directly to Rust and attempt to return
        a :class:`~rateslib.scheduling.NamedCal` object. These will always only use the
        base implementation calendars and ignore any user-implemented calendars.

    Returns
    -------
    NamedCal, Cal, UnionCal

    Notes
    -----

    Please see the :ref:`defaults <defaults-doc>` section of the documentation to discover
    which named calendars are base implemented to *rateslib*.

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
       print(tgt_cal.print(2023, 5))
       tgt_cal.add_bus_days(dt(2023, 1, 3), 5, True)
       type(tgt_cal)

    Calendars can be combined from the pre-existing names using comma separation.

    .. ipython:: python

       tgt_and_nyc_cal = get_calendar("tgt,nyc", named=False)
       print(tgt_and_nyc_cal.print(2023, 5))
       type(tgt_and_nyc_cal)

    """
    if isinstance(calendar, str):
        if named:
            # parse the string directly in Rust
            return NamedCal(calendar)
        else:
            # parse the string in Python and return Rust Cal/UnionCal objects directly
            calendar = calendar.replace(" ", "")
            if calendar in defaults.calendars:
                return defaults.calendars[calendar]
            return _parse_str_calendar(calendar)
    elif isinstance(calendar, NoInput):
        return defaults.calendars["all"]
    else:  # calendar is a Calendar object type
        return calendar


def _parse_str_calendar(calendar: str) -> CalTypes:
    """Parse the calendar string using Python and construct calendar objects."""
    vectors = calendar.split("|")
    if len(vectors) == 1:
        return _parse_str_calendar_no_associated(vectors[0])
    elif len(vectors) == 2:
        return _parse_str_calendar_with_associated(vectors[0], vectors[1])
    else:
        raise ValueError("Cannot use more than one pipe ('|') operator in `calendar`.")


def _parse_str_calendar_no_associated(calendar: str) -> CalTypes:
    calendars = calendar.lower().split(",")
    if len(calendars) == 1:  # only one named calendar is found
        return defaults.calendars[calendars[0]]  # lookup Hashmap
    else:
        # combined calendars are not yet predefined so this does not benefit from hashmap speed
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


def _parse_str_calendar_with_associated(calendar: str, associated_calendar: str) -> CalTypes:
    calendars = calendar.lower().split(",")
    cals = [defaults.calendars[_] for _ in calendars]
    cals_ = []
    for cal in cals:
        if isinstance(cal, Cal):
            cals_.append(cal)
        elif isinstance(cal, NamedCal):
            cals_.extend(cal.union_cal.calendars)
        else:  # is UnionCal
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
