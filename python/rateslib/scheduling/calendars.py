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

from rateslib import calendars
from rateslib.enums.generics import NoInput
from rateslib.scheduling.adjuster import _convert_to_adjuster

if TYPE_CHECKING:
    from rateslib.local_types import CalInput, CalTypes, datetime


def get_calendar(
    calendar: CalInput,
) -> CalTypes:
    """
    Returns a calendar object, possible constructed by the
    :class:`~rateslib.scheduling.CalendarManager`.

    .. role:: red

    .. role:: green

    Parameters
    ----------
    calendar : str, Cal, UnionCal, NamedCal, :red:`required`
        If `str`, then the calendar is returned from pre-calculated values.
        If a specific user defined calendar this is returned without modification.

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

       tgt_and_nyc_cal = get_calendar("tgt,nyc")
       print(tgt_and_nyc_cal.print(2023, 5))
       type(tgt_and_nyc_cal)

    """
    if isinstance(calendar, str):
        return calendars.get(calendar)
    elif isinstance(calendar, NoInput):
        return calendars.get("all")
    else:  # calendar is a Calendar object type
        return calendar


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
