from __future__ import annotations

import calendar as calendar_mod
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from rateslib.calendars.dcfs import _DCF
from rateslib.calendars.rs import (
    _get_modifier,
    _get_rollday,
    get_calendar,
)
from rateslib.default import NoInput, _drb
from rateslib.rs import Cal, Modifier, NamedCal, RollDay, UnionCal

if TYPE_CHECKING:
    from rateslib.typing import CalInput, bool_, datetime_, int_, str_

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.

Modifier.__doc__ = "Enumerable type for modification rules."
RollDay.__doc__ = "Enumerable type for roll day types."


def dcf(
    start: datetime,
    end: datetime,
    convention: str,
    termination: datetime_ = NoInput(0),  # required for 30E360ISDA and ActActICMA
    frequency_months: int_ = NoInput(0),  # req. ActActICMA = ActActISMA = ActActBond
    stub: bool_ = NoInput(0),  # required for ActActICMA = ActActISMA = ActActBond
    roll: str | int_ = NoInput(0),  # required also for ActACtICMA = ...
    calendar: CalInput = NoInput(0),  # required for ActACtICMA = ActActISMA = ActActBond
) -> float:
    """
    Calculate the day count fraction of a period.

    Parameters
    ----------
    start : datetime
        The adjusted start date of the calculation period.
    end : datetime
        The adjusted end date of the calculation period.
    convention : str
        The day count convention of the calculation period accrual. See notes.
    termination : datetime, optional
        The adjusted termination date of the leg. Required only if ``convention`` is
        one of the following values:

        - `"30E360ISDA"` (since end Feb is adjusted to 30 unless it aligns with
          ``termination`` of a leg)
        - `"ACTACTICMA", "ACTACTISMA", "ACTACTBOND", "ACTACTICMA_STUB365F"`, (if the period is
          a stub the ``termination`` of the leg is used to assess front or back stubs and
          adjust the calculation accordingly)

    frequency_months : int, optional
        The number of months according to the frequency of the period. Required only
        with specific values for ``convention``.
    stub : bool, optional
        Required for `"ACTACTICMA", "ACTACTISMA", "ACTACTBOND", "ACTACTICMA_STUB365F"`.
        Non-stub periods will
        return a fraction equal to the frequency, e.g. 0.25 for quarterly.
    roll : str, int, optional
        Used by `"ACTACTICMA", "ACTACTISMA", "ACTACTBOND", "ACTACTICMA_STUB365F"` to project
        regular periods when calculating stubs.
    calendar: str, Calendar, optional
        Required for `"BUS252"` to count business days in period.

    Returns
    --------
    float

    Notes
    -----
    Permitted values for the convention are:

    - `"1"`: Returns 1 for any period.
    - `"1+"`: Returns the number of months between dates divided by 12.
    - `"Act365F"`: Returns actual number of days divided by a fixed 365 denominator.
    - `"Act365F+"`: Returns the number of years and the actual number of days in the fractional year
      divided by a fixed 365 denominator.
    - `"Act360"`: Returns actual number of days divided by a fixed 360 denominator.
    - `"30E360"`, `"EuroBondBasis"`: Months are treated as having 30 days and start
      and end dates are converted under the rule:

      * start day is minimum of (30, start day),
      * end day is minimum of (30, end day).

    - `"30360"`, `"360360"`, `"BondBasis"`: Months are treated as having 30 days
      and start and end dates are converted under the rule:

      * start day is minimum of (30, start day),
      * end day is minimum of (30, end day) if start day >= 30.

    - `"30U360"`: Months are treated as having 30 days and start and end dates are converted
      under the following rules in order:

      * If the ``roll`` is EoM and ``start`` is end-Feb then:

         - start day is 30.
         - end day is 30 ``end`` is also end-Feb.

      * If start day is 30 or 31 then it is converted to 30.
      * End day is converted to 30 if it is 31 and start day is 30.

    - `"30360ISDA"`: Months are treated as having 30 days and start and end dates are
      converted under the rule:

      * start day is converted to 30 if it is a month end.
      * end day is converted to 30 if it is a month end.
      * end day is not converted if it coincides with the leg termination and is
        in February.

    - `"ActAct"`, `"ActActISDA"`: Calendar days between start and end are divided
      by 365 or 366 dependent upon whether they fall within a leap year or not.
    - `"ActActICMA"`, `"ActActISMA"`, `"ActActBond"`, `"ActActICMA_stub365f"`: Returns a fraction
      relevant to the frequency of the schedule if a regular period. If a stub then projects
      a regular period and returns a fraction of that period.
    - `"Bus252"`: Business days between start and end divided by 252. If business days, `start` is
      included whilst `end` is excluded.

    Further information can be found in the
    :download:`2006 ISDA definitions <_static/2006_isda_definitions.pdf>` and
    :download:`2006 ISDA 30360 example <_static/30360isda_2006_example.xls>`.

    Examples
    --------

    .. ipython:: python

       dcf(dt(2000, 1, 1), dt(2000, 4, 3), "Act360")
       dcf(dt(2000, 1, 1), dt(2000, 4, 3), "Act365f")
       dcf(dt(2000, 1, 1), dt(2000, 4, 3), "ActActICMA", dt(2010, 1, 1), 3, False)
       dcf(dt(2000, 1, 1), dt(2000, 4, 3), "ActActICMA", dt(2010, 1, 1), 3, True)

    """
    convention = convention.upper()
    try:
        return _DCF[convention](start, end, termination, frequency_months, stub, roll, calendar)
    except KeyError:
        raise ValueError(
            "`convention` must be in {'Act365f', '1', '1+', 'Act360', "
            "'30360' '360360', 'BondBasis', '30U360', '30E360', 'EuroBondBasis', "
            "'30E360ISDA', 'ActAct', 'ActActISDA', 'ActActICMA', "
            "'ActActISMA', 'ActActBond'}",
        )


# TODO (deprecate): this function on 2.0.0
def create_calendar(rules: list[datetime], week_mask: list[int] | NoInput = NoInput(0)) -> Cal:
    """
    Create a calendar with specific business and holiday days defined.

    .. warning::

       This function is deprecated. Create a :class:`~rateslib.calendars.Cal` object instead.

    Parameters
    ----------
    rules : list[datetime]
        A list of specific holiday dates.
    week_mask : list[int], optional
        Set of days excluded from the working week. [5,6] is Saturday and Sunday.

    Returns
    --------
    Cal
    """
    weekmask = _drb([5, 6], week_mask)
    return Cal(rules, weekmask)


def add_tenor(
    start: datetime,
    tenor: str,
    modifier: str,
    calendar: CalInput = NoInput(0),
    roll: str | int_ = NoInput(0),
    settlement: bool = False,
    mod_days: bool = False,
) -> datetime:
    """
    Add a tenor to a given date under specific modification rules and holiday calendar.

    .. warning::

       Note this function does not validate the ``roll`` input, but expects it to be correct.
       This can be used to correctly replicate a schedule under a given roll day. For example
       a modified 29th May +3M will default to 29th Aug, but can be made to match
       31 Aug with *'eom'* *roll*, or 30th Aug with 30 *roll*.

    Parameters
    ----------
    start : datetime
        The initial date to which to add the tenor.
    tenor : str
        The tenor to add, identified by calendar days, `"D"`, months, `"M"`,
        years, `"Y"` or business days, `"B"`, for example `"10Y"` or `"5B"`.
    modifier : str, optional in {"NONE", "MF", "F", "MP", "P"}
        The modification rule to apply if the tenor is calendar days, months or years.
    calendar : CustomBusinessDay or str, optional
        The calendar for use with business day adjustment and modification.
    roll : str, int, optional
        This is only required if the tenor is given in months or years. Ensures the tenor period
        associates with a schedule's roll day.
    settlement : bool, optional
        Whether to enforce the settlement with an associated settlement calendar. If there is
        no associated settlement calendar this will have no effect.
    mod_days : bool, optional
        If *True* will apply modified rules to day type tenors as well as month and year tenors.
        If *False* will convert "MF" to "F" and "MP" to "P" for day type tenors.

    Returns
    -------
    datetime

    Notes
    ------

    .. ipython:: python
       :suppress:

       from rateslib.calendars import add_tenor, get_calendar, create_calendar, dcf
       from rateslib.scheduling import Schedule
       from rateslib.curves import Curve, LineCurve, index_left
       from rateslib.dual import Dual, Dual2
       from rateslib.periods import FixedPeriod, FloatPeriod, Cashflow, IndexFixedPeriod, IndexCashflow, NonDeliverableCashflow, NonDeliverableFixedPeriod
       from rateslib.legs import FixedLeg, FloatLeg, CustomLeg, FloatLegMtm, FixedLegMtm, IndexFixedLeg, ZeroFixedLeg, ZeroFloatLeg, ZeroIndexLeg
       from rateslib.instruments import FixedRateBond, FloatRateNote, Value, IRS, SBS, FRA, Spread, Fly, BondFuture, Bill, ZCS, FXSwap, ZCIS, IIRS, STIRFuture
       from rateslib.fx import forward_fx
       from rateslib.solver import Solver
       from rateslib.splines import bspldnev_single, PPSpline
       from datetime import datetime as dt
       import pandas as pd
       from pandas import date_range, Series, DataFrame
       pd.set_option("display.float_format", lambda x: '%.2f' % x)
       pd.set_option("display.max_columns", None)
       pd.set_option("display.width", 500)


    Read more about the ``settlement`` argument in the :ref:`calendar user guide <cal-doc>`.

    The ``mod_days`` argument is provided to avoid having to reconfigure *Instrument*
    specifications when a *termination* may differ between months or years, and days or weeks.
    Most *Instruments* will be defined by the typical modified following (*"MF"*) ``modifier``,
    but this would prefer not to always apply.

    .. ipython:: python

       add_tenor(dt(2021, 1, 29), "1M", "MF", "bus")

    while, the following will by default roll into a new month,

    .. ipython:: python

       add_tenor(dt(2021, 1, 22), "8d", "MF", "bus")

    unless,

    .. ipython:: python

       add_tenor(dt(2021, 1, 22), "8d", "MF", "bus", mod_days=True)

    Examples
    --------

    .. ipython:: python

       add_tenor(dt(2022, 2, 28), "3M", "NONE")
       add_tenor(dt(2022, 12, 28), "4b", "F", get_calendar("ldn"))
       add_tenor(dt(2022, 12, 28), "4d", "F", get_calendar("ldn"))
    """  # noqa: E501
    tenor = tenor.upper()
    cal_ = get_calendar(calendar)
    if "D" in tenor:
        return cal_.add_days(start, int(tenor[:-1]), _get_modifier(modifier, mod_days), settlement)
    elif "B" in tenor:
        return cal_.add_bus_days(start, int(tenor[:-1]), settlement)
    elif "Y" in tenor:
        months = int(float(tenor[:-1]) * 12)
        return cal_.add_months(
            start,
            months,
            _get_modifier(modifier, True),
            _get_rollday(roll),
            settlement,
        )
    elif "M" in tenor:
        return cal_.add_months(
            start,
            int(tenor[:-1]),
            _get_modifier(modifier, True),
            _get_rollday(roll),
            settlement,
        )
    elif "W" in tenor:
        return cal_.add_days(
            start,
            int(tenor[:-1]) * 7,
            _get_modifier(modifier, mod_days),
            settlement,
        )
    else:
        raise ValueError("`tenor` must identify frequency in {'B', 'D', 'W', 'M', 'Y'} e.g. '1Y'")


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


def _next_imm_from_som(month: int, year: int, method: str = "imm") -> datetime:
    """Get the next IMM date after the 1st day of the given month and year."""
    _idx = {"imm": 0, "serial_imm": 1, "credit_imm": 2, "credit_imm_HU": 3, "credit_imm_MZ": 4}
    required_month = NEXT_IMM_MAP[month][_idx[method]]
    required_year = year if required_month >= month else year + 1

    if method == "imm" or method == "serial_imm":
        return get_imm(required_month, required_year)
    else:
        return datetime(required_year, required_month, 20)


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
    modifier = modifier.upper()
    return cal_.roll(date, _get_modifier(modifier, True), settlement)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


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


def _get_years_and_months(d1: datetime, d2: datetime) -> tuple[int, int]:
    """
    Get the whole number of years and months between two dates
    """
    years: int = d2.year - d1.year
    if (d2.month == d1.month and d2.day < d1.day) or (d2.month < d1.month):
        years -= 1

    months: int = (d2.month - d1.month) % 12
    return years, months


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def _is_day_type_tenor(tenor: str) -> bool:
    tenor_ = tenor.upper()
    return "D" in tenor_ or "B" in tenor_ or "W" in tenor_


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


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


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


def _get_fx_expiry_and_delivery(
    eval_date: datetime_,
    expiry: str | datetime,
    delivery_lag: int | datetime,
    calendar: CalInput,
    modifier: str,
    eom: bool,
) -> tuple[datetime, datetime]:
    """
    Determines the expiry and delivery date of an FX option using the following rules:

    See Foreign Exchange Option Pricing by Iain Clark

    Parameters
    ----------
    eval_date: datetime
        The evaluation date, which is today (if required)
    expiry: str, datetime
        The expiry date
    delivery_lag: int, datetime
        Number of days, e.g. spot = 2, or a specified datetime for FX settlement after expiry.
    calendar: CalInput
        The calendar used for date rolling. This function makes use of the `settlement` option
        within calendars.
    modifier: str
        Date rule, expected to be "MF" for most FX rate tenors.
    eom: bool
        Whether end-of-month is preserved in tenor date determination.

    Returns
    -------
    tuple of datetime
    """
    if isinstance(expiry, str):
        if isinstance(eval_date, NoInput):
            raise ValueError("`expiry` as string tenor requires `eval_date`.")
        # then the expiry will be implied
        e = expiry.upper()
        if "M" in e or "Y" in e:
            # method
            if isinstance(delivery_lag, datetime):
                raise ValueError(
                    "Cannot determine FXOption expiry and delivery with given parameters.\n"
                    "Supply a `delivery_lag` as integer business days and not a datetime, when "
                    "using a string tenor `expiry`.",
                )
            else:
                spot = get_calendar(calendar).lag(eval_date, delivery_lag, True)
                roll = "eom" if (eom and _is_eom(spot)) else spot.day
                delivery_: datetime = add_tenor(spot, expiry, modifier, calendar, roll, True)
                expiry_ = get_calendar(calendar).add_bus_days(delivery_, -delivery_lag, False)
                return expiry_, delivery_
        else:
            expiry_ = add_tenor(eval_date, expiry, "F", calendar, NoInput(0), False)
    else:
        expiry_ = expiry

    if isinstance(delivery_lag, datetime):
        delivery_ = delivery_lag
    else:
        delivery_ = get_calendar(calendar).lag(expiry_, delivery_lag, True)

    return expiry_, delivery_


_IS_ROLL: dict[str, Callable[..., bool]] = {
    "eom": _is_eom,
    "som": _is_som,
    "imm": _is_imm,
}

__all__ = (
    "add_tenor",
    "Cal",
    "create_calendar",
    "dcf",
    "Modifier",
    "NamedCal",
    "RollDay",
    "UnionCal",
    "get_calendar",
    "get_imm",
    "next_imm",
)
