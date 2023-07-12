from typing import Optional, Union, Dict, Any
from math import floor
from datetime import datetime, timedelta
import calendar as calendar_mod

from dateutil.relativedelta import MO, TH, FR

from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    next_monday,
    next_monday_or_tuesday,
)
from pandas.tseries.offsets import CustomBusinessDay, Easter, Day, DateOffset

CalInput = Optional[Union[CustomBusinessDay, str]]

# Generic holidays
Epiphany = Holiday("Epiphany", month=1, day=6)
GoodFriday = Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)])
EasterMonday = Holiday("Easter Monday", month=1, day=1, offset=[Easter(), Day(1)])
AscentionDay = Holiday("Ascention Day", month=1, day=1, offset=[Easter(), Day(39)])
ChristmasEve = Holiday("Christmas Eve", month=12, day=24)
ChristmasDay = Holiday("Christmas Day", month=12, day=25)
ChristmasDayHoliday = Holiday(
    "Christmas Day Holiday", month=12, day=25, observance=next_monday_or_tuesday
)
BoxingDay = Holiday("Boxing Day", month=12, day=26)
BoxingDayHoliday = Holiday(
    "Boxing Day Holiday", month=12, day=26, observance=next_monday_or_tuesday
)
NewYearsEve = Holiday("New Year's Eve", month=12, day=31)
NewYearsDay = Holiday("New Year's Day", month=1, day=1)
NewYearsDayHoliday = Holiday(
    "New Year's Day Holiday", month=1, day=1, observance=next_monday
)

# US based
USMartinLutherKingJr = Holiday(
    "Dr. Martin Luther King Jr.",
    start_date=datetime(1986, 1, 1),
    month=1,
    day=1,
    offset=DateOffset(weekday=MO(3)),  # type: ignore[arg-type]
)
USPresidentsDay = Holiday(
    "US President" "s Day", month=2, day=1, offset=DateOffset(weekday=MO(3))  # type: ignore[arg-type]
)
USMemorialDay = Holiday(
    "US Memorial Day", month=5, day=31, offset=DateOffset(weekday=MO(-1))  # type: ignore[arg-type]
)
USIndependenceDay = Holiday("US Independence Day", month=7, day=4)
USLabourDay = Holiday("US Labour Day", month=9, day=1, offset=DateOffset(weekday=MO(1)))  # type: ignore[arg-type]
USColumbusDay = Holiday(
    "US Columbus Day", month=10, day=1, offset=DateOffset(weekday=MO(2))  # type: ignore[arg-type]
)
USThanksgivingDay = Holiday(
    "US Thanksgiving", month=11, day=1, offset=DateOffset(weekday=TH(4))  # type: ignore[arg-type]
)

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.

# UK based
UKEarlyMayBankHoliday = Holiday(
    "UK Early May Bank Holiday", month=5, day=1, offset=DateOffset(weekday=MO(1))  # type: ignore[arg-type]
)
UKSpringBankHoliday = Holiday(
    "UK Spring Bank Holiday", month=5, day=31, offset=DateOffset(weekday=MO(-1))  # type: ignore[arg-type]
)
UKSummerBankHoliday = Holiday(
    "UK Summer Bank Holiday", month=8, day=31, offset=DateOffset(weekday=MO(-1))  # type: ignore[arg-type]
)

# EUR based
EULabourDay = Holiday("EU Labour Day", month=5, day=1)
SwedenNational = Holiday("Sweden National Day", month=6, day=6)
MidsummerFriday = Holiday(
    "Swedish Midsummer", month=6, day=25, offset=DateOffset(weekday=FR(-1))  # type: ignore[arg-type]
)

CALENDAR_RULES: Dict[str, list[Any]] = {
    "bus": [],
    "tgt": [
        NewYearsDay,
        GoodFriday,
        EasterMonday,
        EULabourDay,
        ChristmasDay,
        BoxingDay,
    ],
    "ldn": [
        NewYearsDayHoliday,
        GoodFriday,
        EasterMonday,
        UKEarlyMayBankHoliday,
        UKSpringBankHoliday,
        UKSummerBankHoliday,
        ChristmasDayHoliday,
        BoxingDayHoliday,
    ],
    "nyc": [
        NewYearsDayHoliday,
        GoodFriday,
        EasterMonday,
        UKEarlyMayBankHoliday,
        UKSpringBankHoliday,
        UKSummerBankHoliday,
        ChristmasDayHoliday,
        BoxingDayHoliday,
    ],
    "stk": [
        NewYearsDay,
        Epiphany,
        GoodFriday,
        EasterMonday,
        EULabourDay,
        AscentionDay,
        SwedenNational,
        MidsummerFriday,
        ChristmasEve,
        ChristmasDay,
        BoxingDay,
        NewYearsEve,
    ],
    "osl": [
        NewYearsDay,
        Epiphany,
        GoodFriday,
        EasterMonday,
        EULabourDay,
        AscentionDay,
        SwedenNational,
        MidsummerFriday,
        ChristmasEve,
        ChristmasDay,
        BoxingDay,
        NewYearsEve,
    ],
}


def create_calendar(rules: list, weekmask: Optional[str] = None) -> CustomBusinessDay:
    """
    Create a calendar with specific business and holiday days defined.

    Parameters
    ----------
    rules : list[Holiday]
        A list of specific holiday dates defined by the
        ``pandas.tseries.holiday.Holiday`` class.
    weekmask : str, optional
        Set of days as business days. Defaults to *"Mon Tue Wed Thu Fri"*.

    Returns
    --------
    CustomBusinessDay

    Examples
    --------
    .. ipython:: python

       from pandas.tseries.holiday import Holiday
       from pandas import date_range
       TutsBday = Holiday("Tutankhamum Birthday", month=7, day=2)
       pyramid_builder = create_calendar(rules=[TutsBday], weekmask="Tue Wed Thu Fri Sat Sun")
       construction_days = date_range(dt(1999, 6, 25), dt(1999, 7, 5), freq=pyramid_builder)
       construction_days

    """
    weekmask = "Mon Tue Wed Thu Fri" if weekmask is None else weekmask
    return CustomBusinessDay(  # type: ignore[call-arg]
        calendar=AbstractHolidayCalendar(rules=rules),
        weekmask=weekmask,
    )


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def get_calendar(
    calendar: CalInput, kind: bool = False
) -> Union[CustomBusinessDay, tuple[CustomBusinessDay, str]]:
    """
    Returns a calendar object either from an available set or a user defined input.

    Parameters
    ----------
    calendar : str, None, or CustomBusinessDay
        If `None` a blank calendar is returned containing no holidays.
        If `str`, then the calendar is returned from pre-calculated values.
        If a specific user defined calendar this is returned without modification.
    kind : bool
        If `True` will also return the kind of calculation from `"null", "named",
        "custom"`.

    Returns
    -------
    CustomBusinessDay or tuple

    Examples
    --------
    .. ipython:: python

       gbp_cal = get_calendar("ldn")
       gbp_cal.calendar.holidays
       dt(2022, 1, 1) + 5 * gbp_cal
       type(gbp_cal)

    Calendars can be combined from the pre-existing names using comma separation.

    .. ipython:: python

       gbp_and_nyc_cal = get_calendar("ldn,nyc")
       gbp_and_nyc_cal.calendar.holidays

    """
    if calendar is None:
        ret = (create_calendar([], weekmask="Mon Tue Wed Thu Fri Sat Sun"), "null")
    elif isinstance(calendar, str):
        calendars = calendar.lower().split(",")
        if len(calendars) == 1:  # only one named calendar is found
            rules_: list[Any] = CALENDAR_RULES[calendars[0]]
            ret = (
                create_calendar(rules=rules_, weekmask="Mon Tue Wed Thu Fri"),
                "named",
            )
        else:
            rules_ = []
            for c in calendars:
                rules_.extend(CALENDAR_RULES[c])
            ret = (create_calendar(rules_, weekmask="Mon Tue Wed Thu Fri"), "named")
    else:  # calendar is a HolidayCalendar object
        ret = (calendar, "custom")

    return ret if kind else ret[0]


def _is_holiday(date: datetime, calendar: CustomBusinessDay):
    """
    Test whether a given date is a holiday in the given calendar

    Parameters
    ----------
    date : Datetime
        Date to test.
    calendar : Calendar of CustomBusinessDay type
        The holiday calendar to test against.

    Returns
    -------
    bool
    """
    if not isinstance(calendar, CustomBusinessDay):
        raise ValueError("`calendar` must be a `CustomBusinessDay` calendar type.")
    else:
        return not (date + 0 * calendar == date)


def _adjust_date(
    date: datetime,
    modifier: Optional[str],
    calendar: CalInput,
) -> datetime:
    """
    Modify a date under specific rule.

    Parameters
    ----------
    date : datetime
        The date to be adjusted.
    modifier : str or None
        The modification rule, in {"F", "MF", "P", "MP"}. If *None* returns date.
    calendar : calendar, optional
        The holiday calendar object to use. Required only if `modifier` is not *None*.
        If *None* a calendar is created where every day including weekends is valid.

    Returns
    -------
    datetime
    """
    if modifier is None:
        return date
    modifier = modifier.upper()
    if modifier not in ["F", "MF", "P", "MP"]:
        raise ValueError("`modifier` must be in {None, 'F', 'MF', 'P', 'MP'}")

    (adj_op, mod_op) = (
        ("rollforward", "rollback") if "F" in modifier else ("rollback", "rollforward")
    )
    calendar_: CustomBusinessDay = get_calendar(calendar)  # type: ignore[assignment]
    adjusted_date = getattr(calendar_, adj_op)(date)
    if adjusted_date.month != date.month and "M" in modifier:
        adjusted_date = getattr(calendar_, mod_op)(date)
    return adjusted_date.to_pydatetime()


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def add_tenor(
    start: datetime,
    tenor: str,
    modifier: Optional[str],
    calendar: CalInput,
) -> datetime:
    """
    Add a tenor to a given date under specific modification rules and holiday calendar.

    Note this function does **not** implement the `modified following month end` rule,
    where tenors starting on month end are adjusted to end on month end. For example
    a 3-month tenor starting 28th Feb 2022 would be adjusted to end on 31st May 2022.

    Parameters
    ----------
    start : datetime
        The initial date to which to add the tenor.
    tenor : str
        The tenor to add, identified by calendar days, `"D"`, months, `"M"`,
        years, `"Y"` or business days, `"B"`, for example `"10Y"` or `"5B"`.
    modifier : str, optional
        The modification rule to apply if the tenor is calendar days, months or years.
    calendar : CustomBusinessDay or str, optional
        The calendar for use with business day adjustment and modification.

    Returns
    -------
    datetime

    Examples
    --------
    .. ipython:: python
       :suppress:

       from rateslib.calendars import add_tenor, get_calendar, create_calendar, dcf
       from rateslib.curves import Curve, LineCurve, interpolate, index_left, IndexCurve
       from rateslib.dual import Dual, Dual2
       from rateslib.periods import FixedPeriod, FloatPeriod, Cashflow
       from rateslib.legs import FixedLeg, FloatLeg, CustomLeg, FloatLegExchange, FixedLegExchange, FloatLegExchangeMtm, FixedLegExchangeMtm
       from rateslib.instruments import FixedRateBond, FloatRateBond, Value, IRS, SBS, FRA, forward_fx, Spread, Fly, BondFuture, Bill, ZCS, FXSwap, ZCIS, IIRS
       from rateslib.solver import Solver
       from rateslib.splines import bspldnev_single, PPSpline
       from datetime import datetime as dt
       from pandas import date_range, Series, DataFrame

    .. ipython:: python

       add_tenor(dt(2022, 2, 28), "3M", None, None)
       add_tenor(dt(2022, 12, 28), "4b", "F", get_calendar("ldn"))
       add_tenor(dt(2022, 12, 28), "4d", "F", get_calendar("ldn"))
    """
    tenor = tenor.upper()
    if "D" in tenor:
        return _add_days(start, int(tenor[:-1]), modifier, calendar)
    elif "B" in tenor:
        calendar_: CustomBusinessDay = get_calendar(calendar)  # type: ignore[assignment]
        return (start + int(float(tenor[:-1])) * calendar_).to_pydatetime()  # type: ignore[attr-defined]
    elif "Y" in tenor:
        return _add_months(start, int(float(tenor[:-1]) * 12), modifier, calendar)
    elif "M" in tenor:
        return _add_months(start, int(tenor[:-1]), modifier, calendar)
    else:
        raise ValueError(
            "`tenor` must identify frequency in {'B', 'D', 'M', 'Y'} e.g. '1Y'"
        )


def _add_months(
    start: datetime,
    months: int,
    modifier: Optional[str],
    cal: CalInput,
) -> datetime:
    """add a given number of months to an input date"""
    year_roll = floor((start.month + months - 1) / 12)
    month = (start.month + months) % 12
    month = 12 if month == 0 else month
    try:
        end = datetime(start.year + year_roll, month, start.day)
    except ValueError:  # day is out of range for month, i.e. 30 or 31
        end = _get_eom(month, start.year + year_roll)
    return _adjust_date(end, modifier, cal)


def _add_days(
    start: datetime,
    days: int,
    modifier: Optional[str],
    cal: CalInput,
) -> datetime:
    end = start + timedelta(days=days)
    return _adjust_date(end, modifier, cal)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def _is_imm(date: datetime, hmuz=False) -> bool:
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


def dcf(
    start: datetime,
    end: datetime,
    convention: str,
    termination: Optional[datetime] = None,  # required for 30E360ISDA and ActActICMA
    frequency_months: Optional[
        int
    ] = None,  # required for ActActICMA = ActActISMA = ActActBond
    stub: Optional[bool] = None,  # required for ActActICMA = ActActISMA = ActActBond
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
        - `"ACTACTICMA", "ACTACTISMA", "ACTACTBOND"` (if the period is a stub
          the ``termination`` of the leg is used to assess front or back stubs and
          adjust the calculation accordingly)

    frequency_months : int, optional
        The number of months according to the frequency of the period. Required only
        with specific values for ``convention``.
    stub : bool, optional
        Required for `"ACTACTICMA", "ACTACTISMA", "ACTACTBOND"`. Non-stub periods will
        return a fraction equal to the frequency, e.g. 0.25 for quarterly.

    Returns
    --------
    float

    Notes
    -----
    Permitted values for the convention are:

    - `"1"`: Returns 1 for any period.
    - `"1+"`: Returns the number of months between dates divided by 12.
    - `"Act365F"`: Returns actual number of days divided by a fixed 365 denominator.
    - `"Act360"`: Returns actual number of days divided by a fixed 360 denominator.
    - `"30E360"`, `"EuroBondBasis"`: Months are treated as having 30 days and start
      and end dates are converted under the rule:

      * start day is minimum of (30, start day),
      * end day is minimum of (30, end day).

    - `"30360"`, `"360360"`, `"BondBasis"`: Months are treated as having 30 days
      and start and end dates are converted under the rule:

      * start day is minimum of (30, start day),
      * end day is minimum of (30, start day) only if start day was adjusted.

    - `"30360ISDA"`: Months are treated as having 30 days and start and end dates are
      converted under the rule:

      * start day is converted to 30 if it is a month end.
      * end day is converted to 30 if it is a month end.
      * end day is not converted if it coincides with the leg termination and is
        in February.

    - `"ActAct"`, `"ActActISDA"`: Calendar days between start and end are divided
      by 365 or 366 dependent upon whether they fall within a leap year or not.
    - `"ActActICMA"`, `"ActActISMA"`, `"ActActBond"`:

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
    if convention == "ACT365F":
        return (end - start) / timedelta(days=365)
    elif convention == "ACT360":
        return (end - start) / timedelta(days=360)
    elif convention in ["30360", "360360", "BONDBASIS"]:
        ds = min(30, start.day)
        de = min(ds, end.day) if ds == 30 else end.day
        y, m = end.year - start.year, (end.month - start.month) / 12
        return y + m + (de - ds) / 360
    elif convention in ["30E360", "EUROBONDBASIS"]:
        ds, de = min(30, start.day), min(30, end.day)
        y, m = end.year - start.year, (end.month - start.month) / 12
        return y + m + (de - ds) / 360
    elif convention == "30E360ISDA":
        if termination is None:
            raise ValueError(
                "`termination` must be supplied with specified `convention`."
            )

        def _is_end_feb(date):
            if date.month == 2:
                _, end_feb = calendar_mod.monthrange(date.year, 2)
                return date.day == end_feb
            return False

        ds = 30 if (start.day == 31 or _is_end_feb(start)) else start.day
        de = (
            30
            if (end.day == 31 or (_is_end_feb(end) and end != termination))
            else end.day
        )
        y, m = end.year - start.year, (end.month - start.month) / 12
        return y + m + (de - ds) / 360
    elif convention in ["ACTACT", "ACTACTISDA"]:
        if start == end:
            return 0.0

        start_date = datetime.combine(start, datetime.min.time())
        end_date = datetime.combine(end, datetime.min.time())

        year_1_diff = 366 if calendar_mod.isleap(start_date.year) else 365
        year_2_diff = 366 if calendar_mod.isleap(end_date.year) else 365

        total_sum: float = end.year - start.year - 1
        total_sum += (datetime(start.year + 1, 1, 1) - start_date).days / year_1_diff
        total_sum += (end_date - datetime(end.year, 1, 1)).days / year_2_diff
        return total_sum
    elif convention in ["ACTACTICMA", "ACTACTISMA", "ACTACTBOND"]:
        if frequency_months is None:
            raise ValueError(
                "`frequency_months` must be supplied with specified `convention`."
            )
        if termination is None:
            raise ValueError(
                "`termination` must be supplied with specified `convention`."
            )
        if stub is None:
            raise ValueError("`stub` must be supplied with specified `convention`.")
        if not stub:
            return frequency_months / 12
        else:
            if end == termination:  # stub is a BACK stub:
                fwd_end = _add_months(start, frequency_months, None, None)
                fraction = 0.0
                if end > fwd_end:  # stub is LONG
                    fraction += 1
                    fraction += (end - fwd_end) / (
                        _add_months(start, 2 * frequency_months, None, None) - fwd_end
                    )
                else:
                    fraction += (end - start) / (fwd_end - start)
                return fraction * frequency_months / 12
            else:  # stub is a FRONT stub
                prev_start = _add_months(end, -frequency_months, None, None)
                fraction = 0
                if start < prev_start:  # stub is LONG
                    fraction += 1
                    fraction += (prev_start - start) / (
                        prev_start - _add_months(end, -2 * frequency_months, None, None)
                    )
                else:
                    fraction += (end - start) / (end - prev_start)
                return fraction * frequency_months / 12
    elif convention == "1":
        return 1.0
    elif convention == "1+":
        return end.year - start.year + (end.month - start.month) / 12
    else:
        raise ValueError(
            "`convention` must be in {'Act365f', '1', '1+', 'Act360', "
            "'30360' '360360', 'BondBasis', '30E360', 'EuroBondBasis', "
            "'30E360ISDA', 'ActAct', 'ActActISDA', 'ActActICMA', "
            "'ActActISMA', 'ActActBond'}"
        )


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
