from typing import Optional, Union, Dict, Any
from math import floor
from datetime import datetime, timedelta
import calendar as calendar_mod
import warnings

from pandas.tseries.offsets import CustomBusinessDay, Day
from rateslib.default import NoInput
from rateslib.rs import Cal, UnionCal, get_named_calendar, RollDay, Modifier

CalTypes = Union[Cal, UnionCal]
CalInput = Union[CalTypes, str, NoInput]

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.

CALENDARS: Dict[str, CalTypes] = {
    "all": get_named_calendar("all"),
    "bus": get_named_calendar("bus"),
    "tgt": get_named_calendar("tgt"),
    "ldn": get_named_calendar("ldn"),
    "nyc": get_named_calendar("nyc"),
    "stk": get_named_calendar("stk"),
    "osl": get_named_calendar("osl"),
    "zur": get_named_calendar("zur"),
    "tro": get_named_calendar("tro"),
}

# TODO: edit the docs for create_calendar to link to `Cal` on deprecation
# TODO: edit the example on create calendar, or move to `Cal`


def create_calendar(rules: list, week_mask: Optional[str] = None) -> Cal:
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
    weekmask = [5, 6] if week_mask is None else week_mask
    return Cal(rules, weekmask)


def get_calendar(
    calendar: CalInput, kind: bool = False
) -> Union[CalTypes, tuple[CalTypes, str]]:
    """
    Returns a calendar object either from an available set or a user defined input.

    Parameters
    ----------
    calendar : str, Cal, UnionCal
        If `str`, then the calendar is returned from pre-calculated values.
        If a specific user defined calendar this is returned without modification.
    kind : bool
        If `True` will also return the kind of calculation from `"null", "named",
        "custom"`.

    Returns
    -------
    Cal, UnionCal or tuple

    Notes
    -----

    The following named calendars are available and have been back tested against the
    publication of RFR indexes in the relevant geography.

    - *"all"*: Every day is defined as business day including weekends.
    - *"bus"*: Regular weekdays are defined as business days. Saturdays and Sunday are non-business days.
    - *"tgt"*: Target for Europe's ESTR.
    - *"osl"*: Oslo for Norway's NOWA.
    - *"zur"*: Zurich for Switzerland's SARON.
    - *"nyc"*: New York City for US's SOFR.
    - *"ldn"*: London for UK's SONIA.
    - *"stk"*: Stockholm for Sweden's SWESTR.
    - *"tro"*: Toronto for Canada's CORRA.

    The list of generic holidays applied to these calendars is as follows;

    .. list-table:: Calendar generic holidays
       :widths: 51 7 7 7 7 7 7 7
       :header-rows: 1

       * - Holiday
         - *"tgt"*
         - *"osl"*
         - *"zur"*
         - *"nyc"*
         - *"ldn"*
         - *"stk"*
         - *"tro"*
       * - New Years Day
         - X
         - X
         - X
         -
         -
         - X
         -
       * - New Years Day (sun->mon)
         -
         -
         -
         - X
         -
         -
         -
       * - New Years Day (w/e->mon)
         -
         -
         -
         -
         - X
         -
         - X
       * - Berchtoldstag
         -
         -
         - X
         -
         -
         -
         -
       * - Epiphany
         -
         -
         -
         -
         -
         - X
         -
       * - Martin Luther King Day
         -
         -
         -
         - X
         -
         -
         -
       * - President's / Family Day
         -
         -
         -
         - X
         -
         -
         - X
       * - Victoria Day
         -
         -
         -
         -
         -
         -
         - X
       * - Maundy Thursday
         -
         - X
         -
         -
         -
         -
         -
       * - Good Friday
         - X
         - X
         - X
         - X
         - X
         - X
         - X
       * - Easter Monday
         - X
         - X
         - X
         -
         - X
         - X
         -
       * - UK Early May Bank Holiday
         -
         -
         -
         -
         - X
         -
         -
       * - UK Late May Bank Holiday
         -
         -
         -
         -
         - X
         -
         -
       * - EU / US / CA Labour Day (diff)
         - X
         - X
         - X
         - X
         -
         - X
         - X
       * - US Memorial Day
         -
         -
         -
         - X
         -
         -
         -
       * - Ascention Day
         -
         - X
         - X
         -
         -
         - X
         -
       * - Whit Monday
         -
         - X
         -
         -
         -
         -
         -
       * - Midsummer Friday
         -
         -
         -
         -
         -
         - X
         -
       * - National / Constitution Day (diff)
         -
         - X
         - X
         -
         -
         - X
         - X
       * - Juneteenth National Day (sun->mon)
         -
         -
         -
         - X
         -
         -
         -
       * - US Independence Day (sat->fri,sun->mon)
         -
         -
         -
         - X
         -
         -
         -
       * - Civic Holiday
         -
         -
         -
         -
         -
         -
         - X
       * - UK Summer Bank Holiday
         -
         -
         -
         -
         - X
         -
         -
       * - Columbus Day
         -
         -
         -
         - X
         -
         -
         -
       * - US Veteran's Day (sun->mon)
         -
         -
         -
         - X
         -
         -
         -
       * - National Truth
         -
         -
         -
         -
         -
         -
         - X
       * - US / CA Thanksgiving (diff)
         -
         -
         -
         - X
         -
         -
         - X
       * - Remembrance Day
         -
         -
         -
         -
         -
         -
         - X
       * - Christmas Eve
         -
         - X
         -
         -
         -
         - X
         -
       * - Christmas Day
         - X
         - X
         - X
         -
         -
         - X
         -
       * - Christmas Day (sat,sun->mon)
         -
         -
         -
         -
         - X
         -
         - X
       * - Christmas Day (sat->fri,sun->mon)
         -
         -
         -
         - X
         -
         -
         -
       * - Boxing Day
         - X
         - X
         - X
         -
         -
         - X
         -
       * - Boxing Day (sun,mon->tue)
         -
         -
         -
         -
         - X
         -
         - X
       * - New Year's Eve
         -
         -
         -
         -
         -
         - X
         -

    Examples
    --------
    .. ipython:: python

       gbp_cal = get_calendar("ldn")
       gbp_cal.holidays
       gbp_cal.add_bus_days(dt(2023, 1, 3), 5, True)
       type(gbp_cal)

    Calendars can be combined from the pre-existing names using comma separation.

    .. ipython:: python

       gbp_and_nyc_cal = get_calendar("ldn,nyc")
       gbp_and_nyc_cal.holidays

    """
    # TODO: rename calendars or make a more generalist statement about their names.
    if calendar is NoInput.blank:
        ret = CALENDARS["all"], "null"
    elif isinstance(calendar, str):
        calendars = calendar.lower().split(",")
        if len(calendars) == 1:  # only one named calendar is found
            ret = CALENDARS[calendars[0]], "named"
        else:
            cals = [CALENDARS[_] for _ in calendars]
            ret = UnionCal(cals, None), "named"
    else:  # calendar is a Calendar object type
        ret = calendar, "custom"

    return ret if kind else ret[0]


def _get_modifier(modifier: str) -> Modifier:
    try:
        return {
            "F": Modifier.F,
            "MF": Modifier.ModF,
            "P": Modifier.P,
            "MP": Modifier.ModP,
            "NONE": Modifier.Act
        }[modifier.upper()]
    except KeyError:
        raise ValueError("`modifier` must be in {'F', 'MF', 'P', 'MP', 'NONE'}.")


def _get_rollday(roll: Union[str, int, NoInput]) -> RollDay:
    if isinstance(roll, str):
        return {
            "EOM": RollDay.EoM(),
            "SOM": RollDay.SoM(),
            "IMM": RollDay.IMM(),
        }[roll.upper()]
    elif isinstance(roll, int):
        return RollDay.Int(roll)
    return RollDay.Unspecified()


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
    return cal_.roll(date, _get_modifier(modifier), settlement)

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def add_tenor(
    start: datetime,
    tenor: str,
    modifier: str,
    calendar: CalInput = NoInput(0),
    roll: Union[str, int, NoInput] = NoInput(0),
    settlement: bool = False,
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

    Returns
    -------
    datetime

    Examples
    --------
    .. ipython:: python
       :suppress:

       from rateslib.calendars import add_tenor, get_calendar, create_calendar, dcf
       from rateslib.scheduling import Schedule
       from rateslib.curves import Curve, LineCurve, interpolate, index_left, IndexCurve
       from rateslib.dual import Dual, Dual2
       from rateslib.periods import FixedPeriod, FloatPeriod, Cashflow, IndexFixedPeriod, IndexCashflow
       from rateslib.legs import FixedLeg, FloatLeg, CustomLeg, FloatLegMtm, FixedLegMtm, IndexFixedLeg, ZeroFixedLeg, ZeroFloatLeg, ZeroIndexLeg
       from rateslib.instruments import FixedRateBond, FloatRateNote, Value, IRS, SBS, FRA, forward_fx, Spread, Fly, BondFuture, Bill, ZCS, FXSwap, ZCIS, IIRS, STIRFuture
       from rateslib.solver import Solver
       from rateslib.splines import bspldnev_single, PPSpline
       from datetime import datetime as dt
       import pandas as pd
       from pandas import date_range, Series, DataFrame
       pd.set_option("display.float_format", lambda x: '%.2f' % x)
       pd.set_option("display.max_columns", None)
       pd.set_option("display.width", 500)

    .. ipython:: python

       add_tenor(dt(2022, 2, 28), "3M", "NONE")
       add_tenor(dt(2022, 12, 28), "4b", "F", get_calendar("ldn"))
       add_tenor(dt(2022, 12, 28), "4d", "F", get_calendar("ldn"))
    """
    tenor = tenor.upper()
    cal_ = get_calendar(calendar)
    if "D" in tenor:
        return cal_.add_days(start, int(tenor[:-1]), _get_modifier(modifier), settlement)
    elif "B" in tenor:
        return cal_.add_bus_days(start, int(tenor[:-1]), settlement)
    elif "Y" in tenor:
        months = int(float(tenor[:-1]) * 12)
        return cal_.add_months(start, months, _get_modifier(modifier), _get_rollday(roll), settlement)
    elif "M" in tenor:
        return cal_.add_months(start, int(tenor[:-1]), _get_modifier(modifier), _get_rollday(roll), settlement)
    elif "W" in tenor:
        return cal_.add_days(start, int(tenor[:-1]) * 7, _get_modifier(modifier), settlement)
    else:
        raise ValueError("`tenor` must identify frequency in {'B', 'D', 'W', 'M', 'Y'} e.g. '1Y'")


def _get_roll(month: int, year: int, roll: Union[str, int]) -> datetime:
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


def _is_eom_cal(date: datetime, cal: CalInput):
    """Test whether a given date is end of month under a specific calendar"""
    udate = calendar_mod.monthrange(date.year, date.month)[1]
    udate = datetime(date.year, date.month, udate)
    aeom = _adjust_date(udate, "P", cal)
    return date == aeom


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
    termination: Union[datetime, NoInput] = NoInput(0),  # required for 30E360ISDA and ActActICMA
    frequency_months: Union[int, NoInput] = NoInput(0),  # req. ActActICMA = ActActISMA = ActActBond
    stub: Union[bool, NoInput] = NoInput(0),  # required for ActActICMA = ActActISMA = ActActBond
    roll: Union[str, int, NoInput] = NoInput(0),  # required also for ActACtICMA = ...
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
        Currently unused.

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
      * end day is minimum of (30, start day) only if start day was adjusted.

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
            "'30360' '360360', 'BondBasis', '30E360', 'EuroBondBasis', "
            "'30E360ISDA', 'ActAct', 'ActActISDA', 'ActActICMA', "
            "'ActActISMA', 'ActActBond'}"
        )


def _dcf_act365f(start: datetime, end: datetime, *args):
    return (end - start).days / 365.0


def _dcf_act365fplus(start: datetime, end: datetime, *args):
    """count the number of the years and then add a fractional ACT365F peruiod."""
    if end <= datetime(start.year + 1, start.month, start.day):
        return _dcf_act365f(start, end)
    elif end <= datetime(end.year, start.month, start.day):
        return end.year - start.year + _dcf_act365f(datetime(end.year, start.month, start.day), end)
    else:
        years = end.year - start.year - 1
        return years + _dcf_act365f(datetime(end.year - 1, start.month, start.day), end)


def _dcf_act360(start: datetime, end: datetime, *args):
    return (end - start).days / 360.0


def _dcf_30360(start: datetime, end: datetime, *args):
    ds = min(30, start.day)
    de = min(ds, end.day) if ds == 30 else end.day
    y, m = end.year - start.year, (end.month - start.month) / 12.0
    return y + m + (de - ds) / 360.0


def _dcf_30e360(start: datetime, end: datetime, *args):
    ds, de = min(30, start.day), min(30, end.day)
    y, m = end.year - start.year, (end.month - start.month) / 12.0
    return y + m + (de - ds) / 360.0


def _dcf_30e360isda(start: datetime, end: datetime, termination: Union[datetime, NoInput], *args):
    if termination is NoInput.blank:
        raise ValueError("`termination` must be supplied with specified `convention`.")

    def _is_end_feb(date):
        if date.month == 2:
            _, end_feb = calendar_mod.monthrange(date.year, 2)
            return date.day == end_feb
        return False

    ds = 30 if (start.day == 31 or _is_end_feb(start)) else start.day
    de = 30 if (end.day == 31 or (_is_end_feb(end) and end != termination)) else end.day
    y, m = end.year - start.year, (end.month - start.month) / 12.0
    return y + m + (de - ds) / 360.0


def _dcf_actactisda(start: datetime, end: datetime, *args):
    if start == end:
        return 0.0

    start_date = datetime.combine(start, datetime.min.time())
    end_date = datetime.combine(end, datetime.min.time())

    year_1_diff = 366.0 if calendar_mod.isleap(start_date.year) else 365.0
    year_2_diff = 366.0 if calendar_mod.isleap(end_date.year) else 365.0

    total_sum: float = end.year - start.year - 1
    total_sum += (datetime(start.year + 1, 1, 1) - start_date).days / year_1_diff
    total_sum += (end_date - datetime(end.year, 1, 1)).days / year_2_diff
    return total_sum


def _dcf_actacticma(
    start: datetime,
    end: datetime,
    termination: Union[datetime, NoInput],
    frequency_months: Union[int, NoInput],
    stub: Union[bool, NoInput],
    roll: Union[str, int, NoInput],
    calendar: CalInput,
):
    if frequency_months is NoInput.blank:
        raise ValueError("`frequency_months` must be supplied with specified `convention`.")
    if termination is NoInput.blank:
        raise ValueError("`termination` must be supplied with specified `convention`.")
    if stub is NoInput.blank:
        raise ValueError("`stub` must be supplied with specified `convention`.")
    if not stub and frequency_months < 13:  # This is a well defined period that is NOT zero coupon
        return frequency_months / 12.0
    else:
        # Perform stub and zero coupon calculation. Zero coupons handled with an Annual frequency.
        if frequency_months >= 13:
            warnings.warn(
                "Using `convention` 'ActActICMA' with a Period having `frequency` 'Z' is undefined, and "
                "should be avoided.\nFor calculation purposes here the `frequency` is set to 'A'.",
                UserWarning,
            )
            frequency_months = 12  # Will handle Z frequency as a stub period see GH:144

        # roll is used here to roll a negative months forward eg, 30 sep minus 6M = 30/31 March.
        cal_ = get_calendar(calendar)
        if end == termination:  # stub is a BACK stub:
            fwd_end_0, fwd_end_1, fraction = start, start, -1.0
            while (
                end > fwd_end_1
            ):  # Handle Long Stubs which require repeated periods, and Zero frequencies.
                fwd_end_0 = fwd_end_1
                fraction += 1.0
                fwd_end_1 = cal_.add_months(
                    start, (int(fraction) + 1) * frequency_months, _get_modifier("NONE"), _get_rollday(roll), False
                )

            fraction += (end - fwd_end_0) / (fwd_end_1 - fwd_end_0)
            return fraction * frequency_months / 12.0
        else:  # stub is a FRONT stub
            prev_start_0, prev_start_1, fraction = end, end, -1.0
            while (
                start < prev_start_1
            ):  # Handle Long Stubs which require repeated periods, and Zero frequencies.
                prev_start_0 = prev_start_1
                fraction += 1.0
                prev_start_1 = cal_.add_months(
                    end, -(int(fraction) + 1) * frequency_months, _get_modifier("NONE"), _get_rollday(roll), False
                )

            fraction += (prev_start_0 - start) / (prev_start_0 - prev_start_1)
            return fraction * frequency_months / 12.0


def _dcf_actacticma_stub365f(
    start: datetime,
    end: datetime,
    termination: Union[datetime, NoInput],
    frequency_months: Union[int, NoInput],
    stub: Union[bool, NoInput],
    roll: Union[str, int, NoInput],
    calendar: CalInput,
):
    """
    Applies regular actacticma unless a stub period where Act365F is used.
    [designed for Canadian Government Bonds with stubs]
    """
    if frequency_months is NoInput.blank:
        raise ValueError("`frequency_months` must be supplied with specified `convention`.")
    if termination is NoInput.blank:
        raise ValueError("`termination` must be supplied with specified `convention`.")
    if stub is NoInput.blank:
        raise ValueError("`stub` must be supplied with specified `convention`.")
    if not stub:
        return frequency_months / 12.0
    else:
        # roll is used here to roll a negative months forward eg, 30 sep minus 6M = 30/31 March.
        cal_ = get_calendar(calendar)
        if end == termination:  # stub is a BACK stub:
            fwd_end = cal_.add_months(start, frequency_months, _get_modifier("NONE"), _get_rollday(roll), False)
            r = (end - start).days
            s = (fwd_end - start).days
            if end > fwd_end:  # stub is LONG
                d_ = frequency_months / 12.0
                d_ += (r - s) / 365.0
            else:  # stub is SHORT
                if r < (365.0 / (12 / frequency_months)):
                    d_ = r / 365.0
                else:
                    d_ = frequency_months / 12 - (s - r) / 365.0

        else:  # stub is a FRONT stub
            prev_start = cal_.add_months(end, -frequency_months, _get_modifier("NONE"), _get_rollday(roll), False)
            r = (end - start).days
            s = (end - prev_start).days
            if start < prev_start:  # stub is LONG
                d_ = frequency_months / 12.0
                d_ += (r - s) / 365.0
            else:
                if r < 365.0 / (12 / frequency_months):
                    d_ = r / 365.0
                else:
                    d_ = frequency_months / 12 - (s - r) / 365.0

        return d_


def _dcf_1(*args):
    return 1.0


def _dcf_1plus(start: datetime, end: datetime, *args):
    return end.year - start.year + (end.month - start.month) / 12.0


_DCF = {
    "ACT365F": _dcf_act365f,
    "ACT365F+": _dcf_act365fplus,
    "ACT360": _dcf_act360,
    "30360": _dcf_30360,
    "360360": _dcf_30360,
    "BONDBASIS": _dcf_30360,
    "30E360": _dcf_30e360,
    "EUROBONDBASIS": _dcf_30e360,
    "30E360ISDA": _dcf_30e360isda,
    "ACTACT": _dcf_actactisda,
    "ACTACTISDA": _dcf_actactisda,
    "ACTACTICMA": _dcf_actacticma,
    "ACTACTICMA_STUB365F": _dcf_actacticma_stub365f,
    "ACTACTISMA": _dcf_actacticma,
    "ACTACTBOND": _dcf_actacticma,
    "1": _dcf_1,
    "1+": _dcf_1plus,
}

_DCF1d = {
    "ACT365F": 1.0 / 365,
    "ACT365F+": 1.0 / 365,
    "ACT360": 1.0 / 360,
    "30360": 1.0 / 365.25,
    "360360": 1.0 / 365.25,
    "BONDBASIS": 1.0 / 365.25,
    "30E360": 1.0 / 365.25,
    "EUROBONDBASIS": 1.0 / 365.25,
    "30E360ISDA": 1.0 / 365.25,
    "ACTACT": 1.0 / 365.25,
    "ACTACTISDA": 1.0 / 365.25,
    "ACTACTICMA": 1.0 / 365.25,
    "ACTACTICMA_STUB365F": 1 / 365.25,
    "ACTACTISMA": 1.0 / 365.25,
    "ACTACTBOND": 1.0 / 365.25,
    "1": None,
    "1+": None,
}

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
