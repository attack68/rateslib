from __future__ import annotations

import calendar as calendar_mod
import warnings
from datetime import datetime
from typing import TYPE_CHECKING

from rateslib.default import NoInput
from rateslib.rs import Adjuster, Convention, RollDay
from rateslib.scheduling.calendars import get_calendar
from rateslib.scheduling.rollday import _get_rollday

if TYPE_CHECKING:
    from rateslib.typing import Any, CalInput, Callable, bool_, datetime_, int_


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
    :download:`2006 ISDA definitions <https://www.rbccm.com/assets/rbccm/docs/legal/doddfrank/Documents/ISDALibrary/2006%20ISDA%20Definitions.pdf>` and
    :download:`2006 ISDA 30360 example <_static/30360isda_2006_example.xls>`.

    Examples
    --------
    .. ipython:: python
       :suppress:

       from rateslib import dcf

    .. ipython:: python

       dcf(dt(2000, 1, 1), dt(2000, 4, 3), "Act360")
       dcf(dt(2000, 1, 1), dt(2000, 4, 3), "Act365f")
       dcf(dt(2000, 1, 1), dt(2000, 4, 3), "ActActICMA", dt(2010, 1, 1), 3, False)
       dcf(dt(2000, 1, 1), dt(2000, 4, 3), "ActActICMA", dt(2010, 1, 1), 3, True)

    """  # noqa: E501
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


CONVENTIONS_MAP: dict[str, Convention] = {
    "ACT365F": Convention.Act365F,
    "ACT365F+": Convention.Act365FPlus,
    "ACT360": Convention.Act360,
    "30360": Convention.Thirty360,
    "360360": Convention.Thirty360,
    "BONDBASIS": Convention.Thirty360,
    "30E360": Convention.ThirtyE360,
    "EUROBONDBASIS": Convention.ThirtyE360,
    "30E360ISDA": Convention.Thirty360ISDA,
    "ACTACT": Convention.ActActISDA,
    "ACTACTISDA": Convention.ActActISDA,
    "ACTACTICMA": Convention.ActActICMA,
    # "ACTACTICMA_STUB365F": "should panic",
    "ACTACTISMA": Convention.ActActICMA,
    "ACTACTBOND": Convention.ActActICMA,
    "1": Convention.One,
    "1+": Convention.OnePlus,
    "BUS252": Convention.Bus252,
}


def _is_end_feb(date: datetime) -> bool:
    if date.month == 2:
        _, end_feb = calendar_mod.monthrange(date.year, 2)
        return date.day == end_feb
    return False


def _get_convention(convention: str) -> Convention:
    """Convert a user str input into a Convention enum."""
    try:
        return CONVENTIONS_MAP[convention.upper()]
    except KeyError:
        raise ValueError(f"`convention`: {convention}, is not valid.")


def _dcf_act365f(start: datetime, end: datetime, *args: Any) -> float:
    return (end - start).days / 365.0


def _dcf_act365fplus(start: datetime, end: datetime, *args: Any) -> float:
    """count the number of the years and then add a fractional ACT365F period."""
    if end <= datetime(start.year + 1, start.month, start.day):
        return _dcf_act365f(start, end)
    elif end <= datetime(end.year, start.month, start.day):
        return end.year - start.year + _dcf_act365f(datetime(end.year, start.month, start.day), end)
    else:
        years = end.year - start.year - 1
        return years + _dcf_act365f(datetime(end.year - 1, start.month, start.day), end)


def _dcf_act360(start: datetime, end: datetime, *args: Any) -> float:
    return (end - start).days / 360.0


def _dcf_30360(start: datetime, end: datetime, *args: Any) -> float:
    ds = min(30, start.day)
    de = min(ds, end.day) if ds == 30 else end.day
    y, m = end.year - start.year, (end.month - start.month) / 12.0
    return y + m + (de - ds) / 360.0


def _dcf_30u360(
    start: datetime,
    end: datetime,
    termination: datetime | NoInput,
    frequency_months: int | NoInput,
    stub: bool | NoInput,
    roll: str | int | NoInput,
    calendar: CalInput,
) -> float:
    """
    Date adjustment rules (more than one may take effect; apply them in order, and if a date is
    changed in one rule the changed value is used in the following rules):

    - If the investment is EOM and (Date1 is the last day of February) and (Date2 is the last day
      of February), then change D2 to 30.
    - If the investment is EOM and (Date1 is the last day of February), then change D1 to 30.
    - If D2 is 31 and D1 is 30 or 31, then change D2 to 30.
    - If D1 is 31, then change D1 to 30.

    """
    roll_day = _get_rollday(roll)
    _is_eom = roll_day == RollDay.Day(31)

    ds, de = start.day, end.day
    if _is_eom and _is_end_feb(start):
        ds = 30
        if _is_end_feb(end):
            de = 30

    if ds == 31:
        ds = 30

    if de == 31 and ds == 30:
        de = 30

    y, m = end.year - start.year, (end.month - start.month) / 12.0
    return y + m + (de - ds) / 360.0


def _dcf_30e360(start: datetime, end: datetime, *args: Any) -> float:
    ds, de = min(30, start.day), min(30, end.day)
    y, m = end.year - start.year, (end.month - start.month) / 12.0
    return y + m + (de - ds) / 360.0


def _dcf_30e360isda(
    start: datetime,
    end: datetime,
    termination: datetime | NoInput,
    *args: Any,
) -> float:
    if isinstance(termination, NoInput):
        raise ValueError("`termination` must be supplied with specified `convention`.")

    ds = 30 if (start.day == 31 or _is_end_feb(start)) else start.day
    de = 30 if (end.day == 31 or (_is_end_feb(end) and end != termination)) else end.day
    y, m = end.year - start.year, (end.month - start.month) / 12.0
    return y + m + (de - ds) / 360.0


def _dcf_actactisda(start: datetime, end: datetime, *args: Any) -> float:
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
    termination: datetime | NoInput,
    frequency_months: int | NoInput,
    stub: bool | NoInput,
    roll: str | int | NoInput,
    calendar: CalInput,
) -> float:
    if isinstance(termination, NoInput):
        raise ValueError("`termination` must be supplied with specified `convention`.")
    if isinstance(stub, NoInput):
        raise ValueError("`stub` must be supplied with specified `convention`.")
    if isinstance(frequency_months, NoInput):
        raise ValueError("`frequency_months` must be supplied with specified `convention`.")
    elif (
        not stub and frequency_months < 13
    ):  # This is a well defined period that is NOT zero coupon
        return frequency_months / 12.0
    else:
        # Perform stub and zero coupon calculation. Zero coupons handled with an Annual frequency.
        if frequency_months >= 13:
            warnings.warn(
                "Using `convention` 'ActActICMA' with a Period having `frequency` 'Z' is "
                "undefined, and should be avoided.\n"
                "For calculation purposes here the `frequency` is set to 'A'.",
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
                    start,
                    (int(fraction) + 1) * frequency_months,
                    Adjuster.Actual(),
                    _get_rollday(roll),
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
                    end,
                    -(int(fraction) + 1) * frequency_months,
                    Adjuster.Actual(),
                    _get_rollday(roll),
                )

            fraction += (prev_start_0 - start) / (prev_start_0 - prev_start_1)
            return fraction * frequency_months / 12.0


def _dcf_actacticma_stub365f(
    start: datetime,
    end: datetime,
    termination: datetime | NoInput,
    frequency_months: int | NoInput,
    stub: bool | NoInput,
    roll: str | int | NoInput,
    calendar: CalInput,
) -> float:
    """
    Applies regular actacticma unless a stub period where Act365F is used.
    [designed for Canadian Government Bonds with stubs]
    """
    if isinstance(termination, NoInput):
        raise ValueError("`termination` must be supplied with specified `convention`.")
    if isinstance(stub, NoInput):
        raise ValueError("`stub` must be supplied with specified `convention`.")
    if isinstance(frequency_months, NoInput):
        raise ValueError("`frequency_months` must be supplied with specified `convention`.")
    elif not stub:
        return frequency_months / 12.0
    else:
        # roll is used here to roll a negative months forward eg, 30 sep minus 6M = 30/31 March.
        cal_ = get_calendar(calendar)
        if end == termination:  # stub is a BACK stub:
            fwd_end = cal_.add_months(
                start,
                frequency_months,
                Adjuster.Actual(),
                _get_rollday(roll),
            )
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
            prev_start = cal_.add_months(
                end,
                -frequency_months,
                Adjuster.Actual(),
                _get_rollday(roll),
            )
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


def _dcf_1(*args: Any) -> float:
    return 1.0


def _dcf_1plus(start: datetime, end: datetime, *args: Any) -> float:
    return end.year - start.year + (end.month - start.month) / 12.0


def _dcf_bus252(
    start: datetime,
    end: datetime,
    termination: datetime | NoInput,
    frequency_months: int | NoInput,
    stub: bool | NoInput,
    roll: str | int | NoInput,
    calendar: CalInput,
) -> float:
    """
    Counts the number of business days in a range and divides by 252
    [designed for Brazilian interest rate swaps]

    Start will be included if it is a business day
    If start is not a business day it will be rolled and then included.

    End will not be included if it is a business day
    The rolled end will be included if end is not a business day
    """
    if end < start:
        raise ValueError("Cannot return negative DCF for `end` before `start`.")
    elif end == start:
        return 0.0

    cal_ = get_calendar(calendar)
    start_ = cal_.adjust(start, Adjuster.Following())
    end_ = cal_.adjust(end, Adjuster.Previous())
    subtract = -1.0 if end_ == end else 0.0
    if start_ == end_:
        if start_ > start and end_ < end:
            # then logically there is one b.d. between the non-business start and non-business end
            return 1.0 / 252.0
        elif end_ < end:
            # then the business start is permitted to the calculation until the non-business end
            return 1.0 / 252.0
        elif start_ > start:
            # then the business end is not permitted to have occurred and non-business start
            # does not count
            return 0.0
    elif start_ > end_:
        # there are no business days in between start and end
        return 0.0
    dr = cal_.bus_date_range(start_, end_)
    return (len(dr) + subtract) / 252.0


_DCF: dict[str, Callable[..., float]] = {
    "ACT365F": _dcf_act365f,
    "ACT365F+": _dcf_act365fplus,
    "ACT360": _dcf_act360,
    "30360": _dcf_30360,
    "360360": _dcf_30360,
    "BONDBASIS": _dcf_30360,
    "30U360": _dcf_30u360,
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
    "BUS252": _dcf_bus252,
}


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
