from __future__ import annotations

import calendar as calendar_mod
import warnings
from datetime import datetime

from rateslib.calendars.rs import CalInput, _get_modifier, _get_rollday, get_calendar
from rateslib.default import NoInput
from rateslib.rs import Convention

CONVENTIONS_MAP = {
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
    "ACTACTICMA_STUB365F": "should panic",
    "ACTACTISMA": Convention.ActActICMA,
    "ACTACTBOND": Convention.ActActICMA,
    "1": Convention.One,
    "1+": Convention.OnePlus,
    "BUS252": Convention.Bus252,
}


def _get_convention(convention: str) -> Convention:
    try:
        return CONVENTIONS_MAP[convention.upper()]
    except KeyError:
        raise ValueError(f"`convention`: {convention}, is not valid.")


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


def _dcf_30e360isda(start: datetime, end: datetime, termination: datetime | NoInput, *args):
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
    termination: datetime | NoInput,
    frequency_months: int | NoInput,
    stub: bool | NoInput,
    roll: str | int | NoInput,
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
                    _get_modifier("NONE", True),
                    _get_rollday(roll),
                    False,
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
                    _get_modifier("NONE", True),
                    _get_rollday(roll),
                    False,
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
            fwd_end = cal_.add_months(
                start, frequency_months, _get_modifier("NONE", True), _get_rollday(roll), False
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
                end, -frequency_months, _get_modifier("NONE", True), _get_rollday(roll), False
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


def _dcf_1(*args):
    return 1.0


def _dcf_1plus(start: datetime, end: datetime, *args):
    return end.year - start.year + (end.month - start.month) / 12.0


def _dcf_bus252(
    start: datetime,
    end: datetime,
    termination: datetime | NoInput,
    frequency_months: int | NoInput,
    stub: bool | NoInput,
    roll: str | int | NoInput,
    calendar: CalInput,
):
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
    start_ = cal_.roll(start, _get_modifier("F", True), False)
    end_ = cal_.roll(end, _get_modifier("P", True), False)
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
    "BUS252": _dcf_bus252,
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
    "BUS252": 1.0 / 252,
}

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
