from datetime import datetime

from rateslib import defaults
from rateslib.calendars import add_tenor, dcf
from rateslib.default import NoInput

"""
All functions in this module are designed to take a Bond object and return the **fraction**
of the current coupon period associated with the given settlement.

This fraction is used to assess the total accrued calculation at a subsequent stage.
"""


def _acc_linear_proportion_by_days(obj, settlement: datetime, acc_idx: int, *args):
    """
    Return the fraction of an accrual period between start and settlement.

    Method: a linear proportion of actual days between start, settlement and end.
    Measures between unadjusted coupon dates.

    This is a general method, used by many types of bonds, for example by UK Gilts,
    German Bunds.
    """
    r = settlement - obj.leg1.schedule.uschedule[acc_idx]
    s = obj.leg1.schedule.uschedule[acc_idx + 1] - obj.leg1.schedule.uschedule[acc_idx]
    return r / s


def _acc_linear_proportion_by_days_long_stub_split(
    obj,
    settlement: datetime,
    acc_idx: int,
    *args,
):
    """
    For long stub periods this splits the accrued interest into two components.
    Otherwise, returns the regular linear proportion.
    [Designed primarily for US Treasuries]
    """
    if obj.leg1.periods[acc_idx].stub:
        fm = defaults.frequency_months[obj.leg1.schedule.frequency]
        f = 12 / fm
        if obj.leg1.periods[acc_idx].dcf * f > 1:
            # long stub
            quasi_coupon = add_tenor(
                obj.leg1.schedule.uschedule[acc_idx + 1],
                f"-{fm}M",
                "NONE",
                NoInput(0),
                obj.leg1.schedule.roll,
            )
            quasi_start = add_tenor(
                quasi_coupon,
                f"-{fm}M",
                "NONE",
                NoInput(0),
                obj.leg1.schedule.roll,
            )
            if settlement <= quasi_coupon:
                # then first part of long stub
                r = quasi_coupon - settlement
                s = quasi_coupon - quasi_start
                r_ = quasi_coupon - obj.leg1.schedule.uschedule[acc_idx]
                _ = (r_ - r) / s
                return _ / (obj.leg1.periods[acc_idx].dcf * f)
            else:
                # then second part of long stub
                r = obj.leg1.schedule.uschedule[acc_idx + 1] - settlement
                s = obj.leg1.schedule.uschedule[acc_idx + 1] - quasi_coupon
                r_ = quasi_coupon - obj.leg1.schedule.uschedule[acc_idx]
                s_ = quasi_coupon - quasi_start
                _ = r_ / s_ + (s - r) / s
                return _ / (obj.leg1.periods[acc_idx].dcf * f)

    return _acc_linear_proportion_by_days(obj, settlement, acc_idx, *args)


def _acc_30e360(obj, settlement: datetime, acc_idx: int, *args):
    """
    Ignoring the convention on the leg uses "30E360" to determine the accrual fraction.
    Measures between unadjusted date and settlement.
    [Designed primarily for Swedish Government Bonds]

    If stub revert to linear proportioning.
    """
    if obj.leg1.periods[acc_idx].stub:
        return _acc_linear_proportion_by_days(obj, settlement, acc_idx)
    f = 12 / defaults.frequency_months[obj.leg1.schedule.frequency]
    _ = dcf(settlement, obj.leg1.schedule.uschedule[acc_idx + 1], "30e360") * f
    _ = 1 - _
    return _


def _acc_act365_with_1y_and_stub_adjustment(obj, settlement: datetime, acc_idx: int, *args):
    """
    Ignoring the convention on the leg uses "Act365f" to determine the accrual fraction.
    Measures between unadjusted date and settlement.
    Special adjustment if number of days is greater than 365.
    If the period is a stub reverts to a straight line interpolation
    [this is primarily designed for Canadian Government Bonds]
    """
    if obj.leg1.periods[acc_idx].stub:
        return _acc_linear_proportion_by_days(obj, settlement, acc_idx)
    f = 12 / defaults.frequency_months[obj.leg1.schedule.frequency]
    r = (settlement - obj.leg1.schedule.uschedule[acc_idx]).days
    s = (obj.leg1.schedule.uschedule[acc_idx + 1] - obj.leg1.schedule.uschedule[acc_idx]).days
    if r == s:
        _ = 1.0  # then settlement falls on the coupon date
    elif r > 365.0 / f:
        _ = 1.0 - ((s - r) * f) / 365.0  # counts remaining days
    else:
        _ = f * r / 365.0
    return _
