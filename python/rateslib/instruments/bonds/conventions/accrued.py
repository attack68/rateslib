from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from rateslib import defaults
from rateslib.calendars import add_tenor, dcf
from rateslib.default import NoInput

if TYPE_CHECKING:
    from rateslib.typing import Any, BondMixin, Security

"""
All functions in this module are designed to take a Bond object and return the **fraction**
of the current coupon period associated with the given settlement.

This fraction is used to assess the total accrued calculation at a subsequent stage.
"""


class AccrualFunction(Protocol):
    # Callable type for Accrual Functions
    def __call__(
        self, obj: Security | BondMixin, settlement: datetime, acc_idx: int, *args: Any
    ) -> float: ...


def _acc_linear_proportion_by_days(
    obj: Security | BondMixin, settlement: datetime, acc_idx: int, *args: Any
) -> float:
    """
    Return the fraction of an accrual period between start and settlement.

    Method: a linear proportion of actual days between start, settlement and end.
    Measures between unadjusted coupon dates.

    This is a general method, used by many types of bonds, for example by UK Gilts,
    German Bunds.
    """
    r = (settlement - obj.leg1.schedule.uschedule[acc_idx]).days
    s = (obj.leg1.schedule.uschedule[acc_idx + 1] - obj.leg1.schedule.uschedule[acc_idx]).days
    return float(r / s)


def _acc_linear_proportion_by_days_long_stub_split(
    obj: Security | BondMixin,
    settlement: datetime,
    acc_idx: int,
    *args: Any,
) -> float:
    """
    For long stub periods this splits the accrued interest into two components.
    Otherwise, returns the regular linear proportion.
    [Designed primarily for US Treasuries]
    """
    # TODO: handle this union attribute by segregating Securities periods into different
    # categories, perhaps when also integrating deterministic amortised bonds.
    if obj.leg1.periods[acc_idx].stub:  # type: ignore[union-attr]
        fm = defaults.frequency_months[obj.leg1.schedule.frequency]
        f = 12 / fm
        if obj.leg1.periods[acc_idx].dcf * f > 1:  # type: ignore[union-attr]
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

            s_bar_u = (quasi_coupon - quasi_start).days
            if settlement <= quasi_coupon:
                # then first part of long stub
                r_bar_u = (settlement - obj.leg1.schedule.uschedule[acc_idx]).days
                r_u = 0.0
                s_u = 1.0
            else:
                # then second part of long stub
                r_u = (settlement - quasi_coupon).days
                s_u = (obj.leg1.schedule.uschedule[acc_idx + 1] - quasi_coupon).days
                r_bar_u = (quasi_coupon - obj.leg1.schedule.uschedule[acc_idx]).days

            return (r_bar_u / s_bar_u + r_u / s_u) / (obj.leg1.periods[acc_idx].dcf * f)  # type: ignore[union-attr]

    return _acc_linear_proportion_by_days(obj, settlement, acc_idx, *args)


def _acc_30e360_backward(
    obj: Security | BondMixin, settlement: datetime, acc_idx: int, *args: Any
) -> float:
    """
    Ignoring the convention on the leg uses "30E360" to determine the accrual fraction.
    Measures between unadjusted date and settlement.
    [Designed primarily for Swedish Government Bonds]

    If stub revert to linear proportioning.
    """
    if obj.leg1.periods[acc_idx].stub:  # type: ignore[union-attr]
        return _acc_linear_proportion_by_days(obj, settlement, acc_idx)
    f = 12 / defaults.frequency_months[obj.leg1.schedule.frequency]
    _: float = dcf(settlement, obj.leg1.schedule.uschedule[acc_idx + 1], "30e360") * f
    _ = 1 - _
    return _


def _acc_30u360_forward(
    obj: Security | BondMixin, settlement: datetime, acc_idx: int, *args: Any
) -> float:
    """
    Ignoring the convention on the leg uses "30U360" to determine the accrual fraction.
    Measures between unadjusted dates and settlement.
    [Designed primarily for US Corporate/Muni Bonds]
    """
    sch = obj.leg1.schedule
    accrued = dcf(sch.uschedule[acc_idx], settlement, "30u360")
    period = dcf(sch.uschedule[acc_idx], sch.uschedule[acc_idx + 1], "30u360")
    return accrued / period


def _acc_act365_with_1y_and_stub_adjustment(
    obj: Security | BondMixin, settlement: datetime, acc_idx: int, *args: Any
) -> float:
    """
    Ignoring the convention on the leg uses "Act365f" to determine the accrual fraction.
    Measures between unadjusted date and settlement.
    Special adjustment if number of days is greater than 365.
    If the period is a stub reverts to a straight line interpolation
    [this is primarily designed for Canadian Government Bonds]
    """
    if obj.leg1.periods[acc_idx].stub:  # type: ignore[union-attr]
        return _acc_linear_proportion_by_days(obj, settlement, acc_idx)
    f = 12 / defaults.frequency_months[obj.leg1.schedule.frequency]
    r = (settlement - obj.leg1.schedule.uschedule[acc_idx]).days
    s = (obj.leg1.schedule.uschedule[acc_idx + 1] - obj.leg1.schedule.uschedule[acc_idx]).days
    if r == s:
        _: float = 1.0  # then settlement falls on the coupon date
    elif r >= 365.0 / f:
        _ = 1.0 - ((s - r) * f) / 365.0  # counts remaining days
    else:
        _ = f * r / 365.0
    return _


ACC_FRAC_FUNCS: dict[str, AccrualFunction] = {
    "linear_days": _acc_linear_proportion_by_days,
    "linear_days_long_front_split": _acc_linear_proportion_by_days_long_stub_split,
    "30e360_backward": _acc_30e360_backward,
    "30u360_forward": _acc_30u360_forward,
    "act365f_1y": _acc_act365_with_1y_and_stub_adjustment,
}
