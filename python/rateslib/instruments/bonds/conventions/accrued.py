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

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from rateslib.scheduling import dcf

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        _SupportsFixedFloatLeg1,
    )

"""
All functions in this module are designed to take a Bond object and return the **fraction**
of the current coupon period associated with the given settlement.

This fraction is used to assess the total accrued calculation at a subsequent stage.
"""


class AccrualFunction(Protocol):
    # Callable type for Accrual Functions
    def __call__(
        self, obj: _SupportsFixedFloatLeg1, settlement: datetime, acc_idx: int, *args: Any
    ) -> float: ...


def _acc_linear_proportion_by_days(
    obj: _SupportsFixedFloatLeg1, settlement: datetime, acc_idx: int, *args: Any
) -> float:
    """
    Return the fraction of an accrual period between start and settlement.

    Method: a linear proportion of actual days between start, settlement and end.
    Measures between unadjusted coupon dates.

    This is a general method, used by many types of bonds, for example by UK Gilts,
    German Bunds.
    """
    r = (settlement - obj.leg1.schedule.aschedule[acc_idx]).days
    s = (obj.leg1.schedule.aschedule[acc_idx + 1] - obj.leg1.schedule.aschedule[acc_idx]).days
    return float(r / s)


def _acc_linear_proportion_by_days_long_stub_split(
    obj: _SupportsFixedFloatLeg1,
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
    if obj.leg1._regular_periods[acc_idx].period_params.stub:
        f = obj.leg1.schedule.periods_per_annum
        freq = obj.leg1.schedule.frequency_obj
        adjuster = obj.leg1.schedule.accrual_adjuster
        calendar = obj.leg1.schedule.calendar

        if obj.leg1._regular_periods[acc_idx].period_params.dcf * f > 1:
            # long stub

            if acc_idx > 0:
                # then stub is implied to be at the back, must roll forwards
                ustart = obj.leg1.schedule.uschedule[acc_idx]
                astart = obj.leg1.schedule.aschedule[acc_idx]
                quasi_ucoupon = freq.unext(ustart)
                quasi_acoupon = adjuster.adjust(quasi_ucoupon, calendar)
                quasi_uend = freq.unext(quasi_ucoupon)
                quasi_aend = adjuster.adjust(quasi_uend, calendar)
                s_bar_u = (quasi_acoupon - astart).days

                if settlement <= quasi_acoupon:
                    #
                    # |--------------------------|-----------------|---------|
                    # s                     *    qc                e         qe
                    # <-----------s_bar_u-------->
                    # <---r_bar_u----------->                        ==>  (r_bar_u / s_bar_u) / (df)
                    r_bar_u = (settlement - astart).days
                    r_u = 0.0
                    s_u = 1.0
                else:
                    #
                    # |--------------------------|-----------------|---------|
                    # s                          qc             *  e         qe
                    # <-----------s_bar_u--------><------s_u----------------->
                    # <--------r_bar_u-----------><----r_u------>
                    #                                    ==>  (r_bar_u / s_bar_u + r_u / s_u) / (df)
                    r_u = (settlement - quasi_acoupon).days
                    s_u = (quasi_aend - quasi_acoupon).days
                    r_bar_u = (quasi_acoupon - astart).days
            else:
                # then stub is implied to be at the front, must roll backwards
                uend = obj.leg1.schedule.uschedule[acc_idx + 1]
                aend = obj.leg1.schedule.aschedule[acc_idx + 1]
                quasi_ucoupon = freq.uprevious(uend)
                quasi_acoupon = adjuster.adjust(quasi_ucoupon, calendar)
                quasi_ustart = freq.uprevious(quasi_ucoupon)
                quasi_astart = adjuster.adjust(quasi_ustart, calendar)
                s_bar_u = (quasi_acoupon - quasi_astart).days

                if settlement <= quasi_acoupon:
                    #
                    # |--------|-------------------|--------------------------|
                    # qs       s             *     qc                         e
                    # <-----------s_bar_u--------->
                    #          <---r_bar_u--->                       ==>  (r_bar_u / s_bar_u) / (df)
                    r_bar_u = (settlement - obj.leg1.schedule.aschedule[acc_idx]).days
                    r_u = 0.0
                    s_u = 1.0
                else:
                    #
                    # |--------|-------------------|--------------------------|
                    # qs       s                   qc             *           e
                    # <-----------s_bar_u---------><------------s_u----------->
                    #          <-------r_bar_u----><------r_u----->
                    #
                    #                                    ==>  (r_bar_u / s_bar_u + r_u / s_u) / (df)
                    r_u = (settlement - quasi_acoupon).days
                    s_u = (aend - quasi_acoupon).days
                    r_bar_u = (quasi_acoupon - obj.leg1.schedule.aschedule[acc_idx]).days

            return (r_bar_u / s_bar_u + r_u / s_u) / (
                obj.leg1._regular_periods[acc_idx].period_params.dcf * f
            )

    return _acc_linear_proportion_by_days(obj, settlement, acc_idx, *args)


def _acc_30e360_backward(
    obj: _SupportsFixedFloatLeg1, settlement: datetime, acc_idx: int, *args: Any
) -> float:
    """
    Ignoring the convention on the leg uses "30E360" to determine the accrual fraction.
    Measures between unadjusted date and settlement.
    [Designed primarily for Swedish Government Bonds]

    If stub revert to linear proportioning.
    """
    if obj.leg1._regular_periods[acc_idx].period_params.stub:
        return _acc_linear_proportion_by_days(obj, settlement, acc_idx)
    f = obj.leg1.schedule.periods_per_annum
    _: float = (
        dcf(
            start=settlement,
            end=obj.leg1.schedule.aschedule[acc_idx + 1],
            convention="30e360",
            frequency=obj.leg1.schedule.frequency_obj,
        )
        * f
    )
    _ = 1 - _
    return _


def _acc_30u360_forward(
    obj: _SupportsFixedFloatLeg1, settlement: datetime, acc_idx: int, *args: Any
) -> float:
    """
    Ignoring the convention on the leg uses "30U360" to determine the accrual fraction.
    Measures between unadjusted dates and settlement.
    [Designed primarily for US Corporate/Muni Bonds]
    """
    sch = obj.leg1.schedule
    accrued = dcf(
        start=sch.aschedule[acc_idx],
        end=settlement,
        convention="30u360",
        frequency=sch.frequency_obj,
    )
    period = dcf(
        start=sch.aschedule[acc_idx],
        end=sch.aschedule[acc_idx + 1],
        convention="30u360",
        frequency=sch.frequency_obj,
    )
    return accrued / period


def _acc_act365_with_1y_and_stub_adjustment(
    obj: _SupportsFixedFloatLeg1, settlement: datetime, acc_idx: int, *args: Any
) -> float:
    """
    Ignoring the convention on the leg uses "Act365f" to determine the accrual fraction.
    Measures between unadjusted date and settlement.
    Special adjustment if number of days is greater than 365.
    If the period is a stub reverts to a straight line interpolation
    [this is primarily designed for Canadian Government Bonds]
    """
    if obj.leg1._regular_periods[acc_idx].period_params.stub:
        return _acc_linear_proportion_by_days(obj, settlement, acc_idx)
    f = obj.leg1.schedule.periods_per_annum
    r = (settlement - obj.leg1.schedule.aschedule[acc_idx]).days
    s = (obj.leg1.schedule.aschedule[acc_idx + 1] - obj.leg1.schedule.aschedule[acc_idx]).days
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
