from datetime import datetime

from rateslib.calendars import dcf
from rateslib.dual import DualTypes

"""
The calculations for v2 (the interim, regular period discount value) are more standardised
than the other calculations because they exclude the scenarios for stub handling.
"""


def _v2_(obj, ytm: DualTypes, f: int, *args):
    """
    Default method for a single regular period discounted in the regular portion of bond.
    Implies compounding at the same frequency as the coupons.
    """
    return 1 / (1 + ytm / (100 * f))


def _v2_annual(obj, ytm: DualTypes, f: int, *args):
    """
    ytm is expressed annually but coupon payments are on another frequency
    """
    return (1 / (1 + ytm / 100)) ** (1 / f)


"""
The calculations for v1 allow more inputs in order to avoid repeat calculations in the chain.
"""


def _v1_compounded_by_remaining_accrual_fraction(
    obj,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: callable,
    *args,
):
    """
    Determine the discount factor for the first cashflow after settlement.

    The parameter "v2" is a generic discount function which is normally :math:`1/(1+y/f)`

    Method: compounds "v2" by the accrual fraction of the period.
    """
    acc_frac = accrual(obj, settlement, acc_idx)
    if obj.leg1.periods[acc_idx].stub:
        # If it is a stub then the remaining fraction must be scaled by the relative size of the
        # stub period compared with a regular period.
        fd0 = obj.leg1.periods[acc_idx].dcf * f * (1 - acc_frac)
    else:
        # 1 minus acc_fra is the fraction of the period remaining until the next cashflow.
        fd0 = 1 - acc_frac
    return v2**fd0


def _v1_compounded_by_remaining_accrual_frac_except_simple_final_period(
    obj,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: callable,
    *args,
):
    """
    Uses regular fractional compounding except if it is last period, when simple money-mkt
    yield is used instead.
    Introduced for German Bunds.
    """
    if acc_idx == obj.leg1.schedule.n_periods - 1:
        # or \
        # settlement == self.leg1.schedule.uschedule[acc_idx + 1]:
        # then settlement is in last period use simple interest.
        return _v1_simple(obj, ytm, f, settlement, acc_idx, v2, accrual, *args)
    else:
        return _v1_compounded_by_remaining_accrual_fraction(
            obj,
            ytm,
            f,
            settlement,
            acc_idx,
            v2,
            accrual,
            *args,
        )


def _v1_comp_stub_act365f(
    obj,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: callable,
    *args,
):
    """Compounds the yield. In a stub period the act365f DCF is used"""
    if not obj.leg1.periods[acc_idx].stub:
        return _v1_compounded_by_remaining_accrual_fraction(
            obj,
            ytm,
            f,
            settlement,
            acc_idx,
            v2,
            accrual,
            *args,
        )
    else:
        fd0 = dcf(settlement, obj.leg1.schedule.uschedule[acc_idx + 1], "Act365F")
        return v2**fd0


def _v1_simple(
    obj,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: callable,
    *args,
):
    """
    Use simple rates with a yield which matches the frequency of the coupon.
    """
    acc_frac = accrual(obj, settlement, acc_idx)
    if obj.leg1.periods[acc_idx].stub:
        # is a stub so must account for discounting in a different way.
        fd0 = obj.leg1.periods[acc_idx].dcf * f * (1 - acc_frac)
    else:
        fd0 = 1 - acc_frac

    v_ = 1 / (1 + fd0 * ytm / (100 * f))
    return v_


def _v1_simple_1y_adjustment(
    obj,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: callable,
    *args,
):
    """
    Use simple rates with a yield which matches the frequency of the coupon.

    If the stub period is long, then discount the regular part of the stub with the regular
    discount param ``v``.
    """
    acc_frac = accrual(obj, settlement, acc_idx)
    if obj.leg1.periods[acc_idx].stub:
        # is a stub so must account for discounting in a different way.
        fd0 = obj.leg1.periods[acc_idx].dcf * f * (1 - acc_frac)
    else:
        fd0 = 1 - acc_frac

    if fd0 > 1.0:
        v_ = v2 * 1 / (1 + (fd0 - 1) * ytm / (100 * f))
    else:
        v_ = 1 / (1 + fd0 * ytm / (100 * f))
    return v_


def _v3_compounded(
    obj,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    *args,
):
    """
    Final period uses a compounding approach where the power is determined by the DCF of that
    period under the bond's specified convention.
    """
    if obj.leg1.periods[acc_idx].stub:
        # If it is a stub then the remaining fraction must be scaled by the relative size of the
        # stub period compared with a regular period.
        fd0 = obj.leg1.periods[acc_idx].dcf * f
    else:
        fd0 = 1
    return v2**fd0


def _v3_30e360_u_simple(
    obj,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    *args,
):
    """
    The final period is discounted by a simple interest method under a 30E360 convention.

    The YTM is assumed to have the same frequency as the coupons.
    """
    d_ = dcf(obj.leg1.periods[acc_idx].start, obj.leg1.periods[acc_idx].end, "30E360")
    return 1 / (1 + d_ * ytm / 100)  # simple interest


def _v3_simple(
    obj,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: callable,
    *args,
):
    v_ = 1 / (1 + obj.leg1.periods[-2].dcf * ytm / 100.0)
    return v_
