from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from rateslib.calendars import dcf

if TYPE_CHECKING:
    from rateslib.instruments.bonds.conventions.accrued import AccrualFunction
    from rateslib.typing import BondMixin, CurveOption_, DualTypes, Security

"""
The calculations for v2 (the interim, regular period discount value) are more standardised
than the other calculations because they exclude the scenarios for stub handling.
"""


class YtmDiscountFunction(Protocol):
    # Callable Type for discount functions in YTM formula
    def __call__(
        self,
        obj: Security | BondMixin,
        ytm: DualTypes,
        f: int,
        settlement: datetime,
        acc_idx: int,
        v2: DualTypes | None,
        accrual: AccrualFunction,
        period_idx: int,
    ) -> DualTypes: ...


class YtmStubDiscountFunction(Protocol):
    # Callable Type for discount functions in YTM formula
    # This is same as above, except v2 must be pre-calculated and cannot be None
    def __call__(
        self,
        obj: Security | BondMixin,
        ytm: DualTypes,
        f: int,
        settlement: datetime,
        acc_idx: int,
        v2: DualTypes,
        accrual: AccrualFunction,
        period_idx: int,
    ) -> DualTypes: ...


class CashflowFunction(Protocol):
    # Callable Type for cashflow generation in YTM formula
    def __call__(
        self,
        obj: Security | BondMixin,
        ytm: DualTypes,
        f: int,
        acc_idx: int,
        p_idx: int,
        n: int,
        curve: CurveOption_,
    ) -> DualTypes: ...


"""
The calculations for v2:
"""


def _v2_(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes | None,
    accrual: AccrualFunction,
    period_idx: int,
) -> DualTypes:
    """
    Default method for a single regular period discounted in the regular portion of bond.
    Implies compounding at the same frequency as the coupons.
    """
    if v2 is None:
        return 1 / (1 + ytm / (100 * f))
    else:
        return v2


def _v2_annual(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes | None,
    accrual: AccrualFunction,
    period_idx: int,
) -> DualTypes:
    """
    ytm is expressed annually but coupon payments are on another frequency
    """
    if v2 is None:
        return (1 / (1 + ytm / 100)) ** (1 / f)
    else:
        return v2


def _v2_annual_pay_adjust(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes | None,
    accrual: AccrualFunction,
    period_idx: int,
) -> DualTypes:
    if v2 is None:
        # This is the initial, regular determination of v2
        return (1 / (1 + ytm / 100)) ** (1 / f)
    else:
        return v2 ** (1.0 + _pay_adj(obj, period_idx))


"""
The calculations for v1:
"""


def _v1_compounded(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: AccrualFunction,
    period_idx: int,
) -> DualTypes:
    """
    Determine the discount factor for the first cashflow after settlement.

    The parameter "v2" is a generic discount function which is normally :math:`1/(1+y/f)`

    Method: compounds "v2" with exponent in terms of the accrual fraction of the period.
    """
    acc_frac = accrual(obj, settlement, acc_idx)
    if obj.leg1.periods[acc_idx].stub:  # type: ignore[union-attr]
        # If it is a stub then the remaining fraction must be scaled by the relative size of the
        # stub period compared with a regular period.
        fd0 = obj.leg1.periods[acc_idx].dcf * f * (1 - acc_frac)  # type: ignore[union-attr]
    else:
        # 1 minus acc_fra is the fraction of the period remaining until the next cashflow.
        fd0 = 1 - acc_frac
    return v2**fd0


def _v1_simple(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: AccrualFunction,
    period_idx: int,
) -> DualTypes:
    """
    Use simple rates with a yield which matches the frequency of the coupon.
    """
    acc_frac = accrual(obj, settlement, acc_idx)
    if obj.leg1.periods[acc_idx].stub:  # type: ignore[union-attr]
        # is a stub so must account for discounting in a different way.
        fd0 = obj.leg1.periods[acc_idx].dcf * f * (1 - acc_frac)  # type: ignore[union-attr]
    else:
        fd0 = 1 - acc_frac

    v_ = 1 / (1 + fd0 * ytm / (100 * f))
    return v_


def _v1_simple_pay_adjust(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: AccrualFunction,
    period_idx: int,
) -> DualTypes:
    acc_frac = accrual(obj, settlement, acc_idx)
    if obj.leg1.periods[acc_idx].stub:  # type: ignore[union-attr]
        # is a stub so must account for discounting in a different way.
        fd0 = obj.leg1.periods[acc_idx].dcf * f * (1 - acc_frac + _pay_adj(obj, period_idx))  # type: ignore[union-attr]
    else:
        fd0 = 1 - acc_frac + _pay_adj(obj, period_idx)

    v_ = 1 / (1 + fd0 * ytm / (100 * f))
    return v_


def _v1_compounded_pay_adjust(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: AccrualFunction,
    period_idx: int,
) -> DualTypes:
    acc_frac = accrual(obj, settlement, acc_idx)
    if obj.leg1.periods[acc_idx].stub:  # type: ignore[union-attr]
        # If it is a stub then the remaining fraction must be scaled by the relative size of the
        # stub period compared with a regular period.
        fd0 = obj.leg1.periods[acc_idx].dcf * f * (1 - acc_frac + _pay_adj(obj, period_idx))  # type: ignore[union-attr]
    else:
        # 1 minus acc_fra is the fraction of the period remaining until the next cashflow.
        fd0 = 1 - acc_frac + _pay_adj(obj, period_idx)
    return v2**fd0


def _v1_compounded_final_simple(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: AccrualFunction,
    period_idx: int,
) -> DualTypes:
    """
    Uses regular fractional compounding except if it is last period, when simple money-mkt
    yield is used instead.
    Introduced for German Bunds.
    """
    if acc_idx == obj.leg1.schedule.n_periods - 1:
        # or \
        # settlement == self.leg1.schedule.uschedule[acc_idx + 1]:
        # then settlement is in last period use simple interest.
        return _v1_simple(obj, ytm, f, settlement, acc_idx, v2, accrual, period_idx)
    else:
        return _v1_compounded(obj, ytm, f, settlement, acc_idx, v2, accrual, period_idx)


def _v1_compounded_final_simple_pay_adjust(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: AccrualFunction,
    period_idx: int,
) -> DualTypes:
    """
    Uses regular fractional compounding except if it is last period, when simple money-mkt
    yield is used instead.
    Both methods are adjusted to account for pay delays.
    """
    if acc_idx == obj.leg1.schedule.n_periods - 1:
        return _v1_simple_pay_adjust(obj, ytm, f, settlement, acc_idx, v2, accrual, period_idx)
    else:
        # Pay adjustment is not applied if it is not the final period
        return _v1_compounded(obj, ytm, f, settlement, acc_idx, v2, accrual, period_idx)


def _v1_comp_stub_act365f(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: AccrualFunction,
    period_idx: int,
) -> DualTypes:
    """Compounds the yield. In a stub period the act365f DCF is used"""
    if not obj.leg1.periods[acc_idx].stub:  # type: ignore[union-attr]
        return _v1_compounded(obj, ytm, f, settlement, acc_idx, v2, accrual, period_idx)
    else:
        fd0 = dcf(settlement, obj.leg1.schedule.uschedule[acc_idx + 1], "Act365F")
        return v2**fd0


def _v1_simple_long_stub(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: AccrualFunction,
    period_idx: int,
) -> DualTypes:
    """
    Use simple rates with a yield which matches the frequency of the coupon.

    If the stub period is long, then discount the regular part of the stub with the regular
    discount param ``v``.
    """
    if obj.leg1.periods[acc_idx].stub and obj.leg1.periods[acc_idx].dcf * f > 1:  # type: ignore[union-attr]
        # long stub
        acc_frac = accrual(obj, settlement, acc_idx)
        fd0 = obj.leg1.periods[acc_idx].dcf * f * (1 - acc_frac)  # type: ignore[union-attr]
        if fd0 > 1.0:
            # then there is a whole quasi-coupon period until payment of next cashflow
            v_ = v2 * 1 / (1 + (fd0 - 1) * ytm / (100 * f))
        else:
            # this is standard _v1_simple formula
            v_ = 1 / (1 + fd0 * ytm / (100 * f))
        return v_
    else:
        return _v1_simple(obj, ytm, f, settlement, acc_idx, v2, accrual, period_idx)


"""
The calculations for v3:
"""


def _v3_compounded(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: AccrualFunction,
    period_idx: int,
) -> DualTypes:
    """
    Final period uses a compounding approach where the power is determined by the DCF of that
    period under the bond's specified convention.
    """
    if obj.leg1.periods[acc_idx].stub:  # type: ignore[union-attr]
        # If it is a stub then the remaining fraction must be scaled by the relative size of the
        # stub period compared with a regular period.
        fd0 = obj.leg1.periods[acc_idx].dcf * f  # type: ignore[union-attr]
    else:
        fd0 = 1
    return v2**fd0


def _v3_compounded_pay_adjust(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: AccrualFunction,
    period_idx: int,
) -> DualTypes:
    """
    Final period uses a compounding approach where the power is determined by the DCF of that
    period under the bond's specified convention.
    """
    regular_v3 = _v3_compounded(obj, ytm, f, settlement, acc_idx, v2, accrual, period_idx)
    return regular_v3 ** (1.0 + _pay_adj(obj, period_idx))


def _v3_30e360_u_simple(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: AccrualFunction,
    period_idx: int,
) -> DualTypes:
    """
    The final period is discounted by a simple interest method under a 30E360 convention.

    The YTM is assumed to have the same frequency as the coupons.
    """
    d_ = dcf(obj.leg1.periods[acc_idx].start, obj.leg1.periods[acc_idx].end, "30E360")  # type: ignore[union-attr]
    return 1 / (1 + d_ * ytm / 100)  # simple interest


def _v3_simple(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: AccrualFunction,
    period_idx: int,
) -> DualTypes:
    if obj.leg1.periods[acc_idx].stub:  # type: ignore[union-attr]
        # is a stub so must account for discounting in a different way.
        fd0 = obj.leg1.periods[acc_idx].dcf * f  # type: ignore[union-attr]
    else:
        fd0 = 1.0

    v_ = 1 / (1 + fd0 * ytm / (100 * f))
    return v_


def _v3_simple_pay_adjust(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v2: DualTypes,
    accrual: AccrualFunction,
    period_idx: int,
) -> DualTypes:
    if obj.leg1.periods[acc_idx].stub:  # type: ignore[union-attr]
        # is a stub so must account for discounting in a different way.
        fd0 = (1.0 + _pay_adj(obj, period_idx)) * obj.leg1.periods[acc_idx].dcf * f  # type: ignore[union-attr]
    else:
        fd0 = 1.0 + _pay_adj(obj, period_idx)

    v_ = 1 / (1 + fd0 * ytm / (100 * f))
    return v_


V1_FUNCS: dict[str, YtmStubDiscountFunction] = {
    "compounding": _v1_compounded,
    "compounding_pay_adjust": _v1_compounded_pay_adjust,
    "simple": _v1_simple,
    "simple_pay_adjust": _v1_simple_pay_adjust,
    "compounding_final_simple": _v1_compounded_final_simple,
    "compounding_final_simple_pay_adjust": _v1_compounded_final_simple_pay_adjust,  # noqa: E501
    "compounding_stub_act365f": _v1_comp_stub_act365f,
    "simple_long_stub_compounding": _v1_simple_long_stub,
}

V2_FUNCS: dict[str, YtmDiscountFunction] = {
    "regular": _v2_,
    "annual": _v2_annual,
    "annual_pay_adjust": _v2_annual_pay_adjust,
}

V3_FUNCS: dict[str, YtmStubDiscountFunction] = {
    "compounding": _v3_compounded,
    "compounding_pay_adjust": _v3_compounded_pay_adjust,
    "simple": _v3_simple,
    "simple_pay_adjust": _v3_simple_pay_adjust,
    "simple_30e360": _v3_30e360_u_simple,
}


def _c_from_obj(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    acc_idx: int,
    p_idx: int,
    n: int,
    curve: CurveOption_,
) -> DualTypes:
    """
    Return the cashflow as it has been calculated directly on the object according to the
    native schedule and conventions, for the specified period index.
    """
    return obj._period_cashflow(obj.leg1._regular_periods[p_idx], curve)  # type: ignore[arg-type]


def _c_full_coupon(
    obj: Security | BondMixin,
    ytm: DualTypes,
    f: int,
    acc_idx: int,
    p_idx: int,
    n: int,
    curve: CurveOption_,
) -> DualTypes:
    """
    Ignore the native schedule and conventions and return an amount based on the period
    notional, the bond coupon, and the bond frequency.
    """
    return -obj.leg1._regular_periods[p_idx].notional * obj.fixed_rate / (100 * f)  # type: ignore[operator, union-attr]


C_FUNCS: dict[str, CashflowFunction] = {
    "cashflow": _c_from_obj,
    "full_coupon": _c_full_coupon,
}


def _pay_adj(obj: Security | BondMixin, period_idx: int) -> float:
    sch = obj.leg1.schedule
    pd = (sch.pschedule[period_idx + 1] - sch.uschedule[period_idx + 1]).days
    PD = (sch.pschedule[period_idx + 1] - sch.pschedule[period_idx]).days
    return pd / PD
