from rateslib.dual import DualTypes
from datetime import datetime

def _v1_compounded_by_remaining_accrual_fraction(
        obj,
        ytm: DualTypes,
        f: int,
        settlement: datetime,
        acc_idx: int,
        v: DualTypes,
        accrual: callable,
        *args,
):
    """
    Determine the discount factor for the first cashflow after settlement.

    The parameter "v" is a generic discount function which is normally :math:`1/(1+y/f)`

    Method: compounds "v" by the accrual fraction of the period.
    """
    acc_frac = accrual(settlement, acc_idx)
    if obj.leg1.periods[acc_idx].stub:
        # If it is a stub then the remaining fraction must be scaled by the relative size of the
        # stub period compared with a regular period.
        fd0 = obj.leg1.periods[acc_idx].dcf * f * (1 - acc_frac)
    else:
        # 1 minus acc_fra is the fraction of the period remaining until the next cashflow.
        fd0 = 1 - acc_frac
    return v** fd0


def _v1_compounded_by_remaining_accrual_frac_except_simple_final_period(
    self,
    ytm: DualTypes,
    f: int,
    settlement: datetime,
    acc_idx: int,
    v: DualTypes,
    accrual: callable,
    *args,
):
    """
    Uses regular fractional compounding except if it is last period, when simple money-mkt
    yield is used instead.
    Introduced for German Bunds.
    """
    if acc_idx == self.leg1.schedule.n_periods - 1:
        # or \
        # settlement == self.leg1.schedule.uschedule[acc_idx + 1]:
        # then settlement is in last period use simple interest.
        return self._v1_simple(ytm, f, settlement, acc_idx, v, accrual, *args)
    else:
        return self._v1_compounded_by_remaining_accrual_fraction(
            ytm,
            f,
            settlement,
            acc_idx,
            v,
            accrual,
            *args,
        )

