# This file contains bond convention outlines
from datetime import datetime
from typing import Union

from rateslib import defaults
from rateslib.default import NoInput
from rateslib.dual import DualTypes
from rateslib.calendars import add_tenor, dcf

class BondConventions:
    """
    Contains calculation conventions and specifies calculation modes for different bonds of different jurisdictions.
    """

    def _gbp_gb(self):
        """Mode used for UK Gilts"""
        return {
            "accrual_mode": self._acc_lin_days,
            "v1": self._v1_comp,
            "v2": self._v2_,
            "v3": self._v3_dcf_comp,
        }

    def _usd_gb(self):
        """Street convention for US Treasuries"""
        return {
            "accrual_mode": self._acc_lin_days_long_split,
            "v1": self._v1_comp,
            "v2": self._v2_,
            "v3": self._v3_dcf_comp,
        }

    def _usd_gb_tsy(self):
        """Treasury convention for US Treasuries"""
        return {
            "accrual_mode": self._acc_lin_days_long_split,
            "v1": self._v1_simple,
            "v2": self._v2_,
            "v3": self._v3_dcf_comp,
        }

    def _sek_gb(self):
        """Mode used for Swedish GBs."""
        return {
            "accrual_mode": self._acc_30e360,
            "v1": self._v1_comp,
            "v2": self._v2_,
            "v3": self._v3_30e360_u_simple,
        }

    def _acc_lin_days(self, settlement: datetime, acc_idx: int, *args):
        """
        Return the fraction of an accrual period between start and settlement.

        Method: a linear proportion of actual days between start, settlement and end.
        Measures between unadjusted coupon dates.

        This is a general method, used by many types of bonds, for example by UK Gilts, German Bunds.
        """
        r = settlement - self.leg1.schedule.uschedule[acc_idx]
        s = self.leg1.schedule.uschedule[acc_idx + 1] - self.leg1.schedule.uschedule[acc_idx]
        return r / s

    def _acc_lin_days_long_split(self, settlement: datetime, acc_idx: int, *args):
        """
        For long stub periods this splits the accrued interest into two components.
        Otherwise, returns the regular linear proportion.
        [Designed primarily for US Treasuries]
        """
        if self.leg1.periods[acc_idx].stub:
            fm = defaults.frequency_months[self.leg1.schedule.frequency]
            f = 12 / fm
            if self.leg1.periods[acc_idx].dcf * f > 1:
                # long stub
                quasi_coupon = add_tenor(
                    self.leg1.schedule.uschedule[acc_idx + 1],
                    f"-{fm}M",
                    "NONE",
                    NoInput(0),
                    self.leg1.schedule.roll,
                )
                quasi_start = add_tenor(
                    quasi_coupon,
                    f"-{fm}M",
                    "NONE",
                    NoInput(0),
                    self.leg1.schedule.roll,
                )
                if settlement <= quasi_coupon:
                    # then first part of long stub
                    r = quasi_coupon - settlement
                    s = quasi_coupon - quasi_start
                    r_ = quasi_coupon - self.leg1.schedule.uschedule[acc_idx]
                    _ = (r_ - r) / s
                    return _ / (self.leg1.periods[acc_idx].dcf * f)
                else:
                    # then second part of long stub
                    r = self.leg1.schedule.uschedule[acc_idx + 1] - settlement
                    s = self.leg1.schedule.uschedule[acc_idx + 1] - quasi_coupon
                    r_ = quasi_coupon - self.leg1.schedule.uschedule[acc_idx]
                    s_ = quasi_coupon - quasi_start
                    _ = r_ / s_ + (s - r) / s
                    return _ / (self.leg1.periods[acc_idx].dcf * f)

        return self._acc_lin_days(settlement, acc_idx, *args)

    def _acc_30e360(self, settlement: datetime, acc_idx: int, *args):
        """
        Ignoring the convention on the leg uses "30E360" to determine the accrual fraction.
        Measures between unadjusted date and settlement.
        [Designed primarily for Swedish Government Bonds]
        """
        f = 12 / defaults.frequency_months[self.leg1.schedule.frequency]
        _ = dcf(settlement, self.leg1.schedule.uschedule[acc_idx + 1], "30e360") * f
        _ = 1 - _
        return _

    def _v1_comp(
        self,
        ytm: DualTypes,
        f: int,
        settlement: datetime,
        acc_idx: int,
        v: DualTypes,
        accrual_calc_mode: Union[str, NoInput],
        *args,
    ):
        """
        Determine the discount factor for the first cashflow after settlement.

        The parameter "v" is a generic discount function which is normally :math:`1/(1+y/f)`

        Method: compounds "v" by the accrual fraction of the period.
        """
        acc_frac = self._accrued_frac(settlement, accrual_calc_mode, acc_idx)
        if self.leg1.periods[acc_idx].stub:
            # If it is a stub then the remaining fraction must be scaled by the relative size of the
            # stub period compared with a regular period.
            fd0 = self.leg1.periods[acc_idx].dcf * f * (1 - acc_frac)
        else:
            # 1 minus acc_fra is the fraction of the period remaining until the next cashflow.
            fd0 = 1 - acc_frac
        return v**fd0

    def _v1_simple(
        self,
        ytm: DualTypes,
        f: int,
        settlement: datetime,
        acc_idx: int,
        v: DualTypes,
        accrual_calc_mode: Union[str, NoInput],
        *args,
    ):
        """
        Use simple rates with a yield which matches the frequency of the coupon.

        If the stub period is long, then discount the regular part of the stub with the regular discount param ``v``.
        """
        acc_frac = self._accrued_frac(settlement, accrual_calc_mode, acc_idx)
        if self.leg1.periods[acc_idx].stub:
            # is a stub so must account for discounting in a different way.
            fd0 = self.leg1.periods[acc_idx].dcf * f * (1 - acc_frac)
        else:
            fd0 = 1 - acc_frac

        if fd0 > 1.0:
            v_ = v * 1 / (1 + (fd0 - 1) * ytm / (100 * f))
        else:
            v_ = 1 / (1 + fd0 * ytm / (100 * f))

        return v_

    def _v2_(self, ytm: DualTypes, f: int, settlement: datetime, acc_idx: int, *args):
        """
        Default method for a single regular period discounted in the regular portion of bond.
        Implies compounding at the same frequency as the coupons.
        """
        return 1 / (1 + ytm / (100 * f))

    def _v3_dcf_comp(
        self,
        ytm: DualTypes,
        f: int,
        settlement: datetime,
        acc_idx: int,
        v: DualTypes,
        *args,
    ):
        """
        Final period uses a compounding approach where the power is determined by the DCF of that
        period under the bond's specified convention.
        """
        if self.leg1.periods[acc_idx].stub:
            # If it is a stub then the remaining fraction must be scaled by the relative size of the
            # stub period compared with a regular period.
            fd0 = self.leg1.periods[acc_idx].dcf * f
        else:
            fd0 = 1
        return v**fd0

    def _v3_30e360_u_simple(
        self,
        ytm: DualTypes,
        f: int,
        settlement: datetime,
        acc_idx: int,
        v: DualTypes,
        *args,
    ):
        """
        The final period is discounted by a simple interest method under a 30E360 convention.

        The YTM is assumed to have the same frequency as the coupons.
        """
        d_ = dcf(self.leg1.periods[acc_idx].start, self.leg1.periods[acc_idx].end, "30E360")
        return 1 / (1 + d_ * ytm / 100)  # simple interest
