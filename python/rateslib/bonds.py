# This file contains bond convention outlines
from __future__ import annotations

from datetime import datetime

from rateslib import defaults
from rateslib.calendars import add_tenor, dcf
from rateslib.curves import index_left
from rateslib.default import NoInput
from rateslib.dual import DualTypes


class _AccruedAndYTMMethods:
    def _period_index(self, settlement: datetime):
        """
        Get the coupon period index for that which the settlement date fall within.
        Uses unadjusted dates.
        """
        _ = index_left(
            self.leg1.schedule.uschedule,
            len(self.leg1.schedule.uschedule),
            settlement,
        )
        return _

    def _accrued_fraction(self, settlement: datetime, calc_mode: str | NoInput, acc_idx: int):
        """
        Return the accrual fraction of period between last coupon and settlement and
        coupon period left index.

        Branches to a calculation based on the bond `calc_mode`.
        """
        try:
            func = getattr(self, f"_{calc_mode}")["accrual_mode"]
            # func = getattr(self, self._acc_frac_mode_map[calc_mode])
            return func(settlement, acc_idx)
        except KeyError:
            raise ValueError(f"Cannot calculate for `calc_mode`: {calc_mode}")

    def _acc_linear_proportion_by_days(self, settlement: datetime, acc_idx: int, *args):
        """
        Return the fraction of an accrual period between start and settlement.

        Method: a linear proportion of actual days between start, settlement and end.
        Measures between unadjusted coupon dates.

        This is a general method, used by many types of bonds, for example by UK Gilts,
        German Bunds.
        """
        r = settlement - self.leg1.schedule.uschedule[acc_idx]
        s = self.leg1.schedule.uschedule[acc_idx + 1] - self.leg1.schedule.uschedule[acc_idx]
        return r / s

    def _acc_linear_proportion_by_days_long_stub_split(
        self, settlement: datetime, acc_idx: int, *args
    ):
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

        return self._acc_linear_proportion_by_days(settlement, acc_idx, *args)

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

    def _acc_act365_with_1y_and_stub_adjustment(self, settlement: datetime, acc_idx: int, *args):
        """
        Ignoring the convention on the leg uses "Act365f" to determine the accrual fraction.
        Measures between unadjusted date and settlement.
        Special adjustment if number of days is greater than 365.
        If the period is a stub reverts to a straight line interpolation
        [this is primarily designed for Canadian Government Bonds]
        """
        if self.leg1.periods[acc_idx].stub:
            return self._acc_linear_proportion_by_days(settlement, acc_idx)
        f = 12 / defaults.frequency_months[self.leg1.schedule.frequency]
        r = settlement - self.leg1.schedule.uschedule[acc_idx]
        s = self.leg1.schedule.uschedule[acc_idx + 1] - self.leg1.schedule.uschedule[acc_idx]
        if r == s:
            _ = 1.0  # then settlement falls on the coupon date
        elif r.days > 365.0 / f:
            _ = 1.0 - ((s - r).days * f) / 365.0  # counts remaining days
        else:
            _ = f * r.days / 365.0
        return _

    def _v1_compounded_by_remaining_accrual_fraction(
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
        Determine the discount factor for the first cashflow after settlement.

        The parameter "v" is a generic discount function which is normally :math:`1/(1+y/f)`

        Method: compounds "v" by the accrual fraction of the period.
        """
        acc_frac = accrual(settlement, acc_idx)
        if self.leg1.periods[acc_idx].stub:
            # If it is a stub then the remaining fraction must be scaled by the relative size of the
            # stub period compared with a regular period.
            fd0 = self.leg1.periods[acc_idx].dcf * f * (1 - acc_frac)
        else:
            # 1 minus acc_fra is the fraction of the period remaining until the next cashflow.
            fd0 = 1 - acc_frac
        return v**fd0

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
                ytm, f, settlement, acc_idx, v, accrual, *args
            )

    def _v1_comp_stub_act365f(
        self,
        ytm: DualTypes,
        f: int,
        settlement: datetime,
        acc_idx: int,
        v: DualTypes,
        accrual: callable,
        *args,
    ):
        """Compounds the yield. In a stub period the act365f DCF is used"""
        if not self.leg1.periods[acc_idx].stub:
            return self._v1_compounded_by_remaining_accrual_fraction(
                ytm, f, settlement, acc_idx, v, accrual, *args
            )
        else:
            fd0 = dcf(settlement, self.leg1.schedule.uschedule[acc_idx + 1], "Act365F")
            return v**fd0

    def _v1_simple(
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
        Use simple rates with a yield which matches the frequency of the coupon.
        """
        acc_frac = accrual(settlement, acc_idx)
        if self.leg1.periods[acc_idx].stub:
            # is a stub so must account for discounting in a different way.
            fd0 = self.leg1.periods[acc_idx].dcf * f * (1 - acc_frac)
        else:
            fd0 = 1 - acc_frac

        v_ = 1 / (1 + fd0 * ytm / (100 * f))
        return v_

    def _v1_simple_1y_adjustment(
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
        Use simple rates with a yield which matches the frequency of the coupon.

        If the stub period is long, then discount the regular part of the stub with the regular
        discount param ``v``.
        """
        acc_frac = accrual(settlement, acc_idx)
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

    def _v2_(self, ytm: DualTypes, f: int, *args):
        """
        Default method for a single regular period discounted in the regular portion of bond.
        Implies compounding at the same frequency as the coupons.
        """
        return 1 / (1 + ytm / (100 * f))

    def _v2_annual(self, ytm: DualTypes, f: int, *args):
        """
        ytm is expressed annually but coupon payments are on another frequency
        """
        return (1 / (1 + ytm / 100)) ** (1 / f)

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

    def _v3_simple(
        self,
        ytm: DualTypes,
        f: int,
        settlement: datetime,
        acc_idx: int,
        v: DualTypes,
        accrual: callable,
        *args,
    ):
        v_ = 1 / (1 + self.leg1.periods[-2].dcf * ytm / 100.0)
        return v_


class _BondConventions(_AccruedAndYTMMethods):
    """
    Contains calculation conventions and specifies calculation modes for different bonds
    of different jurisdictions.

    For FixedRateBonds the conventions are as follows:

    {
        "accrual": callable that returns a fraction of a period to determine accrued interest.
        "v1": discounting function for the first cashflow of a bond.
        "v2": discounting function for intermediate cashflows of a bond.
        "v3": discounting function for the last cashflow of a bond.
    }
    """

    # FixedRateBonds

    @property
    def _uk_gb(self):
        """Mode used for UK Gilts"""
        return {
            "accrual": self._acc_linear_proportion_by_days,
            "v1": self._v1_compounded_by_remaining_accrual_fraction,
            "v2": self._v2_,
            "v3": self._v3_dcf_comp,
        }

    @property
    def _us_gb(self):
        """Street convention for US Treasuries"""
        return {
            "accrual": self._acc_linear_proportion_by_days_long_stub_split,
            "v1": self._v1_compounded_by_remaining_accrual_fraction,
            "v2": self._v2_,
            "v3": self._v3_dcf_comp,
        }

    @property
    def _us_gb_tsy(self):
        """Treasury convention for US Treasuries"""
        return {
            "accrual": self._acc_linear_proportion_by_days_long_stub_split,
            "v1": self._v1_simple_1y_adjustment,
            "v2": self._v2_,
            "v3": self._v3_dcf_comp,
        }

    @property
    def _se_gb(self):
        """Mode used for Swedish GBs."""
        return {
            "accrual": self._acc_30e360,
            "v1": self._v1_compounded_by_remaining_accrual_fraction,
            "v2": self._v2_,
            "v3": self._v3_30e360_u_simple,
        }

    @property
    def _ca_gb(self):
        """Mode used for Canadian GBs."""
        return {
            "accrual": self._acc_act365_with_1y_and_stub_adjustment,
            "ytm_accrual": self._acc_linear_proportion_by_days,
            "v1": self._v1_compounded_by_remaining_accrual_fraction,
            "v2": self._v2_,
            "v3": self._v3_30e360_u_simple,
        }

    @property
    def _de_gb(self):
        """Mode used for German GBs."""
        return {
            "accrual": self._acc_linear_proportion_by_days,
            "v1": self._v1_compounded_by_remaining_accrual_frac_except_simple_final_period,
            "v2": self._v2_,
            "v3": self._v3_dcf_comp,
        }

    @property
    def _fr_gb(self):
        """Mode used for French OATs."""
        return {
            "accrual": self._acc_linear_proportion_by_days,
            "v1": self._v1_compounded_by_remaining_accrual_fraction,
            "v2": self._v2_,
            "v3": self._v3_dcf_comp,
        }

    @property
    def _it_gb(self):
        """Mode used for Italian BTPs."""
        return {
            "accrual": self._acc_linear_proportion_by_days,
            "v1": self._v1_compounded_by_remaining_accrual_frac_except_simple_final_period,
            "v2": self._v2_annual,
            "v3": self._v3_dcf_comp,
        }

    @property
    def _no_gb(self):
        """Mode used for Norwegian GBs."""
        return {
            "accrual": self._acc_act365_with_1y_and_stub_adjustment,
            "v1": self._v1_comp_stub_act365f,
            "v2": self._v2_,
            "v3": self._v3_dcf_comp,
        }

    @property
    def _nl_gb(self):
        """Mode used for Dutch GBs."""
        return {
            "accrual": self._acc_linear_proportion_by_days_long_stub_split,
            "v1": self._v1_compounded_by_remaining_accrual_frac_except_simple_final_period,
            "v2": self._v2_,
            "v3": self._v3_dcf_comp,
        }

    # Bills

    @property
    def _us_gbb(self):
        """Mode used for US T-Bills"""
        return {
            "accrual": self._acc_linear_proportion_by_days,
            "price_type": self._price_discount,
            "ytm_clone": "us_gb",
        }

    @property
    def _se_gbb(self):
        """Mode used for Swedish T-Bills"""
        return {
            "accrual": self._acc_linear_proportion_by_days,
            "price_type": self._price_simple,
            "ytm_clone": "se_gb",
        }

    @property
    def _uk_gbb(self):
        """Mode used for UK T-Bills"""
        return {
            "accrual": self._acc_linear_proportion_by_days,
            "price_type": self._price_simple,
            "ytm_clone": "uk_gb",
        }

    ### Deprecated Aliases

    @property
    def _ukg(self):
        """deprecated alias"""
        return self._uk_gb

    @property
    def _ust(self):
        """deprecated alias"""
        return self._us_gb

    @property
    def _ustb(self):
        """deprecated alias"""
        return self._us_gbb

    @property
    def _ust_31bii(self):
        """deprecated alias"""
        return self._us_gb_tsy

    @property
    def _sgb(self):
        """deprecated alias"""
        return self._se_gb

    @property
    def _cadgb(self):
        """deprecated alias"""
        return self._ca_gb

    @property
    def _sgbb(self):
        """deprecated alias"""
        return self._se_gbb

    @property
    def _uktb(self):
        """deprecated alias"""
        return self._uk_gbb
