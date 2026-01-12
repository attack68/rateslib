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

from typing import TYPE_CHECKING, Protocol

from rateslib import defaults
from rateslib.curves import index_left
from rateslib.curves.utils import average_rate
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.bonds.protocols import _WithAccrued
from rateslib.legs.amortization import _AmortizationType
from rateslib.scheduling import dcf

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        DualTypes,
        datetime,
        str_,
    )


class _WithRepo(_WithAccrued, Protocol):
    """
    Protocol to determine the *yield-to-maturity* of a bond type *Instrument*.
    """

    def fwd_from_repo(
        self,
        price: DualTypes,
        settlement: datetime,
        forward_settlement: datetime,
        repo_rate: DualTypes,
        convention: str_ = NoInput(0),
        dirty: bool = False,
        method: str = "proceeds",
    ) -> DualTypes:
        """
        Return a forward price implied by a given repo rate.

        Parameters
        ----------
        price : float, Dual, or Dual2
            The initial price of the security at ``settlement``.
        settlement : datetime
            The settlement date of the bond
        forward_settlement : datetime
            The forward date for which to calculate the forward price.
        repo_rate : float, Dual or Dual2
            The rate which is used to calculate values.
        convention : str, optional
            The day count convention applied to the rate. If not given uses default
            values.
        dirty : bool, optional
            Whether the input and output price are specified including accrued interest.
        method : str in {"proceeds", "compounded"}, optional
            The method for determining the forward price.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        Any intermediate (non ex-dividend) cashflows between ``settlement`` and
        ``forward_settlement`` will also be assumed to accrue at ``repo_rate``.
        """
        convention_ = _drb(defaults.convention, convention)
        dcf_ = dcf(settlement, forward_settlement, convention_)
        if not dirty:
            d_price = price + self._accrued(
                settlement=settlement, func=self.kwargs.meta["calc_mode"]._settle_accrual
            )
        else:
            d_price = price
        if self.leg1.amortization._type != _AmortizationType.NoAmortization:
            raise NotImplementedError(
                "method for forward price not available with amortization",
            )  # pragma: no cover
        total_rtn = (
            d_price * (1 + repo_rate * dcf_ / 100) * -self.leg1.settlement_params.notional / 100
        )

        # now systematically deduct coupons paid between settle and forward settle
        settlement_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            settlement,
        )
        fwd_settlement_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            forward_settlement,
        )

        # do not accrue a coupon not received
        settlement_idx += 1 if self.leg1.ex_div(settlement) else 0
        # deduct final coupon if received within period
        fwd_settlement_idx += 1 if self.leg1.ex_div(forward_settlement) else 0

        for p_idx in range(settlement_idx, fwd_settlement_idx):
            # deduct accrued coupon from dirty price
            c_period = self.leg1._regular_periods[p_idx]
            c_cashflow: DualTypes = c_period.cashflow()
            # TODO handle FloatPeriod cashflow fetch if need a curve.
            if method.lower() == "proceeds":
                dcf_ = dcf(c_period.settlement_params.payment, forward_settlement, convention_)
                accrued_coup = c_cashflow * (1 + dcf_ * repo_rate / 100)
                total_rtn -= accrued_coup
            elif method.lower() == "compounded":
                r_bar, d, _ = average_rate(
                    settlement, forward_settlement, convention_, repo_rate, dcf_
                )
                n = (forward_settlement - c_period.settlement_params.payment).days
                accrued_coup = c_cashflow * (1 + d * r_bar / 100) ** n
                total_rtn -= accrued_coup
            else:
                raise ValueError("`method` must be in {'proceeds', 'compounded'}.")

        forward_price: DualTypes = total_rtn / -self.leg1.settlement_params.notional * 100
        if dirty:
            return forward_price
        else:
            return forward_price - self._accrued(
                settlement=forward_settlement, func=self.kwargs.meta["calc_mode"]._settle_accrual
            )

    def repo_from_fwd(
        self,
        price: DualTypes,
        settlement: datetime,
        forward_settlement: datetime,
        forward_price: DualTypes,
        convention: str_ = NoInput(0),
        dirty: bool = False,
    ) -> DualTypes:
        """
        Return an implied repo rate from a forward price.

        Parameters
        ----------
        price : float, Dual, or Dual2
            The initial price of the security at ``settlement``.
        settlement : datetime
            The settlement date of the bond
        forward_settlement : datetime
            The forward date for which to calculate the forward price.
        forward_price : float, Dual or Dual2
            The forward price which iplies the repo rate
        convention : str, optional
            The day count convention applied to the rate. If not given uses default
            values.
        dirty : bool, optional
            Whether the input and output price are specified including accrued interest.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        Any intermediate (non ex-dividend) cashflows between ``settlement`` and
        ``forward_settlement`` will also be assumed to accrue at ``repo_rate``.
        """
        convention_ = _drb(defaults.convention, convention)
        # forward price from repo is linear in repo_rate so reverse calculate with AD
        if not dirty:
            p_t = forward_price + self._accrued(
                settlement=forward_settlement, func=self.kwargs.meta["calc_mode"]._settle_accrual
            )
            p_0 = price + self._accrued(
                settlement=settlement, func=self.kwargs.meta["calc_mode"]._settle_accrual
            )
        else:
            p_t, p_0 = forward_price, price

        dcf_ = dcf(settlement, forward_settlement, convention_)
        numerator = p_t - p_0
        denominator = p_0 * dcf_

        # now systematically deduct coupons paid between settle and forward settle
        settlement_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            settlement,
        )
        fwd_settlement_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            forward_settlement,
        )

        # do not accrue a coupon not received
        settlement_idx += 1 if self.leg1.ex_div(settlement) else 0
        # deduct final coupon if received within period
        fwd_settlement_idx += 1 if self.leg1.ex_div(forward_settlement) else 0

        for p_idx in range(settlement_idx, fwd_settlement_idx):
            # deduct accrued coupon from dirty price
            c_period = self.leg1._regular_periods[p_idx]
            c_cashflow: DualTypes = c_period.cashflow()
            # TODO handle FloatPeriod if it needs a Curve to forecast cashflow
            dcf_ = dcf(
                start=c_period.settlement_params.payment,
                end=forward_settlement,
                convention=convention_,
            )
            numerator += 100 * c_cashflow / -self.leg1.settlement_params.notional
            denominator -= 100 * dcf_ * c_cashflow / -self.leg1.settlement_params.notional

        return numerator / denominator * 100
