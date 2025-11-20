from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from rateslib.enums.generics import NoInput

if TYPE_CHECKING:
    from rateslib.instruments.components.bonds.conventions import AccrualFunction
    from rateslib.typing import DualTypes, FixedLeg, FloatLeg, _KWArgs, datetime  # pragma: no cover


class _WithAccrued(Protocol):
    """
    Protocol to determine the *yield-to-maturity* of a bond type *Instrument*.
    """

    @property
    def leg1(self) -> FixedLeg | FloatLeg: ...

    @property
    def kwargs(self) -> _KWArgs: ...

    def _accrued(self, settlement: datetime, func: AccrualFunction) -> DualTypes:
        """func is the specific accrued function associated with the bond ``calc_mode``"""
        acc_idx = self.leg1._period_index(settlement)
        frac = func(self, settlement, acc_idx)
        if self.leg1.ex_div(settlement):
            frac = frac - 1  # accrued is negative in ex-div period
        _: DualTypes = self._period_cashflow(self.leg1._regular_periods[acc_idx], NoInput(0))
        return frac * _ / -self.leg1._regular_periods[acc_idx].settlement_params.notional * 100

    def accrued(self, settlement: datetime) -> DualTypes:
        """
        Calculate the accrued amount per nominal par value of 100.

        Parameters
        ----------
        settlement : datetime
            The settlement date which to measure accrued interest against.

        Notes
        -----
        Fractionally apportions the coupon payment based on calendar days.

        .. math::

           \\text{Accrued} = \\text{Coupon} \\times \\frac{\\text{Settle - Last Coupon}}{\\text{Next Coupon - Last Coupon}}

        """  # noqa: E501
        return self._accrued(settlement, self.kwargs.meta["calc_mode"]._settle_accrual)
