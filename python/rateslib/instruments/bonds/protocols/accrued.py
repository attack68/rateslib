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

from rateslib.enums.generics import NoInput

if TYPE_CHECKING:
    from rateslib.instruments.bonds.conventions.accrued import AccrualFunction  # pragma: no cover
    from rateslib.typing import (  # pragma: no cover
        Cashflow,
        DualTypes,
        FixedLeg,
        FixedPeriod,
        FloatLeg,
        FloatPeriod,
        _BaseCurveOrDict_,
        _KWArgs,
        datetime,
    )


class _WithAccrued(Protocol):
    """
    Protocol to determine the *yield-to-maturity* of a bond type *Instrument*.
    """

    def _period_cashflow(
        self, period: Cashflow | FixedPeriod | FloatPeriod, rate_curve: _BaseCurveOrDict_
    ) -> DualTypes: ...

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
        The amount of accrued interest is calculated using the following formula:

        .. math::

           &AI = \\xi c_i \\qquad \\text{if not ex-dividend} \\\\
           &AI = (\\xi - 1) c_i \\qquad \\text{if ex-dividend} \\\\

        where :math:`c_i` is the physical ``cashflow`` related to the period in which ``settlement``
        falls, and :math:`\\xi` is a fraction of that amount determined according to the
        calculation mode specific to the :class:`~rateslib.instruments.BondCalcMode`.

        """  # noqa: E501
        return self._accrued(settlement, self.kwargs.meta["calc_mode"]._settle_accrual)
