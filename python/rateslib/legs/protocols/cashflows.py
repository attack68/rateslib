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

from pandas import DataFrame

from rateslib.curves import index_left
from rateslib.enums.generics import NoInput

if TYPE_CHECKING:
    from rateslib.typing import (
        CurveOption_,
        FXForwards_,
        Schedule,
        Sequence,
        _BaseCurve_,
        _BasePeriod,
        _FXVolOption_,
        datetime,
        datetime_,
        str_,
    )


class _WithCashflows(Protocol):
    """
    Protocol to generate cashflows of any *Leg* type.

    """

    @property
    def periods(self) -> Sequence[_BasePeriod]: ...

    def cashflows(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Return aggregated cashflow data for the *Leg*.

        .. warning::

           This method is a convenience method to provide a visual representation of all
           associated calculation data. Calling this method to extracting certain values
           should be avoided. It is more efficent to source relevant parameters or calculations
           from object attributes or other methods directly.

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        index_curve: _BaseCurve, optional
            Used to forecast index values for indexation, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForwards` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary. Or, an
            :class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.
        fx_vol: FXDeltaVolSmile, FXSabrSmile, FXDeltaVolSurface, FXSabrSurface, optional
            The FX volatility *Smile* or *Surface* object used for determining Black calendar
            day implied volatility values.
        base: str, optional
            The currency to convert relevant values into.
        settlement: datetime, optional
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, optional
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        DataFrame
        """
        seq = [
            period.cashflows(
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                index_curve=index_curve,
                fx=fx,
                fx_vol=fx_vol,
                base=base,
                settlement=settlement,
                forward=forward,
            )
            for period in self.periods
        ]
        return DataFrame.from_records(seq)


class _WithExDiv(Protocol):
    """
    Protocol to determine if a *Leg* is ex-dividend on a given settlement.

    """

    @property
    def schedule(self) -> Schedule: ...

    def _period_index(self, settlement: datetime) -> int:
        """
        Get the period index for that which the settlement date fall within.
        Uses adjusted dates.
        """
        _: int = index_left(
            self.schedule.aschedule,
            len(self.schedule.aschedule),
            settlement,
        )
        return _

    def ex_div(self, settlement: datetime) -> bool:
        """
        Return a boolean whether the security is ex-div at the given settlement.

        Parameters
        ----------
        settlement : datetime
            The settlement date to test.

        Returns
        -------
        bool

        Notes
        -----
        Uses the UK DMO convention of returning *False* if ``settlement``
        **is on or before** the ex-div date for a regular coupon period.

        This is evaluated by analysing the attribute ``pschedule3`` of the associated
        :class:`~rateslib.scheduling.Schedule` object of the *Leg*.
        """
        left_period_index = self._period_index(settlement)
        ex_div_date = self.schedule.pschedule3[left_period_index + 1]
        return settlement > ex_div_date
