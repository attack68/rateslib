from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pandas import DataFrame

from rateslib.curves import index_left
from rateslib.enums.generics import NoInput

if TYPE_CHECKING:
    from rateslib.typing import (
        CurveOption_,
        FXForwards_,
        FXVolOption_,
        Schedule,
        _BaseCurve_,
        _BasePeriod,
        datetime,
        datetime_,
        str_,
    )


class _WithCashflows(Protocol):
    @property
    def periods(self) -> list[_BasePeriod]: ...

    def cashflows(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Return aggregated cashflow data for the *Period*.

        .. warning::

           This method is a convenience method to provide a visual representation of all
           associated calculation data. Calling this method to extracting certain values
           should be avoided. It is more efficent to source relevant parameters or calculations
           from object attributes or other methods directly.

        Parameters
        ----------
        XXX

        Returns
        -------
        dict of values
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
