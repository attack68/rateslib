from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pandas import DataFrame

from rateslib.enums.generics import NoInput

if TYPE_CHECKING:
    from rateslib.typing import (
        CurveOption_,
        FXForwards_,
        FXVolOption_,
        Period,
        _BaseCurve_,
        datetime_,
        str_,
    )


class _WithCashflows(Protocol):
    periods: list[Period]

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
