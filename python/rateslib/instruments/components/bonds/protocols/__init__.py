from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.components.bonds.protocols.accrued import _WithAccrued
from rateslib.instruments.components.bonds.protocols.cashflows import _WithExDiv
from rateslib.instruments.components.bonds.protocols.duration import _WithDuration
from rateslib.instruments.components.bonds.protocols.oaspread import _WithOASpread
from rateslib.instruments.components.bonds.protocols.repo import _WithRepo
from rateslib.instruments.components.bonds.protocols.ytm import _WithYTM
from rateslib.instruments.components.protocols import _BaseInstrument
from rateslib.instruments.components.protocols.pricing import (
    _maybe_get_curve_or_dict_maybe_from_solver,
)

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Curves_,
        DataFrame,
        DualTypes,
        FXForwards_,
        FXVolOption_,
        Solver_,
        VolT_,
        _BaseCurve_,
        datetime,
        datetime_,
        str_,
    )


class _BaseBondInstrument(
    _BaseInstrument,
    _WithExDiv,
    _WithDuration,
    _WithRepo,
    _WithYTM,
    _WithOASpread,
):
    def npv(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes | dict[str, DualTypes]:
        _curves = self._parse_curves(curves)
        disc_curve = _maybe_get_curve_or_dict_maybe_from_solver(
            curves_meta=self.kwargs.meta["curves"], curves=_curves, name="disc_curve", solver=solver
        )
        if isinstance(settlement, NoInput):
            settlement_ = self.leg1.schedule.calendar.lag_bus_days(
                disc_curve.nodes.initial,
                self.kwargs.meta["settle"],
                True,
            )
            forward_ = _drb(disc_curve.nodes.initial, forward)
        else:
            settlement_ = settlement
            forward_ = forward  # if NoInput adopts the usual default settings from 'settlement'

        return super().npv(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            local=local,
            settlement=settlement_,
            forward=forward_,
        )

    def cashflows(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        return super()._cashflows_from_legs(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            settlement=settlement,
            forward=forward,
        )

    def local_analytic_rate_fixings(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        return self._local_analytic_rate_fixings_from_legs(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            settlement=settlement,
            forward=forward,
        )

    def analytic_delta(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        leg: int = 1,
    ) -> DualTypes | dict[str, DualTypes]:
        settlement_ = self._maybe_get_settlement(
            settlement=settlement,
            disc_curve=_maybe_get_curve_or_dict_maybe_from_solver(
                curves_meta=self.kwargs.meta["curves"],
                curves=self._parse_curves(curves),
                name="disc_curve",
                solver=solver,
            ),
        )

        return super().analytic_delta(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            local=local,
            settlement=settlement_,
            forward=forward,
            leg=leg,
        )

    def price(self, ytm: DualTypes, settlement: datetime, dirty: bool = False) -> DualTypes:
        """
        Calculate the price of the security per nominal value of 100, given
        yield-to-maturity.

        Parameters
        ----------
        ytm : float
            The yield-to-maturity against which to determine the price.
        settlement : datetime
            The settlement date on which to determine the price.
        dirty : bool, optional
            If `True` will include the
            :meth:`rateslib.instruments.FixedRateBond.accrued` in the price.

        Returns
        -------
        float, Dual, Dual2

        Examples
        --------
        This example is taken from the UK debt management office website.
        The result should be `141.070132` and the bond is ex-div.

        .. ipython:: python

           gilt = FixedRateBond(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               calendar="ldn",
               currency="gbp",
               convention="ActActICMA",
               ex_div=7,
               fixed_rate=8.0
           )
           gilt.ex_div(dt(1999, 5, 27))
           gilt.price(
               ytm=4.445,
               settlement=dt(1999, 5, 27),
               dirty=True
           )

        This example is taken from the Swedish national debt office website.
        The result of accrued should, apparently, be `0.210417` and the clean
        price should be `99.334778`.

        .. ipython:: python

           bond = FixedRateBond(
               effective=dt(2017, 5, 12),
               termination=dt(2028, 5, 12),
               frequency="A",
               calendar="stk",
               currency="sek",
               convention="ActActICMA",
               ex_div=5,
               fixed_rate=0.75
           )
           bond.ex_div(dt(2017, 8, 23))
           bond.accrued(dt(2017, 8, 23))
           bond.price(
               ytm=0.815,
               settlement=dt(2017, 8, 23),
               dirty=False
           )

        """
        return self._price_from_ytm(
            ytm=ytm,
            settlement=settlement,
            calc_mode=NoInput(0),  # will be set to kwargs.meta
            dirty=dirty,
            rate_curve=NoInput(0),
        )

    def _maybe_get_settlement(
        self,
        settlement: datetime_,
        disc_curve: _BaseCurve_,
    ) -> datetime:
        if isinstance(settlement, NoInput):
            return self.leg1.schedule.calendar.lag_bus_days(
                disc_curve.nodes.initial,
                self.kwargs.meta["settle"],
                True,
            )
        else:
            return settlement


__all__ = [
    "_WithYTM",
    "_WithExDiv",
    "_WithAccrued",
    "_WithDuration",
    "_WithRepo",
    "_WithOASpread",
]
