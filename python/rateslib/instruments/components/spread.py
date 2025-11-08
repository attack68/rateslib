from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NoReturn

from pandas import DataFrame

from rateslib.enums.generics import NoInput
from rateslib.instruments.components.protocols import _BaseInstrument
from rateslib.instruments.components.protocols.pricing import (
    _get_fx_maybe_from_solver,
)
from rateslib.periods.components.utils import _maybe_fx_converted

if TYPE_CHECKING:
    from rateslib.typing import (
        Any,
        Curves_,
        DualTypes,
        FXForwards_,
        FXVolOption_,
        Solver_,
        datetime_,
        str_,
    )


class Spread(_BaseInstrument):
    """
    Create a *Spread* of *Instruments*.

    Parameters
    ----------
    instrument1 : _BaseInstrument
        An *Instrument* with the shortest maturity.
    instrument2 : _BaseInstrument
        The *Instrument* with the longest maturity.
    """

    _instruments: Sequence[_BaseInstrument]

    @property
    def instruments(self) -> Sequence[_BaseInstrument]:
        """The *Instruments* contained within the *Portfolio*."""
        return self._instruments

    def __init__(
        self,
        instrument1: _BaseInstrument,
        instrument2: _BaseInstrument,
    ) -> None:
        self._instruments = [instrument1, instrument2]

    def npv(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *Portfolio* by summing individual *Instrument* NPVs.
        """
        local_npv = self._npv_single_core(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            base=base,
        )
        if not local:
            single_value: DualTypes = 0.0
            for k, v in local_npv.items():
                single_value += _maybe_fx_converted(
                    value=v,
                    currency=k,
                    fx=_get_fx_maybe_from_solver(fx=fx, solver=solver),
                    base=base,
                )
            return single_value
        else:
            return local_npv

    def local_analytic_rate_fixings(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        return self._local_analytic_rate_fixings_from_instruments(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            settlement=settlement,
            forward=forward,
        )

    def cashflows(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        return self._cashflows_from_instruments(
            curves=curves,
            solver=solver,
            fx=fx,
            fx_vol=fx_vol,
            settlement=settlement,
            forward=forward,
            base=base,
        )

    def delta(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)

    def exo_delta(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the delta of the *Instrument* measured
        against user defined :class:`~rateslib.dual.Variable`.

        For arguments see
        :meth:`Sensitivities.exo_delta()<rateslib.instruments.Sensitivities.exo_delta>`.
        """
        return super().exo_delta(*args, **kwargs)

    def rate(
        self,
        *,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        rates: list[DualTypes] = []
        for inst in self.instruments:
            rates.append(
                inst.rate(
                    curves=curves,
                    solver=solver,
                    fx=fx,
                    fx_vol=fx_vol,
                    base=base,
                    settlement=settlement,
                    forward=forward,
                    metric=metric,
                )
            )
        return (rates[1] - rates[0]) * 100.0

    def analytic_delta(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`analytic_delta` is not defined for Portfolio.")
