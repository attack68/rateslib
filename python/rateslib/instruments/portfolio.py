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

from collections.abc import Sequence
from typing import TYPE_CHECKING, NoReturn

from pandas import DataFrame

from rateslib import defaults
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.protocols import _BaseInstrument
from rateslib.instruments.protocols.pricing import (
    _get_fx_maybe_from_solver,
)
from rateslib.periods.utils import _maybe_fx_converted

if TYPE_CHECKING:
    from rateslib.typing import (
        Any,
        CurvesT_,
        DualTypes,
        FXForwards_,
        Solver_,
        VolT_,
        datetime_,
        str_,
    )


def _instrument_npv(
    instrument: _BaseInstrument, *args: Any, **kwargs: Any
) -> DualTypes | dict[str, DualTypes]:  # pragma: no cover
    # this function is captured by TestPortfolio pooling but is not registered as a parallel process
    # used for parallel processing with Portfolio.npv
    return instrument.npv(*args, **kwargs)


class Portfolio(_BaseInstrument):
    """
    A collection of :class:`~rateslib.instruments.protocols._BaseInstrument`.

    .. rubric:: Examples

    The following initialises a *Portfolio* of *IRSs*.

    .. ipython:: python
       :suppress:

       from rateslib.instruments import Portfolio, IRS
       from datetime import datetime as dt

    .. ipython:: python

       pf = Portfolio(instruments=[
           IRS(dt(2000, 1, 1), "1y", notional=10e3, spec="eur_irs", curves=["estr"]),
           IRS(dt(2000, 1, 1), "2y", notional=10e3, spec="eur_irs", curves=["estr"]),
           IRS(dt(2000, 1, 1), "3y", notional=10e3, spec="eur_irs", curves=["estr"]),
       ])
       pf.cashflows()

    .. rubric:: Pricing

    Each :class:`~rateslib.instruments.protocols._BaseInstrument` should have
    its own ``curves`` and ``vol`` objects set at its initialisation, according to the
    documentation for that *Instrument*. For the pricing methods ``curves`` and ``vol`` objects,
    these can be universally passed to each *Instrument* but in many cases that would be
    technically impossible since each *Instrument* might require difference pricing objects, e.g.
    if the *Instruments* have difference currencies. For a *Portfolio*
    of three *IRS* in the same currency this would be possible, however.

    Parameters
    ----------
    instruments : list of _BaseInstrument
        The collection of *Instruments*.

    Notes
    -----
    A *Portfolio* is just a container for multiple
    :class:`~rateslib.instruments.protocols._BaseInstrument`.
    There is no concept of a :meth:`~rateslib.instruments.Portfolio.rate`.

    """

    _instruments: Sequence[_BaseInstrument]

    @property
    def instruments(self) -> Sequence[_BaseInstrument]:
        """The *Instruments* contained within the *Portfolio*."""
        return self._instruments

    def __init__(self, instruments: Sequence[_BaseInstrument]) -> None:
        if not isinstance(instruments, Sequence):
            raise ValueError("`instruments` should be a list of Instruments.")
        self._instruments = instruments

    def npv(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *Portfolio* by summing individual *Instrument* NPVs.
        """
        # if the pool is 1 do not do any parallel processing and return the single core func
        if defaults.pool == 1:
            local_npv: dict[str, DualTypes] = self._npv_single_core(
                curves=curves,
                solver=solver,
                fx=fx,
                vol=vol,
                base=base,
                settlement=settlement,
                forward=forward,
            )
        else:
            from functools import partial
            from multiprocessing import Pool

            func = partial(
                _instrument_npv,
                curves=curves,
                solver=solver,
                fx=fx,
                vol=vol,
                base=base,
                local=True,
                forward=forward,
                settlement=settlement,
            )
            p = Pool(defaults.pool)
            results = p.map(func, self.instruments)
            p.close()

            # Aggregate results:
            _ = DataFrame(results).fillna(0.0)
            _ = _.sum()
            local_npv = _.to_dict()  # type: ignore[assignment]

            # ret = {}
            # for result in results:
            #     for ccy in result:
            #         if ccy in ret:
            #             ret[ccy] += result[ccy]
            #         else:
            #             ret[ccy] = result[ccy]

        if not local:
            single_value: DualTypes = 0.0
            base_ = _drb(self.settlement_params.currency, base)
            for k, v in local_npv.items():
                single_value += _maybe_fx_converted(
                    value=v,
                    currency=k,
                    fx=_get_fx_maybe_from_solver(fx=fx, solver=solver),
                    base=base_,
                    forward=forward,
                )
            return single_value
        else:
            return local_npv

    def local_analytic_rate_fixings(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        return self._local_analytic_rate_fixings_from_instruments(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            settlement=settlement,
            forward=forward,
        )

    def cashflows(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        return self._cashflows_from_instruments(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            settlement=settlement,
            forward=forward,
            base=base,
        )

    def rate(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`rate` is not defined for Portfolio.")

    def analytic_delta(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`analytic_delta` is not defined for Portfolio.")
