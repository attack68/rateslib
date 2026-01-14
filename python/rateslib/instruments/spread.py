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

from rateslib.enums.generics import NoInput
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


class Spread(_BaseInstrument):
    """
    A *Spread* of :class:`~rateslib.instruments.protocols._BaseInstrument`.

    .. rubric:: Examples

    The following initialises a purchased bond asset swap *Instrument* whose *rate* is
    the difference between the *IRS* rate and the *fixed rate bond* YTM.

    .. ipython:: python
       :suppress:

       from rateslib.instruments import Spread, IRS, FixedRateBond
       from datetime import datetime as dt

    .. ipython:: python

       irs = IRS(dt(2025, 12, 1), dt(2030, 12, 7), notional=1e6, spec="gbp_irs", curves=["uk_sonia"])
       ukt = FixedRateBond(dt(2024, 12, 7), dt(2030, 12, 7), notional=-1e6, fixed_rate=4.75, spec="uk_gb", metric="ytm", curves=["uk_gb"])
       asw = Spread(ukt, irs)
       asw.cashflows()

    .. rubric:: Pricing

    Each :class:`~rateslib.instruments.protocols._BaseInstrument` should have
    its own ``curves`` and ``vol`` objects set at its initialisation, according to the
    documentation for that *Instrument*. For the pricing methods ``curves`` and ``vol`` objects,
    these can be universally passed to each *Instrument* but in many cases that would be
    technically impossible since each *Instrument* might require difference pricing objects.
    In the above example a bond *Curve* and a swap *Curve* are required separately. For a *Spread*
    of two *IRS* in the same currency this would be possible, however.

    Parameters
    ----------
    instrument1 : _BaseInstrument
        The *Instrument* with the shortest maturity.
    instrument2 : _BaseInstrument
        The *Instrument* with the longest maturity.

    Notes
    -----
    A *Spread* is just a container for two
    :class:`~rateslib.instruments.protocols._BaseInstrument`, with an overload
    for the :meth:`~rateslib.instruments.Spread.rate` method to calculate the
    longer rate minus the shorter (whatever metric is in use for each *Instrument*), which allows
    it to offer a lot of flexibility in *pseudo Instrument* creation.

    """  # noqa: E501

    _instruments: Sequence[_BaseInstrument]
    _rate_scalar = 100.0

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
        local_npv = self._npv_single_core(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
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

    def rate(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
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
                    vol=vol,
                    base=base,
                    settlement=settlement,
                    forward=forward,
                    metric=metric,
                )
            )
        return (rates[1] - rates[0]) * 100.0

    def analytic_delta(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`analytic_delta` is not defined for Portfolio.")
