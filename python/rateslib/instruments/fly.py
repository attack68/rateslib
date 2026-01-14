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

from pandas import DataFrame, DatetimeIndex

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


def _composit_fixings_table(df_result: DataFrame, df: DataFrame) -> DataFrame:
    """
    Add a DataFrame to an existing fixings table by extending or adding to relevant columns.

    Parameters
    ----------
    df_result: The main DataFrame that will be updated
    df: The incoming DataFrame with new data to merge

    Returns
    -------
    DataFrame
    """
    # reindex the result DataFrame
    if df_result.empty:
        return df
    else:
        df_result = df_result.reindex(index=df_result.index.union(df.index))

    # # update existing columns with missing data from the new available data
    # for c in [c for c in df.columns if c in df_result.columns and c[1] in ["dcf", "rates"]]:
    #     df_result[c] = df_result[c].combine_first(df[c])

    # merge by addition existing values with missing filled to zero
    m = [c for c in df.columns if c in df_result.columns]
    if len(m) > 0:
        df_result[m] = df_result[m].add(df[m], fill_value=0.0)

    # append new columns without additional calculation
    a = [c for c in df.columns if c not in df_result.columns]
    if len(a) > 0:
        df_result[a] = df[a]

    # df_result.columns = MultiIndex.from_tuples(df_result.columns)
    return df_result


class Fly(_BaseInstrument):
    """
    A *Butterfly* of :class:`~rateslib.instruments.protocols._BaseInstrument`.

    .. rubric:: Examples

    The following initialises a *Butterfly* of *IRSs*.

    .. ipython:: python
       :suppress:

       from rateslib.instruments import Fly, IRS
       from datetime import datetime as dt

    .. ipython:: python

       fly = Fly(
           instrument1=IRS(dt(2000, 1, 1), "1y", notional=10e6, spec="eur_irs", curves=["estr"]),
           instrument2=IRS(dt(2000, 1, 1), "2y", notional=-5e6, spec="eur_irs", curves=["estr"]),
           instrument3=IRS(dt(2000, 1, 1), "3y", notional=1.75e6, spec="eur_irs", curves=["estr"]),
       )
       fly.cashflows()

    .. rubric:: Pricing

    Each :class:`~rateslib.instruments.protocols._BaseInstrument` should have
    its own ``curves`` and ``vol`` objects set at its initialisation, according to the
    documentation for that *Instrument*. For the pricing methods ``curves`` and ``vol`` objects,
    these can be universally passed to each *Instrument* but in many cases that would be
    technically impossible since each *Instrument* might require difference pricing objects, e.g.
    if the *Instruments* have difference currencies. For a *Fly*
    of three *IRS* in the same currency this would be possible, however.

    Parameters
    ----------
    instrument1 : _BaseInstrument
        The *Instrument* with the shortest maturity.
    instrument2 : _BaseInstrument
        The *Instrument* with the intermediate maturity.
    instrument3 : _BaseInstrument
        The *Instrument* with the longest maturity.

    Notes
    -----
    A *Fly* is just a container for three
    :class:`~rateslib.instruments.protocols._BaseInstrument`, with an overload
    for the :meth:`~rateslib.instruments.Spread.rate` method to calculate twice the
    belly rate minus the wings (whatever metric is in use for each *Instrument*), which allows
    it to offer a lot of flexibility in *pseudo Instrument* creation.

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
        instrument3: _BaseInstrument,
    ) -> None:
        self._instruments = [instrument1, instrument2, instrument3]

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
        local_npv = self._npv_single_core(curves=curves, solver=solver, fx=fx, vol=vol, base=base)
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
        """
        TBD
        """
        df_result = DataFrame(index=DatetimeIndex([], name="obs_dates"))
        for inst in self.instruments:
            try:
                df = inst.local_analytic_rate_fixings(
                    curves=curves,
                    solver=solver,
                    fx=fx,
                    vol=vol,
                    forward=forward,
                    settlement=settlement,
                )
            except AttributeError:
                continue
            df_result = _composit_fixings_table(df_result, df)
        return df_result

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
        return (-rates[0] + 2 * rates[1] - rates[2]) * 100.0

    def analytic_delta(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`analytic_delta` is not defined for Portfolio.")
