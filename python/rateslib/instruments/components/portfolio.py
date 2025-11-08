from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NoReturn

from pandas import DataFrame

from rateslib import defaults
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


def _instrument_npv(
    instrument: _BaseInstrument, *args: Any, **kwargs: Any
) -> DualTypes | dict[str, DualTypes]:  # pragma: no cover
    # this function is captured by TestPortfolio pooling but is not registered as a parallel process
    # used for parallel processing with Portfolio.npv
    return instrument.npv(*args, **kwargs)


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


class Portfolio(_BaseInstrument):
    """
    Create a collection of *Instruments* to group metrics

    Parameters
    ----------
    instruments : list
        This should be a list of *Instruments*.

    Notes
    -----
    When using a :class:`Portfolio` each *Instrument* must either have pricing parameters
    pre-defined using the appropriate :ref:`pricing mechanisms<mechanisms-doc>` or share
    common pricing parameters defined at price time.

    Examples
    --------
    See examples for :class:`Spread` for similar functionality.
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
        # if the pool is 1 do not do any parallel processing and return the single core func
        if defaults.pool == 1:
            local_npv: dict[str, DualTypes] = self._npv_single_core(
                curves=curves,
                solver=solver,
                fx=fx,
                fx_vol=fx_vol,
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
                fx_vol=fx_vol,
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
            local_npv = _.to_dict()

            # ret = {}
            # for result in results:
            #     for ccy in result:
            #         if ccy in ret:
            #             ret[ccy] += result[ccy]
            #         else:
            #             ret[ccy] = result[ccy]

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

    def rate(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`rate` is not defined for Portfolio.")

    def analytic_delta(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`analytic_delta` is not defined for Portfolio.")
