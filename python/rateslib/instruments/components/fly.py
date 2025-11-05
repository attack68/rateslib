from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NoReturn

from pandas import DataFrame, DatetimeIndex

from rateslib.enums.generics import NoInput
from rateslib.instruments.components.protocols import _BaseInstrument
from rateslib.instruments.components.protocols.utils import (
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
    Create a butterfly of *Instruments*.

    Parameters
    ----------
    instrument1 : _BaseInstrument
        An *Instrument* with the shortest maturity.
    instrument2 : _BaseInstrument
        The *Instrument* of the body of the *Fly*.
    instrument3 : _BaseInstrument
        An *Instrument* with the longest maturity.
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
            curves=curves, solver=solver, fx=fx, fx_vol=fx_vol, base=base
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
        """
        Return a DataFrame of financial sensitivity to published interest rate fixings,
        expressed in local **settlement currency** of the *Period*.

        If the *Period* has no sensitivity to rates fixings this *DataFrame* is empty.

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
        settlement: datetime, optional
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, optional
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        DataFrame
        """
        df_result = DataFrame(index=DatetimeIndex([], name="obs_dates"))
        for inst in self.instruments:
            try:
                df = inst.local_analytic_rate_fixings(
                    curves=curves,
                    solver=solver,
                    fx=fx,
                    fx_vol=fx_vol,
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

    def fixings_table(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        approximate: bool = False,
        right: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on the *Instruments*.

        For arguments see :meth:`XCS.fixings_table()<rateslib.instruments.XCS.fixings_table>`,
        and/or :meth:`IRS.fixings_table()<rateslib.instruments.IRS.fixings_table>`

        Returns
        -------
        DataFrame
        """
        df_result = DataFrame(
            index=DatetimeIndex([], name="obs_dates"),
        )
        for inst in self.instruments:
            try:
                df = inst.fixings_table(  # type: ignore[attr-defined]
                    curves=curves,
                    solver=solver,
                    fx=fx,
                    base=base,
                    approximate=approximate,
                    right=right,
                )
            except AttributeError:
                continue
            df_result = _composit_fixings_table(df_result, df)
        return df_result

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
        return (-rates[0] + 2 * rates[1] - rates[2]) * 100.0

    def analytic_delta(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`analytic_delta` is not defined for Portfolio.")
