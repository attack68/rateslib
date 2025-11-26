from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Protocol

from pandas import DataFrame, DatetimeIndex, concat

from rateslib.enums.generics import NoInput
from rateslib.instruments.components.protocols.pricing import (
    _get_fx_maybe_from_solver,
    _maybe_get_curve_or_dict_maybe_from_solver,
    _WithPricingObjs,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        CurvesT_,
        FXForwards_,
        FXVolOption_,
        Solver_,
        _Curves,
        _KWArgs,
        datetime_,
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


class _WithAnalyticRateFixings(_WithPricingObjs, Protocol):
    @property
    def kwargs(self) -> _KWArgs: ...

    def local_analytic_rate_fixings(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
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
        raise NotImplementedError(
            f"{type(self).__name__} must implement `local_analytic_rate_fixings`"
        )

    def _local_analytic_rate_fixings_from_legs(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        assert hasattr(self, "legs")  # noqa: S101

        # this is a generic implementation to handle 2 legs.
        _curves: _Curves = self._parse_curves(curves)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        _fx_maybe_from_solver = _get_fx_maybe_from_solver(fx=fx, solver=solver)

        dfs: list[DataFrame] = []
        for leg, names in zip(
            self.legs,
            [
                ("rate_curve", "disc_curve", "index_curve"),
                ("leg2_rate_curve", "leg2_disc_curve", "leg2_index_curve"),
            ],
            strict=False,
        ):
            dfs.append(
                leg.local_analytic_rate_fixings(
                    rate_curve=_maybe_get_curve_or_dict_maybe_from_solver(
                        _curves_meta, _curves, names[0], solver
                    ),
                    disc_curve=_maybe_get_curve_or_dict_maybe_from_solver(
                        _curves_meta, _curves, names[1], solver
                    ),
                    index_curve=_maybe_get_curve_or_dict_maybe_from_solver(
                        _curves_meta, _curves, names[2], solver
                    ),
                    fx=_fx_maybe_from_solver,
                    fx_vol=fx_vol,
                    settlement=settlement,
                    forward=forward,
                )
            )

        with warnings.catch_warnings():
            # TODO: pandas 2.1.0 has a FutureWarning for concatenating DataFrames with Null entries
            warnings.filterwarnings("ignore", category=FutureWarning)
            df = concat(dfs)
            return df.sort_index()

    def _local_analytic_rate_fixings_from_instruments(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        assert hasattr(self, "instruments")  # noqa: S101

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
