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

import warnings
from typing import TYPE_CHECKING, Protocol

from pandas import DataFrame, concat, isna

from rateslib import defaults
from rateslib.enums.generics import NoInput
from rateslib.instruments.protocols.kwargs import _KWArgs
from rateslib.instruments.protocols.pricing import (
    _get_fx_maybe_from_solver,
    _maybe_get_curve_object_maybe_from_solver,
    _maybe_get_curve_or_dict_object_maybe_from_solver,
    _maybe_get_fx_vol_maybe_from_solver,
    _WithPricingObjs,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        Any,
        CurvesT_,
        FXForwards_,
        Solver_,
        VolT_,
        _Curves,
        _Vol,
        datetime_,
        str_,
    )


class _WithCashflows(_WithPricingObjs, Protocol):
    """
    Protocol to determine cashflows for any *Instrument* type.
    """

    _kwargs: _KWArgs

    @property
    def kwargs(self) -> _KWArgs:
        return self._kwargs

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
        """
        Return aggregated cashflow data for the *Instrument*.

        .. warning::

           This method is a convenience method to provide a visual representation of all
           associated calculation data. Calling this method to extract certain values
           should be avoided. It is more efficient to source relevant parameters or calculations
           from object attributes or other methods directly.

        .. rubric:: Examples

        .. ipython:: python
           :suppress:

           from rateslib import dt, Curve, IRS

        .. ipython:: python

           irs = IRS(dt(2000, 1, 1), "3Y", spec="usd_irs", fixed_rate=1.0)
           irs.cashflows()

        Providing relevant pricing objects will ensure all data that can be calculated is returned.

        .. ipython:: python

           curve = Curve({dt(2000, 1, 1): 1.0, dt(2010, 1, 1): 0.75})
           irs.cashflows(curves=[curve])

        Parameters
        ----------
        curves: _Curves, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        solver: Solver, :green:`optional`
            A :class:`~rateslib.solver.Solver` object containing *Curve*, *Smile*, *Surface*, or
            *Cube* mappings for pricing.
        fx: FXForwards, :green:`optional`
            The :class:`~rateslib.fx.FXForwards` object used for forecasting FX rates, if necessary.
        vol: _Vol, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        base: str, :green:`optional (set to settlement currency)`
            The currency to convert the *local settlement* NPV to.
        settlement: datetime, :green:`optional`
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, :green:`optional`
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        DataFrame
        """
        raise NotImplementedError(f"{type(self).__name__} must implement `cashflows`.")

    def _cashflows_from_legs(
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
        """
        Return aggregated cashflow data for the *Period*.

        .. warning::

           This method is a convenience method to provide a visual representation of all
           associated calculation data. Calling this method to extracting certain values
           should be avoided. It is more efficent to source relevant parameters or calculations
           from object attributes or other methods directly.

        Returns
        -------
        dict of values
        """
        # this is a generalist implementation of an NPV function for an instrument with 2 legs.
        # most instruments may be likely to implement NPV directly to benefit from optimisations
        # specific to that instrument
        assert hasattr(self, "legs")  # noqa: S101

        _curves: _Curves = self._parse_curves(curves)
        _vol: _Vol = self._parse_vol(vol)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        _vol_meta: _Vol = self.kwargs.meta["vol"]
        _fx_maybe_from_solver = _get_fx_maybe_from_solver(fx=fx, solver=solver)

        fx_vol = _maybe_get_fx_vol_maybe_from_solver(_vol_meta, _vol, solver)
        legs_df = [
            self.legs[0].cashflows(
                rate_curve=_maybe_get_curve_or_dict_object_maybe_from_solver(
                    _curves_meta, _curves, "rate_curve", solver
                ),
                disc_curve=_maybe_get_curve_object_maybe_from_solver(
                    _curves_meta, _curves, "disc_curve", solver
                ),
                index_curve=_maybe_get_curve_object_maybe_from_solver(
                    _curves_meta, _curves, "index_curve", solver
                ),
                fx=_fx_maybe_from_solver,
                fx_vol=fx_vol,
                settlement=settlement,
                forward=forward,
                base=base,
            )
        ]

        if len(self.legs) > 1:
            legs_df.append(
                self.legs[1].cashflows(
                    rate_curve=_maybe_get_curve_or_dict_object_maybe_from_solver(
                        _curves_meta, _curves, "leg2_rate_curve", solver
                    ),
                    disc_curve=_maybe_get_curve_object_maybe_from_solver(
                        _curves_meta, _curves, "leg2_disc_curve", solver
                    ),
                    index_curve=_maybe_get_curve_object_maybe_from_solver(
                        _curves_meta, _curves, "leg2_index_curve", solver
                    ),
                    fx=_fx_maybe_from_solver,
                    fx_vol=fx_vol,
                    settlement=settlement,
                    forward=forward,
                    base=base,
                )
            )

        # filter empty or all NaN
        dfs_filtered = [_ for _ in legs_df if not (_.empty or isna(_).all(axis=None))]

        with warnings.catch_warnings():
            # TODO: pandas 2.1.0 has a FutureWarning for concatenating DataFrames with Null entries
            warnings.filterwarnings("ignore", category=FutureWarning)
            _: DataFrame = concat(dfs_filtered, keys=["leg1", "leg2"])
        return _

    def _cashflows_from_instruments(self, *args: Any, **kwargs: Any) -> DataFrame:
        # this is a generalist implementation of an NPV function for an instrument with 2 legs.
        # most instruments may be likely to implement NPV directly to benefit from optimisations
        # specific to that instrument
        assert hasattr(self, "instruments")  # noqa: S101

        with warnings.catch_warnings():
            # TODO: pandas 2.1.0 has a FutureWarning for concatenating DataFrames with Null entries
            warnings.filterwarnings("ignore", category=FutureWarning)
            _: DataFrame = concat(
                [_.cashflows(*args, **kwargs) for _ in self.instruments],
                keys=[f"inst{i}" for i in range(len(self.instruments))],
            )
        return _

    def cashflows_table(
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
        """
        Aggregate the values derived from a
        :meth:`~rateslib.instruments.protocols._WithCashflows.cashflows`, grouped by date,
        settlement currency and collateral.

        .. rubric:: Examples

        .. ipython:: python
           :suppress:

           from rateslib import dt, Curve, IRS

        .. ipython:: python

           irs = IRS(dt(2000, 1, 1), "3Y", spec="usd_irs", fixed_rate=1.0)
           curve = Curve({dt(2000, 1, 1): 1.0, dt(2010, 1, 1): 0.75})
           irs.cashflows_table(curves=[curve])

        Parameters
        ----------
        curves: _Curves, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        solver: Solver, :green:`optional`
            A :class:`~rateslib.solver.Solver` object containing *Curve*, *Smile*, *Surface*, or
            *Cube* mappings for pricing.
        fx: FXForwards, :green:`optional`
            The :class:`~rateslib.fx.FXForwards` object used for forecasting FX rates, if necessary.
        vol: _Vol, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        base: str, :green:`optional (set to settlement currency)`
            The currency to convert the *local settlement* NPV to.
        local: bool, :green:`optional (set as False)`
            An override flag to return a dict of NPV values indexed by string currency.
        settlement: datetime, :green:`optional`
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, :green:`optional`
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        DataFrame
        """
        cashflows = self.cashflows(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            settlement=settlement,
            forward=forward,
        )
        cashflows = cashflows[
            [
                defaults.headers["currency"],
                defaults.headers["collateral"],
                defaults.headers["payment"],
                defaults.headers["cashflow"],
            ]
        ]
        _: DataFrame = cashflows.groupby(  # type: ignore[assignment]
            [
                defaults.headers["currency"],
                defaults.headers["collateral"],
                defaults.headers["payment"],
            ],
            dropna=False,
        )
        _ = _.sum().unstack([0, 1]).droplevel(0, axis=1)
        _.columns.names = ["local_ccy", "collateral_ccy"]
        _.index.names = ["payment"]
        _ = _.sort_index(ascending=True, axis=0).infer_objects().fillna(0.0)
        return _
