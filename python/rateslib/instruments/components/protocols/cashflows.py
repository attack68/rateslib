from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Protocol

from pandas import DataFrame, concat, isna

from rateslib.enums.generics import NoInput
from rateslib.instruments.components.protocols.curves import _WithCurves
from rateslib.instruments.components.protocols.kwargs import _KWArgs
from rateslib.instruments.components.protocols.utils import (
    _get_curve_maybe_from_solver,
    _get_fx_maybe_from_solver,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        Any,
        Curves_,
        FXForwards_,
        FXVolOption_,
        Solver_,
        _Curves,
        datetime_,
        str_,
    )


class _WithCashflows(_WithCurves, Protocol):
    _kwargs: _KWArgs

    @property
    def kwargs(self) -> _KWArgs:
        return self._kwargs

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
        raise NotImplementedError(f"{type(self).__name__} must implement `cashflows`.")

    def _cashflows_from_legs(
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
        # this is a generalist implementation of an NPV function for an instrument with 2 legs.
        # most instruments may be likely to implement NPV directly to benefit from optimisations
        # specific to that instrument
        assert hasattr(self, "legs")

        _curves: _Curves = self._parse_curves(curves)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        _fx_maybe_from_solver = _get_fx_maybe_from_solver(fx, solver)

        leg1_df = self.legs[0].cashflows(
            rate_curve=_get_curve_maybe_from_solver(_curves_meta, _curves, "rate_curve", solver),
            disc_curve=_get_curve_maybe_from_solver(_curves_meta, _curves, "disc_curve", solver),
            index_curve=_get_curve_maybe_from_solver(_curves_meta, _curves, "index_curve", solver),
            fx=_fx_maybe_from_solver,
            fx_vol=fx_vol,
            settlement=settlement,
            forward=forward,
            base=base,
        )

        leg2_df = self.legs[1].cashflows(
            rate_curve=_get_curve_maybe_from_solver(
                _curves_meta, _curves, "leg2_rate_curve", solver
            ),
            disc_curve=_get_curve_maybe_from_solver(
                _curves_meta, _curves, "leg2_disc_curve", solver
            ),
            index_curve=_get_curve_maybe_from_solver(
                _curves_meta, _curves, "leg2_index_curve", solver
            ),
            fx=_fx_maybe_from_solver,
            fx_vol=fx_vol,
            settlement=settlement,
            forward=forward,
            base=base,
        )

        # filter empty or all NaN
        dfs_filtered = [_ for _ in [leg1_df, leg2_df] if not (_.empty or isna(_).all(axis=None))]

        with warnings.catch_warnings():
            # TODO: pandas 2.1.0 has a FutureWarning for concatenating DataFrames with Null entries
            warnings.filterwarnings("ignore", category=FutureWarning)
            _: DataFrame = concat(dfs_filtered, keys=["leg1", "leg2"])
        return _

    def _cashflows_from_instruments(self, *args: Any, **kwargs: Any) -> DataFrame:
        # this is a generalist implementation of an NPV function for an instrument with 2 legs.
        # most instruments may be likely to implement NPV directly to benefit from optimisations
        # specific to that instrument
        assert hasattr(self, "instruments")

        _: DataFrame = concat(
            [_.cashflows(*args, **kwargs) for _ in self.instruments],
            keys=[f"inst{i}" for i in range(len(self.instruments))],
        )
        return _
