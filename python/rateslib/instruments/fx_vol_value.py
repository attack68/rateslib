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

from typing import TYPE_CHECKING, NoReturn

from rateslib import defaults
from rateslib.enums.generics import NoInput, _drb
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface, FXSabrSmile, FXSabrSurface
from rateslib.instruments.protocols import _BaseInstrument
from rateslib.instruments.protocols.kwargs import _KWArgs
from rateslib.instruments.protocols.pricing import (
    _get_fx_forwards_maybe_from_solver,
    _maybe_get_fx_vol_maybe_from_solver,
    _Vol,
)
from rateslib.periods.utils import _validate_fx_as_forwards

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        CurvesT_,
        DualTypes,
        FXForwards_,
        Solver_,
        VolT_,
        datetime_,
        str_,
    )


class FXVolValue(_BaseInstrument):
    """
    A pseudo *Instrument* used to calibrate an *FX Vol Object* within a
    :class:`~rateslib.solver.Solver`.

    .. rubric:: Examples

    Examples
    --------
    The below :class:`~rateslib.fx_volatility.FXDeltaVolSmile` is solved directly
    from calibrating volatility values.

    .. ipython:: python
       :suppress:

       from rateslib.fx_volatility import FXDeltaVolSmile
       from rateslib.instruments import FXVolValue
       from rateslib.solver import Solver

    .. ipython:: python

       smile = FXDeltaVolSmile(
           nodes={0.3: 10.0, 0.7: 10.0},
           eval_date=dt(2023, 3, 16),
           expiry=dt(2023, 6, 16),
           delta_type="forward",
           id="VolSmile",
       )
       instruments = [
           FXVolValue(0.4, vol="VolSmile"),
           FXVolValue(0.6, vol=smile)
       ]
       solver = Solver(curves=[smile], instruments=instruments, s=[8.9, 7.8])
       smile[0.3]
       smile[0.4]
       smile[0.6]
       smile[0.7]

    .. rubric:: Pricing

    An *FX Vol Value* requires, and will calibrate, just one *FX Vol Object*.

    Allowable inputs are:

    .. code-block:: python

       vol = fx_vol_obj | [fx_vol_obj]  #  a single object is detected
       vol = {"fx_vol": fx_vol_obj}  # dict form is explicit

    Currently the only available ``metric`` is *'vol'* which returns the specific volatility value
    for the index value, i.e. a delta-index for a *DeltaVol* type object, or a strike for a
    *SABR* type object.

    .. role:: red

    .. role:: green

    Parameters
    ----------
    index_value : float, Dual, Dual2, :red:`required`
        The value of some index to the *FXVolSmile* or *FXVolSurface*.
    expiry: datetime, :green:`optional`
        The expiry at which to evaluate. This will only be used with *Surfaces*, not *Smiles*.
    metric: str, :green:`optional (set as 'vol')`
        The default metric to return from the ``rate`` method.
    vol: str, FXDeltaVolSmile, FXSabrSmile, FXDeltaVolSurface, FXSabrSurface, :green:`optional`
        The associated object from which to determine the ``rate``.

    """

    _rate_scalar = 1.0

    def __init__(
        self,
        index_value: DualTypes,
        expiry: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
        vol: VolT_ = NoInput(0),
    ):
        user_args = dict(
            expiry=expiry,
            index_value=index_value,
            vol=self._parse_vol(vol),
            metric=metric,
        )
        default_args = dict(convention=defaults.convention, metric="vol", curves=NoInput(0))
        self._kwargs = _KWArgs(
            spec=NoInput(0),
            user_args=user_args,
            default_args=default_args,
            meta_args=["curves", "metric", "vol"],
        )

    def _parse_vol(self, vol: VolT_) -> _Vol:
        if isinstance(vol, _Vol):
            return vol
        return _Vol(fx_vol=vol)

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
        """
        Return a value derived from a *Curve*.

        Parameters
        ----------
        curves : Curve, LineCurve, str or list of such
            Uses only one *Curve*, the one given or the first in the list.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            ``Curves`` from calibrating instruments.
        fx : float, FXRates, FXForwards, optional
            Not used.
        base : str, optional
            Not used.
        vol: float, Dual, Dual2, FXDeltaVolSmile or FXDeltaVolSurface
            The volatility used in calculation.
        metric: str in {"curve_value", "index_value", "cc_zero_rate"}, optional
            Configures which type of value to return from the applicable *Curve*.

        Returns
        -------
        float, Dual, Dual2, Variable

        """
        _vol: _Vol = self._parse_vol(vol)
        metric_ = _drb(self.kwargs.meta["metric"], metric).lower()

        if metric_ == "vol":
            vol_ = _maybe_get_fx_vol_maybe_from_solver(
                vol_meta=self.kwargs.meta["vol"], solver=solver, vol=_vol
            )
            if isinstance(vol_, FXDeltaVolSmile | FXDeltaVolSurface):
                # Must initialise with an ``expiry`` if a Surface is used
                return vol_._get_index(
                    delta_index=self.kwargs.leg1["index_value"], expiry=self.kwargs.leg1["expiry"]
                )
            elif isinstance(vol_, FXSabrSmile):
                fx_ = _validate_fx_as_forwards(
                    _get_fx_forwards_maybe_from_solver(solver=solver, fx=fx)
                )
                # if Sabr VolObj is not initialised with a `pair` this will create an error
                pair: str = vol_.meta.pair  # type: ignore[assignment]
                return vol_.get_from_strike(
                    k=self.kwargs.leg1["index_value"],
                    f=fx_.rate(pair=pair, settlement=vol_.meta.delivery),
                    expiry=self.kwargs.leg1["expiry"],
                )[1]
            elif isinstance(vol_, FXSabrSurface):
                fx_ = _validate_fx_as_forwards(
                    _get_fx_forwards_maybe_from_solver(solver=solver, fx=fx)
                )
                # if Sabr VolObj is not initialised with a `pair` this will create an error
                return vol_.get_from_strike(
                    k=self.kwargs.leg1["index_value"],
                    f=fx_,
                    expiry=self.kwargs.leg1["expiry"],
                )[1]
            else:
                raise ValueError("`vol` as an object must be provided for VolValue.")

        raise ValueError("`metric` must be in {'vol'}.")

    def npv(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`VolValue` instrument has no concept of NPV.")

    def cashflows(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`VolValue` instrument has no concept of cashflows.")

    def analytic_delta(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`VolValue` instrument has no concept of analytic delta.")
