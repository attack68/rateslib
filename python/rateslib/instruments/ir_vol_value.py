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
from rateslib.instruments.irs import IRS
from rateslib.instruments.protocols import _BaseInstrument
from rateslib.instruments.protocols.kwargs import _KWArgs
from rateslib.instruments.protocols.pricing import (
    _maybe_get_ir_vol_maybe_from_solver,
    _Vol,
)
from rateslib.volatility.fx import FXVolObj
from rateslib.volatility.ir import IRSabrSmile

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Any,
        CurvesT_,
        DualTypes,
        FXForwards_,
        Solver_,
        VolT_,
        datetime_,
        str_,
    )


class IRVolValue(_BaseInstrument):
    """
    A pseudo *Instrument* used to calibrate an *IR Vol Object* within a
    :class:`~rateslib.solver.Solver`.

    .. rubric:: Examples

    Examples
    --------
    The below :class:`~rateslib.volatility.FXDeltaVolSmile` is solved directly
    from calibrating volatility values.

    .. ipython:: python
       :suppress:

       from rateslib.volatility import IRSabrSmile
       from rateslib.instruments import IRVolValue
       from rateslib.solver import Solver

    ..
        .. ipython:: python

           smile = IRSabrSmile(
               nodes={"alpha": 0.20, "beta": 0.5, "rho": 0.05, "nu": 0.60},
               eval_date=dt(2026, 2, 12),
               tenor="1y",
               expiry=dt(2027, 2, 12),
               irs_series="usd_irs",
               id="VolSmile",
           )
           instruments = [
               IRVolValue(2.5, vol="VolSmile"),
               IRVolValue(3.5, vol=smile)
           ]
           solver = Solver(curves=[smile], instruments=instruments, s=[8.9, 7.8])
           smile[2.1]
           smile[2.5]
           smile[3.5]
           smile[3.9]

    .. rubric:: Pricing

    An *IR Vol Value* requires, and will calibrate, just one *IR Vol Object*.

    Allowable inputs are:

    .. code-block:: python

       vol = ir_vol_obj | [ir_vol_obj]  #  a single object is detected
       vol = {"ir_vol": ir_vol_obj}  # dict form is explicit

    The ``curves`` must match the pricing for an :class:`~rateslib.instruments.IRS`, since the
    atm-rate is determined directly from an *IRS* instance.

    Currently the only available ``metric`` is *'vol'* which returns the specific volatility value
    for the index value, i.e. a strike for an :class:`~rateslib.instruments.IRS`.

    .. role:: red

    .. role:: green

    Parameters
    ----------
    index_value : float, Dual, Dual2, :red:`required`
        The value of some index to the *IRVolSmile* or *IRVolSurface*.
    expiry: datetime, :green:`optional`
        The expiry at which to evaluate. This will only be used with *Surfaces*, not *Smiles*.
    metric: str, :green:`optional (set as 'vol')`
        The default metric to return from the ``rate`` method.
    vol: str, IRSabrSmile, :green:`optional`
        The associated object from which to determine the ``rate``.
    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.

    """

    _rate_scalar = 1.0

    def __init__(
        self,
        index_value: DualTypes,
        expiry: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
    ):
        user_args = dict(
            expiry=expiry,
            index_value=index_value,
            vol=self._parse_vol(vol),
            metric=metric,
            curves=IRS._parse_curves(curves),
        )
        default_args = dict(convention=defaults.convention, metric="vol", curves=NoInput(0))
        self._kwargs = _KWArgs(
            spec=NoInput(0),
            user_args=user_args,
            default_args=default_args,
            meta_args=["curves", "metric", "vol", "curves"],
        )

    def _parse_vol(self, vol: VolT_) -> _Vol:
        if isinstance(vol, _Vol):
            return vol
        elif isinstance(vol, FXVolObj):
            raise TypeError(
                f"`vol` must be suitable object for IR vol pricing. Got {type(vol).__name__}"
            )
        return _Vol(ir_vol=vol)

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
        _vol: _Vol = self._parse_vol(vol)
        metric_ = _drb(self.kwargs.meta["metric"], metric).lower()

        if metric_ == "vol":
            vol_ = _maybe_get_ir_vol_maybe_from_solver(
                vol_meta=self.kwargs.meta["vol"], solver=solver, vol=_vol
            )

            if isinstance(vol_, IRSabrSmile):
                return vol_.get_from_strike(
                    k=self.kwargs.leg1["index_value"],
                    curves=curves,
                ).vol
            else:
                raise ValueError("`vol` as an object must be provided for VolValue.")

        raise ValueError("`metric` must be in {'vol'}.")

    def npv(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`VolValue` instrument has no concept of NPV.")

    def cashflows(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`VolValue` instrument has no concept of cashflows.")

    def analytic_delta(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("`VolValue` instrument has no concept of analytic delta.")
