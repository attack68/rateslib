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

from functools import cached_property
from typing import TYPE_CHECKING, NoReturn

from rateslib import defaults
from rateslib.data.fixings import _get_irs_series
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import OptionType
from rateslib.instruments.irs import IRS
from rateslib.instruments.protocols import _BaseInstrument
from rateslib.instruments.protocols.kwargs import _KWArgs
from rateslib.instruments.protocols.pricing import (
    _maybe_get_ir_vol_maybe_from_solver,
    _Vol,
)
from rateslib.periods.parameters import _IROptionParams
from rateslib.scheduling import add_tenor
from rateslib.volatility.fx import FXVolObj
from rateslib.volatility.ir import IRSabrCube, IRSabrSmile

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Any,
        CurvesT_,
        DualTypes,
        FXForwards_,
        IRSSeries,
        Solver_,
        VolT_,
        datetime,
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
    strike: float, Variable, str, :red:`required`
        The strike value used as the index value to the pricing model.
        If str, there are two possibilities; {"atm", "{}bps"}. "atm" will produce a strike equal
        to the mid-market *IRS* rate, whilst "20bps" or "-50bps" will yield a strike that number
        of basis points different to the mid-market rate.
    expiry: datetime, str, :red:`required`
        The expiry of the option. If given in string tenor format, e.g. "1M" requires an
        ``eval_date``. See **Notes**.
    tenor: datetime, str, :red:`required`
        The parameter defining the maturity of the underlying :class:`~rateslib.instruments.IRS`.
    irs_series: IRSSeries, str, :red:`required`
        The standard conventions applied to the underlying :class:`~rateslib.instruments.IRS`.
    eval_date: datetime, :green:`optional`
        If expiry is given as string tenor, use eval date to determine the date.
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
        strike: DualTypes | str,
        expiry: datetime | str,
        tenor: datetime | str,
        irs_series: IRSSeries | str,
        *,
        eval_date: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
    ):
        user_args = dict(
            tenor=tenor,
            expiry=expiry,
            strike=strike,
            irs_series=irs_series,
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
        if isinstance(self.kwargs.leg1["expiry"], str):
            if isinstance(eval_date, NoInput):
                raise ValueError("`tenor` as string requires an `eval_date` to quantify.")
            series_ = _get_irs_series(self.kwargs.leg1["irs_series"])
            self.kwargs.leg1["expiry"] = add_tenor(
                start=eval_date,
                tenor=self.kwargs.leg1["expiry"],
                modifier=series_.modifier,
                calendar=series_.calendar,
            )

    @cached_property
    def _ir_option_params(self) -> _IROptionParams:
        return _IROptionParams(
            _expiry=self.kwargs.leg1["expiry"],
            _tenor=self.kwargs.leg1["tenor"],
            _irs_series=_get_irs_series(self.kwargs.leg1["irs_series"]),
            _strike=self.kwargs.leg1["strike"],
            # unused parameters
            _direction=OptionType.Put,
            _metric=defaults.ir_option_metric,
            _option_fixings=NoInput(0),
            _settlement_method=defaults.ir_option_settlement,
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
        vol_ = _maybe_get_ir_vol_maybe_from_solver(
            vol_meta=self.kwargs.meta["vol"], solver=solver, vol=_vol
        )

        metric_ = _drb(self.kwargs.meta["metric"], metric).lower()

        if metric_ == "vol":
            if isinstance(vol_, (IRSabrSmile, IRSabrCube)):
                return vol_.get_from_strike(
                    k=self.kwargs.leg1["strike"],
                    curves=curves,
                    expiry=self.kwargs.leg1["expiry"],
                    tenor=self._ir_option_params.option_fixing.termination,
                ).vol
            else:
                raise ValueError("`vol` as an object must be provided for VolValue.")
        elif metric_ in ["alpha", "beta", "rho", "nu"]:
            if isinstance(vol_, IRSabrSmile | IRSabrCube):
                return vol_._get_sabr_param(
                    expiry=self.kwargs.leg1["expiry"],
                    tenor=self._ir_option_params.option_fixing.termination,
                    param=metric_,
                )
            else:
                raise ValueError(
                    "A SABR parameter `metric` can only be obtained from a SABR type "
                    "IR Volatility object."
                )

        raise ValueError("`metric` must be in {'vol', 'alpha', 'beta', 'rho', 'nu'}.")

    def npv(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError(
            "`VolValue` instrument has no concept of NPV."
        )  # pragma: no cover

    def cashflows(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError(
            "`VolValue` instrument has no concept of cashflows."
        )  # pragma: no cover

    def analytic_delta(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError(
            "`VolValue` instrument has no concept of analytic delta."
        )  # pragma: no cover
