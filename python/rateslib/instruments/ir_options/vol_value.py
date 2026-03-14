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
from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.data.fixings import _get_irs_series
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import OptionPricingModel, OptionType, _get_ir_option_metric
from rateslib.instruments.ir_options.call_put import _BaseIROption
from rateslib.instruments.protocols import _BaseInstrument
from rateslib.instruments.protocols.kwargs import _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _maybe_get_curve_maybe_from_solver,
    _maybe_get_ir_vol_maybe_from_solver,
    _Vol,
)
from rateslib.periods.parameters import _IROptionParams
from rateslib.periods.utils import (
    _get_ir_vol_value_and_forward_maybe_from_obj,
)
from rateslib.rs import IROptionMetric
from rateslib.scheduling import add_tenor
from rateslib.volatility.ir import IRSabrCube, IRSabrSmile
from rateslib.volatility.utils import _OptionModelBachelier, _OptionModelBlack76

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Any,
        CurvesT_,
        DualTypes,
        FXForwards_,
        IRSSeries,
        Solver_,
        VolT_,
        _BaseCurve,
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

    The available ``metric`` are:

    - **'normal_vol'**: which returns a normal volatility in bps suitable for the Bachelier pricing
      formula.
    - **'black_vol_shift_{}'**: same as above but allowing an explicit shift.
    - **'alpha', 'beta', 'rho', 'nu'**: returns the SABR parameters explicitly for a SABR based
      pricing object.

    .. role:: red

    .. role:: green

    Parameters
    ----------
    expiry: datetime, str, :red:`required`
        The expiry of the option. If given in string tenor format, e.g. "1M" requires an
        ``eval_date``. See **Notes**.
    tenor: datetime, str, :red:`required`
        The parameter defining the maturity of the underlying :class:`~rateslib.instruments.IRS`.
    strike: float, Variable, str, :red:`required`
        The strike value used as the index value to the pricing model.
        If str, there are two possibilities; {"atm", "{}bps"}. "atm" will produce a strike equal
        to the mid-market *IRS* rate, whilst "20bps" or "-50bps" will yield a strike that number
        of basis points different to the mid-market rate.
    irs_series: IRSSeries, str, :red:`required`
        The standard conventions applied to the underlying :class:`~rateslib.instruments.IRS`.
    eval_date: datetime, :green:`optional`
        If expiry is given as string tenor, use eval date to determine the date.
    metric: str, IROptionMetric, :green:`optional (set as 'normal_vol')`
        The default metric to return from the ``rate`` method.
    vol: str, IRVolObj, :green:`optional`
        The associated object from which to determine the ``rate``.
    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.

    """

    @property
    def rate_scalar(self) -> float:
        metric_ = self.kwargs.meta["metric"].lower()
        match metric_:
            case "alpha" | "beta" | "rho" | "nu":
                return 1.0
            case "normal_vol":
                return 100.0
            case _ if "black_vol_shift_" in metric_:
                return 100.0
            case _:
                raise NotImplementedError(
                    "The provided metric for IRVolValue is not rate scalar mapped."
                )

    _rate_scalar = 1.0

    def __init__(
        self,
        expiry: datetime | str,
        tenor: datetime | str,
        strike: DualTypes | str,
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
            curves=self._parse_curves(curves),
        )
        default_args = dict(convention=defaults.convention, metric="normal_vol", curves=NoInput(0))
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
        ir_vol = _maybe_get_ir_vol_maybe_from_solver(
            vol=self._parse_vol(vol), vol_meta=self.kwargs.meta["vol"], solver=solver
        )

        metric_ = _drb(self.kwargs.meta["metric"], metric).lower()
        del metric

        if metric_ in ["alpha", "beta", "rho", "nu"]:
            if isinstance(ir_vol, IRSabrSmile):
                return getattr(ir_vol.nodes, metric_)  # type: ignore[no-any-return]
            elif isinstance(ir_vol, IRSabrCube):
                smile: IRSabrSmile = ir_vol.get_smile(  # type: ignore[assignment]
                    expiry=self.kwargs.leg1["expiry"],
                    tenor=self._ir_option_params.option_fixing.termination,
                )
                return getattr(smile.nodes, metric_)  # type: ignore[no-any-return]
            else:
                raise ValueError(
                    "A SABR parameter `metric` can only be obtained from a SABR type vol pricing "
                    "object."
                )

        _curves = self._parse_curves(curves)
        rate_curve = _maybe_get_curve_maybe_from_solver(
            curves=_curves, curves_meta=self.kwargs.meta["curves"], solver=solver, name="rate_curve"
        )
        disc_curve: _BaseCurve = _validate_obj_not_no_input(
            _maybe_get_curve_maybe_from_solver(
                curves=_curves,
                curves_meta=self.kwargs.meta["curves"],
                solver=solver,
                name="disc_curve",
            ),
            name="disc_curve",
        )
        index_curve: _BaseCurve = _validate_obj_not_no_input(
            _maybe_get_curve_maybe_from_solver(
                curves=_curves,
                curves_meta=self.kwargs.meta["curves"],
                solver=solver,
                name="index_curve",
            ),
            name="index_curve",
        )

        metric__ = _get_ir_option_metric(metric_)
        del metric_

        if not hasattr(ir_vol, "get_from_strike"):
            raise TypeError("`vol` for IRVolValue must be of type _BaseIRSmile or _BaseIRCube.")

        params = _get_ir_vol_value_and_forward_maybe_from_obj(
            rate_curve=rate_curve,
            index_curve=index_curve,
            strike=self.kwargs.leg1["strike"],
            ir_vol=ir_vol,
            irs=self._ir_option_params.option_fixing.irs,
            tenor=self._ir_option_params.option_fixing.termination,
            expiry=self._ir_option_params.expiry,
            t_e=ir_vol.meta._t_expiry(self._ir_option_params.expiry),  # type: ignore[union-attr]
        )

        match type(metric__):
            case IROptionMetric.Cash | IROptionMetric.PercentNotional:
                raise ValueError(
                    "`metric` cannot be a cash or monetary quantity for this Instrument type"
                )
            case IROptionMetric.NormalVol:
                if params.pricing_model == OptionPricingModel.Bachelier:
                    return params.vol
                else:
                    return _OptionModelBlack76.convert_to_bachelier(
                        f=params.f, k=params.k, shift=params.shift, t_e=params.t_e, vol=params.vol
                    )
            case IROptionMetric.BlackVolShift:
                required_shift = metric__.shift()
                if params.pricing_model == OptionPricingModel.Bachelier:
                    return _OptionModelBachelier.convert_to_black76(
                        f=params.f, k=params.k, shift=required_shift, t_e=params.t_e, vol=params.vol
                    )
                else:
                    return _OptionModelBlack76.convert_to_new_shift(
                        f=params.f,
                        k=params.k,
                        old_shift=params.shift,
                        target_shift=required_shift,
                        t_e=params.t_e,
                        vol=params.vol,
                    )
            case _:
                raise RuntimeError(  # pragma: no cover
                    "Unexpected error: unmapped IROptionMetric branch - please report."
                )

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
        return _BaseIROption._parse_curves(curves)

    def _parse_vol(self, vol: VolT_) -> _Vol:
        return _BaseIROption._parse_vol(vol)

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
