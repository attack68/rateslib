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

from abc import ABCMeta
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import (
    IROptionMetric,
    SwaptionSettlementMethod,
    _get_ir_option_metric,
)
from rateslib.instruments.irs import IRS
from rateslib.instruments.protocols import _BaseInstrument, _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _maybe_get_curve_maybe_from_solver,
    _maybe_get_ir_vol_maybe_from_solver,
    _Vol,
)
from rateslib.legs import CustomLeg
from rateslib.periods import Cashflow, IRSCallPeriod, IRSPutPeriod
from rateslib.periods.utils import (
    _get_ir_vol_value_and_forward_maybe_from_obj,
)
from rateslib.volatility.fx import FXVolObj
from rateslib.volatility.ir import IRSabrSmile
from rateslib.volatility.ir.utils import _get_ir_expiry_and_payment

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Any,
        CurveOption_,
        CurvesT_,
        DataFrame,
        DualTypes,
        DualTypes_,
        FXForwards_,
        IROptionMetric,
        IRSSeries,
        Sequence,
        Solver_,
        VolT_,
        _BaseCurve,
        _BaseCurve_,
        _BaseIRSOptionPeriod,
        _BaseLeg,
        _IRVolOption_,
        datetime_,
        str_,
    )


@dataclass
class _IRVolPricingMetrics:
    """None elements are used as flags to indicate an element is not yet set."""

    vol: DualTypes
    k: DualTypes
    t_e: DualTypes
    f: DualTypes
    shift: DualTypes


class _BaseIROption(_BaseInstrument, metaclass=ABCMeta):
    """
    Abstract base class for implementing *IROptions*.

    See :class:`~rateslib.instruments.PayerSwaption` and
    :class:`~rateslib.instruments.ReceiverSwaption`.
    """

    _rate_scalar: float = 1.0
    _pricing: _IRVolPricingMetrics

    @property
    def leg1(self) -> CustomLeg:
        """The :class:`~rateslib.legs.CustomLeg` of the *Instrument* containing the
        :class:`~rateslib.periods.FXOptionPeriod`."""
        return self._leg1

    @property
    def leg2(self) -> CustomLeg:
        """The :class:`~rateslib.legs.CustomLeg` of the *Instrument* containing the
        premium :class:`~rateslib.periods.Cashflow`."""
        return self._leg2

    @property
    def legs(self) -> Sequence[_BaseLeg]:
        """A list of the *Legs* of the *Instrument*."""
        return self._legs

    @property
    def _option(self) -> _BaseIRSOptionPeriod:
        return self.leg1.periods[0]  # type: ignore[return-value]

    @property
    def _irs(self) -> IRS:
        return self._option.ir_option_params.option_fixing.irs

    @property
    def _premium(self) -> Cashflow:
        return self.leg2.periods[0]  # type: ignore[return-value]

    @classmethod
    def _parse_curves(cls, curves: CurvesT_) -> _Curves:
        """
        A Swaption has 3 curve requirements. See **Pricing**.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        elif isinstance(curves, list | tuple):
            if len(curves) == 1:
                return _Curves(
                    rate_curve=curves[0],
                    index_curve=curves[0],
                    disc_curve=curves[0],
                    leg2_disc_curve=curves[0],
                )
            elif len(curves) == 2:
                return _Curves(
                    rate_curve=curves[0],
                    disc_curve=curves[1],
                    index_curve=curves[1],
                    leg2_disc_curve=curves[1],
                )
            elif len(curves) == 3:
                return _Curves(
                    rate_curve=curves[0],
                    disc_curve=curves[1],
                    index_curve=curves[2],
                    leg2_disc_curve=curves[1],
                )
            else:
                raise ValueError(
                    f"{type(cls).__name__} requires only 2 curve types. Got {len(curves)}."
                )
        elif isinstance(curves, dict):
            return _Curves(
                rate_curve=curves.get("rate_curve", NoInput(0)),
                disc_curve=curves.get("disc_curve", NoInput(0)),
                index_curve=curves.get("index_curve", NoInput(0)),
                leg2_disc_curve=_drb(
                    curves.get("disc_curve", NoInput(0)),
                    curves.get("leg2_disc_curve", NoInput(0)),
                ),
            )
        elif isinstance(curves, _Curves):
            return curves
        else:  # `curves` is just a single input which is copied across all curves
            return _Curves(
                rate_curve=curves,  # type: ignore[arg-type]
                disc_curve=curves,  # type: ignore[arg-type]
                index_curve=curves,  # type: ignore[arg-type]
                leg2_disc_curve=curves,  # type: ignore[arg-type]
            )

    @classmethod
    def _parse_vol(cls, vol: VolT_) -> _Vol:
        """
        FXoptions requires only a single FXVolObj or a scalar.
        """
        if isinstance(vol, _Vol):
            return vol
        elif isinstance(vol, FXVolObj):
            raise TypeError("`vol` cannot be an FX type vol object and must be IR type vol object.")
        else:
            return _Vol(ir_vol=vol)

    def __init__(
        self,
        expiry: datetime | str,
        tenor: datetime | str,
        strike: DualTypes | str,
        irs_series: IRSSeries | str,
        *,
        notional: DualTypes_ = NoInput(0),
        eval_date: datetime | NoInput = NoInput(0),
        premium: DualTypes_ = NoInput(0),
        payment_lag: str | datetime_ = NoInput(0),
        option_fixings: DualTypes_ = NoInput(0),
        settlement_method: SwaptionSettlementMethod | str_ = NoInput(0),
        metric: IROptionMetric | str_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        spec: str_ = NoInput(0),
        call: bool = True,
    ):
        user_args = dict(
            tenor=tenor,
            expiry=expiry,
            notional=notional,
            strike=strike,
            irs_series=irs_series,
            option_fixings=option_fixings,
            settlement_method=settlement_method,
            leg2_payment_lag=payment_lag,
            leg2_premium=premium,
            metric=metric,
            curves=self._parse_curves(curves),
            vol=self._parse_vol(vol),
        )
        # instrument_args: dict[str, Any] = dict()
        default_args = dict(
            notional=defaults.notional,
            metric=defaults.ir_option_metric,
            settlement_method=defaults.ir_option_settlement,
        )
        self._kwargs = _KWArgs(
            user_args=user_args,
            default_args=default_args,
            spec=spec,
            meta_args=["curves", "vol", "metric"],
        )

        # determine the `expiry` and `delivery` as datetimes if derived from other combinations
        (self.kwargs.leg1["expiry"], self.kwargs.leg2["payment"]) = _get_ir_expiry_and_payment(
            eval_date=eval_date,
            expiry=self.kwargs.leg1["expiry"],
            irs_series=self.kwargs.leg1["irs_series"],
            payment_lag=self.kwargs.leg2["payment_lag"],
        )

        self._leg1 = CustomLeg(
            [
                IRSCallPeriod(  # type: ignore[abstract]
                    expiry=self.kwargs.leg1["expiry"],
                    tenor=self.kwargs.leg1["tenor"],
                    irs_series=self.kwargs.leg1["irs_series"],
                    strike=NoInput(0)
                    if isinstance(self.kwargs.leg1["strike"], str)
                    else self.kwargs.leg1["strike"],
                    notional=self.kwargs.leg1["notional"],
                    option_fixings=self.kwargs.leg1["option_fixings"],
                    metric=self.kwargs.meta["metric"],
                    settlement_method=self.kwargs.leg1["settlement_method"],
                )
                if call
                else IRSPutPeriod(  # type: ignore[abstract]
                    expiry=self.kwargs.leg1["expiry"],
                    tenor=self.kwargs.leg1["tenor"],
                    irs_series=self.kwargs.leg1["irs_series"],
                    strike=NoInput(0)
                    if isinstance(self.kwargs.leg1["strike"], str)
                    else self.kwargs.leg1["strike"],
                    notional=self.kwargs.leg1["notional"],
                    option_fixings=self.kwargs.leg1["option_fixings"],
                    metric=self.kwargs.meta["metric"],
                    settlement_method=self.kwargs.leg1["settlement_method"],
                )
            ]
        )
        self._leg2 = CustomLeg(
            [
                Cashflow(
                    notional=_drb(0.0, self.kwargs.leg2["premium"]),
                    payment=self.kwargs.leg2["payment"],
                    currency=self._leg1.settlement_params.currency,
                ),
            ]
        )
        self._legs = [self._leg1, self._leg2]

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def _set_strike_and_vol(
        self,
        rate_curve: CurveOption_,
        disc_curve: _BaseCurve_,
        index_curve: _BaseCurve_,
        vol: _IRVolOption_,
    ) -> None:
        """
        Set the strike, if necessary, and determine pricing metrics from the volatility objects.

        The strike for the *OptionPeriod* is either; string or numeric.

        If it is string, then a numeric strike must be determined with an associated vol.

        If it is numeric then the volatility must be determined for the given strike.

        Pricing elements are captured and cached so they can be used later by subsequent methods.
        """
        _ir_price_params = _get_ir_vol_value_and_forward_maybe_from_obj(
            rate_curve=rate_curve,
            index_curve=index_curve,
            strike=self.kwargs.leg1["strike"],
            ir_vol=vol,
            irs=self._irs,
            tenor=self._option.ir_option_params.option_fixing.termination,
            expiry=self._option.ir_option_params.expiry,
        )

        if isinstance(vol, IRSabrSmile):
            eval_date = vol.meta.eval_date
        else:
            _ = _validate_obj_not_no_input(disc_curve, "disc_curve")
            eval_date = _.nodes.initial
        t_e_ = self._option.ir_option_params.time_to_expiry(eval_date)

        _pricing = _IRVolPricingMetrics(
            vol=_ir_price_params.vol,
            k=_ir_price_params.k,
            t_e=t_e_,
            f=_ir_price_params.f,
            shift=_ir_price_params.shift,
        )

        # Review section in book regarding Hyper-parameters and Solver interaction
        self._option.ir_option_params.strike = _pricing.k
        self._pricing = _pricing
        # self._option_periods[0].strike = _dual_float(self._pricing.k)

    def _set_premium(
        self,
        rate_curve: CurveOption_,
        disc_curve: _BaseCurve_,
        index_curve: _BaseCurve_,
        pricing: _IRVolPricingMetrics,
    ) -> None:
        """
        Set an unspecified premium on the Option to be equal to the mid-market premium.
        """
        if isinstance(self.kwargs.leg2["premium"], NoInput):
            # then set the CashFlow to mid-market
            npv: DualTypes = self._option.npv(  # type: ignore[assignment]
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                index_curve=index_curve,
                ir_vol=pricing.vol,
                local=False,
                forward=self.kwargs.leg2["payment"],
            )
            self._premium.settlement_params._notional = _dual_float(npv)

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
        metric: IROptionMetric | str_ = NoInput(0),
    ) -> DualTypes:
        _curves = self._parse_curves(curves)
        _vol = self._parse_vol(vol)
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
        ir_vol = _maybe_get_ir_vol_maybe_from_solver(
            vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
        )
        self._set_strike_and_vol(
            rate_curve=rate_curve, disc_curve=disc_curve, index_curve=index_curve, vol=ir_vol
        )

        # Premium is not required for rate and also sets as float
        # Review section: "Hyper-parameters and Solver interaction" before enabling.
        # self._set_premium(curves, fx)

        metric_ = _get_ir_option_metric(_drb(self.kwargs.meta["metric"], metric))
        del metric

        value = self._option.rate(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            ir_vol=ir_vol,
            metric=metric_,
        )
        if (
            metric_ in [IROptionMetric.Cash(), IROptionMetric.PercentNotional()]
            and self.leg2.settlement_params.payment != self.leg1.settlement_params.payment
        ):
            disc_curve_ = _validate_obj_not_no_input(disc_curve, name="disc_curve")
            del disc_curve
            return (
                value
                * disc_curve_[self.leg2.settlement_params.payment]
                / disc_curve_[self.leg1.settlement_params.payment]
            )
        else:
            return value

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
        _curves = self._parse_curves(curves)
        _vol = self._parse_vol(vol)
        rate_curve = _maybe_get_curve_maybe_from_solver(
            curves=_curves,
            curves_meta=self.kwargs.meta["curves"],
            solver=solver,
            name="rate_curve",
        )
        disc_curve = _maybe_get_curve_maybe_from_solver(
            curves=_curves,
            curves_meta=self.kwargs.meta["curves"],
            solver=solver,
            name="disc_curve",
        )
        index_curve = _maybe_get_curve_maybe_from_solver(
            curves=_curves,
            curves_meta=self.kwargs.meta["curves"],
            solver=solver,
            name="index_curve",
        )
        ir_vol = _maybe_get_ir_vol_maybe_from_solver(
            vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
        )
        self._set_strike_and_vol(
            rate_curve=rate_curve, disc_curve=disc_curve, index_curve=index_curve, vol=ir_vol
        )

        self._set_premium(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            pricing=self._pricing,
        )

        if not local:
            base_ = _drb(self.legs[0].settlement_params.currency, base)
        else:
            base_ = base

        opt_npv = self._option.npv(
            rate_curve=rate_curve,  # _validate_obj_not_no_input(rate_curve, "rate curve"),
            disc_curve=disc_curve,
            index_curve=index_curve,
            fx=fx,
            base=base_,
            local=local,
            ir_vol=self._pricing.vol,
            settlement=settlement,
            forward=forward,
        )
        prem_npv = self._premium.npv(
            disc_curve=_maybe_get_curve_maybe_from_solver(
                curves=_curves,
                curves_meta=self.kwargs.meta["curves"],
                solver=solver,
                name="leg2_disc_curve",
            ),
            fx=fx,
            base=base_,
            local=local,
            settlement=settlement,
            forward=forward,
        )
        if local:
            return {k: opt_npv.get(k, 0) + prem_npv.get(k, 0) for k in set(opt_npv) | set(prem_npv)}  # type:ignore[union-attr, arg-type]
        else:
            return opt_npv + prem_npv  # type: ignore[operator]

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
        try:
            _curves = self._parse_curves(curves)
            _vol = self._parse_vol(vol)
            rate_curve = _maybe_get_curve_maybe_from_solver(
                curves=_curves,
                curves_meta=self.kwargs.meta["curves"],
                solver=solver,
                name="rate_curve",
            )
            disc_curve = _maybe_get_curve_maybe_from_solver(
                curves=_curves,
                curves_meta=self.kwargs.meta["curves"],
                solver=solver,
                name="disc_curve",
            )
            index_curve = _maybe_get_curve_maybe_from_solver(
                curves=_curves,
                curves_meta=self.kwargs.meta["curves"],
                solver=solver,
                name="index_curve",
            )
            ir_vol = _maybe_get_ir_vol_maybe_from_solver(
                vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
            )
            self._set_strike_and_vol(
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                index_curve=index_curve,
                vol=ir_vol,
            )
            self._set_premium(
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                index_curve=index_curve,
                pricing=self._pricing,
            )
        except Exception:  # noqa: S110
            pass  # `cashflows` proceed without pricing determined values

        return self._cashflows_from_legs(
            curves=curves,
            solver=solver,
            fx=fx,
            base=base,
            settlement=settlement,
            forward=forward,
            vol=vol,
        )

    # def analytic_greeks(
    #     self,
    #     curves: CurvesT_ = NoInput(0),
    #     solver: Solver_ = NoInput(0),
    #     fx: FXForwards_ = NoInput(0),
    #     vol: FXVol_ = NoInput(0),
    # ) -> dict[str, Any]:
    #     """
    #     Return various pricing metrics of the *FX Option*.
    #
    #     .. rubric:: Examples
    #
    #     .. ipython:: python
    #        :suppress:
    #
    #        from rateslib import Curve, FXCall, dt, FXForwards, FXRates, FXDeltaVolSmile
    #
    #     .. ipython:: python
    #
    #        eur = Curve({dt(2020, 1, 1): 1.0, dt(2021, 1, 1): 0.98})
    #        usd = Curve({dt(2020, 1, 1): 1.0, dt(2021, 1, 1): 0.96})
    #        fxf = FXForwards(
    #            fx_rates=FXRates({"eurusd": 1.10}, settlement=dt(2020, 1, 3)),
    #            fx_curves={"eureur": eur, "eurusd": eur, "usdusd": usd},
    #        )
    #        fxvs = FXDeltaVolSmile(
    #            nodes={0.25: 11.0, 0.5: 9.8, 0.75: 10.7},
    #            delta_type="forward",
    #            eval_date=dt(2020, 1, 1),
    #            expiry=dt(2020, 4, 1)
    #        )
    #        fxc = FXCall(
    #            expiry="3m",
    #            strike=1.10,
    #            eval_date=dt(2020, 1, 1),
    #            spec="eurusd_call",
    #        )
    #        fxc.analytic_greeks(fx=fxf, curves=[eur, usd], vol=fxvs)
    #
    #     Parameters
    #     ----------
    #     curves: _Curves, :green:`optional`
    #         Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
    #     solver: Solver, :green:`optional`
    #         A :class:`~rateslib.solver.Solver` object containing *Curve*, *Smile*, *Surface*, or
    #         *Cube* mappings for pricing.
    #     fx: FXForwards, :green:`optional`
    #         The :class:`~rateslib.fx.FXForwards` object used for forecasting FX rates, if necessary.
    #     vol: _Vol, :green:`optional`
    #         Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
    #
    #     Returns
    #     -------
    #     dict
    #     """
    #     return self._analytic_greeks_set_metrics(
    #         curves=curves,
    #         solver=solver,
    #         fx=fx,
    #         vol=vol,
    #         set_metrics=True,
    #     )
    #
    # def _analytic_greeks_set_metrics(
    #     self,
    #     curves: CurvesT_ = NoInput(0),
    #     solver: Solver_ = NoInput(0),
    #     fx: FXForwards_ = NoInput(0),
    #     vol: FXVol_ = NoInput(0),
    #     set_metrics: bool_ = True,
    # ) -> dict[str, Any]:
    #     """
    #     Return various pricing metrics of the *FX Option*.
    #
    #     Returns
    #     -------
    #     float, Dual, Dual2
    #     """
    #     _curves = self._parse_curves(curves)
    #     _vol = self._parse_vol(vol)
    #     rate_curve = _maybe_get_curve_maybe_from_solver(
    #         curves=_curves, curves_meta=self.kwargs.meta["curves"], solver=solver, name="rate_curve"
    #     )
    #     disc_curve = _maybe_get_curve_maybe_from_solver(
    #         curves=_curves, curves_meta=self.kwargs.meta["curves"], solver=solver, name="disc_curve"
    #     )
    #     fx_vol = _maybe_get_fx_vol_maybe_from_solver(
    #         vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
    #     )
    #     fx_ = _get_fx_forwards_maybe_from_solver(solver=solver, fx=fx)
    #
    #     if set_metrics:
    #         self._set_strike_and_vol(
    #             rate_curve=rate_curve, disc_curve=disc_curve, fx=fx_, vol=fx_vol
    #         )
    #         # self._set_premium(curves, fx)
    #
    #     return self._option.analytic_greeks(
    #         rate_curve=_validate_obj_not_no_input(rate_curve, "rate curve"),
    #         disc_curve=_validate_obj_not_no_input(disc_curve, "disc curve"),
    #         fx=_validate_fx_as_forwards(fx_),
    #         fx_vol=fx_vol,
    #         premium=NoInput(0),
    #         premium_payment=self.kwargs.leg2["payment"],
    #     )
    #
    # def _analytic_greeks_reduced(
    #     self,
    #     curves: CurvesT_ = NoInput(0),
    #     solver: Solver_ = NoInput(0),
    #     fx: FXForwards_ = NoInput(0),
    #     base: str_ = NoInput(0),
    #     vol: FXVol_ = NoInput(0),
    #     set_metrics: bool_ = True,
    # ) -> dict[str, Any]:
    #     """
    #     Return various pricing metrics of the *FX Option*.
    #     """
    #     _curves = self._parse_curves(curves)
    #     _vol = self._parse_vol(vol)
    #     rate_curve = _maybe_get_curve_maybe_from_solver(
    #         curves=_curves, curves_meta=self.kwargs.meta["curves"], solver=solver, name="rate_curve"
    #     )
    #     disc_curve = _maybe_get_curve_maybe_from_solver(
    #         curves=_curves, curves_meta=self.kwargs.meta["curves"], solver=solver, name="disc_curve"
    #     )
    #     fx_vol = _maybe_get_fx_vol_maybe_from_solver(
    #         vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
    #     )
    #     fx_ = _get_fx_forwards_maybe_from_solver(solver=solver, fx=fx)
    #
    #     if set_metrics:
    #         self._set_strike_and_vol(
    #             rate_curve=rate_curve, disc_curve=disc_curve, fx=fx_, vol=fx_vol
    #         )
    #         # self._set_premium(curves, fx)
    #
    #     return self._option._base_analytic_greeks(
    #         rate_curve=_validate_obj_not_no_input(rate_curve, "rate_curve"),
    #         disc_curve=_validate_obj_not_no_input(disc_curve, "disc_curve"),
    #         fx=_validate_fx_as_forwards(fx_),
    #         fx_vol=self._pricing.vol,  # type: ignore[arg-type]  # vol is set and != None
    #         premium=NoInput(0),
    #         _reduced=True,
    #     )  # none of the reduced greeks need a VolObj - faster to reuse from _pricing.vol

    # def analytic_delta(self, *args: Any, leg: int = 1, **kwargs: Any) -> NoReturn:
    #     """Not implemented for Option types.
    #     Use :meth:`~rateslib.instruments._BaseFXOption.analytic_greeks`.
    #     """
    #     raise NotImplementedError("For Option types use `analytic_greeks`.")
    #
    # def _plot_payoff(
    #     self,
    #     window: tuple[float, float] | NoInput = NoInput(0),
    #     curves: CurvesT_ = NoInput(0),
    #     solver: Solver_ = NoInput(0),
    #     fx: FXForwards_ = NoInput(0),
    #     vol: IRVol_ = NoInput(0),
    # ) -> tuple[
    #     np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]
    # ]:
    #     """
    #     Mechanics to determine (x,y) coordinates for payoff at expiry plot.
    #     """
    #
    #     _curves = self._parse_curves(curves)
    #     _vol = self._parse_vol(vol)
    #     rate_curve = _validate_obj_not_no_input(
    #         _maybe_get_curve_maybe_from_solver(
    #             curves=_curves,
    #             curves_meta=self.kwargs.meta["curves"],
    #             solver=solver,
    #             name="rate_curve",
    #         ),
    #         "rate_curve",
    #     )
    #     disc_curve = _validate_obj_not_no_input(
    #         _maybe_get_curve_maybe_from_solver(
    #             curves=_curves,
    #             curves_meta=self.kwargs.meta["curves"],
    #             solver=solver,
    #             name="disc_curve",
    #         ),
    #         "disc curve",
    #     )
    #     fx_vol = _maybe_get_fx_vol_maybe_from_solver(
    #         vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
    #     )
    #     fx_ = _get_fx_forwards_maybe_from_solver(solver=solver, fx=fx)
    #     self._set_strike_and_vol(rate_curve=rate_curve, disc_curve=disc_curve, fx=fx_, vol=fx_vol)
    #     # self._set_premium(curves, fx)
    #
    #     x, y = self._option._payoff_at_expiry(window)
    #     return x, y
    #
    # def plot_payoff(
    #     self,
    #     range: tuple[float, float] | NoInput = NoInput(0),  # noqa: A002
    #     curves: CurvesT_ = NoInput(0),
    #     solver: Solver_ = NoInput(0),
    #     fx: FXForwards_ = NoInput(0),
    #     base: str_ = NoInput(0),
    #     vol: float_ = NoInput(0),
    # ) -> PlotOutput:
    #     """
    #     Return a plot of the payoff at expiry, indexed by the *FXFixing* value.
    #
    #     Parameters
    #     ----------
    #     range: list of float, :green:`optional`
    #         A range of values for the *FXFixing* value at expiry to use as the x-axis.
    #     curves: _Curves, :green:`optional`
    #         Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
    #     solver: Solver, :green:`optional`
    #         A :class:`~rateslib.solver.Solver` object containing *Curve*, *Smile*, *Surface*, or
    #         *Cube* mappings for pricing.
    #     fx: FXForwards, :green:`optional`
    #         The :class:`~rateslib.fx.FXForwards` object used for forecasting FX rates, if necessary.
    #     vol: _Vol, :green:`optional`
    #         Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
    #
    #     Returns
    #     -------
    #     (Figure, Axes, list[Lines2D])
    #     """
    #
    #     x, y = self._plot_payoff(window=range, curves=curves, solver=solver, fx=fx, vol=vol)
    #     return plot([x], [y])  # type: ignore
    #
    # def local_analytic_rate_fixings(
    #     self,
    #     *,
    #     curves: CurvesT_ = NoInput(0),
    #     solver: Solver_ = NoInput(0),
    #     fx: FXForwards_ = NoInput(0),
    #     vol: VolT_ = NoInput(0),
    #     settlement: datetime_ = NoInput(0),
    #     forward: datetime_ = NoInput(0),
    # ) -> DataFrame:
    #     return DataFrame()
    #
    # def spread(
    #     self,
    #     *,
    #     curves: CurvesT_ = NoInput(0),
    #     solver: Solver_ = NoInput(0),
    #     fx: FXForwards_ = NoInput(0),
    #     vol: VolT_ = NoInput(0),
    #     base: str_ = NoInput(0),
    #     settlement: datetime_ = NoInput(0),
    #     forward: datetime_ = NoInput(0),
    # ) -> DualTypes:
    #     """
    #     Not implemented for Option types. Use :meth:`~rateslib.instruments._BaseFXOption.rate`.
    #     """
    #     raise NotImplementedError(f"`spread` is not implemented for type: {type(self).__name__}")


class PayerSwaption(_BaseIROption):
    """
    An *IR Payer* swaption.

    .. warning::

       *Swaptions* are in Beta status introduced in v2.7.0

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib import dt, Curve, PayerSwaption

    .. ipython:: python

       iro = PayerSwaption(
           expiry=dt(2027, 2, 16),
           tenor="6m",
           strike=3.02,
           notional=100e6,
           irs_series="usd_irs",
           premium=10000.0,
       )
       # iro.cashflows()

    .. rubric:: Pricing

    A *Swaption* requires from one to three *Curves*;

    - a ``rate_curve`` used to forecast the rates on the :class:`~rateslib.legs.FloatLeg` of the
      underlying :class:`~rateslib.instruments.IRS`.
    - a ``disc_curve`` used to discount the value of the *Swaption* and the premium under the
      terms of its bilateral collateral agreement.
    - an ``index_curve`` used as the price alignment index rate for the discounting of the
      underlying :class:`~rateslib.instruments.IRS`. This does not necessarily need to equal the
      ``disc_curve``.

    Allowable inputs are:

    .. code-block:: python

       curves = rate_curve | [rate_curve] #  one curve is used as all curves
       curves = [rate_curve, disc_curve]  #  two curves are applied in the given order, index_curve is set equal to disc_curve
       curves = [rate_curve, disc_curve, index_curve]  # three curves applied in the given order
       curves = {
           "rate_curve": rate_curve,
           "disc_curve": disc_curve
           "index_curve": index_curve
       }  # dict form is explicit

    A *Swaption* also requires an *IRVolatility* object or numeric value for the ``vol`` argument.
    If a numeric value is given it is assumed to be a Black (log-normal) volatility without shift.
    Allowed inputs are:

    .. code-block:: python

       vol = 12.0     # a specific Black (log-normal) calendar-day annualized vol until expiry
       vol = vol_obj  # an explicit volatility object, e.g. IRSabrSmile

    The following pricing ``metric`` are available, with examples:

    .. ipython:: python

       curve = Curve(
           nodes={dt(2026, 2, 16): 1.0, dt(2028, 2, 16): 0.941024343401225}, calendar="nyc"
       )

    - **"BlackVol" or "BlackVolShift100", "BlackVolShift200", "BlackVolShift300"**:
      The *rate* method will make the necessary conversions between the different volatility
      representations.

      .. ipython:: python

          iro.rate(curves=[curve], vol=25.16, metric="BlackVol")
          iro.rate(curves=[curve], vol=25.16, metric="BlackVolShift100")
          iro.rate(curves=[curve], vol=25.16, metric="BlackVolShift200")
          iro.rate(curves=[curve], vol=25.16, metric="BlackVolShift300")

    - **"NormalVol"**: the equivalent number of basis point volatility used in the Bachelier
      formula:

      .. ipython:: python

          iro.rate(curves=[curve], vol=25.16, metric="NormalVol")

    - **"Cash"**: the cash premium amount applicable to the 'payment' date, expressed in the
      premium currency.

      .. ipython:: python

          iro.rate(curves=[curve], vol=25.16, metric="Cash")

    - **"PercentNotional"**: the cash premium amount expressed as a percentage of the
      notional.

      .. ipython:: python

          iro.rate(curves=[curve], vol=25.16, metric="PercentNotional")

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .

        .. note::

           The following define **ir option** and generalised **settlement** parameters.

    expiry: datetime, str, :red:`required`
        The expiry of the option. If given in string tenor format, e.g. "1M" requires an
        ``eval_date``. See **Notes**.
    tenor: datetime, str, :red:`required`
        The parameter defining the maturity of the underlying :class:`~rateslib.instruments.IRS`.
    irs_series: IRSSeries, str, :red:`required`
        The standard conventions applied to the underlying :class:`~rateslib.instruments.IRS`.
    strike: float, Variable, str, :red:`required`
        The strike value of the option.
        If str, there are two possibilities; {"atm", "{}bps"}. "atm" will produce a strike equal
        to the mid-market *IRS* rate, whilst "20bps" or "-50bps" will yield a strike that number
        of basis points different to the mid-market rate.
    notional: float, :green:`optional (set by 'defaults')`
        The notional amount expressed in units of LHS of ``pair``.
    eval_date: datetime, :green:`optional`
        Only required if ``expiry`` is given as string tenor.
        Should be entered as today (also called horizon) and **not** spot.
    payment_lag: int or datetime, :green:`optional (set as IRS effective)`
        The number of business days after expiry to pay premium. If a *datetime* is given this will
        set the premium date explicitly.
    settlement_method: SwaptionSettlementMethod, str, :green:`optional (set by 'default')`
        The method for deriving the settlement cashflow or underlying value.

        .. note::

           The following define additional **rate** parameters.

    premium: float, :green:`optional`
        The amount paid for the option. If not given assumes an unpriced *Option* and sets this as
        mid-market premium during pricing.
    option_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the option :class:`~rateslib.data.fixings.IRSFixing`. If a scalar, is used
        directly. If a string identifier, links to the central ``fixings`` object and data loader.

        .. note::

           The following are **meta parameters**.

    metric : str, :green:`optional (set as "pips_or_%")`
        The pricing metric returned by the ``rate`` method. See **Pricing**.
    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.
    vol: str, Smile, Surface, float, Dual, Dual2, Variable
        Pricing objects passed directly to the *Instrument's* methods' ``vol`` argument. See
        **Pricing**.
    spec : str, optional
        An identifier to pre-populate many field with conventional values. See
        :ref:`here<defaults-doc>` for more info and available values.

    """  # noqa: E501

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, call=True, **kwargs)


class ReceiverSwaption(_BaseIROption):
    """
    An *IR Receiver* swaption.

    For parameters and examples see :class:`~rateslib.instruments.PayerSwaption`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, call=False, **kwargs)
