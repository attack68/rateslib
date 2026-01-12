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

from pandas import DataFrame

from rateslib import FXDeltaVolSmile, FXDeltaVolSurface, defaults
from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.data.fixings import _fx_index_set_cross, _get_fx_index
from rateslib.default import PlotOutput, plot
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import FXOptionMetric, _get_fx_delta_type
from rateslib.fx_volatility import FXSabrSmile, FXSabrSurface
from rateslib.instruments.protocols import _BaseInstrument, _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _get_fx_forwards_maybe_from_solver,
    _maybe_get_curve_maybe_from_solver,
    _maybe_get_fx_vol_maybe_from_solver,
    _Vol,
)
from rateslib.legs import CustomLeg
from rateslib.periods import Cashflow, FXCallPeriod, FXPutPeriod
from rateslib.periods.utils import _validate_fx_as_forwards
from rateslib.scheduling.frequency import _get_fx_expiry_and_delivery_and_payment

if TYPE_CHECKING:
    from typing import NoReturn  # pragma: no cover

    import numpy as np  # pragma: no cover

    from rateslib.typing import (  # pragma: no cover
        FX_,
        Any,
        CalInput,
        CurvesT_,
        DualTypes,
        DualTypes_,
        FXForwards,
        FXForwards_,
        FXIndex,
        FXVol_,
        Sequence,
        Solver_,
        VolT_,
        _BaseCurve,
        _BaseCurve_,
        _BaseFXOptionPeriod,
        _BaseLeg,
        _FXVolOption_,
        bool_,
        datetime_,
        float_,
        int_,
        str_,
    )


@dataclass
class _PricingMetrics:
    """None elements are used as flags to indicate an element is not yet set."""

    vol: _FXVolOption_ | None
    k: DualTypes | None
    delta_index: DualTypes | None
    spot: datetime
    t_e: DualTypes | None
    f_d: DualTypes


class _BaseFXOption(_BaseInstrument, metaclass=ABCMeta):
    """
    An abstract base class for implementing *FXOptions*.

    See :class:`~rateslib.instruments.FXCall` and :class:`~rateslib.instruments.FXPut`.
    """

    _rate_scalar: float = 1.0
    _pricing: _PricingMetrics

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
    def _option(self) -> _BaseFXOptionPeriod:
        return self.leg1.periods[0]  # type: ignore[return-value]

    @property
    def _premium(self) -> Cashflow:
        return self.leg2.periods[0]  # type: ignore[return-value]

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
        """
        An FXOption has two curve requirements:

        The *rate curve* is the curve for the LHS of ``pair`` which is the curve typically
        used to convert between spot and forward delta types. However, if the premium currency is
        in the LHS side currency this cure will also be used as a discount curve for that
        payment.

        The *disc curve* is the curve for the RHS side of ``pair``.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        if isinstance(curves, dict):
            rate_curve = curves.get("rate_curve", NoInput(0))
            disc_curve = curves.get("disc_curve", NoInput(0))
            if self._premium.settlement_params.currency == self.kwargs.leg1["pair"][:3]:
                leg2_disc_curve = rate_curve
            else:
                leg2_disc_curve = disc_curve
            return _Curves(
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                leg2_disc_curve=leg2_disc_curve,
            )
        elif isinstance(curves, list | tuple) and len(curves) == 2:
            rate_curve = curves[0]  # type: ignore[assignment]
            disc_curve = curves[1]  # type: ignore[assignment]
            if self.kwargs.leg2["premium_ccy"] == self.kwargs.leg1["pair"].pair[:3]:
                leg2_disc_curve = rate_curve
            else:
                leg2_disc_curve = disc_curve
            return _Curves(
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                leg2_disc_curve=leg2_disc_curve,
            )
        elif isinstance(curves, _Curves):
            return curves
        else:
            raise ValueError(f"{type(self).__name__} requires 2 curve types.")

    @classmethod
    def _parse_vol(cls, vol: VolT_) -> _Vol:
        """
        FXoptions requires only a single FXVolObj or a scalar.
        """
        if isinstance(vol, _Vol):
            return vol
        else:
            return _Vol(fx_vol=vol)

    def __init__(
        self,
        expiry: datetime | str,
        strike: DualTypes | str,
        pair: FXIndex | str_ = NoInput(0),
        *,
        notional: DualTypes_ = NoInput(0),
        eval_date: datetime | NoInput = NoInput(0),
        calendar: CalInput = NoInput(0),
        modifier: str_ = NoInput(0),
        eom: bool_ = NoInput(0),
        delivery_lag: int_ = NoInput(0),
        premium: DualTypes_ = NoInput(0),
        premium_ccy: str_ = NoInput(0),
        payment_lag: str | datetime_ = NoInput(0),
        option_fixings: DualTypes_ = NoInput(0),
        delta_type: str_ = NoInput(0),
        metric: str_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        spec: str_ = NoInput(0),
        call: bool = True,
    ):
        user_args = dict(
            pair=pair,
            expiry=expiry,
            notional=notional,
            strike=strike,
            calendar=calendar,
            eom=eom,
            modifier=modifier,
            delta_type=delta_type,
            option_fixings=option_fixings,
            delivery_lag=delivery_lag,
            leg2_payment_lag=payment_lag,
            leg2_premium=premium,
            leg2_premium_ccy=premium_ccy,
            metric=metric,
            curves=curves,
            vol=self._parse_vol(vol),
        )
        # instrument_args: dict[str, Any] = dict()
        default_args = dict(
            delta_type=defaults.fx_delta_type,
            notional=defaults.notional,
            modifier=defaults.modifier,
            metric="pips_or_%",
            delivery_lag=defaults.fx_delivery_lag,
            leg2_payment_lag=defaults.payment_lag,
            eom=defaults.eom_fx,
        )
        self._kwargs = _KWArgs(
            user_args=user_args,
            default_args=default_args,
            spec=spec,
            meta_args=["curves", "vol", "metric"],
        )

        # This configuration here assumes that the options are physically settled, so do not
        # allow WMR cross methodology to impact forecast rates for FXFixings.
        self.kwargs.leg1["pair"] = _fx_index_set_cross(
            _get_fx_index(self.kwargs.leg1["pair"]),
            allow_cross=False,
        )

        # apply the parse knowing the premium currency
        self._kwargs.leg2["premium_ccy"] = _drb(
            self.kwargs.leg1["pair"].pair[3:], self.kwargs.leg2["premium_ccy"]
        )
        self._kwargs.meta["curves"] = self._parse_curves(self._kwargs.meta["curves"])

        # determine the `expiry` and `delivery` as datetimes if derived from other combinations
        (self.kwargs.leg1["expiry"], self.kwargs.leg1["delivery"], self.kwargs.leg2["payment"]) = (
            _get_fx_expiry_and_delivery_and_payment(
                eval_date=eval_date,
                expiry=self.kwargs.leg1["expiry"],
                delivery_lag=self.kwargs.leg1["delivery_lag"],
                calendar=self.kwargs.leg1["calendar"],
                modifier=self.kwargs.leg1["modifier"],
                eom=self.kwargs.leg1["eom"],
                payment_lag=self.kwargs.leg2["payment_lag"],
            )
        )

        if self.kwargs.leg2["premium_ccy"] not in [
            self.kwargs.leg1["pair"].pair[:3],
            self.kwargs.leg1["pair"].pair[3:],
        ]:
            raise ValueError(
                f"`premium_ccy`: '{self.kwargs.leg2['premium_ccy']}' must be one of option "
                f"currency pair: '{self.kwargs.leg1['pair'].pair}'.",
            )
        elif self.kwargs.leg2["premium_ccy"] == self.kwargs.leg1["pair"].pair[3:]:
            self.kwargs.meta["metric_period"] = "pips"
            self.kwargs.meta["delta_method"] = _get_fx_delta_type(self.kwargs.leg1["delta_type"])
        else:
            self.kwargs.meta["metric_period"] = "percent"
            self.kwargs.meta["delta_method"] = _get_fx_delta_type(
                self.kwargs.leg1["delta_type"] + "_pa"
            )

        self._validate_strike_and_premiums()

        self._leg1 = CustomLeg(
            [
                FXCallPeriod(  # type: ignore[abstract]
                    pair=self.kwargs.leg1["pair"],
                    expiry=self.kwargs.leg1["expiry"],
                    delivery=self.kwargs.leg1["delivery"],
                    strike=(
                        NoInput(0)
                        if isinstance(self.kwargs.leg1["strike"], str)
                        else self.kwargs.leg1["strike"]
                    ),
                    notional=self.kwargs.leg1["notional"],
                    option_fixings=self.kwargs.leg1["option_fixings"],
                    delta_type=self.kwargs.meta["delta_method"],
                    metric=self.kwargs.meta["metric_period"],
                )
                if call
                else FXPutPeriod(  # type: ignore[abstract]
                    pair=self.kwargs.leg1["pair"],
                    expiry=self.kwargs.leg1["expiry"],
                    delivery=self.kwargs.leg1["delivery"],
                    strike=(
                        NoInput(0)
                        if isinstance(self.kwargs.leg1["strike"], str)
                        else self.kwargs.leg1["strike"]
                    ),
                    notional=self.kwargs.leg1["notional"],
                    option_fixings=self.kwargs.leg1["option_fixings"],
                    delta_type=self.kwargs.meta["delta_method"],
                    metric=self.kwargs.meta["metric_period"],
                )
            ]
        )
        self._leg2 = CustomLeg(
            [
                Cashflow(
                    notional=_drb(0.0, self.kwargs.leg2["premium"]),
                    payment=self.kwargs.leg2["payment"],
                    currency=self.kwargs.leg2["premium_ccy"],
                ),
            ]
        )
        self._legs = [self._leg1, self._leg2]

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def _validate_strike_and_premiums(self) -> None:
        if isinstance(self.kwargs.leg1["strike"], str) and not isinstance(
            self.kwargs.leg2["premium"], NoInput
        ):
            raise ValueError(
                "FXOption with string delta as `strike` cannot be initialised with a known "
                "`premium`.\n"
                "Either set `strike` as a defined numeric value, or remove the `premium`.",
            )

    def _set_strike_and_vol(
        self,
        rate_curve: _BaseCurve_,
        disc_curve: _BaseCurve_,
        fx: FX_,
        vol: _FXVolOption_,
    ) -> None:
        """
        Set the strike, if necessary, and determine pricing metrics from the volatility objects.

        The strike for the *OptionPeriod* is either; string or numeric.

        If it is string, then a numeric strike must be determined with an associated vol.

        If it is numeric then the volatility must be determined for the given strike.

        Pricing elements are captured and cached so they can be used later by subsequent methods.
        """
        fx_ = _validate_fx_as_forwards(fx)
        _pricing = _PricingMetrics(
            vol=None,
            k=None,
            delta_index=None,
            spot=fx_.pairs_settlement[self.kwargs.leg1["pair"].pair],
            t_e=None,
            f_d=fx_.rate(self.kwargs.leg1["pair"], self.kwargs.leg1["delivery"]),
        )

        if isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface | FXSabrSmile | FXSabrSurface):
            eval_date = vol.meta.eval_date
        else:
            _ = _validate_obj_not_no_input(disc_curve, "disc_curve")
            eval_date = _.nodes.initial
            _pricing.vol = vol  # Not a vol model so set directly
        _pricing.t_e = self._option.fx_option_params.time_to_expiry(eval_date)
        self._update_pricing_for_strike(
            strike=self.kwargs.leg1["strike"],
            fx=fx_,
            pricing=_pricing,
            vol=vol,
            rate_curve=rate_curve,
        )

        # _PricingMetrics.k is completely specified
        assert _pricing.k is not None  # noqa: S101
        # Review section in book regarding Hyper-parameters and Solver interaction
        self._option.fx_option_params.strike = _pricing.k
        self._pricing = _pricing
        # self._option_periods[0].strike = _dual_float(self._pricing.k)

    def _update_pricing_for_strike(
        self,
        strike: str | DualTypes,
        fx: FXForwards,
        pricing: _PricingMetrics,
        vol: _FXVolOption_,
        rate_curve: _BaseCurve_,
    ) -> None:
        """Update the _PricingMetrics object to populate values."""
        if not isinstance(strike, str):
            # then strike is a numeric quantity
            pricing.k = strike
        else:
            # then strike is a string which must be converted to a numeric value
            strike = strike.lower()
            if strike == "atm_forward":
                pricing.k = fx.rate(self.kwargs.leg1["pair"], self.kwargs.leg1["delivery"])
            elif strike == "atm_spot":
                pricing.k = fx.rate(self.kwargs.leg1["pair"], pricing.spot)
            elif strike == "atm_delta":
                rc = _validate_obj_not_no_input(rate_curve, "rate_curve")
                pricing.delta_index, pricing.vol, pricing.k = (
                    self._option._index_vol_and_strike_from_atm(
                        delta_type=self._option.fx_option_params.delta_type,
                        vol=_validate_obj_not_no_input(vol, "vol"),  # type: ignore[arg-type]
                        w_deli=rc[self.kwargs.leg1["delivery"]],
                        w_spot=rc[pricing.spot],
                        f=fx if isinstance(vol, FXSabrSurface) else pricing.f_d,
                        t_e=pricing.t_e,  # type: ignore[arg-type]
                    )
                )
                return None
            elif strike[-1] == "d":  # representing a delta percentage
                rc = _validate_obj_not_no_input(rate_curve, "rate_curve")
                pricing.delta_index, pricing.vol, pricing.k = (
                    self._option._index_vol_and_strike_from_delta(
                        delta=float(strike[:-1]) / 100.0,
                        delta_type=self.kwargs.meta["delta_method"],
                        vol=_validate_obj_not_no_input(vol, "vol"),  # type: ignore[arg-type]
                        w_deli=rc[self.kwargs.leg1["delivery"]],
                        w_spot=rc[pricing.spot],
                        f=fx if isinstance(vol, FXSabrSurface) else pricing.f_d,
                        t_e=pricing.t_e,  # type: ignore[arg-type]
                    )
                )
                return None

        if pricing.vol is None:
            # vol is only None if vol_ is a VolObj so can be safely type ignored.
            # a numeric vol has already been set on the 'pricing' object.
            # then an explicit strike is set so determine the vol from strike, set and return.
            rc = _validate_obj_not_no_input(rate_curve, "rate_curve")
            pricing.delta_index, pricing.vol, _ = vol.get_from_strike(  # type: ignore[union-attr]
                k=pricing.k,  # type: ignore[arg-type]
                f=pricing.f_d if not isinstance(vol, FXSabrSurface) else fx,  # type: ignore[arg-type]
                expiry=self.kwargs.leg1["expiry"],
                z_w=rc[self.kwargs.leg1["delivery"]] / rc[pricing.spot],
            )
        return None

    def _set_premium(
        self,
        rate_curve: _BaseCurve_,
        disc_curve: _BaseCurve_,
        fx: FXForwards_,
        pricing: _PricingMetrics,
    ) -> None:
        """
        Set an unspecified premium on the Option to be equal to the mid-market premium.
        """
        if isinstance(self.kwargs.leg2["premium"], NoInput):
            # then set the CashFlow to mid-market
            disc_curve_: _BaseCurve = _validate_obj_not_no_input(disc_curve, "disc curve")
            rate_curve_: _BaseCurve = _validate_obj_not_no_input(rate_curve, "rate curve")
            try:
                npv: DualTypes = self._option.npv(  # type: ignore[assignment]
                    rate_curve=rate_curve_,
                    disc_curve=disc_curve_,
                    fx=fx,
                    fx_vol=pricing.vol,  # type: ignore[arg-type]
                    local=False,
                    forward=self.kwargs.leg2["payment"],
                    base=self.kwargs.leg2["premium_ccy"],
                )
            except AttributeError:
                raise ValueError(
                    "`premium` has not been configured for the specified FXOption.\nThis is "
                    "normally determined at mid-market from the given `curves` and `vol` but "
                    "in this case these values do not provide a valid calculation. "
                    "If not required, initialise the "
                    "FXOption with a `premium` of 0.0, and this will be avoided.",
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
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        _curves = self._parse_curves(curves)
        _vol = self._parse_vol(vol)
        rate_curve = _maybe_get_curve_maybe_from_solver(
            curves=_curves, curves_meta=self.kwargs.meta["curves"], solver=solver, name="rate_curve"
        )
        disc_curve = _maybe_get_curve_maybe_from_solver(
            curves=_curves, curves_meta=self.kwargs.meta["curves"], solver=solver, name="disc_curve"
        )
        fx_vol = _maybe_get_fx_vol_maybe_from_solver(
            vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
        )
        fx_ = _get_fx_forwards_maybe_from_solver(solver=solver, fx=fx)
        self._set_strike_and_vol(rate_curve=rate_curve, disc_curve=disc_curve, fx=fx_, vol=fx_vol)

        # Premium is not required for rate and also sets as float
        # Review section: "Hyper-parameters and Solver interaction" before enabling.
        # self._set_premium(curves, fx)

        metric = _drb(self.kwargs.meta["metric"], metric)
        if metric in ["vol", "single_vol"]:
            return _validate_obj_not_no_input(self._pricing.vol, "vol")  # type: ignore[return-value]

        _: DualTypes = self._option.rate(
            rate_curve=_validate_obj_not_no_input(rate_curve, "curve"),
            disc_curve=_validate_obj_not_no_input(disc_curve, "curve"),
            fx=fx_,
            fx_vol=self._pricing.vol,  # type: ignore[arg-type]
            forward=self.kwargs.leg2["payment"],
        )
        if metric == "premium":
            if self._option.fx_option_params.metric == FXOptionMetric.Pips:
                # is expressed in RHS currency
                _ *= self._option.settlement_params.notional / 10000
            else:  # == "percent"
                # is expressed in LHS currency
                _ *= self._option.settlement_params.notional / 100
        return _

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
            curves=_curves, curves_meta=self.kwargs.meta["curves"], solver=solver, name="rate_curve"
        )
        disc_curve = _maybe_get_curve_maybe_from_solver(
            curves=_curves, curves_meta=self.kwargs.meta["curves"], solver=solver, name="disc_curve"
        )
        fx_vol = _maybe_get_fx_vol_maybe_from_solver(
            vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
        )
        fx_ = _get_fx_forwards_maybe_from_solver(solver=solver, fx=fx)
        self._set_strike_and_vol(rate_curve=rate_curve, disc_curve=disc_curve, fx=fx_, vol=fx_vol)

        self._set_premium(
            rate_curve=rate_curve, disc_curve=disc_curve, fx=fx_, pricing=self._pricing
        )

        if not local:
            base_ = _drb(self.legs[0].settlement_params.currency, base)
        else:
            base_ = base

        opt_npv = self._option.npv(
            rate_curve=_validate_obj_not_no_input(rate_curve, "rate curve"),
            disc_curve=_validate_obj_not_no_input(disc_curve, "disc_curve"),
            fx=fx_,
            base=base_,
            local=local,
            fx_vol=self._pricing.vol,  # type: ignore[arg-type]
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
            fx=fx_,
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
            fx_vol = _maybe_get_fx_vol_maybe_from_solver(
                vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
            )
            fx_ = _get_fx_forwards_maybe_from_solver(solver=solver, fx=fx)
            self._set_strike_and_vol(
                rate_curve=rate_curve, disc_curve=disc_curve, fx=fx_, vol=fx_vol
            )
            self._set_premium(
                rate_curve=rate_curve, disc_curve=disc_curve, fx=fx_, pricing=self._pricing
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

    def analytic_greeks(
        self,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: FXVol_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return various pricing metrics of the *FX Option*.

        .. rubric:: Examples

        .. ipython:: python
           :suppress:

           from rateslib import Curve, FXCall, dt, FXForwards, FXRates, FXDeltaVolSmile

        .. ipython:: python

           eur = Curve({dt(2020, 1, 1): 1.0, dt(2021, 1, 1): 0.98})
           usd = Curve({dt(2020, 1, 1): 1.0, dt(2021, 1, 1): 0.96})
           fxf = FXForwards(
               fx_rates=FXRates({"eurusd": 1.10}, settlement=dt(2020, 1, 3)),
               fx_curves={"eureur": eur, "eurusd": eur, "usdusd": usd},
           )
           fxvs = FXDeltaVolSmile(
               nodes={0.25: 11.0, 0.5: 9.8, 0.75: 10.7},
               delta_type="forward",
               eval_date=dt(2020, 1, 1),
               expiry=dt(2020, 4, 1)
           )
           fxc = FXCall(
               expiry="3m",
               strike=1.10,
               eval_date=dt(2020, 1, 1),
               spec="eurusd_call",
           )
           fxc.analytic_greeks(fx=fxf, curves=[eur, usd], vol=fxvs)

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

        Returns
        -------
        dict
        """
        return self._analytic_greeks_set_metrics(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            set_metrics=True,
        )

    def _analytic_greeks_set_metrics(
        self,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: FXVol_ = NoInput(0),
        set_metrics: bool_ = True,
    ) -> dict[str, Any]:
        """
        Return various pricing metrics of the *FX Option*.

        Returns
        -------
        float, Dual, Dual2
        """
        _curves = self._parse_curves(curves)
        _vol = self._parse_vol(vol)
        rate_curve = _maybe_get_curve_maybe_from_solver(
            curves=_curves, curves_meta=self.kwargs.meta["curves"], solver=solver, name="rate_curve"
        )
        disc_curve = _maybe_get_curve_maybe_from_solver(
            curves=_curves, curves_meta=self.kwargs.meta["curves"], solver=solver, name="disc_curve"
        )
        fx_vol = _maybe_get_fx_vol_maybe_from_solver(
            vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
        )
        fx_ = _get_fx_forwards_maybe_from_solver(solver=solver, fx=fx)

        if set_metrics:
            self._set_strike_and_vol(
                rate_curve=rate_curve, disc_curve=disc_curve, fx=fx_, vol=fx_vol
            )
            # self._set_premium(curves, fx)

        return self._option.analytic_greeks(
            rate_curve=_validate_obj_not_no_input(rate_curve, "rate curve"),
            disc_curve=_validate_obj_not_no_input(disc_curve, "disc curve"),
            fx=_validate_fx_as_forwards(fx_),
            fx_vol=fx_vol,
            premium=NoInput(0),
            premium_payment=self.kwargs.leg2["payment"],
        )

    def _analytic_greeks_reduced(
        self,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: FXVol_ = NoInput(0),
        set_metrics: bool_ = True,
    ) -> dict[str, Any]:
        """
        Return various pricing metrics of the *FX Option*.
        """
        _curves = self._parse_curves(curves)
        _vol = self._parse_vol(vol)
        rate_curve = _maybe_get_curve_maybe_from_solver(
            curves=_curves, curves_meta=self.kwargs.meta["curves"], solver=solver, name="rate_curve"
        )
        disc_curve = _maybe_get_curve_maybe_from_solver(
            curves=_curves, curves_meta=self.kwargs.meta["curves"], solver=solver, name="disc_curve"
        )
        fx_vol = _maybe_get_fx_vol_maybe_from_solver(
            vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
        )
        fx_ = _get_fx_forwards_maybe_from_solver(solver=solver, fx=fx)

        if set_metrics:
            self._set_strike_and_vol(
                rate_curve=rate_curve, disc_curve=disc_curve, fx=fx_, vol=fx_vol
            )
            # self._set_premium(curves, fx)

        return self._option._base_analytic_greeks(
            rate_curve=_validate_obj_not_no_input(rate_curve, "rate_curve"),
            disc_curve=_validate_obj_not_no_input(disc_curve, "disc_curve"),
            fx=_validate_fx_as_forwards(fx_),
            fx_vol=self._pricing.vol,  # type: ignore[arg-type]  # vol is set and != None
            premium=NoInput(0),
            _reduced=True,
        )  # none of the reduced greeks need a VolObj - faster to reuse from _pricing.vol

    def analytic_delta(self, *args: Any, leg: int = 1, **kwargs: Any) -> NoReturn:
        """Not implemented for Option types.
        Use :meth:`~rateslib.instruments._BaseFXOption.analytic_greeks`.
        """
        raise NotImplementedError("For Option types use `analytic_greeks`.")

    def _plot_payoff(
        self,
        window: tuple[float, float] | NoInput = NoInput(0),
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: FXVol_ = NoInput(0),
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]
    ]:
        """
        Mechanics to determine (x,y) coordinates for payoff at expiry plot.
        """

        _curves = self._parse_curves(curves)
        _vol = self._parse_vol(vol)
        rate_curve = _validate_obj_not_no_input(
            _maybe_get_curve_maybe_from_solver(
                curves=_curves,
                curves_meta=self.kwargs.meta["curves"],
                solver=solver,
                name="rate_curve",
            ),
            "rate_curve",
        )
        disc_curve = _validate_obj_not_no_input(
            _maybe_get_curve_maybe_from_solver(
                curves=_curves,
                curves_meta=self.kwargs.meta["curves"],
                solver=solver,
                name="disc_curve",
            ),
            "disc curve",
        )
        fx_vol = _maybe_get_fx_vol_maybe_from_solver(
            vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
        )
        fx_ = _get_fx_forwards_maybe_from_solver(solver=solver, fx=fx)
        self._set_strike_and_vol(rate_curve=rate_curve, disc_curve=disc_curve, fx=fx_, vol=fx_vol)
        # self._set_premium(curves, fx)

        x, y = self._option._payoff_at_expiry(window)
        return x, y

    def plot_payoff(
        self,
        range: tuple[float, float] | NoInput = NoInput(0),  # noqa: A002
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: float_ = NoInput(0),
    ) -> PlotOutput:
        """
        Return a plot of the payoff at expiry, indexed by the *FXFixing* value.

        Parameters
        ----------
        range: list of float, :green:`optional`
            A range of values for the *FXFixing* value at expiry to use as the x-axis.
        curves: _Curves, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        solver: Solver, :green:`optional`
            A :class:`~rateslib.solver.Solver` object containing *Curve*, *Smile*, *Surface*, or
            *Cube* mappings for pricing.
        fx: FXForwards, :green:`optional`
            The :class:`~rateslib.fx.FXForwards` object used for forecasting FX rates, if necessary.
        vol: _Vol, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.

        Returns
        -------
        (Figure, Axes, list[Lines2D])
        """

        x, y = self._plot_payoff(window=range, curves=curves, solver=solver, fx=fx, vol=vol)
        return plot([x], [y])  # type: ignore

    def local_analytic_rate_fixings(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        return DataFrame()

    def spread(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes:
        """
        Not implemented for Option types. Use :meth:`~rateslib.instruments._BaseFXOption.rate`.
        """
        raise NotImplementedError(f"`spread` is not implemented for type: {type(self).__name__}")


class FXCall(_BaseFXOption):
    """
    An *FX Call* option.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib import dt, FXCall, FXForwards, FXRates, FXDeltaVolSmile, Curve

    .. ipython:: python

       fxc = FXCall(
           expiry="3m",
           strike=1.10,
           eval_date=dt(2020, 1, 1),
           spec="eurusd_call",
       )
       fxc.cashflows()

    .. rubric:: Pricing

    An *FXOption* requires two discount curves; a curve to discount the cashflow of the LHS
    currency of ``pair``. This is labelled as the *rate curve* and is used to derive the
    difference between spot and forward deltas. The curve labelled as *disc curve* discounts
    cashflows of the RHS of ``pair``. For the premium, depending upon whether it is paid in LHS
    or RHS currency the appropriate curve from *Leg1* will be used and labelled as
    *leg2 disc curve*. Allowable inputs are:

    .. code-block:: python

       curves = [rate_curve, disc_curve]  #  two curves are applied in the given order
       curves = {"rate_curve": rate_curve, "disc_curve": disc_curve}  # dict form is explicit

    An *FXOption* also requires an :class:`~rateslib.fx.FXForwards` as input to the ``fx``
    argument, and an *FXVolatility* object or numeric value for the ``vol`` argument. Allowed
    inputs are:

    .. code-block:: python

       vol = 12.0  #  a specific calendar-day annualized %-volatility until expiry
       vol = vol_obj  # an explicit volatility object, e.g. FXDeltaVolSurface

    The following pricing ``metric`` are available, with examples:

    .. ipython:: python

       eur = Curve({dt(2020, 1, 1): 1.0, dt(2021, 1, 1): 0.98})
       usd = Curve({dt(2020, 1, 1): 1.0, dt(2021, 1, 1): 0.96})
       fxf = FXForwards(
           fx_rates=FXRates({"eurusd": 1.10}, settlement=dt(2020, 1, 3)),
           fx_curves={"eureur": eur, "eurusd": eur, "usdusd": usd},
       )
       fxvs = FXDeltaVolSmile(
           nodes={0.25: 11.0, 0.5: 9.8, 0.75: 10.7},
           expiry=dt(2020, 4, 1),
           eval_date=dt(2020, 1, 1),
           delta_type="forward",
       )

    - **'vol'**: the implied volatility value of the option from a volatility object.

      .. ipython:: python

         fxc.rate(vol=fxvs, curves=[eur, usd], fx=fxf, metric="vol")

    - **'premium'**: the cash premium amount applicable to the 'payment' date, expressed in the
      premium currency.

      .. ipython:: python

         fxc.rate(vol=fxvs, curves=[eur, usd], fx=fxf, metric="premium")

    - **'pips_or_%'**: if the premium currency is LHS of ``pair`` this is a % of notional, whilst if
      the premium currency is RHS this gives a number of pips of the FX rate.

      .. ipython:: python

         fxc.rate(vol=fxvs, curves=[eur, usd], fx=fxf, metric="pips_or_%")

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .

        .. note::

           The following define **fx option** and generalised **settlement** parameters.

    expiry: datetime, str, :red:`required`
        The expiry of the option. If given in string tenor format, e.g. "1M" requires an
        ``eval_date``. See **Notes**.
    strike: float, Variable, str, :red:`required`
        The strike value of the option.
        If str, there are four possibilities; {"atm_forward", "atm_spot", "atm_delta", "%d"}.
        Call % deltas can be given, as "25d".
    pair: str, :red:`required`
        The currency pair for the FX rate which settles the option, in 3-digit codes, e.g. "eurusd".
        May be included as part of ``spec``.
    notional: float, :green:`optional (set by 'defaults')`
        The notional amount expressed in units of LHS of ``pair``.
    eval_date: datetime, :green:`optional`
        Only required if ``expiry`` is given as string tenor.
        Should be entered as today (also called horizon) and **not** spot. Spot is derived
        from ``delivery_lag`` and ``calendar``.
    modifier : str, :green:`optional (set by 'defaults')`
        The modification rule, in {"F", "MF", "P", "MP"} for date evaluation.
    eom: bool, :green:`optional (set by 'defaults')`
        Whether to use end-of-month rolls when expiry is given as a month or year tenor.
    calendar : calendar or str, :green:`optional`
        The holiday calendar object to use. If str, looks up named calendar from
        static data.
    delivery_lag: int, :green:`optional (set by 'defaults')`
        The number of business days after expiry that the physical settlement of the FX
        exchange occurs.
    payment_lag: int or datetime, :green:`optional (set by 'defaults')`
        The number of business days after expiry to pay premium. If a *datetime* is given this will
        set the premium date explicitly.
    premium_ccy: str, :green:`optional (set as RHS of 'pair')`
        The currency in which the premium is paid. Can *only* be one of the two currencies
        in `pair`.
    delta_type: FXDeltaMethod, str, :green:`optional (set by 'defaults')`
        When deriving strike from delta use the equation associated with *'spot'* or *'forward'*
        delta. If premium currency is LHS of ``pair`` then this will produce
        **premium adjusted** delta values. If the `premium_ccy` is RHS of ``pair`` then delta values
        are **unadjusted**.

        .. note::

           The following define additional **rate** parameters.

    premium: float, :green:`optional`
        The amount paid for the option. If not given assumes an unpriced *Option* and sets this as
        mid-market premium during pricing.
    option_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the option :class:`~rateslib.data.fixings.FXFixing`. If a scalar, is used
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

    Notes
    ------

    Date calculations for *FXOption* products are very specific. See *'Expiry and Delivery Rules'*
    in *FX Option Pricing* by I. Clark. *Rateslib* uses calendars with associated settlement
    calendars and the recognised market convention rules to derive dates.

    .. ipython:: python
       :suppress:

       from rateslib import dt
       from rateslib.instruments import FXCall

    .. ipython:: python

       fxc = FXCall(
           pair="eursek",
           expiry="2M",
           eval_date=dt(2024, 6, 19),  # <- Wednesday
           strike=11.0,
           modifier="mf",
           calendar="tgt,stk|fed",
           delivery_lag=2,
           payment_lag=2,
       )
       fxc.kwargs.leg1["delivery"]  # <- '2M' out of spot: Monday 24 Jun 2024: FX delivery
       fxc.kwargs.leg1["expiry"]    # <- '2b' before 'delivery': Option expiry
       fxc.kwargs.leg2["payment"]  # <- '2b' after 'expiry': Premium payment date

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, call=True, **kwargs)


class FXPut(_BaseFXOption):
    """
    An *FX Put* option.

    For parameters and examples see :class:`~rateslib.instruments.FXCall`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, call=False, **kwargs)
