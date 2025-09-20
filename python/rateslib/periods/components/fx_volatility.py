from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from pytz import UTC

import rateslib.errors as err
from rateslib import defaults
from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.dual import dual_exp, dual_log, dual_norm_cdf, dual_norm_pdf, newton_1dim
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import Err, NoInput, Ok, Result, _drb
from rateslib.enums.parameters import (
    FXDeltaMethod,
    FXOptionMetric,
    OptionType,
    _get_fx_delta_type,
    _get_fx_option_metric,
)
from rateslib.fx import FXForwards
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface, FXSabrSmile, FXSabrSurface
from rateslib.fx_volatility.delta_vol import (
    _moneyness_from_atm_delta_one_dimensional,
    _moneyness_from_atm_delta_two_dimensional,
    _moneyness_from_delta_one_dimensional,
    _moneyness_from_delta_two_dimensional,
)
from rateslib.fx_volatility.utils import (
    _black76,
    _d_plus_min_u,
    _delta_type_constants,
    _moneyness_from_atm_delta_closed_form,
    _moneyness_from_delta_closed_form,
    _surface_index_left,
)
from rateslib.periods.components.parameters import _SettlementParams
from rateslib.periods.components.parameters.fx_volatility import _FXOptionParams
from rateslib.periods.components.protocols import (
    _WithAnalyticFXOptionGreeks,
    _WithNPVCashflowsStatic,
)
from rateslib.periods.components.utils import (
    _get_vol_delta_type,
    _get_vol_maybe_from_obj,
    _get_vol_smile_or_raise,
    _get_vol_smile_or_value,
    _validate_fx_as_forwards,
)

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        Arr1dF64,
        DualTypes,
        DualTypes_,
        FXForwards_,
        FXVolOption,
        FXVolOption_,
        Number,
        Series,
        _BaseCurve,
        _BaseCurve_,
        datetime,
        datetime_,
        str_,
    )


class FXOptionPeriod(_WithNPVCashflowsStatic, _WithAnalyticFXOptionGreeks, metaclass=ABCMeta):
    """
    Abstract base class for constructing volatility components of FXOptions.

    Pricing model uses Black 76 log-normal volatility calculations with calendar day time
    reference.

    Parameters
    -----------
    pair: str
        The currency pair for the FX rate which the option is settled. 3-digit code, e.g. "eurusd".
    expiry: datetime
        The expiry of the option: when the fixing and moneyness is determined.
    delivery: datetime
        The delivery date of the underlying FX pair. E.g. typically this would be **spot** as
        measured from the expiry date.
    payment: datetime
        The payment date of the premium associated with the option.
    strike: float, Dual, Dual2
        The strike value of the option.
    notional: float
        The amount in ccy1 (LHS) on which the option is based.
    option_fixing: float, optional
        If an option has already expired this argument is used to fix the price determined at
        expiry.
    delta_type: FXDeltaMethod or str
        When deriving strike from a delta percentage the method used to associate the sensitivity
        to either a spot rate or a forward rate, possibly also premium adjusted.
    metric: str in {"pips", "percent"}, optional
        The pricing metric for the rate of the options.
    """

    fx_option_params: _FXOptionParams
    settlement_params: _SettlementParams
    index_params: None
    non_deliverable_params: None
    rate_params: None

    @abstractmethod
    def __init__(
        self,
        *,
        # option params:
        direction: OptionType,
        payment: datetime,
        pair: str,
        delivery: datetime,
        expiry: datetime,
        strike: DualTypes_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        # currency: str_ = NoInput(0),
        delta_type: FXDeltaMethod | str_ = NoInput(0),
        metric: FXOptionMetric | str_ = NoInput(0),
        option_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        # currency args:
        ex_dividend: datetime_ = NoInput(0),
    ) -> None:
        self.index_params = None
        self.non_deliverable_params = None
        self.rate_params = None
        self.period_params = None
        self.settlement_params = _SettlementParams(
            _notional=_drb(defaults.notional, notional),
            _payment=payment,
            _currency=pair[3:].lower(),
            _notional_currency=pair[:3].lower(),
            _ex_dividend=ex_dividend,
        )
        self.fx_option_params = _FXOptionParams(
            _direction=direction,
            _expiry=expiry,
            _delivery=delivery,
            _delta_type=_get_fx_delta_type(_drb(defaults.fx_delta_type, delta_type)),
            _pair=pair,
            _currency=pair[3:].lower(),
            _strike=strike,
            _metric=_drb(defaults.fx_option_metric, metric),
            _option_fixings=option_fixings,
        )

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def try_unindexed_reference_cashflow(  # type: ignore[override]
        self,
        *,
        rate_curve: _BaseCurve_ = NoInput(0),  # w(.) variety
        disc_curve: _BaseCurve_ = NoInput(0),  # v(.) variety
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        **kwargs: Any,
    ) -> Result[DualTypes]:
        if isinstance(self.fx_option_params.strike, NoInput):
            return Err(ValueError(err.VE_NEEDS_STRIKE))
        k = self.fx_option_params.strike

        if not isinstance(self.fx_option_params.option_fixing.value, NoInput):
            # then the cashflow is defined by a fixing
            fix: DualTypes = self.fx_option_params.option_fixing.value
            phi: OptionType = self.fx_option_params.direction

            if phi == OptionType.Call and k < fix:
                return Ok((fix - k) * self.settlement_params.notional)
            elif phi == OptionType.Put and k > fix:
                return Ok((k - fix) * self.settlement_params.notional)
            else:
                return Ok(0.0)

        else:
            # value is expressed in currency (i.e. pair[3:])
            fx_ = _validate_fx_as_forwards(fx)
            rate_curve_ = _validate_obj_not_no_input(rate_curve, "rate_curve")
            disc_curve_ = _validate_obj_not_no_input(disc_curve, "disc_curve")
            vol_ = _get_vol_maybe_from_obj(
                fx_vol=fx_vol,
                fx=fx_,
                disc_curve=rate_curve_,
                strike=k,
                pair=self.fx_option_params.pair,
                delivery=self.fx_option_params.delivery,
                expiry=self.fx_option_params.expiry,
            )
            expected = _black76(
                F=fx_.rate(self.fx_option_params.pair, self.fx_option_params.delivery),
                K=k,
                t_e=self.fx_option_params.time_to_expiry(disc_curve_.nodes.initial),
                v1=NoInput(0),  # not required
                v2=disc_curve_[self.fx_option_params.delivery]
                / disc_curve_[self.settlement_params.payment],
                vol=vol_ / 100.0,
                phi=self.fx_option_params.direction.value,  # controls calls or put price
            )
            return Ok(expected * self.settlement_params.notional)

    def try_rate(
        self,
        rate_curve: _BaseCurve,
        disc_curve: _BaseCurve,
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        metric: FXOptionMetric | str_ = NoInput(0),
    ) -> Result[DualTypes]:
        """
        Return the pricing metric of the *FXOption*.

        This is priced according to the ``payment`` date of the *OptionPeriod*.

        Parameters
        ----------
        disc_curve: Curve
            The discount *Curve* for the LHS currency. (Not used).
        disc_curve_ccy2: Curve
            The discount *Curve* for the RHS currency.
        fx: float, FXRates, FXForwards, optional
            The object to project the currency pair FX rate at delivery.
        fx_vol: float, Dual, Dual2
            The percentage log-normal volatility to price the option.
        metric: FXOptionMetric  or str, optional
            The metric to return. If *Pips* assumes the premium is in foreign (rhs)
            currency. If *Percent*, the premium is assumed to be domestic (lhs).

        Returns
        -------
        float, Dual, Dual2 or dict of such.
        """
        if not isinstance(metric, NoInput):
            metric_ = _get_fx_option_metric(metric)
        else:  # use metric associated with self
            metric_ = self.fx_option_params.metric

        cash_res = self.try_unindexed_reference_cashflow(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
        )
        if cash_res.is_err:
            return cash_res
        cash: DualTypes = cash_res.unwrap()

        if metric_ == FXOptionMetric.Pips:
            points_premium = cash / self.settlement_params.notional
            return Ok(points_premium * 10000.0)
        else:  # metric_ == FXOptionMetric.Percent:
            fx_ = _validate_fx_as_forwards(fx)
            currency_premium = cash / fx_.rate(
                self.fx_option_params.pair, self.settlement_params.payment
            )
            return Ok(currency_premium / self.settlement_params.notional * 100)

    def rate(
        self,
        *,
        rate_curve: _BaseCurve,
        disc_curve: _BaseCurve,
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return the pricing metric of the *FXOption*.

        This is priced according to the ``payment`` date of the *OptionPeriod*.

        Parameters
        ----------
        disc_curve: Curve
            The discount *Curve* for the LHS currency. (Not used).
        disc_curve_ccy2: Curve
            The discount *Curve* for the RHS currency.
        fx: float, FXRates, FXForwards, optional
            The object to project the currency pair FX rate at delivery.
        base: str, optional
            Not used by `rate`.
        vol: float, Dual, Dual2
            The percentage log-normal volatility to price the option.
        metric: str in {"pips", "percent"}
            The metric to return. If "pips" assumes the premium is in foreign (rhs)
            currency. If "percent", the premium is assumed to be domestic (lhs).

        Returns
        -------
        float, Dual, Dual2 or dict of such.
        """
        return self.try_rate(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
            metric=metric,
        ).unwrap()

    def implied_vol(
        self,
        rate_curve: _BaseCurve,
        disc_curve: _BaseCurve,
        fx: FXForwards,
        premium: DualTypes,
        metric: FXOptionMetric | str_ = NoInput(0),
    ) -> Number:
        """
        Calculate the implied volatility of the FX option.

        Parameters
        ----------
        disc_curve: Curve
            Not used by `implied_vol`.
        disc_curve_ccy2: Curve
            The discount *Curve* for the RHS currency.
        fx: FXForwards
            The object to project the currency pair FX rate at delivery.
        premium: float, Dual, Dual2
            The premium value of the option paid at the appropriate payment date. Expressed
            either in *'pips'* or *'percent'* of notional. Must align with ``metric``.
        metric: str in {"pips", "percent"}, optional
            The manner in which the premium is expressed.

        Returns
        -------
        float, Dual or Dual2
        """
        if isinstance(self.fx_option_params.strike, NoInput):
            raise ValueError(err.VE_NEEDS_STRIKE)
        k = self.fx_option_params.strike
        phi = self.fx_option_params.direction
        metric_ = _get_fx_option_metric(_drb(self.fx_option_params.metric, metric))

        # This function uses newton_1d and is AD safe.

        # convert the premium to a standardised immediate pips value.
        if metric_ == FXOptionMetric.Percent:
            # convert premium to pips form
            premium = (
                premium
                * fx.rate(self.fx_option_params.pair, self.settlement_params.payment)
                * 100.0
            )
        # convert to immediate pips form
        imm_premium = premium * disc_curve[self.settlement_params.payment]

        t_e = self.fx_option_params.time_to_expiry(disc_curve.nodes.initial)
        v2 = disc_curve[self.fx_option_params.delivery]
        f_d = fx.rate(self.fx_option_params.pair, self.fx_option_params.delivery)

        def root(
            vol: DualTypes, f_d: DualTypes, k: DualTypes, t_e: float, v2: DualTypes, phi: float
        ) -> tuple[DualTypes, DualTypes]:
            f0 = _black76(f_d, k, t_e, NoInput(0), v2, vol, phi) * 10000.0 - imm_premium
            sqrt_t = t_e**0.5
            d_plus = _d_plus_min_u(k / f_d, vol * sqrt_t, 0.5)
            f1 = v2 * dual_norm_pdf(phi * d_plus) * f_d * sqrt_t * 10000.0
            return f0, f1

        result = newton_1dim(root, 0.10, args=(f_d, k, t_e, v2, phi))
        _: Number = result["g"] * 100.0
        return _

    # Volatility determinations

    def _index_vol_and_strike_from_atm(
        self,
        delta_type: FXDeltaMethod,
        vol: FXVolOption,
        w_deli: DualTypes,
        w_spot: DualTypes,
        f: DualTypes | FXForwards,
        t_e: DualTypes,
    ) -> tuple[DualTypes | None, DualTypes, DualTypes]:
        """
        This function returns strike and vol, where available, a delta index for an option period
        defined by ATM delta.

        Parameters
        ----------
        delta_type: FXDeltaMethod
            The delta type of the option period.
        vol: DualTypes | Smile | Surface
            The volatility used, either specifici value or a Smile/Surface.
        w_deli: DualTypes
            The relevant discount factor at delivery.
        w_spot: DualTypes
            The relevant discount factor at spot.
        f: DualTypes, FXForwards
            The forward FX rate for delivery. FXForwards is used when a SabrSurface is present.
        t_e: DualTypes
            The time to expiry

        Returns
        -------
        (delta_index, vol, strike)
        """
        # TODO this method branches depending upon eta0 and eta1, but depending upon the
        # type of vol these maybe automatically set equal to each other. Refactoring this would
        # make eliminate repeated type checking for the vol argument.
        vol_delta_type = _get_vol_delta_type(vol, delta_type)

        z_w = w_deli / w_spot
        eta_0, z_w_0, _ = _delta_type_constants(delta_type, z_w, 0.0)  # u: unused
        eta_1, z_w_1, _ = _delta_type_constants(vol_delta_type, z_w, 0.0)  #  u: unused

        if isinstance(vol, FXSabrSmile | FXSabrSurface):
            return self._index_vol_and_strike_from_atm_sabr(f, eta_0, vol)
        else:  # DualTypes | FXDeltaVolSmile | FXDeltaVolSurface
            f_: DualTypes = f  # type: ignore[assignment]
            vol_: DualTypes | FXDeltaVolSmile | FXDeltaVolSurface = vol
            return self._index_vol_and_strike_from_atm_dv(
                f_,
                eta_0,
                eta_1,
                z_w_0,
                z_w_1,
                vol_,
                t_e,
                delta_type,
                vol_delta_type,
                z_w,
            )

    def _index_vol_and_strike_from_atm_sabr(
        self,
        f: DualTypes | FXForwards,
        eta_0: float,
        vol: FXSabrSmile | FXSabrSurface,
    ) -> tuple[DualTypes | None, DualTypes, DualTypes]:
        """Get vol and strike from ATM delta specification under a SABR model."""
        t_e = (self.fx_option_params.expiry - vol.meta.eval_date).days / 365.0
        if isinstance(f, FXForwards):
            f_d: DualTypes = f.rate(self.fx_option_params.pair, self.fx_option_params.delivery)
            # _ad = _set_ad_order_objects([0], [f])  # GH755
        else:
            f_d = f

        def root1d(
            k: DualTypes, f_d: DualTypes, fx: DualTypes | FXForwards, as_float: bool
        ) -> tuple[DualTypes, DualTypes]:
            # if not as_float and isinstance(fx, FXForwards):
            #     _set_ad_order_objects(_ad, [fx])
            dsigma_dk: Number
            sigma, dsigma_dk = vol._d_sabr_d_k_or_f(  # type: ignore[assignment]
                k=k, f=fx, expiry=self.fx_option_params.expiry, as_float=as_float, derivative=1
            )
            f0 = -dual_log(k / f_d) + eta_0 * sigma**2 * t_e
            f1 = -1 / k + eta_0 * 2 * sigma * dsigma_dk * t_e
            return f0, f1

        if isinstance(vol, FXSabrSmile):
            alpha = vol.nodes.alpha
        else:  # FXSabrSurface
            vol_: FXSabrSurface = vol
            expiry_posix = self.fx_option_params.expiry.replace(tzinfo=UTC).timestamp()
            e_idx, _ = _surface_index_left(vol_.meta.expiries_posix, expiry_posix)
            alpha = vol_.smiles[e_idx].nodes.alpha

        root_solver = newton_1dim(
            root1d,
            f_d * dual_exp(eta_0 * alpha**2 * t_e),
            args=(f_d, f),
            pre_args=(True,),  # solve `as_float` in iterations
            final_args=(False,),  # capture AD in final iterations
            raise_on_fail=True,
        )

        k: DualTypes = root_solver["g"]
        v_ = vol.get_from_strike(k, f, self.fx_option_params.expiry)[1]
        return None, v_, k

    def _index_vol_and_strike_from_atm_dv(  # DeltaVol type models
        self,
        f: DualTypes,
        eta_0: float,
        eta_1: float,
        z_w_0: DualTypes,
        z_w_1: DualTypes,
        vol: DualTypes | FXDeltaVolSmile | FXDeltaVolSurface,
        t_e: DualTypes,
        delta_type: FXDeltaMethod,
        vol_delta_type: FXDeltaMethod,
        z_w: DualTypes,
    ) -> tuple[DualTypes | None, DualTypes, DualTypes]:
        """Determine strike from ATM delta specification with DeltaVol models or fixed volatility"""
        if eta_0 == 0.5:  # then delta type is unadjusted
            if eta_1 == 0.5:  # then smile delta type matches: closed form eqn available
                if isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
                    d_i: DualTypes = z_w_1 / 2.0
                    vol_value: DualTypes = _get_vol_smile_or_raise(
                        vol, self.fx_option_params.expiry
                    )[d_i]
                    delta_idx: DualTypes | None = d_i
                else:
                    vol_value = _validate_obj_not_no_input(vol, "vol")  # type: ignore[assignment]
                    delta_idx = None
                u = _moneyness_from_atm_delta_closed_form(vol_value, t_e)
                return delta_idx, vol_value, u * f
            else:  # then smile delta type unmatched: 2-d solver required
                delta: DualTypes = z_w_0 * self.fx_option_params.direction.value / 2.0
                u, delta_idx = _moneyness_from_delta_two_dimensional(
                    delta,
                    delta_type,
                    _get_vol_smile_or_raise(vol, self.fx_option_params.expiry),
                    t_e,
                    z_w,
                    self.fx_option_params.direction.value,
                )
        else:  # then delta type is adjusted,
            if eta_1 == -0.5:  # then smile type matches: use 1-d solver
                u = _moneyness_from_atm_delta_one_dimensional(
                    delta_type,
                    vol_delta_type,
                    _get_vol_smile_or_value(vol, self.fx_option_params.expiry),
                    t_e,
                    z_w,
                    self.fx_option_params.direction,
                )
                delta_idx = z_w_1 * u * 0.5
            else:  # smile delta type unmatched: 2-d solver required
                u, delta_idx = _moneyness_from_atm_delta_two_dimensional(
                    delta_type,
                    _get_vol_smile_or_raise(vol, self.fx_option_params.expiry),
                    t_e,
                    z_w,
                    self.fx_option_params.direction,
                )

        if isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
            vol_value = _get_vol_smile_or_raise(vol, self.fx_option_params.expiry)[delta_idx]
        else:
            vol_value = _validate_obj_not_no_input(vol, "vol")  # type: ignore[assignment]

        # u, delta_idx, delta =
        # self._moneyness_from_delta_three_dimensional(delta_type, vol, t_e, z_w)
        return delta_idx, vol_value, u * f

    def _index_vol_and_strike_from_delta(
        self,
        delta: float,
        delta_type: FXDeltaMethod,
        vol: FXVolOption,
        w_deli: DualTypes,
        w_spot: DualTypes,
        f: DualTypes | FXForwards,
        t_e: DualTypes,
    ) -> tuple[DualTypes | None, DualTypes, DualTypes]:
        """
        This function returns strike and, where available, a delta index for an option period
        defined by a fixed delta percentage.

        Parameters
        ----------
        delta: float
           The delta percent, e.g 0.25.
        delta_type: FXDeltaMethod
           The delta type of the option period.
        vol: DualTypes | Smile | Surface
           The volatility used, either a specific value or a Smile/Surface.
        w_deli: DualTypes
           The relevant discount factor at delivery.
        w_spot: DualTypes
           The relevant discount factor at spot.
        f: DualTypes | FXForwards
           The forward FX rate for delivery. When using a *SabrSurface* this is required in
           *FXForwards* form.
        t_e: DualTypes
           The time to expiry

        Returns
        -------
        (DualTypes, DualTypes)
        """
        vol_delta_type = _get_vol_delta_type(vol, delta_type)
        z_w = w_deli / w_spot

        if isinstance(vol, FXSabrSmile | FXSabrSurface):
            return self._index_vol_and_strike_from_delta_sabr(delta, delta_type, vol, z_w, f)
        else:  # DualTypes | FXDeltaVolSmile | FXDeltaVolSurface
            f_: DualTypes = f  # type: ignore[assignment]
            vol_: DualTypes | FXDeltaVolSmile = vol  # type: ignore[assignment]
            return self._index_vol_and_strike_from_delta_dv(
                f_,
                delta,
                vol_,
                t_e,
                delta_type,
                vol_delta_type,
                z_w,
            )

    def _index_vol_and_strike_from_delta_dv(
        self,
        f: DualTypes,
        delta: float,
        vol: DualTypes | FXDeltaVolSmile | FXDeltaVolSurface,
        t_e: DualTypes,
        delta_type: FXDeltaMethod,
        vol_delta_type: FXDeltaMethod,
        z_w: DualTypes,
    ) -> tuple[DualTypes | None, DualTypes, DualTypes]:
        """Determine strike and delta index for an option by delta % for DeltaVol type models
        or constant volatility"""
        eta_0, z_w_0, _ = _delta_type_constants(delta_type, z_w, 0.0)  # u: unused
        eta_1, z_w_1, _ = _delta_type_constants(vol_delta_type, z_w, 0.0)  # u: unused
        # then delta types are both unadjusted, used closed form.
        if eta_0 == eta_1 and eta_0 == 0.5:
            if isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
                d_i: DualTypes = (-z_w_1 / z_w_0) * (
                    delta - 0.5 * z_w_0 * (self.fx_option_params.direction + 1.0)
                )
                vol_value: DualTypes = _get_vol_smile_or_raise(vol, self.fx_option_params.expiry)[
                    d_i
                ]
                delta_idx: DualTypes | None = d_i
            else:
                vol_value = _validate_obj_not_no_input(vol, "vol")  # type: ignore[assignment]
                delta_idx = None
            u: DualTypes = _moneyness_from_delta_closed_form(
                delta, vol_value, t_e, z_w_0, self.fx_option_params.direction
            )
            return delta_idx, vol_value, u * f
        # then delta types are both adjusted, use 1-d solver.
        elif eta_0 == eta_1 and eta_0 == -0.5:
            u = _moneyness_from_delta_one_dimensional(
                delta,
                delta_type,
                vol_delta_type,
                _get_vol_smile_or_value(vol, self.fx_option_params.expiry),
                t_e,
                z_w,
                self.fx_option_params.direction,
            )
            delta_idx = (-z_w_1 / z_w_0) * (
                delta - z_w_0 * u * (self.fx_option_params.direction + 1.0) * 0.5
            )
        else:  # delta adjustment types are different, use 2-d solver.
            u, delta_idx = _moneyness_from_delta_two_dimensional(
                delta,
                delta_type,
                _get_vol_smile_or_raise(vol, self.fx_option_params.expiry),
                t_e,
                z_w,
                self.fx_option_params.direction,
            )

        _1: DualTypes | None = delta_idx
        _2: DualTypes = u * f
        if isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
            vol_value = _get_vol_smile_or_raise(vol, self.fx_option_params.expiry)[delta_idx]
        else:
            vol_value = _validate_obj_not_no_input(vol, "vol")  # type: ignore[assignment]
        return _1, vol_value, _2

    def _index_vol_and_strike_from_delta_sabr(
        self,
        delta: float,
        delta_type: FXDeltaMethod,
        vol: FXSabrSmile | FXSabrSurface,
        z_w: DualTypes,
        f: DualTypes | FXForwards,
    ) -> tuple[DualTypes | None, DualTypes, DualTypes]:
        eta_0, z_w_0, _ = _delta_type_constants(delta_type, z_w, 0.0)  # u: unused
        t_e = (self.fx_option_params.expiry - vol.meta.eval_date).days / 365.0
        sqrt_t = t_e**0.5
        if isinstance(f, FXForwards):
            f_d: DualTypes = f.rate(self.fx_option_params.pair, self.fx_option_params.delivery)
            # _ad = _set_ad_order_objects([0], [f])  # GH755
        else:
            f_d = f

        def root1d(
            k: DualTypes,
            f_d: DualTypes,
            fx: FXForwards | DualTypes,
            z_w_0: DualTypes,
            delta: float,
            as_float: bool,
        ) -> tuple[DualTypes, DualTypes]:
            # if not as_float and isinstance(fx, FXForwards):
            #     _set_ad_order_objects(_ad, [fx])

            sigma, dsigma_dk = vol._d_sabr_d_k_or_f(
                k=k, f=fx, expiry=self.fx_option_params.expiry, as_float=as_float, derivative=1
            )
            dn0 = -dual_log(k / f_d) / (sigma * sqrt_t) + eta_0 * sigma * sqrt_t
            Phi = dual_norm_cdf(self.fx_option_params.direction * dn0)

            if eta_0 == -0.5:
                z_u_0, dz_u_dk = k / f_d, 1 / f_d
                d_1 = -dz_u_dk * z_w_0 * self.fx_option_params.direction * Phi
            else:
                z_u_0, dz_u_dk = 1.0, 0.0
                d_1 = 0.0

            ddn_dk = (dual_log(k / f_d) / (sigma**2 * sqrt_t) + eta_0 * sqrt_t) * dsigma_dk - 1 / (
                k * sigma * sqrt_t
            )
            d_2 = -z_u_0 * z_w_0 * dual_norm_pdf(self.fx_option_params.direction * dn0) * ddn_dk

            f0 = delta - z_w_0 * z_u_0 * self.fx_option_params.direction * Phi
            f1 = d_1 + d_2
            return f0, f1

        g01 = delta if self.fx_option_params.direction > 0 else max(delta, -0.75)
        if isinstance(vol, FXSabrSmile):
            alpha = vol.nodes.alpha
        else:  # FXSabrSurface
            vol_: FXSabrSurface = vol
            expiry_posix = self.fx_option_params.expiry.replace(tzinfo=UTC).timestamp()
            e_idx, _ = _surface_index_left(vol_.meta.expiries_posix, expiry_posix)
            alpha = vol_.smiles[e_idx].nodes.alpha

        g0 = (
            _moneyness_from_delta_closed_form(
                g01, alpha * 100.0, t_e, z_w_0, self.fx_option_params.direction
            )
            * f_d
        )

        root_solver = newton_1dim(
            root1d,
            g0,
            args=(f_d, f, z_w_0, delta),
            pre_args=(True,),  # solve iterations `as_float`
            final_args=(False,),  # solve final iteration with AD
            raise_on_fail=True,
        )

        k: DualTypes = root_solver["g"]
        v_ = vol.get_from_strike(k, f, self.fx_option_params.expiry)[1]
        return None, v_, k

    def _payoff_at_expiry(
        self, rng: list[float] | NoInput = NoInput(0)
    ) -> tuple[Arr1dF64, Arr1dF64]:
        # used by plotting methods
        if isinstance(self.fx_option_params.strike, NoInput):
            raise ValueError(
                "Cannot return payoff for option without a specified `strike`.",
            )  # pragma: no cover
        if isinstance(rng, NoInput):
            x = np.linspace(0, 20, 1001)
        else:
            x = np.linspace(rng[0], rng[1], 1001)
        k: float = _dual_float(self.fx_option_params.strike)
        _ = (x - k) * self.fx_option_params.direction
        __ = np.zeros(1001)
        if self.fx_option_params.direction > 0:  # call
            y = np.where(x < k, __, _) * self.settlement_params.notional
        else:  # put
            y = np.where(x > k, __, _) * self.settlement_params.notional
        return x, y


class FXCallPeriod(FXOptionPeriod):
    """
    Create an FXCallPeriod.

    For parameters see :class:`~rateslib.periods.FXOptionPeriod`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **{"direction": OptionType.Call, **kwargs})


class FXPutPeriod(FXOptionPeriod):
    """
    Create an FXPutPeriod.

    For parameters see :class:`~rateslib.periods.FXOptionPeriod`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **{"direction": OptionType.Put, **kwargs})
