from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from pytz import UTC

from rateslib import defaults
from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.default import NoInput, _drb
from rateslib.dual import dual_exp, dual_log, dual_norm_cdf, dual_norm_pdf
from rateslib.dual.newton import newton_1dim
from rateslib.dual.utils import _dual_float
from rateslib.fx import FXForwards
from rateslib.fx_volatility import (
    FXDeltaVolSmile,
    FXDeltaVolSurface,
    FXSabrSmile,
    FXSabrSurface,
)
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
)
from rateslib.periods.utils import (
    _get_fx_and_base,
    _get_vol_delta_type,
    _get_vol_smile_or_raise,
    _get_vol_smile_or_value,
    _maybe_local,
)
from rateslib.rs import index_left_f64
from rateslib.splines import evaluate

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        Any,
        Curve,
        DualTypes,
        DualTypes_,
        FXVolOption,
        FXVolOption_,
        Number,
        datetime,
        str_,
    )


class FXOptionPeriod(metaclass=ABCMeta):
    """
    Abstract base class for constructing volatility components of FXOptions.

    Pricing model uses Black 76 log-normal volatility calculations.

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
    delta_type: str in {"forward", "spot", "forward_pa", "spot_pa"}
        When deriving strike from a delta percentage the method used to associate the sensitivity
        to either a spot rate or a forward rate, possibly also premium adjusted.
    metric: str in {"pips", "percent"}, optional
        The pricing metric for the rate of the options.
    """

    # https://www.researchgate.net/publication/275905055_A_Guide_to_FX_Options_Quoting_Conventions/
    style: str = "european"
    kind: str
    phi: float

    @abstractmethod
    def __init__(
        self,
        pair: str,
        expiry: datetime,
        delivery: datetime,
        payment: datetime,
        strike: DualTypes_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        option_fixing: DualTypes_ = NoInput(0),
        delta_type: str_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> None:
        self.pair: str = pair.lower()
        self.currency: str = self.pair[3:]
        self.domestic: str = self.pair[:3]
        self.notional: DualTypes = defaults.notional if isinstance(notional, NoInput) else notional
        self.strike: DualTypes | NoInput = strike
        self.payment: datetime = payment
        self.delivery: datetime = delivery
        self.expiry: datetime = expiry
        self.option_fixing: DualTypes_ = option_fixing
        self.delta_type: str = _drb(defaults.fx_delta_type, delta_type).lower()
        self.metric: str | NoInput = metric

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def cashflows(
        self,
        disc_curve: Curve,
        disc_curve_ccy2: Curve,
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        vol: FXVolOption_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return the properties of the period used in calculating cashflows.

        Parameters
        ----------
        disc_curve: Curve
            The discount *Curve* for the LHS currency.
        disc_curve_ccy2: Curve
            The discount *Curve* for the RHS currency.
        fx: float, FXRates, FXForwards, optional
            The object to project the currency pair FX rate at delivery.
        base: str, optional
            The base currency in which to express the NPV.
        local: bool,
            Whether to display NPV in a currency local to the object.
        vol: float, Dual, Dual2, FXDeltaVolSmile, FXDeltaVolSurface, FXSabrSmile
            The percentage log-normal volatility to price the option.

        Returns
        -------
        dict
        """
        fx_, base = _get_fx_and_base(self.currency, fx, base)
        df, collateral = _dual_float(disc_curve_ccy2[self.payment]), disc_curve_ccy2.meta.collateral
        npv_: dict[str, DualTypes] = self.npv(
            disc_curve, disc_curve_ccy2, fx, base, local=True, vol=vol
        )  # type: ignore[assignment]
        npv: float = _dual_float(npv_[self.currency])

        # TODO: (low-perf) get_vol is called twice for same value, once in npv and once for output
        # This method should not be called to get values used in later calculations becuase it
        # is not efficient. Prefer other ways to get values, i.e. by direct calculation calls.
        if isinstance(fx, FXForwards):
            fx_forward: float | None = _dual_float(fx.rate(self.pair, self.delivery))
        else:
            fx_forward = None

        if isinstance(vol, NoInput) or not isinstance(fx, FXForwards):
            vol_: float | None = None
        else:
            vol_ = _dual_float(self._get_vol_maybe_from_obj(vol, fx, disc_curve))

        return {
            defaults.headers["type"]: type(self).__name__,
            defaults.headers["stub_type"]: "Optionality",
            defaults.headers["pair"]: self.pair,
            defaults.headers["notional"]: _dual_float(self.notional),
            defaults.headers["expiry"]: self.expiry,
            defaults.headers["t_e"]: _dual_float(self._t_to_expiry(disc_curve_ccy2.nodes.initial)),
            defaults.headers["delivery"]: self.delivery,
            defaults.headers["rate"]: fx_forward,
            defaults.headers["strike"]: self.strike,
            defaults.headers["vol"]: vol_,
            defaults.headers["model"]: "Black76",
            defaults.headers["payment"]: self.payment,
            defaults.headers["currency"]: self.currency.upper(),
            defaults.headers["cashflow"]: npv / df,
            defaults.headers["df"]: df,
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: _dual_float(fx_),
            defaults.headers["npv_fx"]: npv * _dual_float(fx_),
            defaults.headers["collateral"]: collateral,
        }

    def npv(
        self,
        disc_curve: Curve,
        disc_curve_ccy2: Curve,
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        vol: FXVolOption_ = NoInput(0),
    ) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *FXOption*.

        Parameters
        ----------
        disc_curve: Curve
            The discount *Curve* for the LHS currency.
        disc_curve_ccy2: Curve
            The discount *Curve* for the RHS currency.
        fx: float, FXRates, FXForwards, optional
            The object to project the currency pair FX rate at delivery.
        base: str, optional
            The base currency in which to express the NPV.
        local: bool,
            Whether to display NPV in a currency local to the object.
        vol: float, Dual, Dual2, FXDeltaVolSmile, FXDeltaVolSurface, FXSabrSmile
            The percentage log-normal volatility to price the option.

        Returns
        -------
        float, Dual, Dual2 or dict of such.
        """
        if self.payment < disc_curve_ccy2.nodes.initial:
            # payment date is in the past avoid issues with fixings or rates
            return _maybe_local(0.0, local, self.currency, NoInput(0), NoInput(0))

        if isinstance(self.strike, NoInput):
            raise ValueError("FXOption must set a `strike` for valuation.")

        if not isinstance(self.option_fixing, NoInput):
            if self.kind == "call" and self.strike < self.option_fixing:
                value = (self.option_fixing - self.strike) * self.notional
            elif self.kind == "put" and self.strike > self.option_fixing:
                value = (self.strike - self.option_fixing) * self.notional
            else:
                return _maybe_local(0.0, local, self.currency, NoInput(0), NoInput(0))
            value *= disc_curve_ccy2[self.payment]

        else:
            # value is expressed in currency (i.e. pair[3:])
            if not isinstance(fx, FXForwards):
                raise ValueError("`fx` must be an FXForwards class for FXOption valuation.")
            vol_ = self._get_vol_maybe_from_obj(vol, fx, disc_curve)

            value = _black76(
                F=fx.rate(self.pair, self.delivery),
                K=self.strike,
                t_e=self._t_to_expiry(disc_curve_ccy2.nodes.initial),
                v1=NoInput(0),  # not required: disc_curve[self.expiry],
                v2=disc_curve_ccy2[self.delivery],
                vol=vol_ / 100.0,
                phi=self.phi,  # controls calls or put price
            )
            value *= self.notional

        return _maybe_local(value, local, self.currency, fx, base)

    def rate(
        self,
        disc_curve: Curve,
        disc_curve_ccy2: Curve,
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: FXVolOption_ = NoInput(0),
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
        npv: DualTypes = self.npv(  # type: ignore[assignment]
            disc_curve,
            disc_curve_ccy2,
            fx,
            self.currency,
            False,
            vol,
        )

        if not isinstance(metric, NoInput):
            metric_ = metric.lower()
        elif not isinstance(self.metric, NoInput):
            metric_ = self.metric.lower()
        else:
            metric_ = defaults.fx_option_metric

        if metric_ == "pips":
            points_premium = (npv / disc_curve_ccy2[self.payment]) / self.notional
            return points_premium * 10000.0
        elif metric_ == "percent":
            if not isinstance(fx, FXForwards):
                raise ValueError("`fx` must be an FXForwards class for FXOption valuation.")
            currency_premium = (npv / disc_curve_ccy2[self.payment]) / fx.rate(
                self.pair,
                self.payment,
            )
            return currency_premium / self.notional * 100
        else:
            raise ValueError("`metric` must be in {'pips', 'percent'}")

    def implied_vol(
        self,
        disc_curve: Curve,
        disc_curve_ccy2: Curve,
        fx: FXForwards,
        premium: DualTypes,
        metric: str | NoInput = NoInput(0),
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
        if isinstance(self.strike, NoInput):
            raise ValueError("FXOption must set a `strike` for valuation.")

        # This function uses newton_1d and is AD safe.

        # convert the premium to a standardised immediate pips value.
        if metric == "percent":
            # convert premium to pips form
            premium = premium * fx.rate(self.pair, self.payment) * 100.0
        # convert to immediate pips form
        imm_premium = premium * disc_curve_ccy2[self.payment]

        t_e = self._t_to_expiry(disc_curve_ccy2.nodes.initial)
        v2 = disc_curve_ccy2[self.delivery]
        f_d = fx.rate(self.pair, self.delivery)

        def root(
            vol: DualTypes, f_d: DualTypes, k: DualTypes, t_e: float, v2: DualTypes, phi: float
        ) -> tuple[DualTypes, DualTypes]:
            f0 = _black76(f_d, k, t_e, NoInput(0), v2, vol, phi) * 10000.0 - imm_premium
            sqrt_t = t_e**0.5
            d_plus = _d_plus_min_u(k / f_d, vol * sqrt_t, 0.5)
            f1 = v2 * dual_norm_pdf(phi * d_plus) * f_d * sqrt_t * 10000.0
            return f0, f1

        result = newton_1dim(root, 0.10, args=(f_d, self.strike, t_e, v2, self.phi))
        _: Number = result["g"] * 100.0
        return _

    def analytic_greeks(
        self,
        disc_curve: Curve,
        disc_curve_ccy2: Curve,
        fx: FXForwards,
        base: str_ = NoInput(0),
        vol: FXVolOption_ = NoInput(0),
        premium: DualTypes_ = NoInput(0),  # expressed in the payment currency
    ) -> dict[str, Any]:
        r"""
        Return the different greeks for the *FX Option*.

        Parameters
        ----------
        disc_curve: Curve
            The discount *Curve* for the LHS currency.
        disc_curve_ccy2: Curve
            The discount *Curve* for the RHS currency.
        fx: FXForwards
            The object to project the relevant forward and spot FX rates. The *'spot'* date is
            assumed to be that applied to the `FXRates` objects for relevant currencies.
        base: str, optional
            Not used by `analytic_greeks`.
        vol: float, Dual, Dual2, FXDeltaVolSmile, FXDeltaVolSurface, FXSabrSmile
            The volatility used in calculation.
        premium: float, Dual, Dual2, optional
            The premium value of the option paid at the appropriate payment date.
            Premium should be expressed in domestic currency.
            If not given calculates and assumes a mid-market premium.

        Returns
        -------
        dict

        Notes
        -----
        **Delta** :math:`\Delta`

        This is the percentage value of the domestic notional in either the *forward* or *spot*
        FX rate. The choice of which is defined by the option's ``delta_type``.

        Delta is also expressed in nominal domestic currency amount.

        **Gamma** :math:`\Gamma`

        This defines by how much *delta* will change for a 1.0 increase in either the *forward*
        or *spot* FX rate. Which rate is determined by the option's ``delta_type``.

        Gamma is also expressed in nominal domestic currency amount for a +1% change in FX rates.

        **Vanna** :math:`\Delta_{\nu}`

        This defines by how much *delta* will change for a 1.0 increase (i.e. 100 log-vols) in
        volatility. The additional

        **Vega** :math:`\nu`

        This defines by how much the PnL of the option will change for a 1.0 increase in
        volatility for a nominal of 1 unit of domestic currency.

        Vega is also expressed in foreign currency for a 0.01 (i.e. 1 log-vol) move higher in vol.

        **Vomma (Volga)** :math:`\nu_{\nu}`

        This defines by how much *vega* will change for a 1.0 increase in volatility.

        These values can be used to estimate PnL for a change in the *forward* or
        *spot* FX rate and the volatility according to,

        .. math::

           \delta P \approx v_{deli} N^{dom} \left ( \Delta \delta f + \frac{1}{2} \Gamma \delta f^2 + \Delta_{\nu} \delta f \delta \sigma \right ) + N^{dom} \left ( \nu \delta \sigma + \frac{1}{2} \nu_{\nu} \delta \sigma^2 \right )

        where :math:`v_{deli}` is the date of FX settlement for *forward* or *spot* rate.

        **Kappa** :math:`\kappa`

        This defines by how much the PnL of the option will change for a 1.0 increase in
        strike for a nominal of 1 unit of domestic currency.

        **Kega** :math:`\left . \frac{dK}{d\sigma} \right|_{\Delta}`

        This defines the rate of change of strike with respect to volatility for a constant delta.

        Raises
        ------
        ValueError: if the ``strike`` is not set on the *Option*.
        """  # noqa: E501
        return self._analytic_greeks(disc_curve, disc_curve_ccy2, fx, base, vol, premium)

    def _analytic_greeks(
        self,
        disc_curve: Curve,
        disc_curve_ccy2: Curve,
        fx: FXForwards,
        base: str_ = NoInput(0),
        vol: FXVolOption_ = NoInput(0),
        premium: DualTypes_ = NoInput(0),  # expressed in the payment currency
        _reduced: bool = False,
    ) -> dict[str, Any]:
        """Calculates `analytic_greeks`, if _reduced only calculates those necessary for
        Strange single_vol calculation.

        _reduced calculates:

        __vol, vega, __bs76, _kappa, _kega, _delta_index, gamma, __strike, __forward, __sqrt_t
        """

        if isinstance(self.strike, NoInput):
            raise ValueError("`strike` must be set to value FXOption.")

        spot = fx.pairs_settlement[self.pair]
        w_spot = disc_curve[spot]
        w_deli = disc_curve[self.delivery]
        if self.delivery != self.payment:
            w_payment = disc_curve[self.payment]
        else:
            w_payment = w_deli
        v_deli = disc_curve_ccy2[self.delivery]
        v_spot = disc_curve_ccy2[spot]
        f_d = fx.rate(self.pair, self.delivery)
        f_t = fx.rate(self.pair, spot)
        u = self.strike / f_d
        sqrt_t = self._t_to_expiry(disc_curve.nodes.initial) ** 0.5

        eta_0, z_w_0, z_u_0 = _delta_type_constants(self.delta_type, w_deli / w_spot, u)

        if isinstance(vol, NoInput):
            raise ValueError("`vol` must be a number quantity or Smile or Surface.")
        elif isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
            eta_1, z_w_1, __ = _delta_type_constants(vol.meta.delta_type, w_deli / w_spot, u)
            res: tuple[DualTypes, DualTypes, DualTypes] = vol.get_from_strike(
                k=self.strike,
                f=f_d,
                expiry=self.expiry,
                w_deli=w_deli,
                w_spot=w_spot,
            )
            delta_idx: DualTypes | None = res[0]
            vol_: DualTypes = res[1]
        elif isinstance(vol, FXSabrSmile):
            eta_1, z_w_1 = eta_0, z_w_0
            res = vol.get_from_strike(k=self.strike, f=f_d, expiry=self.expiry)
            delta_idx = None
            vol_ = res[1]
        elif isinstance(vol, FXSabrSurface):
            eta_1, z_w_1 = eta_0, z_w_0
            # SabrSurface uses FXForwards to derive multiple rates
            res = vol.get_from_strike(k=self.strike, f=fx, expiry=self.expiry)
            delta_idx = None
            vol_ = res[1]
        else:
            eta_1, z_w_1 = eta_0, z_w_0
            delta_idx = None
            vol_ = vol
        vol_ /= 100.0
        vol_sqrt_t = vol_ * sqrt_t

        if "spot" in self.delta_type:
            z_v_0 = v_deli / v_spot
        else:
            z_v_0 = 1.0
        d_eta_0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
        d_plus = _d_plus_min_u(u, vol_sqrt_t, 0.5)
        d_min = _d_plus_min_u(u, vol_sqrt_t, -0.5)
        _is_spot = "spot" in self.delta_type

        _: dict[str, Any] = dict()

        _["gamma"] = self._analytic_gamma(
            _is_spot,
            v_deli,
            v_spot,
            z_w_0,
            self.phi,
            d_plus,
            f_d,
            vol_sqrt_t,
        )
        _["vega"] = self._analytic_vega(v_deli, f_d, sqrt_t, self.phi, d_plus)
        _["_kega"] = self._analytic_kega(
            z_u_0,
            z_w_0,
            eta_0,
            vol_,
            sqrt_t,
            f_d,
            self.phi,
            self.strike,
            d_eta_0,
        )
        _["_kappa"] = self._analytic_kappa(v_deli, self.phi, d_min)
        _["_delta_index"] = delta_idx
        _["__delta_type"] = self.delta_type
        _["__vol"] = vol_
        _["__strike"] = self.strike
        _["__forward"] = f_d
        _["__sqrt_t"] = sqrt_t
        _["__bs76"] = self._analytic_bs76(self.phi, v_deli, f_d, d_plus, self.strike, d_min)
        _["__notional"] = self.notional
        if self.phi > 0:
            _["__class"] = "FXCallPeriod"
        else:
            _["__class"] = "FXPutPeriod"

        if not _reduced:
            _["delta"] = self._analytic_delta(
                premium,
                "_pa" in self.delta_type,
                z_u_0,
                z_w_0,
                d_eta_0,
                self.phi,
                d_plus,
                w_payment,
                w_spot,
                self.notional,
            )
            _[f"delta_{self.pair[:3]}"] = abs(self.notional) * _["delta"]

            _[f"gamma_{self.pair[:3]}_1%"] = (
                _["gamma"] * abs(self.notional) * (f_t if _is_spot else f_d) * 0.01
            )

            _[f"vega_{self.pair[3:]}"] = _["vega"] * abs(self.notional) * 0.01

            _["delta_sticky"] = self._analytic_sticky_delta(
                _["delta"],
                _["vega"],
                v_deli,
                vol,
                sqrt_t,
                vol_,
                self.expiry,
                f_d,
                delta_idx,
                u,
                z_v_0,
                z_w_0,
                z_w_1,
                eta_1,
                d_plus,
                self.strike,
                fx,
            )
            _["vomma"] = self._analytic_vomma(_["vega"], d_plus, d_min, vol_)
            _["vanna"] = self._analytic_vanna(z_w_0, self.phi, d_plus, d_min, vol_)
            # _["vanna"] = self._analytic_vanna(_["vega"], _is_spot, f_t, f_d, d_plus, vol_sqrt_t)

        return _

    @staticmethod
    def _analytic_vega(
        v_deli: DualTypes, f_d: DualTypes, sqrt_t: DualTypes, phi: float, d_plus: DualTypes
    ) -> DualTypes:
        return v_deli * f_d * sqrt_t * dual_norm_pdf(phi * d_plus)

    @staticmethod
    def _analytic_vomma(
        vega: DualTypes,
        d_plus: DualTypes,
        d_min: DualTypes,
        vol: DualTypes,
    ) -> DualTypes:
        return vega * d_plus * d_min / vol

    @staticmethod
    def _analytic_gamma(
        spot: DualTypes,
        v_deli: DualTypes,
        v_spot: DualTypes,
        z_w: DualTypes,
        phi: float,
        d_plus: DualTypes,
        f_d: DualTypes,
        vol_sqrt_t: DualTypes,
    ) -> DualTypes:
        ret = z_w * dual_norm_pdf(phi * d_plus) / (f_d * vol_sqrt_t)
        if spot:
            return ret * z_w * v_spot / v_deli
        return ret

    @staticmethod
    def _analytic_delta(
        premium: DualTypes | NoInput,
        adjusted: bool,
        z_u: DualTypes,
        z_w: DualTypes,
        d_eta: DualTypes,
        phi: float,
        d_plus: DualTypes,
        w_payment: DualTypes,
        w_spot: DualTypes,
        N_dom: DualTypes,
    ) -> DualTypes:
        if not adjusted or isinstance(premium, NoInput):
            # returns unadjusted delta or mid-market premium adjusted delta
            return z_u * z_w * phi * dual_norm_cdf(phi * d_eta)
        else:
            # returns adjusted delta with set premium in domestic (LHS) currency.
            # ASSUMES: if premium adjusted the premium is expressed in LHS currency.
            return z_w * phi * dual_norm_cdf(phi * d_plus) - w_payment / w_spot * premium / N_dom

    @staticmethod
    def _analytic_sticky_delta(
        delta: DualTypes,
        vega: DualTypes,
        v_deli: DualTypes,
        vol: FXVolOption,
        sqrt_t: DualTypes,
        vol_: DualTypes,
        expiry: datetime,
        f_d: DualTypes,
        delta_idx: DualTypes | None,
        u: DualTypes,
        z_v_0: DualTypes,
        z_w_0: DualTypes,
        z_w_1: DualTypes,
        eta_1: float,
        d_plus: DualTypes,
        k: DualTypes,
        fxf: FXForwards,
    ) -> DualTypes:
        dvol_df: DualTypes
        if isinstance(vol, FXSabrSmile):
            _, dvol_df = vol._d_sabr_d_k_or_f(  # type: ignore[assignment]
                k=k,
                f=f_d,
                expiry=expiry,
                as_float=False,
                derivative=2,  # with respect to f
            )
        elif isinstance(vol, FXSabrSurface):
            _, dvol_df = vol._d_sabr_d_k_or_f(  # type: ignore[assignment]
                k=k,
                f=fxf,  # use FXForwards to derive multiple rates
                expiry=expiry,
                as_float=False,
                derivative=2,  # with respect to f
            )
        elif isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
            if isinstance(vol, FXDeltaVolSurface):
                smile: FXDeltaVolSmile = vol.get_smile(expiry)
            else:
                smile = vol
            # d sigma / d delta_idx
            _B = evaluate(smile.nodes.spline.spline, delta_idx, 1) / 100.0  # type: ignore[arg-type]

            if "pa" in vol.meta.delta_type:
                # then smile is adjusted:
                ddelta_idx_df_d: DualTypes = -delta_idx / f_d  # type: ignore[operator]
            else:
                ddelta_idx_df_d = 0.0
            _A = z_w_1 * dual_norm_pdf(-d_plus)
            ddelta_idx_df_d -= _A / (f_d * vol_ * sqrt_t)
            ddelta_idx_df_d /= 1 + _A * ((dual_log(u) / (vol_**2 * sqrt_t) + eta_1 * sqrt_t) * _B)

            dvol_df = _B * z_w_0 / z_v_0 * ddelta_idx_df_d

        else:
            dvol_df = 0.0

        return delta + vega / v_deli * z_v_0 * dvol_df

    @staticmethod
    def _analytic_vanna(
        z_w: DualTypes,
        phi: float,
        d_plus: DualTypes,
        d_min: DualTypes,
        vol: DualTypes,
    ) -> DualTypes:
        return -z_w * dual_norm_pdf(phi * d_plus) * d_min / vol

    # @staticmethod
    # def _analytic_vanna(vega, spot, f_t, f_d, d_plus, vol_sqrt_t):  # Alternative monetary def.
    #     if spot:
    #         return vega / f_t * (1 - d_plus / vol_sqrt_t)
    #     else:
    #         return vega / f_d * (1 - d_plus / vol_sqrt_t)

    @staticmethod
    def _analytic_kega(
        z_u: DualTypes,
        z_w: DualTypes,
        eta: float,
        vol: DualTypes,
        sqrt_t: float,
        f_d: DualTypes,
        phi: float,
        k: DualTypes,
        d_eta: DualTypes,
    ) -> DualTypes:
        if eta < 0:
            # dz_u_du = 1.0
            x = vol * phi * dual_norm_cdf(phi * d_eta) / (f_d * z_u * dual_norm_pdf(phi * d_eta))
        else:
            x = 0.0

        ret = (d_eta - 2.0 * eta * sqrt_t * vol) / (-1 / (k * sqrt_t) + x)
        return ret

    @staticmethod
    def _analytic_kappa(v_deli: DualTypes, phi: float, d_min: DualTypes) -> DualTypes:
        return -v_deli * phi * dual_norm_cdf(phi * d_min)

    @staticmethod
    def _analytic_bs76(
        phi: float,
        v_deli: DualTypes,
        f_d: DualTypes,
        d_plus: DualTypes,
        k: DualTypes,
        d_min: DualTypes,
    ) -> DualTypes:
        return phi * v_deli * (f_d * dual_norm_cdf(phi * d_plus) - k * dual_norm_cdf(phi * d_min))

    def _index_vol_and_strike_from_atm(
        self,
        delta_type: str,
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
        delta_type: str
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
            vol_: DualTypes | FXDeltaVolSmile | FXDeltaVolSurface = vol  # type: ignore[assignment]
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
        t_e = (self.expiry - vol.meta.eval_date).days / 365.0
        if isinstance(f, FXForwards):
            f_d: DualTypes = f.rate(self.pair, self.delivery)
            # _ad = _set_ad_order_objects([0], [f])  # GH755
        else:
            # TODO: mypy should auto detect this
            f_d = f  # type: ignore[assignment]

        def root1d(
            k: DualTypes, f_d: DualTypes, fx: DualTypes | FXForwards, as_float: bool
        ) -> tuple[DualTypes, DualTypes]:
            # if not as_float and isinstance(fx, FXForwards):
            #     _set_ad_order_objects(_ad, [fx])
            dsigma_dk: Number
            sigma, dsigma_dk = vol._d_sabr_d_k_or_f(  # type: ignore[assignment]
                k=k, f=fx, expiry=self.expiry, as_float=as_float, derivative=1
            )
            f0 = -dual_log(k / f_d) + eta_0 * sigma**2 * t_e
            f1 = -1 / k + eta_0 * 2 * sigma * dsigma_dk * t_e
            return f0, f1

        if isinstance(vol, FXSabrSmile):
            alpha = vol.nodes.alpha
        else:  # FXSabrSurface
            # mypy should auto detect this
            vol_: FXSabrSurface = vol  # type: ignore[assignment]
            expiry_posix = self.expiry.replace(tzinfo=UTC).timestamp()
            e_idx = index_left_f64(vol_.expiries_posix, expiry_posix)
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
        v_ = vol.get_from_strike(k, f, self.expiry)[1]
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
        delta_type: str,
        vol_delta_type: str,
        z_w: DualTypes,
    ) -> tuple[DualTypes | None, DualTypes, DualTypes]:
        """Determine strike from ATM delta specification with DeltaVol models or fixed volatility"""
        if eta_0 == 0.5:  # then delta type is unadjusted
            if eta_1 == 0.5:  # then smile delta type matches: closed form eqn available
                if isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
                    d_i: DualTypes = z_w_1 / 2.0
                    vol_value: DualTypes = _get_vol_smile_or_raise(vol, self.expiry)[d_i]
                    delta_idx: DualTypes | None = d_i
                else:
                    vol_value = _validate_obj_not_no_input(vol, "vol")  # type: ignore[assignment]
                    delta_idx = None
                u = _moneyness_from_atm_delta_closed_form(vol_value, t_e)
                return delta_idx, vol_value, u * f
            else:  # then smile delta type unmatched: 2-d solver required
                delta: DualTypes = z_w_0 * self.phi / 2.0
                u, delta_idx = _moneyness_from_delta_two_dimensional(
                    delta, delta_type, _get_vol_smile_or_raise(vol, self.expiry), t_e, z_w, self.phi
                )
        else:  # then delta type is adjusted,
            if eta_1 == -0.5:  # then smile type matches: use 1-d solver
                u = _moneyness_from_atm_delta_one_dimensional(
                    delta_type,
                    vol_delta_type,
                    _get_vol_smile_or_value(vol, self.expiry),
                    t_e,
                    z_w,
                    self.phi,
                )
                delta_idx = z_w_1 * u * 0.5
            else:  # smile delta type unmatched: 2-d solver required
                u, delta_idx = _moneyness_from_atm_delta_two_dimensional(
                    delta_type, _get_vol_smile_or_raise(vol, self.expiry), t_e, z_w, self.phi
                )

        if isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
            vol_value = _get_vol_smile_or_raise(vol, self.expiry)[delta_idx]
        else:
            vol_value = _validate_obj_not_no_input(vol, "vol")  # type: ignore[assignment]

        # u, delta_idx, delta =
        # self._moneyness_from_delta_three_dimensional(delta_type, vol, t_e, z_w)
        return delta_idx, vol_value, u * f

    def _index_vol_and_strike_from_delta(
        self,
        delta: float,
        delta_type: str,
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
        delta_type: str
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
        delta_type: str,
        vol_delta_type: str,
        z_w: DualTypes,
    ) -> tuple[DualTypes | None, DualTypes, DualTypes]:
        """Determine strike and delta index for an option by delta % for DeltaVol type models
        or constant volatility"""
        eta_0, z_w_0, _ = _delta_type_constants(delta_type, z_w, 0.0)  # u: unused
        eta_1, z_w_1, _ = _delta_type_constants(vol_delta_type, z_w, 0.0)  # u: unused
        # then delta types are both unadjusted, used closed form.
        if eta_0 == eta_1 and eta_0 == 0.5:
            if isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
                d_i: DualTypes = (-z_w_1 / z_w_0) * (delta - 0.5 * z_w_0 * (self.phi + 1.0))
                vol_value: DualTypes = _get_vol_smile_or_raise(vol, self.expiry)[d_i]
                delta_idx: DualTypes | None = d_i
            else:
                vol_value = _validate_obj_not_no_input(vol, "vol")  # type: ignore[assignment]
                delta_idx = None
            u: DualTypes = _moneyness_from_delta_closed_form(delta, vol_value, t_e, z_w_0, self.phi)
            return delta_idx, vol_value, u * f
        # then delta types are both adjusted, use 1-d solver.
        elif eta_0 == eta_1 and eta_0 == -0.5:
            u = _moneyness_from_delta_one_dimensional(
                delta,
                delta_type,
                vol_delta_type,
                _get_vol_smile_or_value(vol, self.expiry),
                t_e,
                z_w,
                self.phi,
            )
            delta_idx = (-z_w_1 / z_w_0) * (delta - z_w_0 * u * (self.phi + 1.0) * 0.5)
        else:  # delta adjustment types are different, use 2-d solver.
            u, delta_idx = _moneyness_from_delta_two_dimensional(
                delta, delta_type, _get_vol_smile_or_raise(vol, self.expiry), t_e, z_w, self.phi
            )

        _1: DualTypes | None = delta_idx
        _2: DualTypes = u * f
        if isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
            vol_value = _get_vol_smile_or_raise(vol, self.expiry)[delta_idx]
        else:
            vol_value = _validate_obj_not_no_input(vol, "vol")  # type: ignore[assignment]
        return _1, vol_value, _2

    def _index_vol_and_strike_from_delta_sabr(
        self,
        delta: float,
        delta_type: str,
        vol: FXSabrSmile | FXSabrSurface,
        z_w: DualTypes,
        f: DualTypes | FXForwards,
    ) -> tuple[DualTypes | None, DualTypes, DualTypes]:
        eta_0, z_w_0, _ = _delta_type_constants(delta_type, z_w, 0.0)  # u: unused
        t_e = (self.expiry - vol.meta.eval_date).days / 365.0
        sqrt_t = t_e**0.5
        if isinstance(f, FXForwards):
            f_d: DualTypes = f.rate(self.pair, self.delivery)
            # _ad = _set_ad_order_objects([0], [f])  # GH755
        else:
            # mypy should auto detect this
            f_d = f  # type: ignore[assignment]

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
                k=k, f=fx, expiry=self.expiry, as_float=as_float, derivative=1
            )
            dn0 = -dual_log(k / f_d) / (sigma * sqrt_t) + eta_0 * sigma * sqrt_t
            Phi = dual_norm_cdf(self.phi * dn0)

            if eta_0 == -0.5:
                z_u_0, dz_u_dk = k / f_d, 1 / f_d
                d_1 = -dz_u_dk * z_w_0 * self.phi * Phi
            else:
                z_u_0, dz_u_dk = 1.0, 0.0
                d_1 = 0.0

            ddn_dk = (dual_log(k / f_d) / (sigma**2 * sqrt_t) + eta_0 * sqrt_t) * dsigma_dk - 1 / (
                k * sigma * sqrt_t
            )
            d_2 = -z_u_0 * z_w_0 * dual_norm_pdf(self.phi * dn0) * ddn_dk

            f0 = delta - z_w_0 * z_u_0 * self.phi * Phi
            f1 = d_1 + d_2
            return f0, f1

        g01 = delta if self.phi > 0 else max(delta, -0.75)
        if isinstance(vol, FXSabrSmile):
            alpha = vol.nodes.alpha
        else:  # FXSabrSurface
            # mypy should auto detect this
            vol_: FXSabrSurface = vol  # type: ignore[assignment]
            expiry_posix = self.expiry.replace(tzinfo=UTC).timestamp()
            e_idx = index_left_f64(vol_.expiries_posix, expiry_posix)
            alpha = vol_.smiles[e_idx].nodes.alpha

        g0 = _moneyness_from_delta_closed_form(g01, alpha * 100.0, t_e, z_w_0, self.phi) * f_d

        root_solver = newton_1dim(
            root1d,
            g0,
            args=(f_d, f, z_w_0, delta),
            pre_args=(True,),  # solve iterations `as_float`
            final_args=(False,),  # solve final iteration with AD
            raise_on_fail=True,
        )

        k: DualTypes = root_solver["g"]
        v_ = vol.get_from_strike(k, f, self.expiry)[1]
        return None, v_, k

    def _get_vol_maybe_from_obj(
        self,
        vol: FXVolOption_,
        fx: FXForwards,
        disc_curve: Curve,
    ) -> DualTypes:
        """Return a volatility for the option from a given Smile."""
        # FXOption can have a `strike` that is NoInput, however this internal function should
        # only be performed after a `strike` has been set to number, temporarily or otherwise.
        assert not isinstance(self.strike, NoInput)  # noqa: S101

        if isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface | FXSabrSmile | FXSabrSurface):
            spot = fx.pairs_settlement[self.pair]
            f = fx.rate(self.pair, self.delivery)
            _: tuple[Any, DualTypes, Any] = vol.get_from_strike(
                k=self.strike,
                f=f,
                w_deli=disc_curve[self.delivery],
                w_spot=disc_curve[spot],
                expiry=self.expiry,
            )
            vol_: DualTypes = _[1]
        elif isinstance(vol, NoInput):
            raise ValueError("`vol` cannot be NoInput when provided to pricing function.")
        else:
            vol_ = vol

        return vol_

    def _t_to_expiry(self, now: datetime) -> float:
        # TODO make this a dual, associated with theta
        return (self.expiry - now).days / 365.0

    def _payoff_at_expiry(
        self, rng: list[float] | NoInput = NoInput(0)
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]
    ]:
        # used by plotting methods
        if isinstance(self.strike, NoInput):
            raise ValueError(
                "Cannot return payoff for option without a specified `strike`.",
            )  # pragma: no cover
        if isinstance(rng, NoInput):
            x = np.linspace(0, 20, 1001)
        else:
            x = np.linspace(rng[0], rng[1], 1001)
        k: float = _dual_float(self.strike)
        _ = (x - k) * self.phi
        __ = np.zeros(1001)
        if self.phi > 0:  # call
            y = np.where(x < k, __, _) * self.notional
        else:  # put
            y = np.where(x > k, __, _) * self.notional
        return x, y  # type: ignore[return-value]


class FXCallPeriod(FXOptionPeriod):
    """
    Create an FXCallPeriod.

    For parameters see :class:`~rateslib.periods.FXOptionPeriod`.
    """

    kind = "call"
    phi = 1.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class FXPutPeriod(FXOptionPeriod):
    """
    Create an FXPutPeriod.

    For parameters see :class:`~rateslib.periods.FXOptionPeriod`.
    """

    kind = "put"
    phi = -1.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
