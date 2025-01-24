from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from rateslib import defaults
from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.default import NoInput, _drb
from rateslib.dual import dual_exp, dual_inv_norm_cdf, dual_log, dual_norm_cdf, dual_norm_pdf
from rateslib.dual.newton import newton_1dim, newton_ndim
from rateslib.dual.utils import _dual_float
from rateslib.fx import FXForwards
from rateslib.fx_volatility import (
    FXDeltaVolSmile,
    FXDeltaVolSurface,
    _black76,
    _d_plus_min_u,
    _delta_type_constants,
)
from rateslib.periods.utils import (
    _get_fx_and_base,
    _get_vol_delta_type,
    _get_vol_smile_or_raise,
    _get_vol_smile_or_value,
    _maybe_local,
)
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
        Sequence,
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
        vol: float, Dual, Dual2, FXDeltaVolSmile, FXDeltaVolSurface
            The percentage log-normal volatility to price the option.

        Returns
        -------
        dict
        """
        fx_, base = _get_fx_and_base(self.currency, fx, base)
        df, collateral = _dual_float(disc_curve_ccy2[self.payment]), disc_curve_ccy2.collateral
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
            defaults.headers["t_e"]: _dual_float(self._t_to_expiry(disc_curve_ccy2.node_dates[0])),
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
        vol: float, Dual, Dual2, FXDeltaVolSmile, FXDeltaVolSurface
            The percentage log-normal volatility to price the option.

        Returns
        -------
        float, Dual, Dual2 or dict of such.
        """
        if self.payment < disc_curve_ccy2.node_dates[0]:
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
                t_e=self._t_to_expiry(disc_curve_ccy2.node_dates[0]),
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

        t_e = self._t_to_expiry(disc_curve_ccy2.node_dates[0])
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
            The object to project the relevant forward and spot FX rates.
        base: str, optional
            Not used by `analytic_greeks`.
        vol: float, Dual, Dual2, FXDeltaVolSmile, FXDeltaVolSurface
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
        sqrt_t = self._t_to_expiry(disc_curve.node_dates[0]) ** 0.5

        if isinstance(vol, NoInput):
            raise ValueError("`vol` must be a number quantity or FXDeltaVolSmile or Surface.")
        elif isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
            res: tuple[DualTypes, DualTypes, DualTypes] = vol.get_from_strike(
                self.strike, f_d, w_deli, w_spot, self.expiry
            )
            delta_idx: DualTypes | None = res[0]
            vol_: DualTypes = res[1]
        else:
            delta_idx = None
            vol_ = vol
        vol_ /= 100.0
        vol_sqrt_t = vol_ * sqrt_t
        eta, z_w, z_u = _delta_type_constants(self.delta_type, w_deli / w_spot, u)
        d_eta = _d_plus_min_u(u, vol_sqrt_t, eta)
        d_plus = _d_plus_min_u(u, vol_sqrt_t, 0.5)
        d_min = _d_plus_min_u(u, vol_sqrt_t, -0.5)
        _is_spot = "spot" in self.delta_type

        _: dict[str, Any] = dict()
        _["delta"] = self._analytic_delta(
            premium,
            "_pa" in self.delta_type,
            z_u,
            z_w,
            d_eta,
            self.phi,
            d_plus,
            w_payment,
            w_spot,
            self.notional,
        )
        _[f"delta_{self.pair[:3]}"] = abs(self.notional) * _["delta"]
        _["gamma"] = self._analytic_gamma(
            _is_spot,
            v_deli,
            v_spot,
            z_w,
            self.phi,
            d_plus,
            f_d,
            vol_sqrt_t,
        )
        _[f"gamma_{self.pair[:3]}_1%"] = (
            _["gamma"] * abs(self.notional) * (f_t if _is_spot else f_d) * 0.01
        )
        _["vega"] = self._analytic_vega(v_deli, f_d, sqrt_t, self.phi, d_plus)
        _[f"vega_{self.pair[3:]}"] = _["vega"] * abs(self.notional) * 0.01
        _["vomma"] = self._analytic_vomma(_["vega"], d_plus, d_min, vol_)
        _["vanna"] = self._analytic_vanna(z_w, self.phi, d_plus, d_min, vol_)
        # _["vanna"] = self._analytic_vanna(_["vega"], _is_spot, f_t, f_d, d_plus, vol_sqrt_t)

        _["_kega"] = self._analytic_kega(
            z_u,
            z_w,
            eta,
            vol_,
            sqrt_t,
            f_d,
            self.phi,
            self.strike,
            d_eta,
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

    def _strike_and_index_from_atm(
        self,
        delta_type: str,
        vol: FXVolOption,
        w_deli: DualTypes,
        w_spot: DualTypes,
        f: DualTypes,
        t_e: DualTypes,
    ) -> tuple[DualTypes, DualTypes | None]:
        # TODO this method branches depending upon eta0 and eta1, but depending upon the
        # type of vol these maybe automatcially set equal to each other. Refactorin this would
        # make eliminate repeated type checking for the vol argument.
        vol_delta_type = _get_vol_delta_type(vol, delta_type)

        z_w = w_deli / w_spot
        eta_0, z_w_0, _ = _delta_type_constants(delta_type, z_w, 0.0)  # u: unused
        eta_1, z_w_1, _ = _delta_type_constants(vol_delta_type, z_w, 0.0)  #  u: unused

        # u, delta_idx, delta =
        # self._moneyness_from_delta_three_dimensional(delta_type, vol, t_e, z_w)

        if eta_0 == 0.5:  # then delta type is unadjusted
            if eta_1 == 0.5:  # then smile delta type matches: closed form eqn available
                if isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
                    d_i: DualTypes = z_w_1 / 2.0
                    vol_value: DualTypes = _get_vol_smile_or_raise(vol, self.expiry)[d_i]
                    delta_idx: DualTypes | None = d_i
                else:
                    vol_value = _validate_obj_not_no_input(vol, "vol")  # type: ignore[assignment]
                    delta_idx = None
                u = self._moneyness_from_atm_delta_closed_form(vol_value, t_e)
            else:  # then smile delta type unmatched: 2-d solver required
                delta: DualTypes = z_w_0 * self.phi / 2.0
                u, delta_idx = self._moneyness_from_delta_two_dimensional(
                    delta,
                    delta_type,
                    _get_vol_smile_or_raise(vol, self.expiry),
                    t_e,
                    z_w,
                )
        else:  # then delta type is adjusted,
            if eta_1 == -0.5:  # then smile type matches: use 1-d solver
                u = self._moneyness_from_atm_delta_one_dimensional(
                    delta_type,
                    vol_delta_type,
                    _get_vol_smile_or_value(vol, self.expiry),
                    t_e,
                    z_w,
                )
                delta_idx = z_w_1 * u * 0.5
            else:  # smile delta type unmatched: 2-d solver required
                u, delta_idx = self._moneyness_from_atm_delta_two_dimensional(
                    delta_type,
                    _get_vol_smile_or_raise(vol, self.expiry),
                    t_e,
                    z_w,
                )

        return u * f, delta_idx

    def _strike_and_index_from_delta(
        self,
        delta: float,
        delta_type: str,
        vol: FXVolOption_,
        w_deli: DualTypes,
        w_spot: DualTypes,
        f: DualTypes,
        t_e: DualTypes,
    ) -> tuple[DualTypes, DualTypes | None]:
        vol_delta_type = _get_vol_delta_type(vol, delta_type)

        z_w = w_deli / w_spot
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
            u: DualTypes = self._moneyness_from_delta_closed_form(delta, vol_value, t_e, z_w_0)
        # then delta types are both adjusted, use 1-d solver.
        elif eta_0 == eta_1 and eta_0 == -0.5:
            u = self._moneyness_from_delta_one_dimensional(
                delta,
                delta_type,
                vol_delta_type,
                _get_vol_smile_or_value(vol, self.expiry),
                t_e,
                z_w,
            )
            delta_idx = (-z_w_1 / z_w_0) * (delta - z_w_0 * u * (self.phi + 1.0) * 0.5)
        else:  # delta adjustment types are different, use 2-d solver.
            u, delta_idx = self._moneyness_from_delta_two_dimensional(
                delta,
                delta_type,
                _get_vol_smile_or_raise(vol, self.expiry),
                t_e,
                z_w,
            )

        _1: DualTypes = u * f
        _2: DualTypes | None = delta_idx
        return _1, _2

    def _moneyness_from_atm_delta_closed_form(self, vol: DualTypes, t_e: DualTypes) -> DualTypes:
        """
        Return `u` given premium unadjusted `delta`, of either 'spot' or 'forward' type.

        This function preserves AD.

        Parameters
        -----------
        vol: float, Dual, Dual2
            The volatility (in %, e.g. 10.0) to use in calculations.
        t_e: float,
            The time to expiry.

        Returns
        -------
        float, Dual or Dual2
        """
        return dual_exp((vol / 100.0) ** 2 * t_e / 2.0)

    def _moneyness_from_delta_closed_form(
        self,
        delta: DualTypes,
        vol: DualTypes,
        t_e: DualTypes,
        z_w_0: DualTypes,
    ) -> DualTypes:
        """
        Return `u` given premium unadjusted `delta`, of either 'spot' or 'forward' type.

        This function preserves AD.

        Parameters
        -----------
        delta: float
            The input unadjusted delta for which to determine the moneyness for.
        vol: float, Dual, Dual2
            The volatility (in %, e.g. 10.0) to use in calculations.
        t_e: float, Dual, Dual2
            The time to expiry.
        z_w_0: float, Dual, Dual2
            The scalar for 'spot' or 'forward' delta types.
            If 'forward', this should equal 1.0.
            If 'spot', this should be :math:`w_deli / w_spot`.

        Returns
        -------
        float, Dual or Dual2
        """
        vol_sqrt_t = vol * t_e**0.5 / 100.0
        _: DualTypes = dual_inv_norm_cdf(self.phi * delta / z_w_0)
        _ = dual_exp(vol_sqrt_t * (0.5 * vol_sqrt_t - self.phi * _))
        return _

    def _moneyness_from_atm_delta_one_dimensional(
        self,
        delta_type: str,
        vol_delta_type: str,
        vol: DualTypes | FXDeltaVolSmile,
        t_e: DualTypes,
        z_w: DualTypes,
    ) -> DualTypes:
        def root1d(
            g: DualTypes,
            delta_type: str,
            vol_delta_type: str,
            phi: float,
            sqrt_t_e: float,
            z_w: DualTypes,
            ad: int,
        ) -> tuple[DualTypes, DualTypes]:
            u = g

            eta_0, z_w_0, z_u_0 = _delta_type_constants(delta_type, z_w, u)
            eta_1, z_w_1, z_u_1 = _delta_type_constants(vol_delta_type, z_w, u)
            dz_u_0_du = 0.5 - eta_0

            delta_idx = z_w_1 * z_u_0 / 2.0
            if isinstance(vol, FXDeltaVolSmile):
                vol_: DualTypes = vol[delta_idx] / 100.0
                dvol_ddeltaidx = evaluate(vol.spline, delta_idx, 1) / 100.0
            else:
                vol_ = vol / 100.0
                dvol_ddeltaidx = 0.0
            vol_ = _dual_float(vol_) if ad == 0 else vol_
            dvol_ddeltaidx = _dual_float(dvol_ddeltaidx) if ad == 0 else dvol_ddeltaidx
            vol_sqrt_t = vol_ * sqrt_t_e

            # Calculate function values
            d0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
            _phi0 = dual_norm_cdf(phi * d0)
            f0 = phi * z_w_0 * z_u_0 * (0.5 - _phi0)

            # Calculate derivative values
            ddelta_idx_du = dz_u_0_du * z_w_1 * 0.5

            lnu = dual_log(u) / (vol_**2 * sqrt_t_e)
            dd_du = (
                -1 / (u * vol_sqrt_t) + dvol_ddeltaidx * (lnu + eta_0 * sqrt_t_e) * ddelta_idx_du
            )

            nd0 = dual_norm_pdf(phi * d0)
            f1 = -dz_u_0_du * z_w_0 * phi * _phi0 - z_u_0 * z_w_0 * nd0 * dd_du

            return f0, f1

        if isinstance(vol, FXDeltaVolSmile):
            avg_vol: DualTypes = _dual_float(list(vol.nodes.values())[int(vol.n / 2)])
        else:
            avg_vol = vol
        g01 = self.phi * 0.5 * (z_w if "spot" in delta_type else 1.0)
        g00 = self._moneyness_from_delta_closed_form(g01, avg_vol, t_e, 1.0)

        root_solver = newton_1dim(
            root1d,
            g00,
            args=(delta_type, vol_delta_type, self.phi, t_e**0.5, z_w),
            pre_args=(0,),
            final_args=(1,),
            raise_on_fail=True,
        )

        u: DualTypes = root_solver["g"]
        return u

    def _moneyness_from_delta_one_dimensional(
        self,
        delta: DualTypes,
        delta_type: str,
        vol_delta_type: str,
        vol: FXDeltaVolSmile | DualTypes,
        t_e: DualTypes,
        z_w: DualTypes,
    ) -> DualTypes:
        def root1d(
            g: DualTypes,
            delta: DualTypes,
            delta_type: str,
            vol_delta_type: str,
            phi: float,
            sqrt_t_e: DualTypes,
            z_w: DualTypes,
            ad: int,
        ) -> tuple[DualTypes, DualTypes]:
            u = g

            eta_0, z_w_0, z_u_0 = _delta_type_constants(delta_type, z_w, u)
            eta_1, z_w_1, z_u_1 = _delta_type_constants(vol_delta_type, z_w, u)
            dz_u_0_du = 0.5 - eta_0

            delta_idx = (-z_w_1 / z_w_0) * (delta - z_w_0 * z_u_0 * (phi + 1.0) * 0.5)
            if isinstance(vol, FXDeltaVolSmile):
                vol_: DualTypes = vol[delta_idx] / 100.0
                dvol_ddeltaidx = evaluate(vol.spline, delta_idx, 1) / 100.0
            else:
                vol_ = vol / 100.0
                dvol_ddeltaidx = 0.0
            vol_ = _dual_float(vol_) if ad == 0 else vol_
            dvol_ddeltaidx = _dual_float(dvol_ddeltaidx) if ad == 0 else dvol_ddeltaidx
            vol_sqrt_t = vol_ * sqrt_t_e

            # Calculate function values
            d0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
            _phi0 = dual_norm_cdf(phi * d0)
            f0 = delta - z_w_0 * z_u_0 * phi * _phi0

            # Calculate derivative values
            ddelta_idx_du = dz_u_0_du * z_w_1 * (phi + 1.0) * 0.5

            lnu = dual_log(u) / (vol_**2 * sqrt_t_e)
            dd_du = (
                -1 / (u * vol_sqrt_t) + dvol_ddeltaidx * (lnu + eta_0 * sqrt_t_e) * ddelta_idx_du
            )

            nd0 = dual_norm_pdf(phi * d0)
            f1 = -dz_u_0_du * z_w_0 * phi * _phi0 - z_u_0 * z_w_0 * nd0 * dd_du

            return f0, f1

        if isinstance(vol, FXDeltaVolSmile):
            avg_vol: DualTypes = _dual_float(list(vol.nodes.values())[int(vol.n / 2)])
        else:
            avg_vol = vol
        g01 = delta if self.phi > 0 else max(delta, -0.75)
        g00 = self._moneyness_from_delta_closed_form(g01, avg_vol, t_e, 1.0)

        msg = (
            f"If the delta, {delta:.1f}, is premium adjusted for a call option is it infeasible?"
            if self.phi > 0
            else ""
        )
        try:
            root_solver = newton_1dim(
                root1d,
                g00,
                args=(delta, delta_type, vol_delta_type, self.phi, t_e**0.5, z_w),
                pre_args=(0,),
                final_args=(1,),
            )
        except ValueError as e:
            raise ValueError(f"Newton root solver failed, with error: {e.__str__()}.\n{msg}")

        if root_solver["state"] == -1:
            raise ValueError(
                f"Newton root solver failed, after {root_solver['iterations']} iterations.\n{msg}",
            )

        u: DualTypes = root_solver["g"]
        return u

    def _moneyness_from_delta_two_dimensional(
        self,
        delta: DualTypes,
        delta_type: str,
        vol: FXDeltaVolSmile,
        t_e: DualTypes,
        z_w: DualTypes,
    ) -> tuple[DualTypes, DualTypes]:
        def root2d(
            g: Sequence[DualTypes],
            delta: DualTypes,
            delta_type: str,
            vol_delta_type: str,
            phi: float,
            sqrt_t_e: float,
            z_w: DualTypes,
            ad: int,
        ) -> tuple[list[DualTypes], list[list[DualTypes]]]:
            u, delta_idx = g[0], g[1]

            eta_0, z_w_0, z_u_0 = _delta_type_constants(delta_type, z_w, u)
            eta_1, z_w_1, z_u_1 = _delta_type_constants(vol_delta_type, z_w, u)
            dz_u_0_du = 0.5 - eta_0
            dz_u_1_du = 0.5 - eta_1

            vol_ = vol[delta_idx] / 100.0
            vol_ = _dual_float(vol_) if ad == 0 else vol_
            vol_sqrt_t = vol_ * sqrt_t_e

            # Calculate function values
            d0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
            _phi0 = dual_norm_cdf(phi * d0)
            f0_0: DualTypes = delta - z_w_0 * z_u_0 * phi * _phi0

            d1 = _d_plus_min_u(u, vol_sqrt_t, eta_1)
            _phi1 = dual_norm_cdf(-d1)
            f0_1: DualTypes = delta_idx - z_w_1 * z_u_1 * _phi1

            # Calculate Jacobian values
            dvol_ddeltaidx = evaluate(vol.spline, delta_idx, 1) / 100.0
            dvol_ddeltaidx = _dual_float(dvol_ddeltaidx) if ad == 0 else dvol_ddeltaidx

            dd_du = -1 / (u * vol_sqrt_t)
            nd0 = dual_norm_pdf(phi * d0)
            nd1 = dual_norm_pdf(-d1)
            lnu = dual_log(u) / (vol_**2 * sqrt_t_e)
            dd0_ddeltaidx = (lnu + eta_0 * sqrt_t_e) * dvol_ddeltaidx
            dd1_ddeltaidx = (lnu + eta_1 * sqrt_t_e) * dvol_ddeltaidx

            f1_00: DualTypes = -z_w_0 * dz_u_0_du * phi * _phi0 - z_w_0 * z_u_0 * nd0 * dd_du
            f1_10: DualTypes = -z_w_1 * dz_u_1_du * _phi1 + z_w_1 * z_u_1 * nd1 * dd_du
            f1_01: DualTypes = -z_w_0 * z_u_0 * nd0 * dd0_ddeltaidx
            f1_11: DualTypes = 1.0 + z_w_1 * z_u_1 * nd1 * dd1_ddeltaidx

            return [f0_0, f0_1], [[f1_00, f1_01], [f1_10, f1_11]]

        avg_vol = _dual_float(list(vol.nodes.values())[int(vol.n / 2)])
        g01 = delta if self.phi > 0 else max(delta, -0.75)
        g00 = self._moneyness_from_delta_closed_form(g01, avg_vol, t_e, 1.0)

        msg = (
            f"If the delta, {_dual_float(delta):.1f}, is premium adjusted for a "
            "call option is it infeasible?"
            if self.phi > 0
            else ""
        )
        try:
            root_solver = newton_ndim(
                root2d,
                [g00, abs(g01)],
                args=(delta, delta_type, vol.delta_type, self.phi, t_e**0.5, z_w),
                pre_args=(0,),
                final_args=(1,),
                raise_on_fail=False,
            )
        except ValueError as e:
            raise ValueError(f"Newton root solver failed, with error: {e.__str__()}.\n{msg}")

        if root_solver["state"] == -1:
            raise ValueError(
                f"Newton root solver failed, after {root_solver['iterations']} iterations.\n{msg}",
            )
        u, delta_idx = root_solver["g"][0], root_solver["g"][1]
        return u, delta_idx

    def _moneyness_from_atm_delta_two_dimensional(
        self,
        delta_type: str,
        vol: FXDeltaVolSmile,
        t_e: DualTypes,
        z_w: DualTypes,
    ) -> tuple[DualTypes, DualTypes]:
        def root2d(
            g: list[DualTypes],
            delta_type: str,
            vol_delta_type: str,
            phi: float,
            sqrt_t_e: DualTypes,
            z_w: DualTypes,
            ad: int,
        ) -> tuple[list[DualTypes], list[list[DualTypes]]]:
            u, delta_idx = g[0], g[1]

            eta_0, z_w_0, z_u_0 = _delta_type_constants(delta_type, z_w, u)
            eta_1, z_w_1, z_u_1 = _delta_type_constants(vol_delta_type, z_w, u)
            dz_u_0_du = 0.5 - eta_0
            dz_u_1_du = 0.5 - eta_1

            vol_ = vol[delta_idx] / 100.0
            vol_ = _dual_float(vol_) if ad == 0 else vol_
            vol_sqrt_t = vol_ * sqrt_t_e

            # Calculate function values
            d0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
            _phi0 = dual_norm_cdf(phi * d0)
            f0_0 = phi * z_w_0 * z_u_0 * (0.5 - _phi0)

            d1 = _d_plus_min_u(u, vol_sqrt_t, eta_1)
            _phi1 = dual_norm_cdf(-d1)
            f0_1 = delta_idx - z_w_1 * z_u_1 * _phi1

            # Calculate Jacobian values
            dvol_ddeltaidx = evaluate(vol.spline, delta_idx, 1) / 100.0
            dvol_ddeltaidx = _dual_float(dvol_ddeltaidx) if ad == 0 else dvol_ddeltaidx

            dd_du = -1 / (u * vol_sqrt_t)
            nd0 = dual_norm_pdf(phi * d0)
            nd1 = dual_norm_pdf(-d1)
            lnu = dual_log(u) / (vol_**2 * sqrt_t_e)
            dd0_ddeltaidx = (lnu + eta_0 * sqrt_t_e) * dvol_ddeltaidx
            dd1_ddeltaidx = (lnu + eta_1 * sqrt_t_e) * dvol_ddeltaidx

            f1_00 = phi * z_w_0 * dz_u_0_du * (0.5 - _phi0) - z_w_0 * z_u_0 * nd0 * dd_du
            f1_10 = -z_w_1 * dz_u_1_du * _phi1 + z_w_1 * z_u_1 * nd1 * dd_du
            f1_01 = -z_w_0 * z_u_0 * nd0 * dd0_ddeltaidx
            f1_11 = 1.0 + z_w_1 * z_u_1 * nd1 * dd1_ddeltaidx

            return [f0_0, f0_1], [[f1_00, f1_01], [f1_10, f1_11]]

        avg_vol = _dual_float(list(vol.nodes.values())[int(vol.n / 2)])
        g01 = self.phi * 0.5 * (z_w if "spot" in delta_type else 1.0)
        g00 = self._moneyness_from_delta_closed_form(g01, avg_vol, t_e, 1.0)

        root_solver = newton_ndim(
            root2d,
            [g00, abs(g01)],
            args=(delta_type, vol.delta_type, self.phi, t_e**0.5, z_w),
            pre_args=(0,),
            final_args=(1,),
            raise_on_fail=True,
        )

        u, delta_idx = root_solver["g"][0], root_solver["g"][1]
        return u, delta_idx

    def _moneyness_from_delta_three_dimensional(
        self,
        delta_type: str,
        vol: DualTypes | FXDeltaVolSmile,
        t_e: DualTypes,
        z_w: DualTypes,
    ) -> tuple[DualTypes, DualTypes, DualTypes]:
        """
        Solve the ATM delta problem where delta is not explicit.
        """

        def root3d(
            g: list[DualTypes],
            delta_type: str,
            vol_delta_type: str,
            phi: float,
            sqrt_t_e: DualTypes,
            z_w: DualTypes,
            ad: int,
        ) -> tuple[list[DualTypes], list[list[DualTypes]]]:
            u, delta_idx, delta = g[0], g[1], g[2]

            eta_0, z_w_0, z_u_0 = _delta_type_constants(delta_type, z_w, u)
            eta_1, z_w_1, z_u_1 = _delta_type_constants(vol_delta_type, z_w, u)
            dz_u_0_du = 0.5 - eta_0
            dz_u_1_du = 0.5 - eta_1

            if isinstance(vol, FXDeltaVolSmile):
                vol_: DualTypes = vol[delta_idx] / 100.0
                dvol_ddeltaidx = evaluate(vol.spline, delta_idx, 1) / 100.0
            else:
                vol_ = vol / 100.0
                dvol_ddeltaidx = 0.0
            vol_ = _dual_float(vol_) if ad == 0 else vol_
            vol_sqrt_t = vol_ * sqrt_t_e

            # Calculate function values
            d0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
            _phi0 = dual_norm_cdf(phi * d0)
            f0_0 = delta - z_w_0 * z_u_0 * phi * _phi0

            d1 = _d_plus_min_u(u, vol_sqrt_t, eta_1)
            _phi1 = dual_norm_cdf(-d1)
            f0_1 = delta_idx - z_w_1 * z_u_1 * _phi1

            f0_2 = delta - phi * z_u_0 * z_w_0 / 2.0

            # Calculate Jacobian values
            dvol_ddeltaidx = _dual_float(dvol_ddeltaidx) if ad == 0 else dvol_ddeltaidx

            dd_du = -1 / (u * vol_sqrt_t)
            nd0 = dual_norm_pdf(phi * d0)
            nd1 = dual_norm_pdf(-d1)
            lnu = dual_log(u) / (vol_**2 * sqrt_t_e)
            dd0_ddeltaidx = (lnu + eta_0 * sqrt_t_e) * dvol_ddeltaidx
            dd1_ddeltaidx = (lnu + eta_1 * sqrt_t_e) * dvol_ddeltaidx

            f1_00 = -z_w_0 * dz_u_0_du * phi * _phi0 - z_w_0 * z_u_0 * nd0 * dd_du  # dh0/du
            f1_10 = -z_w_1 * dz_u_1_du * _phi1 + z_w_1 * z_u_1 * nd1 * dd_du  # dh1/du
            f1_20 = -phi * z_w_0 * dz_u_0_du / 2.0  # dh2/du
            f1_01 = -z_w_0 * z_u_0 * nd0 * dd0_ddeltaidx  # dh0/ddidx
            f1_11 = 1.0 + z_w_1 * z_u_1 * nd1 * dd1_ddeltaidx  # dh1/ddidx
            f1_21 = 0.0  # dh2/ddidx
            f1_02 = 1.0  # dh0/ddelta
            f1_12 = 0.0  # dh1/ddelta
            f1_22 = 1.0  # dh2/ddelta

            return [f0_0, f0_1, f0_2], [
                [f1_00, f1_01, f1_02],
                [f1_10, f1_11, f1_12],
                [f1_20, f1_21, f1_22],
            ]

        if isinstance(vol, FXDeltaVolSmile):
            avg_vol: DualTypes = _dual_float(list(vol.nodes.values())[int(vol.n / 2)])
            vol_delta_type = vol.delta_type
        else:
            avg_vol = vol
            vol_delta_type = self.delta_type
        g02 = 0.5 * self.phi * (z_w if "spot" in delta_type else 1.0)
        g01 = g02 if self.phi > 0 else max(g02, -0.75)
        g00 = self._moneyness_from_delta_closed_form(g01, avg_vol, t_e, 1.0)

        root_solver = newton_ndim(
            root3d,
            [g00, abs(g01), g02],
            args=(delta_type, vol_delta_type, self.phi, t_e**0.5, z_w),
            pre_args=(0,),
            final_args=(1,),
            raise_on_fail=True,
        )

        u, delta_idx, delta = root_solver["g"][0], root_solver["g"][1], root_solver["g"][1]
        return u, delta_idx, delta

    def _get_vol_maybe_from_obj(
        self,
        vol: FXVolOption_,
        fx: FXForwards,
        disc_curve: Curve,
    ) -> DualTypes:
        """Return a volatility for the option from a given Smile."""
        # FXOption can have a `strike` that is NoInput, however this internal function should
        # only be performed after a `strike` has been set, temporarily or otherwise.
        assert not isinstance(self.strike, NoInput)  # noqa: S101

        if isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
            spot = fx.pairs_settlement[self.pair]
            f = fx.rate(self.pair, self.delivery)
            _: tuple[Any, DualTypes, Any] = vol.get_from_strike(
                self.strike,
                f,
                disc_curve[self.delivery],
                disc_curve[spot],
                self.expiry,
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
