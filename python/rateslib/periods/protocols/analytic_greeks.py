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

from typing import TYPE_CHECKING, Protocol

from rateslib.dual import dual_log, dual_norm_cdf, dual_norm_pdf
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import FXDeltaMethod
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface, FXSabrSmile, FXSabrSurface
from rateslib.fx_volatility.utils import (
    _d_plus_min_u,
    _delta_type_constants,
)
from rateslib.periods.parameters.fx_volatility import _FXOptionParams
from rateslib.periods.parameters.settlement import _SettlementParams
from rateslib.splines import evaluate

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        DualTypes,
        DualTypes_,
        FXForwards,
        _BaseCurve,
        _FXVolOption,
        _FXVolOption_,
        datetime,
        datetime_,
    )


class _WithAnalyticFXOptionGreeks(Protocol):
    """
    Protocol to derive analytic *FXOption* greeks.
    """

    @property
    def fx_option_params(self) -> _FXOptionParams: ...

    @property
    def settlement_params(self) -> _SettlementParams: ...

    # def try_unindexed_reference_analytic_greeks(
    #     self,
    #     *,
    #     rate_curve: _BaseCurve,
    #     disc_curve: _BaseCurve,
    #     fx: FXForwards,
    #     index_curve: _BaseCurve_ = NoInput(0),
    #     fx_vol: _FXVolOption_ = NoInput(0),
    # ) -> dict[str, Any]:
    #     return self.__base_analytic_greeks(
    #         rate_curve=rate_curve,
    #         disc_curve=disc_curve,
    #         fx=fx,
    #         fx_vol=fx_vol,
    #         premium=NoInput(0),
    #         _reduced=False,
    #     )

    def analytic_greeks(
        self,
        rate_curve: _BaseCurve,
        disc_curve: _BaseCurve,
        fx: FXForwards,
        fx_vol: _FXVolOption_ = NoInput(0),
        premium: DualTypes_ = NoInput(0),  # expressed in the payment currency
        premium_payment: datetime_ = NoInput(0),
    ) -> dict[str, Any]:
        r"""
        Return the different greeks for the *FX Option*.

        Parameters
        ----------
        rate_curve: _BaseCurve
            The discount *Curve* for the LHS currency of ``pair``.
        disc_curve: _BaseCurve
            The discount *Curve* for the RHS currency of ``pair``.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForward` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary.
        fx_vol: FXDeltaVolSmile, FXSabrSmile, FXDeltaVolSurface, FXSabrSurface, optional
            The FX volatility *Smile* or *Surface* object used for determining Black calendar
            day implied volatility values.
        premium: float, Dual, Dual2, optional
            The premium value of the option paid at the appropriate payment date.
            Premium should be expressed in domestic currency.
            If not given calculates and assumes a mid-market premium.
        premium_payment: datetime, optional
            The date that the premium is paid. If not given is assumed to be equal to the
            *payment* associated with the option period *settlement_params*.

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
        raise NotImplementedError(
            "Type {type(self).__name__} has not implmented `anlaytic_greeks`."
        )

    def _base_analytic_greeks(
        self,
        rate_curve: _BaseCurve,  #  w(.)
        disc_curve: _BaseCurve,  # v(.)
        fx: FXForwards,
        fx_vol: _FXVolOption_ = NoInput(0),
        premium: DualTypes_ = NoInput(0),  # expressed in the payment currency
        premium_payment: datetime_ = NoInput(0),
        _reduced: bool = False,
    ) -> dict[str, Any]:
        """Calculates `analytic_greeks`, if _reduced only calculates those necessary for
        Strange single_vol calculation.

        _reduced calculates:

        __vol, vega, __bs76, _kappa, _kega, _delta_index, gamma, __strike, __forward, __sqrt_t
        """
        premium_payment_ = _drb(self.settlement_params.payment, premium_payment)
        if isinstance(self.fx_option_params.strike, NoInput):
            raise ValueError("`strike` must be set to value FXOption.")

        spot = fx.pairs_settlement[self.fx_option_params.pair]
        w_spot = rate_curve[spot]
        w_deli = rate_curve[self.fx_option_params.delivery]
        if self.fx_option_params.delivery != premium_payment_:
            w_payment = rate_curve[premium_payment_]
        else:
            w_payment = w_deli
        v_deli = disc_curve[self.fx_option_params.delivery]
        v_spot = disc_curve[spot]
        f_d = fx.rate(self.fx_option_params.pair, self.fx_option_params.delivery)
        f_t = fx.rate(self.fx_option_params.pair, spot)
        u = self.fx_option_params.strike / f_d
        sqrt_t = self.fx_option_params.time_to_expiry(rate_curve.nodes.initial) ** 0.5

        eta_0, z_w_0, z_u_0 = _delta_type_constants(
            self.fx_option_params.delta_type, w_deli / w_spot, u
        )

        if isinstance(fx_vol, NoInput):
            raise ValueError("`fx_vol` must be a number quantity or Smile or Surface.")
        elif isinstance(fx_vol, FXDeltaVolSmile | FXDeltaVolSurface):
            eta_1, z_w_1, __ = _delta_type_constants(fx_vol.meta.delta_type, w_deli / w_spot, u)
            res: tuple[DualTypes, DualTypes, DualTypes] = fx_vol.get_from_strike(
                k=self.fx_option_params.strike,
                f=f_d,
                expiry=self.fx_option_params.expiry,
                z_w=w_deli / w_spot,
            )
            delta_idx: DualTypes | None = res[0]
            fx_vol_: DualTypes = res[1]
        elif isinstance(fx_vol, FXSabrSmile):
            eta_1, z_w_1 = eta_0, z_w_0
            res = fx_vol.get_from_strike(
                k=self.fx_option_params.strike, f=f_d, expiry=self.fx_option_params.expiry
            )
            delta_idx = None
            fx_vol_ = res[1]
        elif isinstance(fx_vol, FXSabrSurface):
            eta_1, z_w_1 = eta_0, z_w_0
            # SabrSurface uses FXForwards to derive multiple rates
            res = fx_vol.get_from_strike(
                k=self.fx_option_params.strike, f=fx, expiry=self.fx_option_params.expiry
            )
            delta_idx = None
            fx_vol_ = res[1]
        else:
            eta_1, z_w_1 = eta_0, z_w_0
            delta_idx = None
            fx_vol_ = fx_vol
        fx_vol_ /= 100.0
        vol_sqrt_t = fx_vol_ * sqrt_t

        _is_spot = self.fx_option_params.delta_type in [
            FXDeltaMethod.SpotPremiumAdjusted,
            FXDeltaMethod.Spot,
        ]
        if _is_spot:
            z_v_0 = v_deli / v_spot
        else:
            z_v_0 = 1.0
        d_eta_0 = _d_plus_min_u(u, vol_sqrt_t, eta_0)
        d_plus = _d_plus_min_u(u, vol_sqrt_t, 0.5)
        d_min = _d_plus_min_u(u, vol_sqrt_t, -0.5)

        _: dict[str, Any] = dict()

        _["gamma"] = self._analytic_gamma(
            _is_spot,
            v_deli,
            v_spot,
            z_w_0,
            self.fx_option_params.direction,
            d_plus,
            f_d,
            vol_sqrt_t,
        )
        _["vega"] = self._analytic_vega(
            v_deli, f_d, sqrt_t, self.fx_option_params.direction, d_plus
        )
        _["_kega"] = self._analytic_kega(
            z_u_0,
            z_w_0,
            eta_0,
            fx_vol_,
            sqrt_t,
            f_d,
            self.fx_option_params.direction,
            self.fx_option_params.strike,
            d_eta_0,
        )
        _["_kappa"] = self._analytic_kappa(v_deli, self.fx_option_params.direction, d_min)
        _["_delta_index"] = delta_idx
        _["__delta_type"] = self.fx_option_params.delta_type
        _["__vol"] = fx_vol_
        _["__strike"] = self.fx_option_params.strike
        _["__forward"] = f_d
        _["__sqrt_t"] = sqrt_t
        _["__bs76"] = self._analytic_bs76(
            self.fx_option_params.direction,
            v_deli,
            f_d,
            d_plus,
            self.fx_option_params.strike,
            d_min,
        )
        _["__notional"] = self.settlement_params.notional
        if self.fx_option_params.direction > 0:
            _["__class"] = "FXCallPeriod"
        else:
            _["__class"] = "FXPutPeriod"

        if not _reduced:
            _["delta"] = self._analytic_delta(
                premium,
                self.fx_option_params.delta_type
                in [FXDeltaMethod.SpotPremiumAdjusted, FXDeltaMethod.ForwardPremiumAdjusted],
                z_u_0,
                z_w_0,
                d_eta_0,
                self.fx_option_params.direction,
                d_plus,
                w_payment,
                w_spot,
                self.settlement_params.notional,
            )
            _[f"delta_{self.fx_option_params.pair[:3]}"] = (
                abs(self.settlement_params.notional) * _["delta"]
            )

            _[f"gamma_{self.fx_option_params.pair[:3]}_1%"] = (
                _["gamma"]
                * abs(self.settlement_params.notional)
                * (f_t if _is_spot else f_d)
                * 0.01
            )

            _[f"vega_{self.fx_option_params.pair[3:]}"] = (
                _["vega"] * abs(self.settlement_params.notional) * 0.01
            )

            _["delta_sticky"] = self._analytic_sticky_delta(
                _["delta"],
                _["vega"],
                v_deli,
                fx_vol,
                sqrt_t,
                fx_vol_,
                self.fx_option_params.expiry,
                f_d,
                delta_idx,
                u,
                z_v_0,
                z_w_0,
                z_w_1,
                eta_1,
                d_plus,
                self.fx_option_params.strike,
                fx,
            )
            _["vomma"] = self._analytic_vomma(_["vega"], d_plus, d_min, fx_vol_)
            _["vanna"] = self._analytic_vanna(
                z_w_0, self.fx_option_params.direction, d_plus, d_min, fx_vol_
            )
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
        vol: _FXVolOption,
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

            if vol.meta.delta_type in [
                FXDeltaMethod.ForwardPremiumAdjusted,
                FXDeltaMethod.SpotPremiumAdjusted,
            ]:
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
